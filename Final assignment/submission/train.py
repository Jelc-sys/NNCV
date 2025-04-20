"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    GaussianBlur,
)

from unet import Model
import matplotlib.pyplot as plt
import numpy as np


class ApplyToImageOnly:
    """Wrapper to apply a transform only to the image."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, target):
        return self.transform(image), target


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    print("Prediction shape:", prediction.shape)
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--lambda-weight", type=float, default=1, help="Lambda value for dynamic loss weighting")
    parser.add_argument("--alpha-weight", type=float, default=1, help="Alpha value for fixed loss weighting")

    return parser


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Ensure alpha is a tensor if provided and on the correct device is handled later
        if alpha is not None and not isinstance(alpha, torch.Tensor):
            # Convert list/numpy array to tensor, maintain device handling in forward
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha # self.alpha is expected to be the size-19 class weights tensor

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W], C=19
        # targets: [B, H, W] with class indices (0-18, 255)

        # Move self.alpha to the same device as inputs if needed
        if self.alpha is not None and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        # Compute the standard Cross-Entropy loss per pixel
        # F.cross_entropy with reduction='none' handles ignore_index correctly
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none') # shape: [B, H, W]

        # Get the probability of the ground truth class for non-ignored pixels
        # ce_loss is -log(pt) for non-ignored pixels. For ignored, ce_loss is effectively 0.
        pt = torch.exp(-ce_loss) # shape: [B, H, W]
        
        # Compute the modulating factor (1 - pt)^gamma
        modulating_factor = (1 - pt) ** self.gamma # shape: [B, H, W]

        # Compute the focal loss per pixel (before alpha weighting and reduction)
        # For ignored pixels, ce_loss is 0, so focal_loss is also 0.
        focal_loss = modulating_factor * ce_loss # shape: [B, H, W]

        # Apply alpha weighting if provided (assuming alpha is a tensor [C])
        if self.alpha is not None:
            # Create an alpha tensor of the same spatial shape as targets/focal_loss
            alpha_tensor = torch.ones_like(targets, dtype=focal_loss.dtype, device=inputs.device) # Use focal_loss.dtype

            # Create a mask for valid (non-ignored) pixels
            valid_mask = (targets != self.ignore_index)

            # --- Add this DEBUGGING BLOCK ---
            # Check the values being used to index self.alpha
            if valid_mask.sum() > 0:
                 valid_targets = targets[valid_mask]
                 min_target = valid_targets.min()
                 max_target = valid_targets.max()
                 alpha_size = self.alpha.size(0)
                 if min_target < 0 or max_target >= alpha_size:
                     print(f"DEBUG ERROR: Target index out of bounds for alpha!")
                     print(f"  Min valid target ID: {min_target.item()}")
                     print(f"  Max valid target ID: {max_target.item()}")
                     print(f"  Alpha tensor size: {alpha_size}")
                     # Print some example invalid values if possible (careful with tensor size)
                     # try:
                     #     invalid_indices_tensor = valid_targets[(valid_targets < 0) | (valid_targets >= alpha_size)]
                     #     print(f"  Example invalid target IDs: {invalid_indices_tensor[:10].tolist()}") # Print up to 10
                     # except Exception as e:
                     #      print(f" Could not print example invalid indices: {e}")
                     raise ValueError("Invalid target ID found for alpha indexing.") # Stop execution explicitly

            # --- End DEBUGGING BLOCK ---

            # Use valid targets to index into the 1D self.alpha tensor
            # Assign the corresponding alpha values to the valid positions in alpha_tensor
            # This line will now be protected by the check above
            alpha_tensor[valid_mask] = self.alpha[targets[valid_mask]]

            # Multiply the per-pixel focal loss by the per-pixel alpha weights
            focal_loss = focal_loss * alpha_tensor # shape: [B, H, W]


        # Apply reduction
        if self.reduction == 'mean':
            # Calculate mean only over non-ignored pixels
            mask = (targets != self.ignore_index)
            # Sum focal loss for non-ignored pixels and divide by the count of non-ignored pixels
            non_ignored_focal_loss_sum = focal_loss[mask].sum()
            num_non_ignored_pixels = mask.sum()
            return non_ignored_focal_loss_sum / num_non_ignored_pixels if num_non_ignored_pixels > 0 else torch.tensor(0.0, device=inputs.device)
        elif self.reduction == 'sum':
             # Sum over all pixels (ignored pixels have focal_loss=0)
             return focal_loss.sum()
        # else reduction is 'none', return per-pixel loss
        return focal_loss


def mean_dice(preds, targets, num_classes=19, ignore_index=255):
    """Compute the mean dice coefficient for a batch of predictions.
    """

    dice_per_class = []

    # Find class (dim 1) with highest prob in preds and make one-hot
    preds = preds.softmax(1).argmax(1)
    # Add dimension at index 1 to match size
    # preds = preds.unsqueeze(1)
    # targets = targets.unsqueeze(1)
    
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue # Skip the ignored class

        pred_mask = preds == class_id # Binary mask for preds
        target_mask = targets == class_id # Binary mask for targets

        intersection = (pred_mask & target_mask).sum().float()
        total_area = (pred_mask.sum() +  target_mask.sum()).float()

        # Prevent division by zero
        if total_area > 0:
            dice_per_class.append((2.0 * intersection) / total_area)

    # Use tensor operations for HW acceleration
    return torch.mean(torch.stack(dice_per_class))




def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    transform = Compose([
        ToImage(),  # Applies only to image anyway
        RandomRotation(degrees=10),
        RandomResizedCrop(size=(256, 256), scale=(0.75, 2.0)),
        RandomHorizontalFlip(p=0.5),
        ApplyToImageOnly(GaussianBlur(kernel_size=(3, 5))),
        Resize((256, 256)),
        ApplyToImageOnly(ToDtype(torch.float32, scale=True)),
        ApplyToImageOnly(Normalize((0.5,), (0.5,))),
    ])

    # Load the dataset and make a split for training and validation
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # REMOVE############

    # Log a batch of augmented images
    vis_images, vis_labels = next(iter(train_dataloader))
    vis_img_grid = make_grid(vis_images[:8], nrow=4).permute(1, 2, 0).numpy()

    # Convert masks to color
    vis_labels = vis_labels[:8].unsqueeze(1)  # Ensure shape is (B, 1, H, W)
    vis_labels = convert_to_train_id(vis_labels)
    vis_labels = vis_labels.squeeze(1)
    vis_colored_masks = convert_train_id_to_color(vis_labels)
    vis_mask_grid = make_grid(vis_colored_masks, nrow=4).permute(1, 2, 0).numpy()

    wandb.log({
        "augmented_images": [wandb.Image(vis_img_grid)],
        "augmented_masks": [wandb.Image(vis_mask_grid)],
    })
    #############################

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
        dropout=args.dropout
    ).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Focal loss
    focal_loss_alpha_weights = torch.ones(19) # Initialize all to 1.0
    
    # Assign higher weights to hard classes (Object and Human categories)
    focal_loss_alpha_weights[[5, 6, 7]] = 3.0 # 'Object' classes (pole, traffic light, traffic sign)
    focal_loss_alpha_weights[[11, 12]] = 2.0 # 'Human' classes (person, rider)
    focal_loss_alpha_weights = focal_loss_alpha_weights.to(device)

    loss_focal = FocalLoss(gamma=2.0, alpha=focal_loss_alpha_weights,  ignore_index=255)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, labels)
            fcl_loss = loss_focal(outputs, labels)

            

            # Add dice loss to cross-entropy loss
            dice_score = mean_dice(outputs, labels)
            dice_loss = 1 - dice_score

            #Softmax-weighted loss
            #_lambda = args.lambda_weight
            #exp_ce = torch.exp(_lambda * ce_loss)
            #exp_dice = torch.exp(_lambda * dice_loss)
            _alpha = args.alpha_weight # How much emphasis on dice loss? Lower alpha = more dice loss

            w_focal = _alpha
            w_dice = 1 - _alpha
            loss = w_focal * fcl_loss + w_dice * dice_loss
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "ce_loss": ce_loss.item(),
                "focal_loss": fcl_loss.item(),
                "dice_loss": dice_loss.item(),
                "w_focal": w_focal,
                "w_dice": w_dice,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
                "mean_dice_coefficient": dice_score,
                #"lambda-weight": _lambda,
                "alpha-weight": _alpha,
                "dropout": args.dropout,
                "batch-size": args.batch_size, 
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            dice_scores = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                #loss = criterion(outputs, labels)

                dice_loss = 1 - mean_dice(outputs, labels)
                fcl_loss = loss_focal(outputs, labels)

                _alpha = args.alpha_weight # How much emphasis on dice loss? Lower alpha = more dice loss

                w_focal = _alpha
                w_dice = 1 - _alpha
                loss = w_focal * fcl_loss + w_dice * dice_loss

                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    dice = mean_dice(outputs, labels.squeeze(1)).item() # .item() to convert from tensor to float
                    dice_scores.append(dice)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            mean_dice_score = sum(dice_scores) / len(dice_scores)


            wandb.log({
                "valid_loss": valid_loss,
                "mean_dice_score": mean_dice_score
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            print("Validation Loss: {valid_loss:.4f}, Dice score: {mean_dice_score:.4f}")

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)