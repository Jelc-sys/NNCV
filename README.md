# Improving U-Net for Semantic Segmentation on Cityscapes

This repository contains the code used for the final assignment of the course 5LSM0 - Neural Networks for Computer Vision. The project investigates how architectural changes and training strategies affect the performance of U-Net models on semantic segmentation tasks using the Cityscapes dataset. A final, best performing architecture has been submitted to the Codalab peak performance benchmark and the robustness benchmark. The final mean Dice scores compared to the baseline U-Net are:

## Baseline
- mean Dice = 0.137

## Peak performance
- mean Dice = 0.264
  
## Robustness
- mean Dice = 0.227

## Project Goals

- Improve the baseline U-Net architecture for semantic segmentation.
- Evaluate the impact of:
  - Residual connections (ResUNet)
  - Transfer learning with pretrained ResNet-18 and ResNet-34 encoders
  - Data augmentations for robustness
  - Combined Dice and focal loss with hyperparameter tuning

## Project Structure
The 'Final Assignment' folder contains all files relevant to the assignment:
- README.md: Contains information on running the code, as well as the student's Codalab username and academic contact details
- jobscript_slurm.sh: A file with training settings to be used by the Snellius supercomputer
- main.sh: The final hyperparameter settings used to train the architecture submitted to the Codalab benchmarks
- model.py: The architecture that was submitted to Codalab
- process_data.py: The code to preprocess data from the Cityscapes dataset
- train.py: The code run to train the model and log relevant training data to WandB


