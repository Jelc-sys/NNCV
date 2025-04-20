wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 4 \
    --seed 42 \
    --experiment-id "unet-training" \
    --dropout 0 \
    --lambda-weight 0.5 \
    --alpha-weight 0.7