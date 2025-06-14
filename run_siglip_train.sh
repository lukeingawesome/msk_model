CUDA_VISIBLE_DEVICES=1 python train_siglip.py \
    --data_path train.csv \
    --batch_size 48 \
    --image_size 384 \
    --warmup_ratio 0.05 \
    --project_name "knee" \
    --experiment_name "all"