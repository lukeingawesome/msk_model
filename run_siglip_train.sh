CUDA_VISIBLE_DEVICES=2 python train_siglip.py \
    --data_path train2.csv \
    --batch_size 16 \
    --image_size 512 \
    --warmup_ratio 0.05 \
    --project_name "knee" \
    --experiment_name "skyline"