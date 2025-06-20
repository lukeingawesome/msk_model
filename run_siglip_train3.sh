CUDA_VISIBLE_DEVICES=5 python train_siglip3.py \
    --data_path train_all.csv \
    --batch_size 16 \
    --image_size 512 \
    --warmup_ratio 0.05 \
    --project_name "knee" \
    --experiment_name "channel_512"