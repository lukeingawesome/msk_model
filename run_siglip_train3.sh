CUDA_VISIBLE_DEVICES=5 python train_siglip3.py \
    --data_path /data3/private/knee/supplementary/vp/all_predicted_patient_view.csv \
    --batch_size 16 \
    --image_size 512 \
    --warmup_ratio 0.05 \
    --project_name "knee" \
    --experiment_name "channel_512"