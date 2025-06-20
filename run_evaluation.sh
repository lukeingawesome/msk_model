#!/bin/bash

# SigLIP Evaluation Script
# Run zero-shot classification and phrase grounding evaluation

python3 evaluate_siglip.py \
    --model_path /model/workspace/msk/checkpoints/ap/best_model.pt \
    --data_path train.csv \
    --image_size 224 \
    --batch_size 32 \
    --max_text_length 64 \
    --output_dir evaluation_results \
    --save_embeddings \
    --top_k 5

echo "Evaluation completed! Check the results in evaluation_results/ directory" 