# RadDino + SigLIP Training for Knee X-ray Vision-Language Understanding

This repository contains code for training and using a hybrid model that combines RadDino (for medical image encoding) with SigLIP's text encoder for knee X-ray vision-language understanding.

## Overview

This implementation uses RadDino as the image encoder (specialized for medical images) combined with SigLIP's text encoder for contrastive learning. The model learns to associate knee X-ray images with their medical impressions, leveraging RadDino's medical imaging expertise and SigLIP's text understanding.

## Features

- **Contrastive Learning**: Uses SigLIP's sigmoid-based contrastive loss
- **Flexible Image Resolution**: Supports both 224x224 and 448x448 image sizes
- **Wandb Integration**: Complete experiment tracking and visualization
- **Mixed Precision Training**: Optional mixed precision for faster training
- **Gradient Accumulation**: Support for effective larger batch sizes
- **Comprehensive Inference**: Both single image and batch inference capabilities

## Installation

1. Clone the repository and install dependencies:

```bash
pip install -r requirements_siglip.txt
```

## Data Format

The training script expects a CSV file with the following columns:
- `img_path`: Path to the X-ray image file
- `impression`: Medical impression text describing the image
- `split`: Dataset split ('train', 'val', or 'test')

Example CSV structure:
```csv
img_path,impression,split
/path/to/image1.png,"Normal knee X-ray",train
/path/to/image2.png,"Severe OA of both knee joint with varus angulation",train
/path/to/image3.png,"Mild osteoarthritis of the knee",val
```

## Training

### Basic Training Command

```bash
python train_siglip.py \
    --data_path train.csv \
    --batch_size 64 \
    --image_size 224 \
    --epochs 20 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --project_name "knee-siglip" \
    --experiment_name "siglip-224-baseline"
```

### Advanced Training Options

```bash
python train_siglip.py \
    --data_path train.csv \
    --batch_size 32 \
    --image_size 448 \
    --epochs 30 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --temperature 0.07 \
    --max_text_length 512 \
    --gradient_accumulation_steps 2 \
    --mixed_precision \
    --project_name "knee-siglip" \
    --experiment_name "siglip-448-mixed-precision"
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | `train.csv` | Path to training CSV file |
| `--batch_size` | `64` | Training batch size |
| `--image_size` | `224` | Image resolution (224 or 448) |
| `--epochs` | `20` | Number of training epochs |
| `--learning_rate` | `1e-4` | Learning rate |
| `--weight_decay` | `0.01` | Weight decay for optimizer |
| `--warmup_ratio` | `0.05` | Warmup ratio (5% of total steps) |
| `--temperature` | `0.07` | Temperature for contrastive loss |
| `--max_text_length` | `512` | Maximum text sequence length |
| `--gradient_accumulation_steps` | `1` | Gradient accumulation steps |
| `--mixed_precision` | `False` | Use mixed precision training |
| `--project_name` | `knee-siglip` | Wandb project name |
| `--experiment_name` | `None` | Wandb experiment name |

## Inference

### Single Image Inference

Test a single image against multiple text queries:

```bash
python inference_siglip.py \
    --model_path checkpoints/best_model.pt \
    --image_path /path/to/knee_xray.png \
    --image_size 224
```

### Batch Inference

Process multiple images from a CSV file:

```bash
python inference_siglip.py \
    --model_path checkpoints/best_model.pt \
    --csv_path test_images.csv \
    --image_size 224 \
    --batch_size 32 \
    --output_path results.csv
```

## Model Architecture

The `KneeRadDinoSigLIPModel` class combines RadDino and SigLIP:

- **Vision Encoder**: RadDino (microsoft/rad-dino) - specialized for medical images
- **Text Encoder**: SigLIP Text Transformer
- **Projection Layer**: Maps RadDino features to SigLIP embedding space
- **Contrastive Learning**: Sigmoid-based contrastive loss
- **Normalization**: L2 normalization of embeddings

## Training Process

1. **Data Loading**: Images and text are processed using SigLIP processor
2. **Image Encoding**: RadDino processes X-ray images to extract medical-specific features
3. **Text Encoding**: SigLIP text encoder processes medical impressions
4. **Feature Projection**: RadDino features are projected to SigLIP embedding space
5. **Contrastive Loss**: Similarity matrix is computed between image and text embeddings
6. **Optimization**: AdamW optimizer with OneCycleLR scheduler
7. **Validation**: Contrastive accuracy is computed on validation set

## Metrics

- **Contrastive Loss**: Primary training objective
- **Contrastive Accuracy**: Percentage of correct image-text pairs identified
- **Learning Rate Scheduling**: OneCycleLR with warmup

## Wandb Integration

All experiments are logged to Weights & Biases:

- Training and validation loss
- Contrastive accuracy
- Learning rate schedule
- Model parameters count
- Dataset statistics

## File Structure

```
├── train_siglip.py          # Main training script
├── inference_siglip.py      # Inference script
├── requirements_siglip.txt  # Python dependencies
├── README_SIGLIP.md        # This file
└── checkpoints/            # Saved model checkpoints
    ├── best_model.pt       # Best model checkpoint
    └── final_model.pt      # Final model checkpoint
```

## Expected Results

With proper training, you should expect:

- **Training Loss**: Decreasing from ~2.0 to ~0.5
- **Validation Loss**: Decreasing from ~2.0 to ~0.6
- **Contrastive Accuracy**: Increasing from ~20% to ~70-80%
- **Training Time**: ~30-60 minutes per epoch (depending on dataset size and hardware)

## GPU Requirements

- **Minimum**: 8GB VRAM (for 224x224 images, batch size 32)
- **Recommended**: 16GB+ VRAM (for 448x448 images, batch size 64)
- **Mixed Precision**: Can reduce memory usage by ~30-40%

## Common Issues and Solutions

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Enable mixed precision training
3. **Poor Convergence**: Adjust learning rate or warmup ratio
4. **Text Truncation**: Increase max_text_length if needed

## Citation

If you use this code, please cite the original SigLIP paper:

```bibtex
@article{zhai2023sigmoid,
  title={Sigmoid loss for language image pre-training},
  author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
  journal={arXiv preprint arXiv:2303.15343},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 