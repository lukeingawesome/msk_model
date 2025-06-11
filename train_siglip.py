#!/usr/bin/env python3
"""
SigLIP Training Script for Knee X-ray Vision-Language Understanding

This script trains a SigLIP model on knee X-ray images with their corresponding
impression text using contrastive learning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import SiglipModel, SiglipProcessor, AutoModel, AutoTokenizer, AutoProcessor
import wandb
from PIL import Image
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Train SigLIP model for knee X-ray vision-language understanding')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio')
    parser.add_argument('--data_path', type=str, default='train.csv', help='Path to data CSV')
    parser.add_argument('--project_name', type=str, default='knee-siglip', help='Wandb project name')
    parser.add_argument('--experiment_name', type=str, default=None, help='Wandb experiment name')
    parser.add_argument('--image_size', type=int, choices=[224, 384, 512], default=224, help='Image resolution (224, 384, or 512)')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive learning')
    parser.add_argument('--max_text_length', type=int, default=64, help='Maximum text length for tokenization')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    return parser.parse_args()

class KneeXrayDataset(Dataset):
    def __init__(self, df, processor, image_size=224, max_text_length=512):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.image_size = image_size
        self.max_text_length = max_text_length
        
        # Get tokenizer from processor for text processing
        self.tokenizer = processor.tokenizer
        
        # Filter out rows with missing image paths or impressions
        self.df = self.df.dropna(subset=['img_path', 'impression'])
        print(f"Dataset size after filtering: {len(self.df)}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        impression = str(row['impression'])
        
        try:
            # Load and process image
            image = Image.open(img_path).convert('RGB')
            
            # Resize image to the target size
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            
            # Process image separately
            image_inputs = self.processor(
                images=image,
                return_tensors="pt"
            )
            
            # Process text using tokenizer directly
            text_inputs = self.tokenizer(
                impression,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length
            )
            
            # Create attention mask manually (SigLIP tokenizer doesn't provide it)
            input_ids = text_inputs['input_ids']
            # For SigLIP, pad_token_id is 1
            attention_mask = (input_ids != 1).long()
            
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'input_ids': input_ids.squeeze(0),
                'attention_mask': attention_mask.squeeze(0),
                'image_path': img_path,
                'text': impression
            }
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy item if there's an error
            dummy_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
            dummy_text = "Normal knee X-ray"
            
            # Process dummy image separately
            image_inputs = self.processor(
                images=dummy_image,
                return_tensors="pt"
            )
            
            # Process dummy text using tokenizer directly
            text_inputs = self.tokenizer(
                dummy_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length
            )
            
            # Create attention mask manually (SigLIP tokenizer doesn't provide it)
            input_ids = text_inputs['input_ids']
            # For SigLIP, pad_token_id is 1
            attention_mask = (input_ids != 1).long()
            
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'input_ids': input_ids.squeeze(0),
                'attention_mask': attention_mask.squeeze(0),
                'image_path': img_path,
                'text': dummy_text
            }

class SigLIPContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_embeds, text_embeds):
        # Normalize embeddings
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        
        # Calculate similarity matrix
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        
        # Create labels (diagonal should be positive pairs)
        batch_size = image_embeds.size(0)
        labels = torch.arange(batch_size, device=image_embeds.device)
        
        # Calculate losses
        image_loss = nn.functional.cross_entropy(logits, labels)
        text_loss = nn.functional.cross_entropy(logits.t(), labels)
        
        return (image_loss + text_loss) / 2

class KneeRadDinoSigLIPModel(nn.Module):
    def __init__(self, siglip_model_name="google/siglip-base-patch16-224", temperature=0.07):
        super().__init__()
        # Initialize RadDino for image encoding
        self.rad_dino = AutoModel.from_pretrained("microsoft/rad-dino")
        
        # Initialize SigLIP text encoder and processor
        self.siglip = SiglipModel.from_pretrained(siglip_model_name)
        self.temperature = temperature
        
        # Project RadDino features to match SigLIP embedding dimension
        self.image_projection = nn.Linear(
            self.rad_dino.config.hidden_size, 
            self.siglip.config.vision_config.hidden_size
        )
        
        # Use SigLIP's text encoder
        self.text_encoder = self.siglip.text_model
        
    def forward(self, pixel_values, input_ids, attention_mask):
        # Get RadDino image features
        rad_dino_outputs = self.rad_dino(pixel_values=pixel_values)
        # Use [CLS] token representation
        image_features = rad_dino_outputs.last_hidden_state[:, 0]  
        
        # Project to SigLIP embedding space
        image_embeds = self.image_projection(image_features)
        
        # Get SigLIP text features
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeds = text_outputs.pooler_output
        
        # Normalize embeddings
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        
        # Calculate logits for contrastive learning
        logits_per_image = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        logits_per_text = logits_per_image.t()
        
        return {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }

def calculate_contrastive_accuracy(image_embeds, text_embeds):
    """Calculate accuracy for contrastive learning"""
    # Normalize embeddings
    image_embeds = nn.functional.normalize(image_embeds, dim=1)
    text_embeds = nn.functional.normalize(text_embeds, dim=1)
    
    # Calculate similarity matrix
    similarity = torch.matmul(image_embeds, text_embeds.t())
    
    # Get predictions (argmax along each dimension)
    image_preds = torch.argmax(similarity, dim=1)
    text_preds = torch.argmax(similarity, dim=0)
    
    # True labels are the diagonal
    batch_size = image_embeds.size(0)
    labels = torch.arange(batch_size, device=image_embeds.device)
    
    # Calculate accuracies
    image_acc = (image_preds == labels).float().mean()
    text_acc = (text_preds == labels).float().mean()
    
    return (image_acc + text_acc) / 2

def train_model(model, train_loader, val_loader, device, config):
    criterion = SigLIPContrastiveLoss(temperature=config.temperature)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy='cos'
    )
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Create save directory
    save_dir = os.path.join(config.checkpoint_dir, wandb.run.name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        
        # Training loop
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
        
        for batch_idx, batch in enumerate(train_pbar):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Mixed precision forward pass
            if config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values, input_ids, attention_mask)
                    loss = criterion(outputs['image_embeds'], outputs['text_embeds'])
                    loss = loss / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(pixel_values, input_ids, attention_mask)
                loss = criterion(outputs['image_embeds'], outputs['text_embeds'])
                loss = loss / config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            # Calculate accuracy
            with torch.no_grad():
                batch_acc = calculate_contrastive_accuracy(outputs['image_embeds'], outputs['text_embeds'])
                train_acc += batch_acc.item()
            
            train_loss += loss.item() * config.gradient_accumulation_steps
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}',
                'acc': f'{batch_acc.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb every 50 steps
            if batch_idx % 50 == 0:
                wandb.log({
                    "train_step_loss": loss.item() * config.gradient_accumulation_steps,
                    "train_step_accuracy": batch_acc.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "step": epoch * len(train_loader) + batch_idx
                })
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_acc = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Val]')
        with torch.no_grad():
            for batch in val_pbar:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                if config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = model(pixel_values, input_ids, attention_mask)
                        loss = criterion(outputs['image_embeds'], outputs['text_embeds'])
                else:
                    outputs = model(pixel_values, input_ids, attention_mask)
                    loss = criterion(outputs['image_embeds'], outputs['text_embeds'])
                
                batch_acc = calculate_contrastive_accuracy(outputs['image_embeds'], outputs['text_embeds'])
                
                val_loss += loss.item()
                val_acc += batch_acc.item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc.item():.4f}'
                })
        
        # Calculate average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_accuracy": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_acc,
        })
        
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_acc': avg_train_acc,
            'val_acc': avg_val_acc,
            'config': config
        }
        
        # Save every epoch
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_weights.pt'))
            print(f"New best model saved! Validation Loss: {avg_val_loss:.4f}")
    
    # Save final model
    torch.save(checkpoint, os.path.join(save_dir, 'final_model.pt'))
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model_weights.pt'))
    
    print(f"\nTraining completed!")
    print(f"Best model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    print(f"All checkpoints saved in: {save_dir}")
    
    # Save to wandb
    wandb.save(os.path.join(save_dir, 'best_model.pt'))
    wandb.save(os.path.join(save_dir, 'final_model.pt'))

def main():
    args = parse_args()
    
    # Initialize wandb
    wandb.init(
        project=args.project_name,
        name=args.experiment_name,
        config=args
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Total samples: {len(df)}")
    
    # Print dataset statistics
    print("\nDataset split distribution:")
    print(df['split'].value_counts())
    
    # Create datasets
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Initialize processor
    model_name = f"google/siglip-base-patch16-{args.image_size}"
    print(f"Loading SigLIP processor: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = KneeXrayDataset(
        train_df, 
        processor, 
        image_size=args.image_size,
        max_text_length=args.max_text_length
    )
    val_dataset = KneeXrayDataset(
        val_df, 
        processor, 
        image_size=args.image_size,
        max_text_length=args.max_text_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    print(f"Initializing RadDino + SigLIP model: {model_name}")
    model = KneeRadDinoSigLIPModel(
        siglip_model_name=model_name,
        temperature=args.temperature
    ).to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Log model info
    wandb.log({
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset)
    })
    
    # Train model
    print("\nStarting training...")
    train_model(model, train_loader, val_loader, device, args)
    
    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    main() 