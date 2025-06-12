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
from transformers import SiglipModel, SiglipProcessor, AutoModel, AutoTokenizer, AutoProcessor, AutoConfig
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
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio')
    parser.add_argument('--data_path', type=str, default='train.csv', help='Path to data CSV')
    parser.add_argument('--project_name', type=str, default='knee-siglip', help='Wandb project name')
    parser.add_argument('--experiment_name', type=str, default=None, help='Wandb experiment name')
    parser.add_argument('--image_size', type=int, choices=[224, 384, 512], default=224, help='Image resolution (224, 384, or 512)')
    parser.add_argument('--max_text_length', type=int, default=64, help='Maximum text length for tokenization')
    parser.add_argument('--checkpoint_dir', type=str, default='/model/workspace/msk/checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    return parser.parse_args()

class KneeXrayDataset(Dataset):
    def __init__(self, df, processor, tokenizer, image_size=224, max_text_length=512):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.tokenizer = tokenizer  # Use separate tokenizer for medical BERT
        self.image_size = image_size
        self.max_text_length = max_text_length
        
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
            
            # Process text using medical BERT tokenizer
            text_inputs = self.tokenizer(
                impression,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length
            )
            
            # Medical BERT tokenizer provides attention mask
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']
            
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
            
            # Process dummy text using medical BERT tokenizer
            text_inputs = self.tokenizer(
                dummy_text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length
            )
            
            # Medical BERT tokenizer provides attention mask
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']
            
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'input_ids': input_ids.squeeze(0),
                'attention_mask': attention_mask.squeeze(0),
                'image_path': img_path,
                'text': dummy_text
            }

class SigLIPContrastiveLoss(nn.Module):
    def __init__(self, init_logit_scale=10.0, init_logit_bias=-10.0):  # Original SigLIP paper initialization
        super().__init__()
        # Learnable logit scale (replaces fixed temperature)
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        # Learnable logit bias to handle class imbalance
        self.logit_bias = nn.Parameter(torch.tensor(init_logit_bias))
    
    def forward(self, image_embeds, text_embeds):
        """
        SigLIP sigmoid loss for contrastive learning
        """
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Check for NaN/inf inputs
        if not torch.isfinite(image_embeds).all():
            print("Warning: NaN/inf detected in image embeddings")
            image_embeds = torch.nan_to_num(image_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if not torch.isfinite(text_embeds).all():
            print("Warning: NaN/inf detected in text embeddings")
            text_embeds = torch.nan_to_num(text_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize embeddings
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        
        # Clamp logit scale to prevent extreme values
        logit_scale_clamped = torch.clamp(self.logit_scale, min=0.01, max=100.0)
        
        # Calculate similarity matrix
        logits = torch.matmul(image_embeds, text_embeds.t()) * logit_scale_clamped + self.logit_bias
        
        # Check for NaN/inf in logits
        if not torch.isfinite(logits).all():
            print(f"Warning: NaN/inf detected in logits. Scale: {logit_scale_clamped.item():.3f}, Bias: {self.logit_bias.item():.3f}")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Create labels: 1 for positive pairs (diagonal), -1 for negative pairs
        labels = 2 * torch.eye(batch_size, device=device) - 1
        
        # SigLIP sigmoid loss
        # Negative log-likelihood of sigmoid for each pair
        sigmoid_input = labels * logits
        
        # Use stable implementation of log-sigmoid
        loss = nn.functional.softplus(-sigmoid_input).mean()
        
        # Final NaN check
        if not torch.isfinite(loss):
            print("Warning: NaN detected in final loss, using dummy loss")
            loss = torch.tensor(1.0, device=device, requires_grad=True)
        
        return loss

class KneeRadDinoSigLIPModel(nn.Module):
    def __init__(self, siglip_model_name="google/siglip-base-patch16-224"):
        super().__init__()
        # Initialize RadDino for image encoding
        self.rad_dino = AutoModel.from_pretrained("microsoft/rad-dino")
        
        # Initialize BiomedVLP medical BERT for text encoding
        self.medical_bert = AutoModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
        
        # Simplified architecture: direct projection to common embedding dimension
        # Use 512 as a more conservative embedding dimension
        embed_dim = 512
        
        self.image_projection = nn.Sequential(
            nn.Linear(self.rad_dino.config.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1)
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(self.medical_bert.config.hidden_size, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1)
        )
        
        # Initialize projection layer weights conservatively
        for module in self.image_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)  # Smaller gain
                nn.init.zeros_(module.bias)
        
        for module in self.text_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)  # Smaller gain
                nn.init.zeros_(module.bias)
        
    def forward(self, pixel_values, input_ids, attention_mask):
        # Get RadDino image features
        rad_dino_outputs = self.rad_dino(pixel_values=pixel_values)
        # Use [CLS] token representation
        image_features = rad_dino_outputs.last_hidden_state[:, 0]
        
        # Check for NaN in image features
        if not torch.isfinite(image_features).all():
            print("Warning: NaN detected in RadDino features")
            image_features = torch.nan_to_num(image_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Project to common embedding space
        image_embeds = self.image_projection(image_features)
        
        # Get medical BERT text features
        text_outputs = self.medical_bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation (check if pooler_output exists)
        if hasattr(text_outputs, 'pooler_output'):
            text_features = text_outputs.pooler_output
        else:
            # Use [CLS] token from last_hidden_state
            text_features = text_outputs.last_hidden_state[:, 0]
        
        # Check for NaN in text features
        if not torch.isfinite(text_features).all():
            print("Warning: NaN detected in Medical BERT features")
            text_features = torch.nan_to_num(text_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Project to common embedding space
        text_embeds = self.text_projection(text_features)
        
        return {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds
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
    criterion = SigLIPContrastiveLoss()
    
    # Create optimizer for all parameters including loss parameters
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': criterion.parameters(), 'lr': config.learning_rate * 0.1}  # Lower LR for loss params
    ], lr=config.learning_rate, weight_decay=config.weight_decay)
    
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
    
    # Create save directory with absolute path
    checkpoint_dir = os.path.abspath(config.checkpoint_dir)
    save_dir = os.path.join(checkpoint_dir, wandb.run.name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Saving checkpoints to absolute path: {save_dir}")
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    # Move criterion to device
    criterion = criterion.to(device)
    
    for epoch in range(config.epochs):
        model.train()
        criterion.train()
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
                    # Clip gradients to prevent exploding gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)
                    
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
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            # Calculate accuracy and check for NaN
            with torch.no_grad():
                # Check if loss is NaN
                if not torch.isfinite(loss):
                    print(f"NaN loss detected at batch {batch_idx}")
                    print(f"Image embeds stats: min={outputs['image_embeds'].min():.4f}, max={outputs['image_embeds'].max():.4f}, mean={outputs['image_embeds'].mean():.4f}")
                    print(f"Text embeds stats: min={outputs['text_embeds'].min():.4f}, max={outputs['text_embeds'].max():.4f}, mean={outputs['text_embeds'].mean():.4f}")
                    print(f"Logit scale: {criterion.logit_scale.item():.4f}, Logit bias: {criterion.logit_bias.item():.4f}")
                    # Skip this batch
                    continue
                
                batch_acc = calculate_contrastive_accuracy(outputs['image_embeds'], outputs['text_embeds'])
                train_acc += batch_acc.item()
            
            train_loss += loss.item() * config.gradient_accumulation_steps
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}',
                'acc': f'{batch_acc.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'logit_scale': f'{criterion.logit_scale.item():.3f}',
                'logit_bias': f'{criterion.logit_bias.item():.3f}'
            })
            
            # Log to wandb every 50 steps
            if batch_idx % 50 == 0:
                wandb.log({
                    "train_step_loss": loss.item() * config.gradient_accumulation_steps,
                    "train_step_accuracy": batch_acc.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "logit_scale": criterion.logit_scale.item(),
                    "logit_bias": criterion.logit_bias.item(),
                    "step": epoch * len(train_loader) + batch_idx
                })
        
        # Validation loop
        model.eval()
        criterion.eval()
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
            "logit_scale": criterion.logit_scale.item(),
            "logit_bias": criterion.logit_bias.item(),
        })
        
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        print(f"Logit Scale: {criterion.logit_scale.item():.3f}, Logit Bias: {criterion.logit_bias.item():.3f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'criterion_state_dict': criterion.state_dict(),
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
            torch.save({
                'model': model.state_dict(),
                'criterion': criterion.state_dict()
            }, os.path.join(save_dir, 'best_model_weights.pt'))
            print(f"New best model saved! Validation Loss: {avg_val_loss:.4f}")
    
    # Save final model
    torch.save(checkpoint, os.path.join(save_dir, 'final_model.pt'))
    torch.save({
        'model': model.state_dict(),
        'criterion': criterion.state_dict()
    }, os.path.join(save_dir, 'final_model_weights.pt'))
    
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
    df = df.loc[df['label'] == 0].reset_index(drop=True)
    print(f"Total samples: {len(df)}")
    
    # Print dataset statistics
    print("\nDataset split distribution:")
    print(df['split'].value_counts())
    
    # Create datasets
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Initialize processor for images (SigLIP) and tokenizer for text (medical BERT)
    model_name = f"google/siglip-base-patch16-{args.image_size}"
    print(f"Loading SigLIP processor for images: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Initialize medical BERT tokenizer
    print("Loading BiomedVLP-CXR-BERT tokenizer for medical text")
    medical_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
    
    # Create datasets
    train_dataset = KneeXrayDataset(
        train_df, 
        processor,
        medical_tokenizer, 
        image_size=args.image_size,
        max_text_length=args.max_text_length
    )
    val_dataset = KneeXrayDataset(
        val_df, 
        processor,
        medical_tokenizer, 
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
    print(f"Initializing RadDino + Medical BERT model")
    model = KneeRadDinoSigLIPModel(
        siglip_model_name=model_name
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