import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import wandb
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Rad-DINO classifier')
    parser.add_argument('--use_weighted_loss', action='store_true', help='Use weighted loss for imbalanced classes')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='Warmup ratio')
    parser.add_argument('--data_path', type=str, default='/data3/private/knee/supplementary/vp/all.csv', help='Path to data CSV')
    parser.add_argument('--project_name', type=str, default='rad-dino-classification', help='Wandb project name')
    parser.add_argument('--experiment_name', type=str, default=None, help='Wandb experiment name (if None, will be auto-generated)')
    parser.add_argument('--finetune_mode', type=str, choices=['full', 'head', 'last_blocks'], default='full',
                      help='Finetuning mode: full (all layers), head (only classifier), last_blocks (last N blocks)')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of last blocks to finetune when using last_blocks mode')
    return parser.parse_args()

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if 'label' in self.df.columns:
            label = self.df.iloc[idx]['label']
            return image, torch.tensor(label, dtype=torch.long)
        return image

class RadDinoClassifier(nn.Module):
    def __init__(self, num_classes=6, finetune_mode='full', num_blocks=2):
        super().__init__()
        self.rad_dino = AutoModel.from_pretrained("microsoft/rad-dino")
        self.classifier = nn.Linear(self.rad_dino.config.hidden_size, num_classes)
        
        # Set which parameters to train based on finetune_mode
        if finetune_mode == 'head':
            # Freeze all backbone parameters
            for param in self.rad_dino.parameters():
                param.requires_grad = False
        elif finetune_mode == 'last_blocks':
            # Freeze all parameters first
            for param in self.rad_dino.parameters():
                param.requires_grad = False
            # Unfreeze the last N blocks
            for i in range(num_blocks):
                for param in self.rad_dino.encoder.layer[-(i+1)].parameters():
                    param.requires_grad = True
        # 'full' mode: all parameters are trainable by default
        
    def forward(self, pixel_values):
        # Ensure input is float32
        pixel_values = pixel_values.float()
        outputs = self.rad_dino(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token
        logits = self.classifier(pooled_output)
        return logits

def train_model(model, train_loader, val_loader, device, config, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        epochs=config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.warmup_ratio
    )
    
    best_val_f1 = 0
    best_epoch = 0
    
    # Create save directory if it doesn't exist
    save_dir = os.path.join('checkpoints', wandb.run.name)
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        # Training loop with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]')
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images = images.float().to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            if batch_idx % 10 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
        
        # Validation loop with progress bar
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Val]')
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.float().to(device)
                labels = labels.long().to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        # Calculate confusion matrices
        train_cm = confusion_matrix(train_labels, train_preds)
        val_cm = confusion_matrix(val_labels, val_preds)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_accuracy": train_acc,
            "train_f1": train_f1,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "train_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=train_labels,
                preds=train_preds,
                class_names=[f"Class {i}" for i in range(6)]
            ),
            "val_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=val_labels,
                preds=val_preds,
                class_names=[f"Class {i}" for i in range(6)]
            )
        })
        
        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'config': config
        }
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
            print(f"\nNew best model saved! Validation F1: {val_f1:.4f}")
            
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print("\nTraining Confusion Matrix:")
        print(train_cm)
        print("\nValidation Confusion Matrix:")
        print(val_cm)
    
    # Save final model
    torch.save(checkpoint, os.path.join(save_dir, 'final_model.pt'))
    print(f"\nTraining completed!")
    print(f"Best model was from epoch {best_epoch+1} with validation F1: {best_val_f1:.4f}")
    print(f"All checkpoints saved in: {save_dir}")
    
    # Log the best model to wandb
    wandb.save(os.path.join(save_dir, 'best_model.pt'))
    wandb.save(os.path.join(save_dir, 'final_model.pt'))

def main():
    args = parse_args()
    
    # Initialize wandb with command line arguments
    wandb.init(
        project=args.project_name,
        name=args.experiment_name,
        config={
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "warmup_ratio": args.warmup_ratio,
            "use_weighted_loss": args.use_weighted_loss,
            "finetune_mode": args.finetune_mode,
            "num_blocks": args.num_blocks
        }
    )
    
    # Load data
    df = pd.read_csv(args.data_path)
    
    # Create datasets
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    # Calculate class weights if using weighted loss
    class_weights = None
    if args.use_weighted_loss:
        class_counts = train_df['label'].value_counts()
        total_samples = len(train_df)
        class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])
        print("Class weights:", class_weights)
        wandb.log({"class_weights": wandb.Table(
            data=[[i, float(w)] for i, w in enumerate(class_weights)],
            columns=["class", "weight"]
        )})
    
    # Define transforms
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageDataset(train_df, transform=transform)
    val_dataset = ImageDataset(val_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadDinoClassifier(
        finetune_mode=args.finetune_mode,
        num_blocks=args.num_blocks
    ).to(device)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    # Move class weights to device if using weighted loss
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device, args, class_weights)
    
    wandb.finish()

if __name__ == "__main__":
    main() 