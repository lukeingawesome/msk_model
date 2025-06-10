import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel
import pandas as pd
from tqdm import tqdm
import argparse
from train_rad_dino import ImageDataset, RadDinoClassifier
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with Rad-DINO classifier')
    parser.add_argument('--checkpoint_path', type=str, default='/opt/project/msk_model/best_model_ob6vcky2.pt', help='Path to the model checkpoint')
    parser.add_argument('--data_path', type=str, default='/data3/private/knee/supplementary/vp/test.csv', help='Path to data CSV')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Path to save predictions')
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
            
        return image

def main():
    args = parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        # Use default config if not in checkpoint
        config = argparse.Namespace(
            finetune_mode='full',
            num_blocks=2
        )
    else:
        # If checkpoint is just the state dict
        model_state_dict = checkpoint
        config = argparse.Namespace(
            finetune_mode='full',
            num_blocks=2
        )
    
    # Load test data directly
    print(f"Loading test data from {args.data_path}")
    test_df = pd.read_csv(args.data_path)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    test_dataset = ImageDataset(test_df, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RadDinoClassifier(
        finetune_mode=config.finetune_mode,
        num_blocks=config.num_blocks
    ).to(device)
    
    # Load model weights
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Make predictions
    all_preds = []
    all_probs = []
    
    print("Making predictions...")
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference"):
            images = images.float().to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Add predictions to dataframe
    test_df['predicted_label'] = all_preds
    for i in range(6):  # Assuming 6 classes
        test_df[f'prob_class_{i}'] = [prob[i] for prob in all_probs]
    
    # Save results
    print(f"Saving predictions to {args.output_path}")
    test_df.to_csv(args.output_path, index=False)
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total test samples: {len(test_df)}")
    print("\nClass distribution in predictions:")
    print(test_df['predicted_label'].value_counts().sort_index())
    
    # Save class distribution to a separate file
    dist_path = args.output_path.replace('.csv', '_distribution.txt')
    with open(dist_path, 'w') as f:
        f.write("Prediction Distribution:\n")
        f.write(test_df['predicted_label'].value_counts().sort_index().to_string())
    
    print(f"\nDetailed distribution saved to: {dist_path}")

if __name__ == "__main__":
    main() 