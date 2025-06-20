#!/usr/bin/env python3
"""
Simplified SigLIP Evaluation Script

This script provides a simplified version of zero-shot evaluation and phrase grounding
for the trained SigLIP model.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Import model architecture from training script
from train_siglip import KneeRadDinoSigLIPModel

def parse_args():
    parser = argparse.ArgumentParser(description='Simple SigLIP evaluation')
    parser.add_argument('--model_path', type=str, default='/model/workspace/msk/checkpoints/ap/best_model.pt')
    parser.add_argument('--data_path', type=str, default='train.csv')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_path', type=str, default='simple_evaluation_results.csv')
    return parser.parse_args()

class SimpleEvalDataset(Dataset):
    def __init__(self, df, processor, image_size=224):
        self.df = df.dropna(subset=['img_path']).reset_index(drop=True)
        self.processor = processor
        self.image_size = image_size
        print(f"Dataset size: {len(self.df)}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row['img_path']).convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            
            image_inputs = self.processor(images=image, return_tensors="pt")
            
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'image_path': row['img_path'],
                'label': row.get('label', -1),
                'impression': row.get('impression', '')
            }
        except Exception as e:
            print(f"Error loading {row['img_path']}: {e}")
            # Return dummy data
            dummy_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
            image_inputs = self.processor(images=dummy_image, return_tensors="pt")
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'image_path': row['img_path'],
                'label': row.get('label', -1),
                'impression': row.get('impression', '')
            }

def get_oa_prompts():
    """Get OA severity prompts"""
    return {
        'normal': [
            "normal knee X-ray",
            "healthy knee joint",
            "no osteoarthritis"
        ],
        'minimal_oa': [
            "minimal osteoarthritis",
            "mild OA",
            "early degenerative changes"
        ],
        'severe_oa': [
            "severe osteoarthritis",
            "severe OA", 
            "advanced degenerative changes",
            "bone on bone contact"
        ]
    }

def encode_prompts(model, tokenizer, prompts, device):
    """Encode text prompts into embeddings"""
    model.eval()
    embeddings = []
    
    for prompt in prompts:
        text_inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        )
        
        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            # Get text features from medical BERT
            text_outputs = model.medical_bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Use CLS token
            if hasattr(text_outputs, 'pooler_output'):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs.last_hidden_state[:, 0]
            
            # Project to embedding space
            text_embeds = model.text_projection(text_features)
            text_embeds = F.normalize(text_embeds, dim=1)
            
            embeddings.append(text_embeds.cpu())
    
    return torch.cat(embeddings, dim=0)

def extract_image_features(model, dataloader, device):
    """Extract image features from dataset"""
    model.eval()
    all_features = []
    all_info = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            pixel_values = batch['pixel_values'].to(device)
            
            # Get image features from RadDino
            rad_dino_outputs = model.rad_dino(pixel_values=pixel_values)
            image_features = rad_dino_outputs.last_hidden_state[:, 0]
            image_embeds = model.image_projection(image_features)
            image_embeds = F.normalize(image_embeds, dim=1)
            
            all_features.append(image_embeds.cpu())
            
            # Store batch info
            for i in range(len(batch['image_path'])):
                all_info.append({
                    'image_path': batch['image_path'][i],
                    'label': batch['label'][i].item(),
                    'impression': batch['impression'][i]
                })
    
    return torch.cat(all_features, dim=0), all_info

def compute_similarities(image_features, text_features, text_labels):
    """Compute similarities between images and text"""
    similarities = torch.matmul(image_features, text_features.t())
    
    # Get best matches
    best_scores, best_indices = torch.max(similarities, dim=1)
    best_labels = [text_labels[idx] for idx in best_indices]
    
    return best_labels, best_scores.tolist(), similarities

def main():
    args = parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    df = pd.read_csv(args.data_path)
    test_df = df[df['split'] == 'test'].copy()
    print(f"Test samples: {len(test_df)}")
    
    if len(test_df) == 0:
        print("No test data found!")
        return
    
    # Initialize model components
    model_name = f"google/siglip-base-patch16-{args.image_size}"
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", 
        trust_remote_code=True
    )
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = KneeRadDinoSigLIPModel(siglip_model_name=model_name)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Create dataset
    dataset = SimpleEvalDataset(test_df, processor, args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Get prompts
    oa_prompts = get_oa_prompts()
    all_prompts = []
    prompt_labels = []
    
    for category, prompts in oa_prompts.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_labels.append(category)
    
    print(f"Total prompts: {len(all_prompts)}")
    
    # Encode prompts
    print("Encoding text prompts...")
    text_features = encode_prompts(model, tokenizer, all_prompts, device)
    print(f"Text features shape: {text_features.shape}")
    
    # Extract image features
    print("Extracting image features...")
    image_features, image_info = extract_image_features(model, dataloader, device)
    print(f"Image features shape: {image_features.shape}")
    
    # Compute similarities
    print("Computing similarities...")
    predictions, scores, similarity_matrix = compute_similarities(
        image_features, text_features, prompt_labels
    )
    
    # Create results
    results = []
    for i, info in enumerate(image_info):
        # Get top 3 matches
        sims = similarity_matrix[i]
        top3_scores, top3_indices = torch.topk(sims, k=3)
        
        results.append({
            'image_path': info['image_path'],
            'true_label': info['label'],
            'impression': info['impression'],
            'predicted_category': predictions[i],
            'confidence': scores[i],
            'top1_prompt': all_prompts[top3_indices[0]],
            'top1_score': top3_scores[0].item(),
            'top2_prompt': all_prompts[top3_indices[1]],
            'top2_score': top3_scores[1].item(),
            'top3_prompt': all_prompts[top3_indices[2]],
            'top3_score': top3_scores[2].item(),
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    print(f"Results saved to: {args.output_path}")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total test samples: {len(results)}")
    print(f"Average confidence: {np.mean(scores):.4f}")
    
    # Category distribution
    pred_counts = pd.Series(predictions).value_counts()
    print(f"\nPredicted category distribution:")
    for category, count in pred_counts.items():
        print(f"  {category}: {count}")
    
    # Show some examples
    print(f"\n=== Top 5 Most Confident Predictions ===")
    top_confident = results_df.nlargest(5, 'confidence')
    for _, row in top_confident.iterrows():
        print(f"Image: {os.path.basename(row['image_path'])}")
        print(f"  Predicted: {row['predicted_category']} (confidence: {row['confidence']:.4f})")
        print(f"  Best match: '{row['top1_prompt']}'")
        print()

if __name__ == "__main__":
    main() 