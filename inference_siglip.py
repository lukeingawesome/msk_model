#!/usr/bin/env python3
"""
SigLIP Inference Script for Knee X-ray Analysis

This script loads a trained SigLIP model and performs inference on knee X-ray images.
"""

import torch
import torch.nn.functional as F
from transformers import SiglipProcessor
from PIL import Image
import pandas as pd
import numpy as np
import argparse
from train_siglip import KneeRadDinoSigLIPModel
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with trained SigLIP model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--image_path', type=str, help='Path to single image for inference')
    parser.add_argument('--csv_path', type=str, help='Path to CSV file for batch inference')
    parser.add_argument('--image_size', type=int, choices=[224, 448], default=224, help='Image resolution')
    parser.add_argument('--output_path', type=str, default='inference_results.csv', help='Output CSV path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--max_text_length', type=int, default=512, help='Maximum text length')
    return parser.parse_args()

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor, image_size=224, max_text_length=512):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.image_size = image_size
        self.max_text_length = max_text_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Process image
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                size={"height": self.image_size, "width": self.image_size}
            )
            
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'image_path': img_path,
                'index': idx
            }
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return dummy data
            dummy_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
            inputs = self.processor(
                images=dummy_image,
                return_tensors="pt",
                size={"height": self.image_size, "width": self.image_size}
            )
            
            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'image_path': img_path,
                'index': idx
            }

def compute_similarity(image_embeds, text_embeds):
    """Compute cosine similarity between image and text embeddings"""
    image_embeds = F.normalize(image_embeds, dim=1)
    text_embeds = F.normalize(text_embeds, dim=1)
    
    similarity = torch.matmul(image_embeds, text_embeds.t())
    return similarity

def single_image_inference(model, processor, image_path, text_queries, device, image_size=224):
    """Perform inference on a single image with multiple text queries"""
    model.eval()
    
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    
    results = []
    
    with torch.no_grad():
        # Process image once
        image_inputs = processor(
            images=image,
            return_tensors="pt",
            size={"height": image_size, "width": image_size}
        )
        pixel_values = image_inputs['pixel_values'].to(device)
        
        # Get image embedding using RadDino
        rad_dino_outputs = model.rad_dino(pixel_values=pixel_values)
        image_features = rad_dino_outputs.last_hidden_state[:, 0]
        image_embeds = model.image_projection(image_features)
        image_embeds = F.normalize(image_embeds, dim=1)
        
        # Process each text query
        for text_query in text_queries:
            text_inputs = processor(
                text=text_query,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512
            )
            
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            
            # Get text embedding using SigLIP text encoder
            text_outputs = model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            text_embeds = F.normalize(text_outputs.pooler_output, dim=1)
            
            # Calculate similarity
            similarity = torch.matmul(image_embeds, text_embeds.t())
            
            results.append({
                'text_query': text_query,
                'similarity_score': similarity.item()
            })
    
    return results

def batch_inference(model, processor, df, device, args):
    """Perform batch inference on a DataFrame"""
    model.eval()
    
    # Create dataset and dataloader
    dataset = InferenceDataset(df, processor, args.image_size, args.max_text_length)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    all_image_embeds = []
    image_paths = []
    indices = []
    
    print(f"Processing {len(dataset)} images...")
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            
            # Get image embeddings using RadDino
            rad_dino_outputs = model.rad_dino(pixel_values=pixel_values)
            image_features = rad_dino_outputs.last_hidden_state[:, 0]
            image_embeds = model.image_projection(image_features)
            image_embeds = F.normalize(image_embeds, dim=1)
            
            all_image_embeds.append(image_embeds.cpu())
            image_paths.extend(batch['image_path'])
            indices.extend(batch['index'].tolist())
    
    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'image_path': image_paths,
        'original_index': indices
    })
    
    # Add embeddings as separate columns (optional)
    embedding_dim = all_image_embeds.shape[1]
    for i in range(embedding_dim):
        results_df[f'embed_{i}'] = all_image_embeds[:, i].numpy()
    
    return results_df, all_image_embeds

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    
    # Determine model name based on image size
    model_name = f"google/siglip-base-patch16-{args.image_size}"
    
    # Initialize model
    model = KneeRadDinoSigLIPModel(siglip_model_name=model_name)
    
    # Load checkpoint
    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume it's just the state dict
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model.to(device)
    model.eval()
    
    # Initialize processor
    processor = SiglipProcessor.from_pretrained(model_name)
    
    if args.image_path:
        print(f"\nPerforming single image inference on: {args.image_path}")
        
        # Define some common knee X-ray text queries
        text_queries = [
            "Normal knee X-ray",
            "Osteoarthritis of the knee",
            "Knee fracture",
            "Joint space narrowing",
            "Bone spur formation",
            "Severe osteoarthritis",
            "Mild osteoarthritis",
            "Doubtful osteoarthritis",
            "Knee joint effusion",
            "Patellofemoral arthritis"
        ]
        
        results = single_image_inference(
            model, processor, args.image_path, text_queries, device, args.image_size
        )
        
        print("\nSimilarity scores:")
        for result in sorted(results, key=lambda x: x['similarity_score'], reverse=True):
            print(f"'{result['text_query']}': {result['similarity_score']:.4f}")
    
    elif args.csv_path:
        print(f"\nPerforming batch inference on: {args.csv_path}")
        
        # Load CSV
        df = pd.read_csv(args.csv_path)
        print(f"Loaded {len(df)} samples")
        
        # Perform batch inference
        results_df, embeddings = batch_inference(model, processor, df, device, args)
        
        # Save results
        results_df.to_csv(args.output_path, index=False)
        print(f"Results saved to: {args.output_path}")
        
        # Save embeddings separately (optional)
        embedding_path = args.output_path.replace('.csv', '_embeddings.pt')
        torch.save(embeddings, embedding_path)
        print(f"Embeddings saved to: {embedding_path}")
        
    else:
        print("Please provide either --image_path for single image inference or --csv_path for batch inference")

if __name__ == "__main__":
    main() 