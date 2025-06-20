#!/usr/bin/env python3
"""
SigLIP Evaluation Script for Zero-shot Classification and Phrase Grounding

This script evaluates a trained SigLIP model on knee X-ray images for:
1. Zero-shot classification with custom medical prompts
2. Phrase grounding analysis
3. Multiclass classification evaluation
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Import model architecture from training script
from train_siglip import KneeRadDinoSigLIPModel

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SigLIP model for zero-shot classification and phrase grounding')
    parser.add_argument('--model_path', type=str, default='/model/workspace/msk/checkpoints/ap/best_model.pt', help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='train.csv', help='Path to data CSV')
    parser.add_argument('--image_size', type=int, choices=[224, 384, 512], default=224, help='Image resolution')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--max_text_length', type=int, default=64, help='Maximum text length for tokenization')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--save_embeddings', action='store_true', help='Save image and text embeddings')
    parser.add_argument('--top_k', type=int, default=5, help='Top-k predictions to consider')
    return parser.parse_args()

class EvaluationDataset(Dataset):
    def __init__(self, df, processor, tokenizer, image_size=224, max_text_length=64):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_text_length = max_text_length
        
        # Filter out rows with missing image paths
        self.df = self.df.dropna(subset=['img_path'])
        print(f"Evaluation dataset size: {len(self.df)}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['img_path']
        
        try:
            # Load and process image
            image = Image.open(img_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            
            # Process image
            image_inputs = self.processor(
                images=image,
                return_tensors="pt"
            )
            
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'image_path': img_path,
                'label': row.get('label', -1),  # Default to -1 if no label
                'impression': row.get('impression', ''),
                'index': idx
            }
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return dummy data
            dummy_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
            image_inputs = self.processor(
                images=dummy_image,
                return_tensors="pt"
            )
            
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'image_path': img_path,
                'label': row.get('label', -1),
                'impression': row.get('impression', ''),
                'index': idx
            }

def get_custom_oa_prompts():
    """Define custom OA severity prompts for zero-shot classification"""
    return {
        0: [
            "normal knee X-ray",
            "healthy knee joint",
            "no signs of osteoarthritis",
            "normal joint space",
            "no degenerative changes"
        ],
        1: [
            "doubtful osteoarthritis",
            "questionable joint space narrowing",
            "minimal degenerative changes",
            "possible early osteoarthritis",
            "borderline osteoarthritis"
        ],
        2: [
            "minimal osteoarthritis",
            "mild joint space narrowing",
            "early degenerative changes",
            "minimal OA",
            "slight osteoarthritis"
        ],
        3: [
            "moderate osteoarthritis",
            "moderate joint space narrowing",
            "moderate degenerative changes",
            "moderate OA",
            "definite osteoarthritis"
        ],
        4: [
            "severe osteoarthritis",
            "severe joint space narrowing",
            "severe degenerative changes",
            "severe OA",
            "advanced osteoarthritis",
            "bone on bone contact"
        ]
    }

def get_detailed_medical_prompts():
    """Get detailed medical prompts for comprehensive evaluation"""
    return [
        # Normal findings
        "normal knee X-ray with preserved joint space",
        "healthy knee joint without degenerative changes",
        "no evidence of osteoarthritis or fracture",
        
        # OA severity levels
        "minimal osteoarthritis with slight joint space narrowing",
        "mild osteoarthritis with osteophyte formation",
        "moderate osteoarthritis with definite joint space narrowing",
        "severe osteoarthritis with bone-on-bone contact",
        
        # Specific findings
        "osteophyte formation at joint margins",
        "subchondral sclerosis and cyst formation",
        "joint space narrowing medial compartment",
        "joint space narrowing lateral compartment",
        "patellofemoral osteoarthritis",
        "tibiofemoral osteoarthritis",
        
        # Complications
        "knee joint effusion",
        "bone marrow edema",
        "meniscal degeneration",
        "ligament abnormalities",
        
        # Fractures
        "tibial plateau fracture",
        "patella fracture",
        "femoral condyle fracture",
        "stress fracture",
        
        # Other pathologies
        "rheumatoid arthritis changes",
        "inflammatory arthritis",
        "septic arthritis",
        "crystal arthropathy"
    ]

def encode_text_prompts(model, tokenizer, prompts, device, max_length=64):
    """Encode a list of text prompts into embeddings"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="Encoding text prompts"):
            # Tokenize text
            text_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            
            # Get text embedding using medical BERT
            text_outputs = model.medical_bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Use [CLS] token representation
            if hasattr(text_outputs, 'pooler_output'):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs.last_hidden_state[:, 0]
            
            # Project to common embedding space
            text_embeds = model.text_projection(text_features)
            text_embeds = F.normalize(text_embeds, dim=1)
            
            all_embeddings.append(text_embeds.cpu())
    
    return torch.cat(all_embeddings, dim=0)

def extract_image_embeddings(model, dataloader, device):
    """Extract image embeddings from the dataset"""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_paths = []
    all_impressions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting image embeddings"):
            pixel_values = batch['pixel_values'].to(device)
            
            # Get image embeddings using RadDino
            rad_dino_outputs = model.rad_dino(pixel_values=pixel_values)
            image_features = rad_dino_outputs.last_hidden_state[:, 0]
            image_embeds = model.image_projection(image_features)
            image_embeds = F.normalize(image_embeds, dim=1)
            
            all_embeddings.append(image_embeds.cpu())
            all_labels.extend(batch['label'].tolist())
            all_paths.extend(batch['image_path'])
            all_impressions.extend(batch['impression'])
    
    return (
        torch.cat(all_embeddings, dim=0),
        all_labels,
        all_paths,
        all_impressions
    )

def zero_shot_classification(image_embeddings, text_embeddings, prompt_labels, top_k=5):
    """Perform zero-shot classification using cosine similarity"""
    # Calculate similarity matrix
    similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
    
    # Get top-k predictions for each image
    top_k_values, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
    
    predictions = []
    confidences = []
    
    for i in range(len(image_embeddings)):
        pred_indices = top_k_indices[i].tolist()
        pred_scores = top_k_values[i].tolist()
        
        pred_labels = [prompt_labels[idx] for idx in pred_indices]
        
        predictions.append({
            'top_predictions': pred_labels,
            'top_scores': pred_scores,
            'best_prediction': pred_labels[0],
            'best_score': pred_scores[0]
        })
        
        confidences.append(pred_scores[0])
    
    return predictions, confidences

def multiclass_zero_shot_evaluation(image_embeddings, oa_prompts, true_labels, device):
    """Evaluate multiclass zero-shot classification for OA severity"""
    results = {}
    
    # Create class-wise embeddings by averaging prompts for each class
    class_embeddings = {}
    class_names = {0: 'Normal', 1: 'Doubtful', 2: 'Minimal', 3: 'Moderate', 4: 'Severe'}
    
    for class_id, prompts in oa_prompts.items():
        # Average embeddings for each class
        class_embeddings[class_id] = prompts.mean(dim=0, keepdim=True)
    
    # Stack class embeddings
    ordered_classes = sorted(class_embeddings.keys())
    stacked_embeddings = torch.cat([class_embeddings[cls] for cls in ordered_classes], dim=0)
    
    # Calculate similarities
    similarities = torch.matmul(image_embeddings, stacked_embeddings.t())
    predicted_classes = similarities.argmax(dim=1).tolist()
    
    # Convert predictions to actual class labels
    predicted_labels = [ordered_classes[pred] for pred in predicted_classes]
    
    # Filter out invalid labels
    valid_indices = [i for i, label in enumerate(true_labels) if label in ordered_classes]
    valid_true_labels = [true_labels[i] for i in valid_indices]
    valid_predictions = [predicted_labels[i] for i in valid_indices]
    
    if len(valid_true_labels) > 0:
        # Calculate metrics
        accuracy = accuracy_score(valid_true_labels, valid_predictions)
        
        # Classification report
        target_names = [class_names[cls] for cls in ordered_classes]
        class_report = classification_report(
            valid_true_labels, 
            valid_predictions,
            target_names=target_names,
            labels=ordered_classes,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(
            valid_true_labels, 
            valid_predictions,
            labels=ordered_classes
        )
        
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'class_names': class_names,
            'predictions': valid_predictions,
            'true_labels': valid_true_labels,
            'valid_indices': valid_indices
        }
    
    return results

def phrase_grounding_analysis(image_embeddings, text_embeddings, prompts, image_paths, top_k=10):
    """Analyze phrase grounding by finding most similar text-image pairs"""
    similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())
    
    grounding_results = []
    
    # For each text prompt, find most similar images
    for prompt_idx, prompt in enumerate(prompts):
        similarities = similarity_matrix[:, prompt_idx]
        top_k_values, top_k_indices = torch.topk(similarities, k=top_k)
        
        grounding_results.append({
            'prompt': prompt,
            'top_similar_images': [
                {
                    'image_path': image_paths[idx.item()],
                    'similarity': score.item()
                }
                for idx, score in zip(top_k_indices, top_k_values)
            ]
        })
    
    return grounding_results

def save_evaluation_results(results, output_dir):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save multiclass evaluation results
    if 'multiclass_results' in results:
        mc_results = results['multiclass_results']
        
        # Save classification report
        pd.DataFrame(mc_results['classification_report']).T.to_csv(
            os.path.join(output_dir, 'classification_report.csv')
        )
        
        # Save confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            mc_results['confusion_matrix'],
            annot=True,
            fmt='d',
            xticklabels=[mc_results['class_names'][i] for i in range(len(mc_results['class_names']))],
            yticklabels=[mc_results['class_names'][i] for i in range(len(mc_results['class_names']))],
            cmap='Blues'
        )
        plt.title('Confusion Matrix - Zero-shot OA Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save phrase grounding results
    if 'phrase_grounding' in results:
        grounding_df = []
        for result in results['phrase_grounding']:
            for img_result in result['top_similar_images']:
                grounding_df.append({
                    'prompt': result['prompt'],
                    'image_path': img_result['image_path'],
                    'similarity': img_result['similarity']
                })
        
        pd.DataFrame(grounding_df).to_csv(
            os.path.join(output_dir, 'phrase_grounding_results.csv'),
            index=False
        )
    
    # Save detailed predictions
    if 'detailed_predictions' in results:
        pd.DataFrame(results['detailed_predictions']).to_csv(
            os.path.join(output_dir, 'detailed_predictions.csv'),
            index=False
        )
    
    print(f"Results saved to: {output_dir}")

def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from: {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Filter for test split
    test_df = df[df['split'] == 'test'].copy()
    print(f"Test samples: {len(test_df)}")
    
    if len(test_df) == 0:
        print("No test samples found! Please check your data.")
        return
    
    # Print label distribution
    if 'label' in test_df.columns:
        print("\nTest label distribution:")
        print(test_df['label'].value_counts().sort_index())
    
    # Initialize processors
    model_name = f"google/siglip-base-patch16-{args.image_size}"
    processor = AutoProcessor.from_pretrained(model_name)
    medical_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedVLP-CXR-BERT-specialized", 
        trust_remote_code=True
    )
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = KneeRadDinoSigLIPModel(siglip_model_name=model_name)
    
    # Load checkpoint
    if os.path.isfile(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded from checkpoint successfully!")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("Model loaded from weights successfully!")
        else:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    model.to(device)
    model.eval()
    
    # Create evaluation dataset
    eval_dataset = EvaluationDataset(
        test_df, 
        processor, 
        medical_tokenizer,
        image_size=args.image_size,
        max_text_length=args.max_text_length
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Evaluation batches: {len(eval_loader)}")
    
    # Extract image embeddings
    print("\n=== Extracting Image Embeddings ===")
    image_embeddings, true_labels, image_paths, impressions = extract_image_embeddings(
        model, eval_loader, device
    )
    print(f"Extracted embeddings for {len(image_embeddings)} images")
    
    # Get custom prompts
    oa_prompts = get_custom_oa_prompts()
    detailed_prompts = get_detailed_medical_prompts()
    
    # Encode OA severity prompts
    print("\n=== Encoding OA Severity Prompts ===")
    oa_prompt_embeddings = {}
    all_oa_prompts = []
    prompt_to_class = {}
    
    for class_id, prompts in oa_prompts.items():
        class_embeddings = encode_text_prompts(
            model, medical_tokenizer, prompts, device, args.max_text_length
        )
        oa_prompt_embeddings[class_id] = class_embeddings
        
        # Keep track of prompt to class mapping
        for prompt in prompts:
            all_oa_prompts.append(prompt)
            prompt_to_class[prompt] = class_id
    
    # Encode detailed medical prompts
    print("\n=== Encoding Detailed Medical Prompts ===")
    detailed_embeddings = encode_text_prompts(
        model, medical_tokenizer, detailed_prompts, device, args.max_text_length
    )
    
    # Multiclass zero-shot evaluation
    print("\n=== Multiclass Zero-shot Evaluation ===")
    multiclass_results = multiclass_zero_shot_evaluation(
        image_embeddings, oa_prompt_embeddings, true_labels, device
    )
    
    if multiclass_results:
        print(f"Zero-shot Accuracy: {multiclass_results['accuracy']:.4f}")
        print("\nPer-class Performance:")
        for class_name, metrics in multiclass_results['classification_report'].items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                print(f"  {class_name}: F1={metrics['f1-score']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")
    
    # Zero-shot classification with all OA prompts
    print("\n=== Zero-shot Classification with All Prompts ===")
    all_oa_embeddings = torch.cat([emb for emb in oa_prompt_embeddings.values()], dim=0)
    predictions, confidences = zero_shot_classification(
        image_embeddings, all_oa_embeddings, all_oa_prompts, top_k=args.top_k
    )
    
    # Phrase grounding analysis
    print("\n=== Phrase Grounding Analysis ===")
    phrase_grounding_results = phrase_grounding_analysis(
        image_embeddings, detailed_embeddings, detailed_prompts, image_paths, top_k=10
    )
    
    # Prepare detailed predictions
    detailed_predictions = []
    for i, (pred, conf, true_label, img_path, impression) in enumerate(
        zip(predictions, confidences, true_labels, image_paths, impressions)
    ):
        detailed_predictions.append({
            'image_path': img_path,
            'true_label': true_label,
            'predicted_class': prompt_to_class.get(pred['best_prediction'], -1),
            'predicted_prompt': pred['best_prediction'],
            'confidence': pred['best_score'],
            'top_k_predictions': pred['top_predictions'],
            'top_k_scores': pred['top_scores'],
            'original_impression': impression
        })
    
    # Save results
    results = {
        'multiclass_results': multiclass_results,
        'phrase_grounding': phrase_grounding_results,
        'detailed_predictions': detailed_predictions
    }
    
    # Save embeddings if requested
    if args.save_embeddings:
        torch.save({
            'image_embeddings': image_embeddings,
            'oa_prompt_embeddings': oa_prompt_embeddings,
            'detailed_prompt_embeddings': detailed_embeddings,
            'image_paths': image_paths,
            'true_labels': true_labels,
            'oa_prompts': oa_prompts,
            'detailed_prompts': detailed_prompts
        }, os.path.join(args.output_dir, 'embeddings.pt'))
    
    save_evaluation_results(results, args.output_dir)
    
    print(f"\n🎉 Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    
    # Print summary
    if multiclass_results:
        print(f"\n📊 Summary:")
        print(f"  - Test samples: {len(true_labels)}")
        print(f"  - Zero-shot accuracy: {multiclass_results['accuracy']:.4f}")
        print(f"  - Average confidence: {np.mean(confidences):.4f}")
        print(f"  - Phrase grounding prompts: {len(detailed_prompts)}")

if __name__ == "__main__":
    main() 