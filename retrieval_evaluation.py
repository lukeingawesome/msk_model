# Import required libraries
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
warnings.filterwarnings("ignore")

# Import model architecture
from train_siglip import KneeRadDinoSigLIPModel

print("✅ All libraries imported successfully!")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Retrieval evaluation for knee radiology reports')
    
    parser.add_argument('--model_path', 
                       type=str, 
                       default='/model/workspace/msk/checkpoints/ap/best_model_weights.pt',
                       help='Path to the trained model weights')
    
    parser.add_argument('--view_position', 
                       type=str, 
                       choices=['ap', 'pa_rosen', 'skyline'],
                       default='ap',
                       help='View position to evaluate (ap, pa_rosen, or skyline)')
    
    parser.add_argument('--data_path', 
                       type=str, 
                       default='test.csv',
                       help='Path to the test data CSV file')
    
    parser.add_argument('--image_size', 
                       type=int, 
                       default=512,
                       help='Input image size for the model')
    
    parser.add_argument('--batch_size', 
                       type=int, 
                       default=16,
                       help='Batch size for evaluation')
    
    parser.add_argument('--device', 
                       type=str, 
                       default='cuda:2',
                       help='Device to use for evaluation (e.g., cuda:0, cuda:1, cpu)')
    
    parser.add_argument('--output_dir', 
                       type=str, 
                       default='evaluation_results',
                       help='Directory to save evaluation results')
    
    return parser.parse_args()

# Parse arguments
args = parse_arguments()

# Configuration - Use parsed arguments
CONFIG = {
    'model_path': args.model_path,
    'view_position': args.view_position,
    'data_path': args.data_path,
    'image_size': args.image_size,
    'batch_size': args.batch_size,
    'max_text_length': 64,  # Match training text length, not 512
    'device': args.device,
    'output_dir': args.output_dir
}

# Print configuration
print("🔧 Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Set device
device = torch.device(CONFIG['device'] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory with view-specific subdirectory
view_output_dir = os.path.join(CONFIG['output_dir'], CONFIG['view_position'])
os.makedirs(view_output_dir, exist_ok=True)
print(f"Results will be saved to: {view_output_dir}")

# Helper function to get view prompts (matching training format)
def get_oa_prompts(view_position):
    """Get view-specific prompts that match training format"""
    view_prompts = {
        'ap': "This is a ap view of a knee x-ray. ",
        'pa_rosen': "This is a pa view of a knee x-ray. ", 
        'skyline': "This is a skyline view of a knee x-ray. "
    }
    return view_prompts.get(view_position, "This is a unknown view of a knee x-ray. ")

# Load and prepare data
print("📊 Loading and preparing data...")
df = pd.read_csv(CONFIG['data_path'])

# Filter out rows without impressions and the specified view
df = df.dropna(subset=[CONFIG['view_position'], 'impression']).reset_index(drop=True)
# Remove empty impressions
df = df[df['impression'].str.strip() != ''].reset_index(drop=True)

print(f"Total samples with both {CONFIG['view_position']} images and impressions: {len(df)}")

if len(df) == 0:
    print("❌ No samples found with both images and impressions! Please check your data.")
    exit()
else:
    print("✅ Data loaded successfully!")
    # NOTE: Training used NO view prompts (oa_prompt = ''), so we match that format
    print(f"📝 Training format: impression text only (NO view prompts)")
    print(f"📝 Sample text format: {df['impression'].iloc[0][:100]}...")

# Initialize model and processors
print("🔧 Initializing model and processors...")

# Initialize processors with correct image size
model_name = f"google/siglip-base-patch16-{CONFIG['image_size']}"
print(f"Using SigLIP model: {model_name}")

try:
    processor = AutoProcessor.from_pretrained(model_name)
    print("✅ SigLIP processor loaded")
except:
    # Fallback to smaller size if 512 not available
    model_name = "google/siglip-base-patch16-384"
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"⚠️  Fallback to: {model_name}")

# Initialize medical tokenizer
medical_tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", 
    trust_remote_code=True
)
print("✅ Medical BERT tokenizer loaded")

# Load model
print(f"🔄 Loading model from: {CONFIG['model_path']}")
model = KneeRadDinoSigLIPModel(siglip_model_name=model_name)

# Load checkpoint
if os.path.isfile(CONFIG['model_path']):
    checkpoint = torch.load(CONFIG['model_path'], map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded from checkpoint!")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
        print("✅ Model loaded from weights!")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Model loaded!")
else:
    print(f"❌ Model file not found: {CONFIG['model_path']}")
    exit()

model.to(device)
model.eval()
print(f"🎯 Model ready on {device}")

# Dataset class for retrieval evaluation
class RetrievalDataset(Dataset):
    def __init__(self, df, processor, image_size=512, view_position='ap'):
        self.df = df
        self.processor = processor
        self.image_size = image_size
        self.view_position = view_position
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Load and process image
            image = Image.open(row[self.view_position]).convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            
            image_inputs = self.processor(images=image, return_tensors="pt")
            
            # Format impression WITHOUT view prompt to match training format
            formatted_impression = str(row['impression'])  # No view prompt added
            
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'image_path': row[self.view_position],
                'impression': formatted_impression,  # Now includes view prompt
                'pid': row.get('pid', idx),  # Use pid if available, else use index
                'index': idx
            }
        except Exception as e:
            print(f"Error loading {row[self.view_position]}: {e}")
            # Return dummy data for failed images
            dummy_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
            image_inputs = self.processor(images=dummy_image, return_tensors="pt")
            # Use dummy text WITHOUT view prompt to match training format
            dummy_impression = "Normal knee X-ray"  # No view prompt added
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'image_path': row[self.view_position],
                'impression': dummy_impression,  # Formatted dummy text
                'pid': row.get('pid', idx),
                'index': idx
            }

# Create dataset and dataloader
dataset = RetrievalDataset(
    df, 
    processor, 
    CONFIG['image_size'],
    CONFIG['view_position']
)
dataloader = DataLoader(
    dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"📦 Retrieval dataset created:")
print(f"  - View position: {CONFIG['view_position']}")
print(f"  - Total samples: {len(dataset)}")
print(f"  - Batch size: {CONFIG['batch_size']}")

# Extract image embeddings
def extract_image_embeddings(model, dataloader, device):
    """Extract image embeddings from the dataset"""
    model.eval()
    all_embeddings = []
    all_info = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="🖼️  Extracting image embeddings"):
            pixel_values = batch['pixel_values'].to(device)
            
            # Get RadDino image features
            rad_dino_outputs = model.rad_dino(pixel_values=pixel_values)
            image_features = rad_dino_outputs.last_hidden_state[:, 0]  # CLS token
            
            # Project to common embedding space
            image_embeds = model.image_projection(image_features)
            image_embeds = F.normalize(image_embeds, dim=1)
            
            all_embeddings.append(image_embeds.cpu())
            
            # Store batch info
            for i in range(len(batch['image_path'])):
                all_info.append({
                    'image_path': batch['image_path'][i],
                    'impression': batch['impression'][i],
                    'pid': batch['pid'][i],
                    'index': batch['index'][i].item()
                })
    
    return torch.cat(all_embeddings, dim=0), all_info

# Extract text embeddings
def extract_text_embeddings(model, tokenizer, texts, device, max_length=512, batch_size=16):
    """Extract text embeddings from impressions"""
    model.eval()
    all_embeddings = []
    
    # Process texts in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="📝 Extracting text embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            text_inputs = tokenizer(
                batch_texts,
                return_tensors="pt", 
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            
            # Get text embeddings
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

# Extract embeddings
print("🚀 Starting embedding extraction...")
image_embeddings, data_info = extract_image_embeddings(model, dataloader, device)

# Extract text impressions
impressions = [info['impression'] for info in data_info]
text_embeddings = extract_text_embeddings(
    model, medical_tokenizer, impressions, device, 
    CONFIG['max_text_length'], CONFIG['batch_size']
)

print(f"✅ Embeddings extracted!")
print(f"  - Image embeddings shape: {image_embeddings.shape}")
print(f"  - Text embeddings shape: {text_embeddings.shape}")

# Retrieval evaluation functions
def calculate_retrieval_metrics(similarities, k_values=[1, 5, 10]):
    """Calculate Recall@K for retrieval task"""
    n_queries = similarities.shape[0]
    
    # Get ranking indices (sorted by similarity, descending)
    rankings = torch.argsort(similarities, dim=1, descending=True)
    
    metrics = {}
    for k in k_values:
        # Check if ground truth (diagonal) is in top-k
        correct_at_k = 0
        for i in range(n_queries):
            if i in rankings[i, :k]:  # Ground truth is at index i for query i
                correct_at_k += 1
        
        recall_at_k = correct_at_k / n_queries
        metrics[f'R@{k}'] = recall_at_k
    
    return metrics

def perform_retrieval_evaluation(image_embeddings, text_embeddings, data_info):
    """Perform both i2t and t2i retrieval evaluation"""
    
    # Calculate similarity matrix (images x texts)
    similarities = torch.matmul(image_embeddings, text_embeddings.t())
    
    print("🎯 Performing retrieval evaluation...")
    
    # Image-to-Text retrieval (i2t)
    print("📄 Image-to-Text Retrieval:")
    i2t_metrics = calculate_retrieval_metrics(similarities)
    
    # Text-to-Image retrieval (t2i) 
    print("🖼️  Text-to-Image Retrieval:")
    t2i_similarities = similarities.t()  # Transpose for t2i
    t2i_metrics = calculate_retrieval_metrics(t2i_similarities)
    
    # Create i2t retrieval results DataFrame
    i2t_results = []
    rankings = torch.argsort(similarities, dim=1, descending=True)
    
    for i, info in enumerate(data_info):
        # Get top-1 retrieved text
        top1_idx = rankings[i, 0].item()
        retrieved_impression = data_info[top1_idx]['impression']
        
        i2t_results.append({
            'img_path': info['image_path'],
            'pid': info['pid'],
            'ground_truth': info['impression'],
            'retrieved_report_top1': retrieved_impression,
            'similarity_score': similarities[i, top1_idx].item(),
            'rank_of_gt': (rankings[i] == i).nonzero(as_tuple=True)[0].item() + 1
        })
    
    i2t_df = pd.DataFrame(i2t_results)
    
    return i2t_metrics, t2i_metrics, similarities, i2t_df

# Perform retrieval evaluation
i2t_metrics, t2i_metrics, similarity_matrix, i2t_results_df = perform_retrieval_evaluation(
    image_embeddings, text_embeddings, data_info
)

# Display results
print("\n" + "="*80)
print(f"🎯 RETRIEVAL EVALUATION RESULTS - {CONFIG['view_position'].upper()} VIEW".center(80))
print("="*80)

# Image-to-Text Results
print("\n📄 IMAGE-TO-TEXT RETRIEVAL:")
print("┌" + "─"*15 + "┬" + "─"*15 + "┐")
print("│    Metric     │     Score     │")
print("├" + "─"*15 + "┼" + "─"*15 + "┤")
for metric, score in i2t_metrics.items():
    print(f"│ {metric:<13} │    {score:.4f}    │")
print("└" + "─"*15 + "┴" + "─"*15 + "┘")

# Text-to-Image Results  
print("\n🖼️  TEXT-TO-IMAGE RETRIEVAL:")
print("┌" + "─"*15 + "┬" + "─"*15 + "┐")
print("│    Metric     │     Score     │")
print("├" + "─"*15 + "┼" + "─"*15 + "┤")
for metric, score in t2i_metrics.items():
    print(f"│ {metric:<13} │    {score:.4f}    │")
print("└" + "─"*15 + "┴" + "─"*15 + "┘")

# Summary Statistics
print(f"\n💡 SUMMARY:")
print(f"   • Total samples: {len(data_info)}")
print(f"   • View position: {CONFIG['view_position'].upper()}")
print(f"   • Image-to-Text R@1: {i2t_metrics['R@1']:.4f} ({i2t_metrics['R@1']*100:.2f}%)")
print(f"   • Text-to-Image R@1: {t2i_metrics['R@1']:.4f} ({t2i_metrics['R@1']*100:.2f}%)")

# Save i2t results to CSV
i2t_csv_path = os.path.join(view_output_dir, f'i2t_retrieval_results_{CONFIG["view_position"]}.csv')
i2t_results_df.to_csv(i2t_csv_path, index=False)
print(f"\n💾 I2T retrieval results saved to: {i2t_csv_path}")

# Save detailed results
results_file = os.path.join(view_output_dir, f'retrieval_results_{CONFIG["view_position"]}.txt')
with open(results_file, 'w') as f:
    f.write(f"Retrieval Evaluation Results - {CONFIG['view_position'].upper()} View\n")
    f.write("=" * 60 + "\n")
    f.write(f"Model: {CONFIG['model_path']}\n")
    f.write(f"View Position: {CONFIG['view_position']}\n")
    f.write(f"Total Samples: {len(data_info)}\n\n")
    
    f.write("Image-to-Text Retrieval:\n")
    for metric, score in i2t_metrics.items():
        f.write(f"  {metric}: {score:.4f}\n")
    
    f.write("\nText-to-Image Retrieval:\n")
    for metric, score in t2i_metrics.items():
        f.write(f"  {metric}: {score:.4f}\n")

print(f"📄 Detailed results saved to: {results_file}")

# Visualize similarity matrix (sample)
if len(data_info) <= 50:  # Only for small datasets
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix.numpy(),
        cmap='viridis',
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title('Image-Text Similarity Matrix', fontsize=16)
    plt.xlabel('Text Index', fontsize=14)
    plt.ylabel('Image Index', fontsize=14)
    
    sim_matrix_path = os.path.join(view_output_dir, f'similarity_matrix_{CONFIG["view_position"]}.png')
    plt.savefig(sim_matrix_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Similarity matrix saved to: {sim_matrix_path}")

print(f"\n🎉 Retrieval evaluation completed!")
print(f"📁 All results saved in: {view_output_dir}")

# Analysis comparison with zero-shot
print("\n" + "="*80)
print("🔍 ANALYSIS: WHY RETRIEVAL VS ZERO-SHOT PERFORMANCE DIFFERS".center(80))
print("="*80)
print("\n📊 KEY DIFFERENCES:")
print(f"   Retrieval Task:")
print(f"   • Text format: '[impression only]' (max_length={CONFIG['max_text_length']})")
print(f"   • Task: Find matching impression among {len(data_info)} candidates")
print(f"   • Metric: Recall@K (whether correct match is in top-K)")
print(f"   • Difficulty: Must distinguish between similar medical texts")
print(f"")
print(f"   Zero-shot Task:")
print(f"   • Text format: Short prompts like 'doubtful OA', 'severe OA'")
print(f"   • Task: Classify into 5 predefined OA severity classes")
print(f"   • Metric: Classification accuracy")
print(f"   • Difficulty: Much easier with distinct, short class prompts")
print(f"")
print(f"💡 INSIGHTS:")
print(f"   • Retrieval is inherently harder (1-of-{len(data_info)} vs 1-of-5)")
print(f"   • Model was trained WITHOUT view prompts (oa_prompt = '')")
print(f"   • Evaluation now matches training format (no view prompts)")
print(f"   • Text length affects embedding quality")
print(f"   • Similar impressions make retrieval challenging")
print(f"   • Zero-shot benefits from distinct, short class prompts")

# Print usage examples
print("\n" + "="*60)
print("📖 USAGE EXAMPLES:")
print("="*60)
print("# Evaluate AP view retrieval:")
print("python retrieval_evaluation.py --view_position ap")
print()
print("# Evaluate with custom model and settings:")
print("python retrieval_evaluation.py \\")
print("    --model_path /path/to/model.pt \\")
print("    --view_position pa_rosen \\")
print("    --batch_size 32 \\")
print("    --device cuda:1")
print()
print("# Full custom evaluation:")
print("python retrieval_evaluation.py \\")
print("    --model_path /custom/path/model.pt \\")
print("    --view_position skyline \\")
print("    --data_path custom_test.csv \\")
print("    --image_size 384 \\")
print("    --batch_size 8 \\")
print("    --device cuda:0 \\")
print("    --output_dir custom_results")
print("="*60) 