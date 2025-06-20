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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import argparse
warnings.filterwarnings("ignore")

# Import model architecture
from train_siglip import KneeRadDinoSigLIPModel

print("✅ All libraries imported successfully!")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Zero-shot evaluation for knee OA classification')
    
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
    'max_text_length': 64,
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

# Define custom OA prompts for zero-shot classification
def get_custom_oa_prompts():
    """Define custom OA severity prompts for zero-shot classification"""
    return {
        0: [
            "No significant bony abnormality",
        ],
        1: [
            "doubtful OA"
        ],
        2: [
            "minimal OA",
        ],
        3: [
            "moderate OA",
        ],
        4: [
            "severe OA",
        ]
    }

# Define detailed medical prompts for phrase grounding
# Display the prompts
oa_prompts = get_custom_oa_prompts()

print("🔍 Custom OA Prompts:")
for class_id, prompts in oa_prompts.items():
    print(f"  Class {class_id}: {prompts}")

# Load and prepare data
print("📊 Loading and preparing data...")
df = pd.read_csv(CONFIG['data_path'])
df = df.loc[df['kl_severity']!='others'].reset_index(drop=True)
print(f"Total samples in dataset: {len(df)}")

# Filter for test split
test_df = df
print(f"Test samples: {len(test_df)}")

if len(test_df) == 0:
    print("❌ No test samples found! Please check your data.")
else:
    print("✅ Test data loaded successfully!")
    
    # Print label distribution if available
    if 'kl_severity' in test_df.columns:
        print("\n📋 Test label distribution:")
        label_dist = test_df['kl_severity'].value_counts().sort_index()
        for label, count in label_dist.items():
            print(f"  Label {label}: {count} samples")
    
    # Show sample data
    print(f"\n📝 Sample test data:")
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

model.to(device)
model.eval()
print(f"🎯 Model ready on {device}")

# Dataset class for evaluation
class EvaluationDataset(Dataset):
    def __init__(self, df, processor, image_size=512, view_position='ap'):
        self.df = df.dropna(subset=[view_position]).reset_index(drop=True)
        self.processor = processor
        self.image_size = image_size
        self.view_position = view_position
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Load and process image using the specified view position
            image = Image.open(row[self.view_position]).convert('RGB')
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
            
            image_inputs = self.processor(images=image, return_tensors="pt")
            
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'image_path': row[self.view_position],
                'label': row.get('kl_grade', -1),  # Use kl_severity column for KL grade
                'impression': row.get('impression', '')
            }
        except Exception as e:
            print(f"Error loading {row[self.view_position]}: {e}")
            # Return dummy data for failed images
            dummy_image = Image.new('RGB', (self.image_size, self.image_size), color='black')
            image_inputs = self.processor(images=dummy_image, return_tensors="pt")
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'image_path': row[self.view_position],
                'label': row.get('kl_grade', -1),  # Use kl_severity column for KL grade
                'impression': row.get('impression', '')
            }

# Create evaluation dataset and dataloader
eval_dataset = EvaluationDataset(
    test_df, 
    processor, 
    CONFIG['image_size'],
    CONFIG['view_position']
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"📦 Evaluation dataset created:")
print(f"  - View position: {CONFIG['view_position']}")
print(f"  - Total samples: {len(eval_dataset)}")
print(f"  - Batch size: {CONFIG['batch_size']}")
print(f"  - Total batches: {len(eval_loader)}")
print(f"  - Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")

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
                    'label': batch['label'][i].item(),
                    'impression': batch['impression'][i]
                })
    
    return torch.cat(all_embeddings, dim=0), all_info

# Extract embeddings
print("🚀 Starting image embedding extraction...")
image_embeddings, image_info = extract_image_embeddings(model, eval_loader, device)

print(f"✅ Image embeddings extracted!")
print(f"  - Shape: {image_embeddings.shape}")
print(f"  - Device: {image_embeddings.device}")
print(f"  - Info samples: {len(image_info)}")

# Function to encode text prompts
def encode_text_prompts(model, tokenizer, prompts, device, max_length=64):
    """Encode a list of text prompts into embeddings"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for prompt in tqdm(prompts, desc="📝 Encoding text prompts"):
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

# Encode OA severity prompts
print("🔍 Encoding OA severity prompts...")
oa_prompt_embeddings = {}
all_oa_prompts = []
prompt_to_class = {}

for class_id, prompts in oa_prompts.items():
    print(f"  Class {class_id}: {len(prompts)} prompts")
    class_embeddings = encode_text_prompts(
        model, medical_tokenizer, prompts, device, CONFIG['max_text_length']
    )
    oa_prompt_embeddings[class_id] = class_embeddings
    
    # Keep track of prompt to class mapping
    for prompt in prompts:
        all_oa_prompts.append(prompt)
        prompt_to_class[prompt] = class_id

print(f"✅ OA prompts encoded! Total: {len(all_oa_prompts)} prompts")


# Multiclass zero-shot evaluation
def multiclass_zero_shot_evaluation(image_embeddings, oa_prompts, true_labels):
    """Evaluate multiclass zero-shot classification for OA severity"""
    
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
    
    results = {}
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
            'valid_indices': valid_indices,
            'similarities': similarities
        }
    
    return results

# Extract true labels from image info
true_labels = [info['label'] for info in image_info]

# Perform multiclass evaluation
print("🎯 Performing multiclass zero-shot evaluation...")
multiclass_results = multiclass_zero_shot_evaluation(
    image_embeddings, oa_prompt_embeddings, true_labels
)

# Display results
if multiclass_results:
    print(f"✅ Zero-shot Accuracy: {multiclass_results['accuracy']:.4f}")
    print(f"\n📊 Per-class Performance:")
    
    for class_name, metrics in multiclass_results['classification_report'].items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1-score: {metrics['f1-score']:.3f}")
            print(f"    Support: {metrics['support']}")
else:
    print("❌ No valid results - check your data labels")


# Visualize confusion matrix
if multiclass_results and 'confusion_matrix' in multiclass_results:
    plt.figure(figsize=(10, 8))
    
    # Get class names for labels
    class_names = [multiclass_results['class_names'][i] for i in sorted(multiclass_results['class_names'].keys())]
    
    sns.heatmap(
        multiclass_results['confusion_matrix'],
        annot=True,
        fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues',
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - Zero-shot OA Classification', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    confusion_matrix_path = os.path.join(view_output_dir, f'confusion_matrix_{CONFIG["view_position"]}.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    print(f"💾 Confusion matrix saved to: {confusion_matrix_path}")
    
    plt.show()
else:
    print("⚠️ No confusion matrix data available")

# Save evaluation results to file
if multiclass_results:
    results_file = os.path.join(view_output_dir, f'evaluation_results_{CONFIG["view_position"]}.txt')
    with open(results_file, 'w') as f:
        f.write(f"Zero-shot Evaluation Results - {CONFIG['view_position'].upper()} View\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {CONFIG['model_path']}\n")
        f.write(f"View Position: {CONFIG['view_position']}\n")
        f.write(f"Test Samples: {len(multiclass_results['valid_indices'])}\n")
        f.write(f"Accuracy: {multiclass_results['accuracy']:.4f}\n\n")
        
        f.write("Per-class Performance:\n")
        for class_name, metrics in multiclass_results['classification_report'].items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {metrics['precision']:.3f}\n")
                f.write(f"    Recall: {metrics['recall']:.3f}\n")
                f.write(f"    F1-score: {metrics['f1-score']:.3f}\n")
                f.write(f"    Support: {metrics['support']}\n")
    
    print(f"📄 Detailed results saved to: {results_file}")

print(f"\n🎉 Evaluation completed!")
print(f"📁 All results saved in: {view_output_dir}")

# Print usage example
print("\n" + "="*60)
print("📖 USAGE EXAMPLES:")
print("="*60)
print("# Evaluate AP view with default settings:")
print("python zeroshot_evaluation.py --view_position ap")
print()
print("# Evaluate PA Rosen view with custom model:")
print("python zeroshot_evaluation.py --view_position pa_rosen --model_path /path/to/model.pt")
print()
print("# Evaluate Skyline view with custom device and batch size:")
print("python zeroshot_evaluation.py --view_position skyline --device cuda:1 --batch_size 32")
print()
print("# Full custom evaluation:")
print("python zeroshot_evaluation.py \\")
print("    --model_path /custom/path/model.pt \\")
print("    --view_position ap \\")
print("    --data_path custom_test.csv \\")
print("    --image_size 384 \\")
print("    --batch_size 8 \\")
print("    --device cuda:0 \\")
print("    --output_dir custom_results")
print("="*60)
