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
    parser = argparse.ArgumentParser(description='Zero-shot evaluation for knee OA classification using channel-stacked images')
    
    parser.add_argument('--model_path', 
                       type=str, 
                       default='/model/workspace/msk/checkpoints/ap/best_model_weights.pt',
                       help='Path to the trained model weights')
    
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

# Create output directory for channel-stacked evaluation
channel_output_dir = os.path.join(CONFIG['output_dir'], 'channel_stacked')
os.makedirs(channel_output_dir, exist_ok=True)
print(f"Results will be saved to: {channel_output_dir}")

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
    def __init__(self, df, processor, image_size=512):
        # Filter for rows that have at least one valid image path
        required_columns = ['ap', 'pa_rosen', 'skyline', 'lat1', 'lat2']
        available_columns = [col for col in required_columns if col in df.columns]
        
        # Keep rows that have at least one non-null image path
        valid_mask = df[available_columns].notna().any(axis=1)
        self.df = df[valid_mask].reset_index(drop=True)
        
        self.processor = processor
        self.image_size = image_size
        
    def __len__(self):
        return len(self.df)
    
    def create_channel_stacked_image(self, row):
        """Create a 3-channel image by stacking AP, PA, and Skyline views"""
        try:
            # Get image paths
            ap_path = row.get('ap', None)
            pa_path = row.get('pa_rosen', None) 
            skyline_path = row.get('skyline', None)
            
            channels = []
            
            # Load and process AP image as channel 0
            if pd.notna(ap_path) and os.path.exists(ap_path):
                ap_img = Image.open(ap_path).convert('L')  # Convert to grayscale
                ap_img = ap_img.resize((self.image_size, self.image_size), Image.LANCZOS)
                ap_channel = np.array(ap_img)
            else:
                # Create empty channel if image doesn't exist
                ap_channel = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            channels.append(ap_channel)
            
            # Load and process PA image as channel 1
            if pd.notna(pa_path) and os.path.exists(pa_path):
                pa_img = Image.open(pa_path).convert('L')  # Convert to grayscale
                pa_img = pa_img.resize((self.image_size, self.image_size), Image.LANCZOS)
                pa_channel = np.array(pa_img)
            else:
                # Create empty channel if image doesn't exist
                pa_channel = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            channels.append(pa_channel)
            
            # Load and process Skyline image as channel 2
            if pd.notna(skyline_path) and os.path.exists(skyline_path):
                skyline_img = Image.open(skyline_path).convert('L')  # Convert to grayscale
                skyline_img = skyline_img.resize((self.image_size, self.image_size), Image.LANCZOS)
                skyline_channel = np.array(skyline_img)
            else:
                # Create empty channel if image doesn't exist
                skyline_channel = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            channels.append(skyline_channel)
            
            # Stack channels to create 3-channel image (H, W, 3)
            image_array = np.stack(channels, axis=2)
            
            # Convert back to PIL Image
            image = Image.fromarray(image_array, mode='RGB')
            
            return image
            
        except Exception as e:
            print(f"Error creating channel-stacked image: {e}")
            # Return black image on error
            return Image.new('RGB', (self.image_size, self.image_size), color='black')
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Create channel-stacked image
        image = self.create_channel_stacked_image(row)
        
        # Process image
        image_inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'image_path': f"channel_stacked_{idx}",  # Identifier for channel-stacked image
            'label': row.get('kl_grade', -1),
            'impression': row.get('impression', '')
        }

# Create evaluation dataset and dataloader
eval_dataset = EvaluationDataset(
    test_df, 
    processor, 
    CONFIG['image_size']
)
eval_loader = DataLoader(
    eval_dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"📦 Evaluation dataset created:")
print(f"  - Using channel-stacked images (AP=Channel0, PA=Channel1, Skyline=Channel2)")
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

# Display results in a beautiful format
if multiclass_results:
    # Overall accuracy display
    accuracy = multiclass_results['accuracy']
    print("\n" + "="*80)
    print("🎯 ZERO-SHOT EVALUATION RESULTS - CHANNEL STACKED".center(80))
    print("="*80)
    print(f"🔥 OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)".center(80))
    print("="*80)
    
    # Extract overall metrics
    class_report = multiclass_results['classification_report']
    macro_avg = class_report.get('macro avg', {})
    weighted_avg = class_report.get('weighted avg', {})
    
    # Summary metrics table
    print("\n📊 SUMMARY METRICS:")
    print("┌" + "─"*25 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*12 + "┐")
    print("│         Metric          │  Macro Avg  │ Weighted Avg │   Overall   │")
    print("├" + "─"*25 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*12 + "┤")
    print(f"│ Precision               │    {macro_avg.get('precision', 0):.3f}    │    {weighted_avg.get('precision', 0):.3f}     │     N/A     │")
    print(f"│ Recall                  │    {macro_avg.get('recall', 0):.3f}    │    {weighted_avg.get('recall', 0):.3f}     │     N/A     │")
    print(f"│ F1-Score                │    {macro_avg.get('f1-score', 0):.3f}    │    {weighted_avg.get('f1-score', 0):.3f}     │     N/A     │")
    print(f"│ Accuracy                │     N/A     │     N/A      │    {accuracy:.3f}    │")
    print("└" + "─"*25 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*12 + "┘")
    
    # Per-class performance table
    print("\n🏆 PER-CLASS PERFORMANCE:")
    print("┌" + "─"*15 + "┬" + "─"*11 + "┬" + "─"*11 + "┬" + "─"*11 + "┬" + "─"*10 + "┐")
    print("│    Class       │ Precision  │   Recall   │  F1-Score  │ Support  │")
    print("├" + "─"*15 + "┼" + "─"*11 + "┼" + "─"*11 + "┼" + "─"*11 + "┼" + "─"*10 + "┤")
    
    # Sort classes by their numeric value for proper order
    class_names = multiclass_results['class_names']
    ordered_classes = sorted(class_names.keys())
    
    for class_id in ordered_classes:
        class_name = class_names[class_id]
        if class_name in class_report:
            metrics = class_report[class_name]
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                precision = metrics['precision']
                recall = metrics['recall']
                f1_score = metrics['f1-score']
                support = metrics['support']
                
                # Add emoji indicators for performance levels
                f1_emoji = "🟢" if f1_score >= 0.8 else "🟡" if f1_score >= 0.6 else "🔴"
                
                print(f"│ {f1_emoji} {class_name:<12} │   {precision:.3f}    │   {recall:.3f}    │   {f1_score:.3f}    │   {support:>4}   │")
    
    print("└" + "─"*15 + "┴" + "─"*11 + "┴" + "─"*11 + "┴" + "─"*11 + "┴" + "─"*10 + "┘")
    
    # Performance indicators
    print("\n📈 PERFORMANCE INDICATORS:")
    print("🟢 Excellent (F1 ≥ 0.8)   🟡 Good (0.6 ≤ F1 < 0.8)   🔴 Needs Improvement (F1 < 0.6)")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    best_class = max(ordered_classes, key=lambda x: class_report.get(class_names[x], {}).get('f1-score', 0))
    worst_class = min(ordered_classes, key=lambda x: class_report.get(class_names[x], {}).get('f1-score', 0))
    best_f1 = class_report.get(class_names[best_class], {}).get('f1-score', 0)
    worst_f1 = class_report.get(class_names[worst_class], {}).get('f1-score', 0)
    
    print(f"   • Best performing class: {class_names[best_class]} (F1: {best_f1:.3f})")
    print(f"   • Most challenging class: {class_names[worst_class]} (F1: {worst_f1:.3f})")
    print(f"   • Total test samples: {len(multiclass_results['valid_indices'])}")
    print(f"   • Macro F1-Score: {macro_avg.get('f1-score', 0):.3f}")
    print(f"   • Weighted F1-Score: {weighted_avg.get('f1-score', 0):.3f}")
    
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
    
    plt.title('Confusion Matrix - Zero-shot OA Classification (Channel Stacked)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    confusion_matrix_path = os.path.join(channel_output_dir, 'confusion_matrix_channel_stacked.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*80)
    print("💾 CONFUSION MATRIX SAVED".center(80))
    print("="*80)
    print(f"📊 File: {confusion_matrix_path}".center(80))
    print("="*80)
    
    plt.show()
else:
    print("⚠️ No confusion matrix data available")

# Save evaluation results to file
if multiclass_results:
    results_file = os.path.join(channel_output_dir, 'evaluation_results_channel_stacked.txt')
    
    # Extract metrics for beautiful file formatting
    accuracy = multiclass_results['accuracy']
    class_report = multiclass_results['classification_report']
    macro_avg = class_report.get('macro avg', {})
    weighted_avg = class_report.get('weighted avg', {})
    class_names = multiclass_results['class_names']
    ordered_classes = sorted(class_names.keys())
    
    with open(results_file, 'w') as f:
        # Beautiful header for saved file
        f.write("="*80 + "\n")
        f.write("🎯 ZERO-SHOT EVALUATION RESULTS - CHANNEL STACKED\n".center(80))
        f.write("="*80 + "\n\n")
        
        # Model information
        f.write("📋 MODEL INFORMATION:\n")
        f.write("─"*50 + "\n")
        f.write(f"Model Path: {CONFIG['model_path']}\n")
        f.write(f"View Type: Channel Stacked (AP=Ch0, PA=Ch1, Skyline=Ch2)\n")
        f.write(f"Test Samples: {len(multiclass_results['valid_indices'])}\n")
        f.write(f"Image Size: {CONFIG['image_size']}x{CONFIG['image_size']}\n")
        f.write(f"Batch Size: {CONFIG['batch_size']}\n")
        f.write(f"Device: {CONFIG['device']}\n\n")
        
        # Overall performance
        f.write("🔥 OVERALL PERFORMANCE:\n")
        f.write("─"*50 + "\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Macro F1-Score: {macro_avg.get('f1-score', 0):.4f}\n")
        f.write(f"Weighted F1-Score: {weighted_avg.get('f1-score', 0):.4f}\n\n")
        
        # Summary metrics table
        f.write("📊 SUMMARY METRICS:\n")
        f.write("─"*50 + "\n")
        f.write(f"{'Metric':<15} {'Macro Avg':<12} {'Weighted Avg':<12}\n")
        f.write("─"*50 + "\n")
        f.write(f"{'Precision':<15} {macro_avg.get('precision', 0):<12.3f} {weighted_avg.get('precision', 0):<12.3f}\n")
        f.write(f"{'Recall':<15} {macro_avg.get('recall', 0):<12.3f} {weighted_avg.get('recall', 0):<12.3f}\n")
        f.write(f"{'F1-Score':<15} {macro_avg.get('f1-score', 0):<12.3f} {weighted_avg.get('f1-score', 0):<12.3f}\n\n")
        
        # Per-class performance
        f.write("🏆 PER-CLASS PERFORMANCE:\n")
        f.write("─"*65 + "\n")
        f.write(f"{'Class':<12} {'Precision':<11} {'Recall':<11} {'F1-Score':<11} {'Support':<8}\n")
        f.write("─"*65 + "\n")
        
        for class_id in ordered_classes:
            class_name = class_names[class_id]
            if class_name in class_report:
                metrics = class_report[class_name]
                if isinstance(metrics, dict) and 'f1-score' in metrics:
                    precision = metrics['precision']
                    recall = metrics['recall']
                    f1_score = metrics['f1-score']
                    support = metrics['support']
                    
                    # Add performance indicator
                    indicator = "🟢" if f1_score >= 0.8 else "🟡" if f1_score >= 0.6 else "🔴"
                    
                    f.write(f"{indicator} {class_name:<9} {precision:<11.3f} {recall:<11.3f} {f1_score:<11.3f} {support:<8}\n")
        
        f.write("\n" + "─"*65 + "\n")
        f.write("🟢 Excellent (F1 ≥ 0.8)   🟡 Good (0.6 ≤ F1 < 0.8)   🔴 Needs Improvement (F1 < 0.6)\n")
        f.write("="*80 + "\n")
    
    print("\n" + "="*80)
    print("📄 DETAILED RESULTS SAVED".center(80))
    print("="*80)
    print(f"📊 File: {results_file}".center(80))
    print("="*80)

print("\n" + "🎊"*80)
print("🎉 EVALUATION COMPLETED SUCCESSFULLY! 🎉".center(80))
print("🎊"*80)
print(f"📁 All results saved in: {channel_output_dir}".center(80))
print("🎊"*80)

# Print usage example
print("\n" + "="*60)
print("📖 USAGE EXAMPLES:")
print("="*60)
print("# Basic channel-stacked evaluation with default settings:")
print("python zeroshot_evaluation_channel.py")
print()
print("# Channel-stacked evaluation with custom model:")
print("python zeroshot_evaluation_channel.py --model_path /path/to/model.pt")
print()
print("# Channel-stacked evaluation with custom device and batch size:")
print("python zeroshot_evaluation_channel.py --device cuda:1 --batch_size 32")
print()
print("# Full custom channel-stacked evaluation:")
print("python zeroshot_evaluation_channel.py \\")
print("    --model_path /custom/path/model.pt \\")
print("    --data_path custom_test.csv \\")
print("    --image_size 384 \\")
print("    --batch_size 8 \\")
print("    --device cuda:0 \\")
print("    --output_dir custom_results")
print()
print("📋 CHANNEL STACKING:")
print("┌─────────────────────┐")
print("│ Channel 0: AP View  │")
print("│ Channel 1: PA View  │")
print("│ Channel 2: Skyline  │")
print("└─────────────────────┘")
print("="*60)
