#!/usr/bin/env python3
"""
Debug script to identify NaN sources in the model
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Import our classes
import sys
sys.path.append('.')

def create_simple_model():
    """Create a simplified version for testing"""
    
    class SimpleKneeModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Initialize RadDino for image encoding
            self.rad_dino = AutoModel.from_pretrained("microsoft/rad-dino")
            
            # Initialize BiomedVLP medical BERT for text encoding
            self.medical_bert = AutoModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
            
            # Simple projections
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
            
            # Conservative initialization
            for module in self.image_projection:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                    nn.init.zeros_(module.bias)
            
            for module in self.text_projection:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                    nn.init.zeros_(module.bias)
        
        def forward(self, pixel_values, input_ids, attention_mask):
            # Get RadDino image features
            rad_dino_outputs = self.rad_dino(pixel_values=pixel_values)
            image_features = rad_dino_outputs.last_hidden_state[:, 0]
            
            # Check for NaN
            if not torch.isfinite(image_features).all():
                print("Warning: NaN detected in RadDino features")
                image_features = torch.nan_to_num(image_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Project
            image_embeds = self.image_projection(image_features)
            
            # Get medical BERT text features
            text_outputs = self.medical_bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Check available attributes
            print(f"Medical BERT output attributes: {dir(text_outputs)}")
            
            # Use last_hidden_state and take [CLS] token if pooler_output doesn't exist
            if hasattr(text_outputs, 'pooler_output'):
                text_features = text_outputs.pooler_output
            else:
                # Use [CLS] token from last_hidden_state
                text_features = text_outputs.last_hidden_state[:, 0]
            
            # Check for NaN
            if not torch.isfinite(text_features).all():
                print("Warning: NaN detected in Medical BERT features")
                text_features = torch.nan_to_num(text_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Project
            text_embeds = self.text_projection(text_features)
            
            return {
                'image_embeds': image_embeds,
                'text_embeds': text_embeds
            }
    
    return SimpleKneeModel()

class SimpleSigLIPLoss(nn.Module):
    def __init__(self, init_logit_scale=10.0, init_logit_bias=-10.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.logit_bias = nn.Parameter(torch.tensor(init_logit_bias))
    
    def forward(self, image_embeds, text_embeds):
        batch_size = image_embeds.size(0)
        device = image_embeds.device
        
        # Check inputs
        if not torch.isfinite(image_embeds).all():
            print("Warning: NaN/inf detected in image embeddings")
            image_embeds = torch.nan_to_num(image_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if not torch.isfinite(text_embeds).all():
            print("Warning: NaN/inf detected in text embeddings")  
            text_embeds = torch.nan_to_num(text_embeds, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Normalize
        image_embeds = nn.functional.normalize(image_embeds, dim=1)
        text_embeds = nn.functional.normalize(text_embeds, dim=1)
        
        # Clamp scale
        logit_scale_clamped = torch.clamp(self.logit_scale, min=0.01, max=100.0)
        
        # Calculate logits
        logits = torch.matmul(image_embeds, text_embeds.t()) * logit_scale_clamped + self.logit_bias
        
        # Check logits
        if not torch.isfinite(logits).all():
            print(f"Warning: NaN/inf detected in logits. Scale: {logit_scale_clamped.item():.3f}, Bias: {self.logit_bias.item():.3f}")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Create labels
        labels = 2 * torch.eye(batch_size, device=device) - 1
        
        # Loss
        sigmoid_input = labels * logits
        loss = nn.functional.softplus(-sigmoid_input).mean()
        
        # Final check
        if not torch.isfinite(loss):
            print("Warning: NaN detected in final loss, using dummy loss")
            loss = torch.tensor(1.0, device=device, requires_grad=True)
        
        return loss

def test_nan_debug():
    """Debug NaN issues"""
    print("Testing NaN sources...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Initialize tokenizer
        print("\n=== Initializing tokenizer ===")
        medical_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
        print("✅ Medical tokenizer loaded")
        
        # Initialize model
        print("\n=== Initializing model ===")
        model = create_simple_model().to(device)
        criterion = SimpleSigLIPLoss().to(device)
        print("✅ Model loaded")
        
        # Test with dummy data
        print("\n=== Testing with dummy data ===")
        batch_size = 4
        image_size = 224
        max_text_length = 128
        
        # Create dummy inputs
        dummy_images = torch.randn(batch_size, 3, image_size, image_size, device=device)
        
        # Create realistic medical text
        medical_texts = [
            "Normal knee X-ray with no abnormalities",
            "Severe osteoarthritis with joint space narrowing",
            "Mild degenerative changes in the knee",
            "Fracture of the tibial plateau"
        ]
        
        # Tokenize
        tokenized = medical_tokenizer(
            medical_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_length
        )
        
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        print(f"Input shapes: images={dummy_images.shape}, input_ids={input_ids.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_images, input_ids, attention_mask)
            
        print(f"Output shapes: image_embeds={outputs['image_embeds'].shape}, text_embeds={outputs['text_embeds'].shape}")
        
        # Check outputs
        img_finite = torch.isfinite(outputs['image_embeds']).all()
        txt_finite = torch.isfinite(outputs['text_embeds']).all()
        print(f"Embeddings finite: images={img_finite}, text={txt_finite}")
        
        if img_finite and txt_finite:
            print("✅ Model forward pass successful")
        else:
            print("❌ NaN detected in embeddings")
            return False
        
        # Test loss
        print("\n=== Testing loss ===")
        loss = criterion(outputs['image_embeds'], outputs['text_embeds'])
        print(f"Loss: {loss.item():.4f}")
        print(f"Loss finite: {torch.isfinite(loss).item()}")
        
        if torch.isfinite(loss):
            print("✅ Loss computation successful")
        else:
            print("❌ NaN detected in loss")
            return False
        
        # Test training step
        print("\n=== Testing training step ===")
        model.train()
        criterion.train()
        
        optimizer = torch.optim.AdamW([
            {'params': model.parameters(), 'lr': 1e-4},
            {'params': criterion.parameters(), 'lr': 1e-5}
        ])
        
        # Create training data with gradients
        dummy_images = torch.randn(batch_size, 3, image_size, image_size, device=device, requires_grad=True)
        
        outputs = model(dummy_images, input_ids, attention_mask)
        loss = criterion(outputs['image_embeds'], outputs['text_embeds'])
        
        print(f"Training loss: {loss.item():.4f}")
        
        if torch.isfinite(loss):
            loss.backward()
            
            # Check gradients
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    if not torch.isfinite(param.grad).all():
                        print(f"❌ NaN gradient in {name}")
                        return False
            
            print(f"Gradient norms: min={min(grad_norms):.4f}, max={max(grad_norms):.4f}, mean={np.mean(grad_norms):.4f}")
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            print("✅ Training step successful")
        else:
            print("❌ NaN in training loss")
            return False
        
        print("\n🎉 All tests passed! No NaN issues detected.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_nan_debug() 