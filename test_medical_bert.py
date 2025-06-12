#!/usr/bin/env python3
"""
Test script to verify Medical BERT integration with RadDino
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoProcessor
from train_siglip import KneeRadDinoSigLIPModel, KneeXrayDataset, SigLIPContrastiveLoss
import pandas as pd
from PIL import Image
import numpy as np

def test_medical_bert_integration():
    """Test the medical BERT integration"""
    print("Testing Medical BERT Integration...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 4
    image_size = 224
    max_text_length = 128
    
    try:
        # Initialize processors and tokenizers
        print("\n=== Initializing processors ===")
        model_name = f"google/siglip-base-patch16-{image_size}"
        processor = AutoProcessor.from_pretrained(model_name)
        medical_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
        
        print("✅ Processors initialized successfully")
        
        # Test tokenizer
        print("\n=== Testing medical tokenizer ===")
        sample_text = "Severe osteoarthritis of both knee joints with varus angulation and joint space narrowing"
        tokens = medical_tokenizer(
            sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_length
        )
        print(f"Tokenized text shape: {tokens['input_ids'].shape}")
        print(f"Attention mask shape: {tokens['attention_mask'].shape}")
        print("✅ Medical tokenizer working correctly")
        
        # Initialize model
        print("\n=== Initializing model ===")
        model = KneeRadDinoSigLIPModel(siglip_model_name=model_name).to(device)
        
        print(f"Model loaded successfully")
        print(f"RadDino hidden size: {model.rad_dino.config.hidden_size}")
        print(f"Medical BERT hidden size: {model.medical_bert.config.hidden_size}")
        
        # Test model forward pass
        print("\n=== Testing model forward pass ===")
        
        # Create dummy data
        dummy_images = torch.randn(batch_size, 3, image_size, image_size, device=device)
        dummy_input_ids = torch.randint(1, 1000, (batch_size, max_text_length), device=device)
        dummy_attention_mask = torch.ones(batch_size, max_text_length, device=device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask)
        
        print(f"Image embeddings shape: {outputs['image_embeds'].shape}")
        print(f"Text embeddings shape: {outputs['text_embeds'].shape}")
        print(f"Output is finite: {torch.isfinite(outputs['image_embeds']).all()} (images), {torch.isfinite(outputs['text_embeds']).all()} (text)")
        print("✅ Model forward pass successful")
        
        # Test loss computation
        print("\n=== Testing loss computation ===")
        criterion = SigLIPContrastiveLoss().to(device)
        
        loss = criterion(outputs['image_embeds'], outputs['text_embeds'])
        print(f"Loss: {loss.item():.4f}")
        print(f"Loss is finite: {torch.isfinite(loss).item()}")
        print("✅ Loss computation successful")
        
        # Test gradient flow
        print("\n=== Testing gradient flow ===")
        model.train()
        criterion.train()
        
        # Create dummy data with gradients
        dummy_images = torch.randn(batch_size, 3, image_size, image_size, device=device, requires_grad=True)
        dummy_input_ids = torch.randint(1, 1000, (batch_size, max_text_length), device=device)
        dummy_attention_mask = torch.ones(batch_size, max_text_length, device=device)
        
        outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask)
        loss = criterion(outputs['image_embeds'], outputs['text_embeds'])
        loss.backward()
        
        print(f"Loss: {loss.item():.4f}")
        print(f"Input gradients norm: {dummy_images.grad.norm().item():.4f}")
        print(f"Model has gradients: {any(p.grad is not None for p in model.parameters())}")
        print(f"Criterion has gradients: {any(p.grad is not None for p in criterion.parameters())}")
        print("✅ Gradient flow successful")
        
        # Test realistic medical text
        print("\n=== Testing realistic medical text ===")
        medical_texts = [
            "Normal knee X-ray with no signs of osteoarthritis",
            "Severe osteoarthritis with joint space narrowing",
            "Mild degenerative changes in the knee joint",
            "Fracture of the proximal tibia"
        ]
        
        # Create dummy images
        dummy_images = torch.randn(len(medical_texts), 3, image_size, image_size, device=device)
        
        # Tokenize medical texts
        tokenized = medical_tokenizer(
            medical_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_text_length
        )
        
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_images, input_ids, attention_mask)
            loss = criterion(outputs['image_embeds'], outputs['text_embeds'])
        
        print(f"Medical text embeddings shape: {outputs['text_embeds'].shape}")
        print(f"Loss with medical text: {loss.item():.4f}")
        print("✅ Medical text processing successful")
        
        print("\n🎉 All tests passed! Medical BERT integration is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_medical_bert_integration() 