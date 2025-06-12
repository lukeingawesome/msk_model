#!/usr/bin/env python3
"""
Test script to verify SigLIP loss implementation
"""

import torch
import torch.nn as nn
import numpy as np
from train_siglip import SigLIPContrastiveLoss

def test_siglip_loss():
    """Test the SigLIP loss implementation"""
    print("Testing SigLIP Loss Implementation...")
    
    # Test parameters
    batch_size = 8
    embed_dim = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create loss function
    criterion = SigLIPContrastiveLoss().to(device)
    
    print(f"Initial logit_scale: {criterion.logit_scale.item():.4f}")
    print(f"Initial logit_bias: {criterion.logit_bias.item():.4f}")
    
    # Test with random embeddings
    print("\n=== Test 1: Random embeddings ===")
    image_embeds = torch.randn(batch_size, embed_dim, device=device)
    text_embeds = torch.randn(batch_size, embed_dim, device=device)
    
    loss = criterion(image_embeds, text_embeds)
    print(f"Loss with random embeddings: {loss.item():.4f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")
    
    # Test with normalized embeddings
    print("\n=== Test 2: Pre-normalized embeddings ===")
    image_embeds = nn.functional.normalize(torch.randn(batch_size, embed_dim, device=device), dim=1)
    text_embeds = nn.functional.normalize(torch.randn(batch_size, embed_dim, device=device), dim=1)
    
    loss = criterion(image_embeds, text_embeds)
    print(f"Loss with normalized embeddings: {loss.item():.4f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")
    
    # Test with perfect alignment (should give low loss)
    print("\n=== Test 3: Perfect alignment ===")
    perfect_embeds = nn.functional.normalize(torch.randn(batch_size, embed_dim, device=device), dim=1)
    
    loss = criterion(perfect_embeds, perfect_embeds)
    print(f"Loss with perfect alignment: {loss.item():.4f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")
    
    # Test gradient flow
    print("\n=== Test 4: Gradient flow ===")
    image_embeds = torch.randn(batch_size, embed_dim, device=device, requires_grad=True)
    text_embeds = torch.randn(batch_size, embed_dim, device=device, requires_grad=True)
    
    loss = criterion(image_embeds, text_embeds)
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Image embeds grad norm: {image_embeds.grad.norm().item():.4f}")
    print(f"Text embeds grad norm: {text_embeds.grad.norm().item():.4f}")
    print(f"Logit scale grad: {criterion.logit_scale.grad.item():.4f}")
    print(f"Logit bias grad: {criterion.logit_bias.grad.item():.4f}")
    
    # Test training simulation
    print("\n=== Test 5: Training simulation ===")
    optimizer = torch.optim.AdamW([
        {'params': criterion.parameters(), 'lr': 1e-3}
    ])
    
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        
        # Create slightly better aligned embeddings over time
        noise_scale = 1.0 - step * 0.05
        image_embeds = nn.functional.normalize(torch.randn(batch_size, embed_dim, device=device), dim=1)
        text_embeds = nn.functional.normalize(image_embeds + noise_scale * torch.randn(batch_size, embed_dim, device=device), dim=1)
        
        loss = criterion(image_embeds, text_embeds)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Step {step+1}: Loss={loss.item():.4f}, Scale={criterion.logit_scale.item():.3f}, Bias={criterion.logit_bias.item():.3f}")
    
    print(f"\nLoss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"Loss decreased: {losses[-1] < losses[0]}")
    
    print("\n✅ All tests completed successfully!")
    return True

if __name__ == "__main__":
    test_siglip_loss() 