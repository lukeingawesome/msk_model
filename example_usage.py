#!/usr/bin/env python3
"""
Example Usage Script for SigLIP Training

This script demonstrates how to run the SigLIP training with your data.
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import torch
        import transformers
        import wandb
        import pandas as pd
        from PIL import Image
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements_siglip.txt")
        return False

def check_data_format(csv_path="train.csv"):
    """Check if the CSV file has the correct format"""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        required_columns = ['img_path', 'impression', 'split']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"✗ Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Check split distribution
        print("✓ CSV file format is correct")
        print("Split distribution:")
        print(df['split'].value_counts())
        
        # Check for train and val splits
        if 'train' not in df['split'].values:
            print("✗ No 'train' split found in data")
            return False
        if 'val' not in df['split'].values:
            print("✗ No 'val' split found in data")
            return False
        
        print("✓ Required train/val splits found")
        return True
        
    except Exception as e:
        print(f"✗ Error reading CSV file: {e}")
        return False

def run_training_example():
    """Run a basic training example"""
    print("\n" + "="*50)
    print("RUNNING SIGLIP TRAINING EXAMPLE")
    print("="*50)
    
    # Basic training command
    cmd = [
        "python", "train_siglip.py",
        "--data_path", "train.csv",
        "--batch_size", "64",
        "--image_size", "224",
        "--epochs", "2",  # Just 2 epochs for quick test
        "--learning_rate", "1e-4",
        "--warmup_ratio", "0.05",
        "--project_name", "knee-siglip-test",
        "--experiment_name", "quick-test"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✓ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return False

def run_training_with_mixed_precision():
    """Run training with mixed precision and other advanced options"""
    print("\n" + "="*50)
    print("RUNNING ADVANCED SIGLIP TRAINING")
    print("="*50)
    
    cmd = [
        "python", "train_siglip.py",
        "--data_path", "train.csv",
        "--batch_size", "32",
        "--image_size", "224",
        "--epochs", "5",
        "--learning_rate", "5e-5",
        "--weight_decay", "0.01",
        "--warmup_ratio", "0.05",
        "--temperature", "0.07",
        "--gradient_accumulation_steps", "2",
        "--mixed_precision",
        "--project_name", "knee-siglip-advanced",
        "--experiment_name", "mixed-precision-test"
    ]
    
    print("Running advanced training with:")
    print("- Mixed precision training")
    print("- Gradient accumulation (steps=2)")
    print("- Lower learning rate")
    print()
    print("Command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✓ Advanced training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Advanced training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("Training interrupted by user")
        return False

def main():
    print("SigLIP Training Setup Checker and Example Runner")
    print("=" * 50)
    
    # Check requirements
    print("1. Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    # Check data format
    print("\n2. Checking data format...")
    if not check_data_format():
        print("\nPlease ensure your train.csv file has the following columns:")
        print("- img_path: Path to the image file")
        print("- impression: Text description/impression")
        print("- split: 'train', 'val', or 'test'")
        sys.exit(1)
    
    # Ask user what to run
    print("\n3. Choose what to run:")
    print("a) Quick test (2 epochs, basic settings)")
    print("b) Advanced test (5 epochs, mixed precision)")
    print("c) Skip training (just validation)")
    
    choice = input("\nEnter your choice (a/b/c): ").lower().strip()
    
    if choice == 'a':
        success = run_training_example()
    elif choice == 'b':
        success = run_training_with_mixed_precision()
    elif choice == 'c':
        print("Skipping training. Your setup looks good!")
        success = True
    else:
        print("Invalid choice. Exiting.")
        success = False
    
    if success:
        print("\n" + "="*50)
        print("SETUP VALIDATION COMPLETE!")
        print("="*50)
        print("Your SigLIP training environment is ready!")
        print("\nNext steps:")
        print("1. Adjust hyperparameters in train_siglip.py as needed")
        print("2. Run full training: python train_siglip.py --epochs 20")
        print("3. Monitor training with wandb")
        print("4. Use inference_siglip.py for model evaluation")
    else:
        print("\n" + "="*50)
        print("SETUP ISSUES DETECTED")
        print("="*50)
        print("Please fix the issues above before proceeding.")

if __name__ == "__main__":
    main() 