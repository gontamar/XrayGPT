#!/usr/bin/env python3
"""
Test script to verify device compatibility fix.
"""

import torch
import os
from PIL import Image
import numpy as np

def create_test_image(path="test_image.jpg"):
    """Create a test image if it doesn't exist."""
    if not os.path.exists(path):
        # Create a simple test image
        test_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        test_image.save(path)
        print(f"Created test image: {path}")
    return path

def test_device_compatibility():
    """Test device compatibility."""
    print("Testing XrayGPT device compatibility...")
    
    # Check available device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        from xraygpt_model import create_generation_model
        
        # Create test image
        test_image_path = create_test_image()
        
        print("Creating XrayGPT model...")
        model = create_generation_model(device=device)
        
        print("Testing text generation...")
        result = model.generate_response(
            text_input="Describe this medical image",
            image_path=test_image_path,
            max_length=20,  # Short for testing
            device=device
        )
        
        print("✅ SUCCESS: Device compatibility test passed!")
        print(f"Generated text: {result['generated_text'][0]}")
        print(f"Fused features shape: {result['fused_features'].shape}")
        print(f"Device: {result['fused_features'].device}")
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print("Cleaned up test image")
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_device_compatibility()
    exit(0 if success else 1)