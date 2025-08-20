#!/usr/bin/env python3
"""
Demonstration script showing the model architecture and saved data.
"""

import torch
import os
from PIL import Image
import numpy as np
import json

def create_test_image(path="model_demo_image.jpg"):
    """Create a test image."""
    if not os.path.exists(path):
        test_image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        test_image.save(path)
        print(f"✅ Created test image: {path}")
    return path

def demonstrate_model_architecture():
    """Show the model architecture and components."""
    print("🤖 XrayGPT-like Model Architecture")
    print("=" * 50)
    
    from xraygpt_model import create_generation_model
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    model = create_generation_model(device=device)
    
    print("\n📋 Model Components:")
    print(f"  🔤 Tokenizer: {type(model.tokenizer_manager).__name__}")
    print(f"     - Model: {model.tokenizer_manager.tokenizer.__class__.__name__}")
    print(f"     - Vocab Size: {getattr(model.tokenizer_manager.tokenizer, 'vocab_size', 'N/A')}")
    
    print(f"  🧠 Text Embeddings: {type(model.embedding_manager).__name__}")
    print(f"     - Model: {type(model.embedding_manager.embedding_model).__name__}")
    print(f"     - Hidden Size: {model.embedding_manager.model_config.hidden_size if model.embedding_manager.model_config else 'N/A'}")
    
    print(f"  🖼️  Image Processor: {type(model.image_processor).__name__}")
    print(f"     - Processor: BLIP2 + Vision Transformer")
    
    print(f"  🔗 Fusion Layer: {type(model.vision_language_fusion).__name__}")
    print(f"     - Layers: {model.num_fusion_layers}")
    print(f"     - Heads: {model.num_heads}")
    print(f"     - Hidden Dim: {model.d_model}")
    
    print(f"  🎯 Output Head: {type(model.output_head).__name__}")
    print(f"     - Task: Text Generation")
    print(f"     - Vocab Size: {model.vocab_size}")
    
    return model

def demonstrate_generation_process():
    """Show the complete generation process and saved data."""
    print("\n🔄 Generation Process Demonstration")
    print("=" * 50)
    
    # Create model and test image
    model = demonstrate_model_architecture()
    test_image_path = create_test_image()
    
    # Test input
    text_input = "Describe the medical findings in this image."
    
    print(f"\n📝 Input:")
    print(f"  Text: '{text_input}'")
    print(f"  Image: {test_image_path}")
    
    # Generate response
    print(f"\n⚙️  Processing...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    result = model.generate_response(
        text_input=text_input,
        image_path=test_image_path,
        max_length=50,
        temperature=0.8,
        device=device
    )
    
    print(f"\n✅ Generation Complete!")
    print(f"  Generated Text: '{result['generated_text'][0]}'")
    print(f"  Generation Length: {result['generation_length']}")
    
    # Show what data is available
    print(f"\n📊 Available Data in Result:")
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            print(f"  🔢 {key}: {value.shape} ({value.dtype})")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"  📋 {key}: {len(value)} layers, each {value[0].shape}")
        else:
            print(f"  📄 {key}: {type(value).__name__}")
    
    # Save comprehensive results
    print(f"\n💾 Saving Complete Results...")
    from storage_utils import save_xraygpt_complete_results
    
    output_dir = save_xraygpt_complete_results(
        model_outputs=result,
        text_input=text_input,
        image_path=test_image_path,
        base_dir="model_demo_results"
    )
    
    # Show saved files
    print(f"\n📁 Saved Files:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if file.endswith('.json'):
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if 'shape' in data:
                        print(f"  📊 {file}: Shape {data['shape']}")
                    elif 'generated_text' in data:
                        print(f"  💬 {file}: Generated responses")
                    elif 'model_architecture' in data:
                        print(f"  ⚙️  {file}: Model metadata")
                    else:
                        print(f"  📄 {file}: {len(str(data))} characters")
                except:
                    print(f"  📄 {file}: JSON file")
    
    # Clean up
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
        print(f"\n🧹 Cleaned up test image")
    
    return output_dir

def show_model_summary():
    """Show a summary of the model and generation process."""
    print("\n📋 MODEL SUMMARY")
    print("=" * 50)
    
    print("🤖 Response Generation Model:")
    print("  ❌ NOT using pre-trained LLM (GPT, LLaMA, T5)")
    print("  ✅ Using custom MLP head for generation")
    print("  🔧 Architecture: BERT + Vision Transformer + Cross-Modal Attention + Custom MLP")
    
    print("\n💾 Data Saved:")
    print("  📝 Text tokens and embeddings")
    print("  🖼️  Image tensors and patch embeddings")
    print("  🧠 Vision Transformer features (CLS token + patch features)")
    print("  🔗 Fused multimodal features")
    print("  👁️  Cross-modal attention weights")
    print("  💬 Generated responses")
    print("  ⚙️  Model metadata and configuration")
    
    print("\n🎯 Key Points:")
    print("  • Custom implementation, not pre-trained language model")
    print("  • Saves ALL intermediate features and embeddings")
    print("  • Supports both CPU and GPU processing")
    print("  • Vision Transformer integration for advanced image understanding")
    print("  • Cross-modal attention for vision-language fusion")

if __name__ == "__main__":
    print("🚀 XrayGPT Model Information & Data Saving Demo")
    print("=" * 60)
    
    try:
        output_dir = demonstrate_generation_process()
        show_model_summary()
        
        print(f"\n✅ Demo completed successfully!")
        print(f"📂 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()