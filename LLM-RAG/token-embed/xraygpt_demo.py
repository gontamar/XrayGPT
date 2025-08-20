import logging
import os
import torch
from typing import Dict, Any, Optional

# Set logging to only show warnings and errors BEFORE importing other modules
logging.basicConfig(level=logging.WARNING, force=True)

from xraygpt_model import XrayGPTModel, create_xraygpt_config, create_generation_model, create_multitask_model
from storage_utils import save_results, create_output_directory, save_xraygpt_complete_results
import json

logger = logging.getLogger(__name__)


def create_test_image_if_needed(image_path: str) -> str:
    """Create a test image if none provided."""
    if not image_path or not os.path.exists(image_path):
        try:
            from PIL import Image
            import numpy as np
            
            # Create a simple test image
            test_image = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            test_path = "xraygpt_test_image.jpg"
            test_image.save(test_path)
            print(f"Created test image: {test_path}")
            return test_path
        except Exception as e:
            print(f"Could not create test image: {e}")
            raise ValueError("No valid image provided and could not create test image")
    return image_path


def demonstrate_text_generation(model: XrayGPTModel, text_input: str, image_path: str, device: str = 'cpu'):
    """Demonstrate text generation capabilities."""
    print("\n" + "="*60)
    print("TEXT GENERATION DEMONSTRATION")
    print("="*60)
    
    try:
        # Generate response
        result = model.generate_response(
            text_input=text_input,
            image_path=image_path,
            max_length=50,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            device=device
        )
        
        print(f"Input Text: {text_input}")
        print(f"Image Path: {image_path}")
        print(f"\nGenerated Response:")
        for i, text in enumerate(result['generated_text']):
            print(f"  Response {i+1}: {text}")
        
        print(f"\nGeneration Metadata:")
        print(f"  Generation Length: {result.get('generation_length', 'N/A')}")
        print(f"  Fused Features Shape: {result['fused_features'].shape}")
        print(f"  Text Features Shape: {result['text_features'].shape}")
        print(f"  Vision Features Shape: {result['vision_features'].shape}")
        
        # Show attention information
        if 'attention_weights' in result and result['attention_weights']:
            attn_weights = result['attention_weights'][-1]  # Last layer attention
            print(f"  Cross-Modal Attention Shape: {attn_weights.shape}")
            print(f"  Average Attention Score: {attn_weights.mean().item():.4f}")
        
        # Save complete results
        print(f"\n💾 Saving complete generation results...")
        complete_output_dir = save_xraygpt_complete_results(
            model_outputs=result,
            text_input=text_input,
            image_path=image_path,
            base_dir="generation_complete_results"
        )
        
        return result
        
    except Exception as e:
        print(f"Error in text generation: {e}")
        logger.error(f"Text generation failed: {e}")
        return None


def demonstrate_classification(model: XrayGPTModel, text_input: str, image_path: str, device: str = 'cpu'):
    """Demonstrate classification capabilities."""
    print("\n" + "="*60)
    print("CLASSIFICATION DEMONSTRATION")
    print("="*60)
    
    try:
        # Perform classification
        result = model.classify(
            text_input=text_input,
            image_path=image_path,
            device=device
        )
        
        print(f"Input Text: {text_input}")
        print(f"Image Path: {image_path}")
        
        if 'classification_logits' in result:
            logits = result['classification_logits']
            probabilities = result['probabilities']
            predictions = result['predictions']
            confidence = result['confidence']
            
            print(f"\nClassification Results:")
            print(f"  Predicted Class: {predictions.item()}")
            print(f"  Confidence: {confidence.item():.4f}")
            print(f"  Top 3 Probabilities:")
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(3, probabilities.size(-1)))
            for i in range(top_probs.size(-1)):
                class_idx = top_indices[0, i].item()
                prob = top_probs[0, i].item()
                print(f"    Class {class_idx}: {prob:.4f}")
        
        print(f"\nClassification Metadata:")
        print(f"  Fused Features Shape: {result['fused_features'].shape}")
        print(f"  Text Features Shape: {result['text_features'].shape}")
        print(f"  Vision Features Shape: {result['vision_features'].shape}")
        
        return result
        
    except Exception as e:
        print(f"Error in classification: {e}")
        logger.error(f"Classification failed: {e}")
        return None


def demonstrate_multitask(model: XrayGPTModel, text_input: str, image_path: str, device: str = 'cpu'):
    """Demonstrate multitask capabilities."""
    print("\n" + "="*60)
    print("MULTITASK DEMONSTRATION")
    print("="*60)
    
    try:
        # Forward pass for both tasks
        result = model.forward(
            text_input=text_input,
            image_path=image_path,
            device=device,
            task='both'
        )
        
        print(f"Input Text: {text_input}")
        print(f"Image Path: {image_path}")
        
        # Show generation results
        if 'generation_logits' in result:
            gen_logits = result['generation_logits']
            print(f"\nGeneration Logits Shape: {gen_logits.shape}")
            
            # Sample a few tokens for demonstration
            sample_probs = torch.softmax(gen_logits[0, 0, :10], dim=0)  # First 10 vocab items
            print(f"Sample Generation Probabilities (first 10 tokens): {sample_probs.tolist()}")
        
        # Show classification results
        if 'classification_logits' in result:
            cls_logits = result['classification_logits']
            cls_probs = torch.softmax(cls_logits, dim=-1)
            cls_pred = torch.argmax(cls_logits, dim=-1)
            
            print(f"\nClassification Results:")
            print(f"  Predicted Class: {cls_pred.item()}")
            print(f"  Class Probabilities: {cls_probs[0].tolist()}")
        
        print(f"\nMultitask Metadata:")
        print(f"  Fused Features Shape: {result['fused_features'].shape}")
        print(f"  Cross-Modal Attention Layers: {len(result['attention_weights'])}")
        
        return result
        
    except Exception as e:
        print(f"Error in multitask demonstration: {e}")
        logger.error(f"Multitask demonstration failed: {e}")
        return None


def save_xraygpt_results(results: Dict[str, Any], output_dir: str):
    """Save XrayGPT results to files."""
    try:
        # Save generation results
        if 'generated_text' in results:
            gen_file = os.path.join(output_dir, "generated_responses.json")
            with open(gen_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'generated_text': results['generated_text'],
                    'generation_length': results.get('generation_length', 0)
                }, f, indent=2, ensure_ascii=False)
            print(f"  - Generated responses: {gen_file}")
        
        # Save classification results
        if 'predictions' in results:
            cls_file = os.path.join(output_dir, "classification_results.json")
            with open(cls_file, 'w') as f:
                json.dump({
                    'predictions': results['predictions'].tolist(),
                    'probabilities': results['probabilities'].tolist(),
                    'confidence': results['confidence'].tolist()
                }, f, indent=2)
            print(f"  - Classification results: {cls_file}")
        
        # Save attention weights
        if 'attention_weights' in results and results['attention_weights']:
            attn_file = os.path.join(output_dir, "attention_weights.json")
            # Save only the last layer attention for space efficiency
            last_attn = results['attention_weights'][-1]
            with open(attn_file, 'w') as f:
                json.dump({
                    'shape': list(last_attn.shape),
                    'attention_weights': last_attn.detach().cpu().numpy().tolist()
                }, f, indent=2)
            print(f"  - Attention weights: {attn_file}")
        
        # Save feature representations
        features_file = os.path.join(output_dir, "feature_representations.json")
        with open(features_file, 'w') as f:
            feature_data = {}
            
            for key in ['fused_features', 'text_features', 'vision_features']:
                if key in results:
                    tensor = results[key]
                    feature_data[key] = {
                        'shape': list(tensor.shape),
                        'mean': tensor.mean().item(),
                        'std': tensor.std().item(),
                        'sample_values': tensor[0, 0, :5].detach().cpu().numpy().tolist()
                    }
            
            json.dump(feature_data, f, indent=2)
        print(f"  - Feature representations: {features_file}")
        
    except Exception as e:
        print(f"Warning: Could not save some results: {e}")


def main():
    """Main demonstration function."""
    print("XrayGPT-like Multimodal AI System Demonstration")
    print("=" * 60)
    
    # Get user inputs
    text_input = input("Enter your question/prompt: ").strip()
    if not text_input:
        text_input = "What do you see in this medical image?"
    
    image_path = input("Enter image path (or press Enter for test image): ").strip()
    image_path = create_test_image_if_needed(image_path)
    
    # Choose demonstration mode
    print("\nAvailable demonstration modes:")
    print("1. Text Generation")
    print("2. Classification")
    print("3. Multitask (Both)")
    print("4. All Demonstrations")
    
    mode = input("Choose mode (1-4, default: 4): ").strip()
    if not mode:
        mode = "4"
    
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create output directory
    output_dir = create_output_directory("xraygpt_results")
    
    all_results = {}
    
    try:
        if mode in ["1", "4"]:
            print("\n" + "🚀 Creating Generation Model...")
            gen_model = create_generation_model(device=device)
            gen_results = demonstrate_text_generation(gen_model, text_input, image_path, device)
            if gen_results:
                all_results['generation'] = gen_results
        
        if mode in ["2", "4"]:
            print("\n" + "🚀 Creating Classification Model...")
            cls_model = create_multitask_model(num_classes=5, device=device)  # Using multitask for classification
            cls_results = demonstrate_classification(cls_model, text_input, image_path, device)
            if cls_results:
                all_results['classification'] = cls_results
        
        if mode in ["3", "4"]:
            print("\n" + "🚀 Creating Multitask Model...")
            multitask_model = create_multitask_model(num_classes=5, device=device)
            multitask_results = demonstrate_multitask(multitask_model, text_input, image_path, device)
            if multitask_results:
                all_results['multitask'] = multitask_results
        
        # Save all results
        print(f"\n" + "💾 Saving results to: {output_dir}")
        for task_name, results in all_results.items():
            task_dir = os.path.join(output_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            save_xraygpt_results(results, task_dir)
        
        print(f"\n✅ XrayGPT demonstration completed successfully!")
        print(f"📁 All results saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        logger.error(f"Demonstration failed: {e}")
    
    finally:
        # Clean up test image if created
        if image_path == "xraygpt_test_image.jpg" and os.path.exists(image_path):
            os.remove(image_path)
            print(f"\n🧹 Cleaned up test image")


if __name__ == "__main__":
    main()