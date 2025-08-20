import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any


def create_output_directory(base_dir: str = "output_results") -> str:
    """
    Create timestamped output directory for storing results.
    
    Args:
        base_dir: Base directory name for outputs
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_dir}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    return output_dir


def save_results(tokenize_result: Dict[str, Any], embed_result: Dict[str, Any], 
                image_result: Dict[str, Any], base_dir: str = "output_results") -> str:
    """
    Save essential processing results to files.
    
    Args:
        tokenize_result: Results from tokenizer
        embed_result: Results from embedding manager
        image_result: Results from image processor
        base_dir: Base directory name for outputs
        
    Returns:
        Path to created output directory
    """
    output_dir = create_output_directory(base_dir)
    
    # Save text tokens and token IDs
    text_data = {
        "tokens": tokenize_result['tokens'],
        "token_ids": tokenize_result['token_ids']
    }
    
    text_file = os.path.join(output_dir, "text_tokens_and_ids.json")
    with open(text_file, 'w', encoding='utf-8') as f:
        json.dump(text_data, f, indent=2, ensure_ascii=False)
    
    # Save text embeddings
    embeddings = embed_result['embeddings']
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Convert to list for JSON serialization
    embeddings_list = embeddings_np.tolist()
    
    embeddings_file = os.path.join(output_dir, "text_embeddings.json")
    with open(embeddings_file, 'w') as f:
        json.dump({
            "shape": list(embeddings_np.shape),
            "embeddings": embeddings_list
        }, f, indent=2)
    
    # Save image tensor
    if 'image_tensor' in image_result:
        image_tensor = image_result['image_tensor']
        if isinstance(image_tensor, torch.Tensor):
            image_tensor_np = image_tensor.detach().cpu().numpy()
        else:
            image_tensor_np = image_tensor
        
        # Convert to list for JSON serialization
        image_tensor_list = image_tensor_np.tolist()
        
        image_tensor_file = os.path.join(output_dir, "image_tensor.json")
        with open(image_tensor_file, 'w') as f:
            json.dump({
                "shape": list(image_tensor_np.shape),
                "tensor": image_tensor_list
            }, f, indent=2)
    
    # Save image embeddings (patch embeddings)
    if 'patch_embeddings' in image_result:
        patch_embeddings = image_result['patch_embeddings']
        if isinstance(patch_embeddings, torch.Tensor):
            patch_embeddings_np = patch_embeddings.detach().cpu().numpy()
        else:
            patch_embeddings_np = patch_embeddings
        
        # Convert to list for JSON serialization
        patch_embeddings_list = patch_embeddings_np.tolist()
        
        image_embeddings_file = os.path.join(output_dir, "image_embeddings.json")
        with open(image_embeddings_file, 'w') as f:
            json.dump({
                "shape": list(patch_embeddings_np.shape),
                "embeddings": patch_embeddings_list,
                "num_patches": image_result.get('num_patches', 'N/A'),
                "patch_shape": image_result.get('patch_shape', 'N/A'),
                "embed_dim": image_result.get('embed_dim', 'N/A')
            }, f, indent=2)
    
    # Save Vision Transformer features (if available)
    if 'vit_features' in image_result:
        vit_features = image_result['vit_features']
        if isinstance(vit_features, torch.Tensor):
            vit_features_np = vit_features.detach().cpu().numpy()
        else:
            vit_features_np = vit_features
        
        vit_features_list = vit_features_np.tolist()
        
        vit_features_file = os.path.join(output_dir, "vit_features.json")
        with open(vit_features_file, 'w') as f:
            json.dump({
                "shape": list(vit_features_np.shape),
                "features": vit_features_list,
                "vit_type": image_result.get('vit_type', 'N/A'),
                "embed_dim": image_result.get('vit_embed_dim', 'N/A'),
                "num_patches": image_result.get('vit_num_patches', 'N/A')
            }, f, indent=2)
        
        # Save CLS token separately
        if 'vit_cls_token' in image_result:
            cls_token = image_result['vit_cls_token']
            if isinstance(cls_token, torch.Tensor):
                cls_token_np = cls_token.detach().cpu().numpy()
            else:
                cls_token_np = cls_token
            
            cls_token_file = os.path.join(output_dir, "vit_cls_token.json")
            with open(cls_token_file, 'w') as f:
                json.dump({
                    "shape": list(cls_token_np.shape),
                    "cls_token": cls_token_np.tolist()
                }, f, indent=2)
        
        # Save patch features separately
        if 'vit_patch_features' in image_result:
            patch_features = image_result['vit_patch_features']
            if isinstance(patch_features, torch.Tensor):
                patch_features_np = patch_features.detach().cpu().numpy()
            else:
                patch_features_np = patch_features
            
            vit_patch_file = os.path.join(output_dir, "vit_patch_features.json")
            with open(vit_patch_file, 'w') as f:
                json.dump({
                    "shape": list(patch_features_np.shape),
                    "patch_features": patch_features_np.tolist()
                }, f, indent=2)
    
    print(f"Results saved:")
    print(f"  - Text tokens and IDs: {text_file}")
    print(f"  - Text embeddings: {embeddings_file}")
    if 'image_tensor' in image_result:
        print(f"  - Image tensor: {image_tensor_file}")
    if 'patch_embeddings' in image_result:
        print(f"  - Image embeddings: {image_embeddings_file}")
    if 'vit_features' in image_result:
        print(f"  - ViT features: {vit_features_file}")
    if 'vit_cls_token' in image_result:
        print(f"  - ViT CLS token: {cls_token_file}")
    if 'vit_patch_features' in image_result:
        print(f"  - ViT patch features: {vit_patch_file}")
    
    return output_dir


def save_xraygpt_complete_results(model_outputs: Dict[str, Any], 
                                 text_input: str, image_path: str,
                                 base_dir: str = "xraygpt_complete_results") -> str:
    """
    Save complete XrayGPT model results including all intermediate features.
    
    Args:
        model_outputs: Complete outputs from XrayGPT model
        text_input: Original text input
        image_path: Original image path
        base_dir: Base directory name for outputs
        
    Returns:
        Path to created output directory
    """
    output_dir = create_output_directory(base_dir)
    
    # Save input information
    input_info = {
        "text_input": text_input,
        "image_path": image_path,
        "timestamp": datetime.now().isoformat(),
        "model_type": "XrayGPT-like Multimodal System"
    }
    
    input_file = os.path.join(output_dir, "input_info.json")
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(input_info, f, indent=2, ensure_ascii=False)
    
    # Save generated responses
    if 'generated_text' in model_outputs:
        response_data = {
            "generated_text": model_outputs['generated_text'],
            "generation_length": model_outputs.get('generation_length', 0),
            "generation_metadata": {
                "max_length": model_outputs.get('max_length', 'N/A'),
                "temperature": model_outputs.get('temperature', 'N/A'),
                "top_p": model_outputs.get('top_p', 'N/A'),
                "top_k": model_outputs.get('top_k', 'N/A')
            }
        }
        
        response_file = os.path.join(output_dir, "generated_responses.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2, ensure_ascii=False)
    
    # Save classification results
    if 'predictions' in model_outputs:
        classification_data = {
            "predictions": model_outputs['predictions'].tolist() if hasattr(model_outputs['predictions'], 'tolist') else model_outputs['predictions'],
            "probabilities": model_outputs['probabilities'].tolist() if hasattr(model_outputs['probabilities'], 'tolist') else model_outputs['probabilities'],
            "confidence": model_outputs['confidence'].tolist() if hasattr(model_outputs['confidence'], 'tolist') else model_outputs['confidence']
        }
        
        classification_file = os.path.join(output_dir, "classification_results.json")
        with open(classification_file, 'w') as f:
            json.dump(classification_data, f, indent=2)
    
    # Save feature representations
    feature_tensors = ['fused_features', 'text_features', 'vision_features']
    for feature_name in feature_tensors:
        if feature_name in model_outputs:
            tensor = model_outputs[feature_name]
            if isinstance(tensor, torch.Tensor):
                tensor_np = tensor.detach().cpu().numpy()
                
                # Save full tensor data
                tensor_file = os.path.join(output_dir, f"{feature_name}.json")
                with open(tensor_file, 'w') as f:
                    json.dump({
                        "shape": list(tensor_np.shape),
                        "data": tensor_np.tolist(),
                        "statistics": {
                            "mean": float(tensor_np.mean()),
                            "std": float(tensor_np.std()),
                            "min": float(tensor_np.min()),
                            "max": float(tensor_np.max())
                        }
                    }, f, indent=2)
    
    # Save attention weights
    if 'attention_weights' in model_outputs and model_outputs['attention_weights']:
        attention_data = []
        for i, attn_layer in enumerate(model_outputs['attention_weights']):
            if isinstance(attn_layer, torch.Tensor):
                attn_np = attn_layer.detach().cpu().numpy()
                attention_data.append({
                    "layer": i,
                    "shape": list(attn_np.shape),
                    "weights": attn_np.tolist(),
                    "statistics": {
                        "mean": float(attn_np.mean()),
                        "std": float(attn_np.std()),
                        "entropy": float(-np.sum(attn_np * np.log(attn_np + 1e-12), axis=-1).mean())
                    }
                })
        
        attention_file = os.path.join(output_dir, "attention_weights.json")
        with open(attention_file, 'w') as f:
            json.dump(attention_data, f, indent=2)
    
    # Save model configuration and metadata
    model_metadata = {
        "model_architecture": "XrayGPT-like Multimodal System",
        "components": {
            "text_encoder": "BERT-based",
            "image_encoder": "BLIP2 + Vision Transformer",
            "fusion": "Cross-modal Attention",
            "output_head": "Custom MLP"
        },
        "feature_dimensions": {
            "text_features": list(model_outputs['text_features'].shape) if 'text_features' in model_outputs else None,
            "vision_features": list(model_outputs['vision_features'].shape) if 'vision_features' in model_outputs else None,
            "fused_features": list(model_outputs['fused_features'].shape) if 'fused_features' in model_outputs else None
        },
        "attention_info": {
            "num_attention_layers": len(model_outputs['attention_weights']) if 'attention_weights' in model_outputs else 0,
            "attention_heads": model_outputs['attention_weights'][0].shape[1] if 'attention_weights' in model_outputs and model_outputs['attention_weights'] else None
        }
    }
    
    metadata_file = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Print summary
    print(f"\n📁 Complete XrayGPT Results Saved:")
    print(f"  📝 Input info: {input_file}")
    if 'generated_text' in model_outputs:
        print(f"  💬 Generated responses: {response_file}")
    if 'predictions' in model_outputs:
        print(f"  🎯 Classification results: {classification_file}")
    
    for feature_name in feature_tensors:
        if feature_name in model_outputs:
            print(f"  🧠 {feature_name.replace('_', ' ').title()}: {os.path.join(output_dir, f'{feature_name}.json')}")
    
    if 'attention_weights' in model_outputs and model_outputs['attention_weights']:
        print(f"  👁️  Attention weights: {attention_file}")
    
    print(f"  ⚙️  Model metadata: {metadata_file}")
    print(f"  📂 Output directory: {output_dir}")
    
    return output_dir