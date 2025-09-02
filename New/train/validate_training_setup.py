#!/usr/bin/env python3
"""
XRayGPT Training Setup Validation Script
This script validates that all required components are properly set up for training.
"""

import os
import json
import torch
import yaml
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and print status"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description}: {dirpath} - NOT FOUND")
        return False

def check_dataset_structure(dataset_path, dataset_name, expected_files):
    """Check dataset structure and file counts"""
    print(f"\n📊 Validating {dataset_name} Dataset:")
    
    # Check main directory
    if not check_directory_exists(dataset_path, f"{dataset_name} dataset directory"):
        return False
    
    # Check image directory
    image_dir = os.path.join(dataset_path, "image")
    if not check_directory_exists(image_dir, f"{dataset_name} image directory"):
        return False
    
    # Count images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   📸 Found {len(image_files)} images")
    
    # Check annotation file
    annotation_file = os.path.join(dataset_path, "filter_cap.json")
    if check_file_exists(annotation_file, f"{dataset_name} annotations"):
        try:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            print(f"   📝 Found {len(annotations)} annotations")
            
            # Validate annotation structure
            if annotations and isinstance(annotations, list):
                sample = annotations[0]
                required_keys = ['image_id', 'caption']
                missing_keys = [key for key in required_keys if key not in sample]
                if missing_keys:
                    print(f"   ⚠️  Missing keys in annotations: {missing_keys}")
                else:
                    print(f"   ✅ Annotation structure is valid")
            
        except Exception as e:
            print(f"   ❌ Error reading annotations: {e}")
            return False
    
    return True

def check_model_weights():
    """Check if all required model weights are available"""
    print(f"\n🧠 Validating Model Weights:")
    
    # Check Vicuna weights
    vicuna_path = "vicuna_weights"
    if check_directory_exists(vicuna_path, "Vicuna weights directory"):
        required_files = ["config.json", "pytorch_model.bin.index.json"]
        for file in required_files:
            check_file_exists(os.path.join(vicuna_path, file), f"Vicuna {file}")
        
        # Check for model bin files
        bin_files = [f for f in os.listdir(vicuna_path) if f.startswith("pytorch_model") and f.endswith(".bin")]
        print(f"   📦 Found {len(bin_files)} model binary files")
    
    # Check MiniGPT-4 checkpoint
    minigpt4_path = "xraygpt/pretrained_ckpt/pretrained_minigpt4_7b.pth"
    check_file_exists(minigpt4_path, "MiniGPT-4 checkpoint")

def check_configuration_files():
    """Check if configuration files are properly set up"""
    print(f"\n⚙️ Validating Configuration Files:")
    
    config_files = [
        ("xraygpt/configs/models/xraygpt.yaml", "Model config"),
        ("xraygpt/configs/datasets/mimic/defaults.yaml", "MIMIC dataset config"),
        ("xraygpt/configs/datasets/openi/defaults.yaml", "OpenI dataset config"),
        ("train_configs/xraygpt_mimic_pretrain.yaml", "MIMIC training config"),
        ("train_configs/xraygpt_openi_finetune.yaml", "OpenI training config"),
        ("eval_configs/xraygpt_eval.yaml", "Evaluation config")
    ]
    
    for config_file, description in config_files:
        if check_file_exists(config_file, description):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"   ✅ {description} is valid YAML")
            except Exception as e:
                print(f"   ❌ {description} YAML parsing error: {e}")

def check_environment():
    """Check Python environment and dependencies"""
    print(f"\n🐍 Validating Environment:")
    
    # Check Python version
    import sys
    print(f"   🐍 Python version: {sys.version}")
    
    # Check key dependencies
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "Transformers"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("gradio", "Gradio")
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {name} is installed")
        except ImportError:
            print(f"   ❌ {name} is NOT installed")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"   ✅ CUDA is available")
        print(f"   🎮 GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   🎮 GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print(f"   ❌ CUDA is NOT available")

def main():
    """Main validation function"""
    print("🔍 XRayGPT Training Setup Validation")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("train.py"):
        print("❌ Error: Please run this script from the XRayGPT root directory")
        return
    
    # Validate environment
    check_environment()
    
    # Validate datasets
    check_dataset_structure("dataset/mimic", "MIMIC", [])
    check_dataset_structure("dataset/openi", "OpenI", [])
    
    # Validate model weights
    check_model_weights()
    
    # Validate configuration files
    check_configuration_files()
    
    # Check output directories
    print(f"\n📁 Validating Output Directories:")
    output_dirs = [
        "output/xraygpt_mimic_pretrain",
        "output/xraygpt_openi_finetune"
    ]
    
    for output_dir in output_dirs:
        check_directory_exists(output_dir, f"Output directory")
    
    print(f"\n🎯 Validation Summary:")
    print("=" * 30)
    print("If all items above show ✅, you're ready to start training!")
    print("If any items show ❌, please address them before training.")
    print("")
    print("🚂 Training Commands:")
    print("Stage 1: torchrun --nproc-per-node 4 train.py --cfg-path train_configs/xraygpt_mimic_pretrain.yaml")
    print("Stage 2: torchrun --nproc-per-node 1 train.py --cfg-path train_configs/xraygpt_openi_finetune.yaml")

if __name__ == "__main__":
    main()