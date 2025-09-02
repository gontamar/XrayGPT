#!/bin/bash

# XRayGPT Training Setup Script
# This script automates the setup process for XRayGPT training

echo "🚀 XRayGPT Training Setup Script"
echo "================================"

# Check if we're in the XRayGPT directory
if [ ! -f "train.py" ]; then
    echo "❌ Error: Please run this script from the XRayGPT root directory"
    exit 1
fi

echo "📁 Creating directory structure..."
mkdir -p dataset/mimic/image
mkdir -p dataset/openi/image
mkdir -p vicuna_weights
mkdir -p xraygpt/pretrained_ckpt
mkdir -p output/xraygpt_mimic_pretrain
mkdir -p output/xraygpt_openi_finetune

echo "✅ Directories created successfully"

echo "⚙️ Updating configuration files..."

# Update model configuration
echo "Updating xraygpt/configs/models/xraygpt.yaml..."
sed -i 's|llama_model: "./Vicuna_Radiology_fp16/"|llama_model: "./vicuna_weights/"|g' xraygpt/configs/models/xraygpt.yaml

# Update MIMIC dataset configuration
echo "Updating xraygpt/configs/datasets/mimic/defaults.yaml..."
cat > xraygpt/configs/datasets/mimic/defaults.yaml << EOF
datasets:
  mimic:
    data_type: images
    build_info:
      storage: ./dataset/mimic
EOF

# Update OpenI dataset configuration
echo "Updating xraygpt/configs/datasets/openi/defaults.yaml..."
cat > xraygpt/configs/datasets/openi/defaults.yaml << EOF
datasets:
  openi:
    data_type: images
    build_info:
      storage: ./dataset/openi
EOF

echo "✅ Configuration files updated successfully"

echo "📋 Setup Summary:"
echo "=================="
echo "✅ Directory structure created"
echo "✅ Configuration files updated"
echo ""
echo "📥 Next Steps - Download Required Assets:"
echo "1. Download MIMIC-CXR dataset to: dataset/mimic/"
echo "2. Download OpenI dataset to: dataset/openi/"
echo "3. Download Vicuna weights to: vicuna_weights/"
echo "4. Download MiniGPT-4 checkpoint to: xraygpt/pretrained_ckpt/"
echo ""
echo "🔗 Download Links:"
echo "- MIMIC annotations: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EZ6500itBIVMnD7sUztdMQMBVWVe7fuF7ta4FV78hpGSwg?e=wyL7Z7"
echo "- OpenI annotations: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EVYGprPyzdhOjFlQ2aNJbykBj49SwTGBYmC1uJ7TMswaVQ?e=qdqS8U"
echo "- Vicuna weights: https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EWoMYn3x7sdEnM2CdJRwWZgBCkMpLM03bk4GR5W0b3KIQQ?e=q6hEBz"
echo "- MiniGPT-4 checkpoint: https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?pli=1"
echo ""
echo "🚂 Training Commands:"
echo "Stage 1: torchrun --nproc-per-node 4 train.py --cfg-path train_configs/xraygpt_mimic_pretrain.yaml"
echo "Stage 2: torchrun --nproc-per-node 1 train.py --cfg-path train_configs/xraygpt_openi_finetune.yaml"
echo ""
echo "🎉 Setup complete! Follow the training guide for next steps."