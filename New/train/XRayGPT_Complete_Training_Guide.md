# XRayGPT Complete Training Guide - Exact Steps

## 🚀 **COMPLETE TRAINING PIPELINE WITH EXACT PATHS**

### **STEP 1: ENVIRONMENT SETUP**

#### **1.1 Clone Repository**
```bash
git clone https://github.com/mbzuai-oryx/XrayGPT.git
cd XRayGPT
```

#### **1.2 Create Environment (Choose One)**

**Option A: Using Conda**
```bash
conda env create -f env.yml
conda activate xraygpt
```

**Option B: Using Pip**
```bash
conda create -n xraygpt python=3.9
conda activate xraygpt
pip install -r xraygpt_requirements.txt
```

---

### **STEP 2: DOWNLOAD ALL REQUIRED ASSETS**

#### **2.1 Create Directory Structure**
```bash
# Create main directories
mkdir -p dataset/mimic/image
mkdir -p dataset/openi/image
mkdir -p vicuna_weights
mkdir -p xraygpt/pretrained_ckpt
mkdir -p output/xraygpt_mimic_pretrain
mkdir -p output/xraygpt_openi_finetune
```

#### **2.2 Download Datasets**

**MIMIC-CXR Dataset:**
```bash
# Download MIMIC-CXR images from PhysioNet (requires credentialed access)
# URL: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
# Place images in: dataset/mimic/image/

# Download preprocessed annotations
wget "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EZ6500itBIVMnD7sUztdMQMBVWVe7fuF7ta4FV78hpGSwg?e=wyL7Z7" -O dataset/mimic/filter_cap.json
```

**OpenI Dataset:**
```bash
# Download OpenI images from NIH
# URL: https://openi.nlm.nih.gov/faq#collection
# Place images in: dataset/openi/image/

# Download preprocessed annotations
wget "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EVYGprPyzdhOjFlQ2aNJbykBj49SwTGBYmC1uJ7TMswaVQ?e=qdqS8U" -O dataset/openi/filter_cap.json
```

#### **2.3 Download Model Weights**

**Vicuna Weights (Fine-tuned on Medical Data):**
```bash
# Download from SharePoint link
wget "https://mbzuaiac-my.sharepoint.com/:u:/g/personal/omkar_thawakar_mbzuai_ac_ae/EWoMYn3x7sdEnM2CdJRwWZgBCkMpLM03bk4GR5W0b3KIQQ?e=q6hEBz" -O vicuna_weights.zip
unzip vicuna_weights.zip -d vicuna_weights/
```

**MiniGPT-4 Checkpoint:**
```bash
# Download MiniGPT-4 pretrained checkpoint
wget "https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?pli=1" -O xraygpt/pretrained_ckpt/pretrained_minigpt4_7b.pth
```

**Final Directory Structure Should Look Like:**
```
XRayGPT/
├── dataset/
│   ├── mimic/
│   │   ├── image/
│   │   │   ├── abea5eb9-b7c32823-3a14c5ca-77868030-69c83139.jpg
│   │   │   ├── 427446c1-881f5cce-85191ce1-91a58ba9-0a57d3f5.jpg
│   │   │   └── ... (241k images)
│   │   └── filter_cap.json
│   └── openi/
│       ├── image/
│       │   ├── 1.png
│       │   ├── 2.png
│       │   └── ... (3.4k images)
│       └── filter_cap.json
├── vicuna_weights/
│   ├── config.json
│   ├── generation_config.json
│   ├── pytorch_model.bin.index.json
│   ├── pytorch_model-00001-of-00003.bin
│   └── ...
└── xraygpt/pretrained_ckpt/
    └── pretrained_minigpt4_7b.pth
```

---

### **STEP 3: CONFIGURATION FILE UPDATES**

#### **3.1 Update Model Configuration**
**File: `xraygpt/configs/models/xraygpt.yaml`**
```yaml
# CHANGE LINE 15 FROM:
llama_model: "./Vicuna_Radiology_fp16/"
# TO:
llama_model: "./vicuna_weights/"
```

#### **3.2 Update Dataset Configurations**

**File: `xraygpt/configs/datasets/mimic/defaults.yaml`**
```yaml
datasets:
  mimic:
    data_type: images
    build_info:
      # CHANGE FROM:
      # storage: /home/omkarthawakar/fahad/XrayGPT/dataset/mimic_test
      # TO:
      storage: ./dataset/mimic
```

**File: `xraygpt/configs/datasets/openi/defaults.yaml`**
```yaml
datasets:
  openi:
    data_type: images
    build_info:
      # CHANGE FROM:
      # storage: /home/omkarthawakar/fahad/XrayGPT/dataset/openi
      # TO:
      storage: ./dataset/openi
```

#### **3.3 Update Training Configurations**

**File: `train_configs/xraygpt_mimic_pretrain.yaml`**
```yaml
# No changes needed - paths are relative
# But verify output directory exists:
run:
  output_dir: "output/xraygpt_mimic_pretrain"  # ✓ Correct
```

**File: `train_configs/xraygpt_openi_finetune.yaml`**
```yaml
model:
  # VERIFY this path is correct:
  ckpt: './xraygpt/pretrained_ckpt/pretrained_minigpt4_7b.pth'  # ✓ Correct
  
run:
  output_dir: "output/xraygpt_openi_finetune"  # ✓ Correct
  # IMPORTANT: Set resume_ckpt_path to Stage 1 output
  resume_ckpt_path: "output/xraygpt_mimic_pretrain/checkpoint_best.pth"
```

---

### **STEP 4: TRAINING EXECUTION**

#### **4.1 Stage 1: MIMIC Pre-training**

**Single GPU Training:**
```bash
python train.py --cfg-path train_configs/xraygpt_mimic_pretrain.yaml
```

**Multi-GPU Training (Recommended):**
```bash
# For 4 GPUs
torchrun --nproc-per-node 4 train.py --cfg-path train_configs/xraygpt_mimic_pretrain.yaml

# For 8 GPUs
torchrun --nproc-per-node 8 train.py --cfg-path train_configs/xraygpt_mimic_pretrain.yaml

# For specific GPU IDs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 train.py --cfg-path train_configs/xraygpt_mimic_pretrain.yaml
```

**Expected Output:**
- Training logs in console
- Checkpoints saved to: `output/xraygpt_mimic_pretrain/`
- Best checkpoint: `output/xraygpt_mimic_pretrain/checkpoint_best.pth`
- Training time: ~2-3 days on 4 GPUs

#### **4.2 Update Configuration for Stage 2**

**Before Stage 2, update `train_configs/xraygpt_openi_finetune.yaml`:**
```yaml
run:
  # ADD this line to resume from Stage 1:
  resume_ckpt_path: "output/xraygpt_mimic_pretrain/checkpoint_best.pth"
```

#### **4.3 Stage 2: OpenI Fine-tuning**

**Single GPU Training:**
```bash
python train.py --cfg-path train_configs/xraygpt_openi_finetune.yaml
```

**Multi-GPU Training:**
```bash
# Usually single GPU is sufficient for fine-tuning
torchrun --nproc-per-node 1 train.py --cfg-path train_configs/xraygpt_openi_finetune.yaml

# Or 2 GPUs if needed
torchrun --nproc-per-node 2 train.py --cfg-path train_configs/xraygpt_openi_finetune.yaml
```

**Expected Output:**
- Fine-tuning logs in console
- Checkpoints saved to: `output/xraygpt_openi_finetune/`
- Final model: `output/xraygpt_openi_finetune/checkpoint_best.pth`
- Training time: ~4-6 hours on 1 GPU

---

### **STEP 5: MODEL EVALUATION**

#### **5.1 Update Evaluation Configuration**
**File: `eval_configs/xraygpt_eval.yaml`**
```yaml
model:
  # CHANGE FROM:
  # ckpt: './xraygpt/output/path_to_ckpt'
  # TO:
  ckpt: './output/xraygpt_openi_finetune/checkpoint_best.pth'
```

#### **5.2 Run Evaluation**
```bash
python test.py --cfg-path eval_configs/xraygpt_eval.yaml
```

#### **5.3 Run Interactive Demo**
```bash
python demo.py --cfg-path eval_configs/xraygpt_eval.yaml --gpu-id 0
```

---

### **STEP 6: TRAINING MONITORING & TROUBLESHOOTING**

#### **6.1 Monitor Training Progress**
```bash
# Check GPU usage
nvidia-smi

# Monitor training logs
tail -f output/xraygpt_mimic_pretrain/log.txt

# Check checkpoint files
ls -la output/xraygpt_mimic_pretrain/
```

#### **6.2 Resume Training (If Interrupted)**
```yaml
# In training config file, set:
run:
  resume_ckpt_path: "output/xraygpt_mimic_pretrain/checkpoint_X.pth"
```

#### **6.3 Adjust Batch Size (If Out of Memory)**
```yaml
# In training config, reduce batch size:
run:
  batch_size_train: 1  # Reduce from 3 to 1
  batch_size_eval: 1   # Reduce from 3 to 1
```

---

### **STEP 7: VALIDATION & TESTING**

#### **7.1 Validate Training Success**
```bash
# Check if final checkpoint exists
ls -la output/xraygpt_openi_finetune/checkpoint_best.pth

# Test model loading
python -c "
import torch
ckpt = torch.load('output/xraygpt_openi_finetune/checkpoint_best.pth')
print('Model loaded successfully!')
print('Keys:', list(ckpt.keys()))
"
```

#### **7.2 Test on Sample Images**
```bash
# Run demo with sample images
python demo.py --cfg-path eval_configs/xraygpt_eval.yaml --gpu-id 0
```

---

### **STEP 8: HARDWARE REQUIREMENTS**

#### **8.1 Minimum Requirements**
- **GPU**: 1x RTX 3090 (24GB VRAM) or equivalent
- **RAM**: 32GB system RAM
- **Storage**: 500GB free space
- **Training Time**: 3-4 days total

#### **8.2 Recommended Requirements**
- **GPU**: 4x RTX A100 (40GB VRAM each)
- **RAM**: 128GB system RAM
- **Storage**: 1TB NVMe SSD
- **Training Time**: 1-2 days total

---

### **STEP 9: COMMON ISSUES & SOLUTIONS**

#### **9.1 CUDA Out of Memory**
```bash
# Solution 1: Reduce batch size
# In config files, change:
batch_size_train: 1
batch_size_eval: 1

# Solution 2: Use gradient accumulation
# Add to config:
gradient_accumulation_steps: 4
```

#### **9.2 Dataset Path Issues**
```bash
# Verify dataset structure
ls -la dataset/mimic/
ls -la dataset/openi/

# Check annotation files
head -5 dataset/mimic/filter_cap.json
head -5 dataset/openi/filter_cap.json
```

#### **9.3 Model Loading Issues**
```bash
# Check Vicuna weights
ls -la vicuna_weights/
# Should contain: config.json, pytorch_model*.bin files

# Check MiniGPT-4 checkpoint
ls -la xraygpt/pretrained_ckpt/pretrained_minigpt4_7b.pth
```

---

### **STEP 10: FINAL VERIFICATION**

#### **10.1 Complete Training Verification Checklist**
- [ ] Environment setup complete
- [ ] All datasets downloaded and placed correctly
- [ ] All model weights downloaded
- [ ] Configuration files updated with correct paths
- [ ] Stage 1 training completed successfully
- [ ] Stage 2 fine-tuning completed successfully
- [ ] Model evaluation runs without errors
- [ ] Demo interface works correctly

#### **10.2 Expected Final Directory Structure**
```
XRayGPT/
├── output/
│   ├── xraygpt_mimic_pretrain/
│   │   ├── checkpoint_best.pth
│   │   ├── log.txt
│   │   └── ...
│   └── xraygpt_openi_finetune/
│       ├── checkpoint_best.pth  # ← FINAL TRAINED MODEL
│       ├── log.txt
│       └── ...
├── dataset/ (as described above)
├── vicuna_weights/ (as described above)
└── xraygpt/pretrained_ckpt/ (as described above)
```

**🎉 Congratulations! You now have a fully trained XRayGPT model ready for medical X-ray analysis!**