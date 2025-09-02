# XRayGPT Task Scripts Mapping

## 📋 **ACTUAL SCRIPTS FOR EACH TASK IN XRayGPT_Tasks_Table.csv**

The `XRayGPT_Tasks_Table.csv` file is a **reference documentation** I created to list all tasks. Here are the **actual scripts and commands** used for each task:

---

## 🗂️ **DATASET PREPARATION TASKS**

### **Task 1: MIMIC-CXR Dataset**
**Scripts Used:**
```bash
# No automated script - Manual download required
# 1. Register at PhysioNet: https://physionet.org/content/mimic-cxr-jpg/2.0.0/
# 2. Download images manually
# 3. Download preprocessed annotations:
wget "https://mbzuaiac-my.sharepoint.com/..." -O dataset/mimic/filter_cap.json
```

**Dataset Loading Script:**
- **File**: `xraygpt/datasets/datasets/mimic_dataset.py`
- **Builder**: `xraygpt/datasets/builders/image_text_pair_builder.py`

### **Task 2: OpenI Dataset**
**Scripts Used:**
```bash
# No automated script - Manual download required
# 1. Download from: https://openi.nlm.nih.gov/faq#collection
# 2. Download preprocessed annotations:
wget "https://mbzuaiac-my.sharepoint.com/..." -O dataset/openi/filter_cap.json
```

**Dataset Loading Script:**
- **File**: `xraygpt/datasets/datasets/openi_dataset.py`

---

## 🧹 **DATA CLEANING TASKS**

### **Task 3-4: Report Preprocessing & Summary Generation**
**Scripts Used:**
- **No direct scripts in repository** - Preprocessing was done offline using GPT-3.5-turbo
- **Result**: Pre-processed `filter_cap.json` files for both datasets
- **Documentation**: `README-DATASET.md` explains the methodology

---

## 🧠 **MODEL WEIGHTS TASKS**

### **Task 5: Vicuna Base Model**
**Scripts Used:**
```bash
# Download Vicuna weights
wget "https://mbzuaiac-my.sharepoint.com/..." -O vicuna_weights.zip
unzip vicuna_weights.zip -d vicuna_weights/

# Update configuration
# Edit: xraygpt/configs/models/xraygpt.yaml
# Line 15: llama_model: "./vicuna_weights/"
```

### **Task 6: Vicuna Fine-tuning Data**
**Scripts Used:**
```bash
# Download medical conversation data
wget "https://mbzuaiac-my.sharepoint.com/..." -O radiology_data.zip
wget "https://mbzuaiac-my.sharepoint.com/..." -O medical_healthcare_data.zip

# Fine-tuning using FastChat repository
# Refer to: https://github.com/lm-sys/FastChat#fine-tuning
```

### **Task 7: MiniGPT-4 Checkpoint**
**Scripts Used:**
```bash
# Download MiniGPT-4 checkpoint
wget "https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view" -O xraygpt/pretrained_ckpt/pretrained_minigpt4_7b.pth
```

### **Task 8: XRayGPT Pre-trained Model**
**Scripts Used:**
```bash
# Download final trained model (for inference only)
wget "https://mbzuaiac-my.sharepoint.com/..." -O xraygpt_final_model.pth
```

---

## 🚂 **MODEL TRAINING TASKS**

### **Task 9: MIMIC Pre-training Stage**
**Main Script:** `train.py`
```bash
# Single GPU
python train.py --cfg-path train_configs/xraygpt_mimic_pretrain.yaml

# Multi-GPU (Recommended)
torchrun --nproc-per-node 4 train.py --cfg-path train_configs/xraygpt_mimic_pretrain.yaml

# With specific GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 train.py --cfg-path train_configs/xraygpt_mimic_pretrain.yaml
```

**Supporting Scripts:**
- **Runner**: `xraygpt/runners/runner_base.py`
- **Task**: `xraygpt/tasks/image_text_pretrain.py`
- **Model**: `xraygpt/models/mini_gpt4.py`

### **Task 10: OpenI Fine-tuning Stage**
**Main Script:** `train.py`
```bash
# Single GPU (Usually sufficient)
python train.py --cfg-path train_configs/xraygpt_openi_finetune.yaml

# Multi-GPU
torchrun --nproc-per-node 2 train.py --cfg-path train_configs/xraygpt_openi_finetune.yaml
```

---

## ✅ **MODEL VALIDATION TASKS**

### **Task 11: Model Testing**
**Main Script:** `test.py`
```bash
# Run evaluation
python test.py --cfg-path eval_configs/xraygpt_eval.yaml

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python test.py --cfg-path eval_configs/xraygpt_eval.yaml
```

### **Task 12: Demo Evaluation**
**Main Script:** `demo.py`
```bash
# Launch interactive demo
python demo.py --cfg-path eval_configs/xraygpt_eval.yaml --gpu-id 0

# With custom port
python demo.py --cfg-path eval_configs/xraygpt_eval.yaml --gpu-id 0 --port 7860
```

### **Task 13: Online Demo**
**Scripts Used:**
- **Same as Task 12** but deployed on cloud servers
- **Gradio interface** from `demo.py`

---

## 🔧 **ENVIRONMENT SETUP TASKS**

### **Task 14: Dependencies Installation**
**Scripts Used:**
```bash
# Option 1: Conda environment
conda env create -f env.yml
conda activate xraygpt

# Option 2: Pip installation
conda create -n xraygpt python=3.9
conda activate xraygpt
pip install -r xraygpt_requirements.txt
```

### **Task 15: Project Structure**
**Scripts Used:**
```bash
# Automated setup script I created
./setup_xraygpt_training.sh

# Manual setup
mkdir -p dataset/mimic/image
mkdir -p dataset/openi/image
mkdir -p vicuna_weights
mkdir -p xraygpt/pretrained_ckpt
mkdir -p output/xraygpt_mimic_pretrain
mkdir -p output/xraygpt_openi_finetune
```

---

## 🏗️ **MODEL ARCHITECTURE TASKS**

### **Task 16-18: Vision Encoder, Language Model, Q-Former**
**Scripts Used:**
- **No separate scripts** - These are **code components** loaded during training/inference
- **Files**:
  - `xraygpt/models/eva_vit.py` - Vision encoder
  - `xraygpt/models/mini_gpt4.py` - Main model
  - `xraygpt/models/modeling_llama.py` - Language model
  - `xraygpt/models/Qformer.py` - Q-Former

---

## 📊 **EVALUATION METRICS TASKS**

### **Task 19: Accuracy Assessment**
**Scripts Used:**
```bash
# Same as model testing
python test.py --cfg-path eval_configs/xraygpt_eval.yaml
```

### **Task 20: Clinical Validation**
**Scripts Used:**
- **No automated scripts** - Requires manual expert evaluation
- **Custom evaluation scripts** can be created based on specific metrics

---

## 📚 **REFERENCE TASKS**

### **Task 21-22: Paper & Demo References**
**Scripts Used:**
- **No scripts** - These are **documentation links**
- **ArXiv Paper**: https://arxiv.org/abs/2306.07971
- **YouTube Demo**: https://youtu.be/-zzq7bzbUuY

---

## 🛠️ **HELPER SCRIPTS I CREATED**

### **Setup Automation:**
```bash
# Automated setup script
./setup_xraygpt_training.sh

# Validation script
python validate_training_setup.py
```

### **Configuration Updates:**
```bash
# Update dataset paths
sed -i 's|storage: /home/omkarthawakar/fahad/XrayGPT/dataset/mimic_test|storage: ./dataset/mimic|g' xraygpt/configs/datasets/mimic/defaults.yaml

# Update model paths
sed -i 's|llama_model: "./Vicuna_Radiology_fp16/"|llama_model: "./vicuna_weights/"|g' xraygpt/configs/models/xraygpt.yaml
```

---

## 🎯 **SUMMARY**

### **Main Executable Scripts:**
1. **`train.py`** - Training (Tasks 9-10)
2. **`test.py`** - Evaluation (Task 11)
3. **`demo.py`** - Interactive demo (Task 12)
4. **`setup_xraygpt_training.sh`** - Setup automation (Task 15)
5. **`validate_training_setup.py`** - Validation (Helper)

### **Configuration Files:**
- **Training configs**: `train_configs/*.yaml`
- **Evaluation configs**: `eval_configs/*.yaml`
- **Model configs**: `xraygpt/configs/models/*.yaml`
- **Dataset configs**: `xraygpt/configs/datasets/*/defaults.yaml`

### **Manual Tasks:**
- **Dataset downloads** (Tasks 1-2)
- **Model weight downloads** (Tasks 5-8)
- **Clinical validation** (Task 20)

The CSV file serves as a **comprehensive task checklist**, while the actual implementation uses the scripts and commands listed above.