# XRayGPT Project Structure and Flow Analysis

## Project Overview
XRayGPT is a medical vision-language model designed for chest radiograph summarization. It combines a medical visual encoder (MedCLIP) with a fine-tuned language model (Vicuna-7B) to generate interactive summaries from X-ray images.

## Architecture Flow

### 1. **Data Pipeline**
```
Raw X-ray Images + Reports → Data Cleaning → Preprocessed Summaries → Training Data
```

**Components:**
- **MIMIC-CXR Dataset**: 377k images, 227k reports → 114,690 filtered reports with 241k training images
- **OpenI Dataset**: 3,403 high-quality summaries for fine-tuning
- **GPT-3.5 Preprocessing**: Cleans reports, removes comparisons, generates interactive summaries

### 2. **Model Architecture**
```
X-ray Image → MedCLIP Encoder → Q-Former → Vicuna-7B LLM → Medical Summary
```

**Key Components:**
- **Vision Encoder**: MedCLIP (frozen during training)
- **Q-Former**: 32 query tokens for vision-language alignment
- **Language Model**: Vicuna-7B fine-tuned on medical conversations
- **Linear Transformation**: Aligns visual and textual representations

### 3. **Training Pipeline**
```
Stage 1: MIMIC Pre-training → Stage 2: OpenI Fine-tuning → Final Model
```

**Training Stages:**
1. **MIMIC Pre-training**: Large-scale image-text alignment on MIMIC dataset
2. **OpenI Fine-tuning**: High-quality fine-tuning on curated OpenI summaries

## Directory Structure Analysis

```
XRayGPT/
├── xraygpt/                    # Core package
│   ├── models/                 # Model architectures
│   │   ├── mini_gpt4.py       # Main XRayGPT model
│   │   ├── blip2.py           # BLIP-2 components
│   │   ├── eva_vit.py         # Vision transformer
│   │   ├── Qformer.py         # Query transformer
│   │   └── modeling_llama.py  # LLaMA/Vicuna components
│   ├── datasets/              # Dataset handling
│   │   ├── datasets/
│   │   │   ├── mimic_dataset.py    # MIMIC dataset loader
│   │   │   └── openi_dataset.py    # OpenI dataset loader
│   │   └── builders/          # Dataset builders
│   ├── processors/            # Data preprocessing
│   ├── configs/              # Configuration files
│   │   ├── models/           # Model configs
│   │   └── datasets/         # Dataset configs
│   └── runners/              # Training/evaluation runners
├── train_configs/            # Training configurations
├── eval_configs/            # Evaluation configurations
├── train.py                 # Training script
├── test.py                  # Testing script
├── demo.py                  # Interactive demo
└── README.md               # Documentation
```

## Key Configuration Files

### Training Configurations
- **`xraygpt_mimic_pretrain.yaml`**: Stage 1 training on MIMIC dataset
- **`xraygpt_openi_finetune.yaml`**: Stage 2 fine-tuning on OpenI dataset

### Model Configuration
- **`xraygpt.yaml`**: Main model architecture and component settings

### Dataset Configurations
- **`mimic/defaults.yaml`**: MIMIC dataset paths and settings
- **`openi/defaults.yaml`**: OpenI dataset paths and settings

## Workflow Summary

### 1. **Setup Phase**
- Install dependencies (`env.yml` or `xraygpt_requirements.txt`)
- Download datasets (MIMIC-CXR, OpenI)
- Download pre-trained weights (Vicuna, MiniGPT-4)
- Configure paths in YAML files

### 2. **Data Preparation**
- Process raw radiology reports using GPT-3.5
- Generate clean, interactive summaries
- Organize datasets in required folder structure

### 3. **Training Phase**
- **Stage 1**: Pre-train on MIMIC dataset for image-text alignment
- **Stage 2**: Fine-tune on OpenI dataset for high-quality outputs

### 4. **Evaluation Phase**
- Test model performance using `test.py`
- Run interactive demo using `demo.py`
- Validate clinical accuracy and relevance

### 5. **Deployment**
- Use trained model for inference
- Deploy as web demo (Gradio interface)
- Integrate into clinical workflows

## Key Features

1. **Medical Domain Adaptation**: Fine-tuned on 100k patient-doctor conversations
2. **High-Quality Data**: Preprocessed summaries using GPT-3.5
3. **Modular Architecture**: Separate components for vision, language, and alignment
4. **Two-Stage Training**: Progressive training from general to specific
5. **Interactive Interface**: Gradio-based demo for easy testing

## Dependencies and Requirements

- **Deep Learning**: PyTorch, transformers, LAVIS
- **Vision**: PIL, OpenCV for image processing
- **Language**: Vicuna/LLaMA models
- **Data**: WebDataset for efficient data loading
- **Interface**: Gradio for demo deployment

This structure enables efficient training, evaluation, and deployment of the XRayGPT model for medical image analysis and report generation.