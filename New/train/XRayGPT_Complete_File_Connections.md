# XRayGPT Complete File Connections and Functions

## 🏗️ **COMPLETE FILE ARCHITECTURE MAP**

### **📁 ROOT LEVEL FILES**

#### **Entry Point Scripts**
1. **`train.py`** - Main Training Orchestrator
   - **Purpose**: Coordinates entire training pipeline
   - **Imports**: `xraygpt.tasks`, `xraygpt.common.*`, `xraygpt.datasets.builders.*`, `xraygpt.models.*`
   - **Connects to**: All config files, model files, dataset builders, runners
   - **Function**: Parses args → loads config → initializes distributed training → builds datasets/model → runs training

2. **`test.py`** - Main Testing/Evaluation Script
   - **Purpose**: Evaluates trained models
   - **Imports**: Same as train.py but calls `runner.test()` instead of `runner.train()`
   - **Connects to**: eval_configs, trained model checkpoints
   - **Function**: Loads trained model → runs evaluation → generates metrics

3. **`demo.py`** - Interactive Gradio Demo
   - **Purpose**: Web interface for real-time X-ray analysis
   - **Imports**: `xraygpt.common.config`, `xraygpt.common.registry`, model classes
   - **Connects to**: eval_configs, trained model weights, prompts
   - **Function**: Loads model → creates Gradio interface → processes uploaded X-rays → generates summaries

#### **Environment & Setup**
4. **`env.yml`** - Conda Environment Specification
   - **Purpose**: Defines all Python dependencies and versions
   - **Connects to**: All Python files (runtime dependencies)
   - **Contains**: PyTorch, transformers, gradio, PIL, numpy, etc.

5. **`xraygpt_requirements.txt`** - Pip Dependencies
   - **Purpose**: Alternative to conda for package installation
   - **Connects to**: All Python files (runtime dependencies)
   - **Function**: `pip install -r xraygpt_requirements.txt`

#### **Documentation**
6. **`README.md`** - Main Project Documentation
   - **Purpose**: Installation, usage, and architecture overview
   - **Connects to**: All files (documentation reference)
   - **Contains**: Setup instructions, training commands, download links

7. **`README-DATASET.md`** - Dataset Creation Documentation
   - **Purpose**: Explains data preprocessing and quality enhancement
   - **Connects to**: Dataset files, preprocessing scripts
   - **Contains**: GPT-3.5 preprocessing methodology, data statistics

---

### **⚙️ CONFIGURATION FILES**

#### **Training Configurations**
8. **`train_configs/xraygpt_mimic_pretrain.yaml`** - Stage 1 Training Config
   - **Purpose**: MIMIC dataset pretraining parameters
   - **Connects to**: `train.py`, MIMIC dataset builder, model configs
   - **Contains**: Learning rates, batch sizes, epochs, model architecture settings
   - **Function**: Defines first training stage on large MIMIC dataset

9. **`train_configs/xraygpt_openi_finetune.yaml`** - Stage 2 Training Config
   - **Purpose**: OpenI dataset fine-tuning parameters
   - **Connects to**: `train.py`, OpenI dataset builder, pretrained checkpoint
   - **Contains**: Fine-tuning hyperparameters, prompt templates, checkpoint paths
   - **Function**: Defines second training stage on high-quality OpenI data

#### **Evaluation Configuration**
10. **`eval_configs/xraygpt_eval.yaml`** - Evaluation Config
    - **Purpose**: Model evaluation and inference settings
    - **Connects to**: `test.py`, `demo.py`, trained model weights
    - **Contains**: Model checkpoint path, evaluation processors, inference settings

#### **Model Configuration**
11. **`xraygpt/configs/models/xraygpt.yaml`** - Model Architecture Config
    - **Purpose**: Defines model components and their settings
    - **Connects to**: All model files, training/eval configs
    - **Contains**: ViT settings, Q-Former parameters, Vicuna path, image size

12. **`xraygpt/configs/default.yaml`** - Default Global Config
    - **Purpose**: Base configuration template
    - **Connects to**: All other config files (inheritance)
    - **Function**: Provides default values for all configurations

#### **Dataset Configurations**
13. **`xraygpt/configs/datasets/mimic/defaults.yaml`** - MIMIC Dataset Config
    - **Purpose**: MIMIC dataset path and settings
    - **Connects to**: MIMIC dataset builder, training configs
    - **Contains**: Storage path, data type specification

14. **`xraygpt/configs/datasets/openi/defaults.yaml`** - OpenI Dataset Config
    - **Purpose**: OpenI dataset path and settings
    - **Connects to**: OpenI dataset builder, training configs
    - **Contains**: Storage path, data type specification

---

### **🧠 MODEL ARCHITECTURE FILES**

#### **Core Model Implementation**
15. **`xraygpt/models/mini_gpt4.py`** - Main XRayGPT Model
    - **Purpose**: Integrates all model components into XRayGPT
    - **Imports**: `blip2.py`, `Qformer.py`, `modeling_llama.py`, `eva_vit.py`
    - **Connects to**: All training/eval scripts, model configs
    - **Function**: Combines vision encoder + Q-Former + LLM for medical image analysis

16. **`xraygpt/models/blip2.py`** - BLIP-2 Framework
    - **Purpose**: Vision-language pretraining framework
    - **Imports**: `eva_vit.py`, `Qformer.py`, base model classes
    - **Connects to**: mini_gpt4.py, training pipeline
    - **Function**: Handles image-text alignment and multimodal learning

17. **`xraygpt/models/eva_vit.py`** - Vision Transformer (MedCLIP)
    - **Purpose**: Medical-aware image encoder
    - **Connects to**: blip2.py, mini_gpt4.py, image processors
    - **Function**: Encodes X-ray images into visual features

18. **`xraygpt/models/Qformer.py`** - Query Transformer
    - **Purpose**: Cross-modal alignment between vision and language
    - **Connects to**: blip2.py, mini_gpt4.py
    - **Function**: Uses 32 query tokens to bridge visual and textual representations

19. **`xraygpt/models/modeling_llama.py`** - LLaMA/Vicuna Implementation
    - **Purpose**: Language model for text generation
    - **Connects to**: mini_gpt4.py, conversation module
    - **Function**: Generates medical summaries from aligned features

#### **Supporting Model Files**
20. **`xraygpt/models/base_model.py`** - Base Model Class
    - **Purpose**: Common functionality for all models
    - **Connects to**: All model implementations
    - **Function**: Provides base methods for model loading, saving, configuration

21. **`xraygpt/models/blip2_outputs.py`** - BLIP-2 Output Classes
    - **Purpose**: Defines output structures for BLIP-2 models
    - **Connects to**: blip2.py, training/evaluation loops
    - **Function**: Handles model outputs, loss computation

22. **`xraygpt/models/__init__.py`** - Model Package Initializer
    - **Purpose**: Registers all model classes
    - **Connects to**: Registry system, training/eval scripts
    - **Function**: Makes models discoverable by the framework

---

### **📊 DATASET HANDLING FILES**

#### **Dataset Builders**
23. **`xraygpt/datasets/builders/image_text_pair_builder.py`** - Dataset Builders
    - **Purpose**: Constructs MIMIC and OpenI datasets
    - **Imports**: `mimic_dataset.py`, `openi_dataset.py`, base builder
    - **Connects to**: Training scripts, dataset configs
    - **Function**: Builds datasets from raw data and annotations

24. **`xraygpt/datasets/builders/base_dataset_builder.py`** - Base Dataset Builder
    - **Purpose**: Common dataset building functionality
    - **Connects to**: All specific dataset builders
    - **Function**: Provides base methods for dataset construction

25. **`xraygpt/datasets/builders/__init__.py`** - Dataset Builder Registry
    - **Purpose**: Registers dataset builders
    - **Connects to**: Registry system, training scripts
    - **Function**: Makes dataset builders discoverable

#### **Dataset Implementations**
26. **`xraygpt/datasets/datasets/mimic_dataset.py`** - MIMIC Dataset Class
    - **Purpose**: Loads MIMIC X-ray images and captions
    - **Imports**: `caption_datasets.py`, `base_dataset.py`
    - **Connects to**: MIMIC builder, training pipeline
    - **Function**: Handles .jpg images, processes image-caption pairs

27. **`xraygpt/datasets/datasets/openi_dataset.py`** - OpenI Dataset Class
    - **Purpose**: Loads OpenI X-ray images and captions
    - **Imports**: `caption_datasets.py`, `base_dataset.py`
    - **Connects to**: OpenI builder, training pipeline
    - **Function**: Handles .png images, processes high-quality summaries

28. **`xraygpt/datasets/datasets/caption_datasets.py`** - Caption Dataset Base
    - **Purpose**: Base class for image-caption datasets
    - **Connects to**: MIMIC and OpenI dataset classes
    - **Function**: Provides common caption processing functionality

29. **`xraygpt/datasets/datasets/base_dataset.py`** - Base Dataset Class
    - **Purpose**: Fundamental dataset functionality
    - **Connects to**: All dataset implementations
    - **Function**: Provides base methods for data loading and processing

#### **Dataset Utilities**
30. **`xraygpt/datasets/datasets/dataloader_utils.py`** - DataLoader Utilities
    - **Purpose**: Custom data loading functions
    - **Connects to**: Training/evaluation loops
    - **Function**: Handles batch processing, data sampling

31. **`xraygpt/datasets/data_utils.py`** - Data Processing Utilities
    - **Purpose**: Common data processing functions
    - **Connects to**: All dataset classes
    - **Function**: Data transformation, preprocessing utilities

32. **`xraygpt/datasets/__init__.py`** - Dataset Package Initializer
    - **Purpose**: Registers dataset classes
    - **Connects to**: Registry system
    - **Function**: Makes datasets discoverable

---

### **🔄 DATA PROCESSING FILES**

#### **Image Processors**
33. **`xraygpt/processors/blip_processors.py`** - BLIP Image Processors
    - **Purpose**: Image preprocessing for BLIP-2 format
    - **Connects to**: Dataset classes, model training
    - **Function**: Resizes, normalizes, augments X-ray images

34. **`xraygpt/processors/base_processor.py`** - Base Processor Class
    - **Purpose**: Common processing functionality
    - **Connects to**: All specific processors
    - **Function**: Provides base methods for data processing

35. **`xraygpt/processors/randaugment.py`** - Random Augmentation
    - **Purpose**: Data augmentation for training
    - **Connects to**: Image processors, training pipeline
    - **Function**: Applies random transformations to improve generalization

36. **`xraygpt/processors/__init__.py`** - Processor Registry
    - **Purpose**: Registers all processors
    - **Connects to**: Registry system
    - **Function**: Makes processors discoverable

---

### **🏃 TRAINING & EXECUTION FILES**

#### **Task Management**
37. **`xraygpt/tasks/image_text_pretrain.py`** - Pretraining Task
    - **Purpose**: Defines image-text pretraining task
    - **Connects to**: Training scripts, base task
    - **Function**: Manages pretraining workflow and evaluation

38. **`xraygpt/tasks/base_task.py`** - Base Task Class
    - **Purpose**: Common task functionality
    - **Connects to**: All specific tasks
    - **Function**: Provides base methods for task management

39. **`xraygpt/tasks/__init__.py`** - Task Registry
    - **Purpose**: Registers all tasks
    - **Connects to**: Registry system, training scripts
    - **Function**: Makes tasks discoverable

#### **Training Runners**
40. **`xraygpt/runners/runner_base.py`** - Base Training Runner
    - **Purpose**: Manages training and evaluation loops
    - **Connects to**: Training/test scripts, all models and datasets
    - **Function**: Handles distributed training, checkpointing, logging

41. **`xraygpt/runners/__init__.py`** - Runner Registry
    - **Purpose**: Registers training runners
    - **Connects to**: Registry system
    - **Function**: Makes runners discoverable

---

### **💬 CONVERSATION & PROMPTS**

#### **Conversation Management**
42. **`xraygpt/conversation/conversation.py`** - Conversation Handler
    - **Purpose**: Manages multi-turn conversations with the model
    - **Connects to**: Demo script, model inference
    - **Function**: Handles conversation state, prompt formatting

43. **`xraygpt/conversation/__init__.py`** - Conversation Registry
    - **Purpose**: Registers conversation classes
    - **Connects to**: Registry system
    - **Function**: Makes conversation handlers discoverable

#### **Prompt Templates**
44. **`prompts/alignment.txt`** - Training Prompts
    - **Purpose**: Contains diverse prompts for training alignment
    - **Connects to**: Training configs, conversation module
    - **Function**: Provides varied question formats for X-ray analysis

45. **`prompts/alignment_original.txt`** - Original Prompt Templates
    - **Purpose**: Backup/alternative prompt templates
    - **Connects to**: Training pipeline (optional)
    - **Function**: Alternative prompt formulations

---

### **🛠️ COMMON UTILITIES**

#### **Core Infrastructure**
46. **`xraygpt/common/config.py`** - Configuration Management
    - **Purpose**: Loads and manages all configuration files
    - **Connects to**: All scripts, all config files
    - **Function**: Parses YAML configs, merges settings, validates parameters

47. **`xraygpt/common/registry.py`** - Component Registry
    - **Purpose**: Central registry for all components (models, datasets, tasks, etc.)
    - **Connects to**: All __init__.py files, all component classes
    - **Function**: Enables dynamic component discovery and instantiation

48. **`xraygpt/common/dist_utils.py`** - Distributed Training Utilities
    - **Purpose**: Handles multi-GPU and distributed training
    - **Connects to**: Training scripts, runners
    - **Function**: Sets up distributed processes, handles communication

49. **`xraygpt/common/logger.py`** - Logging System
    - **Purpose**: Manages logging across the entire system
    - **Connects to**: All scripts and modules
    - **Function**: Provides structured logging, handles log levels

50. **`xraygpt/common/optims.py`** - Optimizers and Schedulers
    - **Purpose**: Custom optimizers and learning rate schedulers
    - **Connects to**: Training scripts, runners
    - **Function**: Implements LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler

51. **`xraygpt/common/utils.py`** - General Utilities
    - **Purpose**: Common utility functions
    - **Connects to**: All modules
    - **Function**: Provides helper functions for various tasks

52. **`xraygpt/common/gradcam.py`** - Gradient-weighted Class Activation Mapping
    - **Purpose**: Visualization and interpretability
    - **Connects to**: Model evaluation, analysis tools
    - **Function**: Generates attention maps for model interpretability

#### **Package Initializers**
53. **`xraygpt/__init__.py`** - Main Package Initializer
    - **Purpose**: Initializes the entire XRayGPT package
    - **Connects to**: All submodules
    - **Function**: Sets up package structure, imports key components

54. **`xraygpt/common/__init__.py`** - Common Package Initializer
    - **Purpose**: Initializes common utilities
    - **Connects to**: All common modules
    - **Function**: Makes common utilities available

---

## 🔗 **CRITICAL FILE CONNECTIONS FLOW**

### **Training Flow:**
```
train.py → config.py → registry.py → 
├── image_text_pair_builder.py → mimic_dataset.py/openi_dataset.py
├── mini_gpt4.py → blip2.py → eva_vit.py + Qformer.py + modeling_llama.py
├── runner_base.py → image_text_pretrain.py
└── blip_processors.py → Training Loop
```

### **Evaluation Flow:**
```
test.py/demo.py → config.py → registry.py →
├── Trained model weights
├── conversation.py → alignment.txt
└── blip_processors.py → Inference
```

### **Data Flow:**
```
Raw Data → filter_cap.json → 
dataset_builder.py → dataset_classes.py → 
processors.py → Training/Evaluation
```

### **Configuration Flow:**
```
YAML configs → config.py → 
├── Model configs → Model classes
├── Dataset configs → Dataset builders
└── Training configs → Runners
```

This comprehensive mapping shows how every single file in the XRayGPT repository connects and contributes to the overall system functionality.
