# XRayGPT File Flow Diagram

## 🌊 **COMPLETE SYSTEM FLOW**

```mermaid
graph TD
    %% Entry Points
    A[train.py] --> B[config.py]
    C[test.py] --> B
    D[demo.py] --> B
    
    %% Configuration Layer
    B --> E[registry.py]
    F[xraygpt_mimic_pretrain.yaml] --> B
    G[xraygpt_openi_finetune.yaml] --> B
    H[xraygpt_eval.yaml] --> B
    I[xraygpt.yaml] --> B
    J[mimic/defaults.yaml] --> B
    K[openi/defaults.yaml] --> B
    
    %% Registry connects to all components
    E --> L[Models]
    E --> M[Datasets]
    E --> N[Tasks]
    E --> O[Runners]
    E --> P[Processors]
    
    %% Model Architecture
    L --> Q[mini_gpt4.py]
    Q --> R[blip2.py]
    Q --> S[eva_vit.py]
    Q --> T[Qformer.py]
    Q --> U[modeling_llama.py]
    V[base_model.py] --> Q
    W[blip2_outputs.py] --> R
    
    %% Dataset Pipeline
    M --> X[image_text_pair_builder.py]
    X --> Y[mimic_dataset.py]
    X --> Z[openi_dataset.py]
    AA[base_dataset_builder.py] --> X
    BB[caption_datasets.py] --> Y
    BB --> Z
    CC[base_dataset.py] --> BB
    DD[dataloader_utils.py] --> Y
    DD --> Z
    EE[data_utils.py] --> Y
    EE --> Z
    
    %% Processing Pipeline
    P --> FF[blip_processors.py]
    GG[base_processor.py] --> FF
    HH[randaugment.py] --> FF
    FF --> Y
    FF --> Z
    
    %% Task Management
    N --> II[image_text_pretrain.py]
    JJ[base_task.py] --> II
    
    %% Training Execution
    O --> KK[runner_base.py]
    KK --> II
    KK --> Q
    KK --> Y
    KK --> Z
    
    %% Conversation & Prompts
    LL[conversation.py] --> D
    MM[alignment.txt] --> LL
    NN[alignment_original.txt] --> LL
    
    %% Common Utilities
    OO[dist_utils.py] --> A
    OO --> C
    PP[logger.py] --> A
    PP --> C
    PP --> D
    QQ[optims.py] --> KK
    RR[utils.py] --> A
    RR --> C
    RR --> D
    SS[gradcam.py] --> C
    
    %% Environment
    TT[env.yml] --> UU[Runtime Environment]
    VV[xraygpt_requirements.txt] --> UU
    
    %% Documentation
    WW[README.md] --> XX[Documentation]
    YY[README-DATASET.md] --> XX
```

## 🔄 **DETAILED EXECUTION FLOWS**

### **1. TRAINING EXECUTION FLOW**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   train.py  │───▶│  config.py  │───▶│ registry.py │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ dist_utils  │    │ YAML configs│    │All Components│
│ logger.py   │    │ validation  │    │Registration │
│ utils.py    │    └─────────────┘    └─────────────┘
└─────────────┘            │                   │
        │                   ▼                   ▼
        │          ┌─────────────┐    ┌─────────────┐
        │          │Task Builder │    │Model Builder│
        │          │image_text_  │    │mini_gpt4.py │
        │          │pretrain.py  │    └─────────────┘
        │          └─────────────┘            │
        │                   │                   ▼
        │                   ▼          ┌─────────────┐
        │          ┌─────────────┐    │   blip2.py  │
        │          │Dataset      │    │   eva_vit   │
        │          │Builder      │    │   Qformer   │
        │          └─────────────┘    │modeling_llama│
        │                   │          └─────────────┘
        │                   ▼                   │
        │          ┌─────────────┐            │
        │          │MIMIC/OpenI  │            │
        │          │Datasets     │            │
        │          └─────────────┘            │
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│              runner_base.py                         │
│         (Training Loop Orchestrator)                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │   Model     │ │  Datasets   │ │Optimization │   │
│  │ Training    │ │  Loading    │ │ & Logging   │   │
│  └─────────────┘ └─────────────┘ └─────────────┘   │
└─────────────────────────────────────────────────────┘
```

### **2. DATASET PROCESSING FLOW**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Raw X-ray    │    │filter_cap   │    │Dataset      │
│Images       │───▶│.json        │───▶│Config       │
│(.jpg/.png)  │    │(Annotations)│    │(YAML)       │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│Image        │    │Caption      │    │Builder      │
│Processors   │    │Processors   │    │Registration │
│blip_        │    │(Text        │    │(Registry)   │
│processors   │    │Processing)  │    └─────────────┘
└─────────────┘    └─────────────┘            │
        │                   │                   ▼
        ▼                   ▼          ┌─────────────┐
┌─────────────┐    ┌─────────────┐    │image_text_  │
│Augmented    │    │Processed    │    │pair_builder │
│Images       │    │Captions     │    └─────────────┘
│(224x224)    │    │(Tokenized)  │            │
└─────────────┘    └─────────────┘            ▼
        │                   │          ┌─────────────┐
        └───────────────────┼─────────▶│mimic_dataset│
                            │          │openi_dataset│
                            │          └─────────────┘
                            │                   │
                            └───────────────────▼
                                    ┌─────────────┐
                                    │DataLoader   │
                                    │(Batched     │
                                    │ Data)       │
                                    └─────────────┘
```

### **3. MODEL ARCHITECTURE FLOW**
```
┌─────────────┐
│ X-ray Image │
│ (224x224x3) │
└─────────────┘
        │
        ▼
┌─────────────┐    ┌─────────────┐
│ eva_vit.py  │───▶│ Visual      │
│ (MedCLIP    │    │ Features    │
│  Encoder)   │    │ (Frozen)    │
└─────────────┘    └─────────────┘
        │                   │
        ▼                   ▼
┌─────────────┐    ┌─────────────┐
│ Qformer.py  │───▶│ Query       │
│ (32 Query   │    │ Features    │
│  Tokens)    │    │ (Frozen)    │
└─────────────┘    └─────────────┘
        │                   │
        ▼                   ▼
┌─────────────┐    ┌─────────────┐
│Linear       │───▶│ Aligned     │
│Projection   │    │ Features    │
│(Trainable)  │    │ (Trainable) │
└─────────────┘    └─────────────┘
        │                   │
        ▼                   ▼
┌─────────────┐    ┌─────────────┐
│modeling_    │───▶│ Medical     │
│llama.py     │    │ Summary     │
│(Vicuna-7B)  │    │ (Generated) │
└─────────────┘    └─────────────┘
```

### **4. CONFIGURATION CASCADE FLOW**
```
┌─────────────┐
│default.yaml │ (Base Config)
└─────────────┘
        │
        ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│xraygpt.yaml │    │mimic/       │    │openi/       │
│(Model       │    │defaults.yaml│    │defaults.yaml│
│ Config)     │    │(Dataset)    │    │(Dataset)    │
└─────────────┘    └─────────────┘    └─────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                   ┌─────────────┐
                   │Training     │
                   │Configs:     │
                   │- mimic_     │
                   │  pretrain   │
                   │- openi_     │
                   │  finetune   │
                   │- eval       │
                   └─────────────┘
                            │
                            ▼
                   ┌─────────────┐
                   │config.py    │
                   │(Merged      │
                   │ Final       │
                   │ Config)     │
                   └─────────────┘
```

## 🎯 **KEY INTEGRATION POINTS**

### **Registry System Hub**
- **`registry.py`** connects ALL components
- Every `__init__.py` registers components
- Enables dynamic discovery and instantiation

### **Configuration Management**
- **`config.py`** merges all YAML files
- Validates parameters and paths
- Provides unified configuration interface

### **Training Orchestration**
- **`runner_base.py`** coordinates everything
- Manages distributed training
- Handles checkpointing and logging

### **Data Pipeline**
- **Builders** create datasets from configs
- **Processors** handle image/text preprocessing
- **Datasets** provide batched data to training

This flow diagram shows how all 55 files interconnect to create the complete XRayGPT system!