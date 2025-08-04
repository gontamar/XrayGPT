# Generic Multimodal Framework - Complete Usage Guide

## 🎯 Overview

This framework is **completely configuration-driven** with **no hardcoded defaults**. Every parameter must be explicitly specified in your configuration file, making it fully customizable for any domain or use case.

## 🚀 Quick Start

### 1. Choose Your Domain Configuration

Pick one of the pre-built configurations or create your own:

```bash
# Medical domain (X-ray analysis, clinical reports)
cp configs/medical_domain_config.yaml my_config.yaml

# General domain (everyday vision-language tasks)
cp configs/general_domain_config.yaml my_config.yaml

# Custom domain (start from template)
cp configs/custom_domain_template.yaml my_config.yaml
```

### 2. Basic Usage

```python
import yaml
from generic_multimodal_framework.models.generic_multimodal_model import GenericMultiModalModel

# Load your configuration
with open('my_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = GenericMultiModalModel(config)
model.eval()

# Prepare your data
data = {
    "image": your_image,  # PIL Image, torch.Tensor, or numpy array
    "text": "Your descriptive text",
    "query": "What do you want to know about this image?"
}

# Process through the framework
output = model.forward(data, return_attention=True)
print(f"Response: {output.text_response}")
```

## 📋 Configuration Structure

### Required Sections

Every configuration file **must** include these sections:

```yaml
domain:          # Domain specification
validation:      # Data validation rules
tokenization:    # Tokenization settings
encoding:        # Encoder configurations
attention:       # Attention mechanisms
decoding:        # Decoder settings
model:          # Model parameters
```

### Domain Configuration

```yaml
domain:
  name: "your_domain"  # REQUIRED: medical, general, satellite, etc.
  description: "Domain description"
```

### Validation Configuration

```yaml
validation:
  domain: "your_domain"  # REQUIRED: must match domain.name
  domain_name: "your_domain"  # REQUIRED: used in error messages
  min_image_size: [224, 224]  # REQUIRED: minimum image dimensions
  max_image_size: [1024, 1024]  # REQUIRED: maximum image dimensions
  min_text_length: 5  # REQUIRED: minimum text length
  max_text_length: 1000  # REQUIRED: maximum text length
  forbidden_text_patterns: []  # REQUIRED: patterns to filter
  required_channels: [1, 3]  # REQUIRED: allowed image channels
  require_domain_content: false  # REQUIRED: enforce domain content
  convert_to_grayscale: false  # REQUIRED: image preprocessing
  apply_histogram_equalization: false  # REQUIRED: contrast enhancement
  domain_keywords: []  # REQUIRED: keywords for content validation
  domain_replacements: {}  # REQUIRED: text standardization rules
```

### Tokenization Configuration

```yaml
tokenization:
  text:
    type: "text"  # REQUIRED: text, vision, multimodal
    tokenizer_type: "bert"  # REQUIRED: bert, llama, gpt2, t5, auto
    model_name: "bert-base-uncased"  # REQUIRED: HuggingFace model name
    max_length: 512  # REQUIRED: maximum sequence length
    padding: "max_length"  # REQUIRED: padding strategy
    truncation: true  # REQUIRED: truncation setting
    add_special_tokens: true  # REQUIRED: special tokens
    special_tokens: {}  # REQUIRED: custom special tokens
  
  image:
    type: "vision"  # REQUIRED: text, vision, multimodal
    method: "patch"  # REQUIRED: patch, vqvae
    patch_size: 16  # REQUIRED: patch size for vision
    image_size: 224  # REQUIRED: input image size
    vocab_size: 8192  # REQUIRED: vocabulary size
```

### Encoding Configuration

```yaml
encoding:
  device: "cuda"  # REQUIRED: cuda, cpu
  
  text:
    type: "bert"  # REQUIRED: bert, roberta, clip_text, medical_bert
    model_name: "bert-base-uncased"  # REQUIRED: model name
    hidden_size: 768  # REQUIRED: embedding dimension
    device: "cuda"  # REQUIRED: device
    pooling_strategy: "cls"  # REQUIRED: cls, mean, max
    freeze_encoder: false  # REQUIRED: freeze weights
    encoder_type: "bert"  # REQUIRED: specific implementation
    
  image:
    type: "vision_transformer"  # REQUIRED: encoder type
    model_name: "google/vit-base-patch16-224"  # REQUIRED: model name
    hidden_size: 768  # REQUIRED: embedding dimension
    device: "cuda"  # REQUIRED: device
    freeze_encoder: false  # REQUIRED: freeze weights
    encoder_type: "vision_transformer"  # REQUIRED: implementation
    image_size: 224  # REQUIRED: image size
    patch_size: 16  # REQUIRED: patch size
```

### Attention Configuration

```yaml
attention:
  self_attention:
    enabled: true  # REQUIRED: enable self-attention
    num_attention_heads: 12  # REQUIRED: number of heads
    hidden_size: 768  # REQUIRED: hidden dimension
    attention_dropout: 0.1  # REQUIRED: dropout rate
    layer_norm_eps: 1e-12  # REQUIRED: layer norm epsilon
    
  cross_attention:
    enabled: true  # REQUIRED: enable cross-attention
    num_attention_heads: 12  # REQUIRED: number of heads
    hidden_size: 768  # REQUIRED: hidden dimension
    attention_dropout: 0.1  # REQUIRED: dropout rate
    cross_attention_freq: 2  # REQUIRED: frequency
    save_attention: false  # REQUIRED: save attention maps
    
  qformer:
    enabled: false  # REQUIRED: enable Q-Former
    num_query_tokens: 32  # REQUIRED if enabled: query tokens
    num_layers: 6  # REQUIRED if enabled: layers
    cross_attention_freq: 2  # REQUIRED if enabled: frequency
```

### Decoding Configuration

```yaml
decoding:
  type: "llm"  # REQUIRED: llm, transformer
  model_name: "microsoft/DialoGPT-medium"  # REQUIRED: model name
  max_new_tokens: 128  # REQUIRED: max output length
  temperature: 0.7  # REQUIRED: sampling temperature
  top_p: 0.9  # REQUIRED: nucleus sampling
  top_k: 50  # REQUIRED: top-k sampling
  do_sample: true  # REQUIRED: enable sampling
  num_beams: 1  # REQUIRED: beam search
  repetition_penalty: 1.0  # REQUIRED: repetition penalty
  length_penalty: 1.0  # REQUIRED: length penalty
  early_stopping: true  # REQUIRED: early stopping
  pad_token_id: 50256  # REQUIRED: padding token
  eos_token_id: 50256  # REQUIRED: end token
```

### Model Configuration

```yaml
model:
  name: "YourModelName"  # REQUIRED: model name
  device: "cuda"  # REQUIRED: cuda, cpu
  dtype: "float16"  # REQUIRED: float16, float32, bfloat16
  gradient_checkpointing: false  # REQUIRED: memory optimization
  training_mode: false  # REQUIRED: training vs inference
```

## 🎯 Domain-Specific Examples

### Medical Domain

```python
# Medical X-ray analysis
config_path = "configs/medical_domain_config.yaml"
model = GenericMultiModalModel.from_config(config_path)

data = {
    "image": chest_xray_image,
    "text": "Chest X-ray examination",
    "query": "What are the findings in this chest X-ray?"
}

output = model.forward(data)
# Output: "The chest X-ray shows clear lung fields with no acute abnormalities..."
```

### General Domain

```python
# General vision-language understanding
config_path = "configs/general_domain_config.yaml"
model = GenericMultiModalModel.from_config(config_path)

data = {
    "image": everyday_photo,
    "text": "A photo from my vacation",
    "query": "Describe what you see in this image"
}

output = model.forward(data)
# Output: "This image shows a beautiful landscape with mountains..."
```

### Custom Domain (Satellite Imagery)

```python
# Create custom satellite domain config
config = {
    'domain': {'name': 'satellite'},
    'validation': {
        'domain': 'satellite',
        'domain_name': 'satellite',
        'min_image_size': [256, 256],
        'max_image_size': [2048, 2048],
        'domain_keywords': ['satellite', 'aerial', 'geographic', 'terrain'],
        # ... other required parameters
    },
    # ... complete configuration
}

model = GenericMultiModalModel(config)
```

## 🔧 Extending the Framework

### Adding Custom Validators

```python
from generic_multimodal_framework.core.data_validator import BaseDomainValidator

class LegalDomainValidator(BaseDomainValidator):
    def validate_image(self, image):
        # Legal document image validation
        pass
    
    def validate_text(self, text):
        # Legal text validation
        pass

# Register your validator
from generic_multimodal_framework.core.data_validator import DataValidatorFactory
DataValidatorFactory.register_validator("legal", LegalDomainValidator)
```

### Adding Custom Encoders

```python
from generic_multimodal_framework.core.encoder_manager import BaseEncoder

class LegalDocumentEncoder(BaseEncoder):
    def encode(self, inputs):
        # Legal document encoding logic
        pass

# Register your encoder
from generic_multimodal_framework.core.encoder_manager import EncoderManager
EncoderManager.register_encoder("legal_encoder", LegalDocumentEncoder)
```

## ⚠️ Important Notes

1. **No Defaults**: Every parameter must be explicitly specified in your config
2. **Validation**: The framework will raise errors for missing required parameters
3. **Device Consistency**: Ensure all components use the same device
4. **Model Compatibility**: Verify encoder/decoder model compatibility
5. **Memory Management**: Use appropriate dtype and gradient checkpointing for large models

## 🐛 Troubleshooting

### Common Errors

```python
# Missing required parameter
ValueError: "hidden_size must be specified in encoder config"
# Solution: Add hidden_size to your encoder config

# Unknown encoder type
ValueError: "Unknown encoder type: custom_encoder"
# Solution: Register your custom encoder or use supported types

# Device mismatch
RuntimeError: "Expected all tensors to be on the same device"
# Solution: Ensure all configs specify the same device
```

### Debugging Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate configuration before creating model
def validate_config(config):
    required_sections = ['domain', 'validation', 'tokenization', 'encoding', 'attention', 'decoding', 'model']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    return True

validate_config(config)
model = GenericMultiModalModel(config)
```

## 🎉 Summary

The Generic Multimodal Framework is now completely configuration-driven:

- ✅ **No hardcoded values** - everything comes from config
- ✅ **Domain-agnostic** - easily adaptable to any use case
- ✅ **Fully customizable** - every parameter is configurable
- ✅ **Extensible** - easy to add custom components
- ✅ **Production-ready** - robust error handling and validation

Start with one of the provided domain configurations and customize it for your specific needs!