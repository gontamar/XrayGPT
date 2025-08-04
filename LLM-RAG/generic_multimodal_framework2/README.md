# Generic Multimodal AI Framework

A flexible and extensible framework for building multimodal AI models that can handle various domains and data types. Built with inspiration from XrayGPT but designed to be domain-agnostic.

## 🚀 Features

- **Domain-Agnostic Design**: Easily adaptable to medical, general, or custom domains
- **Flexible Data Validation**: Domain-specific validation and conditioning
- **Configurable Tokenization**: Support for multiple tokenizer types (BERT, LLaMA, GPT-2, etc.)
- **Multiple Encoder Options**: Vision Transformers, CLIP, ResNet, domain-specific encoders
- **Advanced Attention Mechanisms**: Self-attention, cross-attention, and Q-Former style attention
- **Flexible Decoding**: Support for various language models and generation strategies
- **Easy Configuration**: YAML-based configuration system
- **Extensible Architecture**: Easy to add new domains, encoders, and attention mechanisms

## 🏗️ Architecture

The framework follows a modular pipeline:

```
Input Data → Validation → Tokenization → Encoding → Attention → Decoding → Output
```

### 1. Data Validation & Conditioning
- Domain-specific validation rules
- Data quality checks (blank images, forbidden text patterns)
- Automatic preprocessing and conditioning
- Support for medical, generic, and custom domains

### 2. Tokenization
- **Text Tokenizers**: BERT, LLaMA, GPT-2, T5, Medical-specific
- **Image Tokenizers**: Vision patches, CLIP vision
- **Multimodal Tokenizers**: Combined text-image tokenization

### 3. Encoding
- **Text Encoders**: BERT, RoBERTa, CLIP Text, Medical BERT
- **Image Encoders**: Vision Transformer, CLIP Vision, ResNet, Medical CLIP
- **Fusion Methods**: Linear alignment, MLP, attention-based fusion

### 4. Attention Mechanisms
- **Self-Attention**: Focus on relevant parts within each modality
- **Cross-Attention**: Attend between text and image features
- **Q-Former**: BLIP-2 style query-based attention

### 5. Decoding
- **LLM Decoders**: Support for various language models
- **Transformer Decoders**: Custom transformer-based decoders
- **Configurable Generation**: Temperature, top-p, beam search, etc.

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd generic_multimodal_framework

# Install dependencies
pip install torch torchvision transformers pillow pyyaml numpy
```

## 🚀 Quick Start

### Basic Usage

```python
import yaml
from generic_multimodal_framework.models.generic_multimodal_model import GenericMultiModalModel

# Load configuration
with open('configs/generic_multimodal_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = GenericMultiModalModel(config)

# Prepare your data
data = {
    "image": your_image,  # PIL Image or torch.Tensor
    "text": "Your text input",
    "query": "What do you see in this image?"
}

# Process through the model
output = model.forward(data, return_attention=True)
print(f"Response: {output.text_response}")
```

### Medical Domain Example

```python
# Configure for medical domain
config['domain']['name'] = 'medical'
config['validation']['domain'] = 'medical'
config['validation']['require_medical_content'] = True

# The framework will automatically:
# - Validate medical images (check for blank X-rays)
# - Filter forbidden text patterns
# - Apply medical-specific conditioning
# - Use appropriate tokenizers and encoders

model = GenericMultiModalModel(config)
medical_output = model.forward({
    "image": xray_image,
    "text": "Chest X-ray findings",
    "query": "What abnormalities are present?"
})
```

## 🔧 Configuration

The framework uses YAML configuration files. Key sections:

### Domain Configuration
```yaml
domain:
  name: "medical"  # medical, generic, custom
  description: "Medical imaging and text analysis"
```

### Validation Configuration
```yaml
validation:
  domain: "medical"
  min_image_size: [224, 224]
  forbidden_text_patterns: ["xxxx", "redacted"]
  require_medical_content: true
```

### Model Components
```yaml
tokenization:
  text:
    type: "bert"
    model_name: "bert-base-uncased"
  image:
    type: "vision_patch"
    patch_size: 16

encoding:
  text:
    type: "bert"
    hidden_size: 768
  image:
    type: "vision_transformer"
    hidden_size: 768

attention:
  self_attention:
    enabled: true
    num_attention_heads: 12
  cross_attention:
    enabled: true
    cross_attention_freq: 2

decoding:
  type: "llm"
  model_name: "microsoft/DialoGPT-medium"
  max_new_tokens: 256
```

## 🎯 Domain-Specific Examples

### Medical Domain (X-ray Analysis)
```python
# Medical configuration focuses on:
# - X-ray image validation
# - Medical terminology processing
# - Clinical report generation

config = {
    'domain': {'name': 'medical'},
    'validation': {
        'forbidden_text_patterns': ['xxxx', 'redacted'],
        'medical_keywords': ['chest', 'lung', 'heart', 'radiograph']
    }
}
```

### General Vision-Language
```python
# General domain for everyday images and text
config = {
    'domain': {'name': 'generic'},
    'validation': {
        'require_medical_content': False
    }
}
```

### Custom Domain
```python
# Extend for your specific domain
from generic_multimodal_framework.core.data_validator import BaseDomainValidator

class CustomDomainValidator(BaseDomainValidator):
    def validate_image(self, image):
        # Your custom image validation logic
        pass
    
    def validate_text(self, text):
        # Your custom text validation logic
        pass

# Register your validator
DataValidatorFactory.register_validator("custom", CustomDomainValidator)
```

## 🔌 Extending the Framework

### Adding New Encoders
```python
from generic_multimodal_framework.core.encoder_manager import BaseEncoder

class CustomEncoder(BaseEncoder):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your encoder
    
    def encode(self, inputs):
        # Your encoding logic
        return embeddings

# Register the encoder
EncoderManager.register_encoder("custom_encoder", CustomEncoder)
```

### Adding New Attention Mechanisms
```python
from generic_multimodal_framework.core.attention_manager import BaseAttentionModule

class CustomAttention(BaseAttentionModule):
    def forward(self, query, key, value, attention_mask=None):
        # Your attention logic
        return attention_output

# Register the attention mechanism
AttentionManager.register_attention("custom_attention", CustomAttention)
```

## 📊 Supported Models and Components

### Text Encoders
- BERT (bert-base-uncased, bert-large-uncased)
- RoBERTa (roberta-base, roberta-large)
- CLIP Text Encoder
- Medical BERT variants

### Image Encoders
- Vision Transformer (ViT-Base, ViT-Large)
- CLIP Vision Encoder
- ResNet variants
- Medical CLIP

### Language Models (Decoders)
- GPT-2 variants
- DialoGPT
- LLaMA (with proper setup)
- T5 variants
- Custom transformer models

## 🧪 Testing

Run the demonstration script:
```bash
python tmp_rovodev_demo_generic_framework.py
```

This will show the complete pipeline in action with sample data.

## 📝 Configuration Examples

### Minimal Configuration
```yaml
domain:
  name: "generic"

validation:
  domain: "generic"

tokenization:
  text:
    type: "bert"
  image:
    type: "vision_patch"

encoding:
  text:
    type: "bert"
  image:
    type: "vision_transformer"

attention:
  cross_attention:
    enabled: true

decoding:
  type: "llm"
  model_name: "microsoft/DialoGPT-small"
```

### Advanced Medical Configuration
```yaml
domain:
  name: "medical"

validation:
  domain: "medical"
  forbidden_text_patterns: ["xxxx", "redacted", "[PHI]"]
  require_medical_content: true
  convert_to_grayscale: true
  apply_histogram_equalization: true

tokenization:
  text:
    type: "medical_text"
    model_name: "emilyalsentzer/Bio_ClinicalBERT"

encoding:
  image:
    type: "medical_clip"
    model_name: "flaviagiammarino/pubmed-clip-vit-base-patch32"

attention:
  qformer:
    enabled: true
    num_query_tokens: 32

decoding:
  type: "llm"
  model_name: "microsoft/BioGPT-Large"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your domain-specific validators, encoders, or attention mechanisms
4. Update the configuration schema if needed
5. Add tests and documentation
6. Submit a pull request

## 📄 License

This project is licensed under the same terms as the original XrayGPT project.

## 🙏 Acknowledgments

- Built upon the architecture of [XrayGPT](https://github.com/mbzuai-oryx/XrayGPT)
- Inspired by [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) and [MiniGPT-4](https://minigpt-4.github.io)
- Uses components from [Transformers](https://huggingface.co/transformers/) library