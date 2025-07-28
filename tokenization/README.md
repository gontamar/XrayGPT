# Generic Tokenizer Manager

A flexible and configurable tokenization system that supports multiple tokenizer types through YAML configuration. This implementation is based on the Transformers library and provides a unified interface for different tokenization needs.

## Features

- **Configuration-driven**: All tokenizer settings are managed through a YAML configuration file
- **Multiple tokenizer support**: BERT, RoBERTa, GPT-2, LLaMA, Vicuna, and custom tokenizers
- **XrayGPT integration**: Specific configurations for XrayGPT use cases
- **Batch processing**: Support for single text and batch tokenization
- **Special tokens**: Easy configuration of special tokens for different models
- **Extensible**: Easy to add new tokenizer configurations

## Quick Start

### Basic Usage

```python
from tokenizer_manager import TokenizerManager

# Create manager and load default tokenizer
manager = TokenizerManager()
manager.load_tokenizer('bert')

# Tokenize text
text = "This is a sample radiology report for testing."
result = manager.tokenize(text)

print("Tokens:", result['tokens'])
print("Token IDs:", result['token_ids'])
```

### XrayGPT Usage

```python
# Use BLIP2 BERT configuration for XrayGPT
manager.load_tokenizer('blip2_bert')
result = manager.tokenize(radiology_report)
```

### Quick Tokenization

```python
from tokenizer_manager import quick_tokenize

result = quick_tokenize("Sample text", tokenizer_name='bert')
```

## Configuration

The system uses a YAML configuration file (`tokenizer_config.yaml`) to define tokenizer settings:

```yaml
tokenizers:
  bert:
    model_name: "bert-base-uncased"
    special_tokens:
      bos_token: "[DEC]"
    max_length: 512
    padding: true
    truncation: true
```

### Available Configurations

- **bert**: BERT base uncased
- **bert_large**: BERT large uncased
- **llama**: LLaMA 2 7B
- **vicuna**: Vicuna 7B v1.5
- **gpt2**: GPT-2
- **roberta**: RoBERTa base

### XrayGPT Specific

- **blip2_bert**: BLIP2 BERT configuration
- **vicuna_llm**: Vicuna LLM configuration

## API Reference

### TokenizerManager

#### Methods

- `load_tokenizer(tokenizer_name, custom_config)`: Load a tokenizer
- `tokenize(text, **kwargs)`: Tokenize input text
- `decode(token_ids)`: Decode token IDs to text
- `list_available_tokenizers()`: List available configurations
- `get_tokenizer_info(name)`: Get tokenizer configuration details

#### Properties

- `tokenizer`: Current loaded tokenizer
- `current_config`: Current tokenizer configuration
- `config`: Full configuration dictionary

## Examples

### Batch Processing

```python
texts = [
    "First radiology report",
    "Second medical document",
    "Third sample text"
]

result = manager.tokenize(texts)
```

### Custom Configuration

```python
custom_config = {
    'model_name': 'bert-base-uncased',
    'special_tokens': {'bos_token': '[CUSTOM]'},
    'max_length': 256
}

manager.load_tokenizer(custom_config=custom_config)
```

### Adding Custom Tokenizers

Edit `tokenizer_config.yaml`:

```yaml
custom_paths:
  my_custom_tokenizer: "/path/to/custom/tokenizer"
```

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- transformers>=4.20.0
- torch>=1.12.0
- PyYAML>=6.0
- numpy>=1.21.0

## Error Handling

The system includes comprehensive error handling and logging:

- Configuration file validation
- Tokenizer loading errors
- Tokenization errors
- Missing tokenizer configurations

## Extending the System

### Adding New Tokenizers

1. Add configuration to `tokenizer_config.yaml`
2. Specify model name and parameters
3. Add any required special tokens

### Custom Preprocessing

Extend the `TokenizerManager` class to add custom preprocessing:

```python
class CustomTokenizerManager(TokenizerManager):
    def preprocess_text(self, text):
        # Custom preprocessing logic
        return processed_text
```

## License

This implementation is designed to be flexible and extensible for various NLP and multimodal applications.