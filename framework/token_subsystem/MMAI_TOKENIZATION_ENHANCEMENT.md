# Enhanced MMAI Tokenization Framework

## Overview

The Multi-Modal AI (MMAI) tokenization framework has been enhanced with **automatic model-based tokenizer selection** capabilities. This update allows the system to automatically detect and select the most appropriate tokenizer based on the specific model type (XrayGPT, MedCLIP, Automotive, BLIP2, etc.) and domain context being used.

## Key Features

### 1. Model-Based Automatic Selection

The framework now supports automatic tokenizer selection based on specific model types:

**Model-Specific Domains:**
- **XRAYGPT**: XrayGPT, MiniGPT4 with medical focus, Vicuna Radiology
- **MEDCLIP**: MedCLIP, Medical CLIP models for medical image-text contrastive learning
- **AUTOMOTIVE**: Tesla, Waymo, Cruise, and other autonomous driving models
- **BLIP2**: Salesforce BLIP2 for general vision-language tasks
- **MINIGPT4**: MiniGPT4 for general multimodal conversation
- **LLAVA**: LLaVA models for visual instruction following
- **FLAMINGO**: Flamingo and OpenFlamingo models

**Application Domains:**
- **MEDICAL_IMAGING**: General medical imaging applications
- **AUTONOMOUS_DRIVING**: Self-driving car applications
- **GENERAL_VISION**: General computer vision tasks
- **CONVERSATIONAL_AI**: Chat and dialogue systems

### 2. Intelligent Model Detection

The system automatically detects the appropriate model type and domain using:

- **Model path analysis**: Automatic detection from model file paths and names
- **Text analysis**: Keywords and domain-specific terminology
- **Modality information**: Model-specific and application-specific flags
- **Priority-based selection**: Higher priority tokenizers preferred for each model type

### 3. Flexible Configuration

- **Priority-based selection**: Higher priority tokenizers are preferred
- **Fallback mechanisms**: Multiple fallback options for robustness
- **Model-specific mappings**: Different tokenizers for different models
- **Manual override**: Option to disable auto-selection and use specific tokenizers

## New Classes and Enums

### DomainType Enum
```python
class DomainType(Enum):
    # Model-specific domains
    XRAYGPT = "xraygpt"
    MEDCLIP = "medclip"
    AUTOMOTIVE = "automotive"
    BLIP2 = "blip2"
    MINIGPT4 = "minigpt4"
    LLAVA = "llava"
    FLAMINGO = "flamingo"
    
    # Application domains
    MEDICAL_IMAGING = "medical_imaging"
    AUTONOMOUS_DRIVING = "autonomous_driving"
    GENERAL_VISION = "general_vision"
    CONVERSATIONAL_AI = "conversational_ai"
    
    # Legacy domains (kept for compatibility)
    MEDICAL = "medical"
    RADIOLOGY = "radiology"
    GENERAL = "general"
    VISION_LANGUAGE = "vision_language"
```

### DomainTokenizerMapping
```python
@dataclass
class DomainTokenizerMapping:
    domain: DomainType
    primary_tokenizer: str
    fallback_tokenizers: List[str]
    model_specific_tokenizers: Dict[str, str]
    context_keywords: List[str]
```

### Enhanced TokenizationConfig
```python
@dataclass
class TokenizationConfig:
    # ... existing fields ...
    domain: Optional[DomainType] = None
    priority: int = 0  # Higher priority = preferred for domain
```

## New Methods in MMAITokenizationFramework

### Automatic Selection Methods

```python
def detect_domain_from_context(self, text: str, modalities: Dict[str, Any] = None) -> DomainType:
    """Automatically detect domain from text context and modalities"""

def get_tokenizer_auto(self, text: str = None, modalities: Dict[str, Any] = None, 
                      model_name: str = None, domain: DomainType = None) -> BaseTokenizer:
    """Automatically select and return the best tokenizer based on context"""

def get_multimodal_tokenizer_auto(self, text: str = None, modalities: Dict[str, Any] = None,
                                model_name: str = None, domain: DomainType = None) -> MultiModalTokenizer:
    """Automatically select and return the best multimodal tokenizer based on context"""
```

### Configuration Methods

```python
def set_auto_selection(self, enabled: bool):
    """Enable or disable automatic tokenizer selection"""

def register_domain_mapping(self, mapping: DomainTokenizerMapping):
    """Register a custom domain mapping"""

def list_available_tokenizers_for_domain(self, domain: DomainType) -> List[str]:
    """List all available tokenizers for a specific domain"""
```

## Enhanced Model Integration

### Updated MiniGPT4 Methods

The `mini_gpt4.py` model now includes enhanced methods:

```python
def encode_with_framework(self, text: str, modalities: dict = None, 
                         tokenizer_name: str = 'xraygpt_multimodal', 
                         auto_select: bool = True, domain: DomainType = None):
    """Encode text using MMAI tokenization framework with automatic selection"""

def encode_radiology_prompt(self, text: str, modalities: dict = None):
    """Specialized encoding for radiology prompts"""

def encode_medical_prompt(self, text: str, modalities: dict = None):
    """Specialized encoding for general medical prompts"""

def get_available_tokenizers_for_context(self, text: str = None, modalities: dict = None) -> dict:
    """Get information about available tokenizers for the given context"""
```

## Usage Examples

### Automatic Model Detection and Setup

```python
from xraygpt.tokenization import setup_model_tokenization, DomainType

# Automatic model type detection from paths
framework = setup_model_tokenization(
    auto_detect=True,
    llama_model_path="./models/xraygpt_vicuna_7b",
    bert_model_path="./models/bert_medical"
)

# Manual model type specification
framework = setup_model_tokenization(
    model_type="medclip",
    model_path="./models/medclip_vit_bert"
)

# Automotive model setup
framework = setup_model_tokenization(
    model_type="automotive", 
    model_path="./models/tesla_autopilot_v3"
)
```

### Automatic Tokenizer Selection

```python
# Automatic selection based on model path
text = "Analyze this autonomous driving scenario"
tokenizer = framework.get_tokenizer_auto(
    text=text,
    model_name="./models/tesla_autopilot_v3"
)

# With modalities for specific model types
modalities = {'automotive': True, 'navigation': True}
multimodal_tokenizer = framework.get_multimodal_tokenizer_auto(
    text=text, 
    modalities=modalities,
    model_name="tesla_model"
)
```

### Manual Domain Specification

```python
# Force specific domain
radiology_tokenizer = framework.get_tokenizer_auto(
    text="X-ray analysis",
    domain=DomainType.RADIOLOGY
)

medical_tokenizer = framework.get_tokenizer_auto(
    text="Patient consultation", 
    domain=DomainType.MEDICAL
)
```

### Model Integration

```python
# In your model class
def process_radiology_input(self, text, image_features):
    # Automatic radiology-specific tokenization
    encoded = self.encode_radiology_prompt(text, {'image': True, 'radiology': True})
    
    # Process with model...
    return self.generate(encoded, image_features)

def process_medical_consultation(self, text):
    # Automatic medical-specific tokenization  
    encoded = self.encode_medical_prompt(text, {'medical': True})
    
    # Process with model...
    return self.generate(encoded)
```

### Configuration and Inspection

```python
# Check what tokenizers are available for a domain
available = framework.list_available_tokenizers_for_domain(DomainType.RADIOLOGY)
print(f"Available radiology tokenizers: {available}")

# Get context information
context_info = model.get_available_tokenizers_for_context(
    text="Chest X-ray analysis", 
    modalities={'image': True, 'radiology': True}
)
print(f"Detected domain: {context_info['detected_domain']}")
print(f"Optimal tokenizer: {context_info['optimal_tokenizer']}")

# Toggle automatic selection
framework.set_auto_selection(False)  # Disable auto-selection
framework.set_auto_selection(True)   # Re-enable auto-selection
```

## Default Domain Mappings

The framework comes with pre-configured domain mappings:

### Radiology Domain
- **Primary**: `radiology_llama` (if available)
- **Fallbacks**: `xraygpt_llama`, `xraygpt_bert`
- **Keywords**: radiology, x-ray, chest, lung, radiograph, imaging, scan
- **Model mappings**: Vicuna_Radiology → radiology_llama

### Medical Domain  
- **Primary**: `xraygpt_llama`
- **Fallbacks**: `radiology_llama`, `xraygpt_bert`
- **Keywords**: medical, patient, diagnosis, treatment, clinical, healthcare

### Vision-Language Domain
- **Primary**: `xraygpt_bert`
- **Fallbacks**: `xraygpt_llama`
- **Keywords**: image, visual, caption, description, multimodal

## Migration Guide

### For Existing Code

Existing code will continue to work without changes. The new automatic selection is opt-in:

```python
# Old way (still works)
tokenizer = framework.get_tokenizer('xraygpt_llama')
encoded = tokenizer.encode(text)

# New way (automatic selection)
tokenizer = framework.get_tokenizer_auto(text=text)
encoded = tokenizer.encode(text)

# Model integration (new methods)
encoded = model.encode_with_framework(text, auto_select=True)  # New automatic
encoded = model.encode_with_framework(text, auto_select=False) # Legacy behavior
```

### For Model Classes

Models can gradually adopt the new methods:

```python
# Add to your model's __init__
self.tokenization_framework = setup_xraygpt_tokenization(llama_model_path, bert_model_path)

# Use new specialized methods
def process_radiology_text(self, text):
    return self.encode_radiology_prompt(text)

def process_medical_text(self, text):  
    return self.encode_medical_prompt(text)
```

## Benefits

1. **Automatic Optimization**: Best tokenizer selected automatically based on context
2. **Domain Specialization**: Specialized tokenizers for medical subdomains
3. **Backward Compatibility**: Existing code continues to work unchanged
4. **Flexibility**: Manual override options when needed
5. **Extensibility**: Easy to add new domains and tokenizers
6. **Robustness**: Multiple fallback mechanisms prevent failures

## Future Enhancements

- **Dynamic tokenizer loading**: Load tokenizers on-demand based on context
- **Performance optimization**: Cache frequently used tokenizers
- **Advanced domain detection**: ML-based domain classification
- **Custom domain registration**: Runtime domain and tokenizer registration
- **Metrics and monitoring**: Track tokenizer selection performance