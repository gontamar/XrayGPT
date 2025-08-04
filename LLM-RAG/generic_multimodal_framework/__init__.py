"""
Generic Multimodal AI Framework

A flexible framework for building multimodal AI systems that can handle various domains
and data types with configurable validation, encoding, attention, and decoding components.

Based on XrayGPT architecture but generalized for multiple domains and use cases.
"""

__version__ = "1.0.0"
__author__ = "Generic Multimodal Framework Team"

from .core.data_validator import (
    MultiModalDataValidator,
    DataValidatorFactory,
    BaseDomainValidator,
    MedicalDomainValidator,
    GenericDomainValidator,
    DataType,
    ValidationResult,
    ValidationOutput
)

from .core.tokenizer_manager import TokenizerManager
from .core.encoder_manager import EncoderManager
from .core.attention_manager import AttentionManager
from .core.decoder_manager import DecoderManager
from .models.generic_multimodal_model import GenericMultiModalModel
from .config.config_manager import ConfigManager

__all__ = [
    "MultiModalDataValidator",
    "DataValidatorFactory", 
    "BaseDomainValidator",
    "MedicalDomainValidator",
    "GenericDomainValidator",
    "DataType",
    "ValidationResult",
    "ValidationOutput",
    "TokenizerManager",
    "EncoderManager", 
    "AttentionManager",
    "DecoderManager",
    "GenericMultiModalModel",
    "ConfigManager"
]