"""
Tokenization Framework for Multi-Modal AI (MMAI) - XrayGPT
Restored and integrated from temporary framework
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from transformers import (
    AutoTokenizer, 
    LlamaTokenizer, 
    BertTokenizer,
    GPT2Tokenizer,
    T5Tokenizer,
    RobertaTokenizer,
    DebertaTokenizer,
    PreTrainedTokenizer
)
import logging
from enum import Enum
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerType(Enum):
    """Enumeration of supported tokenizer types"""
    LLAMA = "llama"
    BERT = "bert"
    GPT2 = "gpt2"
    T5 = "t5"
    ROBERTA = "roberta"
    DEBERTA = "deberta"
    AUTO = "auto"

class DomainType(Enum):
    """Enumeration of supported domain types for automatic tokenizer selection"""
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
    DOCUMENT_AI = "document_ai"
    
    # Traditional domains (kept for compatibility)
    MEDICAL = "medical"
    RADIOLOGY = "radiology"
    GENERAL = "general"
    VISION_LANGUAGE = "vision_language"

@dataclass
class TokenizationConfig:
    """Configuration for tokenization parameters"""
    tokenizer_type: TokenizerType
    model_name_or_path: str
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    add_special_tokens: bool = True
    return_tensors: str = "pt"
    pad_token: Optional[str] = None
    eos_token: Optional[str] = None
    bos_token: Optional[str] = None
    unk_token: Optional[str] = None
    domain: Optional[DomainType] = None
    priority: int = 0  # Higher priority tokenizers are preferred for domain
    
@dataclass
class DomainTokenizerMapping:
    """Mapping configuration for domain-based tokenizer selection"""
    domain: DomainType
    primary_tokenizer: str
    fallback_tokenizers: List[str]
    model_specific_tokenizers: Dict[str, str]  # model_name -> tokenizer_name
    context_keywords: List[str]  # Keywords that indicate this domain

class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers in the framework"""
    
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.tokenizer = self._initialize_tokenizer()
        self._setup_special_tokens()
    
    @abstractmethod
    def _initialize_tokenizer(self) -> PreTrainedTokenizer:
        """Initialize the underlying tokenizer"""
        pass
    
    def _setup_special_tokens(self):
        """Setup special tokens if provided in config"""
        if self.config.pad_token:
            self.tokenizer.pad_token = self.config.pad_token
        if self.config.eos_token:
            self.tokenizer.eos_token = self.config.eos_token
        if self.config.bos_token:
            self.tokenizer.bos_token = self.config.bos_token
        if self.config.unk_token:
            self.tokenizer.unk_token = self.config.unk_token
    
    def encode(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        """Encode text to tokens"""
        encoding_kwargs = {
            'padding': self.config.padding,
            'truncation': self.config.truncation,
            'max_length': self.config.max_length,
            'return_tensors': self.config.return_tensors,
            'add_special_tokens': self.config.add_special_tokens,
            **kwargs
        }
        return self.tokenizer(text, **encoding_kwargs)
    
    def decode(self, token_ids: torch.Tensor, **kwargs) -> str:
        """Decode tokens to text"""
        decode_kwargs = {
            'skip_special_tokens': True,
            'clean_up_tokenization_spaces': True,
            **kwargs
        }
        return self.tokenizer.decode(token_ids, **decode_kwargs)
    
    def batch_decode(self, token_ids: torch.Tensor, **kwargs) -> List[str]:
        """Batch decode tokens to text"""
        decode_kwargs = {
            'skip_special_tokens': True,
            'clean_up_tokenization_spaces': True,
            **kwargs
        }
        return self.tokenizer.batch_decode(token_ids, **decode_kwargs)

class LlamaTokenizerWrapper(BaseTokenizer):
    """Wrapper for LLaMA tokenizer - as used in XrayGPT"""
    
    def _initialize_tokenizer(self) -> LlamaTokenizer:
        logger.info(f"Initializing LLaMA tokenizer from {self.config.model_name_or_path}")
        tokenizer = LlamaTokenizer.from_pretrained(
            self.config.model_name_or_path, 
            use_fast=False
        )
        # Set pad token to eos token as in XrayGPT
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

class BertTokenizerWrapper(BaseTokenizer):
    """Wrapper for BERT tokenizer - as used in XrayGPT's Q-Former"""
    
    def _initialize_tokenizer(self) -> BertTokenizer:
        logger.info(f"Initializing BERT tokenizer from {self.config.model_name_or_path}")
        tokenizer = BertTokenizer.from_pretrained(self.config.model_name_or_path)
        # Add special tokens as in XrayGPT
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

class GPT2TokenizerWrapper(BaseTokenizer):
    """Wrapper for GPT-2 tokenizer"""
    
    def _initialize_tokenizer(self) -> GPT2Tokenizer:
        logger.info(f"Initializing GPT-2 tokenizer from {self.config.model_name_or_path}")
        tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

class T5TokenizerWrapper(BaseTokenizer):
    """Wrapper for T5 tokenizer"""
    
    def _initialize_tokenizer(self) -> T5Tokenizer:
        logger.info(f"Initializing T5 tokenizer from {self.config.model_name_or_path}")
        return T5Tokenizer.from_pretrained(self.config.model_name_or_path)

class RobertaTokenizerWrapper(BaseTokenizer):
    """Wrapper for RoBERTa tokenizer"""
    
    def _initialize_tokenizer(self) -> RobertaTokenizer:
        logger.info(f"Initializing RoBERTa tokenizer from {self.config.model_name_or_path}")
        return RobertaTokenizer.from_pretrained(self.config.model_name_or_path)

class DebertaTokenizerWrapper(BaseTokenizer):
    """Wrapper for DeBERTa tokenizer"""
    
    def _initialize_tokenizer(self) -> DebertaTokenizer:
        logger.info(f"Initializing DeBERTa tokenizer from {self.config.model_name_or_path}")
        return DebertaTokenizer.from_pretrained(self.config.model_name_or_path)

class AutoTokenizerWrapper(BaseTokenizer):
    """Wrapper for AutoTokenizer - automatically detects tokenizer type"""
    
    def _initialize_tokenizer(self) -> PreTrainedTokenizer:
        logger.info(f"Initializing Auto tokenizer from {self.config.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            use_fast=False
        )
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

class TokenizerFactory:
    """Factory class for creating tokenizers"""
    
    _tokenizer_classes = {
        TokenizerType.LLAMA: LlamaTokenizerWrapper,
        TokenizerType.BERT: BertTokenizerWrapper,
        TokenizerType.GPT2: GPT2TokenizerWrapper,
        TokenizerType.T5: T5TokenizerWrapper,
        TokenizerType.ROBERTA: RobertaTokenizerWrapper,
        TokenizerType.DEBERTA: DebertaTokenizerWrapper,
        TokenizerType.AUTO: AutoTokenizerWrapper,
    }
    
    @classmethod
    def create_tokenizer(cls, config: TokenizationConfig) -> BaseTokenizer:
        """Create a tokenizer based on the configuration"""
        tokenizer_class = cls._tokenizer_classes.get(config.tokenizer_type)
        if tokenizer_class is None:
            raise ValueError(f"Unsupported tokenizer type: {config.tokenizer_type}")
        
        return tokenizer_class(config)

class MultiModalTokenizer:
    """Multi-modal tokenizer for handling different modalities"""
    
    def __init__(self, text_tokenizer: BaseTokenizer, modality_configs: Dict[str, Any]):
        self.text_tokenizer = text_tokenizer
        self.modality_configs = modality_configs
        self.image_placeholder = modality_configs.get('image_placeholder', '<ImageHere>')
        self.medical_tokens = modality_configs.get('medical_tokens', {})
    
    def process_multimodal_input(self, text: str, modalities: Dict[str, Any]) -> str:
        """Process multi-modal input by inserting placeholders"""
        processed_text = text
        
        # Handle image modality
        if 'image' in modalities:
            # Insert image placeholder if not already present
            if self.image_placeholder not in processed_text:
                processed_text = f"<Img>{self.image_placeholder}</Img> {processed_text}"
        
        # Handle medical context tokens
        for token_type, token_value in self.medical_tokens.items():
            if token_type in modalities:
                processed_text = f"{token_value} {processed_text}"
        
        return processed_text
    
    def encode_multimodal(self, text: str, modalities: Dict[str, Any] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Encode multi-modal input"""
        if modalities is None:
            modalities = {}
        
        processed_text = self.process_multimodal_input(text, modalities)
        encoding = self.text_tokenizer.encode(processed_text)
        
        # Add modality information to encoding
        encoding['modalities'] = modalities
        encoding['processed_text'] = processed_text
        
        return encoding

class MMAITokenizationFramework:
    """Main framework class for MMAI tokenization"""
    
    def __init__(self):
        self.tokenizers = {}
        self.multimodal_tokenizers = {}
        self.default_configs = self._get_default_configs()
        self.domain_mappings = self._get_default_domain_mappings()
        self.auto_selection_enabled = True
    
    def register_tokenizer(self, name: str, config: TokenizationConfig):
        """Register a new tokenizer"""
        tokenizer = TokenizerFactory.create_tokenizer(config)
        self.tokenizers[name] = tokenizer
        logger.info(f"Registered tokenizer '{name}' of type {config.tokenizer_type}")
    
    def register_multimodal_tokenizer(self, name: str, text_tokenizer_name: str, 
                                    modality_configs: Dict[str, Any]):
        """Register a multi-modal tokenizer"""
        if text_tokenizer_name not in self.tokenizers:
            raise ValueError(f"Text tokenizer '{text_tokenizer_name}' not found")
        
        text_tokenizer = self.tokenizers[text_tokenizer_name]
        multimodal_tokenizer = MultiModalTokenizer(text_tokenizer, modality_configs)
        self.multimodal_tokenizers[name] = multimodal_tokenizer
        logger.info(f"Registered multi-modal tokenizer '{name}'")
    
    def get_tokenizer(self, name: str) -> BaseTokenizer:
        """Get a registered tokenizer"""
        if name not in self.tokenizers:
            raise ValueError(f"Tokenizer '{name}' not found")
        return self.tokenizers[name]
    
    def get_multimodal_tokenizer(self, name: str) -> MultiModalTokenizer:
        """Get a registered multi-modal tokenizer"""
        if name not in self.multimodal_tokenizers:
            raise ValueError(f"Multi-modal tokenizer '{name}' not found")
        return self.multimodal_tokenizers[name]
    
    def detect_model_type_from_path(self, model_path: str) -> Optional[DomainType]:
        """Detect model type from model path or name"""
        if not model_path:
            return None
            
        model_path_lower = model_path.lower()
        
        # Model-specific detection
        model_indicators = {
            DomainType.XRAYGPT: ["xraygpt", "mini_gpt4", "vicuna_radiology", "xray_gpt"],
            DomainType.MEDCLIP: ["medclip", "medical_clip", "med_clip"],
            DomainType.AUTOMOTIVE: ["automotive", "tesla", "waymo", "cruise", "driving", "autonomous"],
            DomainType.BLIP2: ["blip2", "blip-2", "salesforce/blip2"],
            DomainType.MINIGPT4: ["minigpt4", "mini_gpt4", "mini-gpt4"],
            DomainType.LLAVA: ["llava", "llava_v1", "llava-v1.5"],
            DomainType.FLAMINGO: ["flamingo", "openflamingo"]
        }
        
        for domain, indicators in model_indicators.items():
            if any(indicator in model_path_lower for indicator in indicators):
                return domain
        
        return None
    
    def detect_domain_from_context(self, text: str = None, modalities: Dict[str, Any] = None, 
                                 model_path: str = None) -> DomainType:
        """Automatically detect domain from text context, modalities, and model path"""
        
        # First, try to detect from model path (highest priority)
        if model_path:
            model_domain = self.detect_model_type_from_path(model_path)
            if model_domain:
                logger.info(f"Detected domain {model_domain.value} from model path: {model_path}")
                return model_domain
        
        # Check modalities for specific model/domain indicators
        if modalities:
            # Model-specific modality indicators
            if 'xraygpt' in modalities or 'mini_gpt4' in modalities:
                return DomainType.XRAYGPT
            if 'medclip' in modalities or 'medical_clip' in modalities:
                return DomainType.MEDCLIP
            if 'automotive' in modalities or 'driving' in modalities:
                return DomainType.AUTOMOTIVE
            if 'blip2' in modalities:
                return DomainType.BLIP2
            if 'llava' in modalities:
                return DomainType.LLAVA
                
            # Application-specific modality indicators
            if 'radiology' in modalities or 'x-ray' in modalities:
                return DomainType.MEDICAL_IMAGING
            if 'autonomous_driving' in modalities or 'navigation' in modalities:
                return DomainType.AUTONOMOUS_DRIVING
            if 'image' in modalities or 'visual' in modalities:
                return DomainType.GENERAL_VISION
        
        # Check text content for domain keywords if text is provided
        if text:
            text_lower = text.lower()
            domain_scores = {}
            
            for domain, mapping in self.domain_mappings.items():
                score = sum(1 for keyword in mapping.context_keywords if keyword in text_lower)
                if score > 0:
                    domain_scores[domain] = score
            
            # Return domain with highest score
            if domain_scores:
                detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
                logger.info(f"Detected domain {detected_domain.value} from text context")
                return detected_domain
        
        # Default fallback - try to infer from available context
        if modalities and ('medical' in modalities or 'radiology' in modalities):
            return DomainType.XRAYGPT  # Default medical model
        
        # Final fallback
        logger.info("Using GENERAL domain as fallback")
        return DomainType.GENERAL
    
    def get_optimal_tokenizer_for_domain(self, domain: DomainType, model_name: str = None) -> str:
        """Get the optimal tokenizer name for a given domain and optional model"""
        if domain not in self.domain_mappings:
            logger.warning(f"Domain {domain} not found in mappings, using MEDICAL default")
            domain = DomainType.MEDICAL
        
        mapping = self.domain_mappings[domain]
        
        # Check for model-specific tokenizer first
        if model_name:
            for model_key, tokenizer_name in mapping.model_specific_tokenizers.items():
                if model_key.lower() in model_name.lower():
                    if tokenizer_name in self.tokenizers:
                        return tokenizer_name
        
        # Try primary tokenizer
        if mapping.primary_tokenizer in self.tokenizers:
            return mapping.primary_tokenizer
        
        # Try fallback tokenizers
        for fallback in mapping.fallback_tokenizers:
            if fallback in self.tokenizers:
                return fallback
        
        # Final fallback to any available tokenizer
        if self.tokenizers:
            return list(self.tokenizers.keys())[0]
        
        raise ValueError(f"No suitable tokenizer found for domain {domain}")
    
    def get_tokenizer_auto(self, text: str = None, modalities: Dict[str, Any] = None, 
                          model_name: str = None, domain: DomainType = None) -> BaseTokenizer:
        """Automatically select and return the best tokenizer based on context"""
        if not self.auto_selection_enabled:
            # Fallback to default tokenizer
            return self.get_tokenizer('xraygpt_llama')
        
        # Use provided domain or detect from context
        if domain is None:
            domain = self.detect_domain_from_context(text, modalities, model_name)
        
        # Get optimal tokenizer name
        tokenizer_name = self.get_optimal_tokenizer_for_domain(domain, model_name)
        
        logger.info(f"Auto-selected tokenizer '{tokenizer_name}' for domain {domain.value}")
        return self.get_tokenizer(tokenizer_name)
    
    def get_multimodal_tokenizer_auto(self, text: str = None, modalities: Dict[str, Any] = None,
                                    model_name: str = None, domain: DomainType = None) -> MultiModalTokenizer:
        """Automatically select and return the best multimodal tokenizer based on context"""
        
        # Use provided domain or detect from context
        if domain is None:
            domain = self.detect_domain_from_context(text, modalities, model_name)
        
        # Try to get domain-specific multimodal tokenizer first
        domain_multimodal_name = f"{domain.value}_multimodal"
        if domain_multimodal_name in self.multimodal_tokenizers:
            logger.info(f"Using domain-specific multimodal tokenizer: {domain_multimodal_name}")
            return self.get_multimodal_tokenizer(domain_multimodal_name)
        
        # Fallback to default multimodal tokenizer
        if 'xraygpt_multimodal' in self.multimodal_tokenizers:
            return self.get_multimodal_tokenizer('xraygpt_multimodal')
        
        # If no multimodal tokenizer exists, create one on the fly based on domain
        base_tokenizer = self.get_tokenizer_auto(text, modalities, model_name, domain)
        
        # Create domain-specific modality configs
        modality_configs = self._get_domain_specific_modality_config(domain)
        
        logger.info(f"Creating on-the-fly multimodal tokenizer for domain {domain.value}")
        return MultiModalTokenizer(base_tokenizer, modality_configs)
    
    def _get_domain_specific_modality_config(self, domain: DomainType) -> Dict[str, Any]:
        """Get domain-specific modality configuration"""
        base_config = {
            'image_placeholder': '<ImageHere>',
        }
        
        if domain in [DomainType.XRAYGPT, DomainType.MEDICAL_IMAGING, DomainType.MEDCLIP]:
            base_config['medical_tokens'] = {
                'radiology': '[RADIOLOGY]',
                'finding': '[FINDING]',
                'impression': '[IMPRESSION]',
                'medical': '[MEDICAL]'
            }
        elif domain == DomainType.AUTOMOTIVE:
            base_config['automotive_tokens'] = {
                'vehicle': '[VEHICLE]',
                'traffic': '[TRAFFIC]',
                'road': '[ROAD]',
                'navigation': '[NAV]'
            }
        elif domain in [DomainType.BLIP2, DomainType.MINIGPT4, DomainType.LLAVA]:
            base_config['vision_tokens'] = {
                'caption': '[CAPTION]',
                'description': '[DESC]',
                'visual': '[VISUAL]'
            }
        else:
            base_config['general_tokens'] = {
                'context': '[CONTEXT]',
                'task': '[TASK]'
            }
        
        return base_config
    
    def set_auto_selection(self, enabled: bool):
        """Enable or disable automatic tokenizer selection"""
        self.auto_selection_enabled = enabled
        logger.info(f"Automatic tokenizer selection {'enabled' if enabled else 'disabled'}")
    
    def register_domain_mapping(self, mapping: DomainTokenizerMapping):
        """Register a custom domain mapping"""
        self.domain_mappings[mapping.domain] = mapping
        logger.info(f"Registered domain mapping for {mapping.domain}")
    
    def list_available_tokenizers_for_domain(self, domain: DomainType) -> List[str]:
        """List all available tokenizers for a specific domain"""
        if domain not in self.domain_mappings:
            return []
        
        mapping = self.domain_mappings[domain]
        available = []
        
        # Check primary tokenizer
        if mapping.primary_tokenizer in self.tokenizers:
            available.append(mapping.primary_tokenizer)
        
        # Check fallback tokenizers
        for fallback in mapping.fallback_tokenizers:
            if fallback in self.tokenizers and fallback not in available:
                available.append(fallback)
        
        return available
    
    def _get_default_configs(self) -> Dict[str, TokenizationConfig]:
        """Get default configurations for common tokenizers"""
        return {
            # XrayGPT tokenizers
            'xraygpt_llama': TokenizationConfig(
                tokenizer_type=TokenizerType.LLAMA,
                model_name_or_path="decapoda-research/llama-7b-hf",
                max_length=512,
                domain=DomainType.XRAYGPT,
                priority=15
            ),
            'xraygpt_bert': TokenizationConfig(
                tokenizer_type=TokenizerType.BERT,
                model_name_or_path="bert-base-uncased",
                max_length=512,
                domain=DomainType.XRAYGPT,
                priority=12
            ),
            
            # MedCLIP tokenizers
            'medclip_tokenizer': TokenizationConfig(
                tokenizer_type=TokenizerType.BERT,
                model_name_or_path="bert-base-uncased",
                max_length=512,
                domain=DomainType.MEDCLIP,
                priority=14
            ),
            
            # Automotive tokenizers
            'automotive_tokenizer': TokenizationConfig(
                tokenizer_type=TokenizerType.AUTO,
                model_name_or_path="microsoft/DialoGPT-medium",
                max_length=512,
                domain=DomainType.AUTOMOTIVE,
                priority=13
            ),
            
            # BLIP2 tokenizers
            'blip2_tokenizer': TokenizationConfig(
                tokenizer_type=TokenizerType.BERT,
                model_name_or_path="bert-base-uncased",
                max_length=512,
                domain=DomainType.BLIP2,
                priority=12
            ),
            
            # MiniGPT4 tokenizers
            'minigpt4_tokenizer': TokenizationConfig(
                tokenizer_type=TokenizerType.LLAMA,
                model_name_or_path="decapoda-research/llama-7b-hf",
                max_length=512,
                domain=DomainType.MINIGPT4,
                priority=13
            ),
            
            # LLaVA tokenizers
            'llava_tokenizer': TokenizationConfig(
                tokenizer_type=TokenizerType.LLAMA,
                model_name_or_path="decapoda-research/llama-7b-hf",
                max_length=512,
                domain=DomainType.LLAVA,
                priority=13
            ),
            
            # General purpose
            'general_auto': TokenizationConfig(
                tokenizer_type=TokenizerType.AUTO,
                model_name_or_path="microsoft/DialoGPT-medium",
                max_length=512,
                domain=DomainType.GENERAL,
                priority=5
            )
        }
    
    def _get_default_domain_mappings(self) -> Dict[DomainType, DomainTokenizerMapping]:
        """Get default domain-to-tokenizer mappings"""
        return {
            # Model-specific domains
            DomainType.XRAYGPT: DomainTokenizerMapping(
                domain=DomainType.XRAYGPT,
                primary_tokenizer="xraygpt_llama",
                fallback_tokenizers=["xraygpt_bert", "general_auto"],
                model_specific_tokenizers={
                    "xraygpt": "xraygpt_llama",
                    "mini_gpt4": "xraygpt_llama",
                    "vicuna": "xraygpt_llama",
                    "llama": "xraygpt_llama"
                },
                context_keywords=["xraygpt", "medical", "radiology", "x-ray", "chest", "diagnosis"]
            ),
            DomainType.MEDCLIP: DomainTokenizerMapping(
                domain=DomainType.MEDCLIP,
                primary_tokenizer="medclip_tokenizer",
                fallback_tokenizers=["xraygpt_bert", "xraygpt_llama"],
                model_specific_tokenizers={
                    "medclip": "medclip_tokenizer",
                    "clip": "medclip_tokenizer",
                    "medical_clip": "medclip_tokenizer"
                },
                context_keywords=["medclip", "medical", "clip", "contrastive", "medical_imaging"]
            ),
            DomainType.AUTOMOTIVE: DomainTokenizerMapping(
                domain=DomainType.AUTOMOTIVE,
                primary_tokenizer="automotive_tokenizer",
                fallback_tokenizers=["general_auto", "xraygpt_bert"],
                model_specific_tokenizers={
                    "automotive": "automotive_tokenizer",
                    "driving": "automotive_tokenizer",
                    "tesla": "automotive_tokenizer",
                    "waymo": "automotive_tokenizer"
                },
                context_keywords=["automotive", "driving", "car", "vehicle", "traffic", "road", "autonomous"]
            ),
            DomainType.BLIP2: DomainTokenizerMapping(
                domain=DomainType.BLIP2,
                primary_tokenizer="blip2_tokenizer",
                fallback_tokenizers=["xraygpt_bert", "general_auto"],
                model_specific_tokenizers={
                    "blip2": "blip2_tokenizer",
                    "blip": "blip2_tokenizer",
                    "salesforce": "blip2_tokenizer"
                },
                context_keywords=["blip2", "blip", "salesforce", "vision", "language", "captioning"]
            ),
            DomainType.MINIGPT4: DomainTokenizerMapping(
                domain=DomainType.MINIGPT4,
                primary_tokenizer="minigpt4_tokenizer",
                fallback_tokenizers=["xraygpt_llama", "general_auto"],
                model_specific_tokenizers={
                    "minigpt4": "minigpt4_tokenizer",
                    "mini_gpt4": "minigpt4_tokenizer",
                    "gpt4": "minigpt4_tokenizer"
                },
                context_keywords=["minigpt4", "mini_gpt4", "gpt4", "multimodal", "conversation"]
            ),
            DomainType.LLAVA: DomainTokenizerMapping(
                domain=DomainType.LLAVA,
                primary_tokenizer="llava_tokenizer",
                fallback_tokenizers=["xraygpt_llama", "general_auto"],
                model_specific_tokenizers={
                    "llava": "llava_tokenizer",
                    "llava_v1": "llava_tokenizer",
                    "llava_v1.5": "llava_tokenizer"
                },
                context_keywords=["llava", "visual", "instruction", "following", "multimodal"]
            ),
            
            # Application domains
            DomainType.MEDICAL_IMAGING: DomainTokenizerMapping(
                domain=DomainType.MEDICAL_IMAGING,
                primary_tokenizer="xraygpt_llama",
                fallback_tokenizers=["medclip_tokenizer", "xraygpt_bert"],
                model_specific_tokenizers={
                    "medical": "xraygpt_llama",
                    "radiology": "xraygpt_llama",
                    "pathology": "xraygpt_llama"
                },
                context_keywords=["medical", "radiology", "pathology", "x-ray", "ct", "mri", "ultrasound"]
            ),
            DomainType.AUTONOMOUS_DRIVING: DomainTokenizerMapping(
                domain=DomainType.AUTONOMOUS_DRIVING,
                primary_tokenizer="automotive_tokenizer",
                fallback_tokenizers=["general_auto", "xraygpt_bert"],
                model_specific_tokenizers={
                    "tesla": "automotive_tokenizer",
                    "waymo": "automotive_tokenizer",
                    "cruise": "automotive_tokenizer"
                },
                context_keywords=["autonomous", "driving", "self-driving", "navigation", "traffic", "road"]
            ),
            DomainType.GENERAL_VISION: DomainTokenizerMapping(
                domain=DomainType.GENERAL_VISION,
                primary_tokenizer="general_auto",
                fallback_tokenizers=["xraygpt_bert", "xraygpt_llama"],
                model_specific_tokenizers={
                    "clip": "general_auto",
                    "vit": "general_auto",
                    "dino": "general_auto"
                },
                context_keywords=["vision", "image", "visual", "object", "detection", "classification"]
            ),
            DomainType.CONVERSATIONAL_AI: DomainTokenizerMapping(
                domain=DomainType.CONVERSATIONAL_AI,
                primary_tokenizer="general_auto",
                fallback_tokenizers=["xraygpt_llama", "xraygpt_bert"],
                model_specific_tokenizers={
                    "gpt": "general_auto",
                    "chatgpt": "general_auto",
                    "claude": "general_auto",
                    "bard": "general_auto"
                },
                context_keywords=["conversation", "chat", "dialogue", "assistant", "qa", "question"]
            ),
            
            # Legacy domains (kept for compatibility)
            DomainType.RADIOLOGY: DomainTokenizerMapping(
                domain=DomainType.RADIOLOGY,
                primary_tokenizer="xraygpt_llama",
                fallback_tokenizers=["xraygpt_bert", "general_auto"],
                model_specific_tokenizers={
                    "Vicuna_Radiology": "xraygpt_llama",
                    "llama": "xraygpt_llama",
                    "bert": "xraygpt_bert"
                },
                context_keywords=["radiology", "x-ray", "chest", "lung", "radiograph", "imaging", "scan"]
            ),
            DomainType.MEDICAL: DomainTokenizerMapping(
                domain=DomainType.MEDICAL,
                primary_tokenizer="xraygpt_llama",
                fallback_tokenizers=["xraygpt_bert", "general_auto"],
                model_specific_tokenizers={
                    "llama": "xraygpt_llama",
                    "bert": "xraygpt_bert"
                },
                context_keywords=["medical", "patient", "diagnosis", "treatment", "clinical", "healthcare"]
            ),
            DomainType.VISION_LANGUAGE: DomainTokenizerMapping(
                domain=DomainType.VISION_LANGUAGE,
                primary_tokenizer="xraygpt_bert",
                fallback_tokenizers=["xraygpt_llama", "general_auto"],
                model_specific_tokenizers={
                    "bert": "xraygpt_bert",
                    "qformer": "xraygpt_bert"
                },
                context_keywords=["image", "visual", "caption", "description", "multimodal"]
            ),
            DomainType.GENERAL: DomainTokenizerMapping(
                domain=DomainType.GENERAL,
                primary_tokenizer="general_auto",
                fallback_tokenizers=["xraygpt_llama", "xraygpt_bert"],
                model_specific_tokenizers={},
                context_keywords=["general", "conversation", "chat", "dialogue"]
            )
        }
    
    def setup_model_tokenizers(self, model_type: str = "xraygpt", **model_paths):
        """Setup tokenizers for specific model types with automatic domain detection"""
        
        if model_type.lower() == "xraygpt":
            self._setup_xraygpt_tokenizers(**model_paths)
        elif model_type.lower() == "medclip":
            self._setup_medclip_tokenizers(**model_paths)
        elif model_type.lower() == "automotive":
            self._setup_automotive_tokenizers(**model_paths)
        elif model_type.lower() == "blip2":
            self._setup_blip2_tokenizers(**model_paths)
        elif model_type.lower() == "minigpt4":
            self._setup_minigpt4_tokenizers(**model_paths)
        elif model_type.lower() == "llava":
            self._setup_llava_tokenizers(**model_paths)
        else:
            logger.warning(f"Unknown model type: {model_type}, setting up default tokenizers")
            self._setup_default_tokenizers(**model_paths)
    
    def setup_xraygpt_tokenizers(self, llama_model_path: str = None, bert_model_path: str = None):
        """Setup tokenizers specifically for XrayGPT (legacy method for compatibility)"""
        self._setup_xraygpt_tokenizers(llama_model_path=llama_model_path, bert_model_path=bert_model_path)
    
    def _setup_xraygpt_tokenizers(self, llama_model_path: str = None, bert_model_path: str = None, **kwargs):
        """Internal method to setup XrayGPT tokenizers"""
        # Setup LLaMA tokenizer for main language model
        llama_config = TokenizationConfig(
            tokenizer_type=TokenizerType.LLAMA,
            model_name_or_path=llama_model_path or "decapoda-research/llama-7b-hf",
            max_length=512,
            add_special_tokens=True,
            domain=DomainType.XRAYGPT,
            priority=15
        )
        self.register_tokenizer('xraygpt_llama', llama_config)
        
        # Setup BERT tokenizer for Q-Former
        bert_config = TokenizationConfig(
            tokenizer_type=TokenizerType.BERT,
            model_name_or_path=bert_model_path or "bert-base-uncased",
            max_length=512,
            add_special_tokens=True,
            domain=DomainType.XRAYGPT,
            priority=12
        )
        self.register_tokenizer('xraygpt_bert', bert_config)
        
        # Setup multi-modal tokenizer
        modality_configs = {
            'image_placeholder': '<ImageHere>',
            'medical_tokens': {
                'radiology': '[RADIOLOGY]',
                'finding': '[FINDING]',
                'impression': '[IMPRESSION]',
                'medical': '[MEDICAL]'
            }
        }
        self.register_multimodal_tokenizer('xraygpt_multimodal', 'xraygpt_llama', modality_configs)
        
        logger.info("XrayGPT tokenizers setup complete")
    
    def _setup_medclip_tokenizers(self, model_path: str = None, **kwargs):
        """Setup tokenizers for MedCLIP"""
        config = TokenizationConfig(
            tokenizer_type=TokenizerType.BERT,
            model_name_or_path=model_path or "bert-base-uncased",
            max_length=512,
            add_special_tokens=True,
            domain=DomainType.MEDCLIP,
            priority=14
        )
        self.register_tokenizer('medclip_tokenizer', config)
        
        # Setup multimodal tokenizer
        modality_configs = {
            'image_placeholder': '<ImageHere>',
            'medical_tokens': {
                'medical': '[MEDICAL]',
                'clip': '[CLIP]',
                'contrastive': '[CONTRAST]'
            }
        }
        self.register_multimodal_tokenizer('medclip_multimodal', 'medclip_tokenizer', modality_configs)
        logger.info("MedCLIP tokenizers setup complete")
    
    def _setup_automotive_tokenizers(self, model_path: str = None, **kwargs):
        """Setup tokenizers for automotive/autonomous driving models"""
        config = TokenizationConfig(
            tokenizer_type=TokenizerType.AUTO,
            model_name_or_path=model_path or "microsoft/DialoGPT-medium",
            max_length=512,
            add_special_tokens=True,
            domain=DomainType.AUTOMOTIVE,
            priority=13
        )
        self.register_tokenizer('automotive_tokenizer', config)
        
        # Setup multimodal tokenizer
        modality_configs = {
            'image_placeholder': '<ImageHere>',
            'automotive_tokens': {
                'vehicle': '[VEHICLE]',
                'traffic': '[TRAFFIC]',
                'road': '[ROAD]',
                'navigation': '[NAV]'
            }
        }
        self.register_multimodal_tokenizer('automotive_multimodal', 'automotive_tokenizer', modality_configs)
        logger.info("Automotive tokenizers setup complete")
    
    def _setup_blip2_tokenizers(self, model_path: str = None, **kwargs):
        """Setup tokenizers for BLIP2"""
        config = TokenizationConfig(
            tokenizer_type=TokenizerType.BERT,
            model_name_or_path=model_path or "bert-base-uncased",
            max_length=512,
            add_special_tokens=True,
            domain=DomainType.BLIP2,
            priority=12
        )
        self.register_tokenizer('blip2_tokenizer', config)
        
        # Setup multimodal tokenizer
        modality_configs = {
            'image_placeholder': '<ImageHere>',
            'vision_tokens': {
                'caption': '[CAPTION]',
                'description': '[DESC]',
                'visual': '[VISUAL]'
            }
        }
        self.register_multimodal_tokenizer('blip2_multimodal', 'blip2_tokenizer', modality_configs)
        logger.info("BLIP2 tokenizers setup complete")
    
    def _setup_minigpt4_tokenizers(self, llama_model_path: str = None, **kwargs):
        """Setup tokenizers for MiniGPT4"""
        config = TokenizationConfig(
            tokenizer_type=TokenizerType.LLAMA,
            model_name_or_path=llama_model_path or "decapoda-research/llama-7b-hf",
            max_length=512,
            add_special_tokens=True,
            domain=DomainType.MINIGPT4,
            priority=13
        )
        self.register_tokenizer('minigpt4_tokenizer', config)
        
        # Setup multimodal tokenizer
        modality_configs = {
            'image_placeholder': '<ImageHere>',
            'vision_tokens': {
                'caption': '[CAPTION]',
                'description': '[DESC]',
                'conversation': '[CONV]'
            }
        }
        self.register_multimodal_tokenizer('minigpt4_multimodal', 'minigpt4_tokenizer', modality_configs)
        logger.info("MiniGPT4 tokenizers setup complete")
    
    def _setup_llava_tokenizers(self, llama_model_path: str = None, **kwargs):
        """Setup tokenizers for LLaVA"""
        config = TokenizationConfig(
            tokenizer_type=TokenizerType.LLAMA,
            model_name_or_path=llama_model_path or "decapoda-research/llama-7b-hf",
            max_length=512,
            add_special_tokens=True,
            domain=DomainType.LLAVA,
            priority=13
        )
        self.register_tokenizer('llava_tokenizer', config)
        
        # Setup multimodal tokenizer
        modality_configs = {
            'image_placeholder': '<ImageHere>',
            'vision_tokens': {
                'instruction': '[INST]',
                'visual': '[VISUAL]',
                'following': '[FOLLOW]'
            }
        }
        self.register_multimodal_tokenizer('llava_multimodal', 'llava_tokenizer', modality_configs)
        logger.info("LLaVA tokenizers setup complete")
    
    def _setup_default_tokenizers(self, **kwargs):
        """Setup default/general tokenizers"""
        # Setup general auto tokenizer
        try:
            general_config = TokenizationConfig(
                tokenizer_type=TokenizerType.AUTO,
                model_name_or_path="microsoft/DialoGPT-medium",
                max_length=512,
                add_special_tokens=True,
                domain=DomainType.GENERAL,
                priority=5
            )
            self.register_tokenizer('general_auto', general_config)
            
            # Setup basic multimodal tokenizer
            modality_configs = {
                'image_placeholder': '<ImageHere>',
                'general_tokens': {
                    'context': '[CONTEXT]',
                    'task': '[TASK]'
                }
            }
            self.register_multimodal_tokenizer('general_multimodal', 'general_auto', modality_configs)
            
        except Exception as e:
            logger.warning(f"Could not setup default tokenizers: {e}")
        
        logger.info("Default tokenizers setup complete")

# Global framework instance
_framework = None

def get_framework() -> MMAITokenizationFramework:
    """Get the global tokenization framework instance"""
    global _framework
    if _framework is None:
        _framework = MMAITokenizationFramework()
    return _framework

def setup_xraygpt_tokenization(llama_model_path: str = None, bert_model_path: str = None):
    """Setup XrayGPT tokenization with the global framework (legacy method)"""
    framework = get_framework()
    framework.setup_xraygpt_tokenizers(llama_model_path, bert_model_path)
    return framework

def setup_model_tokenization(model_type: str = None, auto_detect: bool = True, **model_paths):
    """
    Setup tokenization for any model type with automatic detection
    
    Args:
        model_type: Specific model type ('xraygpt', 'medclip', 'automotive', etc.)
        auto_detect: Whether to auto-detect model type from paths
        **model_paths: Model paths (llama_model_path, bert_model_path, model_path, etc.)
    
    Returns:
        MMAITokenizationFramework instance
    """
    framework = get_framework()
    
    # Auto-detect model type if not specified
    if model_type is None and auto_detect:
        model_type = _auto_detect_model_type(**model_paths)
        if model_type:
            logger.info(f"Auto-detected model type: {model_type}")
        else:
            model_type = "xraygpt"  # Default fallback
            logger.info(f"Could not auto-detect model type, using default: {model_type}")
    elif model_type is None:
        model_type = "xraygpt"  # Default fallback
    
    # Setup tokenizers for the detected/specified model type
    framework.setup_model_tokenizers(model_type, **model_paths)
    
    return framework

def _auto_detect_model_type(**model_paths) -> Optional[str]:
    """Auto-detect model type from provided model paths"""
    
    # Check all provided paths for model type indicators
    all_paths = []
    for key, path in model_paths.items():
        if path and isinstance(path, str):
            all_paths.append(path.lower())
    
    # Combine all paths for analysis
    combined_path = " ".join(all_paths)
    
    # Model type detection patterns (order matters - more specific first)
    detection_patterns = [
        ("xraygpt", ["xraygpt", "xray_gpt", "vicuna_radiology", "mini_gpt4"]),
        ("medclip", ["medclip", "medical_clip", "med_clip"]),
        ("automotive", ["automotive", "tesla", "waymo", "cruise", "autonomous", "driving"]),
        ("blip2", ["blip2", "blip-2", "salesforce/blip2"]),
        ("minigpt4", ["minigpt4", "mini_gpt4", "mini-gpt4"]),
        ("llava", ["llava", "llava_v1", "llava-v1.5"]),
        ("flamingo", ["flamingo", "openflamingo"])
    ]
    
    for model_type, patterns in detection_patterns:
        if any(pattern in combined_path for pattern in patterns):
            return model_type
    
    return None