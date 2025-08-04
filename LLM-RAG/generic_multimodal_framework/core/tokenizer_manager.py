"""
Tokenizer Manager for Generic Multimodal Framework
Handles domain-specific tokenization based on data type and configuration
"""

import torch
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer, 
    BertTokenizer, 
    LlamaTokenizer,
    GPT2Tokenizer,
    T5Tokenizer
)
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseTokenizer(ABC):
    """Base class for domain-specific tokenizers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # All parameters from config - no hardcoded defaults
        self.max_length = config.get("max_length")
        self.padding = config.get("padding")
        self.truncation = config.get("truncation")
        
    @abstractmethod
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text input"""
        pass
    
    @abstractmethod
    def tokenize_image(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Tokenize image input (if applicable)"""
        pass
    
    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        pass

class TransformerTextTokenizer(BaseTokenizer):
    """Transformer-based text tokenizer"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "bert-base-uncased")
        self.tokenizer_type = config.get("tokenizer_type", "auto")
        
        # Initialize tokenizer based on type
        if self.tokenizer_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        elif self.tokenizer_type == "llama":
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.tokenizer_type == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif self.tokenizer_type == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
        # Add special tokens if specified
        special_tokens = config.get("special_tokens", {})
        if special_tokens:
            self.tokenizer.add_special_tokens(special_tokens)
            
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text input"""
        try:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                add_special_tokens=True
            )
            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "token_type_ids": encoded.get("token_type_ids", None)
            }
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            raise
    
    def tokenize_image(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Image tokenization not supported for text tokenizer"""
        raise NotImplementedError("Image tokenization not supported for text-only tokenizer")
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

class VisionTokenizer(BaseTokenizer):
    """Vision tokenizer for image inputs"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.patch_size = config.get("patch_size", 16)
        self.image_size = config.get("image_size", 224)
        self.vocab_size = config.get("vocab_size", 8192)  # For discrete tokenization
        self.tokenization_method = config.get("method", "patch")  # patch, vqvae, etc.
        
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Text tokenization not supported for vision tokenizer"""
        raise NotImplementedError("Text tokenization not supported for vision-only tokenizer")
    
    def tokenize_image(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Tokenize image input"""
        if isinstance(image, Image.Image):
            # Convert PIL to tensor
            image = torch.from_numpy(np.array(image)).float() / 255.0
            if image.dim() == 2:  # Grayscale
                image = image.unsqueeze(0)
            elif image.dim() == 3 and image.shape[2] in [1, 3]:  # HWC to CHW
                image = image.permute(2, 0, 1)
        
        if self.tokenization_method == "patch":
            return self._patch_tokenize(image)
        else:
            raise NotImplementedError(f"Tokenization method {self.tokenization_method} not implemented")
    
    def _patch_tokenize(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Tokenize image into patches"""
        C, H, W = image.shape
        
        # Ensure image is the right size
        if H != self.image_size or W != self.image_size:
            import torch.nn.functional as F
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Create patches
        num_patches_per_dim = self.image_size // self.patch_size
        patches = image.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(C, num_patches_per_dim * num_patches_per_dim, self.patch_size * self.patch_size)
        patches = patches.permute(1, 0, 2).contiguous().view(num_patches_per_dim * num_patches_per_dim, -1)
        
        # Create attention mask (all patches are valid)
        attention_mask = torch.ones(patches.shape[0], dtype=torch.long)
        
        return {
            "patch_embeddings": patches,
            "attention_mask": attention_mask,
            "num_patches": torch.tensor(patches.shape[0])
        }
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode not applicable for vision tokenizer"""
        raise NotImplementedError("Decode not applicable for vision tokenizer")

class MultiModalTokenizer(BaseTokenizer):
    """Multimodal tokenizer that handles both text and images"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize text tokenizer
        text_config = config.get("text_tokenizer", {})
        text_config.update({"max_length": self.max_length})
        self.text_tokenizer = TransformerTextTokenizer(text_config)
        
        # Initialize vision tokenizer
        vision_config = config.get("vision_tokenizer", {})
        self.vision_tokenizer = VisionTokenizer(vision_config)
        
        # Special tokens for multimodal
        self.image_token = config.get("image_token", "<IMAGE>")
        self.image_start_token = config.get("image_start_token", "<IMG>")
        self.image_end_token = config.get("image_end_token", "</IMG>")
        
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text with multimodal awareness"""
        return self.text_tokenizer.tokenize_text(text)
    
    def tokenize_image(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Tokenize image input"""
        return self.vision_tokenizer.tokenize_image(image)
    
    def tokenize_multimodal(self, text: str, image: Union[Image.Image, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Tokenize both text and image together"""
        # Tokenize text
        text_tokens = self.tokenize_text(text)
        
        # Tokenize image
        image_tokens = self.tokenize_image(image)
        
        # Combine tokens (implementation depends on specific architecture)
        return {
            "text_input_ids": text_tokens["input_ids"],
            "text_attention_mask": text_tokens["attention_mask"],
            "image_embeddings": image_tokens["patch_embeddings"],
            "image_attention_mask": image_tokens["attention_mask"],
            "combined_length": text_tokens["input_ids"].shape[1] + image_tokens["patch_embeddings"].shape[0]
        }
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        return self.text_tokenizer.decode(token_ids)

class TokenizerManager:
    """Manager for creating and managing tokenizers based on domain and data type"""
    
    _tokenizer_registry = {
        "text": TransformerTextTokenizer,
        "vision": VisionTokenizer,
        "multimodal": MultiModalTokenizer
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain = config.get("domain", "generic")
        self.tokenizers = {}
        
        # Initialize tokenizers based on config
        self._initialize_tokenizers()
    
    def _initialize_tokenizers(self):
        """Initialize tokenizers based on configuration"""
        tokenizer_configs = self.config.get("tokenizers", {})
        
        for data_type, tokenizer_config in tokenizer_configs.items():
            tokenizer_type = tokenizer_config.get("type", "text")
            
            if tokenizer_type in self._tokenizer_registry:
                tokenizer_class = self._tokenizer_registry[tokenizer_type]
                self.tokenizers[data_type] = tokenizer_class(tokenizer_config)
            else:
                logger.warning(f"Unknown tokenizer type: {tokenizer_type}")
    
    def get_tokenizer(self, data_type: str) -> BaseTokenizer:
        """Get tokenizer for specific data type"""
        if data_type not in self.tokenizers:
            # Create default tokenizer
            default_config = {"type": "text", "model_name": "bert-base-uncased"}
            self.tokenizers[data_type] = TransformerTextTokenizer(default_config)
            
        return self.tokenizers[data_type]
    
    def tokenize(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize multimodal data"""
        tokenized_data = {}
        
        for key, value in data.items():
            if key.startswith("text") or key in ["caption", "description", "query"]:
                tokenizer = self.get_tokenizer("text")
                tokenized_data[f"{key}_tokens"] = tokenizer.tokenize_text(value)
                
            elif key.startswith("image") or key == "visual_input":
                tokenizer = self.get_tokenizer("vision")
                tokenized_data[f"{key}_tokens"] = tokenizer.tokenize_image(value)
                
            elif key == "multimodal" and isinstance(value, dict):
                # Handle multimodal input
                if "text" in value and "image" in value:
                    tokenizer = self.get_tokenizer("multimodal")
                    tokenized_data[f"{key}_tokens"] = tokenizer.tokenize_multimodal(
                        value["text"], value["image"]
                    )
        
        return tokenized_data
    
    def decode(self, token_ids: torch.Tensor, data_type: str = "text") -> str:
        """Decode tokens back to text"""
        tokenizer = self.get_tokenizer(data_type)
        return tokenizer.decode(token_ids)
    
    @classmethod
    def register_tokenizer(cls, name: str, tokenizer_class: type):
        """Register a new tokenizer type"""
        cls._tokenizer_registry[name] = tokenizer_class