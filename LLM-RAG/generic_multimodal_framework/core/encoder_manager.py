"""
Encoder Manager for Generic Multimodal Framework
Handles domain-specific encoding based on data type and configuration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging
from transformers import (
    AutoModel, 
    BertModel, 
    RobertaModel,
    ViTModel,
    CLIPVisionModel,
    CLIPTextModel
)

logger = logging.getLogger(__name__)

class BaseEncoder(ABC):
    """Base class for domain-specific encoders"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # All parameters from config - no hardcoded defaults
        self.hidden_size = config.get("hidden_size")
        self.device = config.get("device")
        
        if not self.hidden_size:
            raise ValueError("hidden_size must be specified in encoder config")
        if not self.device:
            raise ValueError("device must be specified in encoder config")
        
    @abstractmethod
    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode inputs to embeddings"""
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        """Get output embedding dimension"""
        pass

class TextEncoder(BaseEncoder, nn.Module):
    """Text encoder using transformer models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        nn.Module.__init__(self)
        
        self.model_name = config.get("model_name", "bert-base-uncased")
        self.encoder_type = config.get("encoder_type", "bert")
        self.freeze_encoder = config.get("freeze_encoder", False)
        self.pooling_strategy = config.get("pooling_strategy", "cls")  # cls, mean, max
        
        # Initialize encoder
        if self.encoder_type == "bert":
            self.encoder = BertModel.from_pretrained(self.model_name)
        elif self.encoder_type == "roberta":
            self.encoder = RobertaModel.from_pretrained(self.model_name)
        elif self.encoder_type == "clip_text":
            self.encoder = CLIPTextModel.from_pretrained(self.model_name)
        else:
            self.encoder = AutoModel.from_pretrained(self.model_name)
        
        self.hidden_size = self.encoder.config.hidden_size
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        # Projection layer if needed
        output_dim = config.get("output_dim", self.hidden_size)
        if output_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, output_dim)
            self.hidden_size = output_dim
        else:
            self.projection = None
    
    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode text inputs to embeddings"""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif self.pooling_strategy == "mean":
            # Mean pooling with attention mask
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling_strategy == "max":
            # Max pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            embeddings = torch.max(token_embeddings, 1)[0]
        else:
            embeddings = outputs.last_hidden_state  # Return all tokens
        
        # Apply projection if exists
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        return embeddings
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension"""
        return self.hidden_size

class VisionEncoder(BaseEncoder, nn.Module):
    """Vision encoder using CNN or Vision Transformer models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        nn.Module.__init__(self)
        
        self.model_name = config.get("model_name", "google/vit-base-patch16-224")
        self.encoder_type = config.get("encoder_type", "vit")
        self.freeze_encoder = config.get("freeze_encoder", False)
        self.image_size = config.get("image_size", 224)
        
        # Initialize encoder
        if self.encoder_type == "vit":
            self.encoder = ViTModel.from_pretrained(self.model_name)
        elif self.encoder_type == "clip_vision":
            self.encoder = CLIPVisionModel.from_pretrained(self.model_name)
        elif self.encoder_type == "eva_clip_g":
            # Custom EVA-CLIP implementation (simplified)
            from xraygpt.models.eva_vit import create_eva_vit_g
            self.encoder = create_eva_vit_g(
                img_size=self.image_size,
                drop_path_rate=config.get("drop_path_rate", 0),
                use_grad_checkpoint=config.get("use_grad_checkpoint", False),
                precision=config.get("precision", "fp16")
            )
            self.hidden_size = self.encoder.num_features
        else:
            raise ValueError(f"Unknown vision encoder type: {self.encoder_type}")
        
        if hasattr(self.encoder, 'config') and hasattr(self.encoder.config, 'hidden_size'):
            self.hidden_size = self.encoder.config.hidden_size
        
        # Layer normalization
        self.ln_vision = nn.LayerNorm(self.hidden_size)
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        # Projection layer if needed
        output_dim = config.get("output_dim", self.hidden_size)
        if output_dim != self.hidden_size:
            self.projection = nn.Linear(self.hidden_size, output_dim)
            self.hidden_size = output_dim
        else:
            self.projection = None
    
    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode vision inputs to embeddings"""
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
        elif "patch_embeddings" in inputs:
            # Already tokenized patches
            return inputs["patch_embeddings"]
        else:
            raise ValueError("No valid vision input found")
        
        # Get encoder outputs
        if self.encoder_type in ["vit", "clip_vision"]:
            outputs = self.encoder(pixel_values=pixel_values, return_dict=True)
            embeddings = outputs.last_hidden_state
        else:
            # Custom encoder (e.g., EVA-CLIP)
            embeddings = self.encoder(pixel_values)
        
        # Apply layer normalization
        embeddings = self.ln_vision(embeddings)
        
        # Apply projection if exists
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        
        return embeddings
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension"""
        return self.hidden_size

class AudioEncoder(BaseEncoder, nn.Module):
    """Audio encoder for audio inputs"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        nn.Module.__init__(self)
        
        self.model_name = config.get("model_name", "facebook/wav2vec2-base")
        self.encoder_type = config.get("encoder_type", "wav2vec2")
        self.freeze_encoder = config.get("freeze_encoder", False)
        
        # Initialize encoder
        if self.encoder_type == "wav2vec2":
            from transformers import Wav2Vec2Model
            self.encoder = Wav2Vec2Model.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unknown audio encoder type: {self.encoder_type}")
        
        self.hidden_size = self.encoder.config.hidden_size
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
    
    def encode(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode audio inputs to embeddings"""
        input_values = inputs["input_values"]
        attention_mask = inputs.get("attention_mask", None)
        
        outputs = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return outputs.last_hidden_state
    
    def get_output_dim(self) -> int:
        """Get output embedding dimension"""
        return self.hidden_size

class EncoderManager:
    """Manager for creating and managing encoders based on domain and data type"""
    
    _encoder_registry = {
        "text": TextEncoder,
        "vision": VisionEncoder,
        "audio": AudioEncoder
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain = config.get("domain", "generic")
        self.encoders = {}
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize encoders based on config
        self._initialize_encoders()
    
    def _initialize_encoders(self):
        """Initialize encoders based on configuration"""
        encoder_configs = self.config.get("encoders", {})
        
        for data_type, encoder_config in encoder_configs.items():
            encoder_type = encoder_config.get("type", "text")
            encoder_config["device"] = self.device
            
            if encoder_type in self._encoder_registry:
                encoder_class = self._encoder_registry[encoder_type]
                encoder = encoder_class(encoder_config)
                encoder.to(self.device)
                self.encoders[data_type] = encoder
            else:
                logger.warning(f"Unknown encoder type: {encoder_type}")
    
    def get_encoder(self, data_type: str) -> BaseEncoder:
        """Get encoder for specific data type"""
        if data_type not in self.encoders:
            # Create default encoder
            default_config = {
                "type": "text", 
                "model_name": "bert-base-uncased",
                "device": self.device
            }
            encoder = TextEncoder(default_config)
            encoder.to(self.device)
            self.encoders[data_type] = encoder
            
        return self.encoders[data_type]
    
    def encode(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Encode multimodal data"""
        encoded_data = {}
        
        for key, value in data.items():
            if key.endswith("_tokens"):
                # Determine data type from key
                data_type = key.replace("_tokens", "")
                if data_type.startswith("text"):
                    encoder = self.get_encoder("text")
                elif data_type.startswith("image"):
                    encoder = self.get_encoder("vision")
                elif data_type.startswith("audio"):
                    encoder = self.get_encoder("audio")
                else:
                    encoder = self.get_encoder("text")  # Default
                
                # Move data to device
                if isinstance(value, dict):
                    value = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in value.items()}
                
                encoded_data[f"{data_type}_embeddings"] = encoder.encode(value)
        
        return encoded_data
    
    def get_output_dims(self) -> Dict[str, int]:
        """Get output dimensions for all encoders"""
        return {name: encoder.get_output_dim() for name, encoder in self.encoders.items()}
    
    @classmethod
    def register_encoder(cls, name: str, encoder_class: type):
        """Register a new encoder type"""
        cls._encoder_registry[name] = encoder_class
    
    def to(self, device):
        """Move all encoders to device"""
        self.device = device
        for encoder in self.encoders.values():
            if hasattr(encoder, 'to'):
                encoder.to(device)
        return self
    
    def train(self):
        """Set all encoders to training mode"""
        for encoder in self.encoders.values():
            if hasattr(encoder, 'train'):
                encoder.train()
    
    def eval(self):
        """Set all encoders to evaluation mode"""
        for encoder in self.encoders.values():
            if hasattr(encoder, 'eval'):
                encoder.eval()