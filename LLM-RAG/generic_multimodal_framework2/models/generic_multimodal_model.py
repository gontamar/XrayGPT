"""
Generic Multimodal Model

A flexible multimodal AI model that can handle various domains and data types.
Implements the complete pipeline: validation -> tokenization -> encoding -> attention -> decoding
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass

from ..core.data_validator import MultiModalDataValidator, ValidationOutput
from ..core.tokenizer_manager import TokenizerManager
from ..core.encoder_manager import EncoderManager
from ..core.attention_manager import AttentionManager
from ..core.decoder_manager import DecoderManager

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    """Output from the generic multimodal model"""
    text_response: Union[str, List[str]]
    attention_weights: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None
    validation_results: Optional[ValidationOutput] = None
    metadata: Optional[Dict[str, Any]] = None


class GenericMultiModalModel(nn.Module):
    """
    Generic Multimodal AI Model
    
    A flexible framework that can handle various domains and modalities:
    1. Data Validation - Domain-specific validation and conditioning
    2. Tokenization - Flexible tokenization based on data type
    3. Encoding - Configurable encoders for different modalities
    4. Attention - Self and cross-attention mechanisms
    5. Decoding - Generate readable text responses
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.domain = config.get('domain', 'generic')
        self.device = config.get('device', 'cuda')
        
        # Initialize components
        self._initialize_components()
        
        # Model state
        self.training_mode = True
        
        logger.info(f"Initialized GenericMultiModalModel for domain: {self.domain}")
    
    def _initialize_components(self):
        """Initialize all model components"""
        
        # 1. Data Validator
        validator_config = self.config.get('validator', {})
        validator_config['domain'] = self.domain
        self.validator = MultiModalDataValidator(validator_config)
        
        # 2. Tokenizer Manager
        tokenizer_config = self.config.get('tokenizer', {})
        self.tokenizer_manager = TokenizerManager(tokenizer_config)
        
        # 3. Encoder Manager
        encoder_config = self.config.get('encoder', {})
        self.encoder_manager = EncoderManager(encoder_config)
        
        # 4. Attention Manager
        attention_config = self.config.get('attention', {})
        self.attention_manager = AttentionManager(attention_config)
        
        # 5. Decoder Manager
        decoder_config = self.config.get('decoder', {})
        self.decoder_manager = DecoderManager(decoder_config)
        
        # Move components to device
        self.to(self.device)
    
    def forward(self, 
                image: Optional[torch.Tensor] = None,
                text: Optional[Union[str, List[str]]] = None,
                additional_data: Optional[Dict[str, Any]] = None,
                return_attention: bool = False,
                return_embeddings: bool = False) -> ModelOutput:
        """
        Forward pass through the complete pipeline
        
        Args:
            image: Input image tensor(s)
            text: Input text string(s)
            additional_data: Additional domain-specific data
            return_attention: Whether to return attention weights
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            ModelOutput containing the generated response and optional metadata
        """
        
        # Step 1: Data Validation and Conditioning
        validation_output = self.validator.validate_and_condition(
            image=image,
            text=text,
            additional_data=additional_data
        )
        
        if not validation_output.is_valid:
            logger.warning(f"Data validation failed: {validation_output.message}")
            return ModelOutput(
                text_response="Invalid input data",
                validation_results=validation_output
            )
        
        # Use conditioned data
        conditioned_image = validation_output.conditioned_data.get('image')
        conditioned_text = validation_output.conditioned_data.get('text')
        
        # Step 2: Tokenization
        tokenized_data = self.tokenizer_manager.tokenize(
            image=conditioned_image,
            text=conditioned_text,
            additional_data=validation_output.conditioned_data.get('additional_data')
        )
        
        # Step 3: Encoding
        encoded_data = self.encoder_manager.encode(tokenized_data)
        
        # Step 4: Attention Processing
        attention_output = self.attention_manager.process(
            image_features=encoded_data.get('image_embeddings'),
            text_features=encoded_data.get('text_embeddings'),
            additional_features=encoded_data.get('additional_embeddings'),
            return_attention_weights=return_attention
        )
        
        # Step 5: Decoding
        final_embeddings = attention_output['fused_embeddings']
        attention_mask = attention_output.get('attention_mask')
        
        # Generate text response
        generation_config = self.config.get('generation', {})
        text_response = self.decoder_manager.decode(
            embeddings=final_embeddings,
            attention_mask=attention_mask,
            generation_config=generation_config
        )
        
        # Prepare output
        output = ModelOutput(
            text_response=text_response,
            validation_results=validation_output
        )
        
        if return_attention:
            output.attention_weights = attention_output.get('attention_weights')
        
        if return_embeddings:
            output.embeddings = {
                'encoded': encoded_data,
                'fused': final_embeddings
            }
        
        # Add metadata
        output.metadata = {
            'domain': self.domain,
            'processing_steps': [
                'validation',
                'tokenization', 
                'encoding',
                'attention',
                'decoding'
            ],
            'model_config': self.config.get('model_info', {})
        }
        
        return output
    
    def generate(self, 
                 image: Optional[torch.Tensor] = None,
                 text: Optional[Union[str, List[str]]] = None,
                 additional_data: Optional[Dict[str, Any]] = None,
                 **generation_kwargs) -> str:
        """
        Generate text response (inference mode)
        
        Args:
            image: Input image tensor
            text: Input text
            additional_data: Additional domain-specific data
            **generation_kwargs: Generation configuration overrides
            
        Returns:
            Generated text response
        """
        
        self.eval()
        
        with torch.no_grad():
            # Update generation config if provided
            if generation_kwargs:
                original_config = self.config.get('generation', {})
                self.config['generation'] = {**original_config, **generation_kwargs}
            
            output = self.forward(
                image=image,
                text=text,
                additional_data=additional_data,
                return_attention=False,
                return_embeddings=False
            )
            
            # Restore original config
            if generation_kwargs:
                self.config['generation'] = original_config
        
        if self.training_mode:
            self.train()
        
        return output.text_response
    
    def compute_loss(self, 
                     image: Optional[torch.Tensor] = None,
                     text: Optional[Union[str, List[str]]] = None,
                     target_text: Optional[Union[str, List[str]]] = None,
                     additional_data: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Compute training loss
        
        Args:
            image: Input image tensor
            text: Input text
            target_text: Target text for training
            additional_data: Additional domain-specific data
            
        Returns:
            Loss tensor
        """
        
        # This would need to be implemented based on specific training objectives
        # For now, return a placeholder
        return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def set_domain(self, domain: str):
        """Change the domain configuration"""
        self.domain = domain
        self.config['domain'] = domain
        
        # Reinitialize validator for new domain
        validator_config = self.config.get('validator', {})
        validator_config['domain'] = domain
        self.validator = MultiModalDataValidator(validator_config)
        
        logger.info(f"Changed domain to: {domain}")
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update model configuration"""
        self.config.update(config_updates)
        
        # Reinitialize components if needed
        if 'validator' in config_updates:
            self.validator = MultiModalDataValidator(config_updates['validator'])
        if 'tokenizer' in config_updates:
            self.tokenizer_manager = TokenizerManager(config_updates['tokenizer'])
        if 'encoder' in config_updates:
            self.encoder_manager = EncoderManager(config_updates['encoder'])
        if 'attention' in config_updates:
            self.attention_manager = AttentionManager(config_updates['attention'])
        if 'decoder' in config_updates:
            self.decoder_manager = DecoderManager(config_updates['decoder'])
        
        logger.info("Updated model configuration")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model configuration"""
        return {
            'domain': self.domain,
            'device': self.device,
            'config': self.config,
            'components': {
                'validator': type(self.validator).__name__,
                'tokenizer': type(self.tokenizer_manager).__name__,
                'encoder': type(self.encoder_manager).__name__,
                'attention': type(self.attention_manager).__name__,
                'decoder': type(self.decoder_manager).__name__,
            }
        }
    
    def save_pretrained(self, save_path: str):
        """Save model configuration and weights"""
        import json
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(save_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save model weights
        weights_path = os.path.join(save_path, 'pytorch_model.bin')
        torch.save(self.state_dict(), weights_path)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, device: str = 'cuda'):
        """Load model from saved configuration and weights"""
        import json
        import os
        
        # Load configuration
        config_path = os.path.join(load_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config['device'] = device
        
        # Create model
        model = cls(config)
        
        # Load weights
        weights_path = os.path.join(load_path, 'pytorch_model.bin')
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
        
        logger.info(f"Model loaded from {load_path}")
        return model
    
    def to(self, device):
        """Move model to device"""
        super().to(device)
        self.device = device
        
        # Move components to device
        if hasattr(self, 'encoder_manager'):
            self.encoder_manager.to(device)
        if hasattr(self, 'attention_manager'):
            self.attention_manager.to(device)
        if hasattr(self, 'decoder_manager'):
            self.decoder_manager.set_device(device)
        
        return self
    
    def train(self, mode: bool = True):
        """Set training mode"""
        super().train(mode)
        self.training_mode = mode
        
        # Set components to training mode
        if hasattr(self, 'encoder_manager'):
            self.encoder_manager.train(mode)
        if hasattr(self, 'attention_manager'):
            self.attention_manager.train(mode)
        
        return self
    
    def eval(self):
        """Set evaluation mode"""
        return self.train(False)


def create_model_from_config(config_path: str, device: str = 'cuda') -> GenericMultiModalModel:
    """Create model from configuration file"""
    import json
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config['device'] = device
    return GenericMultiModalModel(config)