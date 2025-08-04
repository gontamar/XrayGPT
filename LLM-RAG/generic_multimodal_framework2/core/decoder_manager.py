"""
Decoder Manager for Generic Multimodal Framework

Handles the decoding of cross-attention outputs back to readable text responses.
Supports multiple decoder types and generation strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
import logging

logger = logging.getLogger(__name__)


class BaseDecoder(ABC):
    """Base class for all decoders"""
    
    @abstractmethod
    def decode(self, 
               embeddings: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               generation_config: Optional[Dict[str, Any]] = None) -> str:
        """Decode embeddings to text"""
        pass
    
    @abstractmethod
    def batch_decode(self, 
                    embeddings: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    generation_config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Decode batch of embeddings to text"""
        pass


class LLMDecoder(BaseDecoder):
    """Decoder using Large Language Models (LLaMA, Vicuna, etc.)"""
    
    def __init__(self, 
                 model_name_or_path: str,
                 tokenizer_name_or_path: Optional[str] = None,
                 device: str = "cuda",
                 torch_dtype: torch.dtype = torch.float16,
                 low_resource: bool = False,
                 max_new_tokens: int = 300,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 num_beams: int = 1,
                 repetition_penalty: float = 1.0,
                 length_penalty: float = 1.0,
                 stop_tokens: Optional[List[str]] = None):
        
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.stop_tokens = stop_tokens or []
        
        # Load tokenizer
        tokenizer_path = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if low_resource:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
            ).to(device)
        
        # Set generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Setup stopping criteria
        self.stopping_criteria = self._setup_stopping_criteria()
        
        logger.info(f"Initialized LLMDecoder with model: {model_name_or_path}")
    
    def _setup_stopping_criteria(self) -> StoppingCriteriaList:
        """Setup stopping criteria based on stop tokens"""
        if not self.stop_tokens:
            return StoppingCriteriaList([])
        
        stop_words_ids = []
        for stop_token in self.stop_tokens:
            ids = self.tokenizer.encode(stop_token, add_special_tokens=False)
            if ids:
                stop_words_ids.append(torch.tensor(ids).to(self.device))
        
        if stop_words_ids:
            return StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        return StoppingCriteriaList([])
    
    def decode(self, 
               embeddings: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               generation_config: Optional[Dict[str, Any]] = None) -> str:
        """Decode single embedding to text"""
        results = self.batch_decode(embeddings.unsqueeze(0), 
                                  attention_mask.unsqueeze(0) if attention_mask is not None else None,
                                  generation_config)
        return results[0]
    
    def batch_decode(self, 
                    embeddings: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    generation_config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Decode batch of embeddings to text"""
        
        # Update generation config if provided
        gen_config = self.generation_config
        if generation_config:
            gen_config = GenerationConfig(**{**self.generation_config.to_dict(), **generation_config})
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                generation_config=gen_config,
                stopping_criteria=self.stopping_criteria,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode generated tokens
        generated_texts = []
        for sequence in outputs.sequences:
            # Remove special tokens and clean up
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            text = self._clean_generated_text(text)
            generated_texts.append(text)
        
        return generated_texts
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean up generated text by removing stop tokens and unwanted patterns"""
        # Remove stop tokens
        for stop_token in self.stop_tokens:
            if stop_token in text:
                text = text.split(stop_token)[0]
        
        # Remove common unwanted patterns
        text = text.strip()
        
        # Remove repeated patterns (simple heuristic)
        lines = text.split('\n')
        if len(lines) > 1 and lines[-1] == lines[-2]:
            text = '\n'.join(lines[:-1])
        
        return text


class StoppingCriteriaSub(StoppingCriteria):
    """Custom stopping criteria for specific tokens"""
    
    def __init__(self, stops: List[torch.Tensor], encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.encounters = encounters
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if len(input_ids[0]) >= len(stop):
                if torch.all((stop == input_ids[0][-len(stop):])).item():
                    return True
        return False


class TransformerDecoder(BaseDecoder):
    """Decoder using Transformer architecture for sequence-to-sequence tasks"""
    
    def __init__(self, 
                 model_name_or_path: str,
                 device: str = "cuda",
                 max_length: int = 512):
        
        self.device = device
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
        
        logger.info(f"Initialized TransformerDecoder with model: {model_name_or_path}")
    
    def decode(self, 
               embeddings: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               generation_config: Optional[Dict[str, Any]] = None) -> str:
        """Decode single embedding to text"""
        results = self.batch_decode(embeddings.unsqueeze(0), 
                                  attention_mask.unsqueeze(0) if attention_mask is not None else None,
                                  generation_config)
        return results[0]
    
    def batch_decode(self, 
                    embeddings: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    generation_config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Decode batch of embeddings to text"""
        
        # Simple generation using the model
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                max_length=self.max_length,
                num_return_sequences=1,
                temperature=generation_config.get('temperature', 1.0) if generation_config else 1.0,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode generated tokens
        generated_texts = []
        for sequence in outputs:
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            generated_texts.append(text.strip())
        
        return generated_texts


class DecoderManager:
    """Manager class for handling different types of decoders"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.decoder = self._initialize_decoder()
        
        logger.info(f"Initialized DecoderManager with decoder type: {config.get('type', 'unknown')}")
    
    def _initialize_decoder(self) -> BaseDecoder:
        """Initialize decoder based on configuration"""
        decoder_type = self.config.get('type', 'llm')
        
        if decoder_type == 'llm':
            return LLMDecoder(
                model_name_or_path=self.config['model_name_or_path'],
                tokenizer_name_or_path=self.config.get('tokenizer_name_or_path'),
                device=self.config.get('device', 'cuda'),
                torch_dtype=getattr(torch, self.config.get('torch_dtype', 'float16')),
                low_resource=self.config.get('low_resource', False),
                max_new_tokens=self.config.get('max_new_tokens', 300),
                temperature=self.config.get('temperature', 1.0),
                top_p=self.config.get('top_p', 0.9),
                num_beams=self.config.get('num_beams', 1),
                repetition_penalty=self.config.get('repetition_penalty', 1.0),
                length_penalty=self.config.get('length_penalty', 1.0),
                stop_tokens=self.config.get('stop_tokens', [])
            )
        
        elif decoder_type == 'transformer':
            return TransformerDecoder(
                model_name_or_path=self.config['model_name_or_path'],
                device=self.config.get('device', 'cuda'),
                max_length=self.config.get('max_length', 512)
            )
        
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
    
    def decode(self, 
               embeddings: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None,
               generation_config: Optional[Dict[str, Any]] = None) -> Union[str, List[str]]:
        """Decode embeddings to text"""
        
        if embeddings.dim() == 2:  # Single sequence
            embeddings = embeddings.unsqueeze(0)
        
        if embeddings.size(0) == 1:
            return self.decoder.decode(embeddings[0], attention_mask, generation_config)
        else:
            return self.decoder.batch_decode(embeddings, attention_mask, generation_config)
    
    def update_generation_config(self, **kwargs):
        """Update generation configuration"""
        if hasattr(self.decoder, 'generation_config'):
            for key, value in kwargs.items():
                setattr(self.decoder.generation_config, key, value)
    
    def set_device(self, device: str):
        """Move decoder to specified device"""
        if hasattr(self.decoder, 'model'):
            self.decoder.model = self.decoder.model.to(device)
        if hasattr(self.decoder, 'device'):
            self.decoder.device = device


# Factory function for creating decoders
def create_decoder(config: Dict[str, Any]) -> DecoderManager:
    """Factory function to create decoder manager"""
    return DecoderManager(config)


# Registry for decoder types
DECODER_REGISTRY = {
    'llm': LLMDecoder,
    'transformer': TransformerDecoder,
}


def register_decoder(name: str, decoder_class: type):
    """Register a new decoder type"""
    DECODER_REGISTRY[name] = decoder_class


def get_available_decoders() -> List[str]:
    """Get list of available decoder types"""
    return list(DECODER_REGISTRY.keys())