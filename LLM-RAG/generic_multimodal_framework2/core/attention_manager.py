"""
Attention Manager for Generic Multimodal Framework
Handles self-attention and cross-attention mechanisms for multimodal learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union, Tuple
from abc import ABC, abstractmethod
import math
import logging

logger = logging.getLogger(__name__)

class BaseAttentionModule(ABC, nn.Module):
    """Base class for attention mechanisms"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.hidden_size = config.get("hidden_size", 768)
        self.num_attention_heads = config.get("num_attention_heads", 12)
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.dropout_prob = config.get("attention_dropout", 0.1)
        
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass for attention mechanism"""
        pass

class SelfAttentionModule(BaseAttentionModule):
    """Self-attention mechanism for focusing on relevant parts of input"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Positional encoding if needed
        self.use_positional_encoding = config.get("use_positional_encoding", True)
        if self.use_positional_encoding:
            max_position = config.get("max_position_embeddings", 512)
            self.position_embeddings = nn.Embedding(max_position, self.hidden_size)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for self-attention
        
        Args:
            hidden_states: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Mask to avoid attention on padding tokens
            position_ids: Position indices for positional encoding
            output_attentions: Whether to return attention weights
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeddings
        
        # Compute query, key, value
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self._transpose_for_scores(query_layer)
        key_layer = self._transpose_for_scores(key_layer)
        value_layer = self._transpose_for_scores(value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            # Expand mask for multi-head attention
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.expand(
                batch_size, self.num_attention_heads, seq_len, seq_len
            )
            attention_scores = attention_scores + (extended_attention_mask * -10000.0)
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        attention_output = self.output_projection(context_layer)
        
        # Residual connection and layer norm
        attention_output = self.layer_norm(attention_output + hidden_states)
        
        if output_attentions:
            return attention_output, attention_probs
        return attention_output
    
    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention computation"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

class CrossAttentionModule(BaseAttentionModule):
    """Cross-attention mechanism for attending between different modalities"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Query from one modality, Key/Value from another
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Cross-attention specific parameters
        self.cross_attention_freq = config.get("cross_attention_freq", 2)
        self.save_attention = config.get("save_attention", False)
        self.attention_map = None
    
    def forward(
        self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        query_attention_mask: Optional[torch.Tensor] = None,
        kv_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for cross-attention
        
        Args:
            query_states: Query embeddings from one modality [batch_size, query_len, hidden_size]
            key_value_states: Key/Value embeddings from another modality [batch_size, kv_len, hidden_size]
            query_attention_mask: Mask for query sequence
            kv_attention_mask: Mask for key/value sequence
            output_attentions: Whether to return attention weights
        """
        batch_size, query_len, hidden_size = query_states.shape
        kv_len = key_value_states.shape[1]
        
        # Compute query, key, value
        query_layer = self.query(query_states)
        key_layer = self.key(key_value_states)
        value_layer = self.value(key_value_states)
        
        # Reshape for multi-head attention
        query_layer = self._transpose_for_scores(query_layer)
        key_layer = self._transpose_for_scores(key_layer)
        value_layer = self._transpose_for_scores(value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if kv_attention_mask is not None:
            # Create cross-attention mask
            extended_attention_mask = kv_attention_mask.unsqueeze(1).unsqueeze(1)
            extended_attention_mask = extended_attention_mask.expand(
                batch_size, self.num_attention_heads, query_len, kv_len
            )
            attention_scores = attention_scores + (extended_attention_mask * -10000.0)
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Save attention map if requested
        if self.save_attention:
            self.attention_map = attention_probs.detach()
        
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Output projection
        attention_output = self.output_projection(context_layer)
        
        # Residual connection and layer norm
        attention_output = self.layer_norm(attention_output + query_states)
        
        if output_attentions:
            return attention_output, attention_probs
        return attention_output
    
    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose tensor for multi-head attention computation"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def get_attention_map(self) -> Optional[torch.Tensor]:
        """Get saved attention map"""
        return self.attention_map

class QFormerAttention(BaseAttentionModule):
    """Q-Former style attention mechanism (similar to BLIP-2)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_query_tokens = config.get("num_query_tokens", 32)
        self.vision_width = config.get("vision_width", 768)
        
        # Initialize query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query_tokens, self.hidden_size)
        )
        nn.init.normal_(self.query_tokens, mean=0.0, std=0.02)
        
        # Self-attention for query tokens
        self.self_attention = SelfAttentionModule(config)
        
        # Cross-attention between queries and vision features
        self.cross_attention = CrossAttentionModule(config)
        
        # Feed-forward network
        self.intermediate = nn.Linear(self.hidden_size, config.get("intermediate_size", 3072))
        self.output = nn.Linear(config.get("intermediate_size", 3072), self.hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(self.hidden_size)
    
    def forward(
        self,
        vision_embeddings: torch.Tensor,
        vision_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for Q-Former attention
        
        Args:
            vision_embeddings: Vision embeddings [batch_size, num_patches, vision_width]
            vision_attention_mask: Mask for vision embeddings
            output_attentions: Whether to return attention weights
        """
        batch_size = vision_embeddings.shape[0]
        
        # Expand query tokens for batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Self-attention on query tokens
        query_output = self.self_attention(query_tokens)
        
        # Cross-attention between queries and vision features
        cross_attention_output = self.cross_attention(
            query_states=query_output,
            key_value_states=vision_embeddings,
            kv_attention_mask=vision_attention_mask,
            output_attentions=output_attentions
        )
        
        if output_attentions:
            cross_attention_output, attention_weights = cross_attention_output
        
        # Feed-forward network
        intermediate_output = self.intermediate(cross_attention_output)
        intermediate_output = self.activation(intermediate_output)
        final_output = self.output(intermediate_output)
        
        # Residual connection and layer norm
        final_output = self.layer_norm(final_output + cross_attention_output)
        
        if output_attentions:
            return final_output, attention_weights
        return final_output

class AttentionManager:
    """Manager for creating and managing attention mechanisms"""
    
    _attention_registry = {
        "self_attention": SelfAttentionModule,
        "cross_attention": CrossAttentionModule,
        "qformer": QFormerAttention
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.attention_modules = {}
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize attention modules based on config
        self._initialize_attention_modules()
    
    def _initialize_attention_modules(self):
        """Initialize attention modules based on configuration"""
        attention_configs = self.config.get("attention_modules", {})
        
        for module_name, attention_config in attention_configs.items():
            attention_type = attention_config.get("type", "self_attention")
            
            if attention_type in self._attention_registry:
                attention_class = self._attention_registry[attention_type]
                attention_module = attention_class(attention_config)
                attention_module.to(self.device)
                self.attention_modules[module_name] = attention_module
            else:
                logger.warning(f"Unknown attention type: {attention_type}")
    
    def get_attention_module(self, module_name: str) -> BaseAttentionModule:
        """Get attention module by name"""
        if module_name not in self.attention_modules:
            # Create default self-attention module
            default_config = {"type": "self_attention", "hidden_size": 768}
            attention_module = SelfAttentionModule(default_config)
            attention_module.to(self.device)
            self.attention_modules[module_name] = attention_module
        
        return self.attention_modules[module_name]
    
    def apply_self_attention(
        self, 
        embeddings: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        module_name: str = "default_self_attention"
    ) -> torch.Tensor:
        """Apply self-attention to embeddings"""
        attention_module = self.get_attention_module(module_name)
        
        if isinstance(attention_module, SelfAttentionModule):
            return attention_module(embeddings, attention_mask)
        else:
            logger.warning(f"Module {module_name} is not a self-attention module")
            return embeddings
    
    def apply_cross_attention(
        self,
        query_embeddings: torch.Tensor,
        key_value_embeddings: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
        module_name: str = "default_cross_attention"
    ) -> torch.Tensor:
        """Apply cross-attention between different modalities"""
        attention_module = self.get_attention_module(module_name)
        
        if isinstance(attention_module, (CrossAttentionModule, QFormerAttention)):
            if isinstance(attention_module, QFormerAttention):
                # For Q-Former, key_value_embeddings are vision embeddings
                return attention_module(key_value_embeddings, kv_mask)
            else:
                return attention_module(query_embeddings, key_value_embeddings, query_mask, kv_mask)
        else:
            logger.warning(f"Module {module_name} is not a cross-attention module")
            return query_embeddings
    
    def create_multimodal_features(
        self,
        text_embeddings: torch.Tensor,
        vision_embeddings: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Create multimodal features using attention mechanisms"""
        
        # Apply self-attention to each modality
        attended_text = self.apply_self_attention(
            text_embeddings, text_mask, "text_self_attention"
        )
        attended_vision = self.apply_self_attention(
            vision_embeddings, vision_mask, "vision_self_attention"
        )
        
        # Apply cross-attention between modalities
        text_to_vision = self.apply_cross_attention(
            attended_text, attended_vision, text_mask, vision_mask, "text_to_vision_attention"
        )
        vision_to_text = self.apply_cross_attention(
            attended_vision, attended_text, vision_mask, text_mask, "vision_to_text_attention"
        )
        
        # Use Q-Former style attention if configured
        if "qformer" in self.attention_modules:
            qformer_output = self.apply_cross_attention(
                None, attended_vision, None, vision_mask, "qformer"
            )
            
            return {
                "text_features": attended_text,
                "vision_features": attended_vision,
                "text_to_vision": text_to_vision,
                "vision_to_text": vision_to_text,
                "qformer_features": qformer_output,
                "fused_features": qformer_output  # Use Q-Former output as main fused features
            }
        else:
            # Simple fusion by concatenation or averaging
            fused_features = torch.cat([text_to_vision, vision_to_text], dim=-1)
            
            return {
                "text_features": attended_text,
                "vision_features": attended_vision,
                "text_to_vision": text_to_vision,
                "vision_to_text": vision_to_text,
                "fused_features": fused_features
            }
    
    @classmethod
    def register_attention_module(cls, name: str, attention_class: type):
        """Register a new attention module type"""
        cls._attention_registry[name] = attention_class
    
    def to(self, device):
        """Move all attention modules to device"""
        self.device = device
        for module in self.attention_modules.values():
            module.to(device)
        return self
    
    def train(self):
        """Set all attention modules to training mode"""
        for module in self.attention_modules.values():
            module.train()
    
    def eval(self):
        """Set all attention modules to evaluation mode"""
        for module in self.attention_modules.values():
            module.eval()