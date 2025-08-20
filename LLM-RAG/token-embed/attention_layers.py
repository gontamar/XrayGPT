import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import logging
from transformers.activations import ACT2FN

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism for self-attention and cross-attention."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: [batch_size, seq_len_q, seq_len_k] or broadcastable
        
        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_k = key.size(1)
        
        # Linear projections and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        
        output = self.w_o(attention_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q, K, V: [batch_size, num_heads, seq_len, d_k]
            mask: [batch_size, 1, seq_len_q, seq_len_k] or broadcastable
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)  # Add head dimension
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class SelfAttentionLayer(nn.Module):
    """Self-attention layer with residual connection and layer norm."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or broadcastable
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        output = self.layer_norm(x + self.dropout(attn_output))
        
        return output, attention_weights


class CrossModalAttention(nn.Module):
    """Cross-modal attention for vision-language interaction (similar to XrayGPT)."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_kv = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Additional projection layers for cross-modal alignment
        self.vision_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)
        
    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None,
                vision_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention: text queries attend to vision keys/values
        
        Args:
            text_features: [batch_size, text_seq_len, d_model]
            vision_features: [batch_size, vision_seq_len, d_model] (e.g., patch embeddings)
            text_mask: [batch_size, text_seq_len]
            vision_mask: [batch_size, vision_seq_len]
        
        Returns:
            enhanced_text: [batch_size, text_seq_len, d_model]
            cross_attention_weights: [batch_size, num_heads, text_seq_len, vision_seq_len]
        """
        # Project features for better cross-modal alignment
        text_proj = self.text_proj(text_features)
        vision_proj = self.vision_proj(vision_features)
        
        # Normalize inputs
        text_norm = self.layer_norm_q(text_proj)
        vision_norm = self.layer_norm_kv(vision_proj)
        
        # Create cross-attention mask if needed
        cross_mask = None
        if text_mask is not None and vision_mask is not None:
            # [batch_size, text_seq_len, 1] * [batch_size, 1, vision_seq_len]
            cross_mask = text_mask.unsqueeze(-1) * vision_mask.unsqueeze(1)
        
        # Cross-modal attention: text attends to vision
        attn_output, attention_weights = self.attention(
            query=text_norm,
            key=vision_norm,
            value=vision_norm,
            mask=cross_mask
        )
        
        # Residual connection
        enhanced_text = text_features + self.dropout(attn_output)
        
        return enhanced_text, attention_weights


class VisionLanguageFusion(nn.Module):
    """Vision-Language fusion module combining self-attention and cross-modal attention."""
    
    def __init__(self, d_model: int, num_heads: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Self-attention layers for each modality
        self.text_self_attention = nn.ModuleList([
            SelfAttentionLayer(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.vision_self_attention = nn.ModuleList([
            SelfAttentionLayer(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Cross-modal attention layers
        self.cross_modal_attention = nn.ModuleList([
            CrossModalAttention(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Final fusion layer
        self.fusion_layer = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        
    def forward(self, text_features: torch.Tensor, vision_features: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None,
                vision_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_features: [batch_size, text_seq_len, d_model]
            vision_features: [batch_size, vision_seq_len, d_model]
            text_mask: [batch_size, text_seq_len]
            vision_mask: [batch_size, vision_seq_len]
        
        Returns:
            Dictionary containing:
                - fused_features: [batch_size, text_seq_len, d_model]
                - text_features: Enhanced text features
                - vision_features: Enhanced vision features
                - attention_weights: Cross-modal attention weights
        """
        current_text = text_features
        current_vision = vision_features
        all_attention_weights = []
        
        # Apply self-attention and cross-modal attention layers
        for i in range(self.num_layers):
            # Self-attention for text
            current_text, text_self_attn = self.text_self_attention[i](current_text, text_mask)
            
            # Self-attention for vision
            current_vision, vision_self_attn = self.vision_self_attention[i](current_vision, vision_mask)
            
            # Cross-modal attention: text attends to vision
            current_text, cross_attn_weights = self.cross_modal_attention[i](
                current_text, current_vision, text_mask, vision_mask
            )
            
            all_attention_weights.append(cross_attn_weights)
        
        # Global vision feature (average pooling)
        if vision_mask is not None:
            vision_lengths = vision_mask.sum(dim=1, keepdim=True).float()
            global_vision = (current_vision * vision_mask.unsqueeze(-1)).sum(dim=1) / vision_lengths
        else:
            global_vision = current_vision.mean(dim=1)  # [batch_size, d_model]
        
        # Expand global vision to match text sequence length
        global_vision_expanded = global_vision.unsqueeze(1).expand(-1, current_text.size(1), -1)
        
        # Concatenate and fuse
        concatenated = torch.cat([current_text, global_vision_expanded], dim=-1)
        fused_features = self.fusion_norm(self.fusion_layer(concatenated))
        
        return {
            'fused_features': fused_features,
            'text_features': current_text,
            'vision_features': current_vision,
            'attention_weights': all_attention_weights,
            'global_vision': global_vision
        }


# Enhanced BERT Attention Components
class BertSelfAttention(nn.Module):
    """Enhanced BERT Self-Attention with cross-attention support."""
    
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            # For cross-attention, key and value come from encoder (vision features)
            encoder_width = getattr(config, 'encoder_width', config.hidden_size)
            self.key = nn.Linear(encoder_width, self.all_head_size)
            self.value = nn.Linear(encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    """BERT Self-Attention Output layer."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """Enhanced BERT Attention with cross-attention support."""
    def __init__(self, config, is_cross_attention=False):
        super().__init__()
        self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    """BERT Intermediate (Feed-Forward) layer."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """BERT Output layer."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """Enhanced BERT Layer with cross-attention support."""
    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = getattr(config, 'chunk_size_feed_forward', 0)
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num
        
        # Add cross-attention based on frequency
        cross_attention_freq = getattr(config, 'cross_attention_freq', 2)
        add_cross_attention = getattr(config, 'add_cross_attention', False)
        
        if add_cross_attention and layer_num % cross_attention_freq == 0:
            self.crossattention = BertAttention(config, is_cross_attention=True)
            self.has_cross_attention = True
        else:
            self.has_cross_attention = False
            
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        # Query-specific layers for cross-attention
        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # Self-attention
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        if query_length > 0:
            query_attention_output = attention_output[:, :query_length, :]

            if self.has_cross_attention:
                assert encoder_hidden_states is not None, "encoder_hidden_states must be given for cross-attention layers"
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                query_attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            layer_output = self.feed_forward_chunk_query(query_attention_output)
            if attention_output.shape[1] > query_length:
                layer_output_text = self.feed_forward_chunk(attention_output[:, query_length:, :])
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            layer_output = self.feed_forward_chunk(attention_output)
            
        outputs = (layer_output,) + outputs
        outputs = outputs + (present_key_value,)
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def feed_forward_chunk_query(self, attention_output):
        intermediate_output = self.intermediate_query(attention_output)
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output


# Utility functions for creating attention masks
def create_padding_mask(sequences: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """Create padding mask for sequences."""
    return (sequences != pad_token_id).float()


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal (lower triangular) mask for autoregressive generation."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask