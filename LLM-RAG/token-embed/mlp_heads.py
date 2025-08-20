import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_layers: int = 2, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # Activation function
        if activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
        
        # Build layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:  # No dropout after last layer
                self.layers.append(nn.Dropout(dropout))
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] or [batch_size, input_dim]
        
        Returns:
            output: [batch_size, seq_len, output_dim] or [batch_size, output_dim]
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                if i < len(self.layers) - 1:  # Apply activation except for last layer
                    x = self.activation(x)
            else:  # Dropout layer
                x = layer(x)
        
        return self.layer_norm(x)


class ResponseGenerationHead(nn.Module):
    """Response generation head for XrayGPT-like text generation."""
    
    def __init__(self, d_model: int, vocab_size: int, hidden_dim: Optional[int] = None,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = d_model * 4  # Standard transformer FFN ratio
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Pre-processing layers
        self.pre_norm = nn.LayerNorm(d_model)
        
        # MLP for feature transformation
        self.feature_mlp = MLP(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=d_model,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Temperature scaling for generation
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, fused_features: torch.Tensor, 
                temperature: Optional[float] = None) -> torch.Tensor:
        """
        Args:
            fused_features: [batch_size, seq_len, d_model]
            temperature: Optional temperature for scaling logits
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Normalize input features
        x = self.pre_norm(fused_features)
        
        # Transform through MLP
        x = self.feature_mlp(x)
        
        # Project to vocabulary space
        logits = self.output_projection(x)
        
        # Apply temperature scaling
        if temperature is not None:
            logits = logits / temperature
        else:
            logits = logits / self.temperature
        
        return logits
    
    def generate_response(self, fused_features: torch.Tensor, 
                         tokenizer, max_length: int = 100,
                         temperature: float = 1.0, top_p: float = 0.9,
                         top_k: int = 50, do_sample: bool = True) -> Dict[str, Any]:
        """
        Generate response using the fused multimodal features.
        
        Args:
            fused_features: [batch_size, seq_len, d_model]
            tokenizer: Tokenizer for decoding
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling or greedy decoding
        
        Returns:
            Dictionary with generated text and metadata
        """
        batch_size = fused_features.size(0)
        device = fused_features.device
        
        # Use the last token's features for generation (or mean pooling)
        if fused_features.size(1) > 1:
            # Use last token features
            generation_features = fused_features[:, -1:, :]  # [batch_size, 1, d_model]
        else:
            generation_features = fused_features
        
        generated_ids = []
        generated_logits = []
        
        # Start with a special token or empty
        current_features = generation_features
        
        for step in range(max_length):
            # Get logits for current step
            logits = self.forward(current_features, temperature)  # [batch_size, 1, vocab_size]
            logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            generated_logits.append(logits)
            
            # Sample next token
            if do_sample:
                # Apply top-k and top-p filtering
                filtered_logits = self._top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_ids.append(next_token)
            
            # Check for end of sequence
            if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                if (next_token == tokenizer.eos_token_id).all():
                    break
        
        # Concatenate generated tokens
        if generated_ids:
            generated_sequence = torch.cat(generated_ids, dim=1)  # [batch_size, gen_length]
        else:
            generated_sequence = torch.empty(batch_size, 0, dtype=torch.long, device=device)
        
        # Decode to text
        generated_texts = []
        for i in range(batch_size):
            tokens = generated_sequence[i].cpu().tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            generated_texts.append(text)
        
        return {
            'generated_text': generated_texts,
            'generated_ids': generated_sequence,
            'logits': torch.stack(generated_logits, dim=1) if generated_logits else None,
            'generation_length': len(generated_ids)
        }
    
    def _top_k_top_p_filtering(self, logits: torch.Tensor, top_k: int = 0, 
                              top_p: float = 1.0, filter_value: float = -float('Inf')) -> torch.Tensor:
        """Filter logits using top-k and/or nucleus (top-p) sampling."""
        top_k = min(top_k, logits.size(-1))  # Safety check
        
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        
        return logits


class ClassificationHead(nn.Module):
    """Classification head for tasks like medical diagnosis classification."""
    
    def __init__(self, d_model: int, num_classes: int, hidden_dim: Optional[int] = None,
                 num_layers: int = 2, dropout: float = 0.1, pooling: str = 'mean'):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = d_model // 2
        
        self.pooling = pooling
        self.num_classes = num_classes
        
        # Pooling layer
        if pooling == 'attention':
            self.attention_pool = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Classification MLP
        self.classifier = MLP(
            input_dim=d_model,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, fused_features: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            fused_features: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] - padding mask
        
        Returns:
            class_logits: [batch_size, num_classes]
        """
        batch_size = fused_features.size(0)
        
        if self.pooling == 'mean':
            # Mean pooling
            if mask is not None:
                lengths = mask.sum(dim=1, keepdim=True).float()
                pooled = (fused_features * mask.unsqueeze(-1)).sum(dim=1) / lengths
            else:
                pooled = fused_features.mean(dim=1)
                
        elif self.pooling == 'max':
            # Max pooling
            if mask is not None:
                fused_features = fused_features.masked_fill(~mask.unsqueeze(-1), -float('inf'))
            pooled = fused_features.max(dim=1)[0]
            
        elif self.pooling == 'cls':
            # Use first token (CLS token)
            pooled = fused_features[:, 0, :]
            
        elif self.pooling == 'attention':
            # Attention pooling
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            pooled, _ = self.attention_pool(cls_tokens, fused_features, fused_features, 
                                         key_padding_mask=~mask if mask is not None else None)
            pooled = pooled.squeeze(1)
        
        # Apply classifier
        class_logits = self.classifier(pooled)
        
        return class_logits


class MultiTaskHead(nn.Module):
    """Multi-task head combining generation and classification."""
    
    def __init__(self, d_model: int, vocab_size: int, num_classes: int,
                 generation_hidden_dim: Optional[int] = None,
                 classification_hidden_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        # Generation head
        self.generation_head = ResponseGenerationHead(
            d_model=d_model,
            vocab_size=vocab_size,
            hidden_dim=generation_hidden_dim,
            dropout=dropout
        )
        
        # Classification head
        self.classification_head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            hidden_dim=classification_hidden_dim,
            dropout=dropout
        )
        
        # Task-specific feature projections
        self.generation_proj = nn.Linear(d_model, d_model)
        self.classification_proj = nn.Linear(d_model, d_model)
        
    def forward(self, fused_features: torch.Tensor, 
                task: str = 'both', mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_features: [batch_size, seq_len, d_model]
            task: 'generation', 'classification', or 'both'
            mask: [batch_size, seq_len] - padding mask
        
        Returns:
            Dictionary with task-specific outputs
        """
        outputs = {}
        
        if task in ['generation', 'both']:
            gen_features = self.generation_proj(fused_features)
            generation_logits = self.generation_head(gen_features)
            outputs['generation_logits'] = generation_logits
        
        if task in ['classification', 'both']:
            cls_features = self.classification_proj(fused_features)
            classification_logits = self.classification_head(cls_features, mask)
            outputs['classification_logits'] = classification_logits
        
        return outputs