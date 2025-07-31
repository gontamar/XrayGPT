"""
Generic Embedding Manager
A flexible embedding system that works with the tokenizer system
and supports different embedding models (BERT, GPT, RoBERTa, LLaMA, etc.).
"""

import torch
from torch import nn
from transformers import BertConfig, AutoConfig, RobertaConfig, GPT2Config
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from tokenizer_class import Tokenizer

# Set up logging
logger = logging.getLogger(__name__)


class BertEmbeddings(nn.Module):
    """
    BERT Embeddings module based on XrayGPT implementation.
    Supports word embeddings, position embeddings, and layer normalization.
    """
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=getattr(config, 'pad_token_id', 0)
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Register position_ids buffer
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.config = config

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        query_embeds: Optional[torch.Tensor] = None, 
        past_key_values_length: int = 0
    ) -> torch.Tensor:
        """Forward pass for BERT embeddings."""
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ].clone()

        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings

            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaEmbeddings(nn.Module):
    """
    RoBERTa Embeddings module.
    Similar to BERT but with different position embedding handling.
    """
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            padding_idx=getattr(config, 'pad_token_id', 1)
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # RoBERTa uses different position embedding offset
        self.padding_idx = getattr(config, 'pad_token_id', 1)
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.config = config

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        past_key_values_length: int = 0
    ) -> torch.Tensor:
        """Forward pass for RoBERTa embeddings."""
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.create_position_ids_from_input_ids(
                input_ids, self.padding_idx, past_key_values_length
            )

        embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def create_position_ids_from_input_ids(self, input_ids, padding_idx, past_key_values_length=0):
        """Create position IDs from input IDs, accounting for padding."""
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx


class GPTEmbeddings(nn.Module):
    """
    GPT-style Embeddings module.
    Uses word embeddings and position embeddings without layer norm.
    """
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        self.config = config

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        past_key_values_length: int = 0
    ) -> torch.Tensor:
        """Forward pass for GPT embeddings."""
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        
        embeddings = self.dropout(embeddings)
        return embeddings


class LlamaEmbeddings(nn.Module):
    """
    LLaMA-style Embeddings module.
    Simple word embeddings without position embeddings (uses RoPE instead).
    """
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0)
        )
        self.config = config

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None, 
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for LLaMA embeddings."""
        embeddings = self.word_embeddings(input_ids)
        return embeddings


class GenericEmbeddings(nn.Module):
    """
    Generic embeddings that can adapt to different architectures.
    Falls back to simple word + position embeddings.
    """
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size,
            padding_idx=getattr(config, 'pad_token_id', 0)
        )
        
        # Optional position embeddings
        if hasattr(config, 'max_position_embeddings') and config.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, 
                config.hidden_size
            )
            self.register_buffer(
                "position_ids", 
                torch.arange(config.max_position_embeddings).expand((1, -1))
            )
            self.use_position_embeddings = True
        else:
            self.use_position_embeddings = False
        
        # Optional layer norm
        if hasattr(config, 'layer_norm_eps'):
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.use_layer_norm = True
        else:
            self.use_layer_norm = False
        
        # Optional dropout
        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.config = config

    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for generic embeddings."""
        embeddings = self.word_embeddings(input_ids)
        
        # Add position embeddings if available
        if self.use_position_embeddings:
            if position_ids is None:
                seq_length = input_ids.size()[1]
                position_ids = self.position_ids[:, :seq_length]
            
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        
        # Apply layer norm if configured
        if self.use_layer_norm:
            embeddings = self.LayerNorm(embeddings)
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        return embeddings


class EmbeddingManager:
    """
    Generic embedding manager that works with different tokenizers and models.
    Provides a unified interface for creating embeddings from text.
    Automatically selects appropriate embedding architecture based on model type.
    """
    
    def __init__(self, tokenizer_config_path: str = "tokenizer_config.yaml"):
        """
        Initialize the EmbeddingManager.
        
        Args:
            tokenizer_config_path: Path to tokenizer configuration file
        """
        self.tokenizer_manager = Tokenizer(tokenizer_config_path)
        self.embedding_model = None
        self.model_config = None
        self.current_tokenizer_name = None
        self.embedding_type = None
        
    def _detect_embedding_type(self, config, model_name: str) -> str:
        """
        Detect the appropriate embedding type based on model configuration.
        
        Args:
            config: Model configuration
            model_name: Model name string
            
        Returns:
            embedding_type: One of 'bert', 'roberta', 'gpt', 'llama', 'generic'
        """
        # Check config type first
        config_type = type(config).__name__.lower()
        
        if 'bert' in config_type:
            return 'bert'
        elif 'roberta' in config_type:
            return 'roberta'
        elif 'gpt' in config_type:
            return 'gpt'
        elif 'llama' in config_type:
            return 'llama'
        
        # Check model name patterns
        model_name_lower = model_name.lower()
        
        if any(pattern in model_name_lower for pattern in ['bert', 'distilbert', 'electra']):
            return 'bert'
        elif any(pattern in model_name_lower for pattern in ['roberta', 'camembert', 'xlm-roberta']):
            return 'roberta'
        elif any(pattern in model_name_lower for pattern in ['gpt', 'gpt2', 'gpt-neo', 'gpt-j']):
            return 'gpt'
        elif any(pattern in model_name_lower for pattern in ['llama', 'alpaca', 'vicuna', 'llama2']):
            return 'llama'
        
        # Default to generic
        return 'generic'
    
    def _create_embedding_model(self, embedding_type: str, config):
        """
        Create the appropriate embedding model based on type.
        
        Args:
            embedding_type: Type of embedding ('bert', 'roberta', 'gpt', 'llama', 'generic')
            config: Model configuration
            
        Returns:
            embedding_model: Appropriate embedding model instance
        """
        if embedding_type == 'bert':
            return BertEmbeddings(config)
        elif embedding_type == 'roberta':
            return RobertaEmbeddings(config)
        elif embedding_type == 'gpt':
            return GPTEmbeddings(config)
        elif embedding_type == 'llama':
            return LlamaEmbeddings(config)
        else:
            return GenericEmbeddings(config)
    
    def _ensure_config_compatibility(self, config, embedding_type: str):
        """
        Ensure config has all required attributes for the embedding type.
        
        Args:
            config: Model configuration
            embedding_type: Type of embedding
            
        Returns:
            config: Updated configuration
        """
        # Set default values if missing
        if not hasattr(config, 'hidden_dropout_prob'):
            config.hidden_dropout_prob = 0.1
        
        if not hasattr(config, 'layer_norm_eps'):
            config.layer_norm_eps = 1e-12
        
        if not hasattr(config, 'max_position_embeddings'):
            config.max_position_embeddings = 512
        
        if not hasattr(config, 'pad_token_id'):
            if embedding_type == 'roberta':
                config.pad_token_id = 1  # RoBERTa default
            else:
                config.pad_token_id = 0  # BERT/GPT default
        
        return config
        
    def load_embedding_model(
        self, 
        tokenizer_name: str = "bert", 
        model_config: Optional[Union[str, Dict]] = None,
        embedding_type: Optional[str] = None,
        device: str = "cpu"
    ) -> None:
        """
        Load embedding model for a specific tokenizer.
        
        Args:
            tokenizer_name: Name of tokenizer configuration to use
            model_config: Model configuration (config name, dict, or None for auto)
            embedding_type: Force specific embedding type ('bert', 'roberta', 'gpt', 'llama', 'generic')
            device: Device to load model on
        """
        # Load the tokenizer
        self.tokenizer_manager.load_tokenizer(tokenizer_name)
        self.current_tokenizer_name = tokenizer_name
        
        # Get tokenizer info for model configuration
        tokenizer_info = self.tokenizer_manager.get_tokenizer_info(tokenizer_name)
        model_name = tokenizer_info['model_name']
        
        try:
            # Load model configuration
            if isinstance(model_config, str):
                # Load config from model name
                config = AutoConfig.from_pretrained(model_config)
            elif isinstance(model_config, dict):
                # Use provided config dict - determine appropriate config class
                if embedding_type == 'roberta':
                    config = RobertaConfig(**model_config)
                elif embedding_type == 'gpt':
                    config = GPT2Config(**model_config)
                else:
                    config = BertConfig(**model_config)
            else:
                # Auto-detect config from model name
                try:
                    config = AutoConfig.from_pretrained(model_name)
                except:
                    # Fallback to BERT config
                    logger.warning(f"Could not load config for {model_name}, using BERT config")
                    config = BertConfig()
            
            # Update vocab size to match tokenizer
            if hasattr(self.tokenizer_manager.tokenizer, 'vocab_size'):
                config.vocab_size = self.tokenizer_manager.tokenizer.vocab_size
            elif hasattr(self.tokenizer_manager.tokenizer, 'vocab'):
                config.vocab_size = len(self.tokenizer_manager.tokenizer.vocab)
            
            # Detect embedding type if not specified
            if embedding_type is None:
                embedding_type = self._detect_embedding_type(config, model_name)
            
            self.embedding_type = embedding_type
            
            # Ensure config compatibility
            config = self._ensure_config_compatibility(config, embedding_type)
            
            # Create appropriate embedding model
            self.embedding_model = self._create_embedding_model(embedding_type, config).to(device)
            self.model_config = config
            
            logger.info(f"Embedding model loaded for tokenizer: {tokenizer_name}")
            logger.info(f"Embedding type: {embedding_type}")
            logger.info(f"Model config: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def create_embeddings(
        self, 
        text: Union[str, List[str]], 
        return_tokens: bool = True,
        return_attention_mask: bool = True,
        device: str = "cpu",
        **tokenize_kwargs
    ) -> Dict[str, Any]:
        """
        Create embeddings from text.
        
        Args:
            text: Input text(s) to embed
            return_tokens: Whether to return token information
            return_attention_mask: Whether to return attention mask
            device: Device for computation
            **tokenize_kwargs: Additional tokenization arguments
            
        Returns:
            Dictionary containing embeddings and related information
        """
        if self.embedding_model is None:
            logger.warning("No embedding model loaded. Loading default BERT model.")
            self.load_embedding_model("bert", device=device)
        
        # Tokenize the text
        tokenize_result = self.tokenizer_manager.tokenize(
            text, 
            return_tensors=True,
            return_attention_mask=return_attention_mask,
            **tokenize_kwargs
        )
        
        # Get input_ids tensor
        input_ids = tokenize_result['encoded']['input_ids'].to(device)
        
        # Create embeddings
        with torch.no_grad():
            embeddings = self.embedding_model(input_ids)
        
        result = {
            'embeddings': embeddings,
            'input_ids': input_ids,
            'text': text
        }
        
        # Add optional returns
        if return_tokens:
            result['tokens'] = tokenize_result['tokens']
            result['token_ids'] = tokenize_result['token_ids']
        
        if return_attention_mask and 'attention_mask' in tokenize_result['encoded']:
            result['attention_mask'] = tokenize_result['encoded']['attention_mask'].to(device)
        
        # Add shape information
        result['embedding_shape'] = embeddings.shape
        result['vocab_size'] = self.model_config.vocab_size if self.model_config else None
        result['hidden_size'] = self.model_config.hidden_size if self.model_config else None
        
        return result
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        if self.embedding_model is None:
            return {"status": "No embedding model loaded"}
        
        info = {
            "status": "Model loaded",
            "tokenizer_name": self.current_tokenizer_name,
            "embedding_type": self.embedding_type,
            "vocab_size": self.model_config.vocab_size,
            "hidden_size": self.model_config.hidden_size,
            "model_class": type(self.embedding_model).__name__
        }
        
        # Add optional attributes if they exist
        if hasattr(self.model_config, 'max_position_embeddings'):
            info["max_position_embeddings"] = self.model_config.max_position_embeddings
        
        if hasattr(self.model_config, 'pad_token_id'):
            info["pad_token_id"] = self.model_config.pad_token_id
            
        if hasattr(self.model_config, 'layer_norm_eps'):
            info["layer_norm_eps"] = self.model_config.layer_norm_eps
            
        return info
    
    def list_available_tokenizers(self) -> List[str]:
        """List available tokenizers for embedding."""
        return self.tokenizer_manager.list_available_tokenizers()


# Convenience functions
def create_embedding_manager(config_path: str = "tokenizer_config.yaml") -> EmbeddingManager:
    """Create and return an EmbeddingManager instance."""
    return EmbeddingManager(config_path)
