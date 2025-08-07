import torch
from torch import nn
from transformers import BertConfig, AutoConfig
import logging
from typing import Dict, List, Optional, Union, Any
from text_tokenizer_class import Tokenizer

# Set up logging
logger = logging.getLogger(__name__)


class BertEmbeddings(nn.Module):
    """
    BERT embeddings module with word and position embeddings.
    Implements the embedding layer from BERT architecture.
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


class EmbeddingManager:
    """
    Manages text embedding creation using BERT architecture.
    Integrates with tokenizer for end-to-end text processing.
    """
    
    def __init__(self, tokenizer_config_path: str = "tokenizer_config.yaml"):
        """
        Initialize embedding manager.
        
        Args:
            tokenizer_config_path: Path to tokenizer configuration file
        """
        self.tokenizer_manager = Tokenizer(tokenizer_config_path, enable_input_processing=False)
        self.embedding_model = None
        self.model_config = None
        self.current_tokenizer_name = None
    
    
    def _ensure_config_compatibility(self, config) -> BertConfig:
        """
        Ensure BERT config has all required attributes with defaults.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            Validated BertConfig object
        """
        # Set default values if missing for BERT
        if not hasattr(config, 'hidden_dropout_prob'):
            config.hidden_dropout_prob = 0.1
        
        if not hasattr(config, 'layer_norm_eps'):
            config.layer_norm_eps = 1e-12
        
        if not hasattr(config, 'max_position_embeddings'):
            config.max_position_embeddings = 512
        
        if not hasattr(config, 'pad_token_id'):
            config.pad_token_id = 0  # BERT default
        
        return config
    
        
    def load_embedding_model(
        self, 
        tokenizer_name: str = "bert", 
        model_config: Optional[Union[str, Dict]] = None,
        device: str = "cpu"
    ) -> None:
        
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
                # Use provided config dict - always use BERT config
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
            
            # Ensure config compatibility
            config = self._ensure_config_compatibility(config)
            
            # Create BERT embedding model
            self.embedding_model = BertEmbeddings(config).to(device)
            self.model_config = config
            
            logger.info(f"BERT embedding model loaded for tokenizer: {tokenizer_name}")
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
