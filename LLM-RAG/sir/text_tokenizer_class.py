import yaml
import os
from typing import Dict, List, Optional, Union, Any
from transformers import AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tokenizer:    
    def __init__(self, config_path: str = "tokenizer_config.yaml", enable_input_processing: bool = True):
        self.config_path = config_path
        self.config = self._load_config()
        self.tokenizer = None
        self.current_config = None
        self.enable_input_processing = enable_input_processing
        
        # Input processing disabled
        self.input_processor = None
        self.enable_input_processing = False
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                if config is None:
                    logger.error(f"Configuration file {self.config_path} is empty or invalid")
                    raise ValueError(f"Configuration file {self.config_path} is empty or invalid")
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")

            raise
    
    def get_tokenizer_info(self, tokenizer_name: str) -> Dict[str, Any]:
        """Get tokenizer configuration."""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        # Check in different sections
        for section in ['tokenizers', 'xraygpt', 'custom_paths']:
            if tokenizer_name in self.config.get(section, {}):
                return self.config[section][tokenizer_name]
        
        raise ValueError(f"Tokenizer '{tokenizer_name}' not found in configuration")
    
    
    def load_tokenizer(self, tokenizer_name: Optional[str] = None, 
                      custom_config: Optional[Dict[str, Any]] = None) -> AutoTokenizer:
        if custom_config:
            config = custom_config
        elif tokenizer_name:
            config = self.get_tokenizer_info(tokenizer_name)
        else:
            # Use default configuration
            default_type = self.config['default']['type']
            config = self.config['tokenizers'][default_type]
        
        # Handle custom paths
        if tokenizer_name in self.config.get('custom_paths', {}):
            model_name_or_path = self.config['custom_paths'][tokenizer_name]
        else:
            model_name_or_path = config['model_name']
        
        try:
            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Add special tokens if specified
            special_tokens = config.get('special_tokens', {})
            if special_tokens:
                tokenizer.add_special_tokens(special_tokens)
                logger.info(f"Added special tokens: {special_tokens}")
            
            self.tokenizer = tokenizer
            self.current_config = config
            
            logger.info(f"Tokenizer loaded: {model_name_or_path}")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error loading tokenizer {model_name_or_path}: {e}")
            raise
    
    def tokenize(self, text: Union[str, List[str]], 
                return_tensors: bool = True,
                return_attention_mask: bool = True,
                return_token_type_ids: bool = False,
                **kwargs) -> Dict[str, Any]:
      
        if self.tokenizer is None:
            logger.warning("No tokenizer loaded. Loading default tokenizer.")
            self.load_tokenizer()
        
        # Merge configuration with provided arguments
        tokenize_kwargs = {
            'padding': self.current_config.get('padding', True),
            'truncation': self.current_config.get('truncation', True),
            'max_length': self.current_config.get('max_length', 512),
            'return_tensors': self.current_config.get('return_tensors', 'pt') if return_tensors else None,
            'return_attention_mask': return_attention_mask,
            'return_token_type_ids': return_token_type_ids,
        }
        
        # Override with any provided kwargs
        tokenize_kwargs.update(kwargs)
        
        try:
            # Tokenize the text
            encoded = self.tokenizer(text, **tokenize_kwargs)
            
            # Also get human-readable tokens for debugging
            if isinstance(text, str):
                tokens = self.tokenizer.tokenize(text)
                token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            else:
                tokens = [self.tokenizer.tokenize(t) for t in text]
                token_ids = [self.tokenizer.encode(t, add_special_tokens=True) for t in text]
            
            result = {
                'encoded': encoded,
                'tokens': tokens,
                'token_ids': token_ids,
                'text': text
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            raise
