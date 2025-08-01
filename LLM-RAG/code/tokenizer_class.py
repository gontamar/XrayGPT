"""
Generic Tokenizer Manager
A flexible tokenization system that supports multiple tokenizer types
and configurations through a YAML configuration file.
"""

import yaml
import os
from typing import Dict, List, Optional, Union, Any
from transformers import AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tokenizer:
    """
    A generic tokenizer manager that handles different tokenizer configurations
    and provides a unified interface for tokenization operations.
    """
    
    def __init__(self, config_path: str = "tokenizer_config.yaml"):
        """
        Initialize the TokenizerManager with a configuration file.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.tokenizer = None
        self.current_config = None
        
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
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.config = self._load_config()
        logger.info("Configuration reloaded")
    
    def list_available_tokenizers(self) -> List[str]:
        """List all available tokenizer configurations."""
        if self.config is None:
            logger.error("Configuration not loaded")
            return []
        
        tokenizers = list(self.config.get('tokenizers', {}).keys())
        xraygpt_tokenizers = list(self.config.get('xraygpt', {}).keys())
        custom_tokenizers = list(self.config.get('custom_paths', {}).keys())
        
        all_tokenizers = tokenizers + xraygpt_tokenizers + custom_tokenizers
        return sorted(set(all_tokenizers))
    
    def get_tokenizer_info(self, tokenizer_name: str) -> Dict[str, Any]:
        """Get information about a specific tokenizer configuration."""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        # Check in different sections
        for section in ['tokenizers', 'xraygpt', 'custom_paths']:
            if tokenizer_name in self.config.get(section, {}):
                return self.config[section][tokenizer_name]
        
        raise ValueError(f"Tokenizer '{tokenizer_name}' not found in configuration")
    
    def load_tokenizer(self, tokenizer_name: Optional[str] = None, 
                      custom_config: Optional[Dict[str, Any]] = None) -> AutoTokenizer:
        """
        Load a tokenizer based on configuration.
        
        Args:
            tokenizer_name (str, optional): Name of the tokenizer from config.
                                          If None, uses default configuration.
            custom_config (dict, optional): Custom configuration to override defaults.
        
        Returns:
            AutoTokenizer: Loaded tokenizer instance
        """
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
        """
        Tokenize input text using the loaded tokenizer.
        
        Args:
            text (str or List[str]): Input text(s) to tokenize
            return_tensors (bool): Whether to return tensors
            return_attention_mask (bool): Whether to return attention mask
            return_token_type_ids (bool): Whether to return token type IDs
            **kwargs: Additional arguments for tokenization
        
        Returns:
            Dict containing tokenization results
        """
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
    
    def decode(self, token_ids: Union[List[int], List[List[int]]], 
               skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text(s)
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded. Call load_tokenizer() first.")
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the current tokenizer."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded. Call load_tokenizer() first.")
        
        return len(self.tokenizer.vocab)
    
    def get_special_tokens(self) -> Dict[str, str]:
        """Get special tokens of the current tokenizer."""
        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded. Call load_tokenizer() first.")
        
        return self.tokenizer.special_tokens_map


# Convenience functions for quick usage
def create_tokenizer_manager(config_path: str = "tokenizer_config.yaml") -> Tokenizer:
    """Create and return a Tokenizer instance."""
    return Tokenizer(config_path)


def quick_tokenize(text: str, tokenizer_name: str = None, 
                  config_path: str = "tokenizer_config.yaml") -> Dict[str, Any]:
    """
    Quick tokenization function for simple use cases.
    
    Args:
        text (str): Text to tokenize
        tokenizer_name (str, optional): Tokenizer configuration name
        config_path (str): Path to configuration file
    
    Returns:
        Dict containing tokenization results
    """
    manager = Tokenizer(config_path)
    manager.load_tokenizer(tokenizer_name)
    return manager.tokenize(text)