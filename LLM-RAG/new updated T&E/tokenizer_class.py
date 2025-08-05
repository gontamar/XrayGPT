"""
Generic Tokenizer Manager
A flexible tokenization system that supports multiple tokenizer types
and configurations through a YAML configuration file.
Includes input validation and preprocessing capabilities.
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
    Includes input validation and preprocessing capabilities.
    """
    
    def __init__(self, config_path: str = "tokenizer_config.yaml", enable_input_processing: bool = True):
        """
        Initialize the TokenizerManager with a configuration file.
        
        Args:
            config_path (str): Path to the YAML configuration file
            enable_input_processing (bool): Whether to enable input validation and preprocessing
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.tokenizer = None
        self.current_config = None
        self.enable_input_processing = enable_input_processing
        
        # Initialize input processor if enabled
        if self.enable_input_processing:
            try:
                from input_processor import InputProcessor
                self.input_processor = InputProcessor(config_path)
            except ImportError:
                logger.warning("Input processor not available.")
                self.input_processor = None
                self.enable_input_processing = False
        else:
            self.input_processor = None
        
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
                validation_profile: str = "default",
                preprocessing_profile: str = "default",
                skip_validation: bool = False,
                skip_preprocessing: bool = False,
                **kwargs) -> Dict[str, Any]:
        """
        Tokenize input text using the loaded tokenizer.
        Includes optional input validation and preprocessing.
        
        Args:
            text (str or List[str]): Input text(s) to tokenize
            return_tensors (bool): Whether to return tensors
            return_attention_mask (bool): Whether to return attention mask
            return_token_type_ids (bool): Whether to return token type IDs
            validation_profile (str): Validation profile to use
            preprocessing_profile (str): Preprocessing profile to use
            skip_validation (bool): Skip input validation
            skip_preprocessing (bool): Skip input preprocessing
            **kwargs: Additional arguments for tokenization
        
        Returns:
            Dict containing tokenization results and validation info
        """
        if self.tokenizer is None:
            logger.warning("No tokenizer loaded. Loading default tokenizer.")
            self.load_tokenizer()
        
        # Initialize result tracking
        validation_results = []
        processed_texts = []
        original_text = text
        
        # Process input validation and preprocessing
        if self.enable_input_processing and self.input_processor and not skip_validation:
            if isinstance(text, str):
                # Single text processing
                if not skip_preprocessing:
                    validation_result = self.input_processor.process_input(
                        text, validation_profile, preprocessing_profile
                    )
                else:
                    validation_result = self.input_processor.validate_input(text, validation_profile)
                
                validation_results.append(validation_result)
                
                if not validation_result.is_valid:
                    logger.warning(f"Input validation failed: {validation_result.errors}")
                    # You can choose to raise an exception or continue with original text
                    # raise ValueError(f"Input validation failed: {validation_result.errors}")
                
                # Use processed text if available, otherwise original
                text = validation_result.processed_text if validation_result.processed_text else text
                processed_texts.append(text)
                
            else:
                # Multiple texts processing
                processed_text_list = []
                for single_text in text:
                    if not skip_preprocessing:
                        validation_result = self.input_processor.process_input(
                            single_text, validation_profile, preprocessing_profile
                        )
                    else:
                        validation_result = self.input_processor.validate_input(single_text, validation_profile)
                    
                    validation_results.append(validation_result)
                    
                    if not validation_result.is_valid:
                        logger.warning(f"Input validation failed for text: {validation_result.errors}")
                    
                    # Use processed text if available, otherwise original
                    processed_text = validation_result.processed_text if validation_result.processed_text else single_text
                    processed_text_list.append(processed_text)
                    processed_texts.append(processed_text)
                
                text = processed_text_list
        else:
            # No processing, use original text
            processed_texts = text if isinstance(text, list) else [text]
        
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
            # Tokenize the (possibly processed) text
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
                'text': text,  # Processed text
                'original_text': original_text,  # Original input
                'processed_texts': processed_texts,
                'input_processing_enabled': self.enable_input_processing,
                'validation_results': validation_results if validation_results else None,
                'validation_profile': validation_profile if self.enable_input_processing else None,
                'preprocessing_profile': preprocessing_profile if self.enable_input_processing else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            raise
