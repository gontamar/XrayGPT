import yaml
import re
import os
from typing import Dict, List, Optional, Union, Any
from transformers import AutoTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Enhanced text tokenizer with preprocessing capabilities.
    Supports BERT and other HuggingFace tokenizers with configurable input processing.
    """
    
    def __init__(self, config_path: str = "tokenizer_config.yaml", enable_input_processing: bool = True):
        """
        Initialize tokenizer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            enable_input_processing: Whether to enable input preprocessing by default
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.tokenizer = None
        self.current_config = None
        self.enable_input_processing = enable_input_processing
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                if config is None:
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
    
    def process_input_text(self, text: Union[str, List[str]], 
                          clean_text: bool = True,
                          normalize_whitespace: bool = True,
                          strip_text: bool = True,
                          lowercase: bool = False,
                          remove_special_chars: bool = False,
                          custom_preprocessing: Optional[callable] = None) -> Union[str, List[str]]:
        """
        Process input text before tokenization.
        
        Args:
            text: Input text string or list of strings
            clean_text: Whether to apply basic text cleaning
            normalize_whitespace: Replace multiple whitespaces with single space
            strip_text: Strip whitespace from beginning and end
            lowercase: Convert text to lowercase
            remove_special_chars: Remove special characters (keeps alphanumeric and basic punctuation)
            custom_preprocessing: Custom preprocessing function to apply
            
        Returns:
            Processed text string or list of strings
        """
        def _process_single_text(single_text: str) -> str:
            processed_text = single_text
            
            if clean_text:
                # Strip whitespace
                if strip_text:
                    processed_text = processed_text.strip()
                
                # Normalize whitespace (replace multiple spaces/tabs/newlines with single space)
                if normalize_whitespace:
                    processed_text = re.sub(r'\s+', ' ', processed_text)
                
                # Convert to lowercase
                if lowercase:
                    processed_text = processed_text.lower()
                
                # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
                if remove_special_chars:
                    processed_text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:\-\'"()]', '', processed_text)
            
            # Apply custom preprocessing if provided
            if custom_preprocessing and callable(custom_preprocessing):
                processed_text = custom_preprocessing(processed_text)
            
            return processed_text.strip()  # Final strip
        
        try:
            # Process based on input type
            if isinstance(text, str):
                processed_text = _process_single_text(text)
                logger.debug(f"Processed single text: '{text[:30]}...' -> '{processed_text[:30]}...'")
                return processed_text
            elif isinstance(text, list):
                processed_texts = [_process_single_text(t) for t in text]
                logger.debug(f"Processed {len(processed_texts)} text inputs")
                return processed_texts
            else:
                raise ValueError(f"Unsupported text input type: {type(text)}")
                
        except Exception as e:
            logger.error(f"Error during text preprocessing: {e}")
            raise
    
    
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
                process_input: bool = None,
                preprocessing_options: Optional[Dict[str, Any]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Tokenize text with optional input preprocessing.
        
        Args:
            text: Input text string or list of strings
            return_tensors: Whether to return tensors
            return_attention_mask: Whether to return attention mask
            return_token_type_ids: Whether to return token type IDs
            process_input: Whether to apply input preprocessing (defaults to self.enable_input_processing)
            preprocessing_options: Dictionary of preprocessing options to override defaults
            **kwargs: Additional tokenization arguments
            
        Returns:
            Dictionary containing tokenization results
        """
        if self.tokenizer is None:
            logger.warning("No tokenizer loaded. Loading default tokenizer.")
            self.load_tokenizer()
        
        # Determine preprocessing settings
        if process_input is None:
            process_input = self.enable_input_processing
        
        # Store original text
        original_text = text
        processed_text = text
        
        # Apply input preprocessing if enabled
        if process_input:
            # Default preprocessing options
            default_preprocessing = {
                'clean_text': True,
                'normalize_whitespace': True,
                'strip_text': True,
                'lowercase': False,
                'remove_special_chars': False,
                'custom_preprocessing': None
            }
            
            # Override with user-provided options
            if preprocessing_options:
                default_preprocessing.update(preprocessing_options)
            
            # Process the input text
            processed_text = self.process_input_text(text, **default_preprocessing)
            logger.debug("Applied input text preprocessing before tokenization")
        
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
            # Tokenize the processed text
            encoded = self.tokenizer(processed_text, **tokenize_kwargs)
            
            # Also get human-readable tokens for debugging
            if isinstance(processed_text, str):
                tokens = self.tokenizer.tokenize(processed_text)
                token_ids = self.tokenizer.encode(processed_text, add_special_tokens=True)
            else:
                tokens = [self.tokenizer.tokenize(t) for t in processed_text]
                token_ids = [self.tokenizer.encode(t, add_special_tokens=True) for t in processed_text]
            
            result = {
                'encoded': encoded,
                'tokens': tokens,
                'token_ids': token_ids,
                'text': processed_text,
                'original_text': original_text,
                'preprocessing_applied': process_input
            }
            
            # Add preprocessing details if applied
            if process_input:
                result['preprocessing_options'] = default_preprocessing if preprocessing_options else default_preprocessing
            
            return result
            
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            raise
