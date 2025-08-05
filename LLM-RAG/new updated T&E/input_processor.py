"""
Input Processing and Validation System
Handles input validation, cleaning, and preprocessing before tokenization.
Configurable through YAML for different validation rules and preprocessing steps.
"""

import re
import yaml
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import unicodedata
import string

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    original_text: str
    processed_text: Optional[str] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class InputProcessor:
    """
    Input processor that validates and preprocesses text before tokenization.
    Uses YAML configuration for validation rules and preprocessing steps.
    """
    
    # Class variable to cache config and avoid repeated loading
    _config_cache = {}
    
    def __init__(self, config_path: str = "tokenizer_config.yaml"):
        """
        Initialize the InputProcessor.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        
        # Use cached config if available
        if config_path in InputProcessor._config_cache:
            self.config = InputProcessor._config_cache[config_path]
        else:
            self.config = self._load_config()
            InputProcessor._config_cache[config_path] = self.config
            
        self.validation_rules = self.config.get('input_validation', {})
        self.preprocessing_rules = self.config.get('input_preprocessing', {})
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                if config is None:
                    logger.error(f"Configuration file {self.config_path} is empty or invalid")
                    raise ValueError(f"Configuration file {self.config_path} is empty or invalid")
                logger.info(f"Input processing configuration loaded from {self.config_path}")
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
        self.validation_rules = self.config.get('input_validation', {})
        self.preprocessing_rules = self.config.get('input_preprocessing', {})
        logger.info("Input processing configuration reloaded")
    
    def _check_length_constraints(self, text: str) -> List[str]:
        """Check text length constraints."""
        errors = []
        length_rules = self.validation_rules.get('length', {})
        
        min_length = length_rules.get('min_length', 0)
        max_length = length_rules.get('max_length', float('inf'))
        
        if len(text) < min_length:
            errors.append(f"Text too short: {len(text)} < {min_length} characters")
        
        if len(text) > max_length:
            errors.append(f"Text too long: {len(text)} > {max_length} characters")
            
        return errors
    
    def _check_forbidden_characters(self, text: str) -> List[str]:
        """Check for forbidden characters."""
        errors = []
        char_rules = self.validation_rules.get('characters', {})
        
        # Check forbidden characters
        forbidden_chars = char_rules.get('forbidden_characters', [])
        for char in forbidden_chars:
            if char in text:
                errors.append(f"Forbidden character found: '{char}'")
        
        # Check forbidden patterns (regex)
        forbidden_patterns = char_rules.get('forbidden_patterns', [])
        for pattern in forbidden_patterns:
            if re.search(pattern, text):
                errors.append(f"Forbidden pattern found: '{pattern}'")
        
        # Check required character sets
        allowed_chars = char_rules.get('allowed_characters', None)
        if allowed_chars:
            for char in text:
                if char not in allowed_chars:
                    errors.append(f"Character not in allowed set: '{char}'")
                    break  # Don't spam with every invalid character
        
        return errors
    
    def _check_content_rules(self, text: str) -> Tuple[List[str], List[str]]:
        """Check content-based validation rules."""
        errors = []
        warnings = []
        content_rules = self.validation_rules.get('content', {})
        
        # Check for valid text format
        if self._is_invalid_format(text):
            errors.append("Invalid text format detected")
        
        # Check for required patterns
        required_patterns = content_rules.get('required_patterns', [])
        for pattern in required_patterns:
            if not re.search(pattern, text):
                errors.append(f"Required pattern not found: '{pattern}'")
        
        # Check for warning patterns
        warning_patterns = content_rules.get('warning_patterns', [])
        for pattern in warning_patterns:
            if re.search(pattern, text):
                warnings.append(f"Warning pattern found: '{pattern}'")
        
        # Check language constraints
        language_rules = content_rules.get('language', {})
        if language_rules.get('ascii_only', False):
            if not all(ord(char) < 128 for char in text):
                errors.append("Non-ASCII characters found (ASCII-only mode)")
        
        # Check for excessive repetition
        max_repetition = content_rules.get('max_character_repetition', 0)
        if max_repetition > 0:
            for char in set(text):
                if char * (max_repetition + 1) in text:
                    errors.append(f"Excessive character repetition: '{char}' repeated more than {max_repetition} times")
        
        return errors, warnings
    
    def _is_invalid_format(self, text: str) -> bool:
        """Check if text has invalid format patterns."""
        # Check for excessive repetition of same character (more than 5)
        for char in set(text):
            if char * 6 in text:
                return True
        
        # Check for meaningful content
        if not self._has_meaningful_content(text):
            return True
        
        # Check for excessive special characters (more than 20% of text)
        special_char_count = len(re.findall(r'[^a-zA-Z0-9\s.,!?-]', text))
        if len(text) > 0 and special_char_count > len(text) * 0.2:
            return True
        
        return False
    
    def _has_meaningful_content(self, text: str) -> bool:
        """Check if text contains meaningful sentences or valid content."""
        text = text.strip()
        
        # Check minimum length for meaningful content
        if len(text) < 2:
            return False
        
        # Check if it's a valid number format
        if self._is_valid_number_format(text):
            return True
        
        # Check for meaningful words (at least 50% should be dictionary-like words)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return False
        
        meaningful_words = 0
        for word in words:
            # Consider words meaningful if they:
            # - Are common English words (basic check)
            # - Have reasonable vowel/consonant distribution
            # - Are not just random letters
            if self._is_meaningful_word(word):
                meaningful_words += 1
        
        # At least 60% of words should be meaningful
        if len(words) > 0 and meaningful_words / len(words) >= 0.6:
            return True
        
        # Check if it contains sentence-like structure
        if self._has_sentence_structure(text):
            return True
        
        return False
    
    def _is_valid_number_format(self, text: str) -> bool:
        """Check if text is a valid number format."""
        # Remove whitespace
        text = text.strip()
        
        # Integer
        if re.match(r'^-?\d+$', text):
            return True
        
        # Float/Decimal
        if re.match(r'^-?\d+\.\d+$', text):
            return True
        
        # Scientific notation
        if re.match(r'^-?\d+\.?\d*[eE][+-]?\d+$', text):
            return True
        
        # Percentage
        if re.match(r'^\d+\.?\d*%$', text):
            return True
        
        # Currency (basic formats)
        if re.match(r'^[\$€£¥]\d+\.?\d*$', text):
            return True
        
        # Phone number formats
        if re.match(r'^[\+]?[\d\s\-\(\)]{10,15}$', text):
            return True
        
        # Date formats (basic)
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', text):
            return True
        
        return False
    
    def _is_meaningful_word(self, word: str) -> bool:
        """Check if a word appears to be meaningful."""
        word = word.lower()
        
        # Too short
        if len(word) < 2:
            return False
        
        # Allow longer words (up to 30 characters for technical/medical terms)
        if len(word) > 30:
            return False
        
        # Common English words (basic list)
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
            'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up',
            'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time',
            'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could',
            'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think',
            'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even',
            'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been',
            'has', 'had', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'if', 'up', 'out',
            'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'two',
            'more', 'very', 'what', 'know', 'just', 'first', 'get', 'over', 'think', 'also', 'your', 'work',
            'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through',
            'when', 'where', 'much', 'go', 'me', 'back', 'with', 'well', 'were', 'been', 'have', 'there', 'who',
            'patient', 'medical', 'doctor', 'hospital', 'treatment', 'diagnosis', 'symptoms', 'medicine', 'health',
            'report', 'test', 'result', 'analysis', 'data', 'information', 'system', 'process', 'method', 'study',
            'temperature', 'blood', 'pressure', 'heart', 'lung', 'brain', 'chest', 'examination', 'normal',
            'abnormal', 'condition', 'disease', 'therapy', 'medication', 'clinical', 'research', 'science',
            'antidisestablishmentarianism', 'pneumonia', 'cardiovascular', 'respiratory', 'neurological'
        }
        
        if word in common_words:
            return True
        
        # Check vowel/consonant distribution (meaningful words usually have vowels)
        vowels = len(re.findall(r'[aeiou]', word))
        consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', word))
        
        # Must have at least one vowel for words longer than 3 characters
        if vowels == 0 and len(word) > 3:
            return False
        
        # Must have at least one consonant for words longer than 2 characters
        if consonants == 0 and len(word) > 2:
            return False
        
        # Check for reasonable vowel/consonant ratio
        total_letters = vowels + consonants
        if total_letters > 0:
            vowel_ratio = vowels / total_letters
            # Reasonable vowel ratio (10% to 70%)
            if vowel_ratio < 0.1 or vowel_ratio > 0.7:
                return False
        
        # Check for patterns that suggest gibberish
        # Reject words with too many consecutive consonants or vowels
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', word):  # 5+ consecutive consonants
            return False
        if re.search(r'[aeiou]{4,}', word):  # 4+ consecutive vowels
            return False
        
        # Check for keyboard patterns (qwerty, asdf, etc.)
        keyboard_patterns = ['qwerty', 'asdf', 'zxcv', 'qaz', 'wsx', 'edc', 'rfv', 'tgb', 'yhn', 'ujm']
        for pattern in keyboard_patterns:
            if pattern in word or pattern[::-1] in word:  # Check reverse too
                return False
        
        # If it's all letters and passes other checks, consider it meaningful
        if re.match(r'^[a-z]+$', word):
            return True
        
        return False
    
    def _has_sentence_structure(self, text: str) -> bool:
        """Check if text has basic sentence structure."""
        # Contains spaces (multiple words)
        if ' ' in text.strip():
            return True
        
        # Contains sentence-ending punctuation
        if re.search(r'[.!?]', text):
            return True
        
        # Contains common sentence patterns
        if re.search(r'\b(is|are|was|were|have|has|had|will|would|can|could|should|may|might)\b', text.lower()):
            return True
        
        return False
    
    def _check_format_rules(self, text: str) -> List[str]:
        """Check format-specific validation rules."""
        errors = []
        format_rules = self.validation_rules.get('format', {})
        
        # Check line count
        max_lines = format_rules.get('max_lines', 0)
        if max_lines > 0:
            line_count = text.count('\n') + 1
            if line_count > max_lines:
                errors.append(f"Too many lines: {line_count} > {max_lines}")
        
        # Check for empty lines
        if not format_rules.get('allow_empty_lines', True):
            if '\n\n' in text or text.startswith('\n') or text.endswith('\n'):
                errors.append("Empty lines not allowed")
        
        # Check for specific format requirements
        if format_rules.get('must_start_with_capital', False):
            if text and not text[0].isupper():
                errors.append("Text must start with capital letter")
        
        if format_rules.get('must_end_with_punctuation', False):
            if text and text[-1] not in '.!?':
                errors.append("Text must end with punctuation")
        
        return errors
    
    def validate_input(self, text: str, validation_profile: str = "default") -> ValidationResult:
        """
        Validate input text according to configured rules.
        
        Args:
            text: Input text to validate
            validation_profile: Validation profile to use (from config)
            
        Returns:
            ValidationResult object with validation status and details
        """
        if not isinstance(text, str):
            return ValidationResult(
                is_valid=False,
                original_text=str(text),
                errors=["Input must be a string"]
            )
        
        # Use specific validation profile if configured
        if validation_profile != "default" and validation_profile in self.validation_rules:
            # Temporarily switch to specific profile
            original_rules = self.validation_rules
            self.validation_rules = self.validation_rules[validation_profile]
        
        errors = []
        warnings = []
        
        try:
            # Basic checks
            if not text.strip():
                errors.append("Empty or whitespace-only text")
                return ValidationResult(
                    is_valid=False,
                    original_text=text,
                    errors=errors
                )
            
            # Length constraints
            errors.extend(self._check_length_constraints(text))
            
            # Character constraints
            errors.extend(self._check_forbidden_characters(text))
            
            # Content rules
            content_errors, content_warnings = self._check_content_rules(text)
            errors.extend(content_errors)
            warnings.extend(content_warnings)
            
            # Format rules
            errors.extend(self._check_format_rules(text))
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                original_text=text,
                errors=errors,
                warnings=warnings
            )
            
        finally:
            # Restore original rules if we switched profiles
            if validation_profile != "default" and validation_profile in self.config.get('input_validation', {}):
                self.validation_rules = original_rules
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        normalization = self.preprocessing_rules.get('unicode_normalization', 'NFC')
        return unicodedata.normalize(normalization, text)
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace."""
        whitespace_rules = self.preprocessing_rules.get('whitespace', {})
        
        # Remove leading/trailing whitespace
        if whitespace_rules.get('strip', True):
            text = text.strip()
        
        # Normalize internal whitespace
        if whitespace_rules.get('normalize_spaces', True):
            text = re.sub(r'\s+', ' ', text)
        
        # Remove extra newlines
        if whitespace_rules.get('normalize_newlines', True):
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        
        return text
    
    def _apply_character_replacements(self, text: str) -> str:
        """Apply character replacements and cleaning."""
        char_rules = self.preprocessing_rules.get('characters', {})
        
        # Character replacements
        replacements = char_rules.get('replacements', {})
        for old_char, new_char in replacements.items():
            text = text.replace(old_char, new_char)
        
        # Pattern replacements (regex)
        pattern_replacements = char_rules.get('pattern_replacements', {})
        for pattern, replacement in pattern_replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove specific characters
        remove_chars = char_rules.get('remove_characters', [])
        for char in remove_chars:
            text = text.replace(char, '')
        
        return text
    
    def _apply_case_normalization(self, text: str) -> str:
        """Apply case normalization."""
        case_rules = self.preprocessing_rules.get('case', {})
        
        case_mode = case_rules.get('normalize_case', None)
        if case_mode == 'lower':
            text = text.lower()
        elif case_mode == 'upper':
            text = text.upper()
        elif case_mode == 'title':
            text = text.title()
        elif case_mode == 'sentence':
            # Capitalize first letter of each sentence
            text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text
    
    def _apply_content_cleaning(self, text: str) -> str:
        """Apply content-specific cleaning."""
        content_rules = self.preprocessing_rules.get('content', {})
        
        # Remove excessive punctuation
        if content_rules.get('normalize_punctuation', False):
            text = re.sub(r'[.]{3,}', '...', text)  # Multiple dots to ellipsis
            text = re.sub(r'[!]{2,}', '!', text)    # Multiple exclamations to single
            text = re.sub(r'[?]{2,}', '?', text)    # Multiple questions to single
        
        # Remove excessive character repetition
        max_repetition = content_rules.get('max_character_repetition', 0)
        if max_repetition > 0:
            pattern = r'(.)\1{' + str(max_repetition) + ',}'
            text = re.sub(pattern, lambda m: m.group(1) * max_repetition, text)
        
        # Fix common typos
        if content_rules.get('fix_common_typos', False):
            typo_fixes = content_rules.get('typo_replacements', {})
            for typo, correction in typo_fixes.items():
                text = re.sub(r'\b' + re.escape(typo) + r'\b', correction, text, flags=re.IGNORECASE)
        
        return text
    
    def preprocess_text(self, text: str, preprocessing_profile: str = "default") -> str:
        """
        Preprocess text according to configured rules.
        
        Args:
            text: Input text to preprocess
            preprocessing_profile: Preprocessing profile to use
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Use specific preprocessing profile if configured
        if preprocessing_profile != "default" and preprocessing_profile in self.preprocessing_rules:
            original_rules = self.preprocessing_rules
            self.preprocessing_rules = self.preprocessing_rules[preprocessing_profile]
        
        try:
            # Apply preprocessing steps in order
            if self.preprocessing_rules.get('enabled', True):
                # Unicode normalization
                text = self._normalize_unicode(text)
                
                # Whitespace cleaning
                text = self._clean_whitespace(text)
                
                # Character replacements
                text = self._apply_character_replacements(text)
                
                # Case normalization
                text = self._apply_case_normalization(text)
                
                # Content cleaning
                text = self._apply_content_cleaning(text)
                
                # Final whitespace cleanup
                text = text.strip()
            
            return text
            
        finally:
            # Restore original rules if we switched profiles
            if preprocessing_profile != "default" and preprocessing_profile in self.config.get('input_preprocessing', {}):
                self.preprocessing_rules = original_rules
    
    def process_input(self, text: str, 
                     validation_profile: str = "default",
                     preprocessing_profile: str = "default",
                     auto_correct: bool = True) -> ValidationResult:
        """
        Complete input processing: validation + preprocessing.
        
        Args:
            text: Input text to process
            validation_profile: Validation profile to use
            preprocessing_profile: Preprocessing profile to use
            auto_correct: Whether to apply preprocessing automatically
            
        Returns:
            ValidationResult with processed text
        """
        # Initial validation
        initial_result = self.validate_input(text, validation_profile)
        
        if not initial_result.is_valid and not auto_correct:
            return initial_result
        
        # Apply preprocessing
        processed_text = self.preprocess_text(text, preprocessing_profile)
        
        # Validate processed text
        final_result = self.validate_input(processed_text, validation_profile)
        
        # Combine results
        return ValidationResult(
            is_valid=final_result.is_valid,
            original_text=text,
            processed_text=processed_text,
            errors=final_result.errors,
            warnings=initial_result.warnings + final_result.warnings
        )
    
    def get_validation_profiles(self) -> List[str]:
        """Get available validation profiles."""
        return list(self.validation_rules.keys())
    
    def get_preprocessing_profiles(self) -> List[str]:
        """Get available preprocessing profiles."""
        return list(self.preprocessing_rules.keys())


# Essential input processing functionality only - removed convenience functions