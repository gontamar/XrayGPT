"""
Generic Data Validation Module
Handles domain-specific data validation and conditioning before tokenization
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DataType(Enum):
    """Supported data types"""
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"
    TABULAR = "tabular"

class ValidationResult(Enum):
    """Validation results"""
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_PREPROCESSING = "needs_preprocessing"

@dataclass
class ValidationOutput:
    """Output structure for validation results"""
    result: ValidationResult
    data: Any
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class BaseDomainValidator(ABC):
    """Base class for domain-specific validators"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain_name = config.get("domain_name", "generic")
        
    @abstractmethod
    def validate_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> ValidationOutput:
        """Validate image data"""
        pass
    
    @abstractmethod
    def validate_text(self, text: str) -> ValidationOutput:
        """Validate text data"""
        pass
    
    @abstractmethod
    def apply_conditioning(self, data: Any, data_type: DataType) -> Any:
        """Apply domain-specific conditioning"""
        pass

class MedicalDomainValidator(BaseDomainValidator):
    """Medical domain validator (completely config-driven)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # All parameters come from config - no hardcoded defaults
        self.min_image_size = tuple(config.get("min_image_size", [224, 224]))
        self.max_image_size = tuple(config.get("max_image_size", [1024, 1024]))
        self.forbidden_text_patterns = config.get("forbidden_text_patterns", [])
        self.required_image_channels = config.get("required_channels", [1, 3])
        
    def validate_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> ValidationOutput:
        """Validate medical image data"""
        try:
            # Convert to PIL Image if needed
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:  # Batch dimension
                    image = image.squeeze(0)
                if image.dim() == 3 and image.shape[0] in [1, 3]:  # CHW format
                    image = image.permute(1, 2, 0)
                image = Image.fromarray((image.numpy() * 255).astype(np.uint8))
            elif isinstance(image, np.ndarray):
                if image.ndim == 3 and image.shape[2] in [1, 3]:
                    image = Image.fromarray(image.astype(np.uint8))
                elif image.ndim == 2:  # Grayscale
                    image = Image.fromarray(image.astype(np.uint8), mode='L')
                else:
                    return ValidationOutput(
                        ValidationResult.INVALID,
                        None,
                        {},
                        f"Invalid image dimensions: {image.shape}"
                    )
            
            # Check if image is blank/empty
            img_array = np.array(image)
            if self._is_blank_image(img_array):
                return ValidationOutput(
                    ValidationResult.INVALID,
                    None,
                    {"reason": "blank_image"},
                    "Image appears to be blank or empty"
                )
            
            # Check image size
            width, height = image.size
            if width < self.min_image_size[0] or height < self.min_image_size[1]:
                return ValidationOutput(
                    ValidationResult.NEEDS_PREPROCESSING,
                    image,
                    {"reason": "size_too_small", "current_size": (width, height)},
                    f"Image size {(width, height)} is smaller than minimum {self.min_image_size}"
                )
            
            if width > self.max_image_size[0] or height > self.max_image_size[1]:
                return ValidationOutput(
                    ValidationResult.NEEDS_PREPROCESSING,
                    image,
                    {"reason": "size_too_large", "current_size": (width, height)},
                    f"Image size {(width, height)} is larger than maximum {self.max_image_size}"
                )
            
            # Check image quality/contrast
            if self._has_poor_quality(img_array):
                return ValidationOutput(
                    ValidationResult.NEEDS_PREPROCESSING,
                    image,
                    {"reason": "poor_quality"},
                    "Image has poor quality or contrast"
                )
            
            return ValidationOutput(
                ValidationResult.VALID,
                image,
                {"size": (width, height), "mode": image.mode}
            )
            
        except Exception as e:
            return ValidationOutput(
                ValidationResult.INVALID,
                None,
                {},
                f"Error validating image: {str(e)}"
            )
    
    def validate_text(self, text: str) -> ValidationOutput:
        """Validate medical text data"""
        if not isinstance(text, str):
            return ValidationOutput(
                ValidationResult.INVALID,
                None,
                {},
                "Text must be a string"
            )
        
        # Check for forbidden patterns
        text_lower = text.lower()
        for pattern in self.forbidden_text_patterns:
            if pattern.lower() in text_lower:
                return ValidationOutput(
                    ValidationResult.INVALID,
                    None,
                    {"forbidden_pattern": pattern},
                    f"Text contains forbidden pattern: {pattern}"
                )
        
        # Check text length
        min_length = self.config.get("min_text_length", 5)
        max_length = self.config.get("max_text_length", 1000)
        
        if len(text.strip()) < min_length:
            return ValidationOutput(
                ValidationResult.INVALID,
                None,
                {"length": len(text)},
                f"Text too short (minimum {min_length} characters)"
            )
        
        if len(text) > max_length:
            return ValidationOutput(
                ValidationResult.NEEDS_PREPROCESSING,
                text,
                {"length": len(text), "reason": "truncation_needed"},
                f"Text too long (maximum {max_length} characters)"
            )
        
        # Check for domain relevance based on config keywords
        domain_keywords = self.config.get("domain_keywords", [])
        
        has_domain_content = any(keyword in text_lower for keyword in domain_keywords) if domain_keywords else True
        if not has_domain_content and self.config.get("require_domain_content", False):
            domain_name = self.config.get("domain_name", "domain-specific")
            return ValidationOutput(
                ValidationResult.NEEDS_PREPROCESSING,
                text,
                {"reason": f"non_{domain_name}_content"},
                f"Text does not appear to contain {domain_name} content"
            )
        
        return ValidationOutput(
            ValidationResult.VALID,
            text,
            {"length": len(text), "has_domain_content": has_domain_content}
        )
    
    def apply_conditioning(self, data: Any, data_type: DataType) -> Any:
        """Apply medical domain conditioning"""
        if data_type == DataType.IMAGE:
            return self._condition_medical_image(data)
        elif data_type == DataType.TEXT:
            return self._condition_medical_text(data)
        else:
            return data
    
    def _is_blank_image(self, img_array: np.ndarray) -> bool:
        """Check if image is blank or nearly blank"""
        # Check if all pixels are the same value
        if img_array.std() < 1e-6:
            return True
        
        # Check if image is mostly black or white
        mean_intensity = img_array.mean()
        if mean_intensity < 5 or mean_intensity > 250:  # Very dark or very bright
            return True
        
        return False
    
    def _has_poor_quality(self, img_array: np.ndarray) -> bool:
        """Check if image has poor quality"""
        # Check contrast
        contrast = img_array.std()
        if contrast < 10:  # Very low contrast
            return True
        
        # Check for excessive noise (simplified)
        if len(img_array.shape) == 3:
            gray = np.mean(img_array, axis=2)
        else:
            gray = img_array
        
        # Simple noise detection using gradient
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        noise_level = np.sqrt(grad_x**2 + grad_y**2).std()
        
        if noise_level > 50:  # High noise
            return True
        
        return False
    
    def _condition_medical_image(self, image: Image.Image) -> Image.Image:
        """Apply medical-specific image conditioning"""
        # Convert to grayscale if needed for X-rays
        if self.config.get("convert_to_grayscale", False) and image.mode != 'L':
            image = image.convert('L')
        
        # Apply histogram equalization for better contrast
        if self.config.get("apply_histogram_equalization", True):
            img_array = np.array(image)
            # Simple histogram equalization
            hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]
            img_equalized = np.interp(img_array.flatten(), bins[:-1], cdf_normalized)
            image = Image.fromarray(img_equalized.reshape(img_array.shape).astype(np.uint8))
        
        return image
    
    def _condition_medical_text(self, text: str) -> str:
        """Apply medical-specific text conditioning"""
        # Remove PHI patterns (simplified)
        import re
        
        # Remove dates
        text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '[DATE]', text)
        
        # Remove potential patient IDs
        text = re.sub(r'\b\d{6,}\b', '[ID]', text)
        
        # Standardize domain terminology from config
        domain_replacements = self.config.get("domain_replacements", {})
        
        for old_term, new_term in domain_replacements.items():
            text = re.sub(old_term, new_term, text, flags=re.IGNORECASE)
        
        return text.strip()

class GenericDomainValidator(BaseDomainValidator):
    """Generic domain validator for general use cases"""
    
    def validate_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> ValidationOutput:
        """Basic image validation"""
        try:
            if isinstance(image, torch.Tensor):
                if image.numel() == 0:
                    return ValidationOutput(ValidationResult.INVALID, None, {}, "Empty tensor")
                # Convert to PIL for validation
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.dim() == 3 and image.shape[0] <= 3:
                    image = image.permute(1, 2, 0)
                image = Image.fromarray((image.numpy() * 255).astype(np.uint8))
            
            if isinstance(image, np.ndarray):
                if image.size == 0:
                    return ValidationOutput(ValidationResult.INVALID, None, {}, "Empty array")
                image = Image.fromarray(image.astype(np.uint8))
            
            if not isinstance(image, Image.Image):
                return ValidationOutput(ValidationResult.INVALID, None, {}, "Invalid image type")
            
            return ValidationOutput(ValidationResult.VALID, image, {"size": image.size})
            
        except Exception as e:
            return ValidationOutput(ValidationResult.INVALID, None, {}, str(e))
    
    def validate_text(self, text: str) -> ValidationOutput:
        """Basic text validation"""
        if not isinstance(text, str):
            return ValidationOutput(ValidationResult.INVALID, None, {}, "Not a string")
        
        if len(text.strip()) == 0:
            return ValidationOutput(ValidationResult.INVALID, None, {}, "Empty text")
        
        return ValidationOutput(ValidationResult.VALID, text, {"length": len(text)})
    
    def apply_conditioning(self, data: Any, data_type: DataType) -> Any:
        """Basic conditioning"""
        return data

class DataValidatorFactory:
    """Factory for creating domain-specific validators"""
    
    _validators = {
        "medical": MedicalDomainValidator,
        "generic": GenericDomainValidator
    }
    
    @classmethod
    def create_validator(cls, domain: str, config: Dict[str, Any]) -> BaseDomainValidator:
        """Create a validator for the specified domain"""
        if domain not in cls._validators:
            logger.warning(f"Unknown domain '{domain}', using generic validator")
            domain = "generic"
        
        validator_class = cls._validators[domain]
        return validator_class(config)
    
    @classmethod
    def register_validator(cls, domain: str, validator_class: type):
        """Register a new domain validator"""
        cls._validators[domain] = validator_class

class MultiModalDataValidator:
    """Main data validation orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.domain = config.get("domain", "generic")
        self.validator = DataValidatorFactory.create_validator(self.domain, config)
        
    def validate_and_condition(self, data: Dict[str, Any]) -> Dict[str, ValidationOutput]:
        """Validate and condition multimodal data"""
        results = {}
        
        for key, value in data.items():
            if key.startswith("image") or key == "visual_input":
                result = self.validator.validate_image(value)
                if result.result in [ValidationResult.VALID, ValidationResult.NEEDS_PREPROCESSING]:
                    conditioned_data = self.validator.apply_conditioning(result.data, DataType.IMAGE)
                    result.data = conditioned_data
                results[key] = result
                
            elif key.startswith("text") or key in ["caption", "description", "query"]:
                result = self.validator.validate_text(value)
                if result.result in [ValidationResult.VALID, ValidationResult.NEEDS_PREPROCESSING]:
                    conditioned_data = self.validator.apply_conditioning(result.data, DataType.TEXT)
                    result.data = conditioned_data
                results[key] = result
                
            else:
                # For other data types, apply basic validation
                results[key] = ValidationOutput(ValidationResult.VALID, value, {})
        
        return results
    
    def is_data_valid(self, validation_results: Dict[str, ValidationOutput]) -> bool:
        """Check if all data passed validation"""
        return all(result.result != ValidationResult.INVALID for result in validation_results.values())
    
    def get_validation_summary(self, validation_results: Dict[str, ValidationOutput]) -> Dict[str, Any]:
        """Get summary of validation results"""
        summary = {
            "total_inputs": len(validation_results),
            "valid": 0,
            "invalid": 0,
            "needs_preprocessing": 0,
            "errors": []
        }
        
        for key, result in validation_results.items():
            if result.result == ValidationResult.VALID:
                summary["valid"] += 1
            elif result.result == ValidationResult.INVALID:
                summary["invalid"] += 1
                if result.error_message:
                    summary["errors"].append(f"{key}: {result.error_message}")
            else:
                summary["needs_preprocessing"] += 1
        
        return summary