"""
BLIP2 Image Processor
Image processing system based on BLIP2 architecture for vision-language models.
"""

import yaml
import os
from typing import Dict, List, Optional, Union, Any, Tuple
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from omegaconf import OmegaConf
import logging

# Import Vision Transformer components when needed to avoid circular imports
def _import_vision_transformer():
    """Lazy import of Vision Transformer to avoid circular dependencies."""
    try:
        from vision_transformer import VisionTransformer, create_eva_vit_g
        return VisionTransformer, create_eva_vit_g
    except ImportError as e:
        logger.error(f"Could not import Vision Transformer: {e}")
        raise ImportError("Vision Transformer module not available")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import shared PatchEmbed class
from patch_embed import PatchEmbed, to_2tuple


class BlipImageBaseProcessor:
    """Base class for BLIP image processors."""
    
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.48145466, 0.4578275, 0.40821073]
        if std is None:
            std = [0.26862954, 0.26130258, 0.27577711]
        
        self.normalize = transforms.Normalize(mean=mean, std=std)


class Blip2ImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)


class ImageProcessor:
    """
    A generic image processor that handles different image preprocessing configurations
    """
    
    def __init__(self, config_path: str = "tokenizer_config.yaml"):
        """
        Initialize the ImageProcessor with a configuration file.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.processor = None
        self.current_config = None
        self.patch_embed = None
        self.vision_transformer = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                if config is None:
                    logger.error(f"Configuration file {self.config_path} is empty or invalid")
                    raise ValueError(f"Configuration file {self.config_path} is empty or invalid")
                logger.info(f"Image processor configuration loaded from {self.config_path}")
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
    
    def get_processor_info(self, processor_name: str) -> Dict[str, Any]:
        """Get image processor configuration."""
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        # Check in image_processors section
        if 'image_processors' in self.config and processor_name in self.config['image_processors']:
            return self.config['image_processors'][processor_name]
        
        raise ValueError(f"Image processor '{processor_name}' not found in configuration")
    
    def load_processor(self, processor_name: Optional[str] = None, 
                      custom_config: Optional[Dict[str, Any]] = None) -> Blip2ImageEvalProcessor:
        """
        Load a BLIP2 image processor based on configuration.
        
        Args:
            processor_name (str, optional): Name of the processor from config.
            custom_config (dict, optional): Custom configuration to override defaults.
        
        Returns:
            Blip2ImageEvalProcessor: Loaded BLIP2 image processor
        """
        if custom_config:
            config = custom_config
        elif processor_name:
            config = self.get_processor_info(processor_name)
        else:
            # Use default BLIP2 configuration
            config = {
                'type': 'blip2',
                'image_size': 224,
                'normalize_mean': [0.48145466, 0.4578275, 0.40821073],
                'normalize_std': [0.26862954, 0.26130258, 0.27577711]
            }
        
        try:
            # Create BLIP2 processor from config
            cfg = OmegaConf.create({
                'image_size': config.get('image_size', 224),
                'mean': config.get('normalize_mean', None),
                'std': config.get('normalize_std', None)
            })
            
            self.processor = Blip2ImageEvalProcessor.from_config(cfg)
            self.current_config = config
            
            logger.info(f"BLIP2 image processor loaded")
            logger.info(f"Image size: {config.get('image_size', 224)}")
            
            # Initialize patch embedding if configured
            if config.get('enable_patch_embed', False):
                self.load_patch_embed(config)
            
            # Initialize Vision Transformer if configured
            if config.get('enable_vision_transformer', False):
                self.load_vision_transformer(config)
            
            return self.processor
            
        except Exception as e:
            logger.error(f"Error loading image processor: {e}")
            raise
    
    def load_patch_embed(self, config: Dict[str, Any], device: str = "cpu") -> PatchEmbed:
        """
        Load patch embedding module based on configuration.
        
        Args:
            config: Configuration dictionary containing patch embed parameters
            device: Device to load the patch embedding on
            
        Returns:
            PatchEmbed: Loaded patch embedding module
        """
        try:
            # Get patch embedding parameters from config
            img_size = config.get('image_size', 224)
            patch_size = config.get('patch_size', 16)
            in_chans = config.get('in_channels', 3)
            embed_dim = config.get('embed_dim', 768)
            
            # Create patch embedding module
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim
            ).to(device)
            
            logger.info(f"Patch embedding loaded: {img_size}x{img_size} -> {patch_size}x{patch_size} patches, embed_dim={embed_dim}")
            logger.info(f"Number of patches: {self.patch_embed.num_patches}")
            
            return self.patch_embed
            
        except Exception as e:
            logger.error(f"Error loading patch embedding: {e}")
            raise
    
    def load_vision_transformer(self, config: Dict[str, Any], device: str = "cpu"):
        """
        Load EVA-CLIP-G Vision Transformer based on configuration.
        
        Args:
            config: Configuration dictionary containing ViT parameters
            device: Device to load the Vision Transformer on
            
        Returns:
            VisionTransformer: Loaded Vision Transformer model
        """
        try:
            # Import Vision Transformer components
            VisionTransformer, create_eva_vit_g = _import_vision_transformer()
            
            # Get Vision Transformer parameters from config
            img_size = config.get('image_size', 224)
            drop_path_rate = config.get('drop_path_rate', 0.4)
            use_checkpoint = config.get('use_checkpoint', False)
            precision = config.get('precision', 'fp16')
            
            # Create EVA-CLIP-G Vision Transformer
            self.vision_transformer = create_eva_vit_g(
                img_size=img_size,
                drop_path_rate=drop_path_rate,
                use_checkpoint=use_checkpoint,
                precision=precision
            ).to(device)
            
            logger.info(f"EVA-CLIP-G Vision Transformer loaded:")
            logger.info(f"  - Image size: {img_size}")
            logger.info(f"  - Embed dim: 1408")
            logger.info(f"  - Depth: 39 layers")
            logger.info(f"  - Heads: 16")
            logger.info(f"  - Drop path rate: {drop_path_rate}")
            logger.info(f"  - Precision: {precision}")
            
            return self.vision_transformer
            
        except Exception as e:
            logger.error(f"Error loading Vision Transformer: {e}")
            raise
    
    def process_image(self, image_path: str, 
                     add_batch_dim: bool = True,
                     device: str = "cpu",
                     create_patches: bool = False,
                     use_vision_transformer: bool = False) -> Dict[str, Any]:
        """
        Process an image using the BLIP2 processor and optionally Vision Transformer.
        
        Args:
            image_path (str): Path to the image file
            add_batch_dim (bool): Whether to add batch dimension
            device (str): Device to move tensor to
            create_patches (bool): Whether to create patch embeddings
            use_vision_transformer (bool): Whether to process through Vision Transformer
        
        Returns:
            Dict containing processed image tensor, patch embeddings, ViT features (if requested), and metadata
        """
        if self.processor is None:
            logger.warning("No image processor loaded. Loading default BLIP2 processor.")
            self.load_processor()
        
        try:
            # Load and convert image
            image = Image.open(image_path).convert("RGB")
            
            # Apply BLIP2 transforms
            image_tensor = self.processor(image)
            
            # Add batch dimension if requested
            if add_batch_dim:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Move to device
            image_tensor = image_tensor.to(device)
            
            result = {
                'image_tensor': image_tensor,
                'original_size': image.size,
                'processed_size': image_tensor.shape,
                'image_path': image_path,
                'device': device
            }
            
            # Create patch embeddings if requested
            if create_patches:
                if self.patch_embed is None:
                    # Load patch embedding with default config if not loaded
                    default_patch_config = {
                        'image_size': 224,
                        'patch_size': 16,
                        'in_channels': 3,
                        'embed_dim': 768
                    }
                    self.load_patch_embed(default_patch_config, device)
                
                # Create patch embeddings
                with torch.no_grad():
                    patch_embeddings = self.patch_embed(image_tensor)
                
                result.update({
                    'patch_embeddings': patch_embeddings,
                    'patch_shape': self.patch_embed.patch_shape,
                    'num_patches': self.patch_embed.num_patches,
                    'embed_dim': patch_embeddings.shape[-1],
                    'patch_size': self.patch_embed.patch_size
                })
                
                logger.info(f"Created patch embeddings: {patch_embeddings.shape}")
            
            # Process through Vision Transformer if requested
            if use_vision_transformer:
                if self.vision_transformer is None:
                    # Load Vision Transformer with default config if not loaded
                    default_vit_config = {
                        'image_size': 224,
                        'drop_path_rate': 0.4,
                        'use_checkpoint': False,
                        'precision': 'fp16'
                    }
                    self.load_vision_transformer(default_vit_config, device)
                
                # Process through Vision Transformer
                with torch.no_grad():
                    if create_patches and 'patch_embeddings' in result:
                        # Use existing patch embeddings (ViT handles dtype conversion automatically)
                        vit_features = self.vision_transformer(patch_embeddings, use_external_patches=True)
                    else:
                        # Use image tensor directly (ViT will create its own patches and handle dtype)
                        vit_features = self.vision_transformer(image_tensor, use_external_patches=False)
                
                result.update({
                    'vit_features': vit_features,
                    'vit_cls_token': vit_features[:, 0, :],  # CLS token
                    'vit_patch_tokens': vit_features[:, 1:, :],  # Patch tokens
                    'vit_embed_dim': vit_features.shape[-1],
                    'vit_sequence_length': vit_features.shape[1]
                })
                
                logger.info(f"Vision Transformer features: {vit_features.shape}")
                logger.info(f"CLS token shape: {vit_features[:, 0, :].shape}")
                logger.info(f"Patch tokens shape: {vit_features[:, 1:, :].shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def process_multiple_images(self, image_paths: List[str],
                               device: str = "cpu") -> Dict[str, Any]:
        """
        Process multiple images and stack them into a batch using BLIP2 processor.
        
        Args:
            image_paths (List[str]): List of image file paths
            device (str): Device to move tensors to
        
        Returns:
            Dict containing batched image tensor and metadata
        """
        if self.processor is None:
            logger.warning("No image processor loaded. Loading default BLIP2 processor.")
            self.load_processor()
        
        try:
            processed_images = []
            original_sizes = []
            
            for image_path in image_paths:
                # Load and convert image
                image = Image.open(image_path).convert("RGB")
                original_sizes.append(image.size)
                
                # Apply BLIP2 transforms
                image_tensor = self.processor(image)
                processed_images.append(image_tensor)
            
            # Stack into batch
            batch_tensor = torch.stack(processed_images, dim=0)
            batch_tensor = batch_tensor.to(device)
            
            result = {
                'image_tensor': batch_tensor,
                'original_sizes': original_sizes,
                'processed_size': batch_tensor.shape,
                'image_paths': image_paths,
                'device': device,
                'batch_size': len(image_paths)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing multiple images: {e}")
            raise
    
    def get_processor_info_current(self) -> Dict[str, Any]:
        """Get information about the currently loaded processor."""
        if self.current_config is None:
            return {"status": "No processor loaded"}
        
        return {
            "type": self.current_config.get('type', 'unknown'),
            "image_size": self.current_config.get('image_size', 'unknown'),
            "normalize_mean": self.current_config.get('normalize_mean', 'unknown'),
            "normalize_std": self.current_config.get('normalize_std', 'unknown'),
            "config_path": self.config_path
        }
