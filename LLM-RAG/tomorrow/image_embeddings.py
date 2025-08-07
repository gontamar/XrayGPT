import torch
from torch import nn
import yaml
import logging
from typing import Dict, List, Optional, Union, Any
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from omegaconf import OmegaConf

# Set up logging
logger = logging.getLogger(__name__)


def to_2tuple(x):
    """Convert input to 2-tuple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding using convolutional projection.
    Converts image into sequence of patch embeddings for Vision Transformer.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
       
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        # Calculate number of patches and grid shape
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        
        # Store configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Convolutional projection layer
        # This effectively divides the image into patches and projects each patch to embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        
        # Verify input dimensions match expected size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # Apply convolutional projection
        # [B, C, H, W] -> [B, embed_dim, H//patch_size, W//patch_size]
        x = self.proj(x)
        
        # Flatten spatial dimensions and transpose
        # [B, embed_dim, H//patch_size, W//patch_size] -> [B, embed_dim, num_patches] -> [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        return x
    
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
    Image processing manager with BLIP2 support and optional patch embeddings.
    Handles image loading, preprocessing, and feature extraction.
    """
    
    def __init__(self, config_path: str = "tokenizer_config.yaml"):
        """
        Initialize image processor.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.processor = None
        self.current_config = None
        self.patch_embed = None
        
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
        """
        Get processor configuration from config file.
        
        Args:
            processor_name: Name of processor in config
            
        Returns:
            Processor configuration dictionary
        """
        if self.config is None:
            raise ValueError("Configuration not loaded")
        
        # Search in image_processors section
        if 'image_processors' in self.config and processor_name in self.config['image_processors']:
            return self.config['image_processors'][processor_name]
        
        raise ValueError(f"Processor '{processor_name}' not found in configuration")
    
    def load_processor(self, processor_name: Optional[str] = None, 
                      custom_config: Optional[Dict[str, Any]] = None) -> Blip2ImageEvalProcessor:
    
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
            
            return self.processor
            
        except Exception as e:
            logger.error(f"Error loading image processor: {e}")
            raise
    
    def load_patch_embed(self, config: Dict[str, Any], device: str = "cpu") -> PatchEmbed:
       
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
    
    
    def process_image(self, image_path: str, 
                     add_batch_dim: bool = True,
                     device: str = "cpu",
                     create_patches: bool = False) -> Dict[str, Any]:
        
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
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
