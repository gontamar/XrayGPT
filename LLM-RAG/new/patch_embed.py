"""
Patch Embedding Module
Shared implementation of patch embedding for Vision Transformers and image processing.
"""

import torch
import torch.nn as nn
from typing import Tuple


def to_2tuple(x):
    """Convert input to 2-tuple."""
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    
    Converts an image into a sequence of patch embeddings using a convolutional projection.
    This is the standard approach used in Vision Transformers.
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        """
        Initialize patch embedding layer.
        
        Args:
            img_size (int): Input image size (assumes square images)
            patch_size (int): Size of each patch (assumes square patches)
            in_chans (int): Number of input channels (typically 3 for RGB)
            embed_dim (int): Embedding dimension for each patch
        """
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
        """
        Forward pass: convert image to patch embeddings.
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Patch embeddings [B, num_patches, embed_dim]
        """
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
    
    def get_patch_info(self) -> dict:
        """
        Get information about the patch embedding configuration.
        
        Returns:
            dict: Configuration information
        """
        return {
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'num_patches': self.num_patches,
            'patch_shape': self.patch_shape,
            'in_channels': self.in_chans,
            'embed_dim': self.embed_dim
        }
    
    def __repr__(self):
        return (f"PatchEmbed(img_size={self.img_size}, patch_size={self.patch_size}, "
                f"in_chans={self.in_chans}, embed_dim={self.embed_dim}, "
                f"num_patches={self.num_patches})")