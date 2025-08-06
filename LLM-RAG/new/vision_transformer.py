"""
EVA-CLIP-G Vision Transformer
Based on EVA, BEIT, timm and DeiT code bases
Implements the Vision Transformer for processing patch embeddings into contextualized representations.
"""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# Import shared PatchEmbed class
from patch_embed import PatchEmbed, to_2tuple

# Set up logging
logger = logging.getLogger(__name__)

# to_2tuple is now imported from patch_embed module

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    """MLP block for Vision Transformer."""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention for Vision Transformer."""
    
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0., 
        proj_drop=0., 
        window_size=None, 
        attn_head_dim=None
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))
            
            # Get pair-wise relative position index
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block for Vision Transformer."""
    
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        drop=0., 
        attn_drop=0., 
        drop_path=0., 
        init_values=None, 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        window_size=None, 
        attn_head_dim=None
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            attn_drop=attn_drop, 
            proj_drop=drop, 
            window_size=window_size, 
            attn_head_dim=attn_head_dim
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# PatchEmbed class is imported from image_embeddings.py to avoid duplication


# RelativePositionBias class removed - not used in EVA-CLIP-G configuration
# EVA-CLIP-G uses absolute position embeddings only


class VisionTransformer(nn.Module):
    """
    EVA-CLIP-G Vision Transformer
    Processes patch embeddings into contextualized representations.
    """
    
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=1000, 
        embed_dim=1408,  # EVA-CLIP-G dimension
        depth=39,        # EVA-CLIP-G depth
        num_heads=16,    # 1408//88 = 16
        mlp_ratio=4.3637, 
        qkv_bias=False, 
        qk_scale=None, 
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm, 
        init_values=None,
        use_abs_pos_emb=True, 
        # Relative position bias parameters removed - not used in EVA-CLIP-G 
        use_mean_pooling=True, 
        init_scale=0.001, 
        use_checkpoint=False
    ):
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        
        # Patch embedding (will be replaced by external patch embeddings)
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embeddings
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # EVA-CLIP-G uses absolute position embeddings only
        self.rel_pos_bias = None

        self.use_checkpoint = use_checkpoint
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer, 
                init_values=init_values,
                window_size=None  # EVA-CLIP-G doesn't use relative position bias
            )
            for i in range(depth)
        ])

        # Initialize weights
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        """Fix initialization weights."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, use_external_patches=False):
        """
        Forward pass through Vision Transformer.
        
        Args:
            x: Input tensor - either image tensor [B, C, H, W] or patch embeddings [B, N, D]
            use_external_patches: If True, x is treated as pre-computed patch embeddings
        """
        # Ensure input tensor matches model dtype
        model_dtype = next(self.parameters()).dtype
        if x.dtype != model_dtype:
            x = x.to(model_dtype)
        
        if use_external_patches:
            # x is already patch embeddings [B, N, D]
            # Need to project to correct dimension if different
            if x.shape[-1] != self.embed_dim:
                # Add projection layer if dimensions don't match
                if not hasattr(self, 'patch_proj'):
                    self.patch_proj = nn.Linear(x.shape[-1], self.embed_dim).to(x.device).to(model_dtype)
                # Ensure projection layer has correct dtype
                if self.patch_proj.weight.dtype != model_dtype:
                    self.patch_proj = self.patch_proj.to(model_dtype)
                x = self.patch_proj(x)
        else:
            # x is image tensor, apply patch embedding
            x = self.patch_embed(x)
        
        batch_size, seq_len, _ = x.size()
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Handle position embeddings with different sequence lengths
        if self.pos_embed is not None:
            # Check if sequence length matches position embeddings
            if x.shape[1] != self.pos_embed.shape[1]:
                logger.warning(f"Sequence length mismatch: input {x.shape[1]} vs pos_embed {self.pos_embed.shape[1]}")
                # Interpolate or truncate position embeddings to match input
                if x.shape[1] < self.pos_embed.shape[1]:
                    # Truncate position embeddings
                    pos_embed = self.pos_embed[:, :x.shape[1], :]
                else:
                    # Interpolate position embeddings
                    pos_embed = self._interpolate_pos_embed(self.pos_embed, x.shape[1])
                x = x + pos_embed
            else:
                x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None)
            else:
                x = blk(x, None)
        
        return x

    def forward(self, x, use_external_patches=False):
        """Forward pass."""
        x = self.forward_features(x, use_external_patches=use_external_patches)
        return x

    def get_intermediate_layers(self, x, use_external_patches=False):
        """Get intermediate layer outputs."""
        # Ensure input tensor matches model dtype
        model_dtype = next(self.parameters()).dtype
        if x.dtype != model_dtype:
            x = x.to(model_dtype)
        
        if use_external_patches:
            if x.shape[-1] != self.embed_dim:
                if not hasattr(self, 'patch_proj'):
                    self.patch_proj = nn.Linear(x.shape[-1], self.embed_dim).to(x.device).to(model_dtype)
                # Ensure projection layer has correct dtype
                if self.patch_proj.weight.dtype != model_dtype:
                    self.patch_proj = self.patch_proj.to(model_dtype)
                x = self.patch_proj(x)
        else:
            x = self.patch_embed(x)
        
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Handle position embeddings with different sequence lengths
        if self.pos_embed is not None:
            if x.shape[1] != self.pos_embed.shape[1]:
                logger.warning(f"Sequence length mismatch: input {x.shape[1]} vs pos_embed {self.pos_embed.shape[1]}")
                if x.shape[1] < self.pos_embed.shape[1]:
                    pos_embed = self.pos_embed[:, :x.shape[1], :]
                else:
                    pos_embed = self._interpolate_pos_embed(self.pos_embed, x.shape[1])
                x = x + pos_embed
            else:
                x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        for blk in self.blocks:
            x = blk(x, None)
            features.append(x)
        
        return features
    
    def _interpolate_pos_embed(self, pos_embed, target_length):
        """
        Interpolate position embeddings to match target sequence length.
        
        Args:
            pos_embed: Original position embeddings [1, N, D]
            target_length: Target sequence length
            
        Returns:
            Interpolated position embeddings [1, target_length, D]
        """
        if pos_embed.shape[1] == target_length:
            return pos_embed
        
        # Extract CLS token and patch embeddings
        cls_pos_embed = pos_embed[:, :1, :]  # [1, 1, D]
        patch_pos_embed = pos_embed[:, 1:, :]  # [1, N-1, D]
        
        if target_length == 1:
            # Only CLS token needed
            return cls_pos_embed
        
        # Calculate grid size for original and target
        orig_patches = patch_pos_embed.shape[1]
        target_patches = target_length - 1  # Subtract 1 for CLS token
        
        orig_size = int(orig_patches ** 0.5)
        target_size = int(target_patches ** 0.5)
        
        if orig_size * orig_size != orig_patches:
            logger.warning(f"Original patch count {orig_patches} is not a perfect square")
            # Simple truncation/padding for non-square cases
            if target_patches <= orig_patches:
                new_patch_pos_embed = patch_pos_embed[:, :target_patches, :]
            else:
                # Pad with zeros
                padding = target_patches - orig_patches
                new_patch_pos_embed = torch.cat([
                    patch_pos_embed,
                    torch.zeros(1, padding, patch_pos_embed.shape[-1], device=patch_pos_embed.device)
                ], dim=1)
        else:
            # Reshape to 2D grid and interpolate
            patch_pos_embed_2d = patch_pos_embed.reshape(1, orig_size, orig_size, -1)
            patch_pos_embed_2d = patch_pos_embed_2d.permute(0, 3, 1, 2)  # [1, D, H, W]
            
            # Interpolate to target size
            new_patch_pos_embed_2d = F.interpolate(
                patch_pos_embed_2d,
                size=(target_size, target_size),
                mode='bicubic',
                align_corners=False
            )
            
            # Reshape back to sequence
            new_patch_pos_embed = new_patch_pos_embed_2d.permute(0, 2, 3, 1).reshape(1, target_patches, -1)
        
        # Concatenate CLS token and patch embeddings
        interpolated_pos_embed = torch.cat([cls_pos_embed, new_patch_pos_embed], dim=1)
        
        logger.info(f"Interpolated position embeddings from {pos_embed.shape[1]} to {target_length}")
        return interpolated_pos_embed


def create_eva_vit_g(
    img_size=224,
    drop_path_rate=0.4,
    use_checkpoint=False,
    precision="fp16"
):
    """
    Create EVA-CLIP-G Vision Transformer.
    
    Args:
        img_size: Input image size
        drop_path_rate: Drop path rate for stochastic depth
        use_checkpoint: Whether to use gradient checkpointing
        precision: Model precision (fp16 or fp32)
    
    Returns:
        VisionTransformer: EVA-CLIP-G model
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=14,
        use_mean_pooling=False,
        embed_dim=1408,
        depth=39,
        num_heads=1408//88,  # 16 heads
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=use_checkpoint,
    )
    
    # Convert to fp16 if requested
    if precision == "fp16":
        def convert_weights_to_fp16(l):
            if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                l.weight.data = l.weight.data.half()
                if l.bias is not None:
                    l.bias.data = l.bias.data.half()
            elif isinstance(l, nn.LayerNorm):
                l.weight.data = l.weight.data.half()
                l.bias.data = l.bias.data.half()
            elif isinstance(l, nn.Parameter):
                l.data = l.data.half()
        
        model.apply(convert_weights_to_fp16)
        
        # Also convert parameter tensors
        for param in model.parameters():
            if param.data.dtype == torch.float32:
                param.data = param.data.half()
    
    logger.info(f"Created EVA-CLIP-G Vision Transformer:")
    logger.info(f"  - Image size: {img_size}")
    logger.info(f"  - Patch size: 14")
    logger.info(f"  - Embed dim: 1408")
    logger.info(f"  - Depth: 39 layers")
    logger.info(f"  - Heads: 16")
    logger.info(f"  - Precision: {precision}")
    
    return model