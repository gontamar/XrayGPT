import torch
from PIL import Image
from torchvision import transforms
from xraygpt.models.eva_vit import PatchEmbed, VisionTransformer
import torch.nn as nn
import torch.nn.functional as F
import os

def load_and_preprocess_image(image_path, img_size=224):
    # Load image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    # Preprocess: resize, center crop, to tensor, normalize (ViT typical normalization)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension
    return image_tensor

def test_linear_proj_with_local_image(image_path):
    # Configuration (should match XrayGPT defaults)
    img_size = 224
    patch_size = 14    # For EVA-CLIP-G, patch_size=14, vision_width=1408
    in_chans = 3
    vision_width = 1408
    embed_dim = 2560   # BLIP2/Minigpt4 default
    num_layers = 2     # For fast test, not the full model
    num_heads = 8

    # 1. Load and preprocess image
    image_tensor = load_and_preprocess_image(image_path, img_size=img_size)

    # 2. Create PatchEmbed and VisionTransformer
    patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=vision_width)
    vit = VisionTransformer(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        embed_dim=vision_width, depth=num_layers, num_heads=num_heads,
        mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm
    )

    # 3. Convert image to patch vectors (patch embedding)
    patch_vectors = patch_embed(image_tensor)
    print("Patch vectors shape:", patch_vectors.shape)  # (batch, num_patches, vision_width)
    print("Patch vectors:", patch_vectors)

    # 4. Visual features from VisionTransformer (visual encoder)
    visual_features = vit(image_tensor)
    print("Visual features shape:", visual_features.shape)  # (batch, num_patches+1, vision_width)
    print("Visual features:", visual_features)

    # 5. Linear projection (as in BLIP2)
    vision_proj = nn.Linear(vision_width, embed_dim)
    projected_features = vision_proj(visual_features)
    print("Projected features shape:", projected_features.shape)  # (batch, num_patches+1, embed_dim)
    print("Projected features:", projected_features) 
    
    # 6. Normalize (as in BLIP2)
    projected_features_norm = F.normalize(projected_features, dim=-1)
    print("Projected features (normalized) shape:", projected_features_norm.shape)
    print("Projected features (normalized):", projected_features_norm)

if __name__ == "__main__":
    local_image_path = "images/example_test_images/img2.png"
    if not os.path.exists(local_image_path):
        print(f"Image path not found: {local_image_path}")
    else:
        test_linear_proj_with_local_image(local_image_path)