import torch
from xraygpt.models.eva_vit import VisionTransformer, PatchEmbed

def test_patch_vectors_to_visual_features():
    # Parameters (should match model's config)
    img_size = 224
    patch_size = 16
    in_chans = 3
    embed_dim = 768
    batch_size = 2

    # Create dummy images (batch_size, channels, height, width)
    dummy_images = torch.randn(batch_size, in_chans, img_size, img_size)

    # Initialize PatchEmbed and VisionTransformer
    patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
    vit = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=2,         
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=torch.nn.LayerNorm
    )

    # 1. Convert image to patch vectors
    patch_vectors = patch_embed(dummy_images)
    print("Patch vectors shape:", patch_vectors.shape)  # (batch_size, num_patches, embed_dim)
    print("Patch vectors shape:", patch_vectors
          )
    # 2. Pass patch vectors through the visual encoder (VisionTransformer)
    # VisionTransformer expects raw images as input and calls patch_embed inside
    visual_features = vit(dummy_images)
    print("Visual features shape:", visual_features.shape)  # (batch_size, num_patches+1, embed_dim)
    print("Visual features shape:", visual_features)

if __name__ == "__main__":
    test_patch_vectors_to_visual_features()