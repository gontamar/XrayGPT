import torch
from torch import nn
from torchvision import transforms
from PIL import Image

#  ViT patch embedding as in xraygpt/models/eva_vit.py
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])

# Load your chest X-ray image
img = Image.open("images/example_test_images/img2.png").convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

# Instantiate patch embedding
patch_embed = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
with torch.no_grad():
    patch_embeddings = patch_embed(img_tensor)

print("Patch embeddings shape:", patch_embeddings.shape)  # [1, num_patches, embed_dim]
print("First patch embedding vector (first 5 dims):", patch_embeddings[0,0,:5])
print("Patch embeddings :", patch_embeddings) 