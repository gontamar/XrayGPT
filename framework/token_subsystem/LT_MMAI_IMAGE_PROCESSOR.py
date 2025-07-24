from transformers import BertTokenizer
from PIL import Image
import torch
from torchvision import transforms
#from xraygpt.models.blip2 import Blip2Base
#from xraygpt.processors.blip_processors import BlipImageBaseProcessor

# --- TEXT: BLIP2 BERT Tokenizer (as in xraygpt/models/blip2.py) ---
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({"bos_token": "[DEC]"})

input_text = "There is a consolidation in the right lower lung zone."
tokens = tokenizer.tokenize(input_text)
token_ids = tokenizer.encode(input_text, return_tensors="pt")
print("Text tokens:", tokens)
print("Text token IDs:", token_ids)

# --- IMAGE: BLIP2 image preprocessor (as in xraygpt/processors/blip_processors.py) ---
# Define the same normalization as BLIP2 uses
normalize = transforms.Normalize(
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711)
)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # BLIP2 uses random crop for train, resize for eval
    transforms.ToTensor(),
    normalize,
])

# Load and process an chest xray image 
img = Image.open("images/example_test_images/img1.png").convert("RGB")
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)  # add batch dimension

print("Image tensor shape:", img_tensor.shape)
print("Image tensor :", img_tensor)
