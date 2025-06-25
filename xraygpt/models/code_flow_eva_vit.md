# Code Flow for `xraygpt/models/eva_vit.py`

Below is a detailed, line-by-line code flow for the provided file.  
**Sections are grouped by class/function for clarity, but within each block, line-by-line logic is explained.**

---

## Imports

1. **Imports standard and third-party packages**:  
   - `math`, `partial` (from functools), `torch`, `torch.nn`, `torch.nn.functional`, `torch.utils.checkpoint`
   - Imports from `timm` for model layers/utilities
   - Imports for model registry and file download from local (`xraygpt.common.dist_utils`)

---

## `_cfg` function

2. **Defines a config dictionary utility for models**  
   - Returns a dictionary with defaults for model parameters  
   - Accepts a URL and extra kwargs for customization

---

## `DropPath` class

3. **Implements stochastic depth (regularization) as a module**  
   - `__init__`: sets the drop probability
   - `forward`: applies `drop_path` from timm (randomly drops inputs during training)
   - `extra_repr`: string summary with drop probability

---

## `Mlp` class

4. **Implements a 2-layer feedforward block, typical in Transformers**
   - `__init__`:  
     - Sets up first linear, activation, second linear, dropout
   - `forward`:  
     - Applies first linear, activation, second linear, dropout  
     - (Note: Dropout after activation commented out for BERT-like behavior)

---

## `Attention` class

5. **Multi-head self-attention with optional relative position bias**
   - `__init__`:  
     - Sets up number of heads and per-head dimension  
     - Optionally uses a custom head dimension  
     - Sets up linear for QKV projections (single layer)  
     - Optionally sets up learnable biases for Q/V  
     - If windowed attention, computes number of relative positions and sets up a learnable bias table  
     - Precomputes relative position index tensor for bias lookup  
     - Sets up dropout and projection
   - `forward`:  
     - Calculates QKV projections (with or without biases)  
     - Reshapes QKV for multi-head attention  
     - Computes dot-product attention scores  
     - Adds relative position bias if present  
     - Applies softmax to get attention weights  
     - Dropout on attention weights  
     - Applies attention weights to values, merges heads  
     - Final output projection and dropout

---

## `Block` class

6. **A standard Transformer encoder block**
   - `__init__`:  
     - Sets up normalization, attention, stochastic depth, normalization, MLP  
     - Optionally, learnable scale parameters (`gamma_1`, `gamma_2`)
   - `forward`:  
     - Applies attention, drop path, adds to input  
     - Applies MLP, drop path, adds to input  
     - If scaling params exist, multiplies outputs by them first

---

## `PatchEmbed` class

7. **Splits an image into non-overlapping patches and embeds them**
   - `__init__`:  
     - Computes patch grid shape, number of patches  
     - Sets up Conv2d to extract patches and project to embedding dim
   - `forward`:  
     - Checks input spatial size  
     - Applies Conv2d, flattens spatial dims, transposes for batch x seq x embed_dim

---

## `RelativePositionBias` class

8. **Computes learnable relative position biases for attention**
   - `__init__`:  
     - Stores window size, computes number of relative distances  
     - Sets up learnable bias table  
     - Precomputes relative position index buffer for lookup  
   - `forward`:  
     - Looks up relative position bias for each pair of tokens  
     - Returns bias in shape (num_heads, seq_len, seq_len)

---

## `VisionTransformer` class

9. **The main Vision Transformer model**
   - `__init__`:  
     - Stores config params  
     - Patch embedding layer  
     - Learnable class token  
     - Optional absolute positional embedding  
     - Dropout after position embedding  
     - Optional shared relative position bias  
     - Sets up stack of Transformer blocks (ModuleList)  
     - Initializes class token and positional embedding (truncated normal)  
     - Applies custom weight init recursively  
     - Calls `fix_init_weight` to rescale attention/MLP projection weights
   - `fix_init_weight`:  
     - For each block, rescales attention and MLP projection weights by sqrt(2 * layer_id)
   - `_init_weights`:  
     - For Linear: truncated normal for weights, 0 for bias  
     - For LayerNorm: 0 for bias, 1 for weight
   - `get_classifier`:  
     - Returns classification head  
   - `reset_classifier`:  
     - Changes the classification head (for fine-tuning on a new task)
   - `forward_features`:  
     - Converts image to patch embeddings  
     - Prepends class token, adds position embedding  
     - Applies dropout  
     - Computes relative position bias if needed  
     - Runs input through each Transformer block  
     - Returns features (not yet logits)
   - `forward`:  
     - Returns output of `forward_features` (not logits)
   - `get_intermediate_layers`:  
     - Like `forward_features`, but returns outputs after every block

---

## `interpolate_pos_embed` function

10. **Interpolates position embeddings from a checkpoint to fit new grid size**
    - Checks for position embedding in checkpoint  
    - Computes old and new grid sizes  
    - If different, interpolates position tokens to new size using bicubic interpolation  
    - Concatenates with extra tokens (class token, etc.)  
    - Updates checkpoint's position embedding

---

## `convert_weights_to_fp16` function

11. **Converts Conv/Linear weights in a model to FP16**
    - Recursively walks the model  
    - For Conv1d, Conv2d, Linear: casts weights and biases to half precision

---

## `create_eva_vit_g` function

12. **Factory for a large EVA Vision Transformer with pretrained weights**
    - Instantiates a VisionTransformer with specified architecture  
    - Downloads pretrained weights  
    - Loads checkpoint, interpolates position embedding if needed  
    - Loads weights into model  
    - Converts model to FP16 if requested  
    - Returns ready-to-use model

---

**In summary:**
- The file defines a modular Vision Transformer implementation (with EVA-specific settings).
- It provides all building blocks: patch embedding, attention, MLP, transformer blocks, relative bias.
- It supports pretrained weight loading and interpolation, and FP16 casting for efficient inference.
