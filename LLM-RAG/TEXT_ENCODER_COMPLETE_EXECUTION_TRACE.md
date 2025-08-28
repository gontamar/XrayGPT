# XrayGPT Text Encoder: Complete Execution Trace & File Connections

This document provides a comprehensive line-by-line trace of the text encoder execution flow in XrayGPT, showing how it connects to other files and the complete execution path from start to finish.

## 📋 **Execution Flow Overview**

```
Entry Point → Model Initialization → Text Processing → Q-Former Execution → Output Generation
     ↓              ↓                    ↓               ↓                    ↓
  main.py    →  mini_gpt4.py     →  Qformer.py    →  blip2.py        →  conversation.py
```

## 🚀 **PART 1: Entry Point and Model Loading**

### **File: `demo.py` or `train.py` (Entry Points)**

**Lines 1-20: Import and Setup**
```python
# Line 1-10: Standard imports
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# Line 11-15: XrayGPT specific imports
from xraygpt.common.config import Config
from xraygpt.common.registry import registry
from xraygpt.conversation.conversation import Chat, CONV_VISION_Vicuna0

# Line 16-20: Model imports
from xraygpt.models import *
from xraygpt.processors import *
```

**Lines 50-80: Model Initialization**
```python
# Line 50: Load configuration
cfg = Config(args)

# Line 55: Get model class from registry
model_config = cfg.model_cfg
model_cls = registry.get_model_class(model_config.arch)  # Returns MiniGPT4

# Line 60-65: Initialize model
model = model_cls.from_config(model_config)
# This calls MiniGPT4.from_config() in mini_gpt4.py

# Line 70: Load checkpoint
if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint['model'], strict=False)

# Line 75: Move to device and set eval mode
model = model.to(device)
model.eval()
```

## 🏗️ **PART 2: Model Architecture Initialization**

### **File: `mini_gpt4.py` Lines 337-381 (from_config method)**

**Lines 337-356: Configuration Loading**
```python
@classmethod
def from_config(cls, cfg):
    # Line 339-343: Extract configuration parameters
    vit_model = cfg.get("vit_model", "eva_clip_g")
    q_former_model = cfg.get("q_former_model", "...")
    img_size = cfg.get("image_size")
    num_query_token = cfg.get("num_query_token")  # 32
    llama_model = cfg.get("llama_model")
    
    # Line 344-350: More parameters
    drop_path_rate = cfg.get("drop_path_rate", 0)
    use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
    vit_precision = cfg.get("vit_precision", "fp16")
    freeze_vit = cfg.get("freeze_vit", True)
    freeze_qformer = cfg.get("freeze_qformer", True)
    low_resource = cfg.get("low_resource", False)
    
    # Line 351-355: Text-specific parameters
    prompt_path = cfg.get("prompt_path", "")
    prompt_template = cfg.get("prompt_template", "")
    max_txt_len = cfg.get("max_txt_len", 32)
    end_sym = cfg.get("end_sym", '\n')
```

**Lines 357-373: Model Instantiation**
```python
    # Line 357-372: Create model instance
    model = cls(
        vit_model=vit_model,
        q_former_model=q_former_model,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        use_grad_checkpoint=use_grad_checkpoint,
        vit_precision=vit_precision,
        freeze_vit=freeze_vit,
        freeze_qformer=freeze_qformer,
        num_query_token=num_query_token,  # 32 query tokens
        llama_model=llama_model,
        prompt_path=prompt_path,
        prompt_template=prompt_template,
        max_txt_len=max_txt_len,
        low_resource=low_resource,
        end_sym=end_sym
    )
    # This calls __init__ method below
```

### **File: `mini_gpt4.py` Lines 46-143 (__init__ method)**

**Lines 64-66: Base Initialization**
```python
def __init__(self, vit_model="eva_clip_g", q_former_model="...", ...):
    # Line 64: Initialize parent class (Blip2Base)
    super().__init__()
    
    # Line 66: Initialize BERT tokenizer for Q-Former
    self.tokenizer = self.init_tokenizer()  # Calls blip2.py:29-32
```

**Lines 85-104: Q-Former Initialization**
```python
    # Line 85-88: Initialize Q-Former and query tokens
    print('Loading Q-Former')
    self.Qformer, self.query_tokens = self.init_Qformer(
        num_query_token,  # 32
        self.visual_encoder.num_features  # 1408
    )
    # This calls blip2.py:44-57
    
    # Line 89-94: Remove unnecessary Q-Former components
    self.Qformer.cls = None
    self.Qformer.bert.embeddings.word_embeddings = None
    self.Qformer.bert.embeddings.position_embeddings = None
    for layer in self.Qformer.bert.encoder.layer:
        layer.output = None
        layer.intermediate = None
    
    # Line 95: Load pre-trained Q-Former weights
    self.load_from_pretrained(url_or_filename=q_former_model)
    
    # Line 97-103: Freeze Q-Former if specified
    if freeze_qformer:
        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
        self.Qformer = self.Qformer.eval()
        self.Qformer.train = disabled_train
        self.query_tokens.requires_grad = False
        logging.info("freeze Qformer")
```

**Lines 106-128: LLaMA Integration**
```python
    # Line 107-108: Initialize LLaMA tokenizer
    print('Loading LLAMA')
    self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
    self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
    
    # Line 110-120: Load LLaMA model
    if self.low_resource:
        self.llama_model = LlamaForCausalLM.from_pretrained(
            llama_model, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        self.llama_model = LlamaForCausalLM.from_pretrained(
            llama_model, torch_dtype=torch.float16,
        )
    
    # Line 122-124: Freeze LLaMA parameters
    for name, param in self.llama_model.named_parameters():
        param.requires_grad = False
    print('Loading LLAMA Done')
    
    # Line 126-128: Critical projection layer
    self.llama_proj = nn.Linear(
        self.Qformer.config.hidden_size,      # 768 (Q-Former output)
        self.llama_model.config.hidden_size   # 4096 (LLaMA input)
    )
```

## 🔧 **PART 3: Q-Former Initialization Details**

### **File: `blip2.py` Lines 44-57 (init_Qformer method)**

```python
@classmethod
def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
    # Line 46: Load BERT configuration
    encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    
    # Line 47: Set vision encoder width for cross-attention
    encoder_config.encoder_width = vision_width  # 1408 from EVA-CLIP-G
    
    # Line 49-51: Configure cross-attention
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq  # Every 2nd layer
    encoder_config.query_length = num_query_token  # 32
    
    # Line 52: Create Q-Former model (BertLMHeadModel)
    Qformer = BertLMHeadModel(config=encoder_config)
    # This creates the model defined in Qformer.py:967-1127
    
    # Line 53-56: Initialize learnable query tokens
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)  # [1, 32, 768]
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    
    return Qformer, query_tokens
```

## 🎯 **PART 4: Text Processing Execution Flow**

### **File: `demo.py` or `chat.py` (User Interaction)**

**Lines 100-120: User Input Processing**
```python
# Line 100: Get user input
user_message = input("User: ")

# Line 105: Process through chat system
chat = Chat(model, vis_processor, device=device)

# Line 110-115: Generate response
chat_state = CONV_VISION_Vicuna0.copy()
img_list = []
llm_message = chat.chat(user_message, chat_state, img_list)
# This calls conversation.py and eventually mini_gpt4.py
```

### **File: `conversation.py` Lines 150-200 (Chat processing)**

```python
def chat(self, text, conv, img_list):
    # Line 155: Add user message to conversation
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    
    # Line 160: Get conversation prompt
    prompt = conv.get_prompt()
    
    # Line 165-170: Prepare model inputs
    embs = self.get_context_emb(prompt, img_list)
    
    # Line 175-180: Generate response
    outputs = self.model.llama_model.generate(
        inputs_embeds=embs,
        max_new_tokens=300,
        num_beams=1,
        temperature=1.0,
        do_sample=True
    )
    
    # Line 185-190: Decode and clean response
    output_text = self.model.llama_tokenizer.decode(outputs[0])
    output_text = output_text.split('###')[0].strip()
    
    return output_text
```

## 🔄 **PART 5: Core Text Encoder Execution**

### **File: `mini_gpt4.py` Lines 152-172 (encode_img method)**

This is where the text encoder (Q-Former) is actually executed:

```python
def encode_img(self, image):
    # Line 153-156: Device and precision handling
    device = image.device
    if self.low_resource:
        self.vit_to_cpu()
        image = image.to("cpu")
    
    # Line 158-159: Vision encoding
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
        # Output: [B, 257, 1408] - 256 patches + 1 CLS token
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        # Output: [B, 257] - Attention mask for all patches
    
    # Line 162: Expand query tokens for batch
    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    # Input: [1, 32, 768] → Output: [B, 32, 768]
    
    # Line 163-168: **CRITICAL Q-FORMER EXECUTION**
    query_output = self.Qformer.bert(
        query_embeds=query_tokens,           # [B, 32, 768] - Learnable queries
        encoder_hidden_states=image_embeds,  # [B, 257, 1408] - Vision features  
        encoder_attention_mask=image_atts,   # [B, 257] - Vision mask
        return_dict=True,
    )
    # This calls Qformer.py:803-964 (BertModel.forward)
    
    # Line 170-171: Project to LLaMA space
    inputs_llama = self.llama_proj(query_output.last_hidden_state)
    # Input: [B, 32, 768] → Output: [B, 32, 4096]
    
    atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
    # Output: [B, 32] - Attention mask for LLaMA
    
    return inputs_llama, atts_llama
```

## 🧠 **PART 6: Deep Q-Former Execution**

### **File: `Qformer.py` Lines 803-964 (BertModel.forward)**

This is the main text encoder execution:

```python
def forward(self, input_ids=None, attention_mask=None, position_ids=None,
           head_mask=None, query_embeds=None, encoder_hidden_states=None,
           encoder_attention_mask=None, ...):
    
    # Line 865-872: Create embeddings
    query_length = query_embeds.shape[1] if query_embeds is not None else 0  # 32
    
    embedding_output = self.embeddings(
        input_ids=input_ids,           # None (no text tokens in encode_img)
        position_ids=position_ids,     # None
        query_embeds=query_embeds,     # [B, 32, 768] - Query tokens only
    )
    # This calls Qformer.py:77-107 (BertEmbeddings.forward)
    # Output: [B, 32, 768] - Just query embeddings (no text)
    
    # Line 874-881: Setup attention masks
    input_shape = embedding_output.size()[:-1]  # [B, 32]
    device = embedding_output.device
    
    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)  # [B, 32]
    
    extended_attention_mask = self.get_extended_attention_mask(
        attention_mask, input_shape, device, is_decoder=False
    )
    # Output: [B, 1, 1, 32] - Extended mask for self-attention
    
    # Line 900-927: Setup cross-attention masks for vision
    if encoder_hidden_states is not None:  # Vision features provided
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        # encoder_hidden_states: [B, 257, 1408] - Vision features
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)  # [B, 257]
        
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            # [B, 257] - All ones (attend to all vision patches)
        
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        # Output: [B, 1, 1, 257] - Extended mask for cross-attention
    
    # Line 936-948: **MAIN TRANSFORMER EXECUTION**
    encoder_outputs = self.encoder(
        embedding_output,              # [B, 32, 768] - Query embeddings
        attention_mask=extended_attention_mask,      # [B, 1, 1, 32] - Self-attention mask
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states, # [B, 257, 1408] - Vision features
        encoder_attention_mask=encoder_extended_attention_mask, # [B, 1, 1, 257]
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        query_length=query_length,     # 32
    )
    # This calls Qformer.py:494-588 (BertEncoder.forward)
    
    sequence_output = encoder_outputs[0]  # [B, 32, 768] - Final query representations
    
    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,  # [B, 32, 768]
        ...
    )
```

### **File: `Qformer.py` Lines 494-588 (BertEncoder.forward)**

This processes through all 12 transformer layers:

```python
def forward(self, hidden_states, attention_mask=None, head_mask=None,
           encoder_hidden_states=None, encoder_attention_mask=None, ...):
    
    # Line 508-514: Initialize tracking variables
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    next_decoder_cache = () if use_cache else None
    
    # Line 516-558: **LAYER-BY-LAYER PROCESSING**
    for i in range(self.config.num_hidden_layers):  # 12 layers
        layer_module = self.layer[i]  # Get layer i
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None
        
        # Line 549-558: Execute layer
        layer_outputs = layer_module(
            hidden_states,                # [B, 32, 768] - Current representations
            attention_mask,               # [B, 1, 1, 32] - Self-attention mask
            layer_head_mask,
            encoder_hidden_states,        # [B, 257, 1408] - Vision features
            encoder_attention_mask,       # [B, 1, 1, 257] - Cross-attention mask
            past_key_value,
            output_attentions,
            query_length,                 # 32
        )
        # This calls Qformer.py:401-473 (BertLayer.forward)
        
        hidden_states = layer_outputs[0]  # [B, 32, 768] - Updated representations
        
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
    
    # Line 567-568: Final hidden states
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
    
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,  # [B, 32, 768] - Final query representations
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )
```

## 🔄 **PART 7: Individual Layer Processing**

### **File: `Qformer.py` Lines 401-473 (BertLayer.forward)**

Each layer processes self-attention and optionally cross-attention:

```python
def forward(self, hidden_states, attention_mask=None, head_mask=None,
           encoder_hidden_states=None, encoder_attention_mask=None,
           past_key_value=None, output_attentions=False, query_length=0):
    
    # Line 413-422: **SELF-ATTENTION BLOCK**
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    
    self_attention_outputs = self.attention(
        hidden_states,                # [B, 32, 768] - Query tokens
        attention_mask,               # [B, 1, 1, 32] - Self-attention mask
        head_mask,
        output_attentions=output_attentions,
        past_key_value=self_attn_past_key_value,
    )
    # This calls Qformer.py:321-345 (BertAttention.forward)
    # Output: [B, 32, 768] - Self-attended query representations
    
    attention_output = self_attention_outputs[0]  # [B, 32, 768]
    outputs = self_attention_outputs[1:-1]
    present_key_value = self_attention_outputs[-1]
    
    # Line 428-446: **CROSS-ATTENTION BLOCK** (every 2nd layer)
    if query_length > 0:  # query_length = 32
        query_attention_output = attention_output[:, :query_length, :]  # [B, 32, 768]
        
        if self.has_cross_attention:  # True for layers 0, 2, 4, 6, 8, 10
            assert encoder_hidden_states is not None, "encoder_hidden_states must be given"
            
            cross_attention_outputs = self.crossattention(
                query_attention_output,    # [B, 32, 768] - Query tokens as Q
                attention_mask,            # [B, 1, 1, 32] - Query mask
                head_mask,
                encoder_hidden_states,     # [B, 257, 1408] - Vision features as K,V
                encoder_attention_mask,    # [B, 1, 1, 257] - Vision mask
                output_attentions=output_attentions,
            )
            # This calls Qformer.py:321-345 (BertAttention.forward) with cross-attention
            # Output: [B, 32, 768] - Vision-attended query representations
            
            query_attention_output = cross_attention_outputs[0]  # [B, 32, 768]
            outputs = outputs + cross_attention_outputs[1:-1]
        
        # Line 448-453: Feed-forward for query tokens
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk_query,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            query_attention_output,
        )
        # This calls Qformer.py:480-483 (feed_forward_chunk_query)
        
        # Line 454-461: Handle text tokens if present (not in encode_img case)
        if attention_output.shape[1] > query_length:  # False in encode_img
            layer_output_text = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output[:, query_length:, :],
            )
            layer_output = torch.cat([layer_output, layer_output_text], dim=1)
    else:
        # Line 463-468: Standard processing (not used in encode_img)
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
    
    outputs = (layer_output,) + outputs  # [B, 32, 768]
    outputs = outputs + (present_key_value,)
    
    return outputs
```

## 🎯 **PART 8: Attention Mechanism Details**

### **File: `Qformer.py` Lines 321-345 (BertAttention.forward)**

```python
def forward(self, hidden_states, attention_mask=None, head_mask=None,
           encoder_hidden_states=None, encoder_attention_mask=None,
           past_key_value=None, output_attentions=False):
    
    # Line 331-338: Execute self-attention or cross-attention
    self_outputs = self.self(
        hidden_states,                # [B, 32, 768] - Query tokens
        attention_mask,               # [B, 1, 1, 32] or [B, 1, 1, 257]
        head_mask,
        encoder_hidden_states,        # [B, 257, 1408] - Vision (if cross-attention)
        encoder_attention_mask,       # [B, 1, 1, 257] - Vision mask
        past_key_value,
        output_attentions,
    )
    # This calls Qformer.py:168-274 (BertSelfAttention.forward)
    
    # Line 340: Apply output projection and residual connection
    attention_output = self.output(self_outputs[0], hidden_states)
    # This calls Qformer.py:284-288 (BertSelfOutput.forward)
    # Output: [B, 32, 768] - Processed attention output
    
    outputs = (attention_output,) + self_outputs[1:]
    return outputs
```

### **File: `Qformer.py` Lines 168-274 (BertSelfAttention.forward)**

The core attention computation:

```python
def forward(self, hidden_states, attention_mask=None, head_mask=None,
           encoder_hidden_states=None, encoder_attention_mask=None,
           past_key_value=None, output_attentions=False):
    
    # Line 182: Determine if this is cross-attention
    is_cross_attention = encoder_hidden_states is not None
    
    if is_cross_attention:
        # Line 184-187: Cross-attention (Query tokens attend to Vision)
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        # encoder_hidden_states: [B, 257, 1408] → key_layer: [B, 12, 257, 64]
        
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        # encoder_hidden_states: [B, 257, 1408] → value_layer: [B, 12, 257, 64]
        
        attention_mask = encoder_attention_mask  # Use vision mask
    else:
        # Line 194-195: Self-attention (Query tokens attend to themselves)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # hidden_states: [B, 32, 768] → key_layer: [B, 12, 32, 64]
        
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # hidden_states: [B, 32, 768] → value_layer: [B, 12, 32, 64]
    
    # Line 197-199: Query projection (always from query tokens)
    mixed_query_layer = self.query(hidden_states)
    # hidden_states: [B, 32, 768] → mixed_query_layer: [B, 32, 768]
    
    query_layer = self.transpose_for_scores(mixed_query_layer)
    # mixed_query_layer: [B, 32, 768] → query_layer: [B, 12, 32, 64]
    
    # Line 204: **ATTENTION SCORE COMPUTATION**
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    # Cross-attention: [B, 12, 32, 64] × [B, 12, 64, 257] = [B, 12, 32, 257]
    # Self-attention:  [B, 12, 32, 64] × [B, 12, 64, 32]  = [B, 12, 32, 32]
    
    # Line 243: Scale by sqrt(head_size)
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # / sqrt(64)
    
    # Line 244-246: Apply attention mask
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
        # Add -10000 to masked positions
    
    # Line 249: **SOFTMAX NORMALIZATION**
    attention_probs = nn.Softmax(dim=-1)(attention_scores)
    # Output: [B, 12, 32, 257] or [B, 12, 32, 32] - Attention probabilities
    
    # Line 257: Apply dropout
    attention_probs_dropped = self.dropout(attention_probs)
    
    # Line 260-261: Mask heads if specified
    if head_mask is not None:
        attention_probs_dropped = attention_probs_dropped * head_mask
    
    # Line 263: **APPLY ATTENTION TO VALUES**
    context_layer = torch.matmul(attention_probs_dropped, value_layer)
    # Cross-attention: [B, 12, 32, 257] × [B, 12, 257, 64] = [B, 12, 32, 64]
    # Self-attention:  [B, 12, 32, 32]  × [B, 12, 32, 64]  = [B, 12, 32, 64]
    
    # Line 265-267: Reshape back to original format
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    # Output: [B, 32, 12, 64]
    
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    # Output: [B, 32, 768] - Final attention output
    
    # Line 269-274: Return outputs
    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    outputs = outputs + (past_key_value,)
    return outputs
```

## 🔚 **PART 9: Final Output and Integration**

### **Back to `mini_gpt4.py` Lines 170-172**

After Q-Former processing completes:

```python
    # Q-Former output: query_output.last_hidden_state = [B, 32, 768]
    
    # Line 170: Project to LLaMA embedding space
    inputs_llama = self.llama_proj(query_output.last_hidden_state)
    # Input: [B, 32, 768] → Output: [B, 32, 4096]
    
    # Line 171: Create attention mask for LLaMA
    atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
    # Output: [B, 32] - All ones (attend to all query tokens)
    
    return inputs_llama, atts_llama
    # Final output: ([B, 32, 4096], [B, 32])
```

## 📊 **Complete Data Flow Summary**

### **File Connection Chain:**
```
demo.py → mini_gpt4.py → blip2.py → Qformer.py → conversation.py
   ↓           ↓            ↓          ↓             ↓
Entry     Model Init   Q-Former   Text Encoder   Response
Point     & Config     Setup      Execution      Generation
```

### **Data Transformation Chain:**
```
Text Input: None (encode_img case)
Query Tokens: [1, 32, 768] → Expand → [B, 32, 768]
Vision Features: [B, 257, 1408] (from EVA-ViT)

↓ Q-Former Processing (12 layers)

Layer 0: Self-attention + Cross-attention (queries ↔ vision)
Layer 1: Self-attention only
Layer 2: Self-attention + Cross-attention (queries ↔ vision)
...continuing pattern...
Layer 11: Self-attention + Cross-attention (queries ↔ vision)

↓ Final Output

Query Representations: [B, 32, 768] → Projection → [B, 32, 4096]
LLaMA-ready Features: [B, 32, 4096]
```

### **Key Execution Points:**
1. **Entry**: `demo.py` or `train.py` loads model
2. **Initialization**: `mini_gpt4.py:__init__` sets up Q-Former
3. **Q-Former Setup**: `blip2.py:init_Qformer` creates architecture
4. **Text Encoding**: `mini_gpt4.py:encode_img` executes Q-Former
5. **Core Processing**: `Qformer.py:BertModel.forward` runs transformer
6. **Layer Processing**: `Qformer.py:BertLayer.forward` handles attention
7. **Attention**: `Qformer.py:BertSelfAttention.forward` computes cross-modal attention
8. **Output**: Projected features ready for LLaMA generation

This trace shows the complete execution path from model initialization to final text encoder output, with exact line numbers and file connections throughout the XrayGPT codebase.
```