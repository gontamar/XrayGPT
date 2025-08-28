# XrayGPT Text Encoder: Complete Line-by-Line Code Trace

This document provides a detailed line-by-line trace of the text encoder (Q-Former) used in XrayGPT, showing how text input flows through the BERT-based architecture with cross-modal attention.

## 📋 **Text Encoder Overview**

The XrayGPT text encoder is based on **Q-Former**, which is a BERT-based model with:
- **32 learnable query tokens** that bridge vision and language
- **Cross-attention layers** every 2nd layer to attend to vision features
- **768-dimensional hidden states** (BERT-base size)
- **12 attention heads** with 64-dimensional head size

## 🔍 **Complete Code Trace**

### **STEP 1: Q-Former Initialization**

**File: `blip2.py` Lines 44-57**

```python
@classmethod
def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
    # Line 46: Load BERT-base configuration
    encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    
    # Line 47: Set vision encoder width for cross-attention
    encoder_config.encoder_width = vision_width  # 1408 for EVA-CLIP-G
    
    # Line 49-51: Configure cross-attention
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq  # Every 2nd layer
    encoder_config.query_length = num_query_token  # 32 query tokens
    
    # Line 52: Create Q-Former model
    Qformer = BertLMHeadModel(config=encoder_config)
    
    # Lines 53-56: Initialize learnable query tokens
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)  # [1, 32, 768]
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    
    return Qformer, query_tokens
```

### **STEP 2: Text Embedding Layer**

**File: `Qformer.py` Lines 50-107 (BertEmbeddings class)**

#### **2.1 Embedding Initialization (Lines 53-75)**

```python
def __init__(self, config):
    super().__init__()
    
    # Line 55-57: Word embeddings (vocabulary → hidden_size)
    self.word_embeddings = nn.Embedding(
        config.vocab_size,      # 30522 (BERT vocab)
        config.hidden_size,     # 768
        padding_idx=config.pad_token_id  # 0
    )
    
    # Line 58-60: Position embeddings (position → hidden_size)
    self.position_embeddings = nn.Embedding(
        config.max_position_embeddings,  # 512
        config.hidden_size               # 768
    )
    
    # Line 64: Layer normalization
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    # Line 65: Dropout
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # Lines 68-70: Position IDs buffer
    self.register_buffer(
        "position_ids", 
        torch.arange(config.max_position_embeddings).expand((1, -1))
    )
```

#### **2.2 Embedding Forward Pass (Lines 77-107)**

```python
def forward(self, input_ids=None, position_ids=None, query_embeds=None, past_key_values_length=0):
    
    # Lines 84-87: Get sequence length
    if input_ids is not None:
        seq_length = input_ids.size()[1]  # Text sequence length
    else:
        seq_length = 0
    
    # Lines 89-93: Create position IDs if not provided
    if position_ids is None:
        position_ids = self.position_ids[
            :, past_key_values_length : seq_length + past_key_values_length
        ].clone()
        # Output: [1, seq_length] - Position indices
    
    # Lines 94-98: Create word + position embeddings
    if input_ids is not None:
        # Line 95: Convert token IDs to embeddings
        embeddings = self.word_embeddings(input_ids)
        # Input: [B, seq_length] → Output: [B, seq_length, 768]
        
        if self.position_embedding_type == "absolute":
            # Line 97: Add position embeddings
            position_embeddings = self.position_embeddings(position_ids)
            # Output: [1, seq_length, 768]
            
            # Line 98: Add word + position embeddings
            embeddings = embeddings + position_embeddings
            # Output: [B, seq_length, 768]
    
        # Lines 100-101: Concatenate query tokens with text embeddings
        if query_embeds is not None:
            embeddings = torch.cat((query_embeds, embeddings), dim=1)
            # query_embeds: [B, 32, 768] + embeddings: [B, seq_length, 768]
            # Output: [B, 32 + seq_length, 768]
    else:
        # Line 103: Use only query embeddings (no text)
        embeddings = query_embeds
        # Output: [B, 32, 768]
    
    # Line 105: Apply layer normalization
    embeddings = self.LayerNorm(embeddings)
    
    # Line 106: Apply dropout
    embeddings = self.dropout(embeddings)
    
    # Line 107: Return final embeddings
    return embeddings
    # Output: [B, 32 + seq_length, 768] or [B, 32, 768]
```

### **STEP 3: Self-Attention Mechanism**

**File: `Qformer.py` Lines 110-250 (BertSelfAttention class)**

#### **3.1 Attention Initialization (Lines 111-146)**

```python
def __init__(self, config, is_cross_attention):
    super().__init__()
    
    # Lines 114-120: Validate hidden size
    if config.hidden_size % config.num_attention_heads != 0:
        raise ValueError("Hidden size must be divisible by num_attention_heads")
    
    # Lines 122-124: Attention configuration
    self.num_attention_heads = config.num_attention_heads  # 12
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 64
    self.all_head_size = self.num_attention_heads * self.attention_head_size  # 768
    
    # Line 126: Query projection (always from hidden_size)
    self.query = nn.Linear(config.hidden_size, self.all_head_size)  # 768 → 768
    
    # Lines 127-132: Key/Value projections
    if is_cross_attention:
        # Cross-attention: K,V from vision features
        self.key = nn.Linear(config.encoder_width, self.all_head_size)    # 1408 → 768
        self.value = nn.Linear(config.encoder_width, self.all_head_size)  # 1408 → 768
    else:
        # Self-attention: K,V from text features
        self.key = nn.Linear(config.hidden_size, self.all_head_size)      # 768 → 768
        self.value = nn.Linear(config.hidden_size, self.all_head_size)    # 768 → 768
    
    # Line 134: Attention dropout
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
```

#### **3.2 Attention Forward Pass (Lines 168-250)**

```python
def forward(self, hidden_states, attention_mask=None, head_mask=None, 
           encoder_hidden_states=None, encoder_attention_mask=None, 
           past_key_value=None, output_attentions=False):
    
    # Line 182: Check if this is cross-attention
    is_cross_attention = encoder_hidden_states is not None
    
    if is_cross_attention:
        # Lines 184-187: Cross-attention (text queries attend to vision)
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        # encoder_hidden_states: [B, 257, 1408] → key_layer: [B, 12, 257, 64]
        
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        # encoder_hidden_states: [B, 257, 1408] → value_layer: [B, 12, 257, 64]
        
        attention_mask = encoder_attention_mask  # Use vision attention mask
        
    else:
        # Lines 194-195: Self-attention (text attends to text)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # hidden_states: [B, seq_len, 768] → key_layer: [B, 12, seq_len, 64]
        
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # hidden_states: [B, seq_len, 768] → value_layer: [B, 12, seq_len, 64]
    
    # Line 197: Query projection
    mixed_query_layer = self.query(hidden_states)
    # hidden_states: [B, seq_len, 768] → mixed_query_layer: [B, seq_len, 768]
    
    # Line 199: Reshape query for multi-head attention
    query_layer = self.transpose_for_scores(mixed_query_layer)
    # mixed_query_layer: [B, seq_len, 768] → query_layer: [B, 12, seq_len, 64]
    
    # Lines 201-210: Compute attention scores
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    # query_layer: [B, 12, seq_len, 64] × key_layer: [B, 12, 64, kv_len]
    # Output: [B, 12, seq_len, kv_len]
    
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    # Scale by sqrt(64) = 8
    
    # Lines 212-220: Apply attention mask
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
        # attention_mask: [B, 1, 1, kv_len] with -10000 for masked positions
    
    # Line 222: Softmax normalization
    attention_probs = nn.Softmax(dim=-1)(attention_scores)
    # Output: [B, 12, seq_len, kv_len] - Attention probabilities
    
    # Line 224: Apply attention dropout
    attention_probs = self.dropout(attention_probs)
    
    # Lines 226-230: Apply attention to values
    context_layer = torch.matmul(attention_probs, value_layer)
    # attention_probs: [B, 12, seq_len, kv_len] × value_layer: [B, 12, kv_len, 64]
    # Output: [B, 12, seq_len, 64]
    
    # Lines 232-237: Reshape back to original format
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    # Output: [B, seq_len, 12, 64]
    
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)
    # Output: [B, seq_len, 768]
    
    # Lines 239-243: Return outputs
    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs
```

### **STEP 4: Transformer Layer**

**File: `Qformer.py` Lines 400-500 (BertLayer class)**

#### **4.1 Layer Forward Pass**

```python
def forward(self, hidden_states, attention_mask=None, head_mask=None,
           encoder_hidden_states=None, encoder_attention_mask=None,
           past_key_value=None, output_attentions=False, query_length=0):
    
    # Self-attention block
    self_attention_outputs = self.attention(
        hidden_states,           # [B, seq_len, 768] - Query tokens + text
        attention_mask,          # [B, 1, 1, seq_len] - Self-attention mask
        head_mask,
        output_attentions=output_attentions,
    )
    attention_output = self_attention_outputs[0]  # [B, seq_len, 768]
    
    # Cross-attention block (every 2nd layer)
    if self.has_cross_attention:
        cross_attention_outputs = self.crossattention(
            attention_output,        # [B, seq_len, 768] - Text queries
            attention_mask,
            head_mask,
            encoder_hidden_states,   # [B, 257, 1408] - Vision features
            encoder_attention_mask,  # [B, 1, 1, 257] - Vision mask
            output_attentions=output_attentions,
        )
        attention_output = cross_attention_outputs[0]  # [B, seq_len, 768]
    
    # Feed-forward network
    layer_output = apply_chunking_to_forward(
        self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim,
        attention_output
    )
    # Output: [B, seq_len, 768]
    
    return (layer_output,)
```

### **STEP 5: Complete Q-Former Model**

**File: `Qformer.py` Lines 600-800 (BertModel class)**

#### **5.1 Q-Former Forward Pass**

```python
def forward(self, input_ids=None, attention_mask=None, position_ids=None,
           head_mask=None, query_embeds=None, encoder_hidden_states=None,
           encoder_attention_mask=None, past_key_values=None,
           use_cache=None, output_attentions=None, output_hidden_states=None,
           return_dict=None, is_decoder=False):
    
    # Step 1: Create embeddings
    embedding_output = self.embeddings(
        input_ids=input_ids,           # [B, text_len] - Text token IDs
        position_ids=position_ids,     # [B, text_len] - Position IDs
        query_embeds=query_embeds,     # [B, 32, 768] - Query tokens
    )
    # Output: [B, 32 + text_len, 768] - Query + text embeddings
    
    # Step 2: Create attention mask
    input_shape = embedding_output.size()[:-1]  # [B, 32 + text_len]
    device = embedding_output.device
    
    if attention_mask is None:
        attention_mask = torch.ones(input_shape, device=device)
    
    # Extend attention mask for cross-attention
    extended_attention_mask = self.get_extended_attention_mask(
        attention_mask, input_shape, device
    )
    # Output: [B, 1, 1, 32 + text_len] - Extended mask
    
    # Step 3: Process through transformer layers
    encoder_outputs = self.encoder(
        embedding_output,              # [B, 32 + text_len, 768]
        attention_mask=extended_attention_mask,  # [B, 1, 1, 32 + text_len]
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,  # [B, 257, 1408] - Vision
        encoder_attention_mask=encoder_attention_mask,  # [B, 1, 1, 257]
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    
    sequence_output = encoder_outputs[0]  # [B, 32 + text_len, 768]
    
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=sequence_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )
```

### **STEP 6: XrayGPT Integration**

**File: `mini_gpt4.py` Lines 162-168**

```python
# In encode_img method - Q-Former processing
query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
# Expand query tokens: [1, 32, 768] → [B, 32, 768]

query_output = self.Qformer.bert(
    query_embeds=query_tokens,           # [B, 32, 768] - Learnable queries
    encoder_hidden_states=image_embeds,  # [B, 257, 1408] - Vision features
    encoder_attention_mask=image_atts,   # [B, 257] - Vision attention mask
    return_dict=True,
)
# Output: query_output.last_hidden_state = [B, 32, 768]
```

## 🔄 **Complete Text Encoder Data Flow**

### **Input Processing:**
1. **Text Tokens:** `[B, text_len]` → Word Embeddings → `[B, text_len, 768]`
2. **Query Tokens:** `[1, 32, 768]` → Expand → `[B, 32, 768]`
3. **Concatenation:** Query + Text → `[B, 32 + text_len, 768]`

### **Transformer Processing:**
4. **Self-Attention:** Text attends to itself and query tokens
5. **Cross-Attention:** Query tokens attend to vision features `[B, 257, 1408]`
6. **Layer-by-Layer:** 12 transformer layers with cross-attention every 2nd layer

### **Output:**
7. **Final Features:** `[B, 32, 768]` - Query token representations with vision-language alignment

### **Key Dimensions:**
- **Text Embeddings:** 768D (BERT-base)
- **Vision Features:** 1408D (EVA-CLIP-G)
- **Query Tokens:** 32 × 768D learnable embeddings
- **Attention Heads:** 12 heads × 64D each
- **Cross-Attention:** Every 2nd layer (layers 2, 4, 6, 8, 10, 12)

This trace shows how XrayGPT's text encoder uses Q-Former to create vision-aware text representations through learnable query tokens and cross-modal attention.