# XrayGPT BERT Token Processing Flow: Q-Former Analysis

## Overview
This document traces the complete flow of tokens through the BERT-based Q-Former component in XrayGPT, showing how query tokens and text tokens are processed, transformed, and integrated with visual features.

## BERT Token Processing Pipeline in Q-Former

### 1. Q-Former Initialization and Setup
**Location**: `blip2.py:45-57` and `mini_gpt4.py:85-104`

#### A. Q-Former Model Creation
```python
# blip2.py:45-57
@classmethod
def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
    encoder_config = BertConfig.from_pretrained("bert-base-uncased")
    encoder_config.encoder_width = vision_width
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_token
    Qformer = BertLMHeadModel(config=encoder_config)
    query_tokens = nn.Parameter(
        torch.zeros(1, num_query_token, encoder_config.hidden_size)
    )
    query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
    return Qformer, query_tokens
```

**Token State**: 
- `query_tokens`: Learnable parameters of shape `[1, 32, 768]` (32 query tokens, 768-dim embeddings)
- BERT config: `bert-base-uncased` with cross-attention enabled

#### B. Q-Former Component Removal
```python
# mini_gpt4.py:89-95
self.Qformer.cls = None
self.Qformer.bert.embeddings.word_embeddings = None
self.Qformer.bert.embeddings.position_embeddings = None
for layer in self.Qformer.bert.encoder.layer:
    layer.output = None
    layer.intermediate = None
```

**Purpose**: Removes unused components since Q-Former only processes query tokens, not text tokens.

### 2. BERT Embedding Processing
**Location**: `Qformer.py:77-107`

#### A. Query Embedding Injection
```python
# Qformer.py:100-103
if query_embeds is not None:
    embeddings = torch.cat((query_embeds, embeddings), dim=1)
else:
    embeddings = query_embeds
```

**Token Flow**:
- Input: `query_embeds` of shape `[batch_size, 32, 768]`
- Process: Concatenated with any input embeddings
- Output: Combined embedding tensor

#### B. Layer Normalization and Dropout
```python
# Qformer.py:105-107
embeddings = self.LayerNorm(embeddings)
embeddings = self.dropout(embeddings)
```

**Token State**: Normalized and regularized query embeddings ready for BERT processing.

### 3. BERT Encoder Token Processing
**Location**: `Qformer.py:494-588`

#### A. Multi-Layer Processing Loop
```python
# Qformer.py:516-558
for i in range(self.config.num_hidden_layers):
    layer_module = self.layer[i]
    layer_outputs = layer_module(
        hidden_states,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        past_key_value,
        output_attentions,
        query_length,
    )
    hidden_states = layer_outputs[0]
```

**Token Processing**: Each BERT layer processes the hidden states through:
1. Self-attention
2. Cross-attention (with vision features)
3. Feed-forward networks

### 4. BERT Layer Token Processing
**Location**: `Qformer.py:401-473`

#### A. Self-Attention Processing
```python
# Qformer.py:416-423
self_attention_outputs = self.attention(
    hidden_states,
    attention_mask,
    head_mask,
    output_attentions=output_attentions,
    past_key_value=self_attn_past_key_value,
)
attention_output = self_attention_outputs[0]
```

**Token Flow**: Query tokens attend to each other through multi-head self-attention.

#### B. Cross-Attention with Vision Features
```python
# Qformer.py:428-446
if query_length > 0:
    query_attention_output = attention_output[:, :query_length, :]
    
    if self.has_cross_attention:
        cross_attention_outputs = self.crossattention(
            query_attention_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,  # Vision features
            encoder_attention_mask,
            output_attentions=output_attentions,
        )
        query_attention_output = cross_attention_outputs[0]
```

**Token Processing**:
- Extracts query tokens from attention output
- Cross-attends query tokens with vision encoder features
- Fuses visual and query information

#### C. Feed-Forward Processing
```python
# Qformer.py:448-461
layer_output = apply_chunking_to_forward(
    self.feed_forward_chunk_query,
    self.chunk_size_feed_forward,
    self.seq_len_dim,
    query_attention_output,
)
```

**Token State**: Query tokens processed through specialized feed-forward networks.

### 5. BERT Self-Attention Token Processing
**Location**: `Qformer.py:168-274`

#### A. Query, Key, Value Computation
```python
# Qformer.py:184-199
if is_cross_attention:
    key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
    value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
    attention_mask = encoder_attention_mask
else:
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    value_layer = self.transpose_for_scores(self.value(hidden_states))

mixed_query_layer = self.query(hidden_states)
query_layer = self.transpose_for_scores(mixed_query_layer)
```

**Token Processing**:
- For cross-attention: Keys/Values from vision features, Queries from BERT tokens
- For self-attention: All from BERT hidden states
- Reshapes for multi-head attention: `[batch, seq_len, hidden] → [batch, heads, seq_len, head_dim]`

#### B. Attention Score Computation
```python
# Qformer.py:204-249
attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
attention_scores = attention_scores / math.sqrt(self.attention_head_size)

if attention_mask is not None:
    attention_scores = attention_scores + attention_mask

attention_probs = nn.Softmax(dim=-1)(attention_scores)
attention_probs_dropped = self.dropout(attention_probs)
```

**Token Flow**: Computes attention weights between query tokens and key tokens.

#### C. Context Vector Computation
```python
# Qformer.py:263-267
context_layer = torch.matmul(attention_probs_dropped, value_layer)
context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
context_layer = context_layer.view(*new_context_layer_shape)
```

**Token State**: Weighted combination of value vectors, reshaped back to `[batch, seq_len, hidden_size]`.

### 6. Vision-Language Token Integration
**Location**: `mini_gpt4.py:162-172`

#### A. Q-Former Forward Pass with Vision Features
```python
# mini_gpt4.py:163-168
query_output = self.Qformer.bert(
    query_embeds=query_tokens,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=image_atts,
    return_dict=True,
)
```

**Token Processing**:
- Input: `query_tokens` [batch, 32, 768], `image_embeds` [batch, 257, 1408]
- Process: BERT processes query tokens with cross-attention to image features
- Output: `query_output.last_hidden_state` [batch, 32, 768]

#### B. Projection to LLaMA Space
```python
# mini_gpt4.py:170
inputs_llama = self.llama_proj(query_output.last_hidden_state)
```

**Token Flow**: Projects Q-Former output from BERT space (768-dim) to LLaMA space (4096-dim).

### 7. BERT Token Processing in Different Modes

#### A. Training Mode (Forward Pass)
**Location**: `mini_gpt4.py:190-250`

```python
# Text tokenization and embedding
to_regress_tokens = self.llama_tokenizer(
    text,
    return_tensors="pt",
    padding="longest",
    truncation=True,
    max_length=self.max_txt_len,
    add_special_tokens=False
).to(image.device)

to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
```

**Token Processing**: Text tokens are processed by LLaMA tokenizer and embedder, not BERT.

#### B. Inference Mode (Chat)
**Location**: `conversation.py:206-219`

```python
# Context embedding preparation
seg_tokens = [
    self.model.llama_tokenizer(
        seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
    for i, seg in enumerate(prompt_segs)
]
seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
```

**Token Flow**: Text segments tokenized by LLaMA tokenizer, embedded, and concatenated with Q-Former outputs.

## BERT vs LLaMA Token Processing Separation

### BERT Domain (Q-Former)
- **Input**: Query tokens (learnable parameters)
- **Processing**: Self-attention + Cross-attention with vision features
- **Output**: Vision-conditioned query representations
- **Tokenizer**: BERT tokenizer (unused in practice)
- **Vocabulary**: BERT vocabulary (30,522 tokens)

### LLaMA Domain (Text Generation)
- **Input**: Text prompts and responses
- **Processing**: Causal language modeling
- **Output**: Generated text tokens
- **Tokenizer**: LLaMA tokenizer
- **Vocabulary**: LLaMA vocabulary (32,000 tokens)

## Token Flow Summary

```
Vision Features [batch, 257, 1408]
         ↓
Query Tokens [batch, 32, 768] ──→ BERT Embeddings
         ↓                              ↓
BERT Self-Attention ←──────────────────────
         ↓
BERT Cross-Attention ←── Vision Features
         ↓
BERT Feed-Forward
         ↓
Q-Former Output [batch, 32, 768]
         ↓
LLaMA Projection [batch, 32, 4096]
         ↓
Concatenate with LLaMA Text Embeddings
         ↓
LLaMA Generation
```

## Key BERT Token Processing Characteristics

### 1. Query-Only Processing
- No text tokens processed through BERT
- Only learnable query tokens used
- Word/position embeddings disabled

### 2. Cross-Modal Attention
- Query tokens attend to vision features
- Bidirectional attention between modalities
- Vision features as keys/values, queries from BERT

### 3. Layer-wise Processing
- 12 BERT layers with alternating cross-attention
- Cross-attention every 2nd layer (configurable)
- Residual connections and layer normalization

### 4. Token Dimension Transformations
- BERT hidden size: 768 dimensions
- Vision feature size: 1408 dimensions  
- LLaMA hidden size: 4096 dimensions
- Projection layers handle dimension mismatches

This BERT token processing pipeline enables XrayGPT to effectively bridge visual and textual modalities through the Q-Former architecture, creating rich multimodal representations for medical image understanding.