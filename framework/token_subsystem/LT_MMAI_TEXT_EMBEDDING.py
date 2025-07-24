import torch
from transformers import BertTokenizer, BertConfig
from torch import nn

# BertEmbeddings from xraygpt/models/Qformer.py
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.config = config

    def forward(self, input_ids=None, position_ids=None, query_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length].clone()

        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings

            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# # Use the same tokenizer as in xraygpt/models/blip2.py
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens({"bos_token": "[DEC]"})

# Example input
text = "This is a sample radiology report for testing."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

# Create config and embedding layer
config = BertConfig()
embeddings = BertEmbeddings(config)

# Get embeddings
emb_out = embeddings(input_ids)
print("Tokens:", tokenizer.tokenize(text))
print("Token IDs:", input_ids)
print("Embedding output shape:", emb_out.shape)  # [batch_size, seq_len, hidden_size]
print("First token embedding vector:\n", emb_out[0,0,:5])  # Print first 5 dims of first token as an example
print("All embeddings:\n", emb_out)