import torch
from xraygpt.models.Qformer import BertSelfAttention
from transformers.models.bert.configuration_bert import BertConfig

def test_self_attention_query_tokens():
    # Configuration (align with Qformer/BERT)
    batch_size = 2
    num_query_tokens = 8
    hidden_size = 768
    num_attention_heads = 12

    # Create BERT config for Qformer
    config = BertConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        vocab_size=30522,  # not used here
        max_position_embeddings=32,  # not used here
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )

    # Create dummy query tokens (batch, num_query_tokens, hidden_size)
    query_tokens = torch.randn(batch_size, num_query_tokens, hidden_size)

    # Initialize BertSelfAttention (is_cross_attention=False for self-attention)
    self_attention = BertSelfAttention(config, is_cross_attention=False)

    # Forward pass: query tokens attend to each other
    output = self_attention(query_tokens)

    # The first element of output is (context_layer,)
    context_layer = output[0]
    print("Input shape (query tokens):", query_tokens.shape)
    print("Input (query tokens):", query_tokens)

    print("Output shape (context layer):", context_layer.shape)
    print("Output (context layer):", context_layer)
    
    assert context_layer.shape == (batch_size, num_query_tokens, hidden_size), "Shape mismatch!"

if __name__ == "__main__":
    test_self_attention_query_tokens()