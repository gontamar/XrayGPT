import torch
from xraygpt.models.Qformer import BertSelfAttention
from transformers.models.bert.configuration_bert import BertConfig

def test_cross_attention_query_to_image():
    batch_size = 2
    num_query_tokens = 8
    num_image_tokens = 49
    hidden_size = 768
    num_attention_heads = 12

    # Add encoder_width to config!
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
    config.encoder_width = hidden_size  # This fixes the error

    query_tokens = torch.randn(batch_size, num_query_tokens, hidden_size)
    image_features = torch.randn(batch_size, num_image_tokens, hidden_size)

    cross_attention = BertSelfAttention(config, is_cross_attention=True)

    output = cross_attention(
        hidden_states=query_tokens,
        encoder_hidden_states=image_features
    )
    context_layer = output[0]
    print("Input shape (query tokens):", query_tokens.shape)
    print("Input (query tokens):", query_tokens)

    print("Input shape (image features):", image_features.shape)
    print("Input (image features):", image_features)

    print("Output shape (context layer):", context_layer.shape)
    print("Output (context layer):", context_layer)
    assert context_layer.shape == (batch_size, num_query_tokens, hidden_size), "Shape mismatch!"

if __name__ == "__main__":
    test_cross_attention_query_to_image()