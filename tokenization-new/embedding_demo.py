"""
Embedding Demo
Demonstrates the embedding functionality integrated with the tokenizer system.
Shows how to create embeddings using different tokenizers and models.
"""

import torch
from embedding_class import EmbeddingManager
from tokenizer_class import Tokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate embedding functionality."""
    
    print("=== Generic Embedding System Demo ===\n")
    
    # Sample text (same as XrayGPT reference)
    sample_text = "This is a sample radiology report for testing."
    
    # Create embedding manager
    embedding_manager = EmbeddingManager()
    
    # Show available tokenizers
    print("Available Tokenizers for Embedding:")
    print("-" * 50)
    available_tokenizers = embedding_manager.list_available_tokenizers()
    for i, tokenizer_name in enumerate(available_tokenizers, 1):
        print(f"{i:2d}. {tokenizer_name}")
    print()
    
    # Ask user to select tokenizer
    print("Select a tokenizer for embedding demonstration:")
    try:
        choice = input(f"Enter your choice (1-{len(available_tokenizers)}) or press Enter for 'bert': ").strip()
        
        if not choice:
            selected_tokenizer = 'bert'
        else:
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_tokenizers):
                selected_tokenizer = available_tokenizers[choice_num - 1]
            else:
                print("Invalid choice, using 'bert' as default")
                selected_tokenizer = 'bert'
    except (ValueError, KeyboardInterrupt):
        print("Using 'bert' as default")
        selected_tokenizer = 'bert'
    
    print(f"\nSelected tokenizer: {selected_tokenizer}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load embedding model
    print(f"\nLoading embedding model for {selected_tokenizer}...")
    try:
        embedding_manager.load_embedding_model(selected_tokenizer, device=device)
        print("✓ Embedding model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading embedding model: {e}")
        print("Falling back to BERT...")
        embedding_manager.load_embedding_model('bert', device=device)
        selected_tokenizer = 'bert'
    
    # Show model information
    print(f"\nEmbedding Model Information:")
    print("-" * 50)
    model_info = embedding_manager.get_embedding_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")
    
    # Create embeddings
    print(f"\nCreating embeddings for text:")
    print(f"Text: '{sample_text}'")
    print("-" * 50)
    
    try:
        result = embedding_manager.create_embeddings(
            sample_text, 
            return_tokens=True,
            return_attention_mask=True,
            device=device
        )
        
        # Display results
        print(f"✓ Embeddings created successfully")
        print(f"Tokens: {result['tokens']}")
        print(f"Token IDs: {result['token_ids']}")
        print(f"Embedding shape: {result['embedding_shape']}")
        print(f"Hidden size: {result['hidden_size']}")
        
        # Show first few dimensions of first token embedding
        embeddings = result['embeddings']
        print(f"\nFirst token embedding (first 10 dimensions):")
        print(embeddings[0, 0, :10])
        print(f"\nAll token embedding:")
        print(embeddings)
        
        # Show attention mask if available
        if 'attention_mask' in result:
            print(f"\nAttention mask: {result['attention_mask']}")
        
    except Exception as e:
        print(f"✗ Error creating embeddings: {e}")
        return
    
    # Additional embedding information
    print(f"\nAdditional Information:")
    print("-" * 40)
    print(f"Model successfully loaded and ready for use!")
    print(f"You can now use this embedding model in your applications.")
    
    print(f"\n" + "="*60)
    print("Demo completed!")
    print("="*60)


if __name__ == "__main__":
    main()