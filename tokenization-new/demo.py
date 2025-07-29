"""
Example usage of the Generic Tokenizer 
This script demonstrates how to use the tokenization system
based on your reference code.
"""

from tokenizer_class import Tokenizer, quick_tokenize
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)


def main():
    """Demonstrate basic usage of the tokenizer """
    
    print("=== Generic Tokenizer Demo ===\n")
    
    # Create instance
    tokenizer = Tokenizer()

    # Show available tokenizers first
    print(" Available Tokenizers:")
    print("-" * 40)
    available_tokenizers = tokenizer.list_available_tokenizers()
    for i, tokenizer_name in enumerate(available_tokenizers, 1):
        print(f"{i:2d}. {tokenizer_name}")
    print()
    
    # Ask user to select a tokenizer
    print(" Select a tokenizer for demonstration:")
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
    
    print(f"\n Selected tokenizer: {selected_tokenizer}")
    
    # Sample text (similar to your reference)
    input_text = "This is a sample radiology report for testing."
    
    # Using Tokenizer class
    print(f"\nUsing Tokenizer with {selected_tokenizer}:")
    print("-" * 40)
    
    # Load selected tokenizer
    print(f"Loading {selected_tokenizer} tokenizer...")
    try:
        tokenizer.load_tokenizer(selected_tokenizer)
    except Exception as e:
        print(f"Error loading {selected_tokenizer}: {e}")
        print("Falling back to BERT tokenizer...")
        tokenizer.load_tokenizer('bert')
        selected_tokenizer = 'bert'
    
    # Tokenize text
    result = tokenizer.tokenize(input_text, return_tensors=False)
    
    print(f"Text: {result['text']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Token IDs: {result['token_ids']}")
    
    # Show basic tokenizer information
    print(f"\nTokenizer Information:")
    print("-" * 40)
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Special tokens: {list(tokenizer.get_special_tokens().keys())}")
    
    # Demonstrate decoding
    print(f"\nDecoding demonstration:")
    print("-" * 40)
    decoded_text = tokenizer.decode(result['token_ids'])
    print(f"Original text: {input_text}")
    print(f"Decoded text:  {decoded_text}")
    print(f"Texts match: {decoded_text.strip() == input_text}")




if __name__ == "__main__":
    main()