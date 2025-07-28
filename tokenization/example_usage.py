"""
Example usage of the Generic Tokenizer Manager
This script demonstrates how to use the tokenization system
based on your reference code.
"""

from tokenizer_manager import TokenizerManager, quick_tokenize
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)


def main():
    """Demonstrate various usage patterns of the tokenizer manager."""
    
    print("=== Generic Tokenizer Manager Demo ===\n")
    
    # Create manager instance
    manager = TokenizerManager()
    
    # Show available tokenizers first
    print("📋 Available Tokenizers:")
    print("-" * 40)
    available_tokenizers = manager.list_available_tokenizers()
    for i, tokenizer_name in enumerate(available_tokenizers, 1):
        print(f"{i:2d}. {tokenizer_name}")
    print()
    
    # Ask user to select a tokenizer
    print("🎯 Select a tokenizer for demonstration:")
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
    
    print(f"\n✅ Selected tokenizer: {selected_tokenizer}")
    
    # Sample text (similar to your reference)
    input_text = "This is a sample radiology report for testing."
    
    # Method 1: Using TokenizerManager class (recommended for complex usage)
    print(f"\n1. Using TokenizerManager with {selected_tokenizer}:")
    print("-" * 40)
    
    # Load selected tokenizer
    print(f"Loading {selected_tokenizer} tokenizer...")
    try:
        manager.load_tokenizer(selected_tokenizer)
    except Exception as e:
        print(f"Error loading {selected_tokenizer}: {e}")
        print("Falling back to BERT tokenizer...")
        manager.load_tokenizer('bert')
        selected_tokenizer = 'bert'
    
    # Tokenize text
    result = manager.tokenize(input_text, return_tensors=False)
    
    print(f"Text: {result['text']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Token IDs: {result['token_ids']}")
    print()
    
    # Method 2: Using XrayGPT specific configuration
    print("2. Using XrayGPT BLIP2 BERT configuration:")
    print("-" * 40)
    
    manager.load_tokenizer('blip2_bert')
    result = manager.tokenize(input_text, return_tensors=False)
    
    print(f"Text: {result['text']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Token IDs: {result['token_ids']}")
    print()
    
    # Method 3: Using LLaMA/Vicuna tokenizer
    print("3. Using Vicuna tokenizer:")
    print("-" * 40)
    
    try:
        manager.load_tokenizer('vicuna')
        result = manager.tokenize(input_text, return_tensors=False)
        
        print(f"Text: {result['text']}")
        print(f"Vicuna Tokens: {result['tokens']}")
        print(f"Vicuna Token IDs: {result['token_ids']}")
    except Exception as e:
        print(f"Note: Vicuna tokenizer not available locally: {e}")
    print()
    
    # Method 4: Quick tokenization function
    print("4. Using quick_tokenize function:")
    print("-" * 40)
    
    result = quick_tokenize(input_text, 'bert')
    print(f"Quick tokenization result: {len(result['tokens'])} tokens")
    print()
    
    # Method 5: Custom configuration on the fly
    print("5. Using custom configuration:")
    print("-" * 40)
    
    custom_config = {
        'model_name': 'bert-base-uncased',
        'special_tokens': {'bos_token': '[CUSTOM_START]'},
        'max_length': 256,
        'padding': True,
        'truncation': True
    }
    
    manager.load_tokenizer(custom_config=custom_config)
    result = manager.tokenize(input_text, return_tensors=False)
    
    print(f"Custom config tokens: {result['tokens']}")
    print()
    
    # Method 6: Batch tokenization
    print("6. Batch tokenization:")
    print("-" * 40)
    
    batch_texts = [
        "This is the first radiology report.",
        "This is the second medical document.",
        "Another sample text for batch processing."
    ]
    
    manager.load_tokenizer('bert')
    batch_result = manager.tokenize(batch_texts, return_tensors=False)
    
    for i, (text, tokens) in enumerate(zip(batch_result['text'], batch_result['tokens'])):
        print(f"Text {i+1}: {text}")
        print(f"Tokens {i+1}: {len(tokens)} tokens")
    print()
    
    # Method 7: Getting tokenizer information
    print("7. Tokenizer information:")
    print("-" * 40)
    
    info = manager.get_tokenizer_info('bert')
    print(f"BERT config: {info}")
    
    print(f"Vocabulary size: {manager.get_vocab_size()}")
    print(f"Special tokens: {manager.get_special_tokens()}")
    print()


def demonstrate_xraygpt_usage():
    """Demonstrate usage specifically for XrayGPT scenarios."""
    
    print("\n=== XrayGPT Specific Usage ===\n")
    
    manager = TokenizerManager()
    
    # XrayGPT radiology report
    radiology_report = """
    FINDINGS: The chest X-ray shows clear lung fields bilaterally. 
    No evidence of pneumonia, pleural effusion, or pneumothorax. 
    Heart size is within normal limits. Bony structures appear intact.
    
    IMPRESSION: Normal chest X-ray.
    """
    
    # Use BLIP2 BERT tokenizer for vision-language tasks
    print("Using BLIP2 BERT tokenizer for radiology report:")
    manager.load_tokenizer('blip2_bert')
    result = manager.tokenize(radiology_report.strip(), return_tensors=False)
    
    print(f"Report length: {len(radiology_report.strip())} characters")
    print(f"Number of tokens: {len(result['tokens'])}")
    print(f"First 10 tokens: {result['tokens'][:10]}")
    print(f"Last 10 tokens: {result['tokens'][-10:]}")
    
    # Demonstrate decoding
    decoded_text = manager.decode(result['token_ids'])
    print(f"Decoded text matches original: {decoded_text.strip() == radiology_report.strip()}")


if __name__ == "__main__":
    main()
    demonstrate_xraygpt_usage()