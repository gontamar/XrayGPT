#!/usr/bin/env python3
"""
Run Examples Script
Execute this to see the tokenizer manager in action with various scenarios.
"""

import time
from tokenizer_manager import TokenizerManager, quick_tokenize


def example_1_basic_usage():
    """Example 1: Basic usage similar to the reference code."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Usage (Reference Code Style)")
    print("=" * 60)
    
    # Your original reference code equivalent
    input_text = "This is a sample radiology report for testing."
    
    manager = TokenizerManager()
    manager.load_tokenizer('bert')
    
    result = manager.tokenize(input_text, return_tensors=False)
    
    print(f"Text: {result['text']}")
    print(f"Tokens: {result['tokens']}")
    print(f"Token IDs: {result['token_ids']}")
    print(f"Number of tokens: {len(result['tokens'])}")
    
    # Show special tokens
    special_tokens = manager.get_special_tokens()
    print(f"Special tokens: {special_tokens}")


def example_2_xraygpt_usage():
    """Example 2: XrayGPT specific usage."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: XrayGPT BLIP2 BERT Usage")
    print("=" * 60)
    
    radiology_report = """
    CLINICAL HISTORY: 45-year-old patient with chest pain.
    
    FINDINGS: The chest radiograph demonstrates clear lung fields bilaterally. 
    No evidence of pneumonia, pleural effusion, or pneumothorax. 
    The cardiac silhouette is within normal limits. 
    Bony structures appear intact without acute fractures.
    
    IMPRESSION: Normal chest radiograph.
    """
    
    manager = TokenizerManager()
    manager.load_tokenizer('blip2_bert')
    
    result = manager.tokenize(radiology_report.strip(), return_tensors=False)
    
    print(f"Report length: {len(radiology_report.strip())} characters")
    print(f"Number of tokens: {len(result['tokens'])}")
    print(f"First 15 tokens: {result['tokens'][:15]}")
    print(f"Last 10 tokens: {result['tokens'][-10:]}")
    
    # Test decoding
    decoded = manager.decode(result['token_ids'])
    print(f"Decoding successful: {len(decoded) > 0}")


def example_3_batch_processing():
    """Example 3: Batch processing multiple texts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)
    
    medical_texts = [
        "Normal chest X-ray with clear lung fields.",
        "CT scan shows no abnormalities in the abdomen.",
        "MRI reveals mild degenerative changes in the spine.",
        "Ultrasound examination of the thyroid is unremarkable.",
        "Blood test results are within normal limits."
    ]
    
    manager = TokenizerManager()
    manager.load_tokenizer('bert')
    
    start_time = time.time()
    result = manager.tokenize(medical_texts, return_tensors=False)
    end_time = time.time()
    
    print(f"Processed {len(medical_texts)} texts in {end_time - start_time:.4f} seconds")
    
    for i, (text, tokens) in enumerate(zip(result['text'], result['tokens'])):
        print(f"\nText {i+1}: {text}")
        print(f"Tokens ({len(tokens)}): {tokens[:8]}...")


def example_4_different_tokenizers():
    """Example 4: Comparing different tokenizers."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Different Tokenizers Comparison")
    print("=" * 60)
    
    test_text = "The patient presents with acute chest pain and shortness of breath."
    
    tokenizers_to_test = ['bert', 'roberta', 'gpt2']
    
    manager = TokenizerManager()
    
    for tokenizer_name in tokenizers_to_test:
        try:
            print(f"\n--- {tokenizer_name.upper()} Tokenizer ---")
            manager.load_tokenizer(tokenizer_name)
            result = manager.tokenize(test_text, return_tensors=False)
            
            print(f"Tokens ({len(result['tokens'])}): {result['tokens']}")
            print(f"Vocab size: {manager.get_vocab_size()}")
            
        except Exception as e:
            print(f"Error with {tokenizer_name}: {e}")


def example_5_custom_configuration():
    """Example 5: Using custom configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Custom Configuration")
    print("=" * 60)
    
    # Custom configuration for specific use case
    custom_config = {
        'model_name': 'bert-base-uncased',
        'special_tokens': {
            'bos_token': '[REPORT_START]',
            'eos_token': '[REPORT_END]'
        },
        'max_length': 256,
        'padding': True,
        'truncation': True
    }
    
    manager = TokenizerManager()
    manager.load_tokenizer(custom_config=custom_config)
    
    medical_text = "Patient shows signs of improvement after treatment."
    result = manager.tokenize(medical_text, return_tensors=False)
    
    print(f"Custom tokenizer loaded with special tokens: {manager.get_special_tokens()}")
    print(f"Text: {result['text']}")
    print(f"Tokens: {result['tokens']}")


def example_6_quick_tokenization():
    """Example 6: Quick tokenization function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Quick Tokenization Function")
    print("=" * 60)
    
    texts = [
        "Quick tokenization test 1",
        "Quick tokenization test 2",
        "Quick tokenization test 3"
    ]
    
    for i, text in enumerate(texts, 1):
        result = quick_tokenize(text, 'bert')
        print(f"Quick test {i}: {len(result['tokens'])} tokens")
        print(f"  Tokens: {result['tokens'][:5]}...")


def example_7_performance_test():
    """Example 7: Performance testing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Performance Testing")
    print("=" * 60)
    
    # Generate longer text for performance testing
    long_text = "This is a comprehensive radiology report. " * 50
    
    manager = TokenizerManager()
    
    tokenizers = ['bert', 'roberta']
    
    for tokenizer_name in tokenizers:
        try:
            print(f"\n--- Performance Test: {tokenizer_name.upper()} ---")
            
            # Load tokenizer
            start_time = time.time()
            manager.load_tokenizer(tokenizer_name)
            load_time = time.time() - start_time
            
            # Tokenize
            start_time = time.time()
            result = manager.tokenize(long_text, return_tensors=False)
            tokenize_time = time.time() - start_time
            
            print(f"Load time: {load_time:.4f}s")
            print(f"Tokenization time: {tokenize_time:.4f}s")
            print(f"Tokens generated: {len(result['tokens'])}")
            print(f"Tokens per second: {len(result['tokens'])/tokenize_time:.2f}")
            
        except Exception as e:
            print(f"Error testing {tokenizer_name}: {e}")


def main():
    """Run all examples."""
    print("🚀 Generic Tokenizer Manager - Examples")
    print("Running comprehensive examples...")
    
    try:
        example_1_basic_usage()
        example_2_xraygpt_usage()
        example_3_batch_processing()
        example_4_different_tokenizers()
        example_5_custom_configuration()
        example_6_quick_tokenization()
        example_7_performance_test()
        
        print("\n" + "=" * 60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n📚 What you can do next:")
        print("1. Modify tokenizer_config.yaml for your specific needs")
        print("2. Integrate the TokenizerManager into your project")
        print("3. Add custom preprocessing/postprocessing")
        print("4. Experiment with different tokenizer configurations")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("💡 Make sure all dependencies are installed and configuration file exists")


if __name__ == "__main__":
    main()