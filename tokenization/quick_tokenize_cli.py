#!/usr/bin/env python3
"""
Quick Tokenize CLI
A simple command-line interface for quick tokenization tasks.
"""

import sys
import argparse
from tokenizer_manager import TokenizerManager, quick_tokenize


def list_tokenizers():
    """List all available tokenizers."""
    print("📋 Available Tokenizers:")
    print("-" * 50)
    
    manager = TokenizerManager()
    available_tokenizers = manager.list_available_tokenizers()
    
    # Group tokenizers by category
    categories = {
        'General Purpose': ['bert', 'bert_large', 'roberta', 'gpt2'],
        'XrayGPT/Medical': ['blip2_bert', 'vicuna_llm'],
        'Large Language Models': ['llama', 'vicuna']
    }
    
    for category, tokenizer_list in categories.items():
        print(f"\n🏷️  {category}:")
        for tokenizer in tokenizer_list:
            if tokenizer in available_tokenizers:
                try:
                    info = manager.get_tokenizer_info(tokenizer)
                    model_name = info.get('model_name', 'N/A')
                    max_length = info.get('max_length', 'N/A')
                    print(f"   • {tokenizer:15s} - {model_name} (max_length: {max_length})")
                except:
                    print(f"   • {tokenizer:15s} - Available")
    
    # Show any other tokenizers not in categories
    other_tokenizers = [t for t in available_tokenizers if not any(t in cat_list for cat_list in categories.values())]
    if other_tokenizers:
        print(f"\n🔧 Other/Custom:")
        for tokenizer in other_tokenizers:
            print(f"   • {tokenizer}")


def interactive_select_tokenizer():
    """Interactively select a tokenizer."""
    manager = TokenizerManager()
    available_tokenizers = manager.list_available_tokenizers()
    
    print("📋 Available Tokenizers:")
    for i, tokenizer_name in enumerate(available_tokenizers, 1):
        print(f"{i:2d}. {tokenizer_name}")
    
    while True:
        try:
            choice = input(f"\nSelect tokenizer (1-{len(available_tokenizers)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_tokenizers):
                return available_tokenizers[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(available_tokenizers)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def tokenize_text(text: str, tokenizer_name: str, show_details: bool = True):
    """Tokenize text and display results."""
    try:
        print(f"\n🔄 Tokenizing with {tokenizer_name}...")
        result = quick_tokenize(text, tokenizer_name)
        
        print(f"\n📄 Text: {text}")
        print(f"🔢 Tokens ({len(result['tokens'])}): {result['tokens']}")
        print(f"🔢 Token IDs: {result['token_ids']}")
        
        if show_details:
            manager = TokenizerManager()
            manager.load_tokenizer(tokenizer_name)
            print(f"📊 Vocabulary size: {manager.get_vocab_size():,}")
            
            special_tokens = manager.get_special_tokens()
            if special_tokens:
                print(f"🏷️  Special tokens: {list(special_tokens.keys())}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Quick Tokenize CLI - Tokenize text using various tokenizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                           # List available tokenizers
  %(prog)s --text "Hello world"             # Interactive tokenizer selection
  %(prog)s --text "Hello world" --tokenizer bert  # Use specific tokenizer
  %(prog)s --interactive                    # Interactive mode
        """
    )
    
    parser.add_argument('--text', '-t', type=str, help='Text to tokenize')
    parser.add_argument('--tokenizer', '-k', type=str, help='Tokenizer to use')
    parser.add_argument('--list', '-l', action='store_true', help='List available tokenizers')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--no-details', action='store_true', help='Hide detailed information')
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        list_tokenizers()
        return
    
    # Handle interactive mode
    if args.interactive:
        from interactive_tokenizer import InteractiveTokenizer
        interactive_tokenizer = InteractiveTokenizer()
        interactive_tokenizer.run()
        return
    
    # Handle text tokenization
    if args.text:
        # Select tokenizer
        if args.tokenizer:
            tokenizer_name = args.tokenizer
            # Validate tokenizer exists
            manager = TokenizerManager()
            available_tokenizers = manager.list_available_tokenizers()
            if tokenizer_name not in available_tokenizers:
                print(f"❌ Tokenizer '{tokenizer_name}' not found.")
                print(f"Available tokenizers: {', '.join(available_tokenizers)}")
                sys.exit(1)
        else:
            print("🎯 Select a tokenizer:")
            tokenizer_name = interactive_select_tokenizer()
        
        # Tokenize
        result = tokenize_text(args.text, tokenizer_name, not args.no_details)
        
        if result:
            print("\n✅ Tokenization completed successfully!")
    
    else:
        # No arguments provided, show help
        parser.print_help()
        print("\n💡 Quick start:")
        print("  python3 quick_tokenize_cli.py --interactive")
        print("  python3 quick_tokenize_cli.py --list")


if __name__ == "__main__":
    main()