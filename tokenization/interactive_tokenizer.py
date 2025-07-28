#!/usr/bin/env python3
"""
Interactive Tokenizer Script
Allows users to select from available tokenizers and tokenize their text interactively.
"""

import sys
from tokenizer_manager import TokenizerManager
from typing import List, Dict, Any


class InteractiveTokenizer:
    """Interactive tokenizer interface for user-friendly tokenization."""
    
    def __init__(self):
        """Initialize the interactive tokenizer."""
        self.manager = TokenizerManager()
        self.current_tokenizer = None
        
    def display_welcome(self):
        """Display welcome message and instructions."""
        print("=" * 70)
        print("🚀 INTERACTIVE TOKENIZER MANAGER")
        print("=" * 70)
        print("Welcome! This tool helps you tokenize text using different tokenizers.")
        print("You can select from various pre-configured tokenizers based on your needs.")
        print()
    
    def display_available_tokenizers(self) -> List[str]:
        """Display available tokenizers with descriptions."""
        print("📋 AVAILABLE TOKENIZERS:")
        print("-" * 50)
        
        available_tokenizers = self.manager.list_available_tokenizers()
        
        # Create detailed descriptions for each tokenizer
        tokenizer_descriptions = {
            'bert': 'BERT Base - General purpose, good for most NLP tasks',
            'bert_large': 'BERT Large - More parameters, better performance',
            'roberta': 'RoBERTa - Improved BERT variant, robust performance',
            'gpt2': 'GPT-2 - Generative model, good for text generation tasks',
            'blip2_bert': 'XrayGPT BLIP2 BERT - Optimized for medical/radiology reports',
            'vicuna_llm': 'XrayGPT Vicuna - Large language model for medical text',
            'llama': 'LLaMA 2 - Meta\'s large language model',
            'vicuna': 'Vicuna - Fine-tuned LLaMA for conversations'
        }
        
        for i, tokenizer_name in enumerate(available_tokenizers, 1):
            description = tokenizer_descriptions.get(tokenizer_name, 'Custom tokenizer configuration')
            print(f"{i:2d}. {tokenizer_name:15s} - {description}")
        
        print()
        return available_tokenizers
    
    def get_user_selection(self, available_tokenizers: List[str]) -> str:
        """Get user's tokenizer selection."""
        while True:
            try:
                print("🎯 SELECT A TOKENIZER:")
                choice = input(f"Enter your choice (1-{len(available_tokenizers)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print("👋 Goodbye!")
                    sys.exit(0)
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_tokenizers):
                    selected_tokenizer = available_tokenizers[choice_num - 1]
                    return selected_tokenizer
                else:
                    print(f"❌ Please enter a number between 1 and {len(available_tokenizers)}")
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
    
    def load_selected_tokenizer(self, tokenizer_name: str) -> bool:
        """Load the selected tokenizer."""
        print(f"\n🔄 Loading {tokenizer_name} tokenizer...")
        try:
            self.manager.load_tokenizer(tokenizer_name)
            self.current_tokenizer = tokenizer_name
            
            # Display tokenizer info
            info = self.manager.get_tokenizer_info(tokenizer_name)
            print(f"✅ Successfully loaded {tokenizer_name}")
            print(f"   Model: {info.get('model_name', 'N/A')}")
            print(f"   Max length: {info.get('max_length', 'N/A')}")
            
            special_tokens = self.manager.get_special_tokens()
            if special_tokens:
                print(f"   Special tokens: {list(special_tokens.keys())}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading {tokenizer_name}: {e}")
            print("💡 This might be due to network issues or model availability.")
            return False
    
    def get_text_input(self) -> str:
        """Get text input from user."""
        print("\n📝 TEXT INPUT:")
        print("Enter your text to tokenize (or 'back' to select different tokenizer, 'q' to quit):")
        print("-" * 50)
        
        try:
            text = input("Your text: ").strip()
            
            if text.lower() == 'q':
                print("👋 Goodbye!")
                sys.exit(0)
            elif text.lower() == 'back':
                return 'BACK'
            elif not text:
                print("❌ Please enter some text to tokenize")
                return self.get_text_input()
            
            return text
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
    
    def tokenize_and_display(self, text: str):
        """Tokenize text and display results."""
        print(f"\n🔤 TOKENIZATION RESULTS:")
        print("-" * 50)
        
        try:
            # Tokenize the text
            result = self.manager.tokenize(text, return_tensors=False)
            
            # Display results
            print(f"📄 Original text: {text}")
            print(f"🔢 Number of tokens: {len(result['tokens'])}")
            print(f"📊 Vocabulary size: {self.manager.get_vocab_size():,}")
            print()
            
            # Show tokens (limit display for very long texts)
            tokens = result['tokens']
            if len(tokens) <= 20:
                print(f"🔤 Tokens: {tokens}")
            else:
                print(f"🔤 First 10 tokens: {tokens[:10]}")
                print(f"🔤 Last 10 tokens: {tokens[-10:]}")
            print()
            
            # Show token IDs (limit display for very long texts)
            token_ids = result['token_ids']
            if len(token_ids) <= 20:
                print(f"🔢 Token IDs: {token_ids}")
            else:
                print(f"🔢 First 10 token IDs: {token_ids[:10]}")
                print(f"🔢 Last 10 token IDs: {token_ids[-10:]}")
            print()
            
            # Test decoding
            decoded_text = self.manager.decode(token_ids)
            matches_original = decoded_text.strip() == text.strip()
            print(f"🔄 Decoding successful: {'✅ Yes' if matches_original else '⚠️  Partial (some tokens may be modified)'}")
            
            if not matches_original and len(text) < 200:
                print(f"🔄 Decoded text: {decoded_text}")
            
        except Exception as e:
            print(f"❌ Error during tokenization: {e}")
    
    def show_usage_examples(self):
        """Show usage examples for different scenarios."""
        print("\n💡 USAGE EXAMPLES:")
        print("-" * 50)
        print("📚 General NLP tasks: Use 'bert' or 'roberta'")
        print("🏥 Medical/Radiology reports: Use 'blip2_bert' (XrayGPT)")
        print("💬 Conversational AI: Use 'vicuna' or 'vicuna_llm'")
        print("📝 Text generation: Use 'gpt2' or 'llama'")
        print("🔬 Research/Large models: Use 'bert_large' or 'llama'")
        print()
    
    def run_interactive_session(self):
        """Run the main interactive session."""
        while True:
            try:
                # Show available tokenizers
                available_tokenizers = self.display_available_tokenizers()
                
                # Show usage examples
                self.show_usage_examples()
                
                # Get user selection
                selected_tokenizer = self.get_user_selection(available_tokenizers)
                
                # Load tokenizer
                if not self.load_selected_tokenizer(selected_tokenizer):
                    continue
                
                # Tokenization loop for current tokenizer
                while True:
                    text = self.get_text_input()
                    
                    if text == 'BACK':
                        break
                    
                    self.tokenize_and_display(text)
                    
                    # Ask if user wants to tokenize more text
                    print("\n🔄 OPTIONS:")
                    print("1. Tokenize another text with same tokenizer")
                    print("2. Select different tokenizer")
                    print("3. Quit")
                    
                    while True:
                        choice = input("Your choice (1-3): ").strip()
                        if choice == '1':
                            break
                        elif choice == '2':
                            break
                        elif choice == '3':
                            print("👋 Goodbye!")
                            return
                        else:
                            print("❌ Please enter 1, 2, or 3")
                    
                    if choice == '2':
                        break
                    elif choice == '3':
                        return
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                print("🔄 Restarting...")
    
    def run(self):
        """Main entry point for the interactive tokenizer."""
        self.display_welcome()
        self.run_interactive_session()


def main():
    """Main function to run the interactive tokenizer."""
    try:
        interactive_tokenizer = InteractiveTokenizer()
        interactive_tokenizer.run()
    except Exception as e:
        print(f"❌ Failed to start interactive tokenizer: {e}")
        print("💡 Make sure all dependencies are installed and configuration file exists.")
        sys.exit(1)


if __name__ == "__main__":
    main()