#!/usr/bin/env python3
"""
Quick validation script for tokenizer setup
Run this script to verify everything is working correctly.
"""

def validate_setup():
    """Validate the tokenizer setup step by step."""
    print("🔍 Validating Generic Tokenizer Manager Setup...\n")
    
    try:
        # Test 1: Import validation
        print("1. Testing imports...")
        from tokenizer_manager import TokenizerManager, quick_tokenize
        print("   ✅ Imports successful")
        
        # Test 2: Configuration loading
        print("\n2. Testing configuration loading...")
        manager = TokenizerManager()
        available_tokenizers = manager.list_available_tokenizers()
        print(f"   ✅ Configuration loaded")
        print(f"   📋 Available tokenizers: {available_tokenizers}")
        
        # Test 3: BERT tokenizer loading (like reference code)
        print("\n3. Testing BERT tokenizer loading...")
        manager.load_tokenizer('bert')
        print("   ✅ BERT tokenizer loaded successfully")
        
        # Test 4: Basic tokenization
        print("\n4. Testing basic tokenization...")
        test_text = "This is a sample radiology report for testing."
        result = manager.tokenize(test_text, return_tensors=False)
        print(f"   ✅ Tokenization successful")
        print(f"   📝 Text: {result['text']}")
        print(f"   🔤 Tokens: {result['tokens'][:10]}..." if len(result['tokens']) > 10 else f"   🔤 Tokens: {result['tokens']}")
        print(f"   🔢 Token IDs: {result['token_ids'][:10]}..." if len(result['token_ids']) > 10 else f"   🔢 Token IDs: {result['token_ids']}")
        print(f"   📊 Total tokens: {len(result['tokens'])}")
        
        # Test 5: XrayGPT specific tokenizer
        print("\n5. Testing XrayGPT BLIP2 BERT tokenizer...")
        manager.load_tokenizer('blip2_bert')
        xray_result = manager.tokenize(test_text, return_tensors=False)
        print("   ✅ XrayGPT tokenizer loaded and tested")
        print(f"   📊 XrayGPT tokens: {len(xray_result['tokens'])}")
        
        # Test 6: Quick tokenization function
        print("\n6. Testing quick tokenization function...")
        quick_result = quick_tokenize("Quick test", 'bert')
        print("   ✅ Quick tokenization successful")
        print(f"   📊 Quick result tokens: {len(quick_result['tokens'])}")
        
        # Test 7: Batch tokenization
        print("\n7. Testing batch tokenization...")
        batch_texts = [
            "First radiology report",
            "Second medical document",
            "Third sample text"
        ]
        batch_result = manager.tokenize(batch_texts, return_tensors=False)
        print("   ✅ Batch tokenization successful")
        print(f"   📊 Batch results: {[len(tokens) for tokens in batch_result['tokens']]} tokens per text")
        
        # Test 8: Tokenizer information
        print("\n8. Testing tokenizer information retrieval...")
        bert_info = manager.get_tokenizer_info('bert')
        vocab_size = manager.get_vocab_size()
        special_tokens = manager.get_special_tokens()
        print("   ✅ Tokenizer information retrieved")
        print(f"   📚 Vocabulary size: {vocab_size}")
        print(f"   🏷️  Special tokens: {list(special_tokens.keys())}")
        
        # Test 9: Decoding
        print("\n9. Testing token decoding...")
        decoded_text = manager.decode(result['token_ids'])
        print("   ✅ Decoding successful")
        print(f"   🔄 Decoded matches original: {decoded_text.strip() == test_text}")
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! Your setup is ready to use.")
        print("="*50)
        
        # Usage summary
        print("\n📖 Quick Usage Summary:")
        print("```python")
        print("from tokenizer_manager import TokenizerManager")
        print("manager = TokenizerManager()")
        print("manager.load_tokenizer('bert')  # or 'blip2_bert' for XrayGPT")
        print("result = manager.tokenize('Your text here')")
        print("print('Tokens:', result['tokens'])")
        print("print('Token IDs:', result['token_ids'])")
        print("```")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Solution: Run 'pip install -r requirements.txt'")
        return False
        
    except FileNotFoundError as e:
        print(f"❌ File Not Found: {e}")
        print("💡 Solution: Make sure 'tokenizer_config.yaml' exists in the current directory")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print("💡 Check the error message above and ensure all dependencies are installed")
        return False


def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...\n")
    
    dependencies = [
        ('transformers', 'transformers'),
        ('torch', 'torch'),
        ('yaml', 'PyYAML'),
        ('numpy', 'numpy')
    ]
    
    missing_deps = []
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"   ✅ {package_name}")
        except ImportError:
            print(f"   ❌ {package_name} - MISSING")
            missing_deps.append(package_name)
    
    if missing_deps:
        print(f"\n💡 Install missing dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    else:
        print("\n✅ All dependencies are available!")
        return True


def main():
    """Main validation function."""
    print("🚀 Generic Tokenizer Manager - Setup Validation")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before proceeding.")
        return
    
    print("\n" + "=" * 50)
    
    # Run full validation
    success = validate_setup()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Run 'python example_usage.py' for detailed examples")
        print("2. Modify 'tokenizer_config.yaml' for custom configurations")
        print("3. Integrate into your existing code")
        print("4. Check 'EXECUTION_GUIDE.md' for more details")
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Check that all files are in the current directory")
        print("2. Verify internet connection for model downloads")
        print("3. Try running individual components")
        print("4. Check 'EXECUTION_GUIDE.md' for detailed steps")


if __name__ == "__main__":
    main()