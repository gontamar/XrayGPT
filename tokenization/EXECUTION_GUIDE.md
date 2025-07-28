# Execution Guide for Generic Tokenizer Manager

## 📋 Prerequisites

1. **Python Environment**: Python 3.7 or higher
2. **Virtual Environment** (recommended): Create an isolated environment

## 🚀 Step-by-Step Execution

### Step 1: Set Up Environment

```bash
# Create a virtual environment (recommended)
python -m venv tokenizer_env

# Activate virtual environment
# On Windows:
tokenizer_env\Scripts\activate
# On macOS/Linux:
source tokenizer_env/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Or install manually:
pip install transformers torch PyYAML numpy
```

### Step 3: Verify Installation

```bash
# Quick verification
python -c "import transformers, torch, yaml; print('All dependencies installed successfully!')"
```

### Step 4: Run Basic Example

```bash
# Run the main example script
python example_usage.py
```

### Step 5: Test Individual Components

#### A. Test Configuration Loading
```bash
python -c "
from tokenizer_manager import TokenizerManager
manager = TokenizerManager()
print('Available tokenizers:', manager.list_available_tokenizers())
"
```

#### B. Test BERT Tokenizer (like your reference code)
```bash
python -c "
from tokenizer_manager import TokenizerManager
manager = TokenizerManager()
manager.load_tokenizer('bert')
result = manager.tokenize('This is a sample radiology report for testing.')
print('Tokens:', result['tokens'])
print('Token IDs:', result['token_ids'])
"
```

#### C. Test Quick Tokenization
```bash
python -c "
from tokenizer_manager import quick_tokenize
result = quick_tokenize('Sample text', 'bert')
print('Quick result:', len(result['tokens']), 'tokens')
"
```

### Step 6: Run XrayGPT Specific Examples

```bash
python -c "
from tokenizer_manager import TokenizerManager
manager = TokenizerManager()
manager.load_tokenizer('blip2_bert')
text = 'FINDINGS: Clear lung fields. IMPRESSION: Normal chest X-ray.'
result = manager.tokenize(text)
print('XrayGPT tokenization successful:', len(result['tokens']), 'tokens')
"
```

## 🔧 Customization Steps

### Step 7: Modify Configuration

1. **Edit tokenizer_config.yaml**:
```bash
# Open in your preferred editor
nano tokenizer_config.yaml
# or
code tokenizer_config.yaml
```

2. **Add custom tokenizer**:
```yaml
tokenizers:
  my_custom:
    model_name: "your-model-name"
    special_tokens:
      bos_token: "[START]"
    max_length: 1024
```

3. **Test custom configuration**:
```bash
python -c "
from tokenizer_manager import TokenizerManager
manager = TokenizerManager()
manager.load_tokenizer('my_custom')
print('Custom tokenizer loaded successfully!')
"
```

### Step 8: Integration with Your Code

Replace your existing tokenization code:

**Before (your reference code):**
```python
from transformers import AutoTokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
tokens = bert_tokenizer.tokenize(input_text)
token_ids = bert_tokenizer.encode(input_text)
```

**After (using our system):**
```python
from tokenizer_manager import TokenizerManager
manager = TokenizerManager()
manager.load_tokenizer('bert')  # or 'blip2_bert' for XrayGPT
result = manager.tokenize(input_text)
tokens = result['tokens']
token_ids = result['token_ids']
```

## 🧪 Testing Different Scenarios

### Test 1: Single Text Tokenization
```bash
python -c "
from tokenizer_manager import TokenizerManager
manager = TokenizerManager()
manager.load_tokenizer('bert')
result = manager.tokenize('Single text example')
print('Success: Single text tokenized')
"
```

### Test 2: Batch Text Tokenization
```bash
python -c "
from tokenizer_manager import TokenizerManager
manager = TokenizerManager()
manager.load_tokenizer('bert')
texts = ['Text 1', 'Text 2', 'Text 3']
result = manager.tokenize(texts)
print('Success: Batch tokenization completed')
"
```

### Test 3: Different Tokenizers
```bash
python -c "
from tokenizer_manager import TokenizerManager
manager = TokenizerManager()
for tokenizer_name in ['bert', 'roberta', 'gpt2']:
    try:
        manager.load_tokenizer(tokenizer_name)
        result = manager.tokenize('Test text')
        print(f'{tokenizer_name}: {len(result[\"tokens\"])} tokens')
    except Exception as e:
        print(f'{tokenizer_name}: Error - {e}')
"
```

## 🐛 Troubleshooting

### Common Issues and Solutions

1. **ModuleNotFoundError: No module named 'transformers'**
   ```bash
   pip install transformers
   ```

2. **YAML configuration errors**
   ```bash
   python -c "import yaml; yaml.safe_load(open('tokenizer_config.yaml'))"
   ```

3. **Tokenizer download issues**
   ```bash
   # Test internet connection and HuggingFace access
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased')"
   ```

4. **Memory issues with large models**
   - Use smaller models first (bert-base instead of bert-large)
   - Reduce max_length in configuration

### Debug Mode
```bash
# Run with debug logging
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from tokenizer_manager import TokenizerManager
manager = TokenizerManager()
manager.load_tokenizer('bert')
"
```

## 📊 Performance Testing

### Benchmark Different Tokenizers
```bash
python -c "
import time
from tokenizer_manager import TokenizerManager

text = 'This is a sample radiology report for testing performance.' * 10
manager = TokenizerManager()

for tokenizer_name in ['bert', 'roberta']:
    try:
        start_time = time.time()
        manager.load_tokenizer(tokenizer_name)
        result = manager.tokenize(text)
        end_time = time.time()
        print(f'{tokenizer_name}: {end_time - start_time:.4f}s, {len(result[\"tokens\"])} tokens')
    except Exception as e:
        print(f'{tokenizer_name}: Error - {e}')
"
```

## 🔄 Next Steps

1. **Integrate into your project**: Replace existing tokenization code
2. **Customize configurations**: Add your specific tokenizer requirements
3. **Extend functionality**: Add preprocessing or postprocessing steps
4. **Performance optimization**: Add caching for frequently used tokenizers

## 📞 Support

If you encounter issues:
1. Check the error messages and logs
2. Verify all dependencies are installed
3. Test with simple examples first
4. Check tokenizer model availability on HuggingFace

## 🎯 Quick Validation Script

Save this as `validate_setup.py` and run it:
```python
#!/usr/bin/env python3
"""Quick validation script for tokenizer setup"""

def validate_setup():
    try:
        # Test imports
        from tokenizer_manager import TokenizerManager, quick_tokenize
        print("✅ Imports successful")
        
        # Test configuration loading
        manager = TokenizerManager()
        print("✅ Configuration loaded")
        
        # Test tokenizer loading
        manager.load_tokenizer('bert')
        print("✅ BERT tokenizer loaded")
        
        # Test tokenization
        result = manager.tokenize("Test text")
        print(f"✅ Tokenization successful: {len(result['tokens'])} tokens")
        
        # Test quick function
        quick_result = quick_tokenize("Quick test", 'bert')
        print("✅ Quick tokenization successful")
        
        print("\n🎉 All tests passed! Your setup is ready to use.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check the installation steps.")

if __name__ == "__main__":
    validate_setup()
```

Run validation:
```bash
python validate_setup.py
```