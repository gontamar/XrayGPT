"""
Utility functions for MMAI tokenization
"""

import torch
from typing import Dict, List, Optional, Union, Any
from .tokenization_framework import get_framework, setup_xraygpt_tokenization

def restore_tokenization_for_model(model, llama_model_path: str = None, bert_model_path: str = None):
    """
    Restore enhanced tokenization capabilities for an existing model
    
    Args:
        model: The model instance to enhance
        llama_model_path: Path to LLaMA model for tokenization
        bert_model_path: Path to BERT model for Q-Former tokenization
    """
    try:
        # Setup framework if not already done
        framework = setup_xraygpt_tokenization(llama_model_path, bert_model_path)
        
        # Add framework reference to model
        model.tokenization_framework = framework
        model.enhanced_tokenization = True
        
        # Add enhanced methods to model
        def get_enhanced_tokenizer(tokenizer_name: str = 'xraygpt_llama'):
            return framework.get_tokenizer(tokenizer_name)
        
        def get_multimodal_tokenizer(tokenizer_name: str = 'xraygpt_multimodal'):
            return framework.get_multimodal_tokenizer(tokenizer_name)
        
        def encode_with_framework(text: str, modalities: dict = None, tokenizer_name: str = 'xraygpt_multimodal'):
            if modalities:
                multimodal_tokenizer = framework.get_multimodal_tokenizer(tokenizer_name)
                return multimodal_tokenizer.encode_multimodal(text, modalities)
            else:
                enhanced_tokenizer = framework.get_tokenizer('xraygpt_llama')
                return enhanced_tokenizer.encode(text)
        
        # Bind methods to model
        model.get_enhanced_tokenizer = get_enhanced_tokenizer
        model.get_multimodal_tokenizer = get_multimodal_tokenizer
        model.encode_with_framework = encode_with_framework
        
        print("Enhanced tokenization capabilities restored for model")
        return True
        
    except Exception as e:
        print(f"Failed to restore enhanced tokenization: {e}")
        model.enhanced_tokenization = False
        return False

def create_medical_prompt_with_tokenization(
    base_prompt: str, 
    findings: List[str] = None, 
    modalities: Dict[str, Any] = None,
    framework_name: str = 'xraygpt_multimodal'
) -> Dict[str, Any]:
    """
    Create a medical prompt with enhanced tokenization
    
    Args:
        base_prompt: Base prompt text
        findings: List of medical findings to include
        modalities: Dictionary of modalities (e.g., {'image': True, 'radiology': True})
        framework_name: Name of the tokenization framework to use
    
    Returns:
        Dictionary containing tokenized prompt and metadata
    """
    try:
        framework = get_framework()
        multimodal_tokenizer = framework.get_multimodal_tokenizer(framework_name)
        
        # Enhance prompt with findings
        enhanced_prompt = base_prompt
        if findings:
            findings_text = " ".join([f"[FINDING] {finding}" for finding in findings])
            enhanced_prompt = f"{findings_text} {enhanced_prompt}"
        
        # Tokenize with modalities
        if modalities is None:
            modalities = {'image': True, 'radiology': True}
        
        result = multimodal_tokenizer.encode_multimodal(enhanced_prompt, modalities)
        
        return {
            'tokenized': result,
            'original_prompt': base_prompt,
            'enhanced_prompt': enhanced_prompt,
            'findings': findings,
            'modalities': modalities
        }
        
    except Exception as e:
        print(f"Enhanced prompt creation failed: {e}")
        # Fallback to basic tokenization
        return {
            'enhanced_prompt': base_prompt,
            'original_prompt': base_prompt,
            'findings': findings,
            'modalities': modalities,
            'error': str(e)
        }

def compare_tokenization_methods(text: str, model_paths: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Compare different tokenization methods for analysis
    
    Args:
        text: Text to tokenize
        model_paths: Dictionary of model paths for different tokenizers
    
    Returns:
        Comparison results
    """
    if model_paths is None:
        model_paths = {
            'llama': 'decapoda-research/llama-7b-hf',
            'bert': 'bert-base-uncased'
        }
    
    try:
        framework = get_framework()
        
        # Setup tokenizers if not already done
        if 'xraygpt_llama' not in framework.tokenizers:
            framework.setup_xraygpt_tokenizers(
                model_paths.get('llama'), 
                model_paths.get('bert')
            )
        
        results = {}
        
        # Test LLaMA tokenizer
        try:
            llama_tokenizer = framework.get_tokenizer('xraygpt_llama')
            llama_result = llama_tokenizer.encode(text)
            results['llama'] = {
                'input_ids': llama_result['input_ids'].tolist(),
                'attention_mask': llama_result['attention_mask'].tolist(),
                'num_tokens': llama_result['input_ids'].shape[-1],
                'decoded': llama_tokenizer.decode(llama_result['input_ids'][0])
            }
        except Exception as e:
            results['llama'] = {'error': str(e)}
        
        # Test BERT tokenizer
        try:
            bert_tokenizer = framework.get_tokenizer('xraygpt_bert')
            bert_result = bert_tokenizer.encode(text)
            results['bert'] = {
                'input_ids': bert_result['input_ids'].tolist(),
                'attention_mask': bert_result['attention_mask'].tolist(),
                'num_tokens': bert_result['input_ids'].shape[-1],
                'decoded': bert_tokenizer.decode(bert_result['input_ids'][0])
            }
        except Exception as e:
            results['bert'] = {'error': str(e)}
        
        # Test multi-modal tokenizer
        try:
            multimodal_tokenizer = framework.get_multimodal_tokenizer('xraygpt_multimodal')
            multimodal_result = multimodal_tokenizer.encode_multimodal(
                text, 
                {'image': True, 'radiology': True}
            )
            results['multimodal'] = {
                'input_ids': multimodal_result['input_ids'].tolist(),
                'attention_mask': multimodal_result['attention_mask'].tolist(),
                'num_tokens': multimodal_result['input_ids'].shape[-1],
                'processed_text': multimodal_result['processed_text'],
                'modalities': multimodal_result['modalities']
            }
        except Exception as e:
            results['multimodal'] = {'error': str(e)}
        
        return results
        
    except Exception as e:
        return {'error': f"Comparison failed: {e}"}

def validate_tokenization_restoration() -> Dict[str, bool]:
    """
    Validate that tokenization has been properly restored
    
    Returns:
        Dictionary of validation results
    """
    results = {
        'framework_available': False,
        'llama_tokenizer': False,
        'bert_tokenizer': False,
        'multimodal_tokenizer': False,
        'encoding_works': False,
        'decoding_works': False
    }
    
    try:
        # Check framework availability
        framework = get_framework()
        results['framework_available'] = True
        
        # Setup basic tokenizers
        framework.setup_xraygpt_tokenizers()
        
        # Check individual tokenizers
        try:
            llama_tokenizer = framework.get_tokenizer('xraygpt_llama')
            results['llama_tokenizer'] = True
        except:
            pass
        
        try:
            bert_tokenizer = framework.get_tokenizer('xraygpt_bert')
            results['bert_tokenizer'] = True
        except:
            pass
        
        try:
            multimodal_tokenizer = framework.get_multimodal_tokenizer('xraygpt_multimodal')
            results['multimodal_tokenizer'] = True
        except:
            pass
        
        # Test encoding/decoding
        if results['llama_tokenizer']:
            try:
                test_text = "This is a test for medical imaging analysis."
                encoded = llama_tokenizer.encode(test_text)
                decoded = llama_tokenizer.decode(encoded['input_ids'][0])
                results['encoding_works'] = True
                results['decoding_works'] = True
            except:
                pass
        
    except Exception as e:
        results['error'] = str(e)
    
    return results