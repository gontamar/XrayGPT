"""
Text Processing Pipeline
Input → Tokenize → Embed
"""

import logging
import os
from text_tokenizer_class import Tokenizer
from text_embedding_class import EmbeddingManager
from image_embeddings import ImageProcessor

# Reduce logging verbosity
logging.getLogger('text_tokenizer_class').setLevel(logging.WARNING)
logging.getLogger('text_embedding_class').setLevel(logging.WARNING)
logging.getLogger('image_embeddings').setLevel(logging.WARNING)



def main():
    """Text and Image Processing Pipeline."""
    
    print("Text and Image Processing Pipeline with Patch Embeddings")
    print("Options: [1] Text [2] Image [3] Image+Patches [4] Image+ViT [5] Complete Pipeline")
    
    # Initialize components
    tokenizer = Tokenizer(enable_input_processing=False)
    embedding_manager = EmbeddingManager()
    image_processor = ImageProcessor()
    
    # Load models from config
    tokenizer.load_tokenizer('bert')
    embedding_manager.load_embedding_model('bert', device='cpu')
    image_processor.load_processor('blip2')
    
    print("System ready")
    
    while True:
        print("\n" + "="*50)
        choice = input("Choose option [1=Text, 2=Image, 3=Patches, 4=ViT, 5=Complete, quit]: ").strip()
        
        if choice.lower() == 'quit':
            break
        
        if choice == '1':
            # Text processing
            text_input = input("Enter text: ").strip()
            if not text_input:
                continue
            
            try:
                # Tokenize and embed text
                tokenize_result = tokenizer.tokenize(text_input, return_tensors=False)
                embed_result = embedding_manager.create_embeddings(text_input, device='cpu')
                
                print(f"\n TEXT RESULTS:")
                print(f"Tokens: {tokenize_result['tokens']}")
                print(f"Token IDs: {tokenize_result['token_ids']}")
                print(f"Text Embeddings Shape: {embed_result['embeddings'].shape}")
                print(f"Sample Values: {embed_result['embeddings'][0, 0, :3].tolist()}")
                
            except Exception as e:
                print(f"Text processing error: {e}")
        
        elif choice == '2':
            # Image processing
            image_path = input("Enter image path: ").strip()
            if not image_path or not os.path.exists(image_path):
                print("Invalid image path!")
                continue
            
            try:
                # Process image
                image_result = image_processor.process_image(image_path, device='cpu')
                
                print(f"\n IMAGE RESULTS:")
                print(f"Image Path: {image_result['image_path']}")
                print(f"Original Size: {image_result['original_size']}")
                print(f"Processed Shape: {image_result['processed_size']}")
                print(f"Sample Tensor Values: {image_result['image_tensor'][0, 0, :3, :3].tolist()}")
                
            except Exception as e:
                print(f"Image processing error: {e}")
        
        elif choice == '3':
            # Image processing with patches
            image_path = input("Enter image path: ").strip()
            if not image_path or not os.path.exists(image_path):
                print("Invalid image path!")
                continue
            
            try:
                # Process image with patch embeddings
                image_result = image_processor.process_image(image_path, device='cpu', create_patches=True)
                
                print(f"\n IMAGE + PATCH RESULTS:")
                print(f"Image Path: {image_result['image_path']}")
                print(f"Original Size: {image_result['original_size']}")
                print(f"Processed Shape: {image_result['processed_size']}")
                print(f"Sample Tensor Values: {image_result['image_tensor'][0, 0, :3, :3].tolist()}")
                
                if 'patch_embeddings' in image_result:
                    print(f"\n PATCH EMBEDDING RESULTS:")
                    print(f"Patch Embeddings Shape: {image_result['patch_embeddings'].shape}")
                    print(f"Number of Patches: {image_result['num_patches']}")
                    print(f"Patch Size: {image_result['patch_size']}")
                    print(f"Embedding Dimension: {image_result['embed_dim']}")
                    print(f"Patch Grid Shape: {image_result['patch_shape']}")
                    print(f"Sample Patch Embedding: {image_result['patch_embeddings'][0, 0, :5].tolist()}")
                
            except Exception as e:
                print(f"Image processing error: {e}")
        
        elif choice == '4':
            # Image processing with Vision Transformer
            image_path = input("Enter image path: ").strip()
            if not image_path or not os.path.exists(image_path):
                print("Invalid image path!")
                continue
            
            try:
                # Process image with Vision Transformer
                image_result = image_processor.process_image(
                    image_path, 
                    device='cpu', 
                    create_patches=True, 
                    use_vision_transformer=True
                )
                
                print(f"\n IMAGE + VISION TRANSFORMER RESULTS:")
                print(f"Image Path: {image_result['image_path']}")
                print(f"Original Size: {image_result['original_size']}")
                print(f"Processed Shape: {image_result['processed_size']}")
                
                if 'patch_embeddings' in image_result:
                    print(f"\n PATCH EMBEDDING RESULTS:")
                    print(f"Patch Embeddings Shape: {image_result['patch_embeddings'].shape}")
                    print(f"Number of Patches: {image_result['num_patches']}")
                    print(f"Patch Size: {image_result['patch_size']}")
                
                if 'vit_features' in image_result:
                    print(f"\n VISION TRANSFORMER RESULTS:")
                    print(f"ViT Features Shape: {image_result['vit_features'].shape}")
                    print(f"CLS Token Shape: {image_result['vit_cls_token'].shape}")
                    print(f"Patch Tokens Shape: {image_result['vit_patch_tokens'].shape}")
                    print(f"Embedding Dimension: {image_result['vit_embed_dim']}")
                    print(f"Sequence Length: {image_result['vit_sequence_length']}")
                    print(f"Sample CLS Token: {image_result['vit_cls_token'][0, :5].tolist()}")
                    
                    print(f"\n PROCESSING FLOW COMPLETE:")
                    print(f"Image → BLIP2 → Patches → EVA-CLIP-G ViT → Contextualized Features")
                    print(f"Final output ready for multimodal models!")
                
            except Exception as e:
                print(f"Vision Transformer processing error: {e}")
        
        elif choice == '5':
            # Complete pipeline: text and image with full processing
            text_input = input("Enter text: ").strip()
            image_path = input("Enter image path: ").strip()
            
            if not text_input or not image_path or not os.path.exists(image_path):
                print("Invalid text or image path!")
                continue
            
            try:
                # Process text
                tokenize_result = tokenizer.tokenize(text_input, return_tensors=False)
                embed_result = embedding_manager.create_embeddings(text_input, device='cpu')
                
                # Process image with full pipeline
                image_result = image_processor.process_image(
                    image_path, 
                    device='cpu', 
                    create_patches=True, 
                    use_vision_transformer=True
                )
                
                print(f"\n TEXT RESULTS:")
                print(f"Tokens: {tokenize_result['tokens']}")
                print(f"Token IDs: {tokenize_result['token_ids']}")
                print(f"Text Embeddings Shape: {embed_result['embeddings'].shape}")
                print(f"Text Embeddings: {embed_result['embeddings']}")
            
                print(f"\n IMAGE PROCESSING RESULTS:")
                print(f"Image Path: {image_result['image_path']}")
                print(f"Original Size: {image_result['original_size']}")
                print(f"Processed Shape: {image_result['processed_size']}")
                
                if 'patch_embeddings' in image_result:
                    print(f"Patch Embeddings Shape: {image_result['patch_embeddings'].shape}")
                    print(f"Patch Embeddings: {image_result['patch_embeddings']}")
                
                if 'vit_features' in image_result:
                    print(f"ViT Features Shape: {image_result['vit_features'].shape}")
                    print(f"CLS Token Shape: {image_result['vit_cls_token'].shape}")
                
                print(f"\n MULTIMODAL PIPELINE COMPLETE:")
                print(f"Text: {embed_result['embeddings'].shape} BERT embeddings")
                print(f"Image: {image_result['vit_features'].shape if 'vit_features' in image_result else image_result['processed_size']} ViT features")
                print(f"Ready for vision-language models (BLIP2, LLaVA, etc.)!")
                
            except Exception as e:
                print(f"Processing error: {e}")
        
        else:
            print("Invalid choice! Use 1, 2, 3, 4, 5, or 'quit'")
    
    print("Done")


if __name__ == "__main__":
    main()
