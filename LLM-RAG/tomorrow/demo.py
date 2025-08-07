
import logging

# Set logging to only show warnings and errors BEFORE importing other modules
logging.basicConfig(level=logging.WARNING, force=True)

import os
from text_tokenizer_class import Tokenizer
from text_embedding_class import EmbeddingManager
from image_embeddings import ImageProcessor


def main():
    
    # Get text input (this was missing in original)
    text_input = input("Enter text to process: ").strip()
    if not text_input:
        text_input = "Hello World! This is a sample text for processing."
        print(f"Using default text: {text_input}")
    
    print("Initializing components...")
    
    # Initialize components
    tokenizer = Tokenizer(enable_input_processing=False)
    embedding_manager = EmbeddingManager()
    image_processor = ImageProcessor()
    
    # Load models from config
    print("Loading models...")
    tokenizer.load_tokenizer('bert')
    embedding_manager.load_embedding_model('bert', device='cpu')
    image_processor.load_processor('blip2')
    
    print("Models loaded successfully!")
  
    # Tokenize and embed text
    tokenize_result = tokenizer.tokenize(text_input, return_tensors=False)
    embed_result = embedding_manager.create_embeddings(text_input, device='cpu')
                
    print(f"\nTEXT RESULTS:")
    print(f"   Text: {tokenize_result['text']}")
    print(f"   Tokens: {tokenize_result['tokens']}")
    print(f"   Tokens IDs: {tokenize_result['token_ids']}")
    print(f"   Embeddings Shape: {embed_result['embeddings'].shape}")
    print(f"   Embeddings: {embed_result['embeddings']}")
    print(f"   Sample Embedding Values: {embed_result['embeddings'][0, 0, :3].tolist()}")
       
    # Image processing
    image_path = input("Enter image path (or press Enter for test image): ").strip()
    
    # Create test image if no path provided
    if not image_path or not os.path.exists(image_path):
        print("Creating test image...")
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image = Image.fromarray(test_array)
        image_path = "demo_test_image.jpg"
        test_image.save(image_path)
        print(f"Test image created: {image_path}")
           
    # Process image
    image_result = image_processor.process_image(image_path, device='cpu')
                
    print(f"\nIMAGE RESULTS:")
    print(f"   Original Size: {image_result['original_size']}")
    print(f"   Processed Shape: {image_result['processed_size']}")
                
    # Process image with patch embeddings
    image_result = image_processor.process_image(image_path, device='cpu', create_patches=True)
                
               
    if 'patch_embeddings' in image_result:
        print(f"\nPATCH EMBEDDING RESULTS:")
        print(f"   Patch Embeddings Shape: {image_result['patch_embeddings'].shape}")
        print(f"   Patch Embeddings : {image_result['patch_embeddings']}")
        print(f"   Number of Patches: {image_result['num_patches']} ({image_result['patch_shape'][0]}x{image_result['patch_shape'][1]} grid)")
        print(f"   Embedding Dimension: {image_result['embed_dim']}")
        print(f"   Sample Values: {image_result['patch_embeddings'][0, 0, :3].tolist()}")
    
    # Clean up test image if we created it
    if image_path == "demo_test_image.jpg" and os.path.exists(image_path):
        os.remove(image_path)
        print(f"\nCleaned up test image")
    
    print(f"\nProcessing completed successfully!")
                
          

if __name__ == "__main__":
    main()
