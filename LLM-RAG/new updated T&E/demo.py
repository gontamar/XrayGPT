"""
Text Processing Pipeline
Input → Process → Tokenize → Embed
"""

from tokenizer_class import Tokenizer
from embedding_class import Embedding
from input_processor import InputProcessor


def process_text(text):
    """Separate function to process input text and remove unwanted patterns."""
    processor = InputProcessor()
    result = processor.process_input(text)
    return result.processed_text if result.processed_text else result.original_text


def main():
    """Text processing pipeline with separate processing function."""
    
    print("Text Processing Pipeline")
    print("Input → Process → Tokenize → Embed")
    
    # Initialize tokenizer and embedding manager
    tokenizer = Tokenizer(enable_input_processing=False)  # Disable built-in processing
    embedding = Embedding()
    
    # Load BERT from config
    tokenizer.load_tokenizer('bert')
    embedding.load_embedding_model('bert', device='cpu')
    
    
    while True:
        # Step 1: Get user input
        user_input = input("\nEnter text (or 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if not user_input:
            continue
        
        try:
            # Step 2: Process text with separate function and validate
            from input_processor import InputProcessor
            processor = InputProcessor()
            validation_result = processor.validate_input(user_input)
            
            if not validation_result.is_valid:
                print(f" Invalid text format: {validation_result.errors}")
                continue
            
            processed_text = process_text(user_input)
            print(f"Processed: {processed_text}")
            
            # Step 3: Tokenize processed text
            tokenize_result = tokenizer.tokenize(processed_text, return_tensors=False)
            tokens = tokenize_result['tokens']
            print(f"Tokens: {tokens}")
            token_ids = tokenize_result['token_ids']
            print(f"Token IDs: {token_ids}")
            
            # Step 4: Create embeddings from tokenization output
            embed_result = embedding.create_embeddings(processed_text, device='cpu')
            embeddings = embed_result['embeddings']
            
            print(f"Embeddings: {embeddings.shape}")
            print(f"Values: {embeddings[0, 0, :3].tolist()}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("Done")


if __name__ == "__main__":
    main()