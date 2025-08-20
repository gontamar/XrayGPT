import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import logging
from text_tokenizer_class import Tokenizer
from text_embedding_class import EmbeddingManager
from image_embeddings import ImageProcessor
from attention_layers import VisionLanguageFusion, create_padding_mask
from mlp_heads import ResponseGenerationHead, ClassificationHead, MultiTaskHead

logger = logging.getLogger(__name__)


class XrayGPTModel(nn.Module):
    """
    XrayGPT-like multimodal model for medical image analysis and text generation.
    Combines vision and language understanding with attention mechanisms.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.d_model = config.get('d_model', 768)
        self.num_heads = config.get('num_heads', 12)
        self.num_fusion_layers = config.get('num_fusion_layers', 6)
        self.vocab_size = config.get('vocab_size', 30522)  # BERT vocab size
        self.num_classes = config.get('num_classes', 10)  # For classification tasks
        self.dropout = config.get('dropout', 0.1)
        
        # Initialize components
        self._init_tokenizer()
        self._init_embedding_manager()
        self._init_image_processor()
        self._init_fusion_layers()
        self._init_output_heads()
        
        # Model state
        self.is_training = True
        
    def _init_tokenizer(self):
        """Initialize text tokenizer."""
        tokenizer_config = self.config.get('tokenizer_config_path', 'tokenizer_config.yaml')
        self.tokenizer_manager = Tokenizer(tokenizer_config, enable_input_processing=False)
        
    def _init_embedding_manager(self):
        """Initialize text embedding manager."""
        self.embedding_manager = EmbeddingManager()
        
    def _init_image_processor(self):
        """Initialize image processor."""
        self.image_processor = ImageProcessor()
        
    def _init_fusion_layers(self):
        """Initialize vision-language fusion layers."""
        self.vision_language_fusion = VisionLanguageFusion(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_fusion_layers,
            dropout=self.dropout
        )
        
        # Dimension alignment layers
        self.vision_dim_align = nn.Linear(self.d_model, self.d_model)
        self.text_dim_align = nn.Linear(self.d_model, self.d_model)
        
        # Positional encodings
        self.vision_pos_encoding = nn.Parameter(torch.randn(1, 196, self.d_model))  # 14x14 patches
        self.text_pos_encoding = nn.Parameter(torch.randn(1, 512, self.d_model))    # Max text length
        
    def _init_output_heads(self):
        """Initialize output heads for different tasks."""
        task_type = self.config.get('task_type', 'generation')
        
        if task_type == 'generation':
            self.output_head = ResponseGenerationHead(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                dropout=self.dropout
            )
        elif task_type == 'classification':
            self.output_head = ClassificationHead(
                d_model=self.d_model,
                num_classes=self.num_classes,
                dropout=self.dropout
            )
        elif task_type == 'multitask':
            self.output_head = MultiTaskHead(
                d_model=self.d_model,
                vocab_size=self.vocab_size,
                num_classes=self.num_classes,
                dropout=self.dropout
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def load_models(self, tokenizer_name: str = 'bert', 
                   image_processor_name: str = 'blip2', device: str = 'cpu'):
        """Load all sub-models."""
        logger.info("Loading XrayGPT models...")
        
        # Load tokenizer
        self.tokenizer_manager.load_tokenizer(tokenizer_name)
        
        # Load embedding model
        self.embedding_manager.load_embedding_model(tokenizer_name, device=device)
        
        # Load image processor
        self.image_processor.load_processor(image_processor_name)
        
        # Update vocab size if needed
        if hasattr(self.tokenizer_manager.tokenizer, 'vocab_size'):
            actual_vocab_size = self.tokenizer_manager.tokenizer.vocab_size
            if actual_vocab_size != self.vocab_size:
                logger.info(f"Updating vocab size from {self.vocab_size} to {actual_vocab_size}")
                self.vocab_size = actual_vocab_size
                # Reinitialize output head with correct vocab size
                self._init_output_heads()
        
        # Move model to device BEFORE processing any data
        self.to(device)
        
        # Ensure all components are on the same device
        if hasattr(self.embedding_manager, 'embedding_model') and self.embedding_manager.embedding_model is not None:
            self.embedding_manager.embedding_model = self.embedding_manager.embedding_model.to(device)
        
        logger.info(f"XrayGPT models loaded successfully on {device}")
    
    def encode_text(self, text: Union[str, List[str]], device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Encode text input to embeddings."""
        # Get text embeddings
        embed_result = self.embedding_manager.create_embeddings(text, device=device)
        text_embeddings = embed_result['embeddings']  # [batch_size, seq_len, d_model]
        
        # Ensure text embeddings are on the correct device
        text_embeddings = text_embeddings.to(device)
        
        # Align dimensions if needed
        if text_embeddings.size(-1) != self.d_model:
            # Ensure text_dim_align is on the correct device
            self.text_dim_align = self.text_dim_align.to(device)
            text_embeddings = self.text_dim_align(text_embeddings)
        
        # Add positional encoding
        seq_len = text_embeddings.size(1)
        if seq_len <= self.text_pos_encoding.size(1):
            # Ensure positional encoding is on the correct device
            self.text_pos_encoding = self.text_pos_encoding.to(device)
            pos_encoding = self.text_pos_encoding[:, :seq_len, :]
            text_embeddings = text_embeddings + pos_encoding
        
        # Create attention mask
        input_ids = embed_result['input_ids']
        text_mask = create_padding_mask(input_ids, pad_token_id=0)
        text_mask = text_mask.to(device)  # Ensure mask is on correct device
        
        return {
            'embeddings': text_embeddings,
            'mask': text_mask,
            'input_ids': input_ids
        }
    
    def encode_image(self, image_path: str, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Encode image input to patch embeddings."""
        # Process image and get patch embeddings
        image_result = self.image_processor.process_image(
            image_path, device=device, create_patches=True
        )
        
        vision_embeddings = image_result['patch_embeddings']  # [batch_size, num_patches, d_model]
        
        # Ensure vision embeddings are on the correct device
        vision_embeddings = vision_embeddings.to(device)
        
        # Align dimensions if needed
        if vision_embeddings.size(-1) != self.d_model:
            # Ensure vision_dim_align is on the correct device
            self.vision_dim_align = self.vision_dim_align.to(device)
            vision_embeddings = self.vision_dim_align(vision_embeddings)
        
        # Add positional encoding
        num_patches = vision_embeddings.size(1)
        if num_patches <= self.vision_pos_encoding.size(1):
            # Ensure positional encoding is on the correct device
            self.vision_pos_encoding = self.vision_pos_encoding.to(device)
            pos_encoding = self.vision_pos_encoding[:, :num_patches, :]
            vision_embeddings = vision_embeddings + pos_encoding
        
        # Create vision mask (all patches are valid)
        batch_size = vision_embeddings.size(0)
        vision_mask = torch.ones(batch_size, num_patches, device=device)
        
        return {
            'embeddings': vision_embeddings,
            'mask': vision_mask,
            'image_tensor': image_result['image_tensor'],
            'patch_info': {
                'num_patches': image_result['num_patches'],
                'patch_shape': image_result['patch_shape'],
                'embed_dim': image_result['embed_dim']
            }
        }
    
    def forward(self, text_input: Union[str, List[str]], image_path: str,
                device: str = 'cpu', task: str = 'generation') -> Dict[str, Any]:
        """
        Forward pass of the XrayGPT model.
        
        Args:
            text_input: Input text (question/prompt)
            image_path: Path to input image
            device: Device to run on
            task: Task type ('generation', 'classification', 'both')
        
        Returns:
            Dictionary containing model outputs
        """
        # Encode inputs
        text_encoded = self.encode_text(text_input, device)
        image_encoded = self.encode_image(image_path, device)
        
        # Extract features and masks
        text_features = text_encoded['embeddings']
        text_mask = text_encoded['mask']
        vision_features = image_encoded['embeddings']
        vision_mask = image_encoded['mask']
        
        # Ensure fusion layer is on correct device
        self.vision_language_fusion = self.vision_language_fusion.to(device)
        
        # Vision-Language Fusion
        fusion_result = self.vision_language_fusion(
            text_features=text_features,
            vision_features=vision_features,
            text_mask=text_mask,
            vision_mask=vision_mask
        )
        
        fused_features = fusion_result['fused_features']
        
        # Ensure output head is on correct device
        self.output_head = self.output_head.to(device)
        
        # Generate outputs based on task
        if isinstance(self.output_head, MultiTaskHead):
            outputs = self.output_head(fused_features, task=task, mask=text_mask)
        elif isinstance(self.output_head, ResponseGenerationHead):
            generation_logits = self.output_head(fused_features)
            outputs = {'generation_logits': generation_logits}
        elif isinstance(self.output_head, ClassificationHead):
            classification_logits = self.output_head(fused_features, mask=text_mask)
            outputs = {'classification_logits': classification_logits}
        
        # Add intermediate results
        outputs.update({
            'fused_features': fused_features,
            'text_features': fusion_result['text_features'],
            'vision_features': fusion_result['vision_features'],
            'attention_weights': fusion_result['attention_weights'],
            'text_mask': text_mask,
            'vision_mask': vision_mask
        })
        
        return outputs
    
    def generate_response(self, text_input: Union[str, List[str]], image_path: str,
                         max_length: int = 100, temperature: float = 1.0,
                         top_p: float = 0.9, top_k: int = 50,
                         device: str = 'cpu') -> Dict[str, Any]:
        """
        Generate text response given text and image inputs.
        
        Args:
            text_input: Input text (question/prompt)
            image_path: Path to input image
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            device: Device to run on
        
        Returns:
            Dictionary with generated response and metadata
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(text_input, image_path, device, task='generation')
            
            # Generate response
            if isinstance(self.output_head, (ResponseGenerationHead, MultiTaskHead)):
                if isinstance(self.output_head, MultiTaskHead):
                    generation_head = self.output_head.generation_head
                else:
                    generation_head = self.output_head
                
                generation_result = generation_head.generate_response(
                    fused_features=outputs['fused_features'],
                    tokenizer=self.tokenizer_manager.tokenizer,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )
                
                outputs.update(generation_result)
            else:
                logger.warning("Model not configured for text generation")
                outputs['generated_text'] = ["Model not configured for generation"]
        
        if self.is_training:
            self.train()
        
        return outputs
    
    def classify(self, text_input: Union[str, List[str]], image_path: str,
                device: str = 'cpu') -> Dict[str, Any]:
        """
        Classify input based on text and image.
        
        Args:
            text_input: Input text
            image_path: Path to input image
            device: Device to run on
        
        Returns:
            Dictionary with classification results
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(text_input, image_path, device, task='classification')
            
            if 'classification_logits' in outputs:
                # Get predictions
                logits = outputs['classification_logits']
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                outputs.update({
                    'probabilities': probabilities,
                    'predictions': predictions,
                    'confidence': torch.max(probabilities, dim=-1)[0]
                })
            else:
                logger.warning("Model not configured for classification")
        
        if self.is_training:
            self.train()
        
        return outputs
    
    def save_model(self, save_path: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'vocab_size': self.vocab_size
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str, device: str = 'cpu'):
        """Load model state."""
        checkpoint = torch.load(load_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.vocab_size = checkpoint['vocab_size']
        logger.info(f"Model loaded from {load_path}")


def create_xraygpt_config(d_model: int = 768, num_heads: int = 12, 
                         num_fusion_layers: int = 6, vocab_size: int = 30522,
                         num_classes: int = 10, task_type: str = 'generation',
                         dropout: float = 0.1) -> Dict[str, Any]:
    """Create default XrayGPT configuration."""
    return {
        'd_model': d_model,
        'num_heads': num_heads,
        'num_fusion_layers': num_fusion_layers,
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'task_type': task_type,  # 'generation', 'classification', 'multitask'
        'dropout': dropout,
        'tokenizer_config_path': 'tokenizer_config.yaml'
    }


# Example usage functions
def create_generation_model(device: str = 'cpu') -> XrayGPTModel:
    """Create XrayGPT model for text generation."""
    config = create_xraygpt_config(task_type='generation')
    model = XrayGPTModel(config)
    model.load_models(device=device)
    return model


def create_classification_model(num_classes: int = 10, device: str = 'cpu') -> XrayGPTModel:
    """Create XrayGPT model for classification."""
    config = create_xraygpt_config(task_type='classification', num_classes=num_classes)
    model = XrayGPTModel(config)
    model.load_models(device=device)
    return model


def create_multitask_model(num_classes: int = 10, device: str = 'cpu') -> XrayGPTModel:
    """Create XrayGPT model for both generation and classification."""
    config = create_xraygpt_config(task_type='multitask', num_classes=num_classes)
    model = XrayGPTModel(config)
    model.load_models(device=device)
    return model