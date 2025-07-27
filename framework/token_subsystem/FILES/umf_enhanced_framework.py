"""
Universal Multimodal Framework (UMF) - Enhanced Implementation
Inspired by XRayGPT architecture with domain-agnostic design

This framework provides a comprehensive, extensible multimodal AI system
that can handle vision-language tasks across various domains.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import logging
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path


# ============================================================================
# CORE TOKENIZATION SYSTEM
# ============================================================================

class ModalityType(Enum):
    """Enumeration of supported modality types"""
    VISION = auto()
    AUDIO = auto()
    TEXT = auto()
    SENSOR = auto()
    VIDEO = auto()
    LIDAR = auto()
    MEDICAL_IMAGE = auto()
    THERMAL = auto()


@dataclass
class TokenConfig:
    """Configuration for tokenization parameters"""
    vocab_size: int = 50000
    max_length: int = 2048
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    
    # Modality-specific tokens
    img_token: str = "[IMG]"
    audio_token: str = "[AUDIO]"
    video_token: str = "[VIDEO]"
    sensor_token: str = "[SENSOR]"
    
    # Domain-specific tokens
    medical_token: str = "[MEDICAL]"
    autonomous_token: str = "[AUTO]"
    robotics_token: str = "[ROBOT]"
    education_token: str = "[EDU]"


class UniversalTokenizer:
    """
    Universal tokenizer that handles multiple modalities and domains
    Provides unified token space across all input types
    """
    
    def __init__(self, config: TokenConfig):
        self.config = config
        self.base_tokenizer = None
        self.modality_tokens = {}
        self.domain_tokens = {}
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize the base tokenizer and add special tokens"""
        # Use a pre-trained tokenizer as base
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased", 
            use_fast=True
        )
        
        # Add modality-specific tokens
        special_tokens = [
            self.config.img_token, self.config.audio_token,
            self.config.video_token, self.config.sensor_token,
            self.config.medical_token, self.config.autonomous_token,
            self.config.robotics_token, self.config.education_token
        ]
        
        self.base_tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        
        # Create token mappings
        for token in special_tokens:
            token_id = self.base_tokenizer.convert_tokens_to_ids(token)
            if 'IMG' in token or 'AUDIO' in token or 'VIDEO' in token or 'SENSOR' in token:
                modality = token.replace('[', '').replace(']', '').lower()
                self.modality_tokens[modality] = token_id
            else:
                domain = token.replace('[', '').replace(']', '').lower()
                self.domain_tokens[domain] = token_id
    
    def encode_multimodal(
        self, 
        text: str, 
        modality: ModalityType, 
        domain: Optional[str] = None,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text with modality and domain context
        
        Args:
            text: Input text to encode
            modality: Type of modality (vision, audio, etc.)
            domain: Optional domain context (medical, autonomous, etc.)
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing input_ids, attention_mask, and metadata
        """
        max_length = max_length or self.config.max_length
        
        # Add modality and domain tokens to text
        modality_token = f"[{modality.name}]"
        processed_text = f"{modality_token} {text}"
        
        if domain:
            domain_token = f"[{domain.upper()}]"
            processed_text = f"{domain_token} {processed_text}"
        
        # Tokenize
        encoding = self.base_tokenizer(
            processed_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'modality': modality,
            'domain': domain,
            'raw_text': text
        }
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        return self.base_tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


# ============================================================================
# ENCODER ARCHITECTURE
# ============================================================================

class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all encoders"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder"""
        pass
    
    @property
    def device(self):
        return next(self.parameters()).device


class VisionEncoder(BaseEncoder):
    """Vision encoder supporting multiple architectures"""
    
    def __init__(
        self, 
        model_name: str = "eva_clip_g",
        img_size: int = 224,
        output_dim: int = 768,
        freeze: bool = False
    ):
        super().__init__(input_dim=img_size*img_size*3, output_dim=output_dim)
        
        self.model_name = model_name
        self.img_size = img_size
        
        # Initialize vision backbone
        if model_name == "eva_clip_g":
            # Use EVA-CLIP as in XRayGPT
            from transformers import CLIPVisionModel
            self.backbone = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        else:
            # Default to ViT
            from transformers import ViTModel
            self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        # Projection layer
        backbone_dim = self.backbone.config.hidden_size
        self.projection = nn.Linear(backbone_dim, output_dim)
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (batch_size, channels, height, width)
        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        features = self.backbone(x).last_hidden_state
        # Use CLS token or mean pooling
        if hasattr(self.backbone.config, 'use_cls_token') and self.backbone.config.use_cls_token:
            pooled_features = features[:, 0]  # CLS token
        else:
            pooled_features = features.mean(dim=1)  # Mean pooling
        
        return self.projection(pooled_features)


class AudioEncoder(BaseEncoder):
    """Audio encoder for speech and sound processing"""
    
    def __init__(self, output_dim: int = 768):
        super().__init__(input_dim=16000, output_dim=output_dim)  # Assuming 16kHz audio
        
        # Use Wav2Vec2 or similar
        from transformers import Wav2Vec2Model
        self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        backbone_dim = self.backbone.config.hidden_size
        self.projection = nn.Linear(backbone_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Audio tensor of shape (batch_size, sequence_length)
        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        features = self.backbone(x).last_hidden_state
        pooled_features = features.mean(dim=1)  # Mean pooling over time
        return self.projection(pooled_features)


class SensorEncoder(BaseEncoder):
    """Encoder for sensor data (LiDAR, IMU, GPS, etc.)"""
    
    def __init__(self, sensor_type: str, input_dim: int, output_dim: int = 768):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        
        self.sensor_type = sensor_type
        
        if sensor_type == "lidar":
            # Point cloud processing
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )
        elif sensor_type == "imu":
            # IMU data processing
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        else:
            # Generic sensor encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ============================================================================
# FUSION MECHANISMS
# ============================================================================

class CrossModalFusion(nn.Module):
    """Cross-modal fusion using attention mechanisms"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
    
    def forward(
        self, 
        query_features: torch.Tensor, 
        key_value_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query_features: Query modality features (batch_size, seq_len, feature_dim)
            key_value_features: Key-value modality features (batch_size, seq_len, feature_dim)
        Returns:
            Fused features (batch_size, seq_len, feature_dim)
        """
        # Cross-attention
        attended_features, _ = self.cross_attention(
            query_features, key_value_features, key_value_features
        )
        
        # Residual connection and normalization
        attended_features = self.norm1(attended_features + query_features)
        
        # Feed-forward network
        ffn_output = self.ffn(attended_features)
        output = self.norm2(ffn_output + attended_features)
        
        return output


class QFormerFusion(nn.Module):
    """Q-Former style fusion as used in BLIP-2 and XRayGPT"""
    
    def __init__(self, feature_dim: int, num_query_tokens: int = 32):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, feature_dim)
        )
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Transformer layers for fusion
        self.fusion_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=feature_dim * 4,
                batch_first=True
            )
            for _ in range(6)
        ])
    
    def forward(self, multimodal_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            multimodal_features: List of feature tensors from different modalities
        Returns:
            Fused query features (batch_size, num_query_tokens, feature_dim)
        """
        batch_size = multimodal_features[0].shape[0]
        
        # Expand query tokens for batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Concatenate all modality features
        all_features = torch.cat([query_tokens] + multimodal_features, dim=1)
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            all_features = layer(all_features)
        
        # Return only the query token features
        return all_features[:, :self.num_query_tokens]


# ============================================================================
# DOMAIN-SPECIFIC IMPLEMENTATIONS
# ============================================================================

class DomainAdapter(nn.Module, ABC):
    """Abstract base class for domain-specific adaptations"""
    
    def __init__(self, domain_name: str):
        super().__init__()
        self.domain_name = domain_name
    
    @abstractmethod
    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        """Adapt features for domain-specific processing"""
        pass
    
    @abstractmethod
    def get_domain_prompt(self) -> str:
        """Get domain-specific prompt template"""
        pass


class MedicalAdapter(DomainAdapter):
    """Medical domain adapter for healthcare applications"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__("medical")
        
        # Medical-specific feature transformation
        self.medical_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Medical terminology embedding
        self.medical_vocab_size = 10000  # Medical terms
        self.medical_embeddings = nn.Embedding(self.medical_vocab_size, feature_dim)
    
    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply medical-specific transformations"""
        return self.medical_projection(features)
    
    def get_domain_prompt(self) -> str:
        return (
            "You are an experienced medical professional. "
            "Analyze the provided medical data and provide accurate, "
            "evidence-based insights. Consider patient safety and "
            "medical best practices in your response."
        )


class AutonomousAdapter(DomainAdapter):
    """Autonomous driving domain adapter"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__("autonomous")
        
        # Spatial reasoning enhancement
        self.spatial_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Traffic scenario understanding
        self.scenario_classifier = nn.Linear(feature_dim, 50)  # 50 traffic scenarios
    
    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply autonomous driving specific transformations"""
        return self.spatial_projection(features)
    
    def get_domain_prompt(self) -> str:
        return (
            "You are an autonomous driving AI system. "
            "Analyze the traffic scenario and provide safe, "
            "efficient navigation decisions. Consider road rules, "
            "pedestrian safety, and traffic flow optimization."
        )


class RoboticsAdapter(DomainAdapter):
    """Robotics domain adapter"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__("robotics")
        
        # Action-oriented feature processing
        self.action_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Manipulation planning
        self.manipulation_head = nn.Linear(feature_dim, 6)  # 6-DOF actions
    
    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply robotics-specific transformations"""
        return self.action_projection(features)
    
    def get_domain_prompt(self) -> str:
        return (
            "You are a robotic system capable of understanding "
            "and executing complex manipulation tasks. "
            "Analyze the environment and plan safe, efficient actions "
            "to accomplish the given objectives."
        )


class EducationAdapter(DomainAdapter):
    """Education domain adapter"""
    
    def __init__(self, feature_dim: int = 768):
        super().__init__("education")
        
        # Pedagogical feature enhancement
        self.pedagogy_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Learning level assessment
        self.difficulty_classifier = nn.Linear(feature_dim, 5)  # 5 difficulty levels
    
    def adapt_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply education-specific transformations"""
        return self.pedagogy_projection(features)
    
    def get_domain_prompt(self) -> str:
        return (
            "You are an AI tutor designed to help students learn effectively. "
            "Provide clear, engaging explanations adapted to the student's "
            "level. Use examples and encourage critical thinking."
        )


# ============================================================================
# CONVERSATION SYSTEM
# ============================================================================

class ConversationStyle(Enum):
    """Different conversation styles for various domains"""
    MEDICAL = auto()
    TECHNICAL = auto()
    EDUCATIONAL = auto()
    CASUAL = auto()
    FORMAL = auto()


@dataclass
class ConversationConfig:
    """Configuration for conversation management"""
    style: ConversationStyle = ConversationStyle.CASUAL
    max_turns: int = 10
    max_context_length: int = 2048
    system_prompt: str = ""
    user_role: str = "User"
    assistant_role: str = "Assistant"
    separator: str = "###"


class UniversalConversation:
    """Universal conversation manager supporting multiple domains and styles"""
    
    def __init__(self, config: ConversationConfig, domain_adapter: Optional[DomainAdapter] = None):
        self.config = config
        self.domain_adapter = domain_adapter
        self.messages = []
        self.context_length = 0
        
        # Set domain-specific system prompt if adapter is provided
        if domain_adapter:
            self.config.system_prompt = domain_adapter.get_domain_prompt()
    
    def add_message(self, role: str, content: str, modality_data: Optional[Dict] = None):
        """Add a message to the conversation"""
        message = {
            'role': role,
            'content': content,
            'modality_data': modality_data,
            'timestamp': torch.tensor(len(self.messages))
        }
        self.messages.append(message)
        
        # Manage context length
        self._manage_context()
    
    def _manage_context(self):
        """Manage conversation context to stay within limits"""
        while len(self.messages) > self.config.max_turns:
            self.messages.pop(0)
    
    def get_conversation_prompt(self) -> str:
        """Generate the full conversation prompt"""
        prompt_parts = []
        
        if self.config.system_prompt:
            prompt_parts.append(f"System: {self.config.system_prompt}")
        
        for message in self.messages:
            role = message['role']
            content = message['content']
            prompt_parts.append(f"{role}: {content}")
        
        return f" {self.config.separator} ".join(prompt_parts)
    
    def reset(self):
        """Reset the conversation"""
        self.messages = []
        self.context_length = 0


# ============================================================================
# MAIN FRAMEWORK CLASS
# ============================================================================

class UniversalMultimodalFramework(nn.Module):
    """
    Main framework class that orchestrates all components
    Provides a unified interface for multimodal AI across domains
    """
    
    def __init__(
        self,
        tokenizer_config: TokenConfig,
        feature_dim: int = 768,
        num_query_tokens: int = 32,
        domain_adapters: Optional[Dict[str, DomainAdapter]] = None
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.tokenizer = UniversalTokenizer(tokenizer_config)
        
        # Initialize encoders
        self.encoders = nn.ModuleDict({
            'vision': VisionEncoder(output_dim=feature_dim),
            'audio': AudioEncoder(output_dim=feature_dim),
            'sensor_lidar': SensorEncoder('lidar', input_dim=1024, output_dim=feature_dim),
            'sensor_imu': SensorEncoder('imu', input_dim=6, output_dim=feature_dim)
        })
        
        # Fusion mechanism
        self.fusion = QFormerFusion(feature_dim, num_query_tokens)
        
        # Domain adapters
        self.domain_adapters = nn.ModuleDict(domain_adapters or {})
        
        # Language model integration
        self.language_projection = nn.Linear(feature_dim, feature_dim)
        
        # Initialize conversation managers for different domains
        self.conversations = {}
    
    def encode_modality(
        self, 
        data: torch.Tensor, 
        modality: ModalityType,
        encoder_name: Optional[str] = None
    ) -> torch.Tensor:
        """Encode data from a specific modality"""
        
        if modality == ModalityType.VISION:
            encoder_name = encoder_name or 'vision'
        elif modality == ModalityType.AUDIO:
            encoder_name = encoder_name or 'audio'
        elif modality == ModalityType.SENSOR:
            encoder_name = encoder_name or 'sensor_lidar'
        else:
            raise ValueError(f"Unsupported modality: {modality}")
        
        if encoder_name not in self.encoders:
            raise ValueError(f"Encoder {encoder_name} not found")
        
        return self.encoders[encoder_name](data)
    
    def fuse_multimodal_features(self, features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse features from multiple modalities"""
        feature_list = list(features_dict.values())
        return self.fusion(feature_list)
    
    def adapt_to_domain(self, features: torch.Tensor, domain: str) -> torch.Tensor:
        """Apply domain-specific adaptations"""
        if domain in self.domain_adapters:
            return self.domain_adapters[domain].adapt_features(features)
        return features
    
    def create_conversation(
        self, 
        domain: str, 
        style: ConversationStyle = ConversationStyle.CASUAL
    ) -> UniversalConversation:
        """Create a new conversation for a specific domain"""
        config = ConversationConfig(style=style)
        domain_adapter = self.domain_adapters.get(domain)
        
        conversation = UniversalConversation(config, domain_adapter)
        conversation_id = f"{domain}_{len(self.conversations)}"
        self.conversations[conversation_id] = conversation
        
        return conversation
    
    def forward(
        self,
        multimodal_data: Dict[str, torch.Tensor],
        text_input: str,
        domain: str,
        modality_types: Dict[str, ModalityType]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire framework
        
        Args:
            multimodal_data: Dictionary of modality data
            text_input: Text input string
            domain: Target domain for adaptation
            modality_types: Mapping of data keys to modality types
            
        Returns:
            Dictionary containing processed features and outputs
        """
        # Encode each modality
        encoded_features = {}
        for key, data in multimodal_data.items():
            modality = modality_types[key]
            encoded_features[key] = self.encode_modality(data, modality)
        
        # Fuse multimodal features
        fused_features = self.fuse_multimodal_features(encoded_features)
        
        # Apply domain adaptation
        adapted_features = self.adapt_to_domain(fused_features, domain)
        
        # Project for language model
        language_features = self.language_projection(adapted_features)
        
        # Tokenize text with domain context
        text_encoding = self.tokenizer.encode_multimodal(
            text_input, 
            ModalityType.TEXT, 
            domain
        )
        
        return {
            'encoded_features': encoded_features,
            'fused_features': fused_features,
            'adapted_features': adapted_features,
            'language_features': language_features,
            'text_encoding': text_encoding
        }
    
    def save_framework(self, path: str):
        """Save the entire framework"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'tokenizer_config': self.tokenizer.config,
            'feature_dim': self.feature_dim
        }, path)
    
    @classmethod
    def load_framework(cls, path: str):
        """Load a saved framework"""
        checkpoint = torch.load(path)
        
        framework = cls(
            tokenizer_config=checkpoint['tokenizer_config'],
            feature_dim=checkpoint['feature_dim']
        )
        
        framework.load_state_dict(checkpoint['model_state_dict'])
        return framework


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_medical_framework() -> UniversalMultimodalFramework:
    """Create a framework configured for medical applications"""
    config = TokenConfig()
    
    domain_adapters = {
        'medical': MedicalAdapter()
    }
    
    framework = UniversalMultimodalFramework(
        tokenizer_config=config,
        domain_adapters=domain_adapters
    )
    
    return framework


def create_autonomous_framework() -> UniversalMultimodalFramework:
    """Create a framework configured for autonomous driving"""
    config = TokenConfig()
    
    domain_adapters = {
        'autonomous': AutonomousAdapter()
    }
    
    framework = UniversalMultimodalFramework(
        tokenizer_config=config,
        domain_adapters=domain_adapters
    )
    
    return framework


def create_general_framework() -> UniversalMultimodalFramework:
    """Create a general-purpose framework with all domain adapters"""
    config = TokenConfig()
    
    domain_adapters = {
        'medical': MedicalAdapter(),
        'autonomous': AutonomousAdapter(),
        'robotics': RoboticsAdapter(),
        'education': EducationAdapter()
    }
    
    framework = UniversalMultimodalFramework(
        tokenizer_config=config,
        domain_adapters=domain_adapters
    )
    
    return framework


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create a medical framework
    medical_framework = create_medical_framework()
    
    # Create sample data
    batch_size = 2
    sample_image = torch.randn(batch_size, 3, 224, 224)
    sample_audio = torch.randn(batch_size, 16000)
    
    multimodal_data = {
        'chest_xray': sample_image,
        'heart_sound': sample_audio
    }
    
    modality_types = {
        'chest_xray': ModalityType.VISION,
        'heart_sound': ModalityType.AUDIO
    }
    
    # Process through framework
    output = medical_framework(
        multimodal_data=multimodal_data,
        text_input="Analyze this chest X-ray for any abnormalities",
        domain="medical",
        modality_types=modality_types
    )
    
    print("Framework output shapes:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        elif isinstance(value, dict) and 'input_ids' in value:
            print(f"{key}: {value['input_ids'].shape}")
    
    # Create and use conversation
    conversation = medical_framework.create_conversation("medical", ConversationStyle.MEDICAL)
    conversation.add_message("Patient", "I have chest pain and shortness of breath")
    conversation.add_message("Doctor", "I'll analyze your X-ray to help with diagnosis")
    
    print(f"\nConversation prompt:\n{conversation.get_conversation_prompt()}")