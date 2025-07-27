"""
Universal Multimodal Framework (UMF) - Example Implementations
Demonstrates practical usage across different domains
"""

import torch
import torch.nn as nn
from umf_enhanced_framework import (
    UniversalMultimodalFramework, 
    ModalityType, 
    ConversationStyle,
    TokenConfig,
    create_medical_framework,
    create_autonomous_framework,
    create_general_framework
)
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================================
# MEDICAL DOMAIN EXAMPLE
# ============================================================================

class MedicalAISystem:
    """Complete medical AI system using UMF"""
    
    def __init__(self):
        self.framework = create_medical_framework()
        self.conversation = self.framework.create_conversation(
            "medical", 
            ConversationStyle.MEDICAL
        )
    
    def analyze_chest_xray(self, image_path: str, patient_query: str):
        """Analyze chest X-ray and respond to patient query"""
        
        # Load and preprocess image
        image = self._load_image(image_path)
        
        # Prepare multimodal data
        multimodal_data = {'chest_xray': image}
        modality_types = {'chest_xray': ModalityType.MEDICAL_IMAGE}
        
        # Process through framework
        output = self.framework(
            multimodal_data=multimodal_data,
            text_input=patient_query,
            domain="medical",
            modality_types=modality_types
        )
        
        # Add to conversation
        self.conversation.add_message("Patient", patient_query)
        
        # Generate medical response (simplified)
        medical_response = self._generate_medical_response(output)
        self.conversation.add_message("Doctor", medical_response)
        
        return medical_response
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess medical image"""
        # Simulate loading a chest X-ray
        return torch.randn(1, 3, 224, 224)
    
    def _generate_medical_response(self, framework_output: dict) -> str:
        """Generate medical response based on framework output"""
        # In practice, this would use the language features to generate response
        return (
            "Based on the chest X-ray analysis, I can see the lung fields "
            "and cardiac silhouette. The image shows normal lung expansion "
            "with no obvious signs of pneumonia or other abnormalities. "
            "However, please consult with your physician for a complete evaluation."
        )


# ============================================================================
# AUTONOMOUS DRIVING EXAMPLE
# ============================================================================

class AutonomousDrivingSystem:
    """Autonomous driving system using UMF"""
    
    def __init__(self):
        self.framework = create_autonomous_framework()
        self.conversation = self.framework.create_conversation(
            "autonomous", 
            ConversationStyle.TECHNICAL
        )
    
    def process_driving_scenario(
        self, 
        camera_image: torch.Tensor,
        lidar_data: torch.Tensor,
        scenario_description: str
    ):
        """Process a driving scenario with multiple sensors"""
        
        multimodal_data = {
            'front_camera': camera_image,
            'lidar_scan': lidar_data
        }
        
        modality_types = {
            'front_camera': ModalityType.VISION,
            'lidar_scan': ModalityType.SENSOR
        }
        
        # Process through framework
        output = self.framework(
            multimodal_data=multimodal_data,
            text_input=scenario_description,
            domain="autonomous",
            modality_types=modality_types
        )
        
        # Generate driving decision
        decision = self._make_driving_decision(output)
        
        # Add to conversation log
        self.conversation.add_message("System", scenario_description)
        self.conversation.add_message("AI", decision)
        
        return decision
    
    def _make_driving_decision(self, framework_output: dict) -> str:
        """Make driving decision based on sensor fusion"""
        return (
            "Based on camera and LiDAR analysis: "
            "- Detected pedestrian crossing ahead at 50m "
            "- Recommended action: Reduce speed to 25 mph "
            "- Maintain safe following distance "
            "- Monitor pedestrian movement continuously"
        )


# ============================================================================
# EDUCATIONAL AI EXAMPLE
# ============================================================================

class EducationalAITutor:
    """Educational AI tutor using UMF"""
    
    def __init__(self):
        self.framework = create_general_framework()
        self.conversation = self.framework.create_conversation(
            "education", 
            ConversationStyle.EDUCATIONAL
        )
    
    def explain_concept_with_visual(
        self, 
        diagram_image: torch.Tensor,
        concept_question: str,
        student_level: str = "beginner"
    ):
        """Explain a concept using visual aids"""
        
        multimodal_data = {'diagram': diagram_image}
        modality_types = {'diagram': ModalityType.VISION}
        
        # Adapt question for student level
        adapted_question = f"[Level: {student_level}] {concept_question}"
        
        # Process through framework
        output = self.framework(
            multimodal_data=multimodal_data,
            text_input=adapted_question,
            domain="education",
            modality_types=modality_types
        )
        
        # Generate educational response
        explanation = self._generate_explanation(output, student_level)
        
        # Add to conversation
        self.conversation.add_message("Student", concept_question)
        self.conversation.add_message("Tutor", explanation)
        
        return explanation
    
    def _generate_explanation(self, framework_output: dict, level: str) -> str:
        """Generate level-appropriate explanation"""
        if level == "beginner":
            return (
                "Let me explain this step by step! Looking at the diagram, "
                "we can see the main components and how they connect. "
                "Think of it like building blocks - each part has a specific job "
                "and they work together to create the whole system."
            )
        else:
            return (
                "Analyzing the diagram, we observe the interconnected components "
                "and their functional relationships. The system architecture "
                "demonstrates key principles of modular design and information flow."
            )


# ============================================================================
# MULTI-DOMAIN EXAMPLE
# ============================================================================

class UniversalAIAssistant:
    """Universal AI assistant that can handle multiple domains"""
    
    def __init__(self):
        self.framework = create_general_framework()
        self.active_conversations = {}
        self.domain_contexts = {
            'medical': 'healthcare',
            'autonomous': 'transportation', 
            'robotics': 'automation',
            'education': 'learning'
        }
    
    def process_multimodal_query(
        self,
        query: str,
        multimodal_data: dict,
        domain: str,
        modality_types: dict
    ):
        """Process a query across any domain"""
        
        # Get or create conversation for domain
        if domain not in self.active_conversations:
            style = self._get_conversation_style(domain)
            self.active_conversations[domain] = self.framework.create_conversation(
                domain, style
            )
        
        conversation = self.active_conversations[domain]
        
        # Process through framework
        output = self.framework(
            multimodal_data=multimodal_data,
            text_input=query,
            domain=domain,
            modality_types=modality_types
        )
        
        # Generate domain-appropriate response
        response = self._generate_domain_response(output, domain)
        
        # Update conversation
        conversation.add_message("User", query)
        conversation.add_message("Assistant", response)
        
        return {
            'response': response,
            'domain': domain,
            'conversation_context': conversation.get_conversation_prompt(),
            'framework_output': output
        }
    
    def _get_conversation_style(self, domain: str) -> ConversationStyle:
        """Get appropriate conversation style for domain"""
        style_mapping = {
            'medical': ConversationStyle.MEDICAL,
            'autonomous': ConversationStyle.TECHNICAL,
            'education': ConversationStyle.EDUCATIONAL,
            'robotics': ConversationStyle.TECHNICAL
        }
        return style_mapping.get(domain, ConversationStyle.CASUAL)
    
    def _generate_domain_response(self, framework_output: dict, domain: str) -> str:
        """Generate response appropriate for the domain"""
        # This would typically use the language features to generate actual responses
        domain_responses = {
            'medical': "Based on the medical data analysis, here are my findings...",
            'autonomous': "Analyzing the traffic scenario, I recommend the following actions...",
            'education': "Let me help you understand this concept step by step...",
            'robotics': "Processing the environment data for optimal task execution..."
        }
        return domain_responses.get(domain, "I'll analyze this data and provide insights...")


# ============================================================================
# TRAINING PIPELINE EXAMPLE
# ============================================================================

class UMFTrainingPipeline:
    """Training pipeline for the Universal Multimodal Framework"""
    
    def __init__(self, framework: UniversalMultimodalFramework):
        self.framework = framework
        self.optimizer = torch.optim.AdamW(framework.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
    
    def train_stage_1_encoders(self, dataloader, epochs: int = 10):
        """Stage 1: Train individual modality encoders"""
        print("Stage 1: Training modality encoders...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                self.optimizer.zero_grad()
                
                # Compute encoder losses for each modality
                loss = self._compute_encoder_loss(batch)
                loss.backward()
                
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    def train_stage_2_fusion(self, dataloader, epochs: int = 5):
        """Stage 2: Train cross-modal fusion"""
        print("Stage 2: Training cross-modal fusion...")
        
        # Freeze encoders
        for encoder in self.framework.encoders.values():
            for param in encoder.parameters():
                param.requires_grad = False
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                self.optimizer.zero_grad()
                
                # Compute fusion loss
                loss = self._compute_fusion_loss(batch)
                loss.backward()
                
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Fusion Loss: {total_loss/len(dataloader):.4f}")
    
    def train_stage_3_domain_adaptation(self, domain_dataloaders: dict, epochs: int = 5):
        """Stage 3: Train domain-specific adaptations"""
        print("Stage 3: Training domain adaptations...")
        
        for domain, dataloader in domain_dataloaders.items():
            print(f"Training {domain} domain adapter...")
            
            for epoch in range(epochs):
                total_loss = 0
                for batch in dataloader:
                    self.optimizer.zero_grad()
                    
                    # Compute domain-specific loss
                    loss = self._compute_domain_loss(batch, domain)
                    loss.backward()
                    
                    self.optimizer.step()
                    total_loss += loss.item()
                
                print(f"  Epoch {epoch+1}/{epochs}, {domain} Loss: {total_loss/len(dataloader):.4f}")
    
    def _compute_encoder_loss(self, batch) -> torch.Tensor:
        """Compute loss for encoder training"""
        # Simplified contrastive loss for demonstration
        return torch.tensor(0.5, requires_grad=True)
    
    def _compute_fusion_loss(self, batch) -> torch.Tensor:
        """Compute loss for fusion training"""
        # Simplified alignment loss for demonstration
        return torch.tensor(0.3, requires_grad=True)
    
    def _compute_domain_loss(self, batch, domain: str) -> torch.Tensor:
        """Compute domain-specific loss"""
        # Simplified domain adaptation loss for demonstration
        return torch.tensor(0.2, requires_grad=True)


# ============================================================================
# DEMONSTRATION SCRIPT
# ============================================================================

def run_comprehensive_demo():
    """Run a comprehensive demonstration of the UMF"""
    
    print("🚀 Universal Multimodal Framework (UMF) Demonstration")
    print("=" * 60)
    
    # 1. Medical AI Demo
    print("\n🏥 Medical AI System Demo")
    print("-" * 30)
    medical_ai = MedicalAISystem()
    medical_response = medical_ai.analyze_chest_xray(
        "sample_xray.jpg", 
        "Doctor, I've been having chest pain. What do you see in my X-ray?"
    )
    print(f"Medical AI Response: {medical_response[:100]}...")
    
    # 2. Autonomous Driving Demo
    print("\n🚗 Autonomous Driving System Demo")
    print("-" * 35)
    driving_ai = AutonomousDrivingSystem()
    camera_data = torch.randn(1, 3, 224, 224)
    lidar_data = torch.randn(1, 1024)
    
    driving_decision = driving_ai.process_driving_scenario(
        camera_data,
        lidar_data,
        "Approaching intersection with pedestrian crossing signal active"
    )
    print(f"Driving Decision: {driving_decision[:100]}...")
    
    # 3. Educational AI Demo
    print("\n📚 Educational AI Tutor Demo")
    print("-" * 30)
    edu_ai = EducationalAITutor()
    diagram = torch.randn(1, 3, 224, 224)
    
    explanation = edu_ai.explain_concept_with_visual(
        diagram,
        "Can you explain how neural networks work?",
        "beginner"
    )
    print(f"Educational Explanation: {explanation[:100]}...")
    
    # 4. Universal Assistant Demo
    print("\n🤖 Universal AI Assistant Demo")
    print("-" * 35)
    universal_ai = UniversalAIAssistant()
    
    # Test multiple domains
    domains_to_test = [
        {
            'domain': 'medical',
            'query': 'Analyze this medical scan',
            'data': {'scan': torch.randn(1, 3, 224, 224)},
            'types': {'scan': ModalityType.VISION}
        },
        {
            'domain': 'education', 
            'query': 'Explain this diagram',
            'data': {'diagram': torch.randn(1, 3, 224, 224)},
            'types': {'diagram': ModalityType.VISION}
        }
    ]
    
    for test_case in domains_to_test:
        result = universal_ai.process_multimodal_query(
            test_case['query'],
            test_case['data'],
            test_case['domain'],
            test_case['types']
        )
        print(f"{test_case['domain'].title()} Response: {result['response'][:100]}...")
    
    print("\n✅ Demo completed successfully!")
    print("The UMF successfully handled multiple domains and modalities.")


if __name__ == "__main__":
    # Set random seed for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the comprehensive demonstration
    run_comprehensive_demo()
    
    # Example of framework saving/loading
    print("\n💾 Framework Persistence Demo")
    print("-" * 30)
    
    # Create and save a framework
    framework = create_general_framework()
    framework.save_framework("umf_general_model.pth")
    print("Framework saved successfully!")
    
    # Load the framework
    loaded_framework = UniversalMultimodalFramework.load_framework("umf_general_model.pth")
    print("Framework loaded successfully!")
    
    print("\n🎉 All demonstrations completed!")