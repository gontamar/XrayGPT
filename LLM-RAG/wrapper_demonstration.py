"""
Demonstration of XrayGPT Module Wrappers
Implementing concepts from Discussion_Summary.docx

This script demonstrates how the wrapper functions implement:
1. Pattern Recognition (like "How do you" → procedural requests)
2. Domain Knowledge Association (like "load current" → electrical measurement)
3. Cross-modal Understanding for medical AI applications
"""

import torch
import numpy as np
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from xraygpt_module_wrappers import (
    MedicalVisualEncoderWrapper,
    MultiModalProjectionWrapper, 
    SelfAttentionWrapper,
    PatternType,
    create_pattern_recognition_pipeline
)

class WrapperDemonstration:
    """Demonstration class showing wrapper functionality"""
    
    def __init__(self):
        self.demo_data = self._create_demo_data()
        
    def _create_demo_data(self) -> Dict[str, torch.Tensor]:
        """Create synthetic demo data for illustration"""
        return {
            "chest_xray_image": torch.randn(1, 3, 224, 224),  # Simulated chest X-ray
            "visual_features": torch.randn(1, 197, 1408),     # EVA-ViT features
            "text_embeddings": torch.randn(1, 32, 768),       # Text embeddings
            "attention_mask": torch.ones(1, 229)              # Attention mask
        }
    
    def demonstrate_pattern_recognition(self):
        """
        Demonstrate Pattern Recognition concept from Discussion Summary
        
        From document: "The phrase 'How do you' is commonly associated with 
        requests for explanations or procedures"
        
        Medical equivalent: Visual patterns in chest X-rays are associated 
        with specific medical conditions
        """
        print("=" * 60)
        print("1. PATTERN RECOGNITION DEMONSTRATION")
        print("=" * 60)
        print("Concept: Visual medical patterns → Diagnostic associations")
        print()
        
        # Create mock visual encoder components
        class MockVisualEncoder:
            def forward_features(self, x):
                return torch.randn(1, 197, 1408)
            
            @property
            def blocks(self):
                class MockBlock:
                    def __init__(self):
                        self.attn = MockAttention()
                return [MockBlock() for _ in range(12)]
        
        class MockAttention:
            def register_forward_hook(self, hook):
                class MockHook:
                    def remove(self): pass
                return MockHook()
        
        class MockLayerNorm:
            def __call__(self, x):
                return x
        
        # Initialize wrapper
        visual_wrapper = MedicalVisualEncoderWrapper(
            MockVisualEncoder(),
            MockLayerNorm()
        )
        
        # Demonstrate pattern recognition
        print("Processing chest X-ray image...")
        pattern_output = visual_wrapper.recognize_medical_patterns(
            self.demo_data["chest_xray_image"],
            return_attention=True,
            pattern_threshold=0.5
        )
        
        print(f"✓ Pattern Type: {pattern_output.pattern_type.value}")
        print(f"✓ Confidence: {pattern_output.confidence:.3f}")
        print(f"✓ Recognized Condition: {pattern_output.metadata['recognized_condition']}")
        print(f"✓ Feature Shape: {pattern_output.features.shape}")
        
        # Show medical pattern library
        print("\nMedical Pattern Library:")
        for condition, details in visual_wrapper.medical_pattern_library.items():
            print(f"  • {condition}: {details['description']}")
        
        print("\n" + "="*60)
        return pattern_output
    
    def demonstrate_domain_knowledge_association(self):
        """
        Demonstrate Domain Knowledge Association from Discussion Summary
        
        From document: "The model recognizes 'load current' as an electrical 
        concept related to current flowing through a circuit"
        
        Medical equivalent: Visual features are associated with clinical 
        terminology and diagnostic concepts
        """
        print("2. DOMAIN KNOWLEDGE ASSOCIATION DEMONSTRATION")
        print("=" * 60)
        print("Concept: Visual features → Clinical terminology mapping")
        print()
        
        # Create mock projection layer
        projection_layer = torch.nn.Linear(1408, 768)
        
        # Initialize wrapper
        projection_wrapper = MultiModalProjectionWrapper(
            projection_layer,
            hidden_size=768
        )
        
        # Demonstrate domain knowledge association
        print("Associating visual features with medical domain knowledge...")
        domain_output = projection_wrapper.associate_domain_knowledge(
            self.demo_data["visual_features"],
            text_context="chest x-ray shows consolidation in lower lobe",
            association_strength_threshold=0.3
        )
        
        print(f"✓ Source Domain: {domain_output.source_domain}")
        print(f"✓ Target Domain: {domain_output.target_domain}")
        print(f"✓ Association Strength: {domain_output.association_strength:.3f}")
        print(f"✓ Mapped Features Shape: {domain_output.mapped_features.shape}")
        
        # Show domain knowledge base
        print("\nDomain Knowledge Base:")
        for domain, concepts in projection_wrapper.domain_knowledge_base.items():
            print(f"  • {domain}:")
            for concept, terms in concepts.items():
                print(f"    - {concept}: {terms[:3]}...")  # Show first 3 terms
        
        print("\nStrongest Associations:")
        for assoc in domain_output.knowledge_context["strongest_associations"]:
            print(f"  • {assoc}")
        
        print("\n" + "="*60)
        return domain_output
    
    def demonstrate_cross_modal_attention(self):
        """
        Demonstrate Cross-modal Understanding
        
        From document: Combining pattern recognition with domain knowledge 
        for comprehensive understanding
        
        Medical equivalent: Cross-modal attention between visual features 
        and language for medical report generation
        """
        print("3. CROSS-MODAL ATTENTION DEMONSTRATION")
        print("=" * 60)
        print("Concept: Visual ↔ Language attention for medical understanding")
        print()
        
        # Create mock attention module
        class MockBertAttention:
            def __call__(self, hidden_states, encoder_hidden_states=None, 
                        encoder_attention_mask=None, output_attentions=False):
                batch_size, seq_len, hidden_size = hidden_states.shape
                attended = torch.randn_like(hidden_states)
                attention_weights = torch.softmax(torch.randn(batch_size, seq_len, seq_len), dim=-1)
                return (attended, attention_weights) if output_attentions else (attended,)
        
        # Initialize wrapper
        attention_wrapper = SelfAttentionWrapper(
            MockBertAttention(),
            attention_type="bert"
        )
        
        # Demonstrate cross-modal attention
        print("Computing cross-modal attention between visual and text features...")
        attended_features, attention_weights, metadata = attention_wrapper.compute_cross_modal_attention(
            query_features=self.demo_data["visual_features"],
            key_features=self.demo_data["text_embeddings"],
            value_features=self.demo_data["text_embeddings"],
            attention_mask=None,
            return_attention_weights=True
        )
        
        print(f"✓ Attended Features Shape: {attended_features.shape}")
        print(f"✓ Attention Weights Shape: {attention_weights.shape}")
        
        # Show attention analysis
        pattern_analysis = metadata["pattern_analysis"]
        print(f"\nAttention Pattern Analysis:")
        print(f"  • Attention Entropy: {pattern_analysis['attention_entropy']:.3f}")
        print(f"  • Attention Sparsity: {pattern_analysis['attention_sparsity']:.3f}")
        print(f"  • Mean Attention: {pattern_analysis['attention_distribution']['mean']:.3f}")
        
        # Show cross-modal insights
        insights = metadata["cross_modal_insights"]
        print(f"\nCross-modal Insights:")
        print(f"  • Dominant Pattern: {insights['dominant_attention_pattern']}")
        print(f"  • Cross-modal Alignment: {insights['cross_modal_alignment']}")
        print(f"  • Information Flow: {insights['information_flow']}")
        
        # Show modality interaction
        if "modality_interaction" in metadata:
            interaction = metadata["modality_interaction"]
            print(f"\nModality Interaction:")
            print(f"  • Visual→Text Attention: {interaction.get('visual_to_text_attention', 0):.3f}")
            print(f"  • Text→Visual Attention: {interaction.get('text_to_visual_attention', 0):.3f}")
            print(f"  • Cross-modal Ratio: {interaction.get('cross_modal_ratio', 0):.3f}")
        
        print("\n" + "="*60)
        return attended_features, attention_weights, metadata
    
    def demonstrate_integrated_pipeline(self):
        """
        Demonstrate integrated pipeline showing how all concepts work together
        
        This shows the complete flow from Discussion Summary:
        Pattern Recognition + Domain Knowledge Association = Comprehensive Understanding
        """
        print("4. INTEGRATED PIPELINE DEMONSTRATION")
        print("=" * 60)
        print("Concept: Complete medical AI understanding pipeline")
        print()
        
        print("Step 1: Pattern Recognition")
        pattern_output = self.demonstrate_pattern_recognition()
        
        print("\nStep 2: Domain Knowledge Association")
        domain_output = self.demonstrate_domain_knowledge_association()
        
        print("\nStep 3: Cross-modal Attention")
        attended_features, attention_weights, metadata = self.demonstrate_cross_modal_attention()
        
        print("\nStep 4: Integrated Analysis")
        print("Combining all components for comprehensive medical understanding...")
        
        # Simulate integrated analysis
        integrated_confidence = (
            pattern_output.confidence * 0.4 +
            domain_output.association_strength * 0.4 +
            metadata["attention_statistics"].get("mean_attention", 0.5) * 0.2
        )
        
        print(f"✓ Integrated Confidence Score: {integrated_confidence:.3f}")
        
        # Generate medical report summary
        medical_summary = self._generate_medical_summary(
            pattern_output, domain_output, metadata
        )
        
        print(f"\nGenerated Medical Summary:")
        print(f"  {medical_summary}")
        
        print("\n" + "="*60)
        
    def _generate_medical_summary(self, pattern_output, domain_output, attention_metadata):
        """Generate a medical summary based on all wrapper outputs"""
        condition = pattern_output.metadata['recognized_condition']
        confidence = pattern_output.confidence
        associations = domain_output.knowledge_context["strongest_associations"]
        attention_pattern = attention_metadata["cross_modal_insights"]["dominant_attention_pattern"]
        
        if condition == "normal":
            return f"Normal chest radiograph with {confidence:.1%} confidence. Clear lung fields with no acute abnormalities."
        elif condition == "pneumonia":
            return f"Findings consistent with pneumonia ({confidence:.1%} confidence). Consolidation pattern identified with {attention_pattern} attention distribution."
        elif condition == "cardiomegaly":
            return f"Enlarged cardiac silhouette suggestive of cardiomegaly ({confidence:.1%} confidence). Cardiac borders show increased prominence."
        else:
            return f"Chest radiograph shows {condition} with {confidence:.1%} confidence. Clinical correlation recommended."
    
    def visualize_attention_patterns(self, attention_weights):
        """Visualize attention patterns (if matplotlib available)"""
        try:
            plt.figure(figsize=(10, 8))
            
            # Plot attention heatmap
            plt.subplot(2, 2, 1)
            sns.heatmap(attention_weights[0].detach().numpy()[:20, :20], 
                       cmap='Blues', cbar=True)
            plt.title('Attention Weights Heatmap')
            plt.xlabel('Key Positions')
            plt.ylabel('Query Positions')
            
            # Plot attention distribution
            plt.subplot(2, 2, 2)
            attention_flat = attention_weights.flatten().detach().numpy()
            plt.hist(attention_flat, bins=50, alpha=0.7, color='skyblue')
            plt.title('Attention Weight Distribution')
            plt.xlabel('Attention Weight')
            plt.ylabel('Frequency')
            
            # Plot attention entropy over positions
            plt.subplot(2, 2, 3)
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
            plt.plot(entropy[0].detach().numpy())
            plt.title('Attention Entropy by Position')
            plt.xlabel('Position')
            plt.ylabel('Entropy')
            
            # Plot max attention positions
            plt.subplot(2, 2, 4)
            max_positions = torch.argmax(attention_weights, dim=-1)
            plt.plot(max_positions[0].detach().numpy())
            plt.title('Max Attention Positions')
            plt.xlabel('Query Position')
            plt.ylabel('Key Position')
            
            plt.tight_layout()
            plt.savefig('attention_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✓ Attention visualization saved as 'attention_analysis.png'")
            
        except ImportError:
            print("⚠ Matplotlib not available for visualization")
        except Exception as e:
            print(f"⚠ Visualization error: {e}")


def main():
    """Main demonstration function"""
    print("XrayGPT Module Wrappers Demonstration")
    print("Based on Discussion_Summary.docx concepts")
    print("=" * 80)
    print()
    
    # Create demonstration instance
    demo = WrapperDemonstration()
    
    # Run individual demonstrations
    print("Running individual component demonstrations...")
    print()
    
    # 1. Pattern Recognition
    pattern_output = demo.demonstrate_pattern_recognition()
    
    # 2. Domain Knowledge Association  
    domain_output = demo.demonstrate_domain_knowledge_association()
    
    # 3. Cross-modal Attention
    attended_features, attention_weights, metadata = demo.demonstrate_cross_modal_attention()
    
    # 4. Integrated Pipeline
    demo.demonstrate_integrated_pipeline()
    
    # 5. Visualization (optional)
    print("5. ATTENTION VISUALIZATION")
    print("=" * 60)
    demo.visualize_attention_patterns(attention_weights)
    
    print("\n" + "="*80)
    print("SUMMARY OF WRAPPER IMPLEMENTATIONS")
    print("="*80)
    print()
    print("✓ MedicalVisualEncoderWrapper:")
    print("  - Implements pattern recognition for medical images")
    print("  - Associates visual patterns with medical conditions")
    print("  - Provides confidence scoring and attention analysis")
    print()
    print("✓ MultiModalProjectionWrapper:")
    print("  - Implements domain knowledge association")
    print("  - Maps visual features to clinical terminology")
    print("  - Maintains medical knowledge base for associations")
    print()
    print("✓ SelfAttentionWrapper:")
    print("  - Implements cross-modal understanding")
    print("  - Analyzes attention patterns between modalities")
    print("  - Provides insights into information flow")
    print()
    print("These wrappers implement the core concepts from Discussion_Summary.docx:")
    print("• Pattern Recognition: 'How do you' → procedural requests")
    print("• Domain Knowledge: 'load current' → electrical measurement")
    print("• Combined Understanding: Pattern + Knowledge = Comprehensive AI")
    print()
    print("In medical context:")
    print("• Visual patterns → Medical conditions")
    print("• Feature mapping → Clinical terminology") 
    print("• Cross-modal attention → Medical report generation")


if __name__ == "__main__":
    main()