"""
Universal Multimodal Framework (UMF) - Interactive Demo
Comprehensive demonstration of multimodal AI across all domains
"""

import torch
import gradio as gr
import numpy as np
from PIL import Image
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

from umf_universal_implementation import (
    UniversalMultimodalFramework, DomainType, ModalityType, 
    MultimodalInput, ConversationManager
)

# ============================================================================
# Demo Configuration and Setup
# ============================================================================

class UMFDemo:
    """Interactive demo for Universal Multimodal Framework"""
    
    def __init__(self, config_path: str = "umf_config.yaml"):
        self.config = self._load_config(config_path)
        self.framework = UniversalMultimodalFramework(config_path)
        self.conversation_manager = ConversationManager()
        
        # Demo data and examples
        self.demo_examples = self._load_demo_examples()
        self.conversation_history = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🚀 Universal Multimodal Framework Demo initialized!")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for demo"""
        return {
            "generation": {
                "max_new_tokens": 300,
                "temperature": 1.0,
                "top_p": 0.9
            }
        }
    
    def _load_demo_examples(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load demo examples for each domain"""
        return {
            "medical": [
                {
                    "title": "Chest X-ray Analysis",
                    "description": "Analyze chest X-ray for abnormalities",
                    "image": "demo_chest_xray.jpg",
                    "prompt": "Analyze this chest X-ray and describe any findings."
                },
                {
                    "title": "CT Scan Review",
                    "description": "Review CT scan for diagnostic insights",
                    "image": "demo_ct_scan.jpg", 
                    "prompt": "What do you observe in this CT scan?"
                }
            ],
            "autonomous": [
                {
                    "title": "Traffic Scene Analysis",
                    "description": "Analyze traffic situation for driving decisions",
                    "image": "demo_traffic_scene.jpg",
                    "prompt": "What should the autonomous vehicle do in this situation?"
                },
                {
                    "title": "Parking Scenario",
                    "description": "Navigate complex parking situation",
                    "image": "demo_parking.jpg",
                    "prompt": "How should the vehicle approach this parking scenario?"
                }
            ],
            "robotics": [
                {
                    "title": "Object Manipulation",
                    "description": "Plan object grasping and manipulation",
                    "image": "demo_robot_scene.jpg",
                    "prompt": "How should the robot pick up and move these objects?"
                },
                {
                    "title": "Navigation Task",
                    "description": "Navigate through cluttered environment",
                    "image": "demo_robot_navigation.jpg",
                    "prompt": "Plan a safe path through this environment."
                }
            ],
            "education": [
                {
                    "title": "Math Problem Solving",
                    "description": "Explain mathematical concepts step by step",
                    "image": "demo_math_problem.jpg",
                    "prompt": "Explain how to solve this geometry problem."
                },
                {
                    "title": "Science Diagram",
                    "description": "Explain scientific concepts from diagrams",
                    "image": "demo_science_diagram.jpg",
                    "prompt": "Explain the process shown in this diagram."
                }
            ],
            "general": [
                {
                    "title": "Image Description",
                    "description": "Describe and analyze general images",
                    "image": "demo_general_image.jpg",
                    "prompt": "Describe what you see in this image."
                },
                {
                    "title": "Visual Question Answering",
                    "description": "Answer questions about visual content",
                    "image": "demo_vqa_image.jpg",
                    "prompt": "What activities are happening in this scene?"
                }
            ]
        }

# ============================================================================
# Core Demo Functions
# ============================================================================

def process_multimodal_input(
    domain: str,
    image: Optional[Image.Image],
    audio: Optional[np.ndarray],
    text: str,
    conversation_history: List[str],
    demo_instance: UMFDemo
) -> Tuple[str, List[str]]:
    """Process multimodal input and generate response"""
    
    try:
        # Convert domain string to enum
        domain_enum = DomainType(domain.lower())
        
        # Create multimodal input
        multimodal_input = MultimodalInput()
        
        if image is not None:
            multimodal_input.vision = image
        
        if audio is not None:
            multimodal_input.audio = torch.tensor(audio, dtype=torch.float32)
        
        if text:
            multimodal_input.text = text
        
        # Build conversation context
        context = " ".join(conversation_history[-6:]) if conversation_history else None
        
        # Generate response
        response = demo_instance.framework.chat(
            inputs=multimodal_input,
            domain=domain_enum,
            user_query=text,
            conversation_history=conversation_history
        )
        
        # Update conversation history
        updated_history = conversation_history + [f"User: {text}", f"Assistant: {response}"]
        
        return response, updated_history
        
    except Exception as e:
        error_msg = f"Error processing input: {str(e)}"
        demo_instance.logger.error(error_msg)
        return error_msg, conversation_history

def reset_conversation() -> Tuple[str, List[str]]:
    """Reset conversation history"""
    return "", []

def load_example(domain: str, example_idx: int, demo_instance: UMFDemo) -> Tuple[Optional[Image.Image], str]:
    """Load example data for demonstration"""
    try:
        examples = demo_instance.demo_examples.get(domain, [])
        if 0 <= example_idx < len(examples):
            example = examples[example_idx]
            
            # In a real implementation, you would load the actual image
            # For demo purposes, create a placeholder image
            placeholder_image = Image.new('RGB', (224, 224), color='lightgray')
            
            return placeholder_image, example["prompt"]
        else:
            return None, ""
    except Exception as e:
        demo_instance.logger.error(f"Error loading example: {str(e)}")
        return None, ""

# ============================================================================
# Gradio Interface Components
# ============================================================================

def create_domain_interface(demo_instance: UMFDemo) -> gr.Interface:
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="Universal Multimodal Framework Demo") as interface:
        
        # Header
        gr.Markdown("""
        # 🚀 Universal Multimodal Framework (UMF) Demo
        
        **Experience multimodal AI across all domains!**
        
        This framework demonstrates how a single architecture can handle:
        - 🏥 **Medical**: Analyze X-rays, CT scans, medical images
        - 🚗 **Autonomous**: Traffic analysis, driving decisions
        - 🤖 **Robotics**: Task planning, object manipulation
        - 📚 **Education**: Concept explanation, problem solving
        - 🌐 **General**: Image description, visual Q&A
        """)
        
        # Domain Selection
        with gr.Row():
            domain_selector = gr.Dropdown(
                choices=["medical", "autonomous", "robotics", "education", "general"],
                value="medical",
                label="Select Domain",
                info="Choose the domain for specialized AI assistance"
            )
        
        # Main Interface
        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Input")
                
                # Image Input
                image_input = gr.Image(
                    type="pil",
                    label="Upload Image",
                    info="Upload an image relevant to your selected domain"
                )
                
                # Audio Input (for future expansion)
                audio_input = gr.Audio(
                    type="numpy",
                    label="Upload Audio (Optional)",
                    visible=False  # Hidden for now
                )
                
                # Text Input
                text_input = gr.Textbox(
                    label="Your Question/Query",
                    placeholder="Ask a question about the uploaded content...",
                    lines=3
                )
                
                # Action Buttons
                with gr.Row():
                    submit_btn = gr.Button("🚀 Process", variant="primary")
                    reset_btn = gr.Button("🔄 Reset", variant="secondary")
            
            # Output Column
            with gr.Column(scale=1):
                gr.Markdown("### 📤 AI Response")
                
                response_output = gr.Textbox(
                    label="AI Response",
                    lines=10,
                    interactive=False
                )
                
                # Conversation History
                conversation_state = gr.State([])
                
                gr.Markdown("### 💬 Conversation History")
                conversation_display = gr.Textbox(
                    label="Previous Exchanges",
                    lines=5,
                    interactive=False,
                    visible=True
                )
        
        # Examples Section
        gr.Markdown("### 📋 Try These Examples")
        
        with gr.Row():
            example_selector = gr.Dropdown(
                choices=[],
                label="Select Example",
                info="Choose a pre-loaded example for the selected domain"
            )
            load_example_btn = gr.Button("📂 Load Example")
        
        # Domain-Specific Information
        domain_info = gr.Markdown("", visible=True)
        
        # Event Handlers
        def update_examples_and_info(domain):
            """Update examples and domain information"""
            examples = demo_instance.demo_examples.get(domain, [])
            example_choices = [f"{i}: {ex['title']}" for i, ex in enumerate(examples)]
            
            # Domain-specific information
            domain_descriptions = {
                "medical": "🏥 **Medical Domain**: Analyze medical images, provide diagnostic insights, explain medical conditions. *Always consult healthcare professionals for medical decisions.*",
                "autonomous": "🚗 **Autonomous Domain**: Analyze traffic scenes, make driving decisions, plan routes. *Prioritizes safety and traffic compliance.*",
                "robotics": "🤖 **Robotics Domain**: Plan manipulation tasks, navigate environments, coordinate robot actions. *Focuses on safe and efficient task execution.*",
                "education": "📚 **Education Domain**: Explain concepts, solve problems, provide learning guidance. *Adapts explanations to learning context.*",
                "general": "🌐 **General Domain**: Describe images, answer visual questions, provide general assistance. *Versatile multimodal understanding.*"
            }
            
            return (
                gr.Dropdown.update(choices=example_choices, value=None),
                domain_descriptions.get(domain, "")
            )
        
        def load_selected_example(domain, example_selection):
            """Load the selected example"""
            if example_selection:
                try:
                    example_idx = int(example_selection.split(":")[0])
                    return load_example(domain, example_idx, demo_instance)
                except:
                    return None, ""
            return None, ""
        
        def process_and_update(domain, image, audio, text, conv_history):
            """Process input and update conversation"""
            response, updated_history = process_multimodal_input(
                domain, image, audio, text, conv_history, demo_instance
            )
            
            # Format conversation history for display
            history_text = "\n".join(updated_history[-6:]) if updated_history else ""
            
            return response, updated_history, history_text
        
        def reset_all():
            """Reset all inputs and conversation"""
            return "", [], "", None, None
        
        # Connect event handlers
        domain_selector.change(
            update_examples_and_info,
            inputs=[domain_selector],
            outputs=[example_selector, domain_info]
        )
        
        load_example_btn.click(
            load_selected_example,
            inputs=[domain_selector, example_selector],
            outputs=[image_input, text_input]
        )
        
        submit_btn.click(
            process_and_update,
            inputs=[domain_selector, image_input, audio_input, text_input, conversation_state],
            outputs=[response_output, conversation_state, conversation_display]
        )
        
        reset_btn.click(
            reset_all,
            outputs=[text_input, conversation_state, conversation_display, image_input, audio_input]
        )
        
        # Initialize with medical domain
        interface.load(
            update_examples_and_info,
            inputs=[gr.State("medical")],
            outputs=[example_selector, domain_info]
        )
    
    return interface

# ============================================================================
# Command Line Interface
# ============================================================================

def cli_demo(demo_instance: UMFDemo):
    """Command line interface for the demo"""
    print("🚀 Universal Multimodal Framework - CLI Demo")
    print("=" * 50)
    
    while True:
        print("\nAvailable domains:")
        for i, domain in enumerate(DomainType, 1):
            print(f"{i}. {domain.value.title()}")
        
        try:
            choice = input("\nSelect domain (1-5) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                break
            
            domain_idx = int(choice) - 1
            domain = list(DomainType)[domain_idx]
            
            print(f"\n🎯 Selected domain: {domain.value.title()}")
            
            # Get user input
            text_query = input("Enter your question: ").strip()
            
            if not text_query:
                print("Please enter a valid question.")
                continue
            
            # Create multimodal input (text only for CLI)
            multimodal_input = MultimodalInput(text=text_query)
            
            # Process and get response
            print("\n🤔 Processing...")
            response = demo_instance.framework.chat(
                inputs=multimodal_input,
                domain=domain,
                user_query=text_query
            )
            
            print(f"\n🤖 AI Response:\n{response}")
            print("-" * 50)
            
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

# ============================================================================
# Main Demo Application
# ============================================================================

def main():
    """Main demo application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal Multimodal Framework Demo")
    parser.add_argument("--config", default="umf_config.yaml", help="Configuration file path")
    parser.add_argument("--interface", choices=["gradio", "cli"], default="gradio", 
                       help="Interface type")
    parser.add_argument("--host", default="127.0.0.1", help="Host for Gradio interface")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio interface")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    
    args = parser.parse_args()
    
    # Initialize demo
    print("🚀 Initializing Universal Multimodal Framework...")
    demo_instance = UMFDemo(args.config)
    
    if args.interface == "gradio":
        print("🌐 Starting Gradio interface...")
        interface = create_domain_interface(demo_instance)
        
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True
        )
    
    elif args.interface == "cli":
        cli_demo(demo_instance)

# ============================================================================
# Utility Functions for Testing
# ============================================================================

def test_all_domains():
    """Test all domains with sample inputs"""
    print("🧪 Testing all domains...")
    
    demo = UMFDemo()
    
    test_cases = {
        DomainType.MEDICAL: "Analyze this medical image for any abnormalities.",
        DomainType.AUTONOMOUS: "What should the vehicle do in this traffic situation?",
        DomainType.ROBOTICS: "How should the robot approach this manipulation task?",
        DomainType.EDUCATION: "Explain this concept in simple terms.",
        DomainType.GENERAL: "Describe what you see in this image."
    }
    
    for domain, query in test_cases.items():
        print(f"\n🎯 Testing {domain.value.title()} domain...")
        
        multimodal_input = MultimodalInput(text=query)
        
        try:
            response = demo.framework.chat(
                inputs=multimodal_input,
                domain=domain,
                user_query=query
            )
            print(f"✅ Response: {response[:100]}...")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()