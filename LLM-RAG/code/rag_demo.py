#!/usr/bin/env python3
"""
Comprehensive RAG System Demo
Demonstrates the complete RAG implementation with real-world examples.
Shows document ingestion, querying, and advanced features.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from rag_system import RAGSystem, RAGConfig, DocumentMetadata, create_rag_system
from tokenizer_class import Tokenizer
from embedding_class import EmbeddingManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGDemo:
    """Comprehensive demo of the RAG system"""
    
    def __init__(self):
        self.rag = None
        self.sample_documents = self._get_sample_documents()
        
    def _get_sample_documents(self):
        """Get sample documents for demonstration"""
        return {
            'medical_reports': [
                {
                    'text': """Chest X-ray Examination Report
Patient: John Doe, Age: 45
Date: 2024-01-15

FINDINGS:
The chest X-ray demonstrates clear lung fields bilaterally with no evidence of pneumonia, pleural effusion, or pneumothorax. The cardiac silhouette appears normal in size and configuration. The mediastinal contours are within normal limits. No acute bony abnormalities are identified.

IMPRESSION:
Normal chest X-ray examination. No acute cardiopulmonary abnormalities detected.""",
                    'metadata': DocumentMetadata(
                        doc_id="xray_001",
                        title="Chest X-ray Report - Normal",
                        source="Radiology Department",
                        category="chest_xray",
                        tags=["normal", "chest", "xray", "cardiopulmonary"]
                    )
                },
                {
                    'text': """CT Abdomen and Pelvis Report
Patient: Jane Smith, Age: 38
Date: 2024-01-16

TECHNIQUE:
Contrast-enhanced CT examination of the abdomen and pelvis was performed.

FINDINGS:
Liver: Normal size and attenuation. No focal lesions identified.
Gallbladder: Normal appearance without stones or wall thickening.
Pancreas: Normal size and enhancement pattern.
Kidneys: Both kidneys show normal size, shape, and enhancement. No hydronephrosis or stones.
Spleen: Normal size and attenuation.
Bowel: No evidence of obstruction or inflammatory changes.

IMPRESSION:
Normal CT examination of the abdomen and pelvis.""",
                    'metadata': DocumentMetadata(
                        doc_id="ct_001",
                        title="CT Abdomen/Pelvis - Normal",
                        source="Radiology Department",
                        category="ct_scan",
                        tags=["normal", "abdomen", "pelvis", "ct"]
                    )
                },
                {
                    'text': """MRI Brain Report
Patient: Robert Johnson, Age: 62
Date: 2024-01-17

TECHNIQUE:
MRI brain examination with and without contrast.

FINDINGS:
Brain parenchyma: No acute infarction, hemorrhage, or mass lesion identified. White matter appears appropriate for patient age with minimal chronic small vessel ischemic changes.
Ventricular system: Normal size and configuration.
Extra-axial spaces: No abnormal fluid collections.
Skull base and calvarium: Unremarkable.

IMPRESSION:
No acute intracranial abnormalities. Minimal age-related white matter changes.""",
                    'metadata': DocumentMetadata(
                        doc_id="mri_001",
                        title="MRI Brain - Age-related Changes",
                        source="Radiology Department",
                        category="mri_brain",
                        tags=["brain", "mri", "age_related", "normal"]
                    )
                },
                {
                    'text': """Mammography Screening Report
Patient: Lisa Brown, Age: 52
Date: 2024-01-18

TECHNIQUE:
Digital mammography with bilateral craniocaudal and mediolateral oblique views.

FINDINGS:
Breast composition: Heterogeneously dense breast tissue.
Right breast: No suspicious masses, calcifications, or architectural distortion.
Left breast: No suspicious masses, calcifications, or architectural distortion.
Lymph nodes: Normal appearing bilateral axillary lymph nodes.

IMPRESSION:
BI-RADS Category 1: Negative mammogram. Recommend routine annual screening.""",
                    'metadata': DocumentMetadata(
                        doc_id="mammo_001",
                        title="Mammography Screening - Negative",
                        source="Radiology Department",
                        category="mammography",
                        tags=["mammography", "screening", "negative", "birads1"]
                    )
                },
                {
                    'text': """Ultrasound Gallbladder Report
Patient: Michael Davis, Age: 41
Date: 2024-01-19

TECHNIQUE:
Real-time ultrasound examination of the gallbladder and right upper quadrant.

FINDINGS:
Gallbladder: Multiple small echogenic foci with posterior acoustic shadowing consistent with cholelithiasis. Gallbladder wall measures 2mm (normal). No pericholecystic fluid.
Common bile duct: Measures 4mm (normal).
Liver: Normal echogenicity and texture.

IMPRESSION:
Cholelithiasis (gallstones) without evidence of acute cholecystitis.""",
                    'metadata': DocumentMetadata(
                        doc_id="us_001",
                        title="Gallbladder Ultrasound - Cholelithiasis",
                        source="Radiology Department",
                        category="ultrasound",
                        tags=["ultrasound", "gallbladder", "cholelithiasis", "stones"]
                    )
                }
            ],
            'research_papers': [
                {
                    'text': """Deep Learning in Medical Imaging: A Comprehensive Review

Abstract:
Deep learning has revolutionized medical imaging analysis, providing unprecedented accuracy in disease detection and diagnosis. This review examines the current state of deep learning applications in radiology, including convolutional neural networks for image classification, object detection for lesion identification, and segmentation techniques for anatomical structure delineation.

Key findings include improved diagnostic accuracy in chest X-ray analysis, enhanced detection of diabetic retinopathy in fundus photography, and automated segmentation of brain tumors in MRI scans. Challenges remain in model interpretability, data standardization, and regulatory approval processes.""",
                    'metadata': DocumentMetadata(
                        doc_id="paper_001",
                        title="Deep Learning in Medical Imaging Review",
                        source="Journal of Medical AI",
                        category="research_paper",
                        tags=["deep_learning", "medical_imaging", "AI", "review"]
                    )
                },
                {
                    'text': """Artificial Intelligence in Radiology: Current Applications and Future Prospects

Introduction:
Artificial Intelligence (AI) is transforming radiology practice through automated image analysis, workflow optimization, and decision support systems. This paper reviews current AI applications in diagnostic imaging and discusses future developments.

Current applications include:
- Automated detection of pneumonia in chest X-rays
- Breast cancer screening in mammography
- Stroke detection in CT scans
- Fracture identification in emergency radiology

Future prospects involve integration with electronic health records, real-time image analysis during acquisition, and personalized treatment recommendations based on imaging biomarkers.""",
                    'metadata': DocumentMetadata(
                        doc_id="paper_002",
                        title="AI in Radiology Applications",
                        source="Radiology AI Journal",
                        category="research_paper",
                        tags=["AI", "radiology", "applications", "future"]
                    )
                }
            ]
        }
    
    def run_complete_demo(self):
        """Run the complete RAG system demonstration"""
        print("🚀 RAG System Comprehensive Demo")
        print("=" * 60)
        
        # 1. System initialization
        self._demo_system_initialization()
        
        # 2. Document ingestion
        self._demo_document_ingestion()
        
        # 3. Basic querying
        self._demo_basic_querying()
        
        # 4. Advanced querying features
        self._demo_advanced_querying()
        
        # 5. System analytics
        self._demo_system_analytics()
        
        # 6. Configuration management
        self._demo_configuration_management()
        
        # 7. Persistence and loading
        self._demo_persistence()
        
        print("\n✅ Demo completed successfully!")
        print("The RAG system is ready for production use.")
    
    def _demo_system_initialization(self):
        """Demonstrate system initialization"""
        print("\n1. 🔧 System Initialization")
        print("-" * 40)
        
        # Create RAG system with default configuration
        print("Creating RAG system with default configuration...")
        self.rag = RAGSystem()
        
        print(f"✓ System initialized successfully")
        print(f"  Device: {self.rag.device}")
        print(f"  Tokenizer: {self.rag.config.tokenizer_type}")
        print(f"  Max tokens per chunk: {self.rag.config.max_tokens_per_chunk}")
        print(f"  Chunk overlap: {self.rag.config.chunk_overlap}")
        
        # Show available tokenizers
        available_tokenizers = self.rag.tokenizer.list_available_tokenizers()
        print(f"  Available tokenizers: {', '.join(available_tokenizers)}")
    
    def _demo_document_ingestion(self):
        """Demonstrate document ingestion"""
        print("\n2. 📄 Document Ingestion")
        print("-" * 40)
        
        # Prepare all documents
        all_documents = []
        all_metadata = []
        
        for category, docs in self.sample_documents.items():
            print(f"\nIngesting {category}...")
            for doc in docs:
                all_documents.append(doc['text'])
                all_metadata.append(doc['metadata'])
        
        # Add documents to RAG system
        doc_ids = self.rag.add_documents(all_documents, all_metadata)
        
        print(f"\n✓ Successfully ingested {len(doc_ids)} documents")
        
        # Show ingestion statistics
        stats = self.rag.get_system_stats()
        print(f"  Total chunks created: {stats['chunks']}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
    
    def _demo_basic_querying(self):
        """Demonstrate basic querying functionality"""
        print("\n3. 🔍 Basic Querying")
        print("-" * 40)
        
        queries = [
            "chest X-ray findings",
            "brain MRI abnormalities", 
            "gallbladder stones",
            "mammography screening results",
            "deep learning medical imaging"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            print("-" * 30)
            
            results = self.rag.query(query, generate_response=False)
            
            print(f"📊 Results:")
            print(f"  Relevant chunks: {results['context_chunks']}")
            print(f"  Context tokens: {results['context_tokens']}")
            print(f"  Token efficiency: {results['retrieval_stats']['token_efficiency']:.1%}")
            
            if results['sources']:
                print(f"  Top sources:")
                for i, source in enumerate(results['sources'][:3], 1):
                    similarity = source['similarity']
                    title = source.get('title', 'Unknown')
                    print(f"    {i}. {title} (similarity: {similarity:.3f})")
            
            # Show a snippet of the context
            context_snippet = results['context'][:200] + "..." if len(results['context']) > 200 else results['context']
            print(f"  Context preview: {context_snippet}")
    
    def _demo_advanced_querying(self):
        """Demonstrate advanced querying features"""
        print("\n4. 🎯 Advanced Querying Features")
        print("-" * 40)
        
        # 1. Query with filters
        print("\n4.1 Filtered Search")
        print("Searching for 'normal findings' in radiology reports only...")
        
        results = self.rag.query(
            "normal findings",
            filters={'category': ['chest_xray', 'ct_scan', 'mri_brain']},
            generate_response=False
        )
        
        print(f"  Found {results['context_chunks']} chunks in radiology reports")
        
        # 2. Token range search
        print("\n4.2 Token Range Search")
        print("Searching for documents with 100-300 tokens...")
        
        token_results = self.rag.vector_store.search_by_token_range(
            "medical examination",
            min_tokens=100,
            max_tokens=300,
            n_results=3
        )
        
        print(f"  Found {len(token_results['chunks'])} chunks in token range")
        for chunk in token_results['chunks']:
            print(f"    - {chunk['chunk_id']}: {chunk['token_count']} tokens")
        
        # 3. Similarity threshold adjustment
        print("\n4.3 Similarity Threshold Adjustment")
        
        for threshold in [0.5, 0.7, 0.9]:
            results = self.rag.query(
                "chest examination",
                similarity_threshold=threshold,
                generate_response=False
            )
            print(f"  Threshold {threshold}: {results['context_chunks']} chunks")
    
    def _demo_system_analytics(self):
        """Demonstrate system analytics and statistics"""
        print("\n5. 📈 System Analytics")
        print("-" * 40)
        
        # Overall system statistics
        stats = self.rag.get_system_stats()
        
        print("📊 System Overview:")
        print(f"  Documents: {stats['documents']}")
        print(f"  Chunks: {stats['chunks']}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
        print(f"  Device: {stats['device']}")
        
        # Vector store statistics
        vector_stats = stats['vector_store_stats']
        print(f"\n📈 Token Distribution:")
        print(f"  Min tokens: {vector_stats['min_tokens']}")
        print(f"  Max tokens: {vector_stats['max_tokens']}")
        print(f"  25th percentile: {vector_stats['token_distribution']['p25']:.1f}")
        print(f"  50th percentile: {vector_stats['token_distribution']['p50']:.1f}")
        print(f"  75th percentile: {vector_stats['token_distribution']['p75']:.1f}")
        print(f"  95th percentile: {vector_stats['token_distribution']['p95']:.1f}")
        
        # Document-level analytics
        print(f"\n📋 Document Analysis:")
        for doc_id in list(self.rag.documents.keys())[:3]:  # Show first 3
            doc_info = self.rag.get_document_info(doc_id)
            print(f"  {doc_id}:")
            print(f"    Chunks: {doc_info['chunks']}")
            print(f"    Total tokens: {doc_info['total_tokens']}")
            print(f"    Category: {doc_info['metadata'].get('category', 'Unknown')}")
    
    def _demo_configuration_management(self):
        """Demonstrate configuration management"""
        print("\n6. ⚙️ Configuration Management")
        print("-" * 40)
        
        # Show current configuration
        print("Current Configuration:")
        config_dict = self.rag.config.to_dict()
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
        
        # Demonstrate loading from config file
        print(f"\n📁 Configuration File Support:")
        config_file = Path("rag_config.yaml")
        if config_file.exists():
            print(f"  ✓ Configuration file found: {config_file}")
            print(f"  You can create RAG systems with: create_rag_system('rag_config.yaml')")
        else:
            print(f"  ⚠️ Configuration file not found: {config_file}")
        
        # Show different configuration options
        print(f"\n🔧 Available Configurations:")
        print(f"  - Default: General purpose RAG")
        print(f"  - Medical: Optimized for medical documents")
        print(f"  - Large docs: For processing large documents")
        print(f"  - Lightweight: Fast, minimal resource usage")
    
    def _demo_persistence(self):
        """Demonstrate system persistence and loading"""
        print("\n7. 💾 Persistence and Loading")
        print("-" * 40)
        
        # Save the current system
        print("Saving RAG system...")
        save_path = self.rag.save_system()
        print(f"✓ System saved to: {save_path}")
        
        # Show what was saved
        save_file = Path(save_path)
        if save_file.exists():
            size_mb = save_file.stat().st_size / (1024 * 1024)
            print(f"  File size: {size_mb:.2f} MB")
        
        # Demonstrate loading (create new system and load)
        print(f"\nDemonstrating system loading...")
        print(f"  Creating new RAG system...")
        new_rag = RAGSystem()
        
        print(f"  Loading saved data...")
        new_rag.load_system(save_path)
        
        # Verify loaded system
        new_stats = new_rag.get_system_stats()
        print(f"✓ System loaded successfully")
        print(f"  Loaded documents: {new_stats['documents']}")
        print(f"  Loaded chunks: {new_stats['chunks']}")
        
        # Test query on loaded system
        test_results = new_rag.query("chest X-ray", generate_response=False)
        print(f"  Test query successful: {test_results['context_chunks']} chunks found")
    
    def run_interactive_demo(self):
        """Run an interactive demo where users can ask questions"""
        print("\n🎮 Interactive RAG Demo")
        print("=" * 40)
        print("Ask questions about the medical documents!")
        print("Type 'quit' to exit, 'stats' for system statistics")
        print("-" * 40)
        
        if self.rag is None:
            print("Initializing RAG system...")
            self.rag = RAGSystem()
            
            # Add sample documents
            all_documents = []
            all_metadata = []
            for category, docs in self.sample_documents.items():
                for doc in docs:
                    all_documents.append(doc['text'])
                    all_metadata.append(doc['metadata'])
            
            self.rag.add_documents(all_documents, all_metadata)
            print("✓ System ready!")
        
        while True:
            try:
                query = input("\n🔍 Your question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if query.lower() == 'stats':
                    stats = self.rag.get_system_stats()
                    print(f"📊 System Stats: {stats['documents']} docs, {stats['chunks']} chunks, {stats['total_tokens']:,} tokens")
                    continue
                
                if not query:
                    continue
                
                print("🔍 Searching...")
                results = self.rag.query(query, generate_response=False)
                
                print(f"\n📊 Found {results['context_chunks']} relevant chunks:")
                
                if results['sources']:
                    for i, source in enumerate(results['sources'][:3], 1):
                        print(f"{i}. {source.get('title', 'Unknown')} (similarity: {source['similarity']:.3f})")
                
                # Show context preview
                if results['context']:
                    context_preview = results['context'][:300] + "..." if len(results['context']) > 300 else results['context']
                    print(f"\n📄 Context preview:\n{context_preview}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main demo function"""
    print("🏥 Medical RAG System Demo")
    print("Comprehensive demonstration of RAG capabilities")
    print("=" * 60)
    
    demo = RAGDemo()
    
    # Check if user wants interactive or complete demo
    print("\nDemo Options:")
    print("1. Complete Demo (automated)")
    print("2. Interactive Demo (ask questions)")
    print("3. Both")
    
    try:
        choice = input("\nSelect option (1-3) or press Enter for complete demo: ").strip()
        
        if choice == "2":
            demo.run_interactive_demo()
        elif choice == "3":
            demo.run_complete_demo()
            print("\n" + "="*60)
            demo.run_interactive_demo()
        else:
            demo.run_complete_demo()
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()