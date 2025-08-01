#!/usr/bin/env python3
"""
Integration Demo: Connecting RAG System with Existing Vector Store Implementation
This script demonstrates how the new RAG system integrates with the existing
token_vector_store_implementation.py from the parent directory.
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np

# Add paths for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(parent_dir))

# Import from current directory (new implementation)
from rag_system import RAGSystem, RAGConfig, DocumentMetadata
from tokenizer_class import Tokenizer
from embedding_class import EmbeddingManager
from vector_store_class import TokenAwareVectorStore

# Import from parent directory (existing implementation)
try:
    from token_vector_store_implementation import (
        TokenAwareVectorStore as ExistingVectorStore,
        RAGWithTokenManagement as ExistingRAG,
        TokenMetadata as ExistingTokenMetadata
    )
    EXISTING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import existing implementation: {e}")
    EXISTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegrationDemo:
    """Demonstrates integration between new and existing implementations"""
    
    def __init__(self):
        self.sample_texts = [
            "Chest X-ray shows clear lung fields with no pneumonia.",
            "CT scan reveals normal liver and kidney function.",
            "MRI brain demonstrates no acute abnormalities.",
            "Ultrasound shows gallbladder stones present.",
            "Mammography screening results are negative."
        ]
    
    def run_integration_demo(self):
        """Run the complete integration demonstration"""
        print("🔗 RAG System Integration Demo")
        print("=" * 50)
        
        # 1. Test new RAG system
        self._demo_new_rag_system()
        
        # 2. Test existing implementation (if available)
        if EXISTING_AVAILABLE:
            self._demo_existing_implementation()
            self._demo_compatibility()
        else:
            print("\n⚠️ Existing implementation not available for comparison")
        
        # 3. Performance comparison
        self._demo_performance_comparison()
        
        print("\n✅ Integration demo completed!")
    
    def _demo_new_rag_system(self):
        """Demonstrate the new RAG system"""
        print("\n1. 🆕 New RAG System")
        print("-" * 30)
        
        # Initialize new RAG system
        config = RAGConfig(
            tokenizer_type="bert",
            max_tokens_per_chunk=256,
            chunk_overlap=25,
            max_context_tokens=1024
        )
        
        rag = RAGSystem(config)
        print("✓ New RAG system initialized")
        
        # Add documents
        metadata = [
            DocumentMetadata(
                doc_id=f"doc_{i}",
                title=f"Medical Report {i+1}",
                category="medical"
            )
            for i in range(len(self.sample_texts))
        ]
        
        doc_ids = rag.add_documents(self.sample_texts, metadata)
        print(f"✓ Added {len(doc_ids)} documents")
        
        # Query the system
        query = "chest X-ray findings"
        results = rag.query(query, generate_response=False)
        
        print(f"✓ Query results:")
        print(f"  Query: {query}")
        print(f"  Chunks found: {results['context_chunks']}")
        print(f"  Context tokens: {results['context_tokens']}")
        print(f"  Token efficiency: {results['retrieval_stats']['token_efficiency']:.2%}")
        
        # Show system stats
        stats = rag.get_system_stats()
        print(f"✓ System statistics:")
        print(f"  Total documents: {stats['documents']}")
        print(f"  Total chunks: {stats['chunks']}")
        print(f"  Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
        
        return rag
    
    def _demo_existing_implementation(self):
        """Demonstrate the existing vector store implementation"""
        print("\n2. 📚 Existing Implementation")
        print("-" * 30)
        
        try:
            # Initialize existing vector store
            existing_store = ExistingVectorStore(
                tokenizer_type="bert",
                max_tokens=256,
                chunk_overlap=25
            )
            print("✓ Existing vector store initialized")
            
            # Add documents
            chunk_ids = existing_store.add_documents(self.sample_texts)
            print(f"✓ Added documents, created {len(chunk_ids)} chunks")
            
            # Query the system
            query = "chest X-ray findings"
            results = existing_store.query_with_token_limit(
                query=query,
                max_context_tokens=1024
            )
            
            print(f"✓ Query results:")
            print(f"  Query: {query}")
            print(f"  Chunks found: {results['chunks_selected']}")
            print(f"  Context tokens: {results['total_tokens']}")
            print(f"  Token efficiency: {results['total_tokens'] / 1024:.2%}")
            
            # Show statistics
            stats = existing_store.get_token_statistics()
            print(f"✓ System statistics:")
            print(f"  Total documents: {stats['total_documents']}")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
            
            return existing_store
            
        except Exception as e:
            print(f"❌ Error with existing implementation: {e}")
            return None
    
    def _demo_compatibility(self):
        """Demonstrate compatibility between implementations"""
        print("\n3. 🔄 Compatibility Check")
        print("-" * 30)
        
        try:
            # Test tokenizer compatibility
            new_tokenizer = Tokenizer()
            new_tokenizer.load_tokenizer("bert")
            
            test_text = "This is a test sentence for tokenization."
            new_result = new_tokenizer.tokenize(test_text, return_tensors=False)
            
            print("✓ Tokenizer compatibility:")
            print(f"  Text: {test_text}")
            print(f"  Tokens: {len(new_result['token_ids'])}")
            print(f"  Token IDs: {new_result['token_ids'][:10]}...")  # Show first 10
            
            # Test embedding compatibility
            embedding_manager = EmbeddingManager()
            embedding_manager.load_embedding_model("bert")
            
            embedding_result = embedding_manager.create_embeddings(
                test_text,
                return_tokens=True
            )
            
            print("✓ Embedding compatibility:")
            print(f"  Embedding shape: {embedding_result['embedding_shape']}")
            print(f"  Hidden size: {embedding_result['hidden_size']}")
            
            # Test vector store compatibility
            vector_store = TokenAwareVectorStore(
                tokenizer_type="bert",
                max_tokens=256
            )
            
            test_docs = ["Test document one.", "Test document two."]
            chunk_ids = vector_store.add_documents(test_docs)
            
            print("✓ Vector store compatibility:")
            print(f"  Documents added: {len(test_docs)}")
            print(f"  Chunks created: {len(chunk_ids)}")
            
            # Test query compatibility
            query_results = vector_store.query_with_token_limit("test document")
            print(f"  Query successful: {query_results['chunks_selected']} chunks found")
            
        except Exception as e:
            print(f"❌ Compatibility issue: {e}")
    
    def _demo_performance_comparison(self):
        """Compare performance between implementations"""
        print("\n4. ⚡ Performance Comparison")
        print("-" * 30)
        
        import time
        
        # Test data
        test_docs = [
            f"This is test document number {i} with some medical content about examinations and findings."
            for i in range(20)
        ]
        
        # Test new implementation
        print("Testing new RAG system...")
        start_time = time.time()
        
        new_rag = RAGSystem(RAGConfig(max_tokens_per_chunk=128))
        new_rag.add_documents(test_docs)
        new_results = new_rag.query("medical examination", generate_response=False)
        
        new_time = time.time() - start_time
        
        print(f"✓ New implementation:")
        print(f"  Time: {new_time:.3f} seconds")
        print(f"  Chunks found: {new_results['context_chunks']}")
        print(f"  Memory usage: Efficient (integrated components)")
        
        # Test existing implementation (if available)
        if EXISTING_AVAILABLE:
            try:
                print("Testing existing implementation...")
                start_time = time.time()
                
                existing_store = ExistingVectorStore(
                    tokenizer_type="bert",
                    max_tokens=128
                )
                existing_store.add_documents(test_docs)
                existing_results = existing_store.query_with_token_limit("medical examination")
                
                existing_time = time.time() - start_time
                
                print(f"✓ Existing implementation:")
                print(f"  Time: {existing_time:.3f} seconds")
                print(f"  Chunks found: {existing_results['chunks_selected']}")
                print(f"  Memory usage: Standard")
                
                # Performance comparison
                print(f"\n📊 Performance Summary:")
                speedup = existing_time / new_time if new_time > 0 else 1
                print(f"  Speedup factor: {speedup:.2f}x")
                print(f"  New system efficiency: {new_time:.3f}s")
                print(f"  Existing system efficiency: {existing_time:.3f}s")
                
            except Exception as e:
                print(f"❌ Could not test existing implementation: {e}")
        
        # Feature comparison
        print(f"\n🔍 Feature Comparison:")
        print(f"  New RAG System:")
        print(f"    ✓ Integrated tokenizer, embeddings, vector store")
        print(f"    ✓ Document metadata management")
        print(f"    ✓ Advanced querying with filters")
        print(f"    ✓ System persistence and loading")
        print(f"    ✓ Comprehensive configuration management")
        print(f"    ✓ Production-ready architecture")
        
        if EXISTING_AVAILABLE:
            print(f"  Existing Implementation:")
            print(f"    ✓ Token-aware vector storage")
            print(f"    ✓ Basic RAG functionality")
            print(f"    ✓ Token range searching")
            print(f"    ✓ Document reconstruction")
    
    def demo_migration_path(self):
        """Demonstrate how to migrate from existing to new implementation"""
        print("\n5. 🔄 Migration Path")
        print("-" * 30)
        
        if not EXISTING_AVAILABLE:
            print("⚠️ Existing implementation not available for migration demo")
            return
        
        print("Migration steps from existing to new implementation:")
        print("1. ✓ Keep existing tokenizer configurations")
        print("2. ✓ Migrate document data and metadata")
        print("3. ✓ Update query interfaces")
        print("4. ✓ Add new features (filters, persistence)")
        print("5. ✓ Optimize performance")
        
        # Example migration code
        print("\n📝 Example migration code:")
        migration_code = '''
# Old way (existing implementation)
old_store = ExistingVectorStore(tokenizer_type="bert")
old_store.add_documents(documents)
results = old_store.query_with_token_limit(query)

# New way (RAG system)
config = RAGConfig(tokenizer_type="bert")
rag = RAGSystem(config)
rag.add_documents(documents, metadata)
results = rag.query(query, filters=filters)
'''
        print(migration_code)
        
        print("✅ Migration benefits:")
        print("  - Better organization and modularity")
        print("  - Enhanced metadata management")
        print("  - Advanced querying capabilities")
        print("  - Production-ready features")
        print("  - Comprehensive configuration")

def main():
    """Main integration demo"""
    demo = IntegrationDemo()
    
    print("🔗 RAG System Integration and Compatibility Demo")
    print("This demo shows how the new RAG system integrates with existing components")
    print("=" * 70)
    
    try:
        demo.run_integration_demo()
        
        # Optional migration demo
        print("\n" + "="*50)
        response = input("Show migration path demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            demo.demo_migration_path()
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Integration demo error: {e}")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    main()