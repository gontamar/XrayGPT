#!/usr/bin/env python3
"""
Complete RAG (Retrieval Augmented Generation) System
Integrates tokenizer, embeddings, and vector store for production-ready RAG applications.
Based on the analysis from rag_blog_analysis.md and token_vector_store_implementation.py
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
import pickle
import uuid
from datetime import datetime
import yaml
from collections import defaultdict

# Import existing components
from tokenizer_class import Tokenizer
from embedding_class import EmbeddingManager
from vector_store_class import TokenAwareVectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    tokenizer_type: str = "bert"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_tokens_per_chunk: int = 512
    chunk_overlap: int = 50
    max_context_tokens: int = 2048
    top_k_retrieval: int = 10
    similarity_threshold: float = 0.7
    storage_path: str = "./rag_data"
    device: str = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGConfig':
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RAGConfig':
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get('rag_config', data))

@dataclass
class DocumentMetadata:
    """Metadata for documents in RAG system"""
    doc_id: str
    title: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_fields: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DocumentProcessor:
    """Handles document preprocessing and chunking for RAG"""
    
    def __init__(self, tokenizer: Tokenizer, max_tokens: int = 512, overlap: int = 50):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap = overlap
        
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Basic cleaning (can be extended)
        text = text.strip()
        
        return text
    
    def chunk_document(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Chunk document into smaller pieces with overlap.
        
        Args:
            text: Document text
            doc_id: Document identifier
            
        Returns:
            List of chunk dictionaries
        """
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Tokenize full document
        tokenization_result = self.tokenizer.tokenize(text, return_tensors=False)
        token_ids = tokenization_result['token_ids']
        tokens = tokenization_result['tokens']
        
        chunks = []
        
        if len(token_ids) <= self.max_tokens:
            # Document fits in one chunk
            chunks.append({
                'chunk_id': f"{doc_id}_chunk_0",
                'text': text,
                'token_ids': token_ids,
                'tokens': tokens,
                'chunk_index': 0,
                'total_chunks': 1,
                'doc_id': doc_id,
                'token_count': len(token_ids)
            })
        else:
            # Split into multiple chunks
            start = 0
            chunk_index = 0
            
            while start < len(token_ids):
                end = min(start + self.max_tokens, len(token_ids))
                chunk_token_ids = token_ids[start:end]
                
                # Decode chunk back to text
                chunk_text = self.tokenizer.decode(chunk_token_ids)
                
                chunks.append({
                    'chunk_id': f"{doc_id}_chunk_{chunk_index}",
                    'text': chunk_text,
                    'token_ids': chunk_token_ids,
                    'tokens': tokens[start:end] if start < len(tokens) else [],
                    'chunk_index': chunk_index,
                    'total_chunks': -1,  # Will be updated
                    'doc_id': doc_id,
                    'token_count': len(chunk_token_ids),
                    'start_token': start,
                    'end_token': end
                })
                
                # Move start position with overlap
                if end == len(token_ids):
                    break
                start = end - self.overlap
                chunk_index += 1
            
            # Update total chunks count
            for chunk in chunks:
                chunk['total_chunks'] = len(chunks)
        
        return chunks

class RAGRetriever:
    """Handles retrieval of relevant documents for queries"""
    
    def __init__(self, vector_store: TokenAwareVectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
    def retrieve(self, 
                query: str, 
                max_context_tokens: int = 2048,
                top_k: int = 10,
                similarity_threshold: float = 0.0,
                filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            max_context_tokens: Maximum tokens for context
            top_k: Number of top results to consider
            similarity_threshold: Minimum similarity score
            filters: Optional filters for metadata
            
        Returns:
            Retrieval results with context and metadata
        """
        # Use vector store's query method
        results = self.vector_store.query_with_token_limit(
            query=query,
            max_context_tokens=max_context_tokens,
            n_results=top_k
        )
        
        if 'error' in results:
            return results
        
        # Apply similarity threshold
        filtered_chunks = []
        for chunk in results['chunks']:
            if chunk['similarity'] >= similarity_threshold:
                filtered_chunks.append(chunk)
        
        # Apply metadata filters if provided
        if filters:
            filtered_chunks = self._apply_filters(filtered_chunks, filters)
        
        # Update results
        results['chunks'] = filtered_chunks
        results['chunks_selected'] = len(filtered_chunks)
        
        # Recalculate total tokens
        total_tokens = sum(
            self.vector_store.metadata[chunk['chunk_id']].token_count 
            for chunk in filtered_chunks
        )
        results['total_tokens'] = total_tokens
        
        return results
    
    def _apply_filters(self, chunks: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply metadata filters to chunks"""
        filtered = []
        
        for chunk in chunks:
            chunk_metadata = self.vector_store.chunks[chunk['chunk_id']]
            
            # Check each filter
            include_chunk = True
            for filter_key, filter_value in filters.items():
                if filter_key in chunk_metadata:
                    if isinstance(filter_value, list):
                        if chunk_metadata[filter_key] not in filter_value:
                            include_chunk = False
                            break
                    else:
                        if chunk_metadata[filter_key] != filter_value:
                            include_chunk = False
                            break
            
            if include_chunk:
                filtered.append(chunk)
        
        return filtered

class RAGGenerator:
    """Handles response generation (placeholder for LLM integration)"""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.model = None
        
    def generate_response(self, 
                         query: str, 
                         context: str, 
                         max_length: int = 512,
                         temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate response using retrieved context.
        
        Args:
            query: Original query
            context: Retrieved context
            max_length: Maximum response length
            temperature: Generation temperature
            
        Returns:
            Generated response with metadata
        """
        # Placeholder implementation
        # In production, this would use a proper LLM
        
        response = f"""Based on the provided context, here's a response to your query: "{query}"

Context Summary:
{context[:500]}...

This is a placeholder response. In a production system, this would be generated by a language model like GPT, LLaMA, or similar."""
        
        return {
            'query': query,
            'response': response,
            'context_length': len(context),
            'response_length': len(response),
            'model': self.model_name or "placeholder",
            'temperature': temperature
        }

class RAGSystem:
    """
    Complete RAG system integrating all components.
    Based on production best practices from the blog analysis.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG system.
        
        Args:
            config: RAG configuration
        """
        self.config = config or RAGConfig()
        
        # Determine device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        # Initialize components
        self._initialize_components()
        
        # Storage
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Document tracking
        self.documents = {}
        self.document_metadata = {}
        
        logger.info(f"RAG System initialized with device: {self.device}")
        logger.info(f"Configuration: {self.config.to_dict()}")
    
    def _initialize_components(self):
        """Initialize all RAG components"""
        try:
            # Initialize tokenizer
            self.tokenizer = Tokenizer()
            self.tokenizer.load_tokenizer(self.config.tokenizer_type)
            
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager()
            self.embedding_manager.load_embedding_model(
                self.config.tokenizer_type, 
                device=self.device
            )
            
            # Initialize vector store
            self.vector_store = TokenAwareVectorStore(
                tokenizer_type=self.config.tokenizer_type,
                embedding_model=self.config.embedding_model,
                max_tokens=self.config.max_tokens_per_chunk,
                chunk_overlap=self.config.chunk_overlap,
                storage_path=str(self.storage_path / "vector_store")
            )
            
            # Initialize document processor
            self.document_processor = DocumentProcessor(
                self.tokenizer,
                self.config.max_tokens_per_chunk,
                self.config.chunk_overlap
            )
            
            # Initialize retriever
            self.retriever = RAGRetriever(self.vector_store, self.embedding_manager)
            
            # Initialize generator
            self.generator = RAGGenerator()
            
            logger.info("All RAG components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG components: {e}")
            raise
    
    def add_documents(self, 
                     documents: List[str], 
                     metadata: Optional[List[DocumentMetadata]] = None,
                     doc_ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of document texts
            metadata: Optional document metadata
            doc_ids: Optional document IDs
            
        Returns:
            List of document IDs added
        """
        if doc_ids is None:
            doc_ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in documents]
        
        if metadata is None:
            metadata = [DocumentMetadata(doc_id=doc_id) for doc_id in doc_ids]
        
        all_chunks = []
        
        for doc_text, doc_meta, doc_id in zip(documents, metadata, doc_ids):
            # Store original document
            self.documents[doc_id] = doc_text
            self.document_metadata[doc_id] = doc_meta
            
            # Process document into chunks
            chunks = self.document_processor.chunk_document(doc_text, doc_id)
            
            # Prepare for vector store
            chunk_texts = [chunk['text'] for chunk in chunks]
            chunk_metadata = [
                {
                    'chunk_index': chunk['chunk_index'],
                    'total_chunks': chunk['total_chunks'],
                    'token_count': chunk['token_count'],
                    **doc_meta.to_dict()
                }
                for chunk in chunks
            ]
            
            # Add to vector store
            chunk_ids = self.vector_store.add_documents(
                documents=chunk_texts,
                doc_ids=[chunk['chunk_id'] for chunk in chunks],
                metadata=chunk_metadata
            )
            
            all_chunks.extend(chunk_ids)
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks")
        
        logger.info(f"Added {len(documents)} documents with {len(all_chunks)} total chunks")
        return doc_ids
    
    def query(self, 
             query: str, 
             max_context_tokens: Optional[int] = None,
             top_k: Optional[int] = None,
             similarity_threshold: Optional[float] = None,
             include_sources: bool = True,
             generate_response: bool = True,
             filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query: User query
            max_context_tokens: Maximum context tokens (uses config default if None)
            top_k: Number of top results (uses config default if None)
            similarity_threshold: Similarity threshold (uses config default if None)
            include_sources: Whether to include source information
            generate_response: Whether to generate a response
            filters: Optional metadata filters
            
        Returns:
            Complete query results with context, sources, and optional response
        """
        # Use config defaults if not specified
        max_context_tokens = max_context_tokens or self.config.max_context_tokens
        top_k = top_k or self.config.top_k_retrieval
        similarity_threshold = similarity_threshold or self.config.similarity_threshold
        
        # Retrieve relevant documents
        retrieval_results = self.retriever.retrieve(
            query=query,
            max_context_tokens=max_context_tokens,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters
        )
        
        if 'error' in retrieval_results:
            return retrieval_results
        
        # Prepare context
        context_chunks = retrieval_results['chunks']
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Prepare sources
        sources = []
        if include_sources:
            for chunk in context_chunks:
                doc_id = chunk['doc_id']
                doc_metadata = self.document_metadata.get(doc_id, {})
                
                sources.append({
                    'doc_id': doc_id,
                    'chunk_id': chunk['chunk_id'],
                    'similarity': chunk['similarity'],
                    'title': getattr(doc_metadata, 'title', None) if hasattr(doc_metadata, 'title') else None,
                    'source': getattr(doc_metadata, 'source', None) if hasattr(doc_metadata, 'source') else None,
                    'category': getattr(doc_metadata, 'category', None) if hasattr(doc_metadata, 'category') else None
                })
        
        # Generate response if requested
        response_data = None
        if generate_response and context:
            response_data = self.generator.generate_response(query, context)
        
        # Compile results
        results = {
            'query': query,
            'context': context,
            'context_chunks': len(context_chunks),
            'context_tokens': retrieval_results['total_tokens'],
            'sources': sources,
            'retrieval_stats': {
                'chunks_available': retrieval_results['chunks_available'],
                'chunks_selected': retrieval_results['chunks_selected'],
                'similarity_threshold': similarity_threshold,
                'token_efficiency': retrieval_results['total_tokens'] / max_context_tokens
            }
        }
        
        if response_data:
            results['response'] = response_data
        
        return results
    
    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """Get information about a specific document"""
        if doc_id not in self.documents:
            return {'error': f'Document {doc_id} not found'}
        
        chunks = self.vector_store.get_document_chunks(doc_id)
        metadata = self.document_metadata.get(doc_id, {})
        
        return {
            'doc_id': doc_id,
            'text': self.documents[doc_id],
            'metadata': metadata.to_dict() if hasattr(metadata, 'to_dict') else metadata,
            'chunks': len(chunks),
            'total_tokens': sum(chunk['token_count'] for chunk in chunks),
            'chunk_details': chunks
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        vector_stats = self.vector_store.get_token_statistics()
        
        return {
            'documents': len(self.documents),
            'chunks': vector_stats.get('total_chunks', 0),
            'total_tokens': vector_stats.get('total_tokens', 0),
            'avg_tokens_per_chunk': vector_stats.get('avg_tokens_per_chunk', 0),
            'config': self.config.to_dict(),
            'device': self.device,
            'vector_store_stats': vector_stats
        }
    
    def save_system(self, filename: Optional[str] = None) -> str:
        """Save the entire RAG system to disk"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_system_{timestamp}.pkl"
        
        filepath = self.storage_path / filename
        
        # Save vector store separately
        vector_store_path = self.vector_store.save_to_disk()
        
        # Save system data
        system_data = {
            'config': self.config.to_dict(),
            'documents': self.documents,
            'document_metadata': {
                k: v.to_dict() if hasattr(v, 'to_dict') else v 
                for k, v in self.document_metadata.items()
            },
            'vector_store_path': vector_store_path,
            'device': self.device,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(system_data, f)
        
        logger.info(f"RAG system saved to {filepath}")
        return str(filepath)
    
    def load_system(self, filepath: str) -> None:
        """Load RAG system from disk"""
        with open(filepath, 'rb') as f:
            system_data = pickle.load(f)
        
        # Load configuration
        self.config = RAGConfig.from_dict(system_data['config'])
        
        # Load documents and metadata
        self.documents = system_data['documents']
        self.document_metadata = {
            k: DocumentMetadata.from_dict(v) if isinstance(v, dict) else v
            for k, v in system_data['document_metadata'].items()
        }
        
        # Reinitialize components with loaded config
        self._initialize_components()
        
        # Load vector store
        if 'vector_store_path' in system_data:
            self.vector_store.load_from_disk(system_data['vector_store_path'])
        
        logger.info(f"RAG system loaded from {filepath}")
        logger.info(f"Loaded {len(self.documents)} documents")

def create_rag_system(config_path: Optional[str] = None) -> RAGSystem:
    """
    Create a RAG system with optional configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Initialized RAG system
    """
    if config_path and Path(config_path).exists():
        config = RAGConfig.from_yaml(config_path)
    else:
        config = RAGConfig()
    
    return RAGSystem(config)

if __name__ == "__main__":
    # Example usage
    print("RAG System for Tokenization Repository")
    print("=" * 50)
    
    # Create RAG system
    rag = RAGSystem()
    
    # Sample documents (medical/radiology theme)
    documents = [
        "Chest X-ray shows clear lung fields with no signs of pneumonia or other acute abnormalities. Heart size appears normal. No pleural effusions detected.",
        "CT scan of the abdomen reveals normal liver, kidney, and spleen appearance. No masses or lesions identified. Bowel loops appear normal.",
        "MRI brain scan demonstrates no acute intracranial abnormalities. White matter appears normal for patient age. No signs of stroke or hemorrhage.",
        "Ultrasound examination of the gallbladder shows no stones or wall thickening. Common bile duct appears normal in caliber.",
        "Mammography screening reveals dense breast tissue but no suspicious masses or calcifications. Recommend routine follow-up in one year."
    ]
    
    # Add documents with metadata
    metadata = [
        DocumentMetadata(
            doc_id=f"report_{i+1}",
            title=f"Medical Report {i+1}",
            source="Hospital Radiology Department",
            category="radiology",
            created_at=datetime.now().isoformat(),
            tags=["medical", "radiology", "report"]
        )
        for i in range(len(documents))
    ]
    
    doc_ids = rag.add_documents(documents, metadata)
    print(f"Added {len(doc_ids)} documents to RAG system")
    
    # Query the system
    queries = [
        "chest X-ray findings",
        "brain abnormalities",
        "normal examination results"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        results = rag.query(query, generate_response=False)
        
        print(f"Found {results['context_chunks']} relevant chunks")
        print(f"Context tokens: {results['context_tokens']}")
        print(f"Token efficiency: {results['retrieval_stats']['token_efficiency']:.2%}")
        
        if results['sources']:
            print("Sources:")
            for source in results['sources'][:3]:  # Show top 3
                print(f"  - {source['title']} (similarity: {source['similarity']:.3f})")
    
    # Show system statistics
    print(f"\nSystem Statistics:")
    print("-" * 30)
    stats = rag.get_system_stats()
    print(f"Documents: {stats['documents']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Total tokens: {stats['total_tokens']}")
    print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")