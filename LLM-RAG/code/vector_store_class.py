"""
Token-Aware Vector Store Implementation
Integrates with existing tokenizer_class.py and embedding_class.py
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
import hashlib
import logging
from pathlib import Path
import pickle
import uuid
from datetime import datetime

# Import your existing classes
from tokenizer_class import Tokenizer
from embedding_class import EmbeddingManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TokenMetadata:
    """Metadata for token storage in vector stores"""
    token_count: int
    tokenizer_name: str
    embedding_model: str
    chunk_id: Optional[int] = None
    original_doc_id: Optional[str] = None
    token_positions: Optional[List[Tuple[int, int]]] = None
    overlap_tokens: Optional[int] = None
    created_at: Optional[str] = None
    document_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenMetadata':
        """Create from dictionary"""
        return cls(**data)

class TokenAwareVectorStore:
    """
    A vector store implementation that efficiently handles token storage
    and retrieval with awareness of token limits and chunking strategies.
    Integrates with your existing Tokenizer and EmbeddingManager classes.
    """
    
    def __init__(self, 
                 tokenizer_type: str = "bert",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_tokens: int = 512,
                 chunk_overlap: int = 50,
                 storage_path: str = "./vector_store_data"):
        """
        Initialize the token-aware vector store.
        
        Args:
            tokenizer_type: Type of tokenizer to use (from your config)
            embedding_model: Embedding model name
            max_tokens: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            storage_path: Path to store vector data
        """
        self.tokenizer_type = tokenizer_type
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize your existing components
        self.tokenizer = Tokenizer()
        self.embedding_manager = EmbeddingManager()
        
        # Load tokenizer and embedding model
        self._initialize_models()
        
        # In-memory storage (for demo - in production, use proper vector DB)
        self.documents = {}  # doc_id -> document content
        self.chunks = {}     # chunk_id -> chunk data
        self.embeddings = {} # chunk_id -> embedding vector
        self.metadata = {}   # chunk_id -> TokenMetadata
        self.index = {}      # Simple similarity search index
        
        logger.info(f"TokenAwareVectorStore initialized with {tokenizer_type} tokenizer and {embedding_model} embeddings")
    
    def _initialize_models(self):
        """Initialize tokenizer and embedding models"""
        try:
            # Load your tokenizer
            self.tokenizer.load_tokenizer(self.tokenizer_type)
            logger.info(f"Loaded tokenizer: {self.tokenizer_type}")
            
            # Load embedding model using your EmbeddingManager
            self.embedding_manager.load_model(self.embedding_model)
            logger.info(f"Loaded embedding model: {self.embedding_model}")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Fallback to BERT
            self.tokenizer.load_tokenizer("bert")
            self.embedding_manager.load_model("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Fallback to BERT tokenizer and default embeddings")
    
    def chunk_tokens(self, token_ids: List[int]) -> List[List[int]]:
        """
        Split token IDs into chunks with overlap for better context preservation.
        
        Args:
            token_ids: List of token IDs to chunk
            
        Returns:
            List of token ID chunks
        """
        if len(token_ids) <= self.max_tokens:
            return [token_ids]
        
        chunks = []
        start = 0
        
        while start < len(token_ids):
            end = min(start + self.max_tokens, len(token_ids))
            chunk = token_ids[start:end]
            chunks.append(chunk)
            
            # Move start position considering overlap
            if end == len(token_ids):
                break
            start = end - self.chunk_overlap
            
        return chunks
    
    def add_documents(self, 
                     documents: List[str], 
                     doc_ids: Optional[List[str]] = None,
                     metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add documents to the vector store with token-aware processing.
        
        Args:
            documents: List of documents to add
            doc_ids: Optional list of document IDs
            metadata: Optional metadata for each document
            
        Returns:
            List of chunk IDs created
        """
        if doc_ids is None:
            doc_ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in documents]
        
        if metadata is None:
            metadata = [{} for _ in documents]
        
        all_chunk_ids = []
        
        for doc_idx, (document, doc_id, doc_meta) in enumerate(zip(documents, doc_ids, metadata)):
            logger.info(f"Processing document {doc_id}")
            
            # Store original document
            self.documents[doc_id] = document
            
            # Tokenize the document using your tokenizer
            tokenization_result = self.tokenizer.tokenize(document, return_tensors=False)
            token_ids = tokenization_result['token_ids']
            tokens = tokenization_result['tokens']
            
            # Handle token limits through chunking
            token_chunks = self.chunk_tokens(token_ids)
            
            for chunk_idx, chunk_token_ids in enumerate(token_chunks):
                # Decode chunk back to text for embedding
                chunk_text = self.tokenizer.decode(chunk_token_ids)
                
                # Generate embedding using your EmbeddingManager
                try:
                    embedding_result = self.embedding_manager.get_embeddings([chunk_text])
                    embedding = embedding_result['embeddings'][0]  # Get first (and only) embedding
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {chunk_idx}: {e}")
                    continue
                
                # Create chunk ID
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                all_chunk_ids.append(chunk_id)
                
                # Create comprehensive metadata
                token_metadata = TokenMetadata(
                    token_count=len(chunk_token_ids),
                    tokenizer_name=self.tokenizer_type,
                    embedding_model=self.embedding_model,
                    chunk_id=chunk_idx,
                    original_doc_id=doc_id,
                    overlap_tokens=self.chunk_overlap,
                    created_at=datetime.now().isoformat(),
                    document_hash=hashlib.md5(document.encode()).hexdigest()
                )
                
                # Store chunk data
                self.chunks[chunk_id] = {
                    'text': chunk_text,
                    'token_ids': chunk_token_ids,
                    'tokens': [tokens[i] for i in range(len(chunk_token_ids)) if i < len(tokens)],
                    'doc_id': doc_id,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(token_chunks),
                    **doc_meta
                }
                
                # Store embedding and metadata
                self.embeddings[chunk_id] = embedding
                self.metadata[chunk_id] = token_metadata
                
                logger.debug(f"Added chunk {chunk_id} with {len(chunk_token_ids)} tokens")
        
        logger.info(f"Added {len(all_chunk_ids)} chunks from {len(documents)} documents")
        return all_chunk_ids
    
    def query_with_token_limit(self, 
                              query: str, 
                              max_context_tokens: int = 2048,
                              n_results: int = 10) -> Dict[str, Any]:
        """
        Query the vector store with token-aware context management.
        
        Args:
            query: Query string
            max_context_tokens: Maximum tokens to include in context
            n_results: Number of results to retrieve initially
            
        Returns:
            Dictionary with selected documents and token information
        """
        # Generate query embedding
        try:
            query_embedding_result = self.embedding_manager.get_embeddings([query])
            query_embedding = query_embedding_result['embeddings'][0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return {'error': str(e)}
        
        # Calculate similarities
        similarities = []
        for chunk_id, chunk_embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append((chunk_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Select documents within token limit
        selected_chunks = []
        selected_metadata = []
        total_tokens = 0
        
        for chunk_id, similarity in similarities[:n_results]:
            chunk_data = self.chunks[chunk_id]
            chunk_metadata = self.metadata[chunk_id]
            doc_tokens = chunk_metadata.token_count
            
            if total_tokens + doc_tokens <= max_context_tokens:
                selected_chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_data['text'],
                    'similarity': similarity,
                    'doc_id': chunk_data['doc_id']
                })
                selected_metadata.append({
                    **chunk_metadata.to_dict(),
                    'similarity': similarity
                })
                total_tokens += doc_tokens
            else:
                break
        
        return {
            'query': query,
            'chunks': selected_chunks,
            'metadata': selected_metadata,
            'total_tokens': total_tokens,
            'max_context_tokens': max_context_tokens,
            'chunks_selected': len(selected_chunks),
            'chunks_available': len(similarities)
        }
    
    def search_by_token_range(self, 
                             query: str,
                             min_tokens: int = 10,
                             max_tokens: int = 500,
                             n_results: int = 5) -> Dict[str, Any]:
        """
        Search for documents within a specific token range.
        
        Args:
            query: Query string
            min_tokens: Minimum token count
            max_tokens: Maximum token count
            n_results: Number of results to return
            
        Returns:
            Search results filtered by token range
        """
        # Filter chunks by token range first
        filtered_chunks = {}
        for chunk_id, metadata in self.metadata.items():
            if min_tokens <= metadata.token_count <= max_tokens:
                filtered_chunks[chunk_id] = self.embeddings[chunk_id]
        
        if not filtered_chunks:
            return {
                'query': query,
                'chunks': [],
                'message': f'No chunks found with token count between {min_tokens} and {max_tokens}'
            }
        
        # Generate query embedding
        try:
            query_embedding_result = self.embedding_manager.get_embeddings([query])
            query_embedding = query_embedding_result['embeddings'][0]
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return {'error': str(e)}
        
        # Calculate similarities for filtered chunks
        similarities = []
        for chunk_id, chunk_embedding in filtered_chunks.items():
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append((chunk_id, similarity))
        
        # Sort and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[:n_results]
        
        # Prepare results
        results = []
        for chunk_id, similarity in similarities:
            chunk_data = self.chunks[chunk_id]
            chunk_metadata = self.metadata[chunk_id]
            
            results.append({
                'chunk_id': chunk_id,
                'text': chunk_data['text'],
                'similarity': similarity,
                'token_count': chunk_metadata.token_count,
                'doc_id': chunk_data['doc_id']
            })
        
        return {
            'query': query,
            'chunks': results,
            'token_filter': {'min': min_tokens, 'max': max_tokens},
            'total_filtered': len(filtered_chunks)
        }
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            doc_id: Original document ID
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        for chunk_id, chunk_data in self.chunks.items():
            if chunk_data['doc_id'] == doc_id:
                chunk_metadata = self.metadata[chunk_id]
                chunks.append({
                    'chunk_id': chunk_id,
                    'chunk_idx': chunk_data['chunk_idx'],
                    'text': chunk_data['text'],
                    'tokens': chunk_data['tokens'],
                    'token_count': chunk_metadata.token_count,
                    'metadata': chunk_metadata.to_dict()
                })
        
        # Sort by chunk index
        return sorted(chunks, key=lambda x: x['chunk_idx'])
    
    def reconstruct_document(self, doc_id: str) -> str:
        """
        Reconstruct original document from chunks (approximate).
        
        Args:
            doc_id: Original document ID
            
        Returns:
            Reconstructed document text
        """
        if doc_id in self.documents:
            return self.documents[doc_id]
        
        chunks = self.get_document_chunks(doc_id)
        
        if not chunks:
            return ""
        
        # Simple reconstruction - remove overlap for better results
        reconstructed = ""
        for i, chunk in enumerate(chunks):
            if i == 0:
                reconstructed = chunk['text']
            else:
                # Try to remove overlap by finding common text
                prev_text = reconstructed
                curr_text = chunk['text']
                
                # Simple overlap removal (can be improved)
                words_prev = prev_text.split()
                words_curr = curr_text.split()
                
                # Find overlap
                overlap_found = False
                for j in range(min(self.chunk_overlap, len(words_prev), len(words_curr))):
                    if words_prev[-j-1:] == words_curr[:j+1]:
                        reconstructed += " " + " ".join(words_curr[j+1:])
                        overlap_found = True
                        break
                
                if not overlap_found:
                    reconstructed += " " + curr_text
        
        return reconstructed
    
    def get_token_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about token usage in the vector store.
        
        Returns:
            Dictionary with token statistics
        """
        if not self.metadata:
            return {'message': 'No data in vector store'}
        
        token_counts = [meta.token_count for meta in self.metadata.values()]
        
        return {
            'total_chunks': len(token_counts),
            'total_documents': len(self.documents),
            'total_tokens': sum(token_counts),
            'avg_tokens_per_chunk': np.mean(token_counts),
            'min_tokens': min(token_counts),
            'max_tokens': max(token_counts),
            'tokenizer_type': self.tokenizer_type,
            'embedding_model': self.embedding_model,
            'max_tokens_per_chunk': self.max_tokens,
            'chunk_overlap': self.chunk_overlap,
            'token_distribution': {
                'p25': np.percentile(token_counts, 25),
                'p50': np.percentile(token_counts, 50),
                'p75': np.percentile(token_counts, 75),
                'p95': np.percentile(token_counts, 95),
            }
        }
    
    def save_to_disk(self, filename: Optional[str] = None) -> str:
        """
        Save vector store data to disk.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"vector_store_{timestamp}.pkl"
        
        filepath = self.storage_path / filename
        
        data = {
            'documents': self.documents,
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'metadata': {k: v.to_dict() for k, v in self.metadata.items()},
            'config': {
                'tokenizer_type': self.tokenizer_type,
                'embedding_model': self.embedding_model,
                'max_tokens': self.max_tokens,
                'chunk_overlap': self.chunk_overlap
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Vector store saved to {filepath}")
        return str(filepath)
    
    def load_from_disk(self, filepath: str) -> None:
        """
        Load vector store data from disk.
        
        Args:
            filepath: Path to saved vector store file
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.documents = data['documents']
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']
        self.metadata = {k: TokenMetadata.from_dict(v) for k, v in data['metadata'].items()}
        
        # Update config if available
        if 'config' in data:
            config = data['config']
            self.tokenizer_type = config.get('tokenizer_type', self.tokenizer_type)
            self.embedding_model = config.get('embedding_model', self.embedding_model)
            self.max_tokens = config.get('max_tokens', self.max_tokens)
            self.chunk_overlap = config.get('chunk_overlap', self.chunk_overlap)
        
        logger.info(f"Vector store loaded from {filepath}")
        logger.info(f"Loaded {len(self.documents)} documents, {len(self.chunks)} chunks")

class RAGWithTokenManagement:
    """
    RAG implementation using token-aware vector store with your existing components
    """
    
    def __init__(self, 
                 vector_store: TokenAwareVectorStore,
                 max_context_tokens: int = 2048):
        """
        Initialize RAG system.
        
        Args:
            vector_store: TokenAwareVectorStore instance
            max_context_tokens: Maximum tokens for context
        """
        self.vector_store = vector_store
        self.max_context_tokens = max_context_tokens
        
        logger.info(f"RAG system initialized with max context tokens: {max_context_tokens}")
    
    def answer_query(self, query: str, include_sources: bool = True) -> Dict[str, Any]:
        """
        Answer a query using RAG with token management.
        
        Args:
            query: User query
            include_sources: Whether to include source information
            
        Returns:
            Answer with source information and token usage
        """
        # Retrieve relevant context with token limits
        context_data = self.vector_store.query_with_token_limit(
            query=query,
            max_context_tokens=self.max_context_tokens
        )
        
        if 'error' in context_data:
            return context_data
        
        # Prepare context for LLM
        context_chunks = context_data['chunks']
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Prepare sources
        sources = []
        if include_sources:
            sources = [
                {
                    'doc_id': chunk['doc_id'],
                    'chunk_id': chunk['chunk_id'],
                    'similarity': chunk['similarity']
                }
                for chunk in context_chunks
            ]
        
        # In a real implementation, you would call your LLM here
        # For now, we'll return the context and metadata
        
        return {
            'query': query,
            'context': context,
            'context_chunks': len(context_chunks),
            'context_tokens': context_data['total_tokens'],
            'sources': sources,
            'token_efficiency': context_data['total_tokens'] / self.max_context_tokens,
            'chunks_available': context_data['chunks_available'],
            'chunks_selected': context_data['chunks_selected']
        }
    
    def get_relevant_documents(self, 
                              query: str, 
                              token_range: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
        """
        Get relevant documents for a query with optional token filtering.
        
        Args:
            query: Search query
            token_range: Optional (min_tokens, max_tokens) tuple
            
        Returns:
            List of relevant documents
        """
        if token_range:
            min_tokens, max_tokens = token_range
            results = self.vector_store.search_by_token_range(
                query=query,
                min_tokens=min_tokens,
                max_tokens=max_tokens
            )
        else:
            results = self.vector_store.query_with_token_limit(query=query)
        
        return results.get('chunks', [])

if __name__ == "__main__":
    # Example usage
    print("Token-Aware Vector Store for RAG_LLM/Tokenization")
    print("=" * 60)
    
    # Initialize vector store
    vector_store = TokenAwareVectorStore(
        tokenizer_type="bert",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=256,
        chunk_overlap=25
    )
    
    # Sample documents
    documents = [
        "This is a medical report about chest X-ray findings. The patient shows clear lungs with no signs of pneumonia.",
        "Radiology report indicates normal heart size and clear lung fields. No acute cardiopulmonary abnormalities detected.",
        "CT scan reveals no significant abnormalities. Patient's condition appears stable with normal organ function."
    ]
    
    # Add documents
    chunk_ids = vector_store.add_documents(documents)
    print(f"Added {len(chunk_ids)} chunks to vector store")
    
    # Query with token management
    query = "chest X-ray findings"
    results = vector_store.query_with_token_limit(query, max_context_tokens=500)
    
    print(f"\nQuery: {query}")
    print(f"Found {results['chunks_selected']} relevant chunks using {results['total_tokens']} tokens")
    
    # Show statistics
    stats = vector_store.get_token_statistics()
    print(f"\nVector Store Statistics:")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")