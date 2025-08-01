#!/usr/bin/env python3
"""
Token-Aware Vector Store Implementation
Based on analysis of ChromaDB, LangChain, and Pinecone patterns
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import hashlib

@dataclass
class TokenMetadata:
    """Metadata for token storage in vector stores"""
    token_count: int
    tokenizer_name: str
    chunk_id: Optional[int] = None
    original_doc_id: Optional[str] = None
    token_positions: Optional[List[Tuple[int, int]]] = None
    overlap_tokens: Optional[int] = None

class TokenAwareVectorStore:
    """
    A vector store implementation that efficiently handles token storage
    and retrieval with awareness of token limits and chunking strategies.
    """
    
    def __init__(self, 
                 tokenizer,
                 embedding_model,
                 vector_store,
                 max_tokens: int = 512,
                 chunk_overlap: int = 50):
        """
        Initialize the token-aware vector store.
        
        Args:
            tokenizer: Tokenizer for text processing
            embedding_model: Model for generating embeddings
            vector_store: Underlying vector store (ChromaDB, Pinecone, etc.)
            max_tokens: Maximum tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap
        
    def chunk_tokens(self, tokens: List[str]) -> List[List[str]]:
        """
        Split tokens into chunks with overlap for better context preservation.
        
        Args:
            tokens: List of tokens to chunk
            
        Returns:
            List of token chunks
        """
        if len(tokens) <= self.max_tokens:
            return [tokens]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk = tokens[start:end]
            chunks.append(chunk)
            
            # Move start position considering overlap
            if end == len(tokens):
                break
            start = end - self.chunk_overlap
            
        return chunks
    
    def add_documents(self, 
                     documents: List[str], 
                     doc_ids: Optional[List[str]] = None,
                     metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add documents to the vector store with token-aware processing.
        
        Args:
            documents: List of documents to add
            doc_ids: Optional list of document IDs
            metadata: Optional metadata for each document
        """
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadata is None:
            metadata = [{} for _ in documents]
            
        for doc_idx, (document, doc_id, doc_meta) in enumerate(zip(documents, doc_ids, metadata)):
            # Tokenize the document
            tokens = self.tokenizer.encode(document)
            
            # Handle token limits through chunking
            token_chunks = self.chunk_tokens(tokens)
            
            for chunk_idx, chunk_tokens in enumerate(token_chunks):
                # Generate embedding for the chunk
                chunk_text = self.tokenizer.decode(chunk_tokens)
                embedding = self.embedding_model.encode(chunk_text)
                
                # Create comprehensive metadata
                chunk_metadata = {
                    **doc_meta,
                    "original_doc_id": doc_id,
                    "chunk_id": chunk_idx,
                    "token_count": len(chunk_tokens),
                    "total_chunks": len(token_chunks),
                    "tokenizer": self.tokenizer.__class__.__name__,
                    "max_tokens": self.max_tokens,
                    "chunk_overlap": self.chunk_overlap,
                    "token_start": chunk_idx * (self.max_tokens - self.chunk_overlap),
                    "token_end": chunk_idx * (self.max_tokens - self.chunk_overlap) + len(chunk_tokens),
                    "document_hash": hashlib.md5(document.encode()).hexdigest()
                }
                
                # Store in vector database
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                self.vector_store.add(
                    ids=[chunk_id],
                    documents=[chunk_text],
                    embeddings=[embedding],
                    metadatas=[chunk_metadata]
                )
    
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
        # Embed the query
        query_embedding = self.embedding_model.encode(query)
        
        # Retrieve initial results
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Select documents within token limit
        selected_docs = []
        selected_metadata = []
        total_tokens = 0
        
        for doc, meta, distance in zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        ):
            doc_tokens = meta.get('token_count', 0)
            
            if total_tokens + doc_tokens <= max_context_tokens:
                selected_docs.append(doc)
                selected_metadata.append({**meta, 'distance': distance})
                total_tokens += doc_tokens
            else:
                break
        
        return {
            'documents': selected_docs,
            'metadatas': selected_metadata,
            'total_tokens': total_tokens,
            'max_context_tokens': max_context_tokens,
            'query': query
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
        query_embedding = self.embedding_model.encode(query)
        
        # Query with token-based filtering
        results = self.vector_store.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 2,  # Get more to filter
            where={
                "$and": [
                    {"token_count": {"$gte": min_tokens}},
                    {"token_count": {"$lte": max_tokens}}
                ]
            },
            include=["documents", "metadatas", "distances"]
        )
        
        # Limit to requested number of results
        return {
            'documents': results['documents'][0][:n_results],
            'metadatas': results['metadatas'][0][:n_results],
            'distances': results['distances'][0][:n_results],
            'token_filter': {'min': min_tokens, 'max': max_tokens}
        }
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific document.
        
        Args:
            doc_id: Original document ID
            
        Returns:
            List of chunks with metadata
        """
        results = self.vector_store.get(
            where={"original_doc_id": doc_id},
            include=["documents", "metadatas"]
        )
        
        # Sort chunks by chunk_id
        chunks = []
        for doc, meta in zip(results['documents'], results['metadatas']):
            chunks.append({
                'chunk_id': meta.get('chunk_id'),
                'document': doc,
                'metadata': meta
            })
        
        return sorted(chunks, key=lambda x: x['chunk_id'])
    
    def reconstruct_document(self, doc_id: str) -> str:
        """
        Reconstruct original document from chunks (approximate).
        
        Args:
            doc_id: Original document ID
            
        Returns:
            Reconstructed document text
        """
        chunks = self.get_document_chunks(doc_id)
        
        if not chunks:
            return ""
        
        # Simple reconstruction - in practice, you'd want more sophisticated overlap handling
        reconstructed = ""
        for chunk in chunks:
            if chunk['chunk_id'] == 0:
                reconstructed = chunk['document']
            else:
                # Remove overlap tokens for better reconstruction
                overlap_size = self.chunk_overlap
                tokens = self.tokenizer.encode(chunk['document'])
                if len(tokens) > overlap_size:
                    non_overlap_tokens = tokens[overlap_size:]
                    non_overlap_text = self.tokenizer.decode(non_overlap_tokens)
                    reconstructed += " " + non_overlap_text
        
        return reconstructed
    
    def get_token_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about token usage in the vector store.
        
        Returns:
            Dictionary with token statistics
        """
        # Get all documents
        all_docs = self.vector_store.get(include=["metadatas"])
        
        token_counts = [meta.get('token_count', 0) for meta in all_docs['metadatas']]
        
        return {
            'total_chunks': len(token_counts),
            'total_tokens': sum(token_counts),
            'avg_tokens_per_chunk': np.mean(token_counts) if token_counts else 0,
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'token_distribution': {
                'p25': np.percentile(token_counts, 25) if token_counts else 0,
                'p50': np.percentile(token_counts, 50) if token_counts else 0,
                'p75': np.percentile(token_counts, 75) if token_counts else 0,
                'p95': np.percentile(token_counts, 95) if token_counts else 0,
            }
        }

# Example usage and integration patterns
class RAGWithTokenManagement:
    """
    Example RAG implementation using token-aware vector store
    """
    
    def __init__(self, token_vector_store, llm_model, max_context_tokens=2048):
        self.vector_store = token_vector_store
        self.llm = llm_model
        self.max_context_tokens = max_context_tokens
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """
        Answer a query using RAG with token management.
        
        Args:
            query: User query
            
        Returns:
            Answer with source information and token usage
        """
        # Retrieve relevant context with token limits
        context_data = self.vector_store.query_with_token_limit(
            query=query,
            max_context_tokens=self.max_context_tokens
        )
        
        # Prepare context for LLM
        context = "\n\n".join(context_data['documents'])
        
        # Generate response (placeholder - replace with actual LLM call)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        # response = self.llm.generate(prompt)
        
        return {
            'query': query,
            'context_documents': len(context_data['documents']),
            'context_tokens': context_data['total_tokens'],
            'sources': [meta.get('original_doc_id') for meta in context_data['metadatas']],
            # 'answer': response,
            'token_efficiency': context_data['total_tokens'] / self.max_context_tokens
        }

# Example integration with different vector stores
def create_chroma_token_store(collection_name: str = "token_aware_collection"):
    """
    Create a token-aware vector store using ChromaDB.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    
    # Initialize components
    client = chromadb.Client()
    collection = client.create_collection(collection_name)
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    return TokenAwareVectorStore(
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        vector_store=collection,
        max_tokens=256,
        chunk_overlap=25
    )

if __name__ == "__main__":
    # Example usage
    print("Token-Aware Vector Store Implementation")
    print("This implementation provides:")
    print("1. Automatic token chunking with overlap")
    print("2. Token-aware retrieval and context management")
    print("3. Token statistics and document reconstruction")
    print("4. Integration patterns for RAG applications")
    print("5. Support for multiple vector store backends")