# Vector Store Token Storage Analysis

## Overview
This analysis explores how vector stores handle token storage and embedding management across different implementations, focusing on ChromaDB, LangChain, and Pinecone.

## Key Findings from Codebase Analysis

### 1. ChromaDB Token Handling

#### Core Collection Operations
ChromaDB provides a comprehensive API for storing and retrieving embeddings with associated tokens/documents:

```python
# From ChromaDB Collection.py
def add(self, ids, embeddings=None, metadatas=None, documents=None, images=None, uris=None):
    """Add embeddings to the data store.
    Args:
        ids: The ids of the embeddings you wish to add
        embeddings: The embeddings to add. If None, embeddings will be computed based on the documents
        metadatas: The metadata to associate with the embeddings
        documents: The documents to associate with the embeddings
    """

def query(self, query_embeddings=None, query_texts=None, n_results=10, where=None, include=["metadatas", "documents", "distances"]):
    """Get the n_results nearest neighbor embeddings for provided query_embeddings or query_texts."""

def upsert(self, ids, embeddings=None, metadatas=None, documents=None):
    """Update the embeddings, metadatas or documents for provided ids, or create them if they don't exist."""
```

#### Token Processing in Embedding Functions
ChromaDB handles tokenization through specialized embedding functions:

```python
# From onnx_mini_lm_l6_v2.py
class ONNXMiniLM_L6_V2EmbeddingFunction:
    def __init__(self):
        self.tokenizer = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def __call__(self, input: Documents) -> Embeddings:
        # Tokenize documents
        encoded = [self.tokenizer.encode(d) for d in batch]
        
        # Handle token limits
        if any(len(tokens.ids) > self.max_tokens for tokens in encoded):
            # Truncate or handle oversized documents
            
        # Convert to embeddings
        return embeddings
```

### 2. Token Storage Patterns

#### A. Document-Token-Embedding Triplets
Vector stores typically store three related pieces of information:
1. **Original Document/Text**: The raw text content
2. **Tokens**: Processed/tokenized version of the text
3. **Embeddings**: Vector representations of the tokens

#### B. Metadata Association
```python
# Example storage pattern
{
    "id": "doc_123",
    "document": "The quick brown fox jumps over the lazy dog",
    "metadata": {
        "source": "example.txt",
        "tokens": ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
        "token_count": 9,
        "chunk_id": 1
    },
    "embedding": [0.1, 0.2, -0.3, ...]
}
```

### 3. Token Management Strategies

#### A. Chunking for Token Limits
```python
# Common pattern for handling token limits
def chunk_tokens(text, max_tokens=512, overlap=50):
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(chunk)
    
    return chunks
```

#### B. Token Metadata Storage
Vector stores often store token-related metadata:
- Token count per document
- Tokenization method used
- Token overlap information
- Original text boundaries

### 4. Use Cases for Token Storage in Vector Stores

#### A. Retrieval Augmented Generation (RAG)
```python
# RAG workflow with token awareness
def rag_query(query, vector_store):
    # 1. Tokenize and embed query
    query_tokens = tokenizer.encode(query)
    query_embedding = embedding_model(query_tokens)
    
    # 2. Retrieve similar documents
    results = vector_store.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    
    # 3. Use token metadata for context window management
    context_tokens = 0
    selected_docs = []
    
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        doc_tokens = metadata.get('token_count', 0)
        if context_tokens + doc_tokens <= MAX_CONTEXT_TOKENS:
            selected_docs.append(doc)
            context_tokens += doc_tokens
    
    return selected_docs
```

#### B. Semantic Search with Token Filtering
```python
# Search with token-based filtering
def semantic_search_with_token_filter(query, min_tokens=10, max_tokens=500):
    results = vector_store.query(
        query_texts=[query],
        where={
            "$and": [
                {"token_count": {"$gte": min_tokens}},
                {"token_count": {"$lte": max_tokens}}
            ]
        }
    )
    return results
```

#### C. Token-Aware Indexing
```python
# Indexing with token metadata
def index_documents_with_tokens(documents, vector_store):
    for doc_id, document in enumerate(documents):
        # Tokenize
        tokens = tokenizer.encode(document)
        
        # Create embedding
        embedding = embedding_model(tokens)
        
        # Store with token metadata
        vector_store.add(
            ids=[f"doc_{doc_id}"],
            documents=[document],
            embeddings=[embedding],
            metadatas=[{
                "token_count": len(tokens),
                "tokens": tokens,  # Optional: store actual tokens
                "tokenizer": "sentence-transformers/all-MiniLM-L6-v2"
            }]
        )
```

### 5. Advanced Token Storage Patterns

#### A. Hierarchical Token Storage
```python
# Store tokens at multiple granularities
{
    "document_id": "doc_123",
    "full_document": "...",
    "chunks": [
        {
            "chunk_id": "chunk_123_1",
            "text": "First paragraph...",
            "tokens": ["first", "paragraph", ...],
            "token_positions": [(0, 5), (6, 15), ...],
            "embedding": [...]
        },
        {
            "chunk_id": "chunk_123_2", 
            "text": "Second paragraph...",
            "tokens": ["second", "paragraph", ...],
            "token_positions": [(16, 22), (23, 32), ...],
            "embedding": [...]
        }
    ]
}
```

#### B. Token-Level Embeddings
```python
# Store embeddings for individual tokens
def store_token_level_embeddings(text, vector_store):
    tokens = tokenizer.encode(text)
    
    for i, token in enumerate(tokens):
        token_embedding = get_token_embedding(token, context=tokens)
        
        vector_store.add(
            ids=[f"token_{text_id}_{i}"],
            documents=[tokenizer.decode([token])],
            embeddings=[token_embedding],
            metadatas=[{
                "token_position": i,
                "context_window": tokens[max(0, i-5):i+6],
                "document_id": text_id
            }]
        )
```

### 6. Performance Considerations

#### A. Token Caching
- Cache tokenized versions to avoid re-tokenization
- Store token counts for quick filtering
- Pre-compute token boundaries for chunking

#### B. Memory Management
- Store tokens separately from embeddings for large documents
- Use token IDs instead of full token strings
- Implement token compression for storage efficiency

### 7. Integration Examples

#### A. LangChain Integration
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# LangChain automatically handles tokenization
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=OpenAIEmbeddings(),
    metadatas=[{"tokens": len(tokenizer.encode(doc))} for doc in documents]
)
```

#### B. Custom Token-Aware Vector Store
```python
class TokenAwareVectorStore:
    def __init__(self, tokenizer, embedding_model, vector_store):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.vector_store = vector_store
    
    def add_documents(self, documents):
        for doc_id, document in enumerate(documents):
            # Process tokens
            tokens = self.tokenizer.encode(document)
            chunks = self.chunk_tokens(tokens)
            
            # Store each chunk with token metadata
            for chunk_id, chunk_tokens in enumerate(chunks):
                chunk_text = self.tokenizer.decode(chunk_tokens)
                embedding = self.embedding_model(chunk_tokens)
                
                self.vector_store.add(
                    ids=[f"{doc_id}_{chunk_id}"],
                    documents=[chunk_text],
                    embeddings=[embedding],
                    metadatas=[{
                        "original_doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "token_count": len(chunk_tokens),
                        "token_start": chunk_id * (self.max_tokens - self.overlap),
                        "token_end": chunk_id * (self.max_tokens - self.overlap) + len(chunk_tokens)
                    }]
                )
```

## Conclusion

Vector stores can effectively be used for token storage by:
1. **Storing token metadata** alongside embeddings
2. **Managing token limits** through chunking strategies
3. **Enabling token-aware retrieval** for better context management
4. **Supporting hierarchical token storage** for different granularities
5. **Optimizing performance** through caching and compression

This approach enables sophisticated applications like RAG systems, semantic search, and token-level analysis while maintaining the benefits of vector similarity search.