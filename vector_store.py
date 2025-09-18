"""
Vector Store Manager for DocQueryAI

This module handles Chroma database operations and embeddings integration
with Ollama for document storage and retrieval.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

from config import (
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    OLLAMA_BASE_URL
)


@dataclass
class DocumentChunk:
    """Data class for document chunks with metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str
    page_number: Optional[int] = None


class VectorStoreManager:
    """Manages Chroma vector database operations and Ollama embeddings"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.chroma_client = None
        self.collection_name = "documents"
        self.logger = logging.getLogger(__name__)
        
    def initialize_vector_store(self) -> bool:
        """
        Initialize local Chroma database with persistence and Ollama embeddings
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize Ollama embeddings with connection validation
            try:
                self.embeddings = OllamaEmbeddings(
                    model=EMBEDDING_MODEL,
                    base_url=OLLAMA_BASE_URL
                )
                
                # Test embedding connection
                test_embedding = self.embeddings.embed_query("test connection")
                if not test_embedding or len(test_embedding) == 0:
                    raise ConnectionError(f"Failed to generate test embedding with model {EMBEDDING_MODEL}")
                    
            except Exception as e:
                error_msg = str(e).lower()
                if "connection" in error_msg or "refused" in error_msg:
                    raise ConnectionError(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Please ensure Ollama is running.")
                elif "not found" in error_msg or "model" in error_msg:
                    raise RuntimeError(f"Embedding model '{EMBEDDING_MODEL}' not found. Please install it: ollama pull {EMBEDDING_MODEL}")
                else:
                    raise RuntimeError(f"Failed to initialize embeddings: {str(e)}")
            
            # Create vector database directory with error handling
            try:
                os.makedirs(VECTOR_DB_PATH, exist_ok=True)
            except PermissionError:
                raise PermissionError(f"Cannot create vector database directory at {VECTOR_DB_PATH}. Please check permissions.")
            except Exception as e:
                raise RuntimeError(f"Failed to create database directory: {str(e)}")
            
            # Initialize Chroma client with persistence
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=VECTOR_DB_PATH,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Chroma client: {str(e)}")
            
            # Initialize LangChain Chroma vector store
            try:
                self.vector_store = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=VECTOR_DB_PATH
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize vector store: {str(e)}")
            
            self.logger.info("Vector store initialized successfully")
            return True
            
        except (ConnectionError, RuntimeError, PermissionError) as e:
            self.logger.error(f"Vector store initialization failed: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during vector store initialization: {str(e)}")
            return False
    
    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of text chunks to add
            metadata: List of metadata dictionaries for each chunk
            
        Returns:
            bool: True if documents added successfully, False otherwise
        """
        if not self.vector_store:
            self.logger.warning("Vector store not initialized, attempting to initialize...")
            success = self.initialize_vector_store()
            if not success:
                self.logger.error("Failed to initialize vector store")
                return False
            
        if len(chunks) != len(metadata):
            self.logger.error("Chunks and metadata lists must have same length")
            return False
            
        try:
            # Create Document objects
            documents = [
                Document(page_content=chunk, metadata=meta)
                for chunk, meta in zip(chunks, metadata)
            ]
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            self.logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search against the vector database
        
        Args:
            query: Search query string
            k: Number of similar documents to return
            
        Returns:
            List of Document objects with similar content
        """
        if not self.vector_store:
            self.logger.error("Vector store not initialized")
            return []
            
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search(query, k=k)
            
            self.logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        Perform similarity search with relevance scores
        
        Args:
            query: Search query string
            k: Number of similar documents to return
            
        Returns:
            List of tuples (Document, score)
        """
        if not self.vector_store:
            self.logger.error("Vector store not initialized")
            return []
            
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            self.logger.info(f"Found {len(results)} similar documents with scores")
            return results
            
        except Exception as e:
            self.logger.error(f"Similarity search with score failed: {str(e)}")
            return []
    
    def clear_store(self) -> bool:
        """
        Clear all documents from the vector store
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        if not self.vector_store:
            self.logger.error("Vector store not initialized")
            return False
            
        try:
            # Delete the collection and recreate it
            if self.chroma_client:
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                except Exception:
                    pass  # Collection might not exist
                
                # Reinitialize the vector store
                self.vector_store = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=VECTOR_DB_PATH
                )
            
            self.logger.info("Vector store cleared successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear vector store: {str(e)}")
            return False
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection
        
        Returns:
            int: Number of documents in the collection
        """
        if not self.vector_store or not self.chroma_client:
            return 0
            
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            return collection.count()
        except Exception as e:
            self.logger.error(f"Failed to get collection count: {str(e)}")
            return 0
    
    def is_initialized(self) -> bool:
        """
        Check if vector store is properly initialized
        
        Returns:
            bool: True if initialized, False otherwise
        """
        return self.vector_store is not None and self.embeddings is not None


class EmbeddingPipeline:
    """Handles batch embedding generation for document chunks"""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self.logger = logging.getLogger(__name__)
    
    def connect_to_ollama_embedding_model(self) -> bool:
        """
        Connect to Ollama embedding model and verify it's working
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if not self.vector_store_manager.embeddings:
                self.logger.error("Embeddings not initialized in vector store manager")
                return False
            
            # Test the connection with a simple embedding
            test_text = "Connection test"
            test_embedding = self.vector_store_manager.embeddings.embed_query(test_text)
            
            if not test_embedding or len(test_embedding) == 0:
                raise Exception("Empty embedding returned")
            
            self.logger.info(f"Successfully connected to Ollama embedding model: {EMBEDDING_MODEL}")
            self.logger.info(f"Embedding dimension: {len(test_embedding)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Ollama embedding model: {str(e)}")
            return False
    
    def generate_batch_embeddings(self, chunks: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for document chunks in batches
        
        Args:
            chunks: List of text chunks to embed
            batch_size: Number of chunks to process in each batch
            
        Returns:
            List of embedding vectors
        """
        if not self.vector_store_manager.embeddings:
            self.logger.error("Embeddings not initialized")
            return []
        
        embeddings = []
        failed_chunks = []
        
        try:
            # Process chunks in batches to avoid overwhelming the model
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                try:
                    # Generate embeddings for the batch
                    batch_embeddings = self.vector_store_manager.embeddings.embed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    
                    self.logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
                    
                except Exception as batch_error:
                    self.logger.warning(f"Failed to embed batch {i//batch_size + 1}: {str(batch_error)}")
                    
                    # Try individual chunks in the failed batch
                    for j, chunk in enumerate(batch):
                        try:
                            chunk_embedding = self.vector_store_manager.embeddings.embed_query(chunk)
                            embeddings.append(chunk_embedding)
                        except Exception as chunk_error:
                            self.logger.error(f"Failed to embed chunk {i + j}: {str(chunk_error)}")
                            failed_chunks.append(i + j)
                            # Add empty embedding as placeholder
                            embeddings.append([])
            
            if failed_chunks:
                self.logger.warning(f"Failed to generate embeddings for {len(failed_chunks)} chunks")
            
            self.logger.info(f"Successfully generated {len([e for e in embeddings if e])} embeddings out of {len(chunks)} chunks")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {str(e)}")
            return []
    
    def process_document_chunks(self, chunks: List[str], metadata: List[Dict[str, Any]], 
                              batch_size: int = 10) -> bool:
        """
        Process document chunks through the complete embedding pipeline
        
        Args:
            chunks: List of text chunks
            metadata: List of metadata for each chunk
            batch_size: Batch size for embedding generation
            
        Returns:
            bool: True if processing successful, False otherwise
        """
        if len(chunks) != len(metadata):
            self.logger.error("Chunks and metadata lists must have same length")
            return False
        
        try:
            # Ensure vector store is initialized
            if not self.vector_store_manager.is_initialized():
                self.logger.info("Vector store not initialized, initializing now...")
                success = self.vector_store_manager.initialize_vector_store()
                if not success:
                    self.logger.error("Failed to initialize vector store")
                    return False
            
            # Verify connection to embedding model
            if not self.connect_to_ollama_embedding_model():
                return False
            
            # Filter out empty chunks
            valid_chunks = []
            valid_metadata = []
            
            for chunk, meta in zip(chunks, metadata):
                if chunk.strip():
                    valid_chunks.append(chunk)
                    valid_metadata.append(meta)
            
            if not valid_chunks:
                self.logger.warning("No valid chunks to process")
                return False
            
            self.logger.info(f"Processing {len(valid_chunks)} valid chunks")
            
            # Add documents to vector store (embeddings are generated automatically)
            success = self.vector_store_manager.add_documents(valid_chunks, valid_metadata)
            
            if success:
                self.logger.info("Document chunks processed successfully through embedding pipeline")
            else:
                self.logger.error("Failed to process document chunks")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Document chunk processing failed: {str(e)}")
            return False
    
    def handle_embedding_failures(self, error: Exception, chunk_index: int = None) -> str:
        """
        Handle embedding failures with appropriate error messages
        
        Args:
            error: The exception that occurred
            chunk_index: Index of the chunk that failed (if applicable)
            
        Returns:
            str: User-friendly error message
        """
        error_str = str(error).lower()
        
        if "connection" in error_str or "refused" in error_str:
            return "Cannot connect to Ollama service. Please ensure Ollama is running and try again."
        elif "model" in error_str and "not found" in error_str:
            return f"Embedding model '{EMBEDDING_MODEL}' not found. Please install it using: ollama pull {EMBEDDING_MODEL}"
        elif "timeout" in error_str:
            return "Embedding generation timed out. The document might be too large or Ollama is overloaded."
        elif "memory" in error_str or "out of memory" in error_str:
            return "Not enough memory to generate embeddings. Try processing smaller documents or restart Ollama."
        else:
            chunk_info = f" for chunk {chunk_index}" if chunk_index is not None else ""
            return f"Embedding generation failed{chunk_info}: {str(error)}"


# Global instances for the application
vector_store_manager = VectorStoreManager()
embedding_pipeline = EmbeddingPipeline(vector_store_manager)