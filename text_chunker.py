"""
Text chunking module for splitting documents into manageable chunks.
Uses LangChain's RecursiveCharacterTextSplitter for intelligent text splitting.
"""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import hashlib
import re


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks using LangChain's RecursiveCharacterTextSplitter.
    
    Args:
        text: Input text to be chunked
        chunk_size: Maximum size of each chunk (default: 1000 characters)
        overlap: Number of characters to overlap between chunks (default: 200)
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Initialize the text splitter with configured parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=[
            "\n\n",  # Paragraph breaks
            "\n",    # Line breaks
            ". ",    # Sentence endings
            "! ",    # Exclamation endings
            "? ",    # Question endings
            "; ",    # Semicolon breaks
            ", ",    # Comma breaks
            " ",     # Word breaks
            ""       # Character breaks (fallback)
        ]
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    
    # Filter out very small chunks (less than 50 characters)
    filtered_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 50]
    
    return filtered_chunks


def extract_page_info(chunk: str) -> int:
    """
    Extract page number information from chunk text.
    
    Args:
        chunk: Text chunk that may contain page markers
        
    Returns:
        Page number if found, otherwise 1
    """
    # Look for page break markers
    page_match = re.search(r'--- Page Break ---', chunk)
    if page_match:
        # Try to find the last page number before this chunk
        # This is a simplified approach - in practice, you might want more sophisticated tracking
        return 1
    
    # Default to page 1 if no page info found
    return 1


def create_chunk_id(chunk: str, index: int) -> str:
    """
    Create a unique identifier for a chunk.
    
    Args:
        chunk: Text content of the chunk
        index: Index of the chunk in the document
        
    Returns:
        Unique chunk identifier
    """
    # Create hash from chunk content for uniqueness
    chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
    return f"chunk_{index}_{chunk_hash}"


def create_metadata(chunks: List[str], source: str, document_title: str = None) -> List[Dict[str, Any]]:
    """
    Create metadata for each chunk including source and page information.
    
    Args:
        chunks: List of text chunks
        source: Source document name/path
        document_title: Optional document title
        
    Returns:
        List of metadata dictionaries for each chunk
    """
    metadata_list = []
    
    for i, chunk in enumerate(chunks):
        # Extract page information from chunk
        page_num = extract_page_info(chunk)
        
        # Create unique chunk ID
        chunk_id = create_chunk_id(chunk, i)
        
        # Build metadata dictionary
        metadata = {
            "source": source,
            "chunk_id": chunk_id,
            "chunk_index": i,
            "chunk_size": len(chunk),
            "page_number": page_num,
            "total_chunks": len(chunks)
        }
        
        # Add document title if provided
        if document_title:
            metadata["document_title"] = document_title
        
        # Add preview of chunk content (first 100 characters)
        metadata["preview"] = chunk[:100] + "..." if len(chunk) > 100 else chunk
        
        metadata_list.append(metadata)
    
    return metadata_list


def create_documents(chunks: List[str], metadata_list: List[Dict[str, Any]]) -> List[Document]:
    """
    Create LangChain Document objects from chunks and metadata.
    
    Args:
        chunks: List of text chunks
        metadata_list: List of metadata dictionaries
        
    Returns:
        List of LangChain Document objects
    """
    if len(chunks) != len(metadata_list):
        raise ValueError("Number of chunks must match number of metadata entries")
    
    documents = []
    for chunk, metadata in zip(chunks, metadata_list):
        doc = Document(
            page_content=chunk,
            metadata=metadata
        )
        documents.append(doc)
    
    return documents


def process_document_chunks(text: str, source: str, document_title: str = None, 
                          chunk_size: int = 1000, overlap: int = 200) -> List[Document]:
    """
    Complete document chunking pipeline: chunk text and create Document objects.
    
    Args:
        text: Input text to be processed
        source: Source document name/path
        document_title: Optional document title
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of LangChain Document objects ready for embedding
    """
    # Step 1: Chunk the text
    chunks = chunk_text(text, chunk_size, overlap)
    
    if not chunks:
        return []
    
    # Step 2: Create metadata for each chunk
    metadata_list = create_metadata(chunks, source, document_title)
    
    # Step 3: Create Document objects
    documents = create_documents(chunks, metadata_list)
    
    return documents


def get_chunk_stats(documents: List[Document]) -> Dict[str, Any]:
    """
    Get statistics about the chunked documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Dictionary with chunk statistics
    """
    if not documents:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0
        }
    
    chunk_sizes = [len(doc.page_content) for doc in documents]
    
    return {
        "total_chunks": len(documents),
        "total_characters": sum(chunk_sizes),
        "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes)
    }