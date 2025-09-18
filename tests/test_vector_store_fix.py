#!/usr/bin/env python3
"""
Quick test to verify vector store fix
"""

import logging
from vector_store import vector_store_manager, embedding_pipeline
from text_chunker import process_document_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_store_fix():
    """Test the vector store fix"""
    try:
        logger.info("Testing vector store fix...")
        
        # Create test content
        test_content = "This is a test document for DocQueryAI. It contains sample content for testing."
        
        # Process document chunks
        documents = process_document_chunks(
            text=test_content,
            source="test.pdf",
            document_title="Test Document"
        )
        
        logger.info(f"Created {len(documents)} document chunks")
        
        # Clear any existing data
        if vector_store_manager.is_initialized():
            vector_store_manager.clear_store()
        
        # Process through embedding pipeline
        chunks = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]
        
        success = embedding_pipeline.process_document_chunks(
            chunks=chunks,
            metadata=metadata_list,
            batch_size=5
        )
        
        if success:
            logger.info("✅ Vector store fix successful!")
            
            # Verify storage
            doc_count = vector_store_manager.get_collection_count()
            logger.info(f"✅ Vector store contains {doc_count} documents")
            
            return True
        else:
            logger.error("❌ Vector store fix failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_vector_store_fix()
    print(f"Test result: {'PASSED' if success else 'FAILED'}")