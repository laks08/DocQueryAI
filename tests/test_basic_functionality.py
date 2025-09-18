#!/usr/bin/env python3
"""
Basic functionality test for DocQueryAI components
"""

import sys
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported successfully"""
    try:
        logger.info("Testing imports...")
        
        # Test config import
        from config import OLLAMA_MODEL, EMBEDDING_MODEL, APP_TITLE
        logger.info(f"✅ Config imported - Model: {OLLAMA_MODEL}, Embedding: {EMBEDDING_MODEL}")
        
        # Test document processor
        from document_processor import validate_pdf, process_pdf
        logger.info("✅ Document processor imported")
        
        # Test text chunker
        from text_chunker import process_document_chunks, get_chunk_stats
        logger.info("✅ Text chunker imported")
        
        # Test vector store
        from vector_store import vector_store_manager, embedding_pipeline
        logger.info("✅ Vector store imported")
        
        # Test chat engine
        from chat_engine import initialize_rag_system
        logger.info("✅ Chat engine imported")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {str(e)}")
        return False

def test_vector_store_initialization():
    """Test vector store initialization"""
    try:
        logger.info("Testing vector store initialization...")
        
        from vector_store import vector_store_manager
        
        # Initialize vector store
        success = vector_store_manager.initialize_vector_store()
        
        if success:
            logger.info("✅ Vector store initialized successfully")
            
            # Test basic operations
            is_initialized = vector_store_manager.is_initialized()
            logger.info(f"✅ Vector store status: {'initialized' if is_initialized else 'not initialized'}")
            
            # Get collection count
            count = vector_store_manager.get_collection_count()
            logger.info(f"✅ Collection count: {count}")
            
            return True
        else:
            logger.error("❌ Vector store initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {str(e)}")
        return False

def test_chat_engine_initialization():
    """Test chat engine initialization"""
    try:
        logger.info("Testing chat engine initialization...")
        
        from chat_engine import initialize_rag_system
        
        # Initialize RAG system
        chat_engine, rag_success = initialize_rag_system()
        
        if chat_engine:
            logger.info("✅ Chat engine initialized")
            
            # Test LLM availability
            is_available = chat_engine.is_available()
            logger.info(f"✅ LLM availability: {'available' if is_available else 'not available'}")
            
            # Test RAG status
            rag_status = chat_engine.get_rag_status()
            logger.info(f"✅ RAG status: {rag_status}")
            
            return True
        else:
            logger.error("❌ Chat engine initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Chat engine test failed: {str(e)}")
        return False

def test_text_processing():
    """Test text processing functionality"""
    try:
        logger.info("Testing text processing...")
        
        from text_chunker import process_document_chunks, get_chunk_stats
        
        # Test text
        test_text = """
        This is a test document for DocQueryAI.
        
        Chapter 1: Introduction
        DocQueryAI is a PDF chatbot application that allows users to upload documents and ask questions about them.
        
        Chapter 2: Features
        The application includes PDF processing, text chunking, vector embeddings, and natural language querying.
        
        Chapter 3: Architecture
        The system uses Streamlit for the interface, Ollama for LLM, and Chroma for vector storage.
        """
        
        # Process document chunks
        documents = process_document_chunks(
            text=test_text,
            source="test_document.txt",
            document_title="Test Document"
        )
        
        if documents:
            logger.info(f"✅ Text processing successful - {len(documents)} chunks created")
            
            # Get statistics
            stats = get_chunk_stats(documents)
            logger.info(f"✅ Chunk statistics: {stats}")
            
            return True
        else:
            logger.error("❌ Text processing failed - no chunks created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Text processing test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all basic functionality tests"""
    logger.info("🚀 Starting DocQueryAI basic functionality tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Vector Store Test", test_vector_store_initialization),
        ("Chat Engine Test", test_chat_engine_initialization),
        ("Text Processing Test", test_text_processing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! DocQueryAI basic functionality is working.")
        return True
    else:
        logger.error("⚠️ Some tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)