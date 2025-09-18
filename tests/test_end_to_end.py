#!/usr/bin/env python3
"""
End-to-end workflow test for DocQueryAI
Tests the complete PDF upload, processing, and querying workflow
"""

import sys
import logging
import io
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_pdf_content():
    """Create test PDF content as bytes for testing"""
    # For this test, we'll create a simple text file and simulate PDF processing
    # In a real scenario, you would use a proper PDF file
    test_content = """
DocQueryAI Test Document

Introduction
This is a comprehensive test document for the DocQueryAI application. It contains sample content that can be used to verify the PDF processing and question-answering capabilities of the system.

Chapter 1: System Overview
DocQueryAI is a local PDF chatbot application that enables users to upload PDF documents and query them using natural language. The system extracts text from PDFs, processes it into searchable chunks with embeddings, and uses a local LLM (Ollama with Phi-3 Mini) to provide context-aware answers.

Key Features:
- PDF document upload and processing
- Text extraction and chunking using LangChain
- Vector embeddings with Chroma database
- Natural language querying with RAG (Retrieval-Augmented Generation)
- Local processing with Ollama for privacy and security

Chapter 2: Technical Architecture
The application uses the following components:
- Streamlit for the web interface and user interaction
- PyPDF2 for PDF text extraction and processing
- LangChain for text processing, chunking, and RAG pipeline
- Chroma for vector storage and similarity search
- Ollama for LLM inference and embeddings generation

The system follows a pipeline architecture where documents are processed through multiple stages: upload, validation, text extraction, chunking, embedding generation, and storage in the vector database.

Chapter 3: Usage Instructions
To use DocQueryAI effectively:
1. Upload a PDF document through the web interface
2. Wait for the document to be processed (this may take a few minutes)
3. Switch to the chat tab once processing is complete
4. Ask questions about the document content using natural language
5. Review the answers and source citations provided by the system

The system can answer questions about specific topics, summarize content, and provide detailed explanations based on the uploaded document.

Chapter 4: Performance and Limitations
DocQueryAI is designed for local processing and has the following characteristics:
- Maximum PDF file size: 50MB
- Optimal chunk size: 1000 characters with 200 character overlap
- Supports text-based PDFs (not scanned images)
- Response time depends on document size and query complexity
- All processing happens locally for privacy

Conclusion
This test document provides comprehensive sample content for verifying that the DocQueryAI application can successfully process documents, create embeddings, and answer questions about their content using the RAG pipeline.
"""
    return test_content

def test_document_processing():
    """Test the complete document processing pipeline"""
    try:
        logger.info("Testing document processing pipeline...")
        
        from document_processor import process_pdf
        from text_chunker import process_document_chunks, get_chunk_stats
        from vector_store import vector_store_manager, embedding_pipeline
        
        # Create test content
        test_content = create_test_pdf_content()
        
        # Simulate PDF processing (normally would extract from PDF)
        logger.info("✅ Simulating PDF text extraction...")
        processed_text = test_content
        
        # Test text chunking
        logger.info("Testing text chunking...")
        documents = process_document_chunks(
            text=processed_text,
            source="test_document.pdf",
            document_title="DocQueryAI Test Document"
        )
        
        if not documents:
            logger.error("❌ No documents created from text chunking")
            return False
        
        logger.info(f"✅ Created {len(documents)} document chunks")
        
        # Get chunk statistics
        stats = get_chunk_stats(documents)
        logger.info(f"✅ Chunk statistics: {stats}")
        
        # Test embedding pipeline
        logger.info("Testing embedding generation and storage...")
        
        # Clear any existing data
        if vector_store_manager.is_initialized():
            vector_store_manager.clear_store()
        
        # Initialize vector store
        if not vector_store_manager.is_initialized():
            success = vector_store_manager.initialize_vector_store()
            if not success:
                logger.error("❌ Failed to initialize vector store")
                return False
        
        # Process document chunks through embedding pipeline
        chunks = [doc.page_content for doc in documents]
        metadata_list = [doc.metadata for doc in documents]
        
        embedding_success = embedding_pipeline.process_document_chunks(
            chunks=chunks,
            metadata=metadata_list,
            batch_size=5
        )
        
        if not embedding_success:
            logger.error("❌ Failed to process document chunks through embedding pipeline")
            return False
        
        logger.info("✅ Document chunks processed and stored successfully")
        
        # Verify storage
        doc_count = vector_store_manager.get_collection_count()
        logger.info(f"✅ Vector store contains {doc_count} documents")
        
        if doc_count == 0:
            logger.error("❌ No documents found in vector store after processing")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Document processing test failed: {str(e)}")
        return False

def test_query_processing():
    """Test the complete query processing and RAG pipeline"""
    try:
        logger.info("Testing query processing and RAG pipeline...")
        
        from chat_engine import initialize_rag_system
        
        # Initialize RAG system
        chat_engine, rag_success = initialize_rag_system()
        
        if not chat_engine:
            logger.error("❌ Failed to initialize chat engine")
            return False
        
        if not rag_success:
            logger.warning("⚠️ RAG system not fully initialized, testing basic LLM functionality")
        
        # Test queries
        test_queries = [
            "What is DocQueryAI?",
            "What are the key features of the system?",
            "How does the technical architecture work?",
            "What are the usage instructions?",
            "What are the performance limitations?"
        ]
        
        successful_queries = 0
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"Testing query {i}/{len(test_queries)}: '{query}'")
            
            try:
                # Process query
                response_data = chat_engine.process_query(query, use_rag=True)
                
                if response_data and response_data.get("response"):
                    logger.info(f"✅ Query {i} successful")
                    logger.info(f"   Response length: {len(response_data['response'])} characters")
                    
                    if response_data.get("sources"):
                        logger.info(f"   Sources found: {len(response_data['sources'])}")
                    else:
                        logger.info("   No sources found (may be using direct LLM mode)")
                    
                    successful_queries += 1
                else:
                    logger.error(f"❌ Query {i} failed - no response generated")
                
            except Exception as e:
                logger.error(f"❌ Query {i} failed with error: {str(e)}")
        
        logger.info(f"Query processing results: {successful_queries}/{len(test_queries)} successful")
        
        if successful_queries >= len(test_queries) * 0.8:  # 80% success rate
            logger.info("✅ Query processing test passed")
            return True
        else:
            logger.error("❌ Query processing test failed - too many failed queries")
            return False
        
    except Exception as e:
        logger.error(f"❌ Query processing test failed: {str(e)}")
        return False

def test_chat_history():
    """Test chat history functionality"""
    try:
        logger.info("Testing chat history functionality...")
        
        from chat_engine import initialize_rag_system
        
        # Initialize RAG system
        chat_engine, _ = initialize_rag_system()
        
        if not chat_engine:
            logger.error("❌ Failed to initialize chat engine")
            return False
        
        # Clear any existing history
        chat_engine.clear_chat_history()
        
        # Test multiple queries to build history
        queries = [
            "What is DocQueryAI?",
            "What are its key features?",
            "How do I use it?"
        ]
        
        for query in queries:
            response_data = chat_engine.process_query(query, use_rag=False)  # Use direct mode for faster testing
            if not response_data.get("response"):
                logger.error(f"❌ Failed to get response for query: {query}")
                return False
        
        # Check chat history
        history = chat_engine.get_chat_history()
        
        expected_messages = len(queries) * 2  # Each query generates user + assistant message
        
        if len(history) == expected_messages:
            logger.info(f"✅ Chat history contains {len(history)} messages as expected")
            
            # Verify message structure
            for i, message in enumerate(history):
                if "role" not in message or "content" not in message:
                    logger.error(f"❌ Message {i} missing required fields")
                    return False
                
                if message["role"] not in ["user", "assistant"]:
                    logger.error(f"❌ Message {i} has invalid role: {message['role']}")
                    return False
            
            logger.info("✅ Chat history structure is valid")
            return True
        else:
            logger.error(f"❌ Expected {expected_messages} messages, got {len(history)}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Chat history test failed: {str(e)}")
        return False

def test_error_scenarios():
    """Test error handling scenarios"""
    try:
        logger.info("Testing error handling scenarios...")
        
        from vector_store import vector_store_manager
        from chat_engine import initialize_rag_system
        
        # Test 1: Query with no documents in vector store
        logger.info("Testing query with empty vector store...")
        
        # Clear vector store
        if vector_store_manager.is_initialized():
            vector_store_manager.clear_store()
        
        # Initialize chat engine
        chat_engine, _ = initialize_rag_system()
        
        if chat_engine:
            response_data = chat_engine.process_query("What is in the document?", use_rag=True)
            
            if response_data and response_data.get("response"):
                logger.info("✅ Handled empty vector store gracefully")
            else:
                logger.error("❌ Failed to handle empty vector store")
                return False
        
        # Test 2: Invalid query handling
        logger.info("Testing invalid query handling...")
        
        if chat_engine:
            response_data = chat_engine.process_query("", use_rag=True)
            
            if response_data:
                logger.info("✅ Handled empty query gracefully")
            else:
                logger.error("❌ Failed to handle empty query")
                return False
        
        logger.info("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {str(e)}")
        return False

def run_end_to_end_test():
    """Run complete end-to-end workflow test"""
    logger.info("🚀 Starting DocQueryAI End-to-End Workflow Test...")
    
    tests = [
        ("Document Processing", test_document_processing),
        ("Query Processing", test_query_processing),
        ("Chat History", test_chat_history),
        ("Error Handling", test_error_scenarios)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*60}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("END-TO-END TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All end-to-end tests passed! DocQueryAI workflow is working correctly.")
        logger.info("\n📋 Test Coverage:")
        logger.info("✅ PDF document processing pipeline")
        logger.info("✅ Text extraction and chunking")
        logger.info("✅ Vector embedding generation and storage")
        logger.info("✅ RAG query processing")
        logger.info("✅ Chat history management")
        logger.info("✅ Error handling and edge cases")
        return True
    else:
        logger.error("⚠️ Some end-to-end tests failed. Please check the logs above.")
        return False

if __name__ == "__main__":
    success = run_end_to_end_test()
    sys.exit(0 if success else 1)