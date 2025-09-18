"""
DocQueryAI - Main Streamlit Application
PDF Chatbot with local LLM and vector search capabilities

This is the main entry point that orchestrates all components:
- Document processing pipeline
- Vector store management
- Chat engine with RAG capabilities
- Streamlit web interface
"""

import streamlit as st
import logging
import time
import sys
import traceback
from typing import Optional, Dict, Any
from enum import Enum

# Import our modules
from config import (
    APP_TITLE, 
    APP_DESCRIPTION, 
    MAX_FILE_SIZE_MB,
    ALLOWED_FILE_TYPES
)
from document_processor import validate_pdf, process_pdf
from text_chunker import process_document_chunks, get_chunk_stats
from vector_store import vector_store_manager, embedding_pipeline
from chat_engine import initialize_rag_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Application state is now managed through Streamlit session state


class ProcessingStatus(Enum):
    """Enumeration for document processing status"""
    IDLE = "idle"
    UPLOADING = "uploading"
    EXTRACTING = "extracting_text"
    CHUNKING = "chunking"
    EMBEDDING = "creating_embeddings"
    READY = "ready"
    ERROR = "error"


@st.cache_resource
def initialize_application() -> tuple:
    """
    Initialize all application components in proper sequence.
    This function is cached and will only run once per Streamlit session.
    
    Returns:
        tuple: (chat_engine, rag_initialized, startup_error)
    """
    try:
        logger.info("🚀 Starting DocQueryAI application initialization (one-time setup)...")
        
        # Step 1: Initialize vector store
        logger.info("Initializing vector store...")
        vector_init_success = vector_store_manager.initialize_vector_store()
        if not vector_init_success:
            logger.warning("Vector store initialization failed - RAG will not be available")
        
        # Step 2: Initialize RAG system (LLM + vector store integration)
        logger.info("Initializing RAG system...")
        try:
            chat_engine, rag_success = initialize_rag_system()
            
            if rag_success:
                logger.info("RAG system initialized successfully")
            else:
                logger.warning("RAG system initialization failed - falling back to basic chat")
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            startup_error = f"RAG system initialization failed: {str(e)}"
            return None, False, startup_error
        
        # Step 3: Verify all components are working
        logger.info("Verifying component status...")
        if chat_engine and chat_engine.is_available():
            logger.info("✅ LLM service is available")
        else:
            logger.warning("⚠️ LLM service is not available")
        
        if vector_store_manager.is_initialized():
            logger.info("✅ Vector store is initialized")
        else:
            logger.warning("⚠️ Vector store is not initialized")
        
        logger.info("🎉 DocQueryAI application initialization complete!")
        return chat_engine, rag_success, None
        
    except Exception as e:
        logger.error(f"Application initialization failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        startup_error = str(e)
        return None, False, startup_error


def shutdown_application():
    """
    Gracefully shutdown application components.
    """
    try:
        logger.info("Shutting down DocQueryAI application...")
        
        # Clear chat history
        if app_state.chat_engine:
            app_state.chat_engine.clear_chat_history()
            logger.info("Chat history cleared")
        
        # Clear vector store if needed
        # Note: We don't clear the vector store on shutdown as it's persistent
        # and users may want to keep their processed documents
        
        logger.info("Application shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during application shutdown: {str(e)}")


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = ProcessingStatus.IDLE
    
    if 'error_message' not in st.session_state:
        st.session_state.error_message = ""
    
    if 'success_message' not in st.session_state:
        st.session_state.success_message = ""
    
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = ""
    
    if 'document_ready' not in st.session_state:
        st.session_state.document_ready = False
    
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None
    
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
    
    # Chat interface state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0


def display_header():
    """Display the application header and description"""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title(APP_TITLE)
    st.markdown(f"*{APP_DESCRIPTION}*")
    st.markdown("---")


def display_status_indicators():
    """Display comprehensive processing status indicators and progress bars"""
    status = st.session_state.processing_status
    
    # Create columns for status display
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if status == ProcessingStatus.IDLE:
            st.info("📄 Ready to upload PDF document")
        elif status == ProcessingStatus.UPLOADING:
            st.info("📤 Uploading document...")
            st.caption("Validating file format and size...")
        elif status == ProcessingStatus.EXTRACTING:
            st.info("📖 Extracting text from PDF...")
            st.caption("Reading pages and extracting content...")
        elif status == ProcessingStatus.CHUNKING:
            st.info("✂️ Processing text into chunks...")
            st.caption("Splitting text for optimal AI processing...")
        elif status == ProcessingStatus.EMBEDDING:
            st.info("🧠 Creating embeddings...")
            st.caption("Generating AI embeddings for search...")
        elif status == ProcessingStatus.READY:
            st.success(f"✅ Document ready: {st.session_state.uploaded_file_name}")
            st.caption("Ready for questions and chat!")
        elif status == ProcessingStatus.ERROR:
            st.error("❌ Processing failed")
            st.caption("Check error details below")
    
    with col2:
        # Document status with more details
        if st.session_state.document_ready:
            st.success("📄 Document Loaded")
            if hasattr(st.session_state, 'document_stats'):
                stats = st.session_state.document_stats
                st.caption(f"{stats.get('chunks', 0)} chunks")
        else:
            st.warning("📄 No Document")
            st.caption("Upload a PDF to start")
    
    with col3:
        # RAG system status with connection details
        if st.session_state.rag_initialized:
            st.success("🤖 AI Ready")
            if st.session_state.chat_engine:
                rag_status = st.session_state.chat_engine.get_rag_status()
                if rag_status['rag_ready']:
                    st.caption("Full RAG available")
                else:
                    st.caption("Basic chat only")
        else:
            st.warning("🤖 AI Not Ready")
            st.caption("Initializing...")
    
    # Enhanced progress bar for active processing
    if status in [ProcessingStatus.UPLOADING, ProcessingStatus.EXTRACTING, 
                  ProcessingStatus.CHUNKING, ProcessingStatus.EMBEDDING]:
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            # Main progress bar
            if status == ProcessingStatus.UPLOADING:
                progress_value = 0.1
                progress_text = "Uploading and validating PDF..."
            elif status == ProcessingStatus.EXTRACTING:
                progress_value = 0.3
                progress_text = "Extracting text from PDF pages..."
            elif status == ProcessingStatus.CHUNKING:
                progress_value = 0.6
                progress_text = "Creating text chunks for processing..."
            elif status == ProcessingStatus.EMBEDDING:
                progress_value = 0.8
                progress_text = "Generating AI embeddings..."
            
            progress_bar = st.progress(progress_value)
            st.caption(progress_text)
            
            # Add estimated time remaining
            if hasattr(st.session_state, 'processing_start_time'):
                import time
                elapsed = time.time() - st.session_state.processing_start_time
                if status == ProcessingStatus.EXTRACTING:
                    estimated_total = elapsed / 0.3
                elif status == ProcessingStatus.CHUNKING:
                    estimated_total = elapsed / 0.6
                elif status == ProcessingStatus.EMBEDDING:
                    estimated_total = elapsed / 0.8
                else:
                    estimated_total = elapsed / 0.1
                
                remaining = max(0, estimated_total - elapsed)
                if remaining > 0:
                    st.caption(f"⏱️ Estimated time remaining: {remaining:.0f}s")


def display_error_messages():
    """Display error messages with comprehensive troubleshooting guidance"""
    if st.session_state.error_message:
        st.error(f"**Error:** {st.session_state.error_message}")
        
        # Add helpful suggestions based on error type
        error_msg = st.session_state.error_message.lower()
        
        if "ollama" in error_msg or "connection" in error_msg:
            st.info("""
            **🔧 Troubleshooting Ollama Connection:**
            1. **Check Ollama Service:** Make sure Ollama is installed and running
            2. **Start Service:** Run `ollama serve` in your terminal
            3. **Install Models:** 
               - `ollama pull phi3:mini`
               - `ollama pull nomic-embed-text`
            4. **Verify Connection:** Check if Ollama is accessible at http://localhost:11434
            5. **Restart:** Try restarting both Ollama and this application
            """)
        
        elif "model" in error_msg and ("not found" in error_msg or "pull" in error_msg):
            st.info("""
            **🤖 Missing AI Models:**
            The required AI models are not installed. Please run these commands:
            ```bash
            ollama pull phi3:mini
            ollama pull nomic-embed-text
            ```
            Then restart the application.
            """)
        
        elif "pdf" in error_msg or "file" in error_msg:
            st.info(f"""
            **📄 File Requirements:**
            - File must be a valid PDF format
            - Maximum file size: {MAX_FILE_SIZE_MB}MB
            - Supported formats: {', '.join(ALLOWED_FILE_TYPES)}
            - Ensure the PDF contains readable text (not just images)
            """)
        
        elif "memory" in error_msg or "size" in error_msg:
            st.info("""
            **💾 Memory Issues:**
            - Try uploading a smaller PDF file (under 10MB)
            - Close other applications to free up memory
            - Restart the application
            - Check available system memory
            """)
        
        elif "embedding" in error_msg:
            st.info("""
            **🧠 Embedding Generation Issues:**
            - Ensure Ollama is running with the nomic-embed-text model
            - Try restarting Ollama: `ollama serve`
            - Check if the document text is readable
            - Try processing a smaller document first
            """)
        
        elif "vector" in error_msg or "database" in error_msg:
            st.info("""
            **🗄️ Database Issues:**
            - The vector database may be corrupted
            - Try clearing the document and reprocessing
            - Restart the application
            - Check disk space availability
            """)
        
        elif "timeout" in error_msg:
            st.info("""
            **⏱️ Timeout Issues:**
            - The operation took too long to complete
            - Try processing a smaller document
            - Ensure Ollama has sufficient resources
            - Check your system performance
            """)
        
        else:
            st.info("""
            **🔍 General Troubleshooting:**
            - Try restarting the application
            - Ensure all dependencies are installed
            - Check the application logs for more details
            - Try with a different document
            """)


def display_success_messages():
    """Display success messages with enhanced formatting and actionable guidance"""
    if st.session_state.success_message:
        st.success(st.session_state.success_message)
        
        # Add actionable next steps for document processing success
        if st.session_state.document_ready and "processed successfully" in st.session_state.success_message:
            st.info("""
            **🚀 What's Next?**
            1. **Switch to the Chat tab** to start asking questions
            2. **Try asking specific questions** about the document content
            3. **Use natural language** - ask as you would ask a human
            4. **Check the sources** provided with each answer for verification
            """)
            
            # Add quick action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("💬 Go to Chat", use_container_width=True):
                    st.session_state.active_tab = "chat"
                    st.rerun()
            
            with col2:
                if st.button("📊 View Stats", use_container_width=True):
                    if hasattr(st.session_state, 'document_stats'):
                        stats = st.session_state.document_stats
                        st.balloons()
                        st.info(f"""
                        **📈 Document Statistics:**
                        - **Chunks:** {stats.get('chunks', 0)}
                        - **Characters:** {stats.get('characters', 0):,}
                        - **Database entries:** {stats.get('database_docs', 0)}
                        """)
            
            with col3:
                if st.button("🔄 Process Another", use_container_width=True):
                    clear_previous_data()
                    st.rerun()


def create_file_upload_widget():
    """Create and handle the PDF file upload widget"""
    st.subheader("📤 Upload PDF Document")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help=f"Upload a PDF file (max {MAX_FILE_SIZE_MB}MB)",
        key="pdf_uploader"
    )
    
    # Display file information if uploaded
    if uploaded_file is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("File Name", uploaded_file.name)
        
        with col2:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.metric("File Size", f"{file_size_mb:.1f} MB")
        
        with col3:
            st.metric("File Type", uploaded_file.type)
        
        # Validate file immediately
        is_valid, validation_error = validate_pdf(uploaded_file)
        
        if not is_valid:
            st.error(f"**File Validation Failed:** {validation_error}")
            return None
        else:
            st.success("✅ File validation passed")
            return uploaded_file
    
    return None


def clear_previous_data():
    """Clear previous document data and reset states"""
    # Clear vector store
    if vector_store_manager.is_initialized():
        vector_store_manager.clear_store()
    
    # Reset session state
    st.session_state.document_ready = False
    st.session_state.uploaded_file_name = ""
    st.session_state.error_message = ""
    st.session_state.success_message = ""
    st.session_state.processing_status = ProcessingStatus.IDLE
    
    # Clear chat history
    st.session_state.chat_history = []
    if st.session_state.chat_engine:
        st.session_state.chat_engine.clear_chat_history()


def process_uploaded_document(uploaded_file):
    """Process uploaded PDF document through the complete pipeline with comprehensive error handling and detailed feedback"""
    progress_container = None
    
    try:
        import time
        
        # Set initial status and track processing start time
        st.session_state.processing_status = ProcessingStatus.UPLOADING
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.error_message = ""
        st.session_state.success_message = ""
        st.session_state.processing_start_time = time.time()
        
        # Create a detailed progress container
        progress_container = st.empty()
        
        with progress_container.container():
            st.subheader("📊 Processing Progress")
            
            # Overall progress
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            # Detailed step progress
            step_progress = st.progress(0)
            step_status = st.empty()
            
            # Processing details
            details_expander = st.expander("📋 Processing Details", expanded=True)
            with details_expander:
                details_text = st.empty()
        
        # Step 1: Extract text from PDF
        try:
            overall_status.text("📖 Step 1/4: Extracting text from PDF...")
            st.session_state.processing_status = ProcessingStatus.EXTRACTING
            overall_progress.progress(0.25)
            step_progress.progress(0.1)
            
            details_text.markdown("""
            **Current Step:** Text Extraction
            - 🔍 Validating PDF structure
            - 📄 Reading PDF pages
            - 🔤 Extracting text content
            - 🧹 Cleaning extracted text
            """)
            
            step_status.text("🔍 Validating PDF and reading pages...")
            step_progress.progress(0.3)
            
            processed_text, extraction_error = process_pdf(uploaded_file)
            
            if extraction_error:
                raise ValueError(f"Text extraction failed: {extraction_error}")
            
            if not processed_text or not processed_text.strip():
                raise ValueError("No readable text could be extracted from the PDF. This may be a scanned document or contain only images.")
            
            # Show extraction success details
            text_length = len(processed_text)
            word_count = len(processed_text.split())
            
            step_status.text(f"✅ Text extraction complete!")
            step_progress.progress(1.0)
            
            details_text.markdown(f"""
            **✅ Text Extraction Complete**
            - 📊 **Characters extracted:** {text_length:,}
            - 📝 **Estimated words:** {word_count:,}
            - 📄 **File size:** {uploaded_file.size / 1024:.1f} KB
            - ⏱️ **Processing time:** {time.time() - st.session_state.processing_start_time:.1f}s
            """)
            
            time.sleep(0.5)  # Brief pause to show completion
                
        except ValueError as e:
            st.session_state.error_message = str(e)
            st.session_state.processing_status = ProcessingStatus.ERROR
            if progress_container:
                progress_container.empty()
            st.rerun()
            return
        except Exception as e:
            st.session_state.error_message = f"Unexpected error during text extraction: {str(e)}"
            st.session_state.processing_status = ProcessingStatus.ERROR
            if progress_container:
                progress_container.empty()
            st.rerun()
            return
        
        # Step 2: Chunk the text
        try:
            overall_status.text("✂️ Step 2/4: Processing text into chunks...")
            st.session_state.processing_status = ProcessingStatus.CHUNKING
            overall_progress.progress(0.5)
            step_progress.progress(0.1)
            
            details_text.markdown("""
            **Current Step:** Text Chunking
            - 📏 Analyzing text structure
            - ✂️ Splitting into optimal chunks
            - 🏷️ Creating metadata for each chunk
            - 📊 Validating chunk quality
            """)
            
            from config import CHUNK_SIZE, CHUNK_OVERLAP
            
            step_status.text("📏 Analyzing text structure...")
            step_progress.progress(0.3)
            
            # Create document chunks with metadata
            documents = process_document_chunks(
                text=processed_text,
                source=uploaded_file.name,
                document_title=uploaded_file.name.replace('.pdf', ''),
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP
            )
            
            step_status.text("✂️ Creating text chunks...")
            step_progress.progress(0.7)
            
            if not documents:
                raise ValueError("Failed to create document chunks. The text may be too short or contain invalid characters.")
            
            # Get chunk statistics
            chunk_stats = get_chunk_stats(documents)
            logger.info(f"Created {chunk_stats['total_chunks']} chunks from document")
            
            # Validate chunk quality
            if chunk_stats['total_chunks'] == 0:
                raise ValueError("No valid chunks were created from the document")
            if chunk_stats['avg_chunk_size'] < 50:
                raise ValueError("Document chunks are too small. The document may not contain enough readable text.")
            
            step_status.text("✅ Text chunking complete!")
            step_progress.progress(1.0)
            
            details_text.markdown(f"""
            **✅ Text Chunking Complete**
            - 📊 **Total chunks:** {chunk_stats['total_chunks']}
            - 📏 **Average chunk size:** {chunk_stats['avg_chunk_size']:.0f} characters
            - 📈 **Min/Max chunk size:** {chunk_stats['min_chunk_size']}/{chunk_stats['max_chunk_size']} chars
            - 🔧 **Chunk overlap:** {CHUNK_OVERLAP} characters
            - ⏱️ **Processing time:** {time.time() - st.session_state.processing_start_time:.1f}s
            """)
            
            time.sleep(0.5)  # Brief pause to show completion
                
        except ValueError as e:
            st.session_state.error_message = str(e)
            st.session_state.processing_status = ProcessingStatus.ERROR
            if progress_container:
                progress_container.empty()
            st.rerun()
            return
        except Exception as e:
            st.session_state.error_message = f"Unexpected error during text chunking: {str(e)}"
            st.session_state.processing_status = ProcessingStatus.ERROR
            if progress_container:
                progress_container.empty()
            st.rerun()
            return
        
        # Step 3: Create embeddings and store in vector database
        try:
            overall_status.text("🧠 Step 3/4: Creating embeddings and storing in database...")
            st.session_state.processing_status = ProcessingStatus.EMBEDDING
            overall_progress.progress(0.75)
            step_progress.progress(0.1)
            
            details_text.markdown("""
            **Current Step:** Embedding Generation
            - 🔌 Connecting to Ollama embedding service
            - 🧠 Generating AI embeddings for each chunk
            - 🗄️ Storing embeddings in vector database
            - ✅ Verifying database integrity
            """)
            
            step_status.text("🔌 Initializing vector database...")
            step_progress.progress(0.2)
            
            # Initialize vector store if not already done
            if not vector_store_manager.is_initialized():
                vector_init_success = vector_store_manager.initialize_vector_store()
                if not vector_init_success:
                    raise ConnectionError("Failed to initialize vector database. Please check if Ollama is running and the embedding model is available.")
            
            step_status.text("🧠 Generating embeddings...")
            step_progress.progress(0.4)
            
            # Extract chunks and metadata for embedding pipeline
            chunks = [doc.page_content for doc in documents]
            metadata_list = [doc.metadata for doc in documents]
            
            # Validate chunks before processing
            if not chunks or len(chunks) == 0:
                raise ValueError("No valid chunks available for embedding generation")
            
            step_status.text(f"🔄 Processing {len(chunks)} chunks in batches...")
            step_progress.progress(0.6)
            
            # Process through embedding pipeline with error handling
            embedding_success = embedding_pipeline.process_document_chunks(
                chunks=chunks,
                metadata=metadata_list,
                batch_size=10
            )
            
            if not embedding_success:
                raise RuntimeError("Failed to create embeddings for document chunks. This may be due to Ollama connection issues or model availability.")
            
            step_status.text("🗄️ Storing in vector database...")
            step_progress.progress(0.9)
            
            # Verify storage
            doc_count = vector_store_manager.get_collection_count()
            if doc_count == 0:
                raise RuntimeError("No documents found in vector store after processing")
            
            step_status.text("✅ Embedding generation complete!")
            step_progress.progress(1.0)
            
            details_text.markdown(f"""
            **✅ Embedding Generation Complete**
            - 🧠 **Embeddings created:** {len(chunks)}
            - 🗄️ **Documents in database:** {doc_count}
            - 🔧 **Batch size:** 10 chunks per batch
            - 🤖 **Embedding model:** nomic-embed-text
            - ⏱️ **Processing time:** {time.time() - st.session_state.processing_start_time:.1f}s
            """)
            
            time.sleep(0.5)  # Brief pause to show completion
                
        except ConnectionError as e:
            st.session_state.error_message = f"Connection error: {str(e)}"
            st.session_state.processing_status = ProcessingStatus.ERROR
            if progress_container:
                progress_container.empty()
            st.rerun()
            return
        except RuntimeError as e:
            st.session_state.error_message = f"Embedding error: {str(e)}"
            st.session_state.processing_status = ProcessingStatus.ERROR
            if progress_container:
                progress_container.empty()
            st.rerun()
            return
        except Exception as e:
            st.session_state.error_message = f"Unexpected error during embedding generation: {str(e)}"
            st.session_state.processing_status = ProcessingStatus.ERROR
            if progress_container:
                progress_container.empty()
            st.rerun()
            return
        
        # Step 4: Finalize processing
        try:
            overall_status.text("✅ Step 4/4: Finalizing document processing...")
            overall_progress.progress(0.9)
            step_progress.progress(0.1)
            
            details_text.markdown("""
            **Current Step:** Finalization
            - ✅ Verifying all processing steps
            - 🔍 Validating database integrity
            - 📊 Generating final statistics
            - 🎉 Preparing for chat interface
            """)
            
            step_status.text("🔍 Verifying processing results...")
            step_progress.progress(0.5)
            
            # Verify documents were added to vector store
            doc_count = vector_store_manager.get_collection_count()
            if doc_count == 0:
                raise RuntimeError("No documents found in vector store after processing. The embedding process may have failed silently.")
            
            step_status.text("📊 Generating final statistics...")
            step_progress.progress(0.8)
            
            # Store document statistics for display
            st.session_state.document_stats = {
                'chunks': chunk_stats['total_chunks'],
                'characters': chunk_stats['total_characters'],
                'avg_chunk_size': chunk_stats['avg_chunk_size'],
                'database_docs': doc_count
            }
            
            # Complete processing
            overall_progress.progress(1.0)
            step_progress.progress(1.0)
            overall_status.text("🎉 Document processing complete!")
            step_status.text("✅ Ready for questions!")
            
            total_time = time.time() - st.session_state.processing_start_time
            
            details_text.markdown(f"""
            **🎉 Processing Complete!**
            - ✅ **All steps completed successfully**
            - 📄 **File:** {uploaded_file.name}
            - 📊 **Total chunks:** {chunk_stats['total_chunks']}
            - 📝 **Total characters:** {chunk_stats['total_characters']:,}
            - 📏 **Average chunk size:** {chunk_stats['avg_chunk_size']:.0f} characters
            - 🗄️ **Documents in database:** {doc_count}
            - ⏱️ **Total processing time:** {total_time:.1f} seconds
            
            **🚀 You can now ask questions about your document in the Chat tab!**
            """)
            
            # Update session state
            st.session_state.processing_status = ProcessingStatus.READY
            st.session_state.document_ready = True
            st.session_state.success_message = f"""
            🎉 **Document processed successfully in {total_time:.1f} seconds!**
            
            **📊 Processing Summary:**
            - **File:** {uploaded_file.name}
            - **Chunks created:** {chunk_stats['total_chunks']}
            - **Total characters:** {chunk_stats['total_characters']:,}
            - **Average chunk size:** {chunk_stats['avg_chunk_size']:.0f} characters
            - **Documents in database:** {doc_count}
            
            **🚀 Ready for questions!** Switch to the Chat tab to start asking questions about your document.
            """
            
            # Show completion for a moment before clearing
            time.sleep(3)
            if progress_container:
                progress_container.empty()
            
            logger.info(f"Successfully processed document: {uploaded_file.name} in {total_time:.1f}s")
            st.rerun()
            
        except RuntimeError as e:
            st.session_state.error_message = str(e)
            st.session_state.processing_status = ProcessingStatus.ERROR
            if progress_container:
                progress_container.empty()
            st.rerun()
            return
        except Exception as e:
            st.session_state.error_message = f"Unexpected error during finalization: {str(e)}"
            st.session_state.processing_status = ProcessingStatus.ERROR
            if progress_container:
                progress_container.empty()
            st.rerun()
            return
        
    except Exception as e:
        # Catch-all for any unexpected errors
        logger.error(f"Unexpected document processing error: {str(e)}")
        st.session_state.error_message = f"An unexpected error occurred during document processing: {str(e)}"
        st.session_state.processing_status = ProcessingStatus.ERROR
        
        # Clear progress indicators
        if progress_container:
            progress_container.empty()
        
        st.rerun()


def create_chat_interface():
    """Create the chat interface for document querying"""
    if not st.session_state.get('document_ready', False):
        st.info("📄 Please upload and process a PDF document first to enable chat functionality.")
        return
    
    if not st.session_state.get('chat_engine'):
        st.error("❌ Chat engine not initialized. Please restart the application.")
        return
    
    st.subheader("💬 Chat with your Document")
    
    # Display chat history
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.write(f"**{i}.** {source['source']} (Page {source['page_number']})")
                            st.caption(source['content_preview'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Process query through RAG pipeline
                    response_data = st.session_state.chat_engine.process_query(prompt, use_rag=True)
                    
                    # Display response
                    st.write(response_data["response"])
                    
                    # Display sources if available
                    if response_data.get("sources"):
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(response_data["sources"], 1):
                                st.write(f"**{i}.** {source['source']} (Page {source['page_number']})")
                                st.caption(source['content_preview'])
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response_data["response"],
                        "sources": response_data.get("sources", [])
                    })
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })


def main():
    """
    Main application entry point that orchestrates all components.
    """
    try:
        # Initialize Streamlit page configuration
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state
        initialize_session_state()
        
        # Initialize application components (cached - only runs once)
        if 'app_initialized' not in st.session_state:
            with st.spinner("🚀 Initializing DocQueryAI (one-time setup)..."):
                chat_engine, rag_initialized, startup_error = initialize_application()
                
                # Store in session state
                st.session_state.chat_engine = chat_engine
                st.session_state.rag_initialized = rag_initialized
                st.session_state.startup_error = startup_error
                st.session_state.app_initialized = True
                
                if startup_error:
                    st.error(f"❌ Application initialization failed: {startup_error}")
                    st.info("""
                    **Troubleshooting Steps:**
                    1. Ensure Ollama is running: `ollama serve`
                    2. Install required models:
                       - `ollama pull phi3:mini`
                       - `ollama pull nomic-embed-text`
                    3. Restart this application
                    """)
                    st.stop()
        
        # Display header
        display_header()
        
        # Show startup error if any
        if st.session_state.get('startup_error'):
            st.error(f"❌ Application Error: {st.session_state.startup_error}")
            st.stop()
        
        # Display status indicators
        display_status_indicators()
        
        # Display error and success messages
        display_error_messages()
        display_success_messages()
        
        # Create main interface tabs
        tab1, tab2 = st.tabs(["📤 Upload Document", "💬 Chat"])
        
        with tab1:
            # File upload interface
            uploaded_file = create_file_upload_widget()
            
            if uploaded_file is not None:
                # Process document button
                if st.button("🔄 Process Document", type="primary", use_container_width=True):
                    # Clear previous data
                    clear_previous_data()
                    
                    # Process the uploaded document
                    process_uploaded_document(uploaded_file)
        
        with tab2:
            # Chat interface
            create_chat_interface()
        
        # Sidebar with application info and controls
        with st.sidebar:
            st.header("📊 Application Status")
            
            # System status
            if st.session_state.get('chat_engine') and st.session_state.chat_engine.is_available():
                st.success("🤖 LLM: Available")
            else:
                st.error("🤖 LLM: Not Available")
            
            if vector_store_manager.is_initialized():
                doc_count = vector_store_manager.get_collection_count()
                st.success(f"🗄️ Vector Store: {doc_count} documents")
            else:
                st.error("🗄️ Vector Store: Not Initialized")
            
            if st.session_state.get('document_ready', False):
                st.success("📄 Document: Ready")
            else:
                st.warning("📄 Document: Not Loaded")
            
            st.divider()
            
            # Application controls
            st.header("🔧 Controls")
            
            if st.button("🗑️ Clear All Data", use_container_width=True):
                clear_previous_data()
                st.success("All data cleared!")
                st.rerun()
            
            if st.button("🔄 Restart Application", use_container_width=True):
                # Clear cached resources and session state
                st.cache_resource.clear()
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.success("Application will restart...")
                st.rerun()
            
            st.divider()
            
            # Help and information
            st.header("ℹ️ Help")
            st.info("""
            **How to use:**
            1. Upload a PDF document
            2. Click "Process Document"
            3. Switch to Chat tab
            4. Ask questions about your document
            
            **Requirements:**
            - Ollama running locally
            - Models: phi3:mini, nomic-embed-text
            """)
    
    except Exception as e:
        logger.error(f"Critical application error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        st.error(f"❌ Critical Error: {str(e)}")
        st.info("Please restart the application and check the logs.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        shutdown_application()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        shutdown_application()
        sys.exit(1)
