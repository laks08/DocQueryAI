"""
Chat engine for RAG pipeline and LLM interaction.
Handles Ollama LLM integration, query processing, and retrieval-augmented generation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, BaseMessage, Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import (
    OLLAMA_MODEL, 
    OLLAMA_BASE_URL, 
    LLM_TEMPERATURE,
    MAX_CHAT_HISTORY,
    SIMILARITY_SEARCH_K
)
from vector_store import vector_store_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatEngine:
    """
    Handles LLM integration, RAG pipeline, and query processing for the DocQueryAI system.
    """
    
    def __init__(self):
        """Initialize the chat engine with Ollama LLM and RAG capabilities."""
        self.llm = None
        self.rag_chain = None
        self.chat_history: List[BaseMessage] = []
        self.vector_store_manager = vector_store_manager
        self._initialize_llm()
        self._create_rag_chain()
    
    def _initialize_llm(self) -> None:
        """
        Initialize ChatOllama with phi3:mini model and configure settings.
        """
        try:
            self.llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=LLM_TEMPERATURE,
                # Context window management - phi3:mini supports up to 4k tokens
                num_ctx=4096,
                # Disable streaming for simpler implementation
                streaming=False
            )
            
            # Test the connection with error handling
            try:
                test_response = self.llm.invoke([HumanMessage(content="Hello")])
                if not test_response:
                    raise RuntimeError("LLM returned empty response during connection test")
                logger.info(f"LLM initialized successfully with model: {OLLAMA_MODEL}")
            except Exception as test_error:
                error_msg = str(test_error).lower()
                if "connection" in error_msg or "refused" in error_msg:
                    raise ConnectionError(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Please ensure Ollama is running.")
                elif "not found" in error_msg or "model" in error_msg:
                    raise RuntimeError(f"Model '{OLLAMA_MODEL}' not found. Please install it: ollama pull {OLLAMA_MODEL}")
                elif "timeout" in error_msg:
                    raise TimeoutError(f"Connection to Ollama timed out. Please check if Ollama is responsive.")
                else:
                    raise RuntimeError(f"LLM connection test failed: {str(test_error)}")
            
        except (ConnectionError, RuntimeError, TimeoutError):
            # Re-raise specific errors
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise ConnectionError(f"Unexpected error connecting to Ollama service: {str(e)}")
    
    def _create_rag_chain(self) -> None:
        """
        Create retrieval chain that combines similarity search with LLM.
        """
        try:
            # Create a custom prompt template for RAG
            rag_prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Always cite which parts of the context you used to answer the question.

Context:
{context}

Question: {question}

Helpful Answer:"""
            
            self.rag_prompt = PromptTemplate(
                template=rag_prompt_template,
                input_variables=["context", "question"]
            )
            
            logger.info("RAG chain components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to create RAG chain: {str(e)}")
            self.rag_prompt = None
    
    def process_query(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Process a user query using RAG pipeline or direct LLM interaction.
        
        Args:
            question: The user's question
            use_rag: Whether to use RAG (retrieval-augmented generation)
            
        Returns:
            Dictionary containing response, sources, and metadata
        """
        try:
            if use_rag and self.vector_store_manager.is_initialized():
                return self._process_rag_query(question)
            else:
                return self._process_direct_query(question)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"Sorry, I encountered an error processing your question: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _process_rag_query(self, question: str) -> Dict[str, Any]:
        """
        Process query using RAG pipeline with document retrieval.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        try:
            # Perform similarity search to get relevant documents
            retrieved_docs = self.vector_store_manager.similarity_search(
                question, 
                k=SIMILARITY_SEARCH_K
            )
            
            if not retrieved_docs:
                return {
                    "response": "I don't have any relevant information in the uploaded documents to answer your question.",
                    "sources": [],
                    "retrieved_docs": 0
                }
            
            # Format retrieved chunks as context
            context = self._format_context_from_docs(retrieved_docs)
            
            # Generate response using RAG prompt
            formatted_prompt = self.rag_prompt.format(
                context=context,
                question=question
            )
            
            message = HumanMessage(content=formatted_prompt)
            response = self.llm.invoke([message])
            
            # Extract response text
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            # Extract source information
            sources = self._extract_sources_from_docs(retrieved_docs)
            
            # Update chat history
            self._update_chat_history(question, answer)
            
            return {
                "response": answer,
                "sources": sources,
                "retrieved_docs": len(retrieved_docs),
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query processing: {str(e)}")
            return {
                "response": f"Sorry, I encountered an error during document retrieval: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _process_direct_query(self, question: str) -> Dict[str, Any]:
        """
        Process query directly with LLM without document retrieval.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            message = HumanMessage(content=question)
            response = self.llm.invoke([message])
            
            # Extract response text
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            # Update chat history
            self._update_chat_history(question, answer)
            
            return {
                "response": answer,
                "sources": [],
                "mode": "direct"
            }
            
        except Exception as e:
            logger.error(f"Error in direct query processing: {str(e)}")
            return {
                "response": f"Sorry, I encountered an error: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _update_chat_history(self, question: str, answer: str) -> None:
        """
        Update the chat history with the latest question and answer.
        
        Args:
            question: The user's question
            answer: The LLM's answer
        """
        # Add new messages
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        
        # Trim history if it exceeds maximum length
        if len(self.chat_history) > MAX_CHAT_HISTORY * 2:  # *2 because each exchange has 2 messages
            self.chat_history = self.chat_history[-MAX_CHAT_HISTORY * 2:]
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the chat history in a format suitable for display.
        
        Returns:
            List of dictionaries with 'role' and 'content' keys
        """
        history = []
        for message in self.chat_history:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        
        return history
    
    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
        logger.info("Chat history cleared")
    
    def _format_context_from_docs(self, docs: List[Document]) -> str:
        """
        Format retrieved documents as context for the LLM prompt.
        
        Args:
            docs: List of retrieved Document objects
            
        Returns:
            Formatted context string
        """
        if not docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Extract metadata for source attribution
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page_number', 'Unknown')
            
            # Format each document chunk
            context_part = f"[Source {i}: {source}, Page {page}]\n{doc.page_content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _extract_sources_from_docs(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract source information from retrieved documents.
        
        Args:
            docs: List of retrieved Document objects
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        for doc in docs:
            source_info = {
                "source": doc.metadata.get('source', 'Unknown'),
                "page_number": doc.metadata.get('page_number', 'Unknown'),
                "chunk_id": doc.metadata.get('chunk_id', 'Unknown'),
                "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source_info)
        
        return sources
    
    def format_response_with_sources(self, response_data: Dict[str, Any]) -> str:
        """
        Format response with source attribution for display.
        
        Args:
            response_data: Response data from process_query
            
        Returns:
            Formatted response string with sources
        """
        response = response_data.get("response", "")
        sources = response_data.get("sources", [])
        
        if not sources:
            return response
        
        # Add source attribution
        formatted_response = response + "\n\n**Sources:**\n"
        for i, source in enumerate(sources, 1):
            source_line = f"{i}. {source['source']}"
            if source['page_number'] != 'Unknown':
                source_line += f" (Page {source['page_number']})"
            formatted_response += source_line + "\n"
        
        return formatted_response
    
    def is_available(self) -> bool:
        """
        Check if the LLM service is available.
        
        Returns:
            True if the LLM is available, False otherwise
        """
        try:
            if self.llm is None:
                return False
            
            # Simple test query
            test_response = self.llm.invoke([HumanMessage(content="test")])
            return True
            
        except Exception as e:
            logger.warning(f"LLM availability check failed: {str(e)}")
            return False
    
    def is_rag_ready(self) -> bool:
        """
        Check if RAG pipeline is ready (LLM + vector store available).
        
        Returns:
            True if RAG is ready, False otherwise
        """
        return (
            self.is_available() and 
            self.vector_store_manager.is_initialized() and
            self.vector_store_manager.get_collection_count() > 0
        )
    
    def get_rag_status(self) -> Dict[str, Any]:
        """
        Get detailed status of RAG pipeline components.
        
        Returns:
            Dictionary with status information
        """
        return {
            "llm_available": self.is_available(),
            "vector_store_initialized": self.vector_store_manager.is_initialized(),
            "documents_loaded": self.vector_store_manager.get_collection_count(),
            "rag_ready": self.is_rag_ready(),
            "chat_history_length": len(self.chat_history)
        }


def initialize_llm() -> ChatEngine:
    """
    Initialize and return a ChatEngine instance.
    
    Returns:
        Configured ChatEngine instance
        
    Raises:
        ConnectionError: If Ollama service is not available
    """
    return ChatEngine()


def create_rag_chain(chat_engine: ChatEngine) -> bool:
    """
    Create and configure the complete RAG chain.
    
    Args:
        chat_engine: Initialized ChatEngine instance
        
    Returns:
        True if RAG chain created successfully, False otherwise
    """
    try:
        # Ensure vector store is initialized
        if not chat_engine.vector_store_manager.is_initialized():
            success = chat_engine.vector_store_manager.initialize_vector_store()
            if not success:
                logger.error("Failed to initialize vector store for RAG chain")
                return False
        
        # Verify LLM is available
        if not chat_engine.is_available():
            logger.error("LLM is not available for RAG chain")
            return False
        
        # RAG chain components are already created in _create_rag_chain
        logger.info("RAG chain created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create RAG chain: {str(e)}")
        return False


def initialize_rag_system() -> Tuple[ChatEngine, bool]:
    """
    Initialize complete RAG system with LLM and vector store.
    
    Returns:
        Tuple of (ChatEngine instance, success status)
    """
    try:
        # Initialize chat engine
        chat_engine = initialize_llm()
        
        # Initialize vector store
        vector_store_success = chat_engine.vector_store_manager.initialize_vector_store()
        
        if not vector_store_success:
            logger.warning("Vector store initialization failed - RAG will not be available")
            return chat_engine, False
        
        # Create RAG chain
        rag_success = create_rag_chain(chat_engine)
        
        if rag_success:
            logger.info("Complete RAG system initialized successfully")
        else:
            logger.warning("RAG chain creation failed - falling back to direct LLM mode")
        
        return chat_engine, rag_success
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise