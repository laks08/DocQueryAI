# Design Document

## Overview

DocQueryAI is a local PDF chatbot built with Streamlit that enables natural language querying of PDF documents. The system uses a RAG (Retrieval-Augmented Generation) architecture with local components: Ollama for LLM inference and embeddings, Chroma for vector storage, and LangChain for orchestration.

## Architecture

The application follows a simple pipeline architecture:

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Storage → Query Interface → Retrieval → LLM Response
```

### Core Components:

- **Frontend**: Streamlit web interface
- **Document Processing**: PyPDF2/pdfplumber for text extraction
- **Text Processing**: LangChain text splitters for chunking
- **Embeddings**: Ollama embedding models
- **Vector Database**: Chroma (local persistence)
- **LLM**: Ollama with Phi-3 Mini model
- **Orchestration**: LangChain for RAG pipeline

## Components and Interfaces

### 1. Streamlit Frontend (`app.py`)

**Purpose**: Web interface for file upload and chat interaction

**Key Functions**:

- File upload widget for PDF selection
- Chat interface with message history
- Progress indicators for processing
- Error message display

**State Management**:

- `st.session_state` for chat history
- Document processing status
- Current document metadata

### 2. Document Processor (`document_processor.py`)

**Purpose**: Handle PDF text extraction and preprocessing

**Key Functions**:

```python
def extract_text_from_pdf(pdf_file) -> str
def validate_pdf(pdf_file) -> bool
def clean_text(raw_text: str) -> str
```

**Dependencies**: PyPDF2 or pdfplumber for PDF parsing

### 3. Text Chunker (`text_chunker.py`)

**Purpose**: Split documents into manageable chunks for embedding

**Key Functions**:

```python
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]
def create_metadata(chunks: List[str], source: str) -> List[dict]
```

**Strategy**: Use LangChain's RecursiveCharacterTextSplitter with configurable chunk size and overlap

### 4. Vector Store Manager (`vector_store.py`)

**Purpose**: Handle Chroma database operations and embeddings

**Key Functions**:

```python
def initialize_vector_store() -> Chroma
def add_documents(chunks: List[str], metadata: List[dict]) -> None
def similarity_search(query: str, k: int = 4) -> List[Document]
def clear_store() -> None
```

**Configuration**:

- Local Chroma persistence directory
- Ollama embeddings integration
- Configurable similarity search parameters

### 5. Chat Engine (`chat_engine.py`)

**Purpose**: Orchestrate RAG pipeline and LLM interaction

**Key Functions**:

```python
def initialize_llm() -> ChatOllama
def create_rag_chain() -> Chain
def process_query(question: str, chat_history: List) -> str
def format_response(response: str, sources: List[Document]) -> str
```

**LLM Configuration**:

- Model: `phi3:mini`
- Temperature: 0.1 (for consistent responses)
- Context window management

### 6. Configuration Manager (`config.py`)

**Purpose**: Centralized configuration and constants

**Settings**:

```python
OLLAMA_MODEL = "phi3:mini"
EMBEDDING_MODEL = "nomic-embed-text"  # or sentence-transformers fallback
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_DB_PATH = "./chroma_db"
MAX_CHAT_HISTORY = 10
```

## Data Models

### Document Chunk

```python
@dataclass
class DocumentChunk:
    content: str
    metadata: dict
    chunk_id: str
    source: str
    page_number: Optional[int]
```

### Chat Message

```python
@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = None
```

### Processing Status

```python
class ProcessingStatus(Enum):
    IDLE = "idle"
    UPLOADING = "uploading"
    EXTRACTING = "extracting_text"
    CHUNKING = "chunking"
    EMBEDDING = "creating_embeddings"
    READY = "ready"
    ERROR = "error"
```

## Error Handling

### Error Categories:

1. **File Upload Errors**: Invalid PDF, file size limits, corrupted files
2. **Processing Errors**: Text extraction failures, chunking issues
3. **Ollama Connection Errors**: Service unavailable, model not found
4. **Vector Store Errors**: Database connection, embedding failures
5. **Query Errors**: Empty results, LLM generation failures

### Error Handling Strategy:

- Graceful degradation with user-friendly messages
- Retry mechanisms for transient failures
- Logging for debugging (using Python logging module)
- Clear error states in Streamlit UI

### Error Recovery:

```python
def handle_ollama_error(error: Exception) -> str:
    if "connection" in str(error).lower():
        return "Ollama service is not running. Please start Ollama and try again."
    elif "model" in str(error).lower():
        return "Phi-3 Mini model not found. Please ensure it's installed."
    else:
        return f"LLM error: {str(error)}"
```

## Testing Strategy

### Manual Testing:

- Upload a sample PDF and verify text extraction
- Test basic question-answer flow
- Verify error handling with invalid files
- Check chat history functionality

## Performance Considerations

### Optimization Strategies:

1. **Lazy Loading**: Initialize components only when needed
2. **Caching**: Use `@st.cache_data` for expensive operations
3. **Chunking Strategy**: Balance chunk size vs. retrieval accuracy
4. **Memory Management**: Clear unused embeddings and chat history

### Resource Limits:

- Maximum PDF size: 50MB
- Maximum chunks per document: 1000
- Chat history limit: 10 messages
- Vector store cleanup after session

### Monitoring:

- Processing time metrics
- Memory usage tracking
- Error rate monitoring
- User interaction analytics (local only)

## Security and Privacy

### File Validation:

- Check file is valid PDF format
- Limit file size to 50MB maximum
- All processing happens locally (no external API calls)
