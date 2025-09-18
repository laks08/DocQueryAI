# Configuration file with constants and settings for DocQueryAI

# Ollama Configuration
OLLAMA_MODEL = "phi3:mini"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Text Processing Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Database Configuration
VECTOR_DB_PATH = "./chroma_db"

# Chat Configuration
MAX_CHAT_HISTORY = 10

# File Upload Configuration
MAX_FILE_SIZE_MB = 50
ALLOWED_FILE_TYPES = [".pdf"]

# UI Configuration
APP_TITLE = "DocQueryAI - PDF Chatbot"
APP_DESCRIPTION = "Upload PDF documents and query them using natural language"

# Processing Configuration
MAX_CHUNKS_PER_DOCUMENT = 1000
SIMILARITY_SEARCH_K = 4

# LLM Configuration
LLM_TEMPERATURE = 0.1