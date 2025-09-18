# DocQueryAI - Local PDF Chatbot

A powerful local PDF chatbot application that enables users to upload PDF documents and query them using natural language. Built with Streamlit, LangChain, and Ollama for complete privacy and local processing.

![DocQueryAI Demo](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

- **📄 PDF Document Processing**: Upload and process PDF documents with intelligent text extraction
- **🧠 Local LLM Integration**: Uses Ollama with Phi-3 Mini for completely local AI processing
- **🔍 RAG (Retrieval-Augmented Generation)**: Advanced document querying with context-aware responses
- **💬 Interactive Chat Interface**: Natural language conversations about your documents
- **📚 Source Attribution**: Every answer includes citations to specific document sections
- **🔒 Complete Privacy**: All processing happens locally - no data leaves your machine
- **⚡ Real-time Processing**: Efficient document chunking and embedding generation
- **🎯 Smart Search**: Vector-based similarity search for relevant content retrieval

## 🛠️ Technology Stack

- **Frontend**: Streamlit for web interface
- **Backend**: Python with LangChain for document processing
- **LLM**: Ollama (Phi-3 Mini model)
- **Embeddings**: Nomic Embed Text model
- **Vector Database**: Chroma for document storage and retrieval
- **PDF Processing**: PyPDF2 for text extraction
- **Text Processing**: LangChain's RecursiveCharacterTextSplitter

## 📋 Prerequisites

- **Python 3.9+**
- **Ollama** installed and running
- **Git** (for cloning the repository)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/DocQueryAI.git
cd DocQueryAI
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Start Ollama

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Start Ollama service
ollama serve &

# Install required models
ollama pull phi3:mini
ollama pull nomic-embed-text
```

### 5. Run the Application

```bash
# Option 1: Use the startup script (recommended)
./run_app.sh

# Option 2: Manual start
streamlit run app.py --server.headless true --server.port 8501 &
```

### 6. Access the Application

Open your browser and navigate to: `http://localhost:8501`

## 📖 How to Use

1. **Upload Document**:

   - Go to the "Upload Document" tab
   - Select a PDF file (max 50MB)
   - Click "Process Document"

2. **Wait for Processing**:

   - The app will extract text, create chunks, and generate embeddings
   - Progress is shown with detailed status updates

3. **Ask Questions**:

   - Switch to the "Chat" tab
   - Ask natural language questions about your document
   - Review answers with source citations

4. **Explore Features**:
   - Check the sidebar for system status
   - Use the clear data option to process new documents
   - Monitor chat history during your session

## 🧪 Testing

The project includes comprehensive tests:

```bash
# Run basic functionality tests
python test_basic_functionality.py

# Run end-to-end workflow tests
python test_end_to_end.py

# Run vector store tests
python test_vector_store_fix.py
```

## 📁 Project Structure

```
DocQueryAI/
├── app.py                      # Main Streamlit application
├── chat_engine.py             # LLM and RAG pipeline
├── vector_store.py            # Vector database management
├── document_processor.py      # PDF processing utilities
├── text_chunker.py           # Text chunking and metadata
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── run_app.sh               # Application startup script
├── stop_app.sh              # Application stop script
├── activate_env.sh          # Environment activation script
├── tests/                   # Test files
│   ├── test_basic_functionality.py
│   ├── test_end_to_end.py
│   └── test_vector_store_fix.py
├── .kiro/specs/            # Project specifications
└── README.md               # This file
```

## ⚙️ Configuration

Key configuration options in `config.py`:

- **OLLAMA_MODEL**: LLM model (default: "phi3:mini")
- **EMBEDDING_MODEL**: Embedding model (default: "nomic-embed-text")
- **CHUNK_SIZE**: Text chunk size (default: 1000)
- **CHUNK_OVERLAP**: Chunk overlap (default: 200)
- **MAX_FILE_SIZE_MB**: Maximum PDF size (default: 50MB)

## 🔧 Troubleshooting

### Common Issues

1. **Ollama Connection Error**:

   ```bash
   # Ensure Ollama is running
   ollama serve &

   # Check if models are installed
   ollama list
   ```

2. **Model Not Found**:

   ```bash
   # Install required models
   ollama pull phi3:mini
   ollama pull nomic-embed-text
   ```

3. **Port Already in Use**:

   ```bash
   # Stop existing processes
   ./stop_app.sh

   # Or manually kill processes
   pkill -f "streamlit run app.py"
   ```

4. **Memory Issues**:
   - Try smaller PDF files (under 10MB)
   - Restart Ollama service
   - Check available system memory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** for providing local LLM capabilities
- **LangChain** for document processing framework
- **Streamlit** for the web interface
- **Chroma** for vector database functionality
- **Microsoft** for the Phi-3 Mini model

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review the test files for examples
3. Open an issue on GitHub
4. Check Ollama documentation for model-specific issues

## 🔮 Future Enhancements

- [ ] Support for multiple document formats (Word, PowerPoint, etc.)
- [ ] Advanced document management and organization
- [ ] Custom model fine-tuning capabilities
- [ ] Multi-language support
- [ ] Document comparison features
- [ ] Export conversation history
- [ ] Advanced search filters and options

---

**Built with ❤️ for local AI and privacy-focused document processing**
