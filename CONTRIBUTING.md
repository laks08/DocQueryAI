# Contributing to DocQueryAI

Thank you for your interest in contributing to DocQueryAI! This document provides guidelines and information for contributors.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/DocQueryAI.git
   cd DocQueryAI
   ```
3. **Set up the development environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## 🛠️ Development Setup

### Prerequisites

- Python 3.9+
- Ollama installed and running
- Git

### Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if any)
pip install pytest black flake8 mypy

# Start Ollama
ollama serve &

# Install required models
ollama pull phi3:mini
ollama pull nomic-embed-text
```

## 🧪 Testing

Before submitting any changes, make sure all tests pass:

```bash
# Run basic functionality tests
python test_basic_functionality.py

# Run end-to-end tests
python test_end_to_end.py

# Run vector store tests
python test_vector_store_fix.py

# Test the application manually
./run_app.sh
```

## 📝 Code Style

We follow Python best practices:

- **PEP 8** for code formatting
- **Type hints** where appropriate
- **Docstrings** for all functions and classes
- **Meaningful variable names**
- **Comments** for complex logic

### Code Formatting

```bash
# Format code with black (if installed)
black *.py

# Check with flake8 (if installed)
flake8 *.py
```

## 🔄 Making Changes

1. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:

   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**:

   ```bash
   # Run all tests
   python test_basic_functionality.py
   python test_end_to_end.py

   # Test the application
   ./run_app.sh
   ```

4. **Commit your changes**:

   ```bash
   git add .
   git commit -m "Add: brief description of your changes"
   ```

5. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] All tests pass
- [ ] Code follows the project style
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive

### Pull Request Description

Please include:

- **What** changes you made
- **Why** you made these changes
- **How** to test the changes
- **Screenshots** (if UI changes)

### Example PR Description

```markdown
## Changes Made

- Added support for Word documents
- Implemented new document processor for .docx files
- Updated UI to show supported file types

## Why

Users requested support for Word documents in addition to PDFs.

## Testing

- Added tests in test_document_processor.py
- Tested with various .docx files
- Verified UI updates work correctly

## Screenshots

[Include screenshots if applicable]
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description** of the bug
2. **Steps to reproduce**
3. **Expected behavior**
4. **Actual behavior**
5. **Environment details**:
   - OS (macOS, Windows, Linux)
   - Python version
   - Ollama version
   - Browser (if applicable)

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**

1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**

- OS: [e.g. macOS 14.0]
- Python: [e.g. 3.9.7]
- Ollama: [e.g. 0.1.0]
- Browser: [e.g. Chrome 120.0]
```

## 💡 Feature Requests

We welcome feature requests! Please:

1. **Check existing issues** to avoid duplicates
2. **Describe the feature** clearly
3. **Explain the use case**
4. **Consider implementation** if possible

## 🏗️ Project Structure

Understanding the codebase:

```
DocQueryAI/
├── app.py                 # Main Streamlit application
├── chat_engine.py        # LLM and RAG pipeline
├── vector_store.py       # Vector database management
├── document_processor.py # PDF processing utilities
├── text_chunker.py      # Text chunking and metadata
├── config.py            # Configuration settings
└── tests/               # Test files
```

### Key Components

- **app.py**: Main application entry point and UI
- **chat_engine.py**: Handles LLM interactions and RAG pipeline
- **vector_store.py**: Manages Chroma database and embeddings
- **document_processor.py**: PDF text extraction and validation
- **text_chunker.py**: Text splitting and metadata creation

## 🎯 Areas for Contribution

We especially welcome contributions in:

- **New document formats** (Word, PowerPoint, etc.)
- **UI/UX improvements**
- **Performance optimizations**
- **Additional LLM models**
- **Better error handling**
- **Documentation improvements**
- **Test coverage**
- **Accessibility features**

## 📞 Getting Help

If you need help:

1. **Check the README** for basic setup
2. **Review existing issues** on GitHub
3. **Run the tests** to understand expected behavior
4. **Ask questions** in GitHub Discussions
5. **Open an issue** for bugs or feature requests

## 🙏 Recognition

Contributors will be:

- Listed in the README
- Mentioned in release notes
- Credited in commit history

Thank you for contributing to DocQueryAI! 🚀
