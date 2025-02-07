# Local LLM RAG Application

## System Architecture
![System Architecture](assets/rag_architecture.png)

### Components:
1. **Document Processing**
   - Supports PDF and HTML documents
   - Uses LangChain for document loading and chunking
   - Implements RecursiveCharacterTextSplitter for optimal chunk sizes

2. **Vector Database**
   - Uses ChromaDB for efficient vector storage
   - Implements sentence-transformers for embeddings
   - Supports collection management and querying

3. **Local LLM Integration**
   - Uses Llama 2 (7B quantized model)
   - GPU acceleration support
   - Custom prompt templates for improved responses

4. **API Layer**
   - FastAPI framework
   - RESTful endpoints
   - Swagger documentation

## Setup Instructions

### Prerequisites
- Python 3.12
- Docker and Docker Compose
- NVIDIA GPU (optional, for acceleration)

### Local Development
```bash
# Clone repository
git clone https://github.com/yourusername/rag-project.git
cd rag-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r src/requirements.txt

# Download LLM
# Place llama-2-7b.Q4_K_M.gguf in src/models/

# Run application
uvicorn src.rag_app.main:app --reload
```

## RAG Implementation Choices
1. **Document Chunking**
    - Chunk size: 1000 characters
    - Overlap: 100 characters
    - Rationale: Balances context preservation with query relevance

2. **Vector Database**
    - Model: all-MiniLM-L6-v2
    - Rationale: Good performance/speed trade-off for semantic search

3. **LLM Configuration**
    - Context window: 4096 tokens
    - Temperature: 0.7
    - Rationale: Balanced between creativity and accuracy

## Future Improvements

1. **Technical Enhancements** 
    - Implement batch processing for large documents
    - Add support for more document types
    - Improve error handling and logging


2. **Features**
    - Add document summarization
    - Implement user authentication
    - Add support for multiple LLM models


3. **Performance**
    - Optimize chunking strategy
    - Implement caching
    - Add load balancing for multiple users
