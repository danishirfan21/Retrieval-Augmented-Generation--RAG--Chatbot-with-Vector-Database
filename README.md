# RAG Financial Chatbot with Vector Database

A production-ready Retrieval-Augmented Generation (RAG) chatbot for answering domain-specific questions on financial reports. Built with Python, FastAPI, LangChain, Pinecone, and OpenAI.

## Features

- **RAG Architecture**: Combines vector database retrieval with LLM generation
- **Vector Database**: Pinecone for efficient similarity search
- **Embeddings**: Sentence-transformers (all-MiniLM-L6-v2) for text vectorization
- **LLM Integration**: OpenAI GPT-3.5/4 for natural language generation
- **RESTful API**: FastAPI with automatic documentation
- **Docker Support**: Containerized deployment
- **Scalable Design**: Modular architecture for easy extension

## Architecture Overview

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   FastAPI       │
│   Endpoints     │
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│   RAG Chain        │
│  (LangChain)       │
└─────┬───────┬──────┘
      │       │
      ▼       ▼
┌──────────┐ ┌──────────────┐
│ Pinecone │ │ OpenAI API   │
│ Retrieval│ │ Generation   │
└──────────┘ └──────────────┘
```

## Tech Stack

- **Backend**: Python 3.10+, FastAPI
- **RAG Framework**: LangChain
- **Vector Database**: Pinecone
- **Embeddings**: Sentence-Transformers (HuggingFace)
- **LLM**: OpenAI GPT-3.5-turbo
- **Containerization**: Docker & Docker Compose
- **Documentation**: OpenAPI/Swagger

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py           # API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   └── embeddings.py       # Embedding generation
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py        # Pinecone retriever
│   │   └── chain.py            # RAG chain logic
│   └── services/
│       ├── __init__.py
│       └── pinecone_service.py # Pinecone operations
├── data/
│   └── sample_docs/            # Sample financial documents
├── notebooks/
│   └── demo.ipynb              # Jupyter demo notebook
├── scripts/
│   └── ingest_documents.py     # Document ingestion script
├── .env.example                # Example environment variables
├── .gitignore
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Docker Compose configuration
└── README.md
```

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key
- Pinecone API key (free tier available)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Retrieval-Augmented Generation (RAG) Chatbot with Vector Database"
```

### 2. Set Up Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=financial-rag-chatbot
```

### 3. Install Dependencies

#### Option A: Local Installation

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Docker Installation

```bash
# Build Docker image
docker-compose build
```

## Usage

### Step 1: Ingest Documents

Before using the chatbot, you need to ingest documents into the vector database.

#### Local:

```bash
python scripts/ingest_documents.py
```

This will:
1. Load documents from `data/sample_docs/`
2. Split them into chunks
3. Generate embeddings using sentence-transformers
4. Upload vectors to Pinecone

### Step 2: Start the API Server

#### Option A: Local

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Option B: Docker

```bash
docker-compose up
```

The API will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Step 3: Query the Chatbot

#### Using cURL:

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was TechCorp'\''s revenue in Q1 2024?",
    "top_k": 5
  }'
```

#### Using Python:

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "question": "What was TechCorp's revenue in Q1 2024?",
        "top_k": 5
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

#### Using the Interactive API Docs:

1. Navigate to http://localhost:8000/docs
2. Click on the `/api/v1/query` endpoint
3. Click "Try it out"
4. Enter your question and click "Execute"

## API Endpoints

### Health Check

```
GET /api/v1/health
```

Returns API health status.

### Query (Simple)

```
POST /api/v1/query
```

**Request Body:**
```json
{
  "question": "What is the gross margin?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What is the gross margin?",
  "answer": "According to the Q1 2024 report, TechCorp's gross margin was 68%...",
  "sources": ["quarterly_report_q1_2024.txt"],
  "retrieved_docs": [...]
}
```

### Chat (With History)

```
POST /api/v1/chat
```

**Request Body:**
```json
{
  "question": "What about the operating margin?",
  "chat_history": [
    {
      "question": "What was the revenue?",
      "answer": "The revenue was $450 million..."
    }
  ],
  "top_k": 5
}
```

### Index Statistics

```
GET /api/v1/stats
```

Returns Pinecone index statistics (document count, dimension, etc.).

## Configuration

All configuration is managed through environment variables. See `.env.example` for available options:

- **API Keys**: OpenAI and Pinecone credentials
- **Model Settings**: Embedding model, LLM model, temperature, max tokens
- **Retrieval Settings**: Top-K, chunk size, chunk overlap
- **API Settings**: Host, port

## Sample Questions

Try these questions with the sample financial documents:

1. "What was TechCorp's total revenue in Q1 2024?"
2. "What is the net revenue retention rate?"
3. "How much did TechCorp invest in R&D in 2023?"
4. "What are the main risk factors mentioned in the report?"
5. "What is the customer acquisition cost?"
6. "Explain the Rule of 40 metric and TechCorp's score"
7. "What are TechCorp's strategic priorities for 2024?"

## Adding Your Own Documents

1. Place your documents (TXT or PDF) in `data/sample_docs/`
2. Run the ingestion script:
   ```bash
   python scripts/ingest_documents.py
   ```
3. Query the API with domain-specific questions

## Development

### Running Tests

```bash
pytest
```

### Jupyter Notebook Demo

Explore the RAG system interactively:

```bash
jupyter notebook notebooks/demo.ipynb
```

### Code Structure

- **app/core/**: Core utilities (config, embeddings)
- **app/services/**: External service integrations (Pinecone)
- **app/rag/**: RAG-specific logic (retrieval, chain)
- **app/api/**: API routes and models
- **scripts/**: Utility scripts (ingestion)

## Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Production Considerations

1. **Security**:
   - Use secrets management (AWS Secrets Manager, HashiCorp Vault)
   - Enable HTTPS/TLS
   - Implement rate limiting
   - Add authentication/authorization

2. **Scalability**:
   - Use multiple Uvicorn workers
   - Deploy behind load balancer (NGINX, AWS ALB)
   - Consider caching layer (Redis)
   - Monitor with Prometheus/Grafana

3. **Monitoring**:
   - Add logging (structured logging with JSON)
   - Track metrics (response time, error rate)
   - Set up alerts

## Troubleshooting

### Common Issues

**Issue**: "Index not found" error
- **Solution**: Run the ingestion script first to create and populate the index

**Issue**: OpenAI API rate limits
- **Solution**: Implement exponential backoff, reduce max_tokens, or upgrade API tier

**Issue**: Slow response times
- **Solution**: Reduce top_k, use smaller embedding model, or cache frequent queries

**Issue**: Out of memory during ingestion
- **Solution**: Process documents in smaller batches, reduce chunk_size

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance

- **Embedding Generation**: ~100 chunks/second (CPU)
- **Query Response Time**: ~2-3 seconds (including LLM)
- **Index Size**: ~1KB per document chunk
- **Concurrent Requests**: Supports multiple concurrent queries

## Windows quick setup (PowerShell)

## Run frontend separately (optional)

You can serve the minimal web UI independently from the backend and point it at your API.

1) Start the backend (terminal A):

  - C:\rag_venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

2) Start the frontend (terminal B):

  - C:\rag_venv\Scripts\python.exe -m http.server 5500 --directory app/static

3) Open the UI and pass the API base as a query param:

  - http://127.0.0.1:5500/index.html?api=http://127.0.0.1:8000

Notes:
- If port 8000 is in use, change the backend port (e.g., 8010) and update the api parameter accordingly.
- CORS is enabled for development (allow_origins=["*"]). Adjust for production.

If you're on Windows and prefer a small helper script, a `scripts/setup.ps1` helper is included to automate common steps (create venv, install deps, copy `.env.example`). After running that, you can optionally ingest documents and start the server.

Basic commands (PowerShell):

```powershell
# Copy example env and edit .env with your API keys
Copy-Item .env.example .env
notepad .env

# Run the setup helper (installs deps and creates venv)
.\scripts\setup.ps1

# Run setup + ingest documents + run server in foreground
.\scripts\setup.ps1 -Ingest -RunServer

# If you prefer to start the server in the background (uses venv python if present)
.\scripts\start_server.ps1

# If using Docker
docker-compose up --build -d
```

Notes:
- Edit `.env` before running ingestion to ensure OpenAI and Pinecone keys are present.
- If `sentence-transformers` or `torch` fails to install via pip, install a compatible `torch` wheel first (CPU wheel works for local dev).
- The setup script is intended for development convenience; in production use appropriate secrets management.

## Limitations

- Requires API keys (OpenAI, Pinecone)
- OpenAI API has usage costs
- Context window limited by LLM (4K-8K tokens)
- Retrieval accuracy depends on embedding quality

## Future Enhancements

- [ ] Add support for more document types (DOCX, HTML)
- [ ] Implement hybrid search (vector + keyword)
- [ ] Add response streaming
- [ ] Create web UI with React
- [ ] Support multiple vector databases
- [ ] Add evaluation metrics (RAGAS)
- [ ] Implement user feedback loop
- [ ] Add multilingual support

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **LangChain**: RAG orchestration framework
- **Pinecone**: Vector database platform
- **OpenAI**: Language model API
- **Sentence-Transformers**: Embedding models
- **FastAPI**: Modern Python web framework

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

Built with ❤️ for demonstrating RAG systems, vector databases, and modern ML engineering practices.
