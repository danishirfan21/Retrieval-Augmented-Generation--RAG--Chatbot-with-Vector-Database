# Quick Start Guide

Get your RAG Financial Chatbot running in 5 minutes!

## Prerequisites

- Python 3.10+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Pinecone API key ([Get free tier here](https://www.pinecone.io/))

## Step-by-Step Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY=sk-...
# - PINECONE_API_KEY=...
# - PINECONE_ENVIRONMENT=gcp-starter (or your environment)
```

### 3. Setup Pinecone Index

```bash
python scripts/setup_pinecone.py
```

This creates your vector database index.

### 4. Ingest Sample Documents

```bash
python scripts/ingest_documents.py
```

This processes the sample financial documents and uploads them to Pinecone.

### 5. Start the API Server

```bash
uvicorn app.main:app --reload
```

Server will start at http://localhost:8000

### 6. Test It Out!

Open your browser and go to:
- **Interactive Docs**: http://localhost:8000/docs

Try this query:
```json
{
  "question": "What was TechCorp's revenue in Q1 2024?",
  "top_k": 5
}
```

## Quick Test with cURL

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was TechCorps revenue in Q1 2024?", "top_k": 5}'
```

## Using Docker (Alternative)

```bash
# Set environment variables in .env file first!

# Build and run
docker-compose up

# API available at http://localhost:8000
```

## Next Steps

1. **Add Your Documents**: Place PDFs or TXT files in `data/sample_docs/`
2. **Re-run Ingestion**: `python scripts/ingest_documents.py`
3. **Explore Notebook**: `jupyter notebook notebooks/demo.ipynb`
4. **Customize**: Modify settings in `.env` file

## Common Issues

**"Index not found"**: Run `python scripts/setup_pinecone.py` first

**"Invalid API key"**: Check your `.env` file has correct keys

**Slow responses**: Normal for first query (model loading). Subsequent queries are faster.

## Sample Questions to Try

- "What was TechCorp's total revenue in Q1 2024?"
- "What is the net revenue retention rate?"
- "What are the main risk factors?"
- "How much was invested in R&D?"
- "What are the strategic priorities for 2024?"

## Resources

- Full documentation: See README.md
- API docs: http://localhost:8000/docs
- Notebook demo: `notebooks/demo.ipynb`

Need help? Check the README.md or create an issue!
