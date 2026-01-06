# PDF-RAG Backend

A FastAPI backend for the PDF-RAG application. Provides RAG-powered Q&A with precision citations, returning exact page numbers and bounding box coordinates for source text.

## 🚀 Tech Stack

| Category | Technology |
|----------|------------|
| **Framework** | FastAPI |
| **Language** | Python 3.10+ |
| **Orchestration** | LlamaIndex |
| **Database** | PostgreSQL (with pgvector) |
| **PDF Parsing** | PyMuPDF (fitz) |
| **Validation** | Pydantic |
| **Server** | Uvicorn |

## 📋 Prerequisites

- **Python** 3.10+
- **Docker** (for PostgreSQL with pgvector)
- **OpenAI API Key**

## ⚙️ Installation

### Option 1: Local Development

1. **Clone and navigate:**
   ```bash
   cd Backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   .\venv\Scripts\activate   # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

5. **Start PostgreSQL with pgvector:**
   ```bash
   docker run -d --name pdfrag-db \
     -e POSTGRES_USER=postgres \
     -e POSTGRES_PASSWORD=postgres \
     -e POSTGRES_DB=pdfrag \
     -p 5432:5432 \
     pgvector/pgvector:pg16
   ```

6. **Run the server:**
   ```bash
   uvicorn app.main:app --reload
   ```

### Option 2: Docker Compose (Recommended)

1. **Set environment variable:**
   ```bash
   export OPENAI_API_KEY=sk-your-key-here
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

The API will be available at [http://localhost:8000](http://localhost:8000).

## 📚 API Documentation

Once running, access interactive API docs:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health/` | GET | Health check |
| `/api/documents/upload` | POST | Upload PDF |
| `/api/documents/` | GET | List all documents |
| `/api/documents/{id}` | GET | Get document |
| `/api/documents/{id}` | DELETE | Delete document |
| `/api/chat/` | POST | Send chat message |

## 📂 Project Structure

```
Backend/
├── app/
│   ├── __init__.py
│   ├── main.py               # FastAPI application entry
│   ├── config.py             # Pydantic settings
│   ├── api/
│   │   ├── deps.py           # Dependencies
│   │   └── routes/
│   │       ├── health.py     # Health endpoints
│   │       ├── documents.py  # Document CRUD
│   │       └── chat.py       # RAG Q&A
│   ├── core/
│   │   ├── pdf_parser.py     # PyMuPDF extraction
│   │   └── rag_engine.py     # LlamaIndex RAG
│   ├── db/
│   │   ├── database.py       # SQLAlchemy setup
│   │   └── models.py         # ORM models
│   ├── models/
│   │   └── schemas.py        # Pydantic schemas
│   └── services/
│       ├── document_service.py
│       └── chat_service.py
├── tests/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | **Required.** Your OpenAI API key |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection URL |
| `OPENAI_MODEL` | `gpt-4o` | LLM model for generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `512` | Text chunk size for indexing |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `SIMILARITY_TOP_K` | `5` | Number of chunks to retrieve |

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=app
```

## 🐳 Docker Commands

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run linting: `ruff check .`
4. Run tests: `pytest`
5. Commit with clear messages
6. Push and create a Pull Request

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
