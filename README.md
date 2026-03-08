# Semantic Search Engine

A semantic search engine using ChromaDB for vector storage and Sentence Transformers for embeddings.

## Project Structure

```text
.
├── backend/                # FastAPI backend
│   ├── app.py             # Main API application
│   └── semantic_cache.py  # Semantic caching logic
├── frontend/               # React (Vite) frontend
│   └── ...
├── scripts/                # Data processing scripts
│   ├── phase1_embed.py    # Embedding generation
│   └── phase2_cluster.py  # Clustering/indexing
├── data/                   # Data storage
│   ├── dataset/           # Raw datasets
│   └── chroma_store/      # ChromaDB vector database
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Multi-container setup
├── requirements.txt        # Python dependencies
├── run.bat                 # Backend execution script
└── process_data.bat        # Data processing script
```

## How to Execute

### Option 1: Using Docker (Recommended)

This method sets up both the backend and frontend in containers.

1.  **Build and start:**
    ```bash
    docker-compose up --build
    ```
2.  **Access:**
    - Backend: `http://localhost:8000`
    - Frontend: `http://localhost:5173`

### Option 2: Normal Execution (Uvicorn)

Manual execution of the backend using Uvicorn.

1.  **Setup Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```
2.  **Run Backend:**
    ```bash
    uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
    ```
3.  **Run Frontend (Separate Terminal):**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

### Option 3: Using .bat File (Windows)

Quick execution using the provided batch script.

1.  **Run Script:**
    Double-click `run.bat` or run in terminal:
    ```cmd
    .\run.bat
    ```

## Data Processing

To process the initial dataset and populate the vector store:

```cmd
.\process_data.bat
```
