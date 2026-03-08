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

## Screenshots & Workflow

Follow these steps to understand the application workflow:

### 1. Initial Setup and Data Processing
Before running the application, the dataset must be embedded and stored in ChromaDB.
![Data Processing](screenshots/Screenshot%202026-03-08%20172702.png)
*   **Step:** Run `.\process_data.bat` to trigger the embedding and clustering pipeline.

### 2. Launching the Application
The backend and frontend can be started together or separately.
![Starting the Server](screenshots/Screenshot%202026-03-08%20173840.png)
*   **Step:** Use `.\run.bat` or `docker-compose up` to launch the FastAPI backend and React frontend.

### 3. Search Interface Overview
The main interface allows for natural language queries.
![UI Overview](screenshots/Screenshot%202026-03-08%20173924.png)
*   **Step:** Open `http://localhost:5173` to access the Semantic Search Engine UI.

### 4. Performing a Semantic Search
Enter a query to find relevant documents based on meaning rather than just keywords.
![Search Query](screenshots/Screenshot%202026-03-08%20173953.png)
*   **Step:** Type a natural language query (e.g., "space exploration" or "computer hardware") into the search bar.

### 5. Reviewing Search Results
The results show similarity scores, categories, and dominant clusters.
![Search Results](screenshots/Screenshot%202026-03-08%20174000.png)
*   **Step:** Observe how the engine returns the most relevant documents with match percentages.

### 6. Semantic Caching in Action
Subsequent similar queries are served from the semantic cache for faster response times.
![Cache Hit](screenshots/Screenshot%202026-03-08%20174025.png)
*   **Step:** Notice the "Cache Hit" badge and reduced response time when searching for similar topics.

### 7. Cluster Distribution
The engine analyzes which topic cluster the query belongs to.
![Cluster Analysis](screenshots/Screenshot%202026-03-08%20174104.png)
*   **Step:** View the "Dominant Cluster" (e.g., Science & Medicine) assigned to your query.
