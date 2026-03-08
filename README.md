# Trademarkia Semantic Search System

A high-performance semantic search service built on the 20 Newsgroups dataset. This system utilizes deep learning embeddings, fuzzy clustering for semantic partitioning, and a custom-built semantic cache to provide fast, relevant search results across nearly 20,000 documents.

## Architecture

```text
Dataset (20k .txt files) → Phase 1: Embedding → ChromaDB (Vector Store) + embeddings.npy
                                     ↓
                          Phase 2: Fuzzy Clustering → cluster_results.pkl
                                     ↓
                          Phase 3: Semantic Cache (In-Memory, Cluster-Partitioned)
                                     ↓
                          Phase 4: FastAPI Service (app.py) ← User HTTP Requests
```

## Getting Started

### Path A: Local Development (Windows CMD)
1.  **Initialize environment:**
    ```cmd
    python -m venv venv
    venv\Scripts\activate
    ```
2.  **Install dependencies:**
    ```cmd
    pip install -r requirements.txt
    ```
3.  **Process data (if starting from scratch):**
    ```cmd
    python phase1_embed.py
    python phase2_cluster.py
    ```
4.  **Run service:**
    ```cmd
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

### Path B: Local Development (Linux/macOS)
1.  **Initialize environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Process data:**
    ```bash
    python3 phase1_embed.py
    python3 phase2_cluster.py
    ```
4.  **Run service:**
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000
    ```

### Path C: Docker (Recommended)
Launch the entire system with a single command:
```bash
docker-compose up --build
```

## API Reference

### 1. POST `/query`
Performs a semantic search.

**Windows CMD:**
```cmd
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d "{\"query\": \"How do I repair a flat tire?\"}"
```

**Linux/macOS:**
```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query": "How do I repair a flat tire?"}'
```

**Example Response (Cache Miss):**
```json
{
  "query": "How do I repair a flat tire?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "Snippet 1...\n---\nSnippet 2...",
  "dominant_cluster": 3
}
```

### 2. GET `/cache/stats`
Returns cache performance and usage statistics.

```bash
curl "http://localhost:8000/cache/stats"
```

### 3. DELETE `/cache`
Flushes all entries and resets statistics.

**Windows CMD:**
```cmd
curl -X DELETE "http://localhost:8000/cache"
```

**Linux/macOS:**
```bash
curl -X DELETE "http://localhost:8000/cache"
```

## Design Decisions

- **Embedding Model (`all-MiniLM-L6-v2`)**: Selected for its excellent balance between inference speed and semantic quality, making it ideal for local search systems.
- **Fuzzy Clustering (C-Means)**: Unlike hard clustering, fuzzy clustering acknowledges that newsgroup documents often span multiple topics (e.g., a post about space technology relates to both Science and Engineering).
- **Cluster-Partitioned Cache**: By partitioning the cache into semantic clusters, lookups are restricted to relevant buckets. This reduces search complexity from $O(N)$ to approximately $O(N/K)$, ensuring the system remains fast as the cache grows.
- **Similarity Threshold**: Controlled by `SIMILARITY_THRESHOLD` (default 0.85). Values > 0.90 are very strict; values < 0.80 may lead to false positives where unrelated queries are incorrectly matched.

## File Manifest

- `app.py`: FastAPI service script with singleton state management.
- `semantic_cache.py`: Core logic for the cluster-partitioned semantic cache.
- `phase1_embed.py`: Script to load, clean, and embed the raw dataset.
- `phase2_cluster.py`: Script for UMAP reduction and Fuzzy C-Means clustering.
- `requirements.txt`: Pinned project dependencies.
- `Dockerfile`: Production-ready container configuration.
- `docker-compose.yml`: Orchestration for persistent vector storage.
- `embeddings.npy`: Pre-computed document vectors.
- `cleaned_docs.pkl`: Serialized metadata for all processed documents.
- `cluster_results.pkl`: Cluster assignments and metadata.
- `chroma_store/`: Persistent vector database directory.
