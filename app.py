"""
Phase 4: FastAPI Semantic Search Service
Requirements:
- fastapi
- uvicorn
- pydantic
- chromadb
- sentence-transformers
- numpy
- pickle
"""

import os
import time
import pickle
import numpy as np
import chromadb
from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from semantic_cache import SemanticCache

# --- Pydantic Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The natural language query string")

class SearchResultMetadata(BaseModel):
    category: str
    word_count: int
    cluster_id: int

class SearchResult(BaseModel):
    id: str
    text: str
    metadata: SearchResultMetadata
    similarity: float

class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: List[SearchResult]
    dominant_cluster: int
    dominant_cluster_name: str
    cluster_distribution: List[float]
    response_time_ms: float
    cache_hit_cluster: Optional[int] = None
    similarity_to_centroid: Optional[float] = None

class StatsResponse(BaseModel):
    hits: int
    misses: int
    evictions: int
    hit_rate: float
    size: int
    per_cluster_hits: Dict[str, int]

class FlushResponse(BaseModel):
    message: str

# --- Cluster Mapping ---
CLUSTER_NAMES = {
    0: "Religion & Politics",
    1: "Sports",
    2: "Science & Medicine",
    3: "Vehicles & Transportation",
    4: "Computers & Technology"
}

# --- Global State ---
app_state = {
    "model": None,
    "chroma_collection": None,
    "cache": None,
    "cleaned_docs": None,
    "cluster_results": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown of the FastAPI service.
    Loads all heavy assets (embeddings, docs, models, vector store) into memory once.
    """
    print("Initializing Semantic Search Service...")
    
    # 1. Load data assets
    try:
        with open('cleaned_docs.pkl', 'rb') as f:
            app_state["cleaned_docs"] = pickle.load(f)
        
        with open('cluster_results.pkl', 'rb') as f:
            app_state["cluster_results"] = pickle.load(f)
            
        # 2. Initialize Model
        app_state["model"] = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Initialize ChromaDB
        from chromadb.config import Settings
        chroma_client = chromadb.PersistentClient(
            path="./chroma_store",
            settings=Settings(anonymized_telemetry=False)
        )
        app_state["chroma_collection"] = chroma_client.get_collection(name="newsgroups")
        
        # 4. Update ChromaDB Metadata with Cluster Info
        print("Syncing cluster metadata with ChromaDB (this may take a moment)...")
        dominant_clusters = app_state["cluster_results"]["dominant_clusters"]
        ids = [doc['doc_id'] for doc in app_state["cleaned_docs"]]
        metadatas = [{"dominant_cluster": int(dominant_clusters[i])} for i in range(len(ids))]
        
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            app_state["chroma_collection"].update(
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

        # 5. Initialize Semantic Cache
        threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.85))
        app_state["cache"] = SemanticCache(similarity_threshold=threshold)
        print(f"Service ready with similarity_threshold={threshold}.")
    except Exception as e:
        print(f"CRITICAL ERROR during startup: {e}")
        raise e
        
    yield
    print("Shutting down service...")

app = FastAPI(title="Trademarkia Semantic Search", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

# --- Endpoints ---

@app.post("/query", response_model=QueryResponse)
async def perform_search(request: QueryRequest):
    """
    Performs a semantic search with caching and detailed result metadata.
    """
    start_time = time.time()
    try:
        query_text = request.query
        
        # a) Embed query
        query_embedding = app_state["model"].encode(query_text)
        
        # b) Determine dominant cluster
        dominant_cluster = app_state["cache"].get_dominant_cluster(query_embedding)
        distribution = app_state["cache"].get_cluster_distribution(query_embedding)
        cluster_name = CLUSTER_NAMES.get(dominant_cluster, "General Topics")
        
        # c) Cache Lookup
        cache_entry = app_state["cache"].lookup(query_embedding, dominant_cluster)
        
        if cache_entry:
            duration = round((time.time() - start_time) * 1000, 2)
            sim = float(app_state["cache"]._cosine_similarity(query_embedding, cache_entry.query_embedding))
            results = [SearchResult(**r) for r in cache_entry.results]
            
            return QueryResponse(
                query=query_text,
                cache_hit=True,
                matched_query=cache_entry.query_text,
                similarity_score=round(sim, 4),
                result=results,
                dominant_cluster=dominant_cluster,
                dominant_cluster_name=cluster_name,
                cluster_distribution=distribution,
                response_time_ms=duration,
                cache_hit_cluster=dominant_cluster,
                similarity_to_centroid=None
            )
        
        # d) Cache Miss: Query ChromaDB
        n_results = 3
        results = app_state["chroma_collection"].query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where={"dominant_cluster": dominant_cluster}
        )
        
        # e) Format results
        search_results = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            text = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            sim = round(1.0 - (results['distances'][0][i] / 2.0), 4) 
            
            search_results.append({
                "id": doc_id,
                "text": text[:500],
                "metadata": {
                    "category": meta.get("original_category", "unknown"),
                    "word_count": len(text.split()),
                    "cluster_id": dominant_cluster
                },
                "similarity": sim
            })
        
        duration = round((time.time() - start_time) * 1000, 2)
        
        # f) Store in cache
        app_state["cache"].store(query_text, query_embedding, dominant_cluster, search_results, distribution)
        
        # g) Return
        return QueryResponse(
            query=query_text,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result=[SearchResult(**r) for r in search_results],
            dominant_cluster=dominant_cluster,
            dominant_cluster_name=cluster_name,
            cluster_distribution=distribution,
            response_time_ms=duration,
            cache_hit_cluster=None,
            similarity_to_centroid=None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search Error: {str(e)}")

@app.get("/cache/stats", response_model=StatsResponse)
async def get_cache_stats():
    if not app_state["cache"]:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    return app_state["cache"].get_stats()

@app.delete("/cache", response_model=FlushResponse)
async def flush_cache():
    if not app_state["cache"]:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    app_state["cache"].flush()
    return FlushResponse(message="Cache cleared.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
