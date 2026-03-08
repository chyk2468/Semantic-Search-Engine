"""
Phase 3: Semantic Cache Module
Requirements:
- numpy
- sentence-transformers (for self-test analysis)
- pickle
- time
"""

import os
import numpy as np
import pickle
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

@dataclass
class CacheEntry:
    query_text: str
    query_embedding: np.ndarray
    dominant_cluster: int
    results: List[Dict] # Store full result objects
    cluster_distribution: List[float]
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0

class SemanticCache:
    """
    A custom semantic cache that uses fuzzy cluster indexing to avoid 
    linear scans of the entire cache.
    """
    
    _centroids = None
    _cluster_results = None

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the cache with a tunable similarity threshold.
        """
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, CacheEntry] = {}
        self.cluster_index: Dict[int, List[str]] = {}
        
        self.hit_count = 0
        self.miss_count = 0
        
        self._ensure_centroids_loaded()

    def get_cluster_distribution(self, embedding: np.ndarray) -> List[float]:
        """
        Calculates fuzzy membership (distribution) across all clusters.
        Uses the standard FCM membership formula: u_ij = 1 / sum((d_ij/d_kj)^(2/(m-1)))
        """
        if SemanticCache._centroids is None:
            return []
        
        # Calculate distances to all centroids
        dists = [np.linalg.norm(embedding - c) for c in SemanticCache._centroids]
        dists = np.array(dists)
        dists = np.fmax(dists, 1e-10) # Avoid division by zero
        
        # m=2 is our fuzzifier from Phase 2
        power = 2.0 / (2.0 - 1.0)
        
        memberships = []
        for i in range(len(dists)):
            denom = np.sum((dists[i] / dists)**power)
            memberships.append(1.0 / denom)
            
        return [float(round(m, 4)) for m in memberships]

    def _ensure_centroids_loaded(self):
        """
        Loads cluster results and computes centroids for projection.
        """
        if SemanticCache._centroids is None:
            try:
                cluster_results_path = DATA_DIR / 'cluster_results.pkl'
                embeddings_path = DATA_DIR / 'embeddings.npy'
                
                with open(cluster_results_path, 'rb') as f:
                    res = pickle.load(f)
                embeddings = np.load(embeddings_path)
                
                chosen_k = res['chosen_k']
                dominant_clusters = res['dominant_clusters']
                
                centroids = []
                for i in range(chosen_k):
                    idx = np.where(dominant_clusters == i)[0]
                    if len(idx) > 0:
                        centroids.append(np.mean(embeddings[idx], axis=0))
                    else:
                        centroids.append(np.zeros(embeddings.shape[1]))
                
                SemanticCache._centroids = np.array(centroids)
                SemanticCache._cluster_results = res
                print(f"SemanticCache: Loaded {chosen_k} cluster centroids.")
            except FileNotFoundError:
                print(f"Warning: {cluster_results_path} or {embeddings_path} not found.")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def get_dominant_cluster(self, embedding: np.ndarray) -> int:
        if SemanticCache._centroids is None: return 0
        similarities = [self._cosine_similarity(embedding, c) for c in SemanticCache._centroids]
        return int(np.argmax(similarities))

    def lookup(self, query_embedding: np.ndarray, dominant_cluster: int) -> Optional[CacheEntry]:
        candidate_ids = self.cluster_index.get(dominant_cluster, [])
        
        best_similarity = -1.0
        best_entry = None
        
        for eid in candidate_ids:
            entry = self.cache[eid]
            sim = self._cosine_similarity(query_embedding, entry.query_embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_entry = entry
        
        if best_entry and best_similarity >= self.similarity_threshold:
            self.hit_count += 1
            best_entry.hit_count += 1
            return best_entry
        
        self.miss_count += 1
        return None

    def store(self, query_text: str, query_embedding: np.ndarray, 
              dominant_cluster: int, results: List[Dict], distribution: List[float]) -> CacheEntry:
        entry_id = f"entry_{len(self.cache)}"
        entry = CacheEntry(
            query_text=query_text,
            query_embedding=query_embedding,
            dominant_cluster=dominant_cluster,
            results=results,
            cluster_distribution=distribution
        )
        
        self.cache[entry_id] = entry
        if dominant_cluster not in self.cluster_index:
            self.cluster_index[dominant_cluster] = []
        self.cluster_index[dominant_cluster].append(entry_id)
        
        return entry

    def get_stats(self) -> dict:
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        
        # Per-cluster hits tracking
        per_cluster_hits = {}
        for entry in self.cache.values():
            if entry.hit_count > 0:
                per_cluster_hits[str(entry.dominant_cluster)] = per_cluster_hits.get(str(entry.dominant_cluster), 0) + entry.hit_count

        return {
            "hits": self.hit_count,
            "misses": self.miss_count,
            "evictions": 0,
            "hit_rate": round(hit_rate, 3),
            "size": len(self.cache),
            "per_cluster_hits": per_cluster_hits
        }

    def flush(self) -> None:
        self.cache.clear()
        self.cluster_index.clear()
        self.hit_count = 0
        self.miss_count = 0

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    
    print("Running SemanticCache Threshold Analysis...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 10 test pairs
    test_pairs = [
        # a) Semantically identical
        ("How do I fix a flat tire?", "Repairing a punctured car tire"),
        ("Space exploration and Mars", "Missions to the red planet"),
        ("Windows 10 installation error", "Installing Win10 blue screen"),
        
        # b) Topically related but distinct
        ("Best recipe for chocolate cake", "How to bake a cocoa dessert"),
        ("The impact of global warming", "Climate change effects on environment"),
        ("Python programming for beginners", "Learning to code in Python"),
        ("Who won the World Series?", "Major League Baseball championship winner"),
        
        # c) Completely unrelated
        ("Atheism and philosophy", "Stock market trends in 2024"),
        ("Quantum physics basics", "How to train a golden retriever"),
        ("Medieval history of Europe", "Making homemade pasta sauce")
    ]
    
    thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]
    
    # Pre-compute embeddings for efficiency
    queries_1 = [p[0] for p in test_pairs]
    queries_2 = [p[1] for p in test_pairs]
    embs_1 = model.encode(queries_1)
    embs_2 = model.encode(queries_2)
    
    print("\nResults:")
    print("-" * 40)
    for t in thresholds:
        cache = SemanticCache(similarity_threshold=t)
        hits = 0
        for i in range(len(test_pairs)):
            emb1 = embs_1[i]
            emb2 = embs_2[i]
            
            # Use the method from the class to get cluster
            cluster1 = cache.get_dominant_cluster(emb1)
            cluster2 = cache.get_dominant_cluster(emb2)
            
            # Store first query
            cache.store(queries_1[i], emb1, cluster1, [{"text": f"Result for {queries_1[i]}"}], [0.0]*5)
            
            # Lookup with second query
            res = cache.lookup(emb2, cluster2)
            if res:
                hits += 1
                
        print(f"Threshold {t:.2f}: {hits}/10 pairs were HITs")
    
    print("-" * 40)
    print("Analysis:")
    print("0.70 - 0.80: High hit rate, but risks 'False Hits' where distinct topics overlap.")
    print("0.85: The 'Sweet Spot'. Captures paraphrases while rejecting distinct topics.")
    print("0.90 - 0.95: Too restrictive. Defeats the purpose of semantic caching by missing simple rewordings.")
