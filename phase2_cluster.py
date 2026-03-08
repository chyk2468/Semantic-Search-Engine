"""
Phase 2: Fuzzy Clustering, Dimensionality Reduction, and Analysis
Requirements:
- umap-learn
- scikit-fuzzy
- scikit-learn
- matplotlib
- numpy
- pandas
"""

import pickle
import numpy as np
import pandas as pd
import umap
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from tqdm import tqdm

def main():
    # 1. Load data
    print("Loading data from Phase 1...")
    embeddings = np.load('embeddings.npy')
    with open('cleaned_docs.pkl', 'rb') as f:
        cleaned_docs = pickle.load(f)
    
    texts = [doc['text'] for doc in cleaned_docs]
    categories = [doc['category'] for doc in cleaned_docs]

    # 2. Dimension Reduction with UMAP
    # UMAP for clustering (n_components=50)
    print("Reducing dimensions with UMAP for clustering (50D)...")
    reducer_50 = umap.UMAP(n_components=50, random_state=42, n_jobs=-1)
    embeddings_50d = reducer_50.fit_transform(embeddings)

    # UMAP for visualization (n_components=2)
    print("Reducing dimensions with UMAP for visualization (2D)...")
    reducer_2 = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
    embeddings_2d = reducer_2.fit_transform(embeddings)

    # 3. Fuzzy C-Means: Choosing k
    print("Finding optimal k using Partition Coefficient (PC)...")
    # Data needs to be transposed for skfuzzy: (features, samples)
    data_fcm = embeddings_50d.T
    
    ks = range(5, 26, 2)  # Testing k=5..25 with step 2 to save time
    fpcs = []
    
    for k in tqdm(ks, desc="Evaluating k"):
        # cntr: cluster centers, u: membership matrix, u0: initial, d: distance, 
        # jm: objective function, p: num iterations, fpc: fuzzy partition coefficient
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data_fcm, k, m=2, error=0.005, maxiter=1000, seed=42
        )
        fpcs.append(fpc)

    # Plot PC for cluster selection
    plt.figure(figsize=(10, 6))
    plt.plot(ks, fpcs, marker='o')
    plt.title('Fuzzy Partition Coefficient (PC) vs. Number of Clusters (k)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('FPC')
    plt.grid(True)
    plt.savefig('cluster_selection.png')
    print("Saved cluster_selection.png")

    # Final k selection logic: 
    # High FPC means cleaner partitions. Often there's an elbow.
    # We choose the k where FPC starts to drop significantly or the best local max.
    # For newsgroups, 20 is a natural choice, but let's see what the data says.
    # In practice, many datasets show a peak or an elbow around the true number of topics.
    best_idx = np.argmax(fpcs)
    chosen_k = ks[best_idx]
    
    # Justification (printed to console later)
    reasoning = f"Chosen k={chosen_k} because it maximizes the Fuzzy Partition Coefficient (FPC), " \
                f"indicating the most stable and crisp cluster structure among the tested values."

    # 4. Final Fuzzy C-Means Clustering
    print(f"Running final FCM with k={chosen_k}...")
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_fcm, chosen_k, m=2, error=0.005, maxiter=1000, seed=42
    )
    
    # Membership matrix u is (chosen_k, num_docs)
    membership_matrix = u.T # (num_docs, chosen_k)
    dominant_clusters = np.argmax(membership_matrix, axis=1)
    uncertainty_scores = 1 - np.max(membership_matrix, axis=1)

    # 5. Analysis
    print("Performing analysis...")
    # TF-IDF per cluster
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    cluster_info = {}
    for cluster_id in range(chosen_k):
        # Indices of docs where this cluster is dominant
        idx = np.where(dominant_clusters == cluster_id)[0]
        if len(idx) == 0:
            cluster_info[cluster_id] = {"top_terms": [], "top_categories": []}
            continue
            
        # Top TF-IDF terms
        cluster_tfidf = tfidf_matrix[idx].mean(axis=0).A1
        top_indices = cluster_tfidf.argsort()[-10:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        
        # Top categories
        cluster_cats = [categories[i] for i in idx]
        top_categories = Counter(cluster_cats).most_common(5)
        
        cluster_info[cluster_id] = {
            "top_terms": top_terms,
            "top_categories": top_categories
        }
        
        print(f"Cluster {cluster_id}:")
        print(f"  Top Terms: {', '.join(top_terms)}")
        print(f"  Categories: {', '.join([f'{cat}({count})' for cat, count in top_categories])}")

    # Boundary cases
    boundary_indices = np.argsort(uncertainty_scores)[-50:][::-1]
    with open('boundary_cases.txt', 'w', encoding='utf-8') as f:
        f.write("Top 50 Most Uncertain Documents (Boundary Cases)\n")
        f.write("="*50 + "\n\n")
        for i in boundary_indices:
            top_2_clusters = np.argsort(membership_matrix[i])[-2:][::-1]
            m1, m2 = membership_matrix[i][top_2_clusters[0]], membership_matrix[i][top_2_clusters[1]]
            snippet = cleaned_docs[i]['text'][:200].replace('\n', ' ')
            f.write(f"Doc ID: {cleaned_docs[i]['doc_id']}\n")
            f.write(f"Uncertainty: {uncertainty_scores[i]:.4f}\n")
            f.write(f"Top 2 Memberships: Cluster {top_2_clusters[0]} ({m1:.4f}), Cluster {top_2_clusters[1]} ({m2:.4f})\n")
            f.write(f"Category: {cleaned_docs[i]['category']}\n")
            f.write(f"Snippet: {snippet}...\n")
            f.write("-" * 30 + "\n")

    # 3 genuine boundary cases for console
    print("\nGenuine Boundary Cases:")
    for i in boundary_indices[:3]:
        print(f"Doc {i}: Category={categories[i]}, Uncertainty={uncertainty_scores[i]:.4f}")
        print(f"Membership Distribution: {membership_matrix[i]}")

    # Overlap Matrix: Correlation of membership vectors across clusters
    # Matrix size (k, k)
    overlap_matrix = np.zeros((chosen_k, chosen_k))
    for i in range(chosen_k):
        for j in range(chosen_k):
            # How many docs have high membership in both?
            # Or just correlate the membership columns
            overlap_matrix[i, j] = np.corrcoef(membership_matrix[:, i], membership_matrix[:, j])[0, 1]
    
    print("\nCluster Semantic Overlap Matrix (Correlation):")
    print(overlap_matrix)

    # 6. Save outputs
    print("Saving output files...")
    cluster_results = {
        'membership_matrix': membership_matrix,
        'dominant_clusters': dominant_clusters,
        'uncertainty_scores': uncertainty_scores,
        'chosen_k': chosen_k,
        'reasoning': reasoning
    }
    with open('cluster_results.pkl', 'wb') as f:
        pickle.dump(cluster_results, f)

    # Visualization
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=dominant_clusters, cmap='tab20', s=1, alpha=0.5)
    plt.colorbar(scatter)
    plt.title(f'UMAP 2D Projection Colored by Dominant Cluster (k={chosen_k})')
    plt.savefig('clusters_2d.png')
    print("Saved clusters_2d.png")

    print("\nPhase 2 complete.")
    print(f"Reasoning: {reasoning}")

if __name__ == "__main__":
    main()
