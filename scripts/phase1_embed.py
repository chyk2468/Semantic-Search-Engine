"""
Phase 1: Data Preparation, Embedding Generation, and Storage
Requirements:
- sentence-transformers
- chromadb
- numpy
- tqdm
"""

import os
import pickle
import numpy as np
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def clean_text(text: str) -> str:
    """
    Cleans the raw newsgroup text by stripping headers and metadata.
    Follows a two-pass approach:
    1. Skip everything until the first double-newline (common newsgroup body start).
    2. Filter remaining lines for stray header prefixes.
    """
    # Pass 1: Find body start
    parts = text.split('\n\n', 1)
    if len(parts) > 1:
        body = parts[1]
    else:
        body = text
        
    lines = body.split('\n')
    cleaned_lines = []
    
    strip_prefixes = (
        'From:', 'Subject:', 'Organization:', 'Lines:', 
        'NNTP-Posting-Host:', 'X-', 'Path:', 'Newsgroups:',
        'Date:', 'Message-ID:', 'References:', 'Sender:',
        'Reply-To:', 'Distribution:', 'News-Software:',
        'Article-I.D.:', 'Originator:', 'Nntp-Posting-Host:'
    )
    
    for line in lines:
        if any(line.startswith(prefix) for prefix in strip_prefixes):
            continue
        cleaned_lines.append(line)
        
    cleaned_text = '\n'.join(cleaned_lines).strip()
    return cleaned_text

def main():
    dataset_root = DATA_DIR / 'dataset'
    if not dataset_root.exists():
        print(f"Error: Dataset directory {dataset_root} not found.")
        return

    raw_docs = []
    print("Walking dataset directory...")
    for file_path in tqdm(list(dataset_root.rglob('*.txt'))):
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
                category = file_path.parent.name
                raw_docs.append({
                    'text': content,
                    'category': category,
                    'source_file': str(file_path)
                })
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")

    print(f"Loaded {len(raw_docs)} raw documents.")

    # Step 2: PREPROCESS FIRST
    cleaned_docs = []
    seen_texts = set()
    discarded_short = 0
    discarded_duplicate = 0

    print("Cleaning corpus (stripping headers, de-duplicating)...")
    for doc in tqdm(raw_docs):
        # 2.1 CLEAN TEXT BEFORE EMBEDDING
        cleaned_content = clean_text(doc['text'])
        
        if len(cleaned_content) < 50:
            discarded_short += 1
            continue
            
        if cleaned_content in seen_texts:
            discarded_duplicate += 1
            continue
            
        seen_texts.add(cleaned_content)
        
        doc_id = f"doc_{len(cleaned_docs)}"
        cleaned_docs.append({
            'doc_id': doc_id,
            'text': cleaned_content, # Storing CLEAN text
            'category': doc['category'],
            'source_file': doc['source_file']
        })

    print(f"Final corpus stats: Kept {len(cleaned_docs)}, Discarded {discarded_short + discarded_duplicate}")

    # Step 3: Embed the CLEAN text only
    print("Generating embeddings for CLEAN text using 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts_to_embed = [doc['text'] for doc in cleaned_docs]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    
    # Step 5: Store CLEAN docs in ChromaDB
    print("Persisting CLEAN documents to ChromaDB...")
    from chromadb.config import Settings
    chroma_client = chromadb.PersistentClient(
        path=str(DATA_DIR / "chroma_store"),
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        chroma_client.delete_collection("newsgroups")
    except:
        pass
    
    collection = chroma_client.create_collection(name="newsgroups")
    
    batch_size = 5000
    for i in range(0, len(cleaned_docs), batch_size):
        batch = cleaned_docs[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size].tolist()
        batch_ids = [d['doc_id'] for d in batch]
        batch_metadatas = [{'doc_id': d['doc_id'], 'original_category': d['category'], 'cleaned_text': d['text'][:500]} for d in batch]
        batch_documents = [d['text'] for d in batch] # Adding CLEAN text to ChromaDB
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )

    print("Saving output files...")
    np.save(DATA_DIR / 'embeddings.npy', embeddings)
    with open(DATA_DIR / 'cleaned_docs.pkl', 'wb') as f:
        pickle.dump(cleaned_docs, f)

    print("Phase 1 complete.")

if __name__ == "__main__":
    main()
