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

def clean_text(text: str) -> str:
    """
    Cleans the raw newsgroup text by stripping headers and metadata.
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # Prefix list to strip headers as requested
    # Decision: Strip headers because they contain metadata (emails, hosts, subjects) 
    # that can skew semantic similarity towards structural patterns rather than content.
    strip_prefixes = (
        'From:', 'Subject:', 'Organization:', 'Lines:', 
        'NNTP-Posting-Host:', 'X-', 'Path:', 'Newsgroups:',
        'Date:', 'Message-ID:', 'References:', 'Sender:',
        'Reply-To:', 'Distribution:', 'News-Software:',
        'Article-I.D.:', 'Originator:', 'Nntp-Posting-Host:'
    )
    
    in_header = True
    for line in lines:
        # If we hit an empty line, we assume the header block is done
        if in_header and not line.strip():
            in_header = False
            continue
            
        # Strip lines that match known prefixes even outside the main header block
        # just to be safe, or skip the entire header block.
        if any(line.startswith(prefix) for prefix in strip_prefixes):
            continue
            
        cleaned_lines.append(line)
        
    cleaned_text = '\n'.join(cleaned_lines).strip()
    return cleaned_text

def main():
    dataset_root = Path('./dataset')
    if not dataset_root.exists():
        print(f"Error: Dataset directory {dataset_root} not found.")
        return

    raw_docs = []
    print("Walking dataset directory...")
    # Walk recursively to find all .txt files
    # Parent folder name is the category
    for file_path in tqdm(list(dataset_root.rglob('*.txt'))):
        try:
            # Using latin-1 encoding as it's more robust for old newsgroup data
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

    # Cleaning and De-duplication
    cleaned_docs = []
    seen_texts = set()
    discarded_short = 0
    discarded_duplicate = 0

    print("Cleaning corpus...")
    for doc in tqdm(raw_docs):
        cleaned_content = clean_text(doc['text'])
        
        # Decision: Remove posts shorter than 50 characters after cleaning.
        # Rationale: Very short posts (e.g., "Thanks!", "I agree") lack enough semantic 
        # information to be useful for search or meaningful embeddings.
        if len(cleaned_content) < 50:
            discarded_short += 1
            continue
            
        # Decision: Remove duplicate documents (exact text match).
        # Rationale: Exact duplicates inflate the index and can bias search results.
        if cleaned_content in seen_texts:
            discarded_duplicate += 1
            continue
            
        seen_texts.add(cleaned_content)
        
        doc_id = f"doc_{len(cleaned_docs)}"
        cleaned_docs.append({
            'doc_id': doc_id,
            'text': cleaned_content,
            'category': doc['category'],
            'source_file': doc['source_file']
        })

    total_kept = len(cleaned_docs)
    total_discarded = discarded_short + discarded_duplicate
    print(f"Final corpus stats:")
    print(f"  Total docs kept: {total_kept}")
    print(f"  Total discarded: {total_discarded} (Short: {discarded_short}, Duplicates: {discarded_duplicate})")

    # Embedding Generation
    print("Generating embeddings using 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts_to_embed = [doc['text'] for doc in cleaned_docs]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)
    
    print(f"Embedding shape: {embeddings.shape}")

    # Persistence: ChromaDB
    print("Persisting to ChromaDB...")
    from chromadb.config import Settings
    chroma_client = chromadb.PersistentClient(
        path="./chroma_store",
        settings=Settings(anonymized_telemetry=False)
    )
    # Delete existing collection if it exists to ensure a fresh start as per Phase 1 requirements
    try:
        chroma_client.delete_collection("newsgroups")
    except:
        pass
    
    collection = chroma_client.create_collection(name="newsgroups")
    
    # Add documents in batches to ChromaDB
    batch_size = 5000
    for i in range(0, len(cleaned_docs), batch_size):
        batch = cleaned_docs[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size].tolist()
        batch_ids = [d['doc_id'] for d in batch]
        batch_metadatas = [{'doc_id': d['doc_id'], 'original_category': d['category'], 'cleaned_text': d['text'][:1000]} for d in batch]
        batch_documents = [d['text'] for d in batch]
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )

    # Save outputs for Phase 2
    print("Saving output files...")
    np.save('embeddings.npy', embeddings)
    with open('cleaned_docs.pkl', 'wb') as f:
        pickle.dump(cleaned_docs, f)

    print("Phase 1 complete.")

if __name__ == "__main__":
    main()
