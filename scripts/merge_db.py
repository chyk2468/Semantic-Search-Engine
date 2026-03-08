import os

def merge_files(output_file, chunk_dir):
    chunk_files = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if ".part" in f])
    
    with open(output_file, 'wb') as out_f:
        for chunk_file in chunk_files:
            print(f"Reading {chunk_file}...")
            with open(chunk_file, 'rb') as f:
                out_f.write(f.read())
                
    print(f"Merged {len(chunk_files)} chunks into {output_file}")

if __name__ == "__main__":
    db_path = "data/chroma_store/chroma.sqlite3"
    chunk_dir = "data/chroma_store/chroma.sqlite3_chunks"
    
    if os.path.exists(chunk_dir):
        merge_files(db_path, chunk_dir)
    else:
        print(f"Chunk directory {chunk_dir} not found.")
