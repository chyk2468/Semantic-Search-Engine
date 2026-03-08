import os

def merge_files(output_file, chunk_dir, cleanup=True):
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping merge.")
        return

    chunk_files = sorted([os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if ".part" in f])
    
    if not chunk_files:
        print(f"No chunks found in {chunk_dir}.")
        return

    with open(output_file, 'wb') as out_f:
        for chunk_file in chunk_files:
            print(f"Reading {chunk_file}...")
            with open(chunk_file, 'rb') as f:
                out_f.write(f.read())
                
    print(f"Merged {len(chunk_files)} chunks into {output_file}")

    if cleanup:
        for chunk_file in chunk_files:
            os.remove(chunk_file)
        print(f"Cleaned up {len(chunk_files)} chunks.")

if __name__ == "__main__":
    db_path = "data/chroma_store/chroma.sqlite3"
    chunk_dir = "data/chroma_store/chroma.sqlite3_chunks"
    
    # Enable cleanup when running in Docker (checking for environment variable or directory existence)
    is_docker = os.path.exists('/.dockerenv')
    
    if os.path.exists(chunk_dir):
        merge_files(db_path, chunk_dir, cleanup=is_docker)
    else:
        print(f"Chunk directory {chunk_dir} not found.")
