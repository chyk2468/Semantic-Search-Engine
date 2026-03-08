import os

def split_file(file_path, chunk_size_mb=99):
    chunk_size = chunk_size_mb * 1024 * 1024
    file_name = os.path.basename(file_path)
    dir_name = os.path.dirname(file_path)
    output_dir = os.path.join(dir_name, f"{file_name}_chunks")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(file_path, 'rb') as f:
        chunk_num = 0
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            
            output_file = os.path.join(output_dir, f"{file_name}.part{chunk_num:03d}")
            with open(output_file, 'wb') as chunk_f:
                chunk_f.write(chunk_data)
            print(f"Created {output_file}")
            chunk_num += 1
            
    print(f"Split {file_path} into {chunk_num} chunks in {output_dir}")

if __name__ == "__main__":
    db_path = "data/chroma_store/chroma.sqlite3"
    if os.path.exists(db_path):
        split_file(db_path)
    else:
        print(f"File {db_path} not found.")
