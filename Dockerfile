# Use a slim Python 3.10 image to keep the footprint small
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for some Python packages (like build tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker layer caching.
# This ensures that we don't re-download all packages unless requirements.txt changes.
COPY requirements.txt .

# Install dependencies using the CPU-only version of torch to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core application files and pre-computed assets from Phase 1 & 2
# We do NOT copy the raw dataset or processing scripts to keep the production image lean.
COPY app.py .
COPY semantic_cache.py .
COPY embeddings.npy .
COPY cleaned_docs.pkl .
COPY cluster_results.pkl .
COPY chroma_store/ ./chroma_store/

# Expose the FastAPI default port
EXPOSE 8000

# Run the FastAPI service using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
