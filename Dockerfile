# Use a slim Python 3.10 image to keep the footprint small
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for some Python packages (like build tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker layer caching.
COPY requirements.txt .

# Install dependencies using the CPU-only version of torch to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core application files and pre-computed assets
# We copy everything needed for a clean production startup.
COPY backend/ ./backend/
COPY data/ ./data/
COPY static/ ./static/
COPY scripts/ ./scripts/

# Merge the database chunks before running the application
RUN python scripts/merge_db.py

# Expose the FastAPI default port
EXPOSE 8000

# Run the FastAPI service using uvicorn
# Using 0.0.0.0 is critical for Docker networking.
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
