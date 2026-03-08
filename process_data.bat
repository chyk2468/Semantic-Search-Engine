@echo off
if not exist venv (
    echo Error: Virtual environment 'venv' not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo Activating environment...
call venv\Scripts\activate

echo Running Phase 1: Embedding Generation...
python phase1_embed.py
if %errorlevel% neq 0 (
    echo Error during Phase 1.
    pause
    exit /b %errorlevel%
)

echo Running Phase 2: Fuzzy Clustering...
python phase2_cluster.py
if %errorlevel% neq 0 (
    echo Error during Phase 2.
    pause
    exit /b %errorlevel%
)

echo.
echo Data processing complete. Assets generated: embeddings.npy, cleaned_docs.pkl, cluster_results.pkl, chroma_store/
pause
