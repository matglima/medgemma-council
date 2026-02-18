# Use a lightweight Python base
FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (build-essential needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy setup files first to cache dependencies
COPY requirements.txt setup.py ./

# Install dependencies (including dev dependencies for testing)
# Note: In a real GPU env, you'd install llama-cpp-python with CUDA support here
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ".[dev]"

# Copy the source code
COPY src/ src/
COPY tests/ tests/
COPY data/ data/
COPY app.py .
COPY app_gradio.py .
COPY council_cli.py .
COPY conftest.py .
COPY README.md .

# Install the project in editable mode
RUN pip install -e .

# Default command: Run tests to ensure integrity
CMD ["pytest", "tests/", "-v"]

# Instructions for running the app:
# docker run -p 8501:8501 medgemma-council streamlit run app.py
# docker run -p 7860:7860 medgemma-council python app_gradio.py
