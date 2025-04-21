FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy app code
COPY ./src/csgy6613_ai_project/retrieval-and-generation-pipeline/ ./retrieval-and-generation-pipeline/
COPY ./datasets/ ./datasets/
RUN pip install --no-cache-dir -r ./retrieval-and-generation-pipeline/requirements.txt

# Run the script
CMD ["python", "./retrieval-and-generation-pipeline/query_and_response_generation.py"]
