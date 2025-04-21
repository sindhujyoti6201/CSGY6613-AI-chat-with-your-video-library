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
COPY . ./feature-pipeline/


RUN pip install --no-cache-dir -r ./feature-pipeline/requirements.txt

# Run the script
CMD ["python", "./feature-pipeline/embed_and_upload.py"]
