# Use Python 3.11 slim based on Debian Bookworm
FROM python:3.11-slim-bookworm

# 1. Install system dependencies for OpenCV and Git
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Download the SAM model weights *during build*
# This ensures the model is baked into the image and doesn't download on every run.
RUN wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 5. Copy the application code
COPY pipeline.py .
COPY index.html .

# 6. Expose the port
EXPOSE 8000

# 7. Run the application
CMD ["uvicorn", "pipeline:app", "--host", "0.0.0.0", "--port", "8000"]