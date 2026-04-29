FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional but common)
RUN apt-get update && apt-get install -y \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker layer caching)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Make sure the startup script is executable
RUN chmod +x start_server.sh

# Use the script as the container entrypoint
ENTRYPOINT ["./start_server.sh"]