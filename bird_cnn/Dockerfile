FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better Docker layer caching)
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make sure the startup script is executable
RUN chmod +x start_server.sh

# Use the script as the container entrypoint
ENTRYPOINT ["./start_server.sh"]