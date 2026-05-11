FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    bash \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

Copy . .

RUN chmod +x start_server.sh

ENTRYPOINT ["bash", "./start_server.sh"]