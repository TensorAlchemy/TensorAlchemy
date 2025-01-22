FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV AXON_PORT=8101

RUN apt update && \
    apt install -y gcc curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN echo "import warnings; warnings.simplefilter('ignore')" > /usr/local/lib/python3.10/site-packages/sitecustomize.py

CMD ["python3.10", "/app/neurons/miners/Inpainter/main.py"]
