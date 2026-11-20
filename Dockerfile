FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN python3.12 -m venv /venv
ENV PATH="/venv/bin:$PATH"
COPY requirements-fhe.txt .
RUN pip install --no-cache-dir -r requirements-fhe.txt
COPY cipherchat ./cipherchat
COPY server ./server
COPY client ./client
COPY weights ./weights
EXPOSE 8000 8501
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
