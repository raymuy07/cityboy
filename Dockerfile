FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY model_weights.pth .
COPY model_weights_v3.pth .
COPY .env .

# Telethon session files — copy them from the NUC before building:
#   scp guy@100.85.199.23:~/cityboy/cityboy_listener.session .
#   scp guy@100.85.199.23:~/cityboy/cityboy_bot.session .
COPY cityboy_listener.session cityboy_bot.session ./

CMD ["python", "inference.py", "--listen"]
