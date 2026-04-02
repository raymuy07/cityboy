# CityBoy — Emergency Backup

Telegram bot that monitors `@tzevaadomm` for Pikud HaOref pre-alert images, runs a MobileNetV3 model, and posts predictions to `@cityboy_alerts`.

**Only run this if Guy asks you to — running two instances at the same time will break things.**

## Setup

### 1. Get the secret files from Guy

You need these files (not in the repo for security):
- `.env` — Telegram API credentials
- `cityboy_listener.session` — Telegram user session
- `cityboy_bot.session` — Telegram bot session
- `model_weights.pth` — V1 model weights
- `model_weights_v3.pth` — V3 model weights

Put them all in the repo root folder alongside `inference.py`.

### 2. Install Docker

- **Windows:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Mac:** `brew install --cask docker`
- **Linux:** `sudo apt install docker.io`

### 3. Build

```bash
docker build -t cityboy .
```

Takes a few minutes the first time (downloads PyTorch).

### 4. Run

```bash
docker run -d --restart unless-stopped --name cityboy -e PYTHONUNBUFFERED=1 cityboy
```

### 5. Verify

```bash
docker logs -f cityboy
```

You should see:
```
Listening on @tzevaadomm for trigger images ...
Posting alerts to @cityboy_alerts
Bot is ready. Waiting for events ...
```

Press `Ctrl+C` to stop following logs — the bot keeps running in the background.

## Useful commands

| What | Command |
|---|---|
| Check if running | `docker ps` |
| View logs | `docker logs -f cityboy` |
| Stop | `docker stop cityboy` |
| Start again | `docker start cityboy` |
| Full restart | `docker rm -f cityboy && docker build -t cityboy . && docker run -d --restart unless-stopped --name cityboy -e PYTHONUNBUFFERED=1 cityboy` |

## Important

- **Never run this while Guy's instance is running** (NUC or cloud). Two instances sharing the same Telegram session will conflict.
- If you see `Server closed the connection` in logs — that's normal, Telethon reconnects automatically.
- The `.env` and `.session` files contain Guy's Telegram credentials — don't share them.
