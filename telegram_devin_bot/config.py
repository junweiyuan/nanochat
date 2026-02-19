import os

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
DEVIN_API_KEY = os.environ["DEVIN_API_KEY"]
DEVIN_API_BASE = "https://api.devin.ai/v1"
POLL_INTERVAL = 10
ALLOWED_USERS = os.environ.get("ALLOWED_TELEGRAM_USERS", "")
