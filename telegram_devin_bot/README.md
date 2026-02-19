# Telegram Devin Bot

A Telegram bot that lets you interact with [Devin](https://devin.ai) directly from Telegram.

## Features

- **Create sessions** — Start new Devin sessions with `/new <prompt>`
- **Chat** — Send messages to your active Devin session by typing normally
- **Status updates** — Automatic notifications when Devin finishes or needs input
- **Session management** — List, switch between, and monitor multiple sessions
- **Access control** — Restrict bot access to specific Telegram users

## Setup

### 1. Create a Telegram Bot

1. Open Telegram and message [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the bot token

### 2. Get a Devin API Key

1. Go to [app.devin.ai](https://app.devin.ai) > Settings > API
2. Generate a new API key

### 3. Configure Environment Variables

```bash
export TELEGRAM_BOT_TOKEN="your-telegram-bot-token"
export DEVIN_API_KEY="your-devin-api-key"

# Optional: restrict access to specific Telegram usernames or user IDs (comma-separated)
export ALLOWED_TELEGRAM_USERS="your_username,123456789"
```

### 4. Install and Run

```bash
pip install -r requirements.txt
python bot.py
```

## Commands

| Command | Description |
|---------|-------------|
| `/start` | Show welcome message and available commands |
| `/new <prompt>` | Create a new Devin session |
| `/msg <text>` | Send a message to the active session |
| `/status` | Check the status of your active session |
| `/sessions` | List recent Devin sessions |
| `/select <id>` | Switch to a different session |
| `/link` | Get the web URL for your active session |

You can also just type a message without any command to send it to your active Devin session.

## How It Works

1. You send a prompt via `/new` to create a Devin session
2. The bot polls the Devin API for status changes and notifies you
3. You can send follow-up messages that get forwarded to Devin
4. When Devin finishes or creates a PR, you get notified automatically
