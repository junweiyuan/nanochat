import asyncio
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from config import TELEGRAM_BOT_TOKEN, POLL_INTERVAL, ALLOWED_USERS
from devin_client import DevinClient

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

devin = DevinClient()

chat_sessions: dict[int, str] = {}

last_seen_status: dict[str, str] = {}


def is_allowed(update: Update) -> bool:
    if not ALLOWED_USERS:
        return True
    allowed = {u.strip() for u in ALLOWED_USERS.split(",") if u.strip()}
    username = update.effective_user.username or ""
    user_id = str(update.effective_user.id)
    return username in allowed or user_id in allowed


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    text = (
        "Welcome to the Devin Telegram Bot!\n\n"
        "Commands:\n"
        "/new <prompt> - Start a new Devin session\n"
        "/msg <text> - Send a message to your active session\n"
        "/status - Check your active session status\n"
        "/sessions - List your recent Devin sessions\n"
        "/select <session_id> - Switch to a different session\n"
        "/link - Get the URL for your active session\n\n"
        "Or just send a regular message to chat with your active Devin session."
    )
    await update.message.reply_text(text)


async def new_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    prompt = " ".join(context.args) if context.args else ""
    if not prompt:
        await update.message.reply_text("Usage: /new <prompt>\nExample: /new Fix the login bug in auth.py")
        return

    await update.message.reply_text(f"Creating Devin session...")
    try:
        result = await devin.create_session(prompt)
        session_id = result["session_id"]
        url = result.get("url", f"https://app.devin.ai/sessions/{session_id}")
        chat_sessions[update.effective_chat.id] = session_id
        last_seen_status[session_id] = "running"
        await update.message.reply_text(
            f"Session created!\n"
            f"ID: `{session_id}`\n"
            f"URL: {url}\n\n"
            f"Send messages here and they'll be forwarded to Devin.",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        await update.message.reply_text(f"Failed to create session: {e}")


async def send_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    session_id = chat_sessions.get(update.effective_chat.id)
    if not session_id:
        await update.message.reply_text("No active session. Use /new <prompt> to start one.")
        return
    message = " ".join(context.args) if context.args else ""
    if not message:
        await update.message.reply_text("Usage: /msg <message>")
        return
    try:
        await devin.send_message(session_id, message)
        await update.message.reply_text("Message sent to Devin.")
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        await update.message.reply_text(f"Failed to send message: {e}")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    session_id = chat_sessions.get(update.effective_chat.id)
    if not session_id:
        await update.message.reply_text("No active session. Use /new <prompt> to start one.")
        return
    try:
        details = await devin.get_session(session_id)
        status_enum = details.get("status_enum", "unknown")
        title = details.get("title", "N/A")
        url = details.get("url", f"https://app.devin.ai/sessions/{session_id}")
        pr = details.get("pull_request", {})
        pr_url = pr.get("url", "") if pr else ""
        text = (
            f"Session: `{session_id}`\n"
            f"Title: {title}\n"
            f"Status: {status_enum}\n"
            f"URL: {url}"
        )
        if pr_url:
            text += f"\nPR: {pr_url}"
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        await update.message.reply_text(f"Failed to get status: {e}")


async def list_sessions(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    try:
        result = await devin.list_sessions(limit=10)
        sessions = result.get("sessions", [])
        if not sessions:
            await update.message.reply_text("No sessions found.")
            return
        lines = ["Recent Devin sessions:\n"]
        for s in sessions:
            sid = s.get("session_id", "?")
            title = s.get("title", "Untitled")
            st = s.get("status_enum", "?")
            lines.append(f"- `{sid[:8]}...` | {st} | {title}")
        active = chat_sessions.get(update.effective_chat.id, "")
        if active:
            lines.append(f"\nActive: `{active}`")
        lines.append("\nUse /select <session_id> to switch sessions.")
        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        await update.message.reply_text(f"Failed to list sessions: {e}")


async def select_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    if not context.args:
        await update.message.reply_text("Usage: /select <session_id>")
        return
    session_id = context.args[0]
    try:
        details = await devin.get_session(session_id)
        chat_sessions[update.effective_chat.id] = session_id
        status_enum = details.get("status_enum", "unknown")
        await update.message.reply_text(
            f"Switched to session `{session_id}`\nStatus: {status_enum}",
            parse_mode="Markdown",
        )
    except Exception as e:
        logger.error(f"Failed to select session: {e}")
        await update.message.reply_text(f"Failed to select session: {e}")


async def link(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    session_id = chat_sessions.get(update.effective_chat.id)
    if not session_id:
        await update.message.reply_text("No active session.")
        return
    try:
        details = await devin.get_session(session_id)
        url = details.get("url", f"https://app.devin.ai/sessions/{session_id}")
        await update.message.reply_text(url)
    except Exception as e:
        await update.message.reply_text(f"https://app.devin.ai/sessions/{session_id}")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update):
        return
    session_id = chat_sessions.get(update.effective_chat.id)
    if not session_id:
        await update.message.reply_text(
            "No active session. Use /new <prompt> to start one, or /sessions to pick an existing one."
        )
        return
    message = update.message.text
    try:
        await devin.send_message(session_id, message)
        await update.message.reply_text("Sent to Devin.")
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        await update.message.reply_text(f"Failed to send message: {e}")


async def poll_sessions(app: Application) -> None:
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        for chat_id, session_id in list(chat_sessions.items()):
            try:
                details = await devin.get_session(session_id)
                current_status = details.get("status_enum", "unknown")
                prev_status = last_seen_status.get(session_id)
                if prev_status and prev_status != current_status:
                    status_msg = f"Session `{session_id[:8]}...` status: {prev_status} -> {current_status}"
                    if current_status == "finished":
                        pr = details.get("pull_request", {})
                        pr_url = pr.get("url", "") if pr else ""
                        status_msg += "\nDevin has finished working."
                        if pr_url:
                            status_msg += f"\nPR: {pr_url}"
                    elif current_status == "blocked":
                        status_msg += "\nDevin is blocked and waiting for your input."
                    await app.bot.send_message(
                        chat_id=chat_id,
                        text=status_msg,
                        parse_mode="Markdown",
                    )
                last_seen_status[session_id] = current_status
            except Exception as e:
                logger.debug(f"Poll error for session {session_id}: {e}")


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", new_session))
    app.add_handler(CommandHandler("msg", send_msg))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("sessions", list_sessions))
    app.add_handler(CommandHandler("select", select_session))
    app.add_handler(CommandHandler("link", link))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def run():
        async with app:
            await app.start()
            await app.updater.start_polling()
            poll_task = asyncio.create_task(poll_sessions(app))
            logger.info("Bot is running. Press Ctrl+C to stop.")
            try:
                await asyncio.Event().wait()
            finally:
                poll_task.cancel()
                await app.updater.stop()
                await app.stop()

    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
