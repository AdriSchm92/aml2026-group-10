"""Training takes a long time :)

Optional alerts when a job ends. Set:
  DISCORD_WEBHOOK_URL — incoming webhook
  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  RUN_NAME — optional short prefix
Missing vars = no-op. All sends are best-effort (timeouts, never raise to caller)."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request

# Short timeouts so a dead endpoint never blocks shutdown.
_URL_TIMEOUT = 8.0
_MAX_LEN = 1800


def _post_json(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=_URL_TIMEOUT) as resp:  # noqa: S310
        resp.read(256)


def _post_discord(webhook: str, text: str) -> None:
    # Discord "content" max ~2000; we trim earlier.
    _post_json(webhook, {"content": text[:_MAX_LEN]})


def _post_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    _post_json(
        url,
        {
            "chat_id": chat_id,
            "text": text[:_MAX_LEN],
            "disable_web_page_preview": True,
        },
    )


def training_event(title: str, body: str = "") -> None:
    """Send one message to any configured channel(s). Safe to call always."""
    d_url = (os.environ.get("DISCORD_WEBHOOK_URL") or "").strip()
    t_tok = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    t_chat = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
    name = (os.environ.get("RUN_NAME") or "").strip()

    has_discord = bool(d_url)
    has_telegram = bool(t_tok and t_chat)
    if not has_discord and not has_telegram:
        if t_tok or t_chat:
            print("notify: need both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or set DISCORD_WEBHOOK_URL. Skipping.")
        return

    msg = f"{name + ': ' if name else ''}{title}"
    if body:
        msg = f"{msg}\n{body.strip()}"
    if len(msg) > _MAX_LEN:
        msg = msg[: _MAX_LEN - 1] + "…"

    if d_url:
        try:
            _post_discord(d_url, msg)
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            print(f"notify: Discord failed (ignored): {type(e).__name__}")

    if t_tok and t_chat:
        try:
            _post_telegram(t_tok, t_chat, msg)
        except (urllib.error.URLError, socket.timeout, OSError) as e:
            print(f"notify: Telegram failed (ignored): {type(e).__name__}")
    elif t_tok or t_chat:
        print("notify: need both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (other channels still sent).")
