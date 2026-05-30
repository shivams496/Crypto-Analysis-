"""
zoro/alerts.py — Telegram and email alert senders

Both functions fail loudly (print warning) instead of silently.
"""
from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests

from .config import config


def send_telegram(message: str) -> bool:
    """
    Send *message* to the configured Telegram chat.
    Returns True on success, False on failure.
    """
    token   = config.TELEGRAM_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID

    if not token or not chat_id:
        print("[WARN] alerts.send_telegram: TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set in .env")
        return False

    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            print(f"[WARN] alerts.send_telegram: HTTP {resp.status_code} — {resp.text[:200]}")
            return False
        return True
    except requests.RequestException as e:
        print(f"[WARN] alerts.send_telegram: request failed — {e}")
        return False


def send_email(subject: str, body: str) -> bool:
    """
    Send email via Gmail SMTP.
    Returns True on success, False on failure.
    """
    sender   = config.GMAIL_ADDRESS
    password = config.GMAIL_APP_PW
    receiver = sender   # send to self by default

    if not sender or not password:
        print("[WARN] alerts.send_email: GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set in .env")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = receiver
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        return True
    except smtplib.SMTPAuthenticationError:
        print("[WARN] alerts.send_email: authentication failed — check GMAIL_APP_PASSWORD in .env")
        return False
    except Exception as e:
        print(f"[WARN] alerts.send_email: {e}")
        return False


def signal_message(symbol: str, direction: str, price: float,
                   confidence: int, gates: list[str], stop_loss: float) -> str:
    """Format a human-readable signal alert."""
    gate_str = " · ".join(gates) if gates else "none"
    emoji    = "🟢" if direction == "LONG" else ("🔴" if direction == "SHORT" else "⏳")
    return (
        f"{emoji} <b>ZORO SIGNAL — {symbol}</b>\n"
        f"Direction  : {direction}\n"
        f"Price      : ${price:,.2f}\n"
        f"Confidence : {confidence}/100\n"
        f"Stop Loss  : ${stop_loss:,.2f}\n"
        f"Gates      : {gate_str}"
    )
