"""
Send (or, by default, mock-send) the flag email.

By default this writes a .eml file to outbox/ so the full flow can be
demoed and inspected with zero external credentials, and with zero risk
of an actual email reaching a real inbox during testing. Setting
SMTP_HOST (see .env.example) switches to real delivery via SMTP -- this
is a conscious opt-in, not something that happens by having the code
installed, since sending real email on someone's behalf should always be
a deliberate choice.

Real delivery also still writes a copy into outbox/ (prefixed "sent-")
so there is always a local audit record of what actually went out,
consistent with the audit trail the rest of the pipeline keeps.
"""

from __future__ import annotations

import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from src.tools.email_draft import DraftEmail

OUTBOX_DIR = Path(__file__).resolve().parent.parent.parent / "outbox"


def _send_via_smtp(email: DraftEmail) -> None:
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    sender = os.environ.get("SMTP_FROM", user)

    if not user or not password:
        raise RuntimeError(
            "SMTP_HOST is set but SMTP_USER / SMTP_PASSWORD are missing -- "
            "see .env.example for what's required."
        )

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = email.to
    msg["Subject"] = email.subject
    msg.set_content(email.body)

    with smtplib.SMTP(host, port, timeout=20) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)


def send_email(email: DraftEmail, match_id: int) -> Path:
    OUTBOX_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    smtp_configured = bool(os.environ.get("SMTP_HOST"))
    if smtp_configured:
        _send_via_smtp(email)
        filepath = OUTBOX_DIR / f"sent-match-{match_id}_{timestamp}.eml"
    else:
        filepath = OUTBOX_DIR / f"match-{match_id}_{timestamp}.eml"

    filepath.write_text(
        f"To: {email.to}\nSubject: {email.subject}\n\n{email.body}",
        encoding="utf-8",
    )
    return filepath
