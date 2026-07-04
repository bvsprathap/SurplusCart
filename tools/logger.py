"""
tools/logger.py

In-memory message logger for the food rescue simulation pipeline.
No LLM calls anywhere here.

Channels:
  "whatsapp_simulated"  — messages to volunteers / stores
  "a2a_negotiation"     — care-home negotiation messages
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler

_log: List[Dict] = []
_lock = threading.Lock()
_current_run_id: Optional[str] = None


def setup_cloud_logging(run_id: str):
    try:
        client = google.cloud.logging.Client(project="decent-rampart-500008-n6")
        handler = CloudLoggingHandler(client, name="food-to-go-agent")
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)
    except Exception:
        pass  # Cloud logging optional for local runs


def set_run_id(run_id: str) -> None:
    """
    Set the run_id for all subsequent log entries.
    Call this at the start of each simulation run.
    """
    global _current_run_id
    with _lock:
        _current_run_id = run_id


def log_message(to: str, channel: str, content: str) -> None:
    """
    Append a structured log entry.

    Args:
        to:      Recipient identifier (volunteer_id, store_id, care_home_id,
                 or a human-readable name).
        channel: "whatsapp_simulated" | "a2a_negotiation"
        content: The message body (plain text or JSON string).
    """
    entry: Dict = {
        "recipient": to,
        "channel": channel,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": _current_run_id,
    }
    with _lock:
        _log.append(entry)

    try:
        logging.info(json.dumps({
            "run_id": _current_run_id,
            "recipient": to,
            "channel": channel,
            "content": content,
            "timestamp": entry["timestamp"]
        }))
    except Exception:
        pass  # Cloud logging optional for local runs


def get_message_log() -> List[Dict]:
    """
    Return all messages logged this run.
    Returns a shallow copy so the caller can't accidentally mutate the log.
    """
    with _lock:
        return list(_log)


def clear_log() -> None:
    """
    Clear all log entries. Call at the start of each new simulation run
    (after set_run_id, before any pipeline work begins).
    """
    with _lock:
        _log.clear()
