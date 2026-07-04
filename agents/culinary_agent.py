"""
agents/culinary_agent.py

Culinary Agent — Phase 1.5 of the food rescue pipeline.
Enriches the Matchmaker's offer by grouping items into plausible dish suggestions.
Replaces the [DISH_FRAMING] placeholder in offer_message.

Usage:
    dish_framing = await run_culinary(offered_items)
    final_message = offer.offer_message.replace("[DISH_FRAMING]", dish_framing)
"""

from __future__ import annotations

import logging
import os
from typing import List

import google.auth
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from tools.models import OfferedItem

logger = logging.getLogger(__name__)

# Vertex AI config
if not os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") and not os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE"):
    try:
        _, _project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = _project_id
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    except Exception:
        pass

_MODEL = "gemini-2.5-flash"

_SYSTEM_INSTRUCTION = """\
You are a Culinary Assistant for a food rescue system in Chennai, Tamil Nadu, India.
Your job is to suggest practical, culturally appropriate dishes that care homes can prepare using a list of offered surplus ingredients.

Rules:
1. Only suggest dishes where AT LEAST 2 of the key ingredients are present in the offered list.
2. Group the items into 2-3 dish suggestions maximum.
3. If a dish is suggested, name the dish (e.g., Sambar, Poriyal, Rasam, Pongal, Kootu, Veg Biryani), list which offered items it uses, and give one brief sentence of practical context (e.g., "Good for a nutritious breakfast for the children").
4. Items that don't fit any dish must be listed plainly at the end under a line starting with "also available: " followed by the item names, separated by commas.
5. Only use items from the provided offered list. Never invent or assume other ingredients are provided.
6. Never mention quantities or units.
7. Output plain text only. Do not use JSON. Do not use markdown headers (like #, ##, or ###) or bold text (like **). Do not use bullet points or lists unless requested. Keep it flowing as plain paragraphs.
8. If there are fewer than 2 items, or if the items cannot reasonably be combined into any dish, do not suggest any dishes. Simply list all items as "also available: X, Y".
9. Ensure the suggested dishes match the dietary constraints implied (e.g., if there are no eggs or meat, do not suggest meat/egg dishes; if the list is veg-only, suggest purely vegetarian South Indian dishes).
"""

async def run_culinary(offered_items: List[OfferedItem]) -> str:
    """
    Take a list of OfferedItem and generate Chennai culturally-appropriate dish suggestions.
    
    Returns:
        A plain text string to replace [DISH_FRAMING] in offer_message.
    """
    if not offered_items:
        return "also available: none"

    # Quick check: if we have fewer than 2 items, we cannot combine. Graceful fallback.
    if len(offered_items) < 2:
        items_str = ", ".join(item.item for item in offered_items)
        return f"also available: {items_str}"

    agent = Agent(
        name="culinary",
        model=_MODEL,
        instruction=_SYSTEM_INSTRUCTION,
    )

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name="culinary_run",
    )

    session = await session_service.create_session(
        app_name="culinary_run",
        user_id="system",
    )

    # Build input prompt
    items_list = []
    for item in offered_items:
        items_list.append(f"- {item.item} (is_essential={item.is_essential})")
    
    user_message = "Here is the list of offered ingredients:\n" + "\n".join(items_list) + "\n\nSuggest dishes and list remaining items under 'also available'."

    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    last_nonempty: str = ""
    async for event in runner.run_async(
        user_id="system",
        session_id=session.id,
        new_message=user_content,
    ):
        if not (hasattr(event, "content") and event.content):
            continue
        parts = getattr(event.content, "parts", None) or []
        event_text = "".join(
            part.text
            for part in parts
            if getattr(part, "text", None)
        )
        if event_text:
            last_nonempty = event_text

    if not last_nonempty:
        # Fallback in case of empty LLM response
        items_str = ", ".join(item.item for item in offered_items)
        return f"also available: {items_str}"

    return last_nonempty.strip()
