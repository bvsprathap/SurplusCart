"""
agents/care_home_agent.py

Care Home A2A Negotiation Layer.

Two components:
1. CareHomeAgent — LLM-powered ADK Agent simulating a care home
   coordinator responding to food offers. Uses output_schema for
   structured CareHomeResponse output.

2. run_negotiation() — Deterministic orchestration function managing
   the full negotiation exchange for one care home. NOT an LLM agent.

This file handles negotiation only. Phase 2 processing (sourcing,
stock deduction, Order creation) is the ORCHESTRATOR's responsibility.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Set

import google.auth
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from data.data_model import CareHome, FoodCatalogItem
from tools.constraint_tools import StockLedger
from tools.logger import log_message
from tools.models import (
    CareHomeAction,
    CareHomeResponse,
    MatchmakerOffer,
    NegotiationResult,
    NegotiationTurn,
    OfferedItem,
    OrderLineItem,
    _REJECTION_MESSAGE,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vertex AI / GenAI configuration
# ---------------------------------------------------------------------------

if not os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") and not os.environ.get(
    "GOOGLE_GENAI_USE_ENTERPRISE"
):
    try:
        _, _project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = _project_id
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    except Exception:
        pass

_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(care_home: CareHome) -> str:
    """Build a parameterised system prompt for the care home agent."""

    constraints = []
    if care_home.hard_constraints.vegetarian_only:
        constraints.append(
            "This is a STRICTLY VEGETARIAN home. You must NEVER accept "
            "chicken, eggs, or any meat/fish products under any circumstances."
        )
    if care_home.hard_constraints.has_young_children:
        constraints.append(
            "This home has young children. Prioritise accepting dairy "
            "(milk, curd, paneer), eggs (if not vegetarian), and "
            "protein-rich items."
        )

    memory_lines = []
    for note in care_home.memory_notes:
        if note.type == "exclude":
            memory_lines.append(
                f"- You prefer NOT to receive {note.item} — politely "
                f"exclude it if offered."
            )
        elif note.type == "max_quantity" and note.value is not None:
            memory_lines.append(
                f"- You can accept at most {note.value} units of "
                f"{note.item} — reduce if offered more."
            )

    memory_section = ""
    if memory_lines:
        memory_section = (
            "\n\nYour preferences from past experience:\n"
            + "\n".join(memory_lines)
        )

    return f"""\
You are the coordinator of {care_home.name}, a care home in Chennai, Tamil Nadu.
You have {care_home.resident_count} residents and {care_home.storage_capacity_kg} kg of cold/dry storage capacity.

{chr(10).join(constraints) if constraints else "No special dietary restrictions."}
{memory_section}

You are receiving a food donation offer from the Food Rescue system.
Respond as a real care home coordinator would — warm, practical, and decisive.

For EACH turn, respond with one or more actions:
- accept_all: accept everything as offered (use when the offer is good)
- reduce_item: want less of a specific item (specify item and requested_quantity)
- exclude_item: do not need a specific item today (specify item)
- request_item: ask if another item is available (specify item name)
- flag_urgent: mark a specific item as critically needed today (specify item)
- reject_all: reject the entire offer (ONLY if the offer is completely unsuitable)

You may combine multiple actions in a single response. For example:
"accept most items, reduce rice to 10 kg, flag milk as urgent, exclude sugar"

Important rules:
- Be realistic about your storage capacity — refuse excess if it genuinely won't fit
- You may reduce quantities but do not increase beyond what was offered
- flag_urgent is for items you critically need today — use sparingly
- reject_all should be very rare — only for completely unsuitable offers
- Your rationale should be brief, natural language — like a real coordinator speaking
"""


# ---------------------------------------------------------------------------
# Agent runner helper (same pattern as matchmaker)
# ---------------------------------------------------------------------------

async def _run_care_home_turn(
    runner: Runner,
    session_id: str,
    message_text: str,
) -> CareHomeResponse:
    """Send a message to the care home agent, return structured response."""

    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message_text)],
    )

    last_nonempty: str = ""
    async for event in runner.run_async(
        user_id="system",
        session_id=session_id,
        new_message=user_content,
    ):
        if not (hasattr(event, "content") and event.content):
            continue
        parts = getattr(event.content, "parts", None) or []
        event_text = "".join(
            part.text for part in parts if getattr(part, "text", None)
        )
        if event_text:
            last_nonempty = event_text

    if not last_nonempty:
        raise RuntimeError("Care home agent returned no text response")

    # Parse the structured JSON response
    data = json.loads(last_nonempty)
    actions = [CareHomeAction(**a) for a in data.get("actions", [])]
    rationale = data.get("rationale", "")
    return CareHomeResponse(actions=actions, rationale=rationale)


# ---------------------------------------------------------------------------
# run_negotiation — deterministic orchestration
# ---------------------------------------------------------------------------

async def run_negotiation(
    care_home: CareHome,
    offer: MatchmakerOffer,
    ledger: StockLedger,
    catalog: List[FoodCatalogItem],
    run_id: str,
) -> NegotiationResult:
    """
    Manage the full negotiation exchange for one care home.

    This is deterministic orchestration code, NOT an LLM agent.
    The LLM is only used for the care home's responses.

    Parameters
    ----------
    care_home : CareHome profile from WorldConfig.
    offer     : MatchmakerOffer with [DISH_FRAMING] already replaced.
    ledger    : Current StockLedger (read-only — caller deducts after).
    catalog   : Full food catalog for is_essential lookups.
    run_id    : Simulation run identifier for logging.

    Returns
    -------
    NegotiationResult
    """
    catalog_map = {c.name.lower(): c for c in catalog}
    transcript: List[NegotiationTurn] = []
    turn_counter = 0

    # Build lookup: item_name_lower -> OfferedItem
    offered_map: Dict[str, OfferedItem] = {
        oi.item.lower(): oi for oi in offer.offered_items
    }

    # --- Create care home agent ---
    agent = Agent(
        name=f"care_home_{care_home.care_home_id}",
        model=_MODEL,
        instruction=_build_system_prompt(care_home),
        output_schema=CareHomeResponse,
    )

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name=f"negotiation_{care_home.care_home_id}",
    )

    session = await session_service.create_session(
        app_name=f"negotiation_{care_home.care_home_id}",
        user_id="system",
    )

    # ── Round 1: Send offer ──────────────────────────────────────────────────

    turn_counter += 1
    transcript.append(NegotiationTurn(
        turn_number=turn_counter, speaker="system", action="offer",
    ))
    log_message(
        to=care_home.care_home_id,
        channel="a2a_negotiation",
        content=f"[OFFER] {offer.offer_message}",
    )

    # Send to care home agent
    response = await _run_care_home_turn(
        runner, session.id, offer.offer_message,
    )

    # Log each action from the care home
    for action in response.actions:
        turn_counter += 1
        transcript.append(NegotiationTurn(
            turn_number=turn_counter,
            speaker="care_home",
            action=action.action,
            item=action.item,
            quantity=action.requested_quantity,
        ))

    log_message(
        to=care_home.care_home_id,
        channel="a2a_negotiation",
        content=f"[RESPONSE] {response.rationale}",
    )

    # ── Process round 1 actions ──────────────────────────────────────────────

    result = _process_actions(
        response=response,
        offered_map=offered_map,
        catalog_map=catalog_map,
        ledger=ledger,
        care_home=care_home,
        transcript=transcript,
        turn_counter=turn_counter,
        is_round_2=False,
    )

    # If rejected, return immediately
    if result["status"] == "rejected":
        log_message(
            to=care_home.care_home_id,
            channel="a2a_negotiation",
            content=f"[SYSTEM] {_REJECTION_MESSAGE}",
        )
        return NegotiationResult(
            care_home_id=care_home.care_home_id,
            status="rejected",
            agreed_items=[],
            urgent_item_names=[],
            negotiation_transcript=result["transcript"],
            rejection_message=_REJECTION_MESSAGE,
        )

    # ── Round 2: Re-offer if request_item was fulfilled ──────────────────────

    if result["needs_reoffer"] and result["new_items"]:
        turn_counter = result["turn_counter"]

        # Build revised offer message
        revised_items_desc = []
        for item_name, qty in result["new_items"].items():
            revised_items_desc.append(f"- {item_name}: {qty:.0f} units")

        revised_message = (
            f"Great news! We can also include the following items "
            f"you requested:\n"
            + "\n".join(revised_items_desc)
            + "\n\nDoes this work for you? You may accept, reduce, "
            "or exclude any of these additions."
        )

        turn_counter += 1
        result["transcript"].append(NegotiationTurn(
            turn_number=turn_counter, speaker="system", action="offer",
        ))
        log_message(
            to=care_home.care_home_id,
            channel="a2a_negotiation",
            content=f"[RE-OFFER] {revised_message}",
        )

        # Send revised offer
        r2_response = await _run_care_home_turn(
            runner, session.id, revised_message,
        )

        for action in r2_response.actions:
            turn_counter += 1
            result["transcript"].append(NegotiationTurn(
                turn_number=turn_counter,
                speaker="care_home",
                action=action.action,
                item=action.item,
                quantity=action.requested_quantity,
            ))

        log_message(
            to=care_home.care_home_id,
            channel="a2a_negotiation",
            content=f"[RESPONSE R2] {r2_response.rationale}",
        )

        # Process round 2 actions
        r2_result = _process_actions(
            response=r2_response,
            offered_map=result["agreed_map"],
            catalog_map=catalog_map,
            ledger=ledger,
            care_home=care_home,
            transcript=result["transcript"],
            turn_counter=turn_counter,
            is_round_2=True,
        )

        # Merge round 2 results
        result["agreed_map"] = r2_result["agreed_map"]
        result["urgent_names"] = result["urgent_names"] | r2_result["urgent_names"]
        result["transcript"] = r2_result["transcript"]
        result["turn_counter"] = r2_result["turn_counter"]

    # ── Build final NegotiationResult ────────────────────────────────────────

    agreed_items = _build_agreed_items(result["agreed_map"], offered_map)

    # ── Enforce hard constraints at logic level ──────────────────────────
    # The LLM is not reliable for hard dietary constraints.
    # Vegetarian enforcement: remove chicken/eggs if vegetarian_only.
    if care_home.hard_constraints.vegetarian_only:
        _NON_VEG = {"chicken", "eggs"}
        agreed_items = [
            it for it in agreed_items
            if it.item.lower() not in _NON_VEG
        ]

    # Deduplicate and sort urgent names
    urgent_sorted = sorted(result["urgent_names"])

    return NegotiationResult(
        care_home_id=care_home.care_home_id,
        status="agreed",
        agreed_items=agreed_items,
        urgent_item_names=urgent_sorted,
        negotiation_transcript=result["transcript"],
        rejection_message=None,
    )


# ---------------------------------------------------------------------------
# Action processing (deterministic)
# ---------------------------------------------------------------------------

def _process_actions(
    response: CareHomeResponse,
    offered_map: Dict[str, OfferedItem],
    catalog_map: Dict[str, FoodCatalogItem],
    ledger: StockLedger,
    care_home: CareHome,
    transcript: List[NegotiationTurn],
    turn_counter: int,
    is_round_2: bool,
) -> dict:
    """
    Process a CareHomeResponse's actions deterministically.

    Returns a dict with:
      status: "agreed" | "rejected"
      agreed_map: {item_lower: (OfferedItem, accepted_qty)}
      urgent_names: set of essential item names flagged urgent
      needs_reoffer: bool (only True if request_item was fulfilled)
      new_items: {item_name: qty} for newly added requested items
      transcript: updated transcript
      turn_counter: updated counter
    """
    # Start with all offered items accepted at full quantity.
    # offered_map may be Dict[str, OfferedItem] (round 1) or
    # Dict[str, (OfferedItem, float)] (round 2 — from prior agreed_map).
    agreed_map: Dict[str, tuple] = {}
    for name_lower, val in offered_map.items():
        if isinstance(val, tuple):
            oi, prev_qty = val
            agreed_map[name_lower] = (oi, prev_qty)
        else:
            # OfferedItem directly
            agreed_map[name_lower] = (val, val.offered_quantity)

    urgent_names: Set[str] = set()
    needs_reoffer = False
    new_items: Dict[str, float] = {}

    for action in response.actions:
        act = action.action

        # ── reject_all ───────────────────────────────────────────────────
        if act == "reject_all":
            return {
                "status": "rejected",
                "agreed_map": {},
                "urgent_names": set(),
                "needs_reoffer": False,
                "new_items": {},
                "transcript": transcript,
                "turn_counter": turn_counter,
            }

        # ── accept_all ───────────────────────────────────────────────────
        if act == "accept_all":
            # Everything stays as is — agreed_map already has full qty
            continue

        # ── flag_urgent ──────────────────────────────────────────────────
        if act == "flag_urgent" and action.item:
            item_lower = action.item.lower()
            cat = catalog_map.get(item_lower)
            # Only essential items can be urgent — silently ignore others
            if cat and cat.is_essential:
                urgent_names.add(action.item.lower())
            continue

        # ── reduce_item ──────────────────────────────────────────────────
        if act == "reduce_item" and action.item:
            item_lower = action.item.lower()
            if item_lower in agreed_map and action.requested_quantity is not None:
                oi, _ = agreed_map[item_lower]
                # Clamp to [0, offered_quantity]
                new_qty = max(0.0, min(
                    action.requested_quantity, oi.offered_quantity,
                ))
                agreed_map[item_lower] = (oi, new_qty)
            continue

        # ── exclude_item ─────────────────────────────────────────────────
        if act == "exclude_item" and action.item:
            item_lower = action.item.lower()
            agreed_map.pop(item_lower, None)
            continue

        # ── request_item ─────────────────────────────────────────────────
        if act == "request_item" and action.item:
            item_lower = action.item.lower()

            if is_round_2:
                # In round 2, log but do NOT fulfill further requests
                logger.info(
                    "Round 2 request_item '%s' from %s — not fulfilled",
                    action.item, care_home.care_home_id,
                )
                continue

            # Check if item is available in ledger
            cross_totals = ledger.get_cross_store_totals()
            available = cross_totals.get(item_lower, 0.0)

            if available <= 0:
                logger.info(
                    "Requested item '%s' not available in ledger for %s",
                    action.item, care_home.care_home_id,
                )
                continue

            # Estimate a reasonable quantity based on resident count
            cat = catalog_map.get(item_lower)
            unit = cat.unit if cat else "units"
            # Heuristic: ~0.5 units per resident, capped at available
            reasonable_qty = min(
                care_home.resident_count * 0.5,
                available,
            )
            reasonable_qty = max(1.0, reasonable_qty)

            is_essential = cat.is_essential if cat else False
            new_oi = OfferedItem(
                item=action.item,
                unit=unit,
                offered_quantity=reasonable_qty,
                is_essential=is_essential,
            )
            agreed_map[item_lower] = (new_oi, reasonable_qty)
            new_items[action.item] = reasonable_qty
            needs_reoffer = True
            continue

    return {
        "status": "agreed",
        "agreed_map": agreed_map,
        "urgent_names": urgent_names,
        "needs_reoffer": needs_reoffer,
        "new_items": new_items,
        "transcript": transcript,
        "turn_counter": turn_counter,
    }


def _build_agreed_items(
    agreed_map: Dict[str, tuple],
    original_offered_map: Dict[str, OfferedItem],
) -> List[OrderLineItem]:
    """Convert agreed_map to a list of OrderLineItem."""
    items = []
    for item_lower, (oi, accepted_qty) in agreed_map.items():
        # offered_quantity is what we originally offered (or what was added
        # via request_item)
        items.append(OrderLineItem(
            item=oi.item,
            unit=oi.unit,
            offered_quantity=oi.offered_quantity,
            accepted_quantity=accepted_qty,
        ))
    return items
