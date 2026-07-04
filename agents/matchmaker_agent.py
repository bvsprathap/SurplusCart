"""
agents/matchmaker_agent.py

Matchmaker LlmAgent — Phase 1 of the food rescue pipeline.
Produces a MatchmakerOffer for a single care home.

Architecture:
- Pre-processing (hard_constraint_filter, ledger snapshot) done by caller
- This agent ONLY reasons about quantities and builds the offer
- Post-processing (negotiation, single_store_candidate, Order creation)
  handled by the orchestrator in later phases
- Must NOT call single_store_candidate, deduct from StockLedger,
  or create Order objects

Usage:
    offer = await run_matchmaker(
        care_home=care_home,
        eligible_items=filtered_items,
        catalog=catalog,
        cross_store_totals=ledger.get_cross_store_totals(),
        cross_store_snapshot=dict(cross_store_totals),
        excluded_item_names=["chicken", "eggs"],
    )
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List
from pydantic import BaseModel

import google.auth
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from data.data_model import CareHome, DailyFoodItem, FoodCatalogItem
from tools.guardrails import MatchmakerOfferGuardrail
from tools.models import MatchmakerOffer, OfferedItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class OfferedItemSchema(BaseModel):
    item: str
    unit: str
    offered_quantity: float
    is_essential: bool


class MatchmakerResponseSchema(BaseModel):
    offered_items: List[OfferedItemSchema]
    rationale: str
    offer_message: str
    expected_today_statement: str


# ---------------------------------------------------------------------------
# Vertex AI / GenAI configuration
# ---------------------------------------------------------------------------
# Match the pattern from agents/agent.py — use Vertex AI with ADC.
# Safe to re-set if already configured.
if not os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") and not os.environ.get("GOOGLE_GENAI_USE_ENTERPRISE"):
    try:
        _, _project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = _project_id
        os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    except Exception:
        pass  # Tests may configure via GOOGLE_API_KEY instead

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard urgency request text — never LLM-generated.
# Only mentions essential items.
URGENCY_REQUEST_TEXT = (
    "Please let us know which of the essential items listed above "
    "are most critical for you today so we can prioritise delivery."
)

# Gemini model for the Matchmaker agent
_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# System instruction — static, shared across all care-home runs
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = """\
You are a Food Rescue Matchmaker agent for a food rescue coordination \
system operating in Chennai, India.

Your job: analyse available surplus food inventory and build an \
optimal donation offer for a single care home.

## Reasoning Guidelines

1. **Expiry urgency**: Offer items with the fewest days_to_expiry \
first — they must be moved before they spoil.

2. **Memory-note caps**: If the care home has a max_quantity \
memory note for an item, cap your offered quantity at that value. \
Mention the cap and reason briefly in your rationale.

3. **Right-sizing**: Scale quantities to the care home's size. \
A home with 30 residents needs less than one with 60. Never \
offer more than storage_capacity_kg allows in total weight. \
Use 1 kg ≈ 1 unit as a rough weight approximation.

4. **Children priority**: If has_young_children is true, \
prioritise milk and eggs even if other items have lower expiry. \
Children's nutrition is critical.

5. **No padding**: Only include items the care home can use. \
Do not add low-value items just to clear stock.

6. **Essential awareness**: Mark is_essential correctly per \
the catalog classification provided.

## Output Format

Return ONLY a JSON object with exactly these fields:

{
  "offered_items": [
    {"item": "<name>", "unit": "<kg or units>", \
"offered_quantity": <positive number>, "is_essential": <true or false>}
  ],
  "rationale": "<Brief explanation of decisions — mention any \
caps applied, priority reasoning, and trade-offs>",
  "offer_message": "<Warm, professional message listing items, \
quantities, and units. MUST contain the literal placeholder \
[DISH_FRAMING] on its own line where dish suggestions will be \
inserted later by another agent.>",
  "expected_today_statement": "<What we realistically expect \
to deliver today based on the available stock>"
}

## CRITICAL RULES
- Every offered_quantity MUST be > 0.
- No offered_quantity may exceed the available quantity.
- offer_message MUST contain the literal placeholder [DISH_FRAMING].
- is_essential must match the catalog flag for each item exactly.
- Only offer items from the eligible list — never invent items.
"""


# ---------------------------------------------------------------------------
# Auto-accept path  (STEP 4 — no LLM)
# ---------------------------------------------------------------------------

def _build_auto_accept_offer(
    care_home: CareHome,
    eligible_items: List[DailyFoodItem],
    catalog: List[FoodCatalogItem],
    cross_store_totals: Dict[str, float],
) -> MatchmakerOffer:
    """
    Deterministic offer for non-negotiating care homes (home_05).
    Offered_quantity = full remaining quantity per eligible item.
    No Gemini call — fast and deterministic.
    """
    catalog_map = {c.name.lower(): c for c in catalog}

    offered: List[OfferedItem] = []
    for item in eligible_items:
        qty = cross_store_totals.get(item.name.lower(), 0.0)
        if qty <= 0:
            continue
        cat = catalog_map.get(item.name.lower())
        offered.append(OfferedItem(
            item=item.name,
            unit=item.unit,
            offered_quantity=qty,
            is_essential=cat.is_essential if cat else False,
        ))

    return MatchmakerOffer(
        care_home_id=care_home.care_home_id,
        offered_items=offered,
        rationale=(
            "Auto-accept: all remaining eligible items offered "
            "at full available quantity."
        ),
        offer_message=(
            "[Auto-accept] Full remaining inventory offered to "
            f"{care_home.name}."
        ),
        expected_today_statement=(
            "All listed items expected for delivery today."
        ),
        urgency_request="",  # Not applicable for auto-accept homes
    )


# ---------------------------------------------------------------------------
# User-message builder (dynamic context per care home)
# ---------------------------------------------------------------------------

def _build_user_message(
    care_home: CareHome,
    eligible_items: List[DailyFoodItem],
    catalog: List[FoodCatalogItem],
    cross_store_totals: Dict[str, float],
) -> str:
    """Build the user-message payload the LLM will reason over."""
    catalog_map = {c.name.lower(): c for c in catalog}

    lines = [
        "## Care Home Profile",
        f"- Name: {care_home.name}",
        f"- ID: {care_home.care_home_id}",
        f"- Residents: {care_home.resident_count}",
        f"- Storage capacity: {care_home.storage_capacity_kg} kg",
        f"- Vegetarian only: {care_home.hard_constraints.vegetarian_only}",
        f"- Has young children: {care_home.hard_constraints.has_young_children}",
    ]

    if care_home.memory_notes:
        lines.append("- Memory notes:")
        for note in care_home.memory_notes:
            if note.type == "max_quantity":
                lines.append(f"  - {note.item}: max {note.value}")
            elif note.type == "exclude":
                lines.append(
                    f"  - {note.item}: excluded (already filtered out)"
                )

    lines.extend(["", "## Eligible Items (cross-store aggregate)", ""])

    for item in eligible_items:
        cat = catalog_map.get(item.name.lower())
        is_essential = cat.is_essential if cat else False
        qty = cross_store_totals.get(item.name.lower(), item.quantity)
        lines.append(
            f"- {item.name}: {qty:.1f} {item.unit}, "
            f"expires in {item.days_to_expiry} day(s), "
            f"essential={is_essential}"
        )

    lines.extend(["", "Build the offer now."])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _safe_json_loads(text: str) -> dict:
    """
    Parse JSON from LLM response, repairing common issues.

    Gemini sometimes returns JSON with unescaped newlines inside string
    values (e.g. offer_message). This function tries multiple strategies.
    """
    import re

    # First, try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract the outermost { ... } block (skip any prefix/suffix)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        extracted = text[brace_start:brace_end + 1]
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

        # Strategy 3: fix unescaped newlines inside string values
        repaired = re.sub(
            r'"((?:[^"\\]|\\.)*?)"',
            lambda m: '"' + m.group(1).replace("\n", "\\n").replace("\r", "\\r") + '"',
            extracted,
            flags=re.DOTALL,
        )
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Strategy 4: strip markdown code fences
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    logger.error(
        "JSON parse failed after all repair strategies. "
        "Raw response (first 500 chars): %s",
        text[:500],
    )
    raise json.JSONDecodeError(
        "Could not parse LLM response as JSON after repair attempts",
        text, 0,
    )


def _parse_offer_response(
    care_home_id: str,
    response_text: str,
    catalog: List[FoodCatalogItem],
) -> MatchmakerOffer:
    """Parse LLM JSON response into a MatchmakerOffer."""
    data = _safe_json_loads(response_text)

    catalog_map = {c.name.lower(): c for c in catalog}

    offered_items: List[OfferedItem] = []
    for item_data in data.get("offered_items", []):
        cat = catalog_map.get(item_data["item"].lower())
        # Force is_essential to match catalog — don't trust LLM for this
        is_essential = cat.is_essential if cat else item_data.get("is_essential", False)
        offered_items.append(OfferedItem(
            item=item_data["item"],
            unit=item_data["unit"],
            offered_quantity=float(item_data["offered_quantity"]),
            is_essential=is_essential,
        ))

    offer_message = data.get("offer_message", "")
    # Safety net: inject [DISH_FRAMING] if LLM forgot it
    if "[DISH_FRAMING]" not in offer_message:
        offer_message += "\n\n[DISH_FRAMING]"

    return MatchmakerOffer(
        care_home_id=care_home_id,
        offered_items=offered_items,
        rationale=data.get("rationale", ""),
        offer_message=offer_message,
        expected_today_statement=data.get("expected_today_statement", ""),
        urgency_request=URGENCY_REQUEST_TEXT,  # Always standard text
    )


# ---------------------------------------------------------------------------
# Agent runner helper
# ---------------------------------------------------------------------------

async def _run_agent_turn(
    runner: Runner,
    session_id: str,
    content: types.Content,
) -> str:
    """Run one turn of the Matchmaker agent and return the final text.

    ADK streams events; the model's final complete response is emitted as a
    single event whose content.parts[0].text contains the full JSON.
    We track the last non-empty text seen — which is that final complete
    response — rather than concatenating incremental chunks (which would
    produce double-output on non-streaming models) or overwriting with the
    last part (which could be an incomplete fragment if the model yields
    multiple parts in a single content block).
    """
    last_nonempty: str = ""
    async for event in runner.run_async(
        user_id="system",
        session_id=session_id,
        new_message=content,
    ):
        if not (hasattr(event, "content") and event.content):
            continue
        parts = getattr(event.content, "parts", None) or []
        # Collect all text within this single event's parts
        event_text = "".join(
            part.text
            for part in parts
            if getattr(part, "text", None)
        )
        if event_text:
            last_nonempty = event_text

    if not last_nonempty:
        raise RuntimeError("Matchmaker agent returned no text response")
    return last_nonempty


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_matchmaker(
    care_home: CareHome,
    eligible_items: List[DailyFoodItem],
    catalog: List[FoodCatalogItem],
    cross_store_totals: Dict[str, float],
    cross_store_snapshot: Dict[str, float],
    excluded_item_names: List[str],
) -> MatchmakerOffer:
    """
    Build a MatchmakerOffer for a single care home.

    Called by the orchestrator in fixed sequential order
    (home_01 → home_02 → home_03 → home_04 → home_05).
    Each run sees the ledger as it stands after all previous homes
    have negotiated.

    Pre-processing (hard_constraint_filter, ledger snapshot) is the
    caller's responsibility. Post-processing (negotiation, sourcing,
    Order creation) happens after this function returns.

    Parameters
    ----------
    care_home          : CareHome profile from WorldConfig.
    eligible_items     : Output of hard_constraint_filter — only items
                         this care home can receive.
    catalog            : Full FoodCatalogItem list for is_essential lookup.
    cross_store_totals : Aggregated cross-store view from
                         ledger.get_cross_store_totals().
    cross_store_snapshot: Frozen copy of cross_store_totals for guardrail
                         validation (taken before this call).
    excluded_item_names: Item names removed by hard_constraint_filter
                         (for guardrail validation).

    Returns
    -------
    MatchmakerOffer
    """
    # --- Auto-accept path (STEP 4) ---
    if not care_home.negotiates_via_a2a:
        logger.info(
            "Auto-accept path for %s — skipping LLM",
            care_home.care_home_id,
        )
        return _build_auto_accept_offer(
            care_home, eligible_items, catalog, cross_store_totals,
        )

    # --- LLM path (STEPS 2-3) ---
    logger.info(
        "Running Matchmaker LLM for %s (%d eligible items)",
        care_home.care_home_id,
        len(eligible_items),
    )

    agent = Agent(
        name="matchmaker",
        model=_MODEL,
        instruction=_SYSTEM_INSTRUCTION,
        output_schema=MatchmakerResponseSchema,
    )

    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name="matchmaker_run",
    )

    session = await session_service.create_session(
        app_name="matchmaker_run",
        user_id="system",
    )

    user_message = _build_user_message(
        care_home, eligible_items, catalog, cross_store_totals,
    )
    user_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    # --- First attempt ---
    response_text = await _run_agent_turn(
        runner, session.id, user_content,
    )
    offer = _parse_offer_response(
        care_home.care_home_id, response_text, catalog,
    )

    # --- Validate (STEP 3) ---
    try:
        MatchmakerOfferGuardrail(
            offer=offer,
            cross_store_snapshot=cross_store_snapshot,
            excluded_item_names=excluded_item_names,
        )
        logger.info(
            "Offer for %s passed validation on first attempt",
            care_home.care_home_id,
        )
        return offer
    except Exception as first_error:
        logger.warning(
            "Offer for %s failed validation: %s — retrying",
            care_home.care_home_id,
            first_error,
        )

    # --- ONE retry with error feedback ---
    error_content = types.Content(
        role="user",
        parts=[types.Part.from_text(
            text=(
                f"Your offer FAILED validation with this error:\n"
                f"{first_error}\n\n"
                f"Please fix the issues and resubmit the corrected "
                f"JSON offer. Remember: all quantities must be > 0 "
                f"and must not exceed available stock."
            ),
        )],
    )

    retry_text = await _run_agent_turn(
        runner, session.id, error_content,
    )
    offer = _parse_offer_response(
        care_home.care_home_id, retry_text, catalog,
    )

    try:
        MatchmakerOfferGuardrail(
            offer=offer,
            cross_store_snapshot=cross_store_snapshot,
            excluded_item_names=excluded_item_names,
        )
        logger.info(
            "Offer for %s passed validation on retry",
            care_home.care_home_id,
        )
        return offer
    except Exception as retry_error:
        raise RuntimeError(
            f"Matchmaker offer for {care_home.care_home_id} failed "
            f"validation after retry.\n"
            f"First error: {first_error}\n"
            f"Retry error: {retry_error}"
        ) from retry_error
