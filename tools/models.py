"""
tools/models.py

Pipeline output Pydantic models shared across all agents.
No LLM calls anywhere in this file.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Negotiation transcript
# ---------------------------------------------------------------------------

class NegotiationTurn(BaseModel):
    turn_number: int
    speaker: Literal["system", "care_home"]
    action: Literal[
        "offer",
        "accept_all",
        "reduce_item",
        "exclude_item",
        "request_item",
        "flag_urgent",
        "reject_all",       # Care home rejects the entire offer outright.
                            # System responds: "Noted. Will connect with you
                            # on another day." — no revised offer is made.
    ]
    item: Optional[str] = None
    quantity: Optional[float] = None


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class OrderLineItem(BaseModel):
    item: str
    unit: str
    offered_quantity: float
    accepted_quantity: float


class Order(BaseModel):
    order_id: str
    care_home_id: str
    store_id: str
    items: List[OrderLineItem]
    urgent_essential_items: List[str] = []
    negotiation_transcript: List[NegotiationTurn] = []
    final_notice: Dict[str, Any] = {}  # {arriving_today, deferred, message}


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

class Delivery(BaseModel):
    delivery_id: str
    store_id: str
    # Hard constraint: at most 2 orders per delivery run
    order_ids: Annotated[List[str], Field(max_length=2)]
    method: Literal["volunteer", "store_truck", "commercial"]
    volunteer_id: Optional[str] = None
    pickup_time: Optional[str] = None


# ---------------------------------------------------------------------------
# Matchmaker offer (Phase 1 output)
# ---------------------------------------------------------------------------

class OfferedItem(BaseModel):
    item: str
    unit: str
    offered_quantity: float
    is_essential: bool  # from catalog — used by Culinary agent later


class MatchmakerOffer(BaseModel):
    care_home_id: str
    offered_items: List[OfferedItem]
    rationale: str                  # brief natural-language explanation
    offer_message: str              # full message to care home; contains
                                    # [DISH_FRAMING] placeholder for Culinary
    expected_today_statement: str   # what we expect to deliver today
    urgency_request: str            # standard text asking care home to flag
                                    # their most urgent essential items


# ---------------------------------------------------------------------------
# Care Home A2A response (Phase 2 input — LLM output schema)
# ---------------------------------------------------------------------------

class CareHomeAction(BaseModel):
    action: Literal[
        "accept_all",
        "reduce_item",
        "exclude_item",
        "request_item",
        "flag_urgent",
        "reject_all",
    ]
    item: Optional[str] = None
    requested_quantity: Optional[float] = None


class CareHomeResponse(BaseModel):
    """Structured response from the care home LLM agent.
    Used as output_schema on the ADK Agent."""
    actions: List[CareHomeAction]
    rationale: str  # brief natural language explanation


# ---------------------------------------------------------------------------
# Negotiation result (output of run_negotiation)
# ---------------------------------------------------------------------------

_REJECTION_MESSAGE = "Noted. Will connect with you on another day."


class NegotiationResult(BaseModel):
    """Result of a complete negotiation exchange for one care home.

    Essential-only validation for urgent_item_names is enforced at
    the logic level in run_negotiation(), not as a model validator,
    because the model has no access to the food catalog.
    """
    care_home_id: str
    status: Literal["agreed", "rejected"]
    agreed_items: List[OrderLineItem] = []
    # Deduplicated, sorted list of essential items flagged as urgent.
    # Uses List[str] instead of Set[str] for JSON serialization safety.
    urgent_item_names: List[str] = []
    negotiation_transcript: List[NegotiationTurn] = []
    rejection_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Dispatch statistics (output of run_dispatch)
# ---------------------------------------------------------------------------

class DispatchStats(BaseModel):
    """Aggregate statistics for a single dispatch run."""
    total_deliveries: int = 0
    volunteer_assigned: int = 0
    store_truck_assigned: int = 0
    commercial_assigned: int = 0
    volunteers_unavailable: int = 0          # count who were unavailable today
    urgent_items_forced_fallback: int = 0    # deliveries with urgent items that
                                             # needed store truck or commercial
    detours_bundled: int = 0                 # successful 15-min bundles
