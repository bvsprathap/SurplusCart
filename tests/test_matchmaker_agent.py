"""
tests/test_matchmaker_agent.py

Test suite for the Matchmaker agent and its guardrails.

Structure:
  - TestMatchmakerOfferGuardrail  (pure Python — no LLM)
  - TestAutoAcceptPath            (pure Python — no LLM)
  - TestMatchmakerLLM             (live Gemini — skipped if no auth)

Run:  .venv/Scripts/pytest tests/test_matchmaker_agent.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.data_model import (
    CareHome,
    DailyFoodItem,
    FoodCatalogItem,
    HardConstraints,
    MemoryNote,
)
from tools.constraint_tools import hard_constraint_filter
from tools.guardrails import MatchmakerOfferGuardrail
from tools.models import MatchmakerOffer, OfferedItem


# ---------------------------------------------------------------------------
# Check LLM availability
# ---------------------------------------------------------------------------

def _llm_available() -> bool:
    """Return True if either Vertex AI ADC or API key is configured."""
    try:
        import google.auth
        _, _ = google.auth.default()
        return True
    except Exception:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return bool(key and not key.startswith("your_"))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _catalog() -> list[FoodCatalogItem]:
    return [
        FoodCatalogItem(name="milk",    is_perishable=True,  is_essential=True,  push_threshold_days=1, unit="units", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="chicken", is_perishable=True,  is_essential=True,  push_threshold_days=1, unit="kg", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="eggs",    is_perishable=True,  is_essential=True,  push_threshold_days=1, unit="units", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="rice",    is_perishable=False, is_essential=True,  push_threshold_days=7, unit="kg", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="sugar",   is_perishable=False, is_essential=False, push_threshold_days=7, unit="kg", approx_weight_kg=1.0, cap_category="test"),
    ]


def _make_care_home(
    care_home_id: str = "home_01",
    name: str = "Test Home",
    vegetarian: bool = False,
    has_young_children: bool = False,
    negotiates_via_a2a: bool = True,
    memory_notes: list[MemoryNote] | None = None,
    resident_count: int = 30,
    storage_capacity_kg: float = 200.0,
) -> CareHome:
    return CareHome(
        care_home_id=care_home_id,
        name=name,
        latitude=13.0,
        longitude=80.2,
        hard_constraints=HardConstraints(
            vegetarian_only=vegetarian,
            has_young_children=has_young_children,
        ),
        resident_count=resident_count,
        storage_capacity_kg=storage_capacity_kg,
        negotiates_via_a2a=negotiates_via_a2a,
        memory_notes=memory_notes or [],
    )


def _make_offer(
    items: list[OfferedItem] | None = None,
    care_home_id: str = "home_01",
    offer_message: str = "Offer.\n\n[DISH_FRAMING]\n\nThank you.",
) -> MatchmakerOffer:
    if items is None:
        items = [OfferedItem(item="milk", unit="units",
                             offered_quantity=20.0, is_essential=True)]
    return MatchmakerOffer(
        care_home_id=care_home_id,
        offered_items=items,
        rationale="Test rationale",
        offer_message=offer_message,
        expected_today_statement="All items expected today",
        urgency_request=(
            "Please let us know which of the essential items "
            "listed above are most critical for you today so "
            "we can prioritise delivery."
        ),
    )


def _eligible_items() -> list[DailyFoodItem]:
    """A small set of eligible items for tests."""
    return [
        DailyFoodItem(name="milk",  days_to_expiry=1, quantity=50.0,  unit="units"),
        DailyFoodItem(name="rice",  days_to_expiry=5, quantity=30.0,  unit="kg"),
        DailyFoodItem(name="eggs",  days_to_expiry=1, quantity=40.0,  unit="units"),
        DailyFoodItem(name="sugar", days_to_expiry=6, quantity=10.0,  unit="kg"),
    ]


def _cross_store_totals() -> dict[str, float]:
    return {"milk": 50.0, "rice": 30.0, "eggs": 40.0, "sugar": 10.0}


# ===========================================================================
# 1. MatchmakerOfferGuardrail  (pure Python)
# ===========================================================================

class TestMatchmakerOfferGuardrail:

    def test_valid_offer_passes(self):
        offer = _make_offer()
        guard = MatchmakerOfferGuardrail(
            offer=offer,
            cross_store_snapshot={"milk": 50.0},
        )
        assert guard.offer is offer

    def test_rejects_negative_offered_quantity(self):
        items = [OfferedItem(item="milk", unit="units",
                             offered_quantity=-5.0, is_essential=True)]
        with pytest.raises(Exception, match="must be > 0"):
            MatchmakerOfferGuardrail(offer=_make_offer(items=items))

    def test_rejects_zero_offered_quantity(self):
        items = [OfferedItem(item="milk", unit="units",
                             offered_quantity=0.0, is_essential=True)]
        with pytest.raises(Exception, match="must be > 0"):
            MatchmakerOfferGuardrail(offer=_make_offer(items=items))

    def test_rejects_quantity_exceeding_snapshot(self):
        items = [OfferedItem(item="milk", unit="units",
                             offered_quantity=100.0, is_essential=True)]
        with pytest.raises(Exception, match="exceeds cross-store"):
            MatchmakerOfferGuardrail(
                offer=_make_offer(items=items),
                cross_store_snapshot={"milk": 50.0},
            )

    def test_quantity_within_snapshot_passes(self):
        items = [OfferedItem(item="milk", unit="units",
                             offered_quantity=30.0, is_essential=True)]
        guard = MatchmakerOfferGuardrail(
            offer=_make_offer(items=items),
            cross_store_snapshot={"milk": 50.0},
        )
        assert guard.offer is not None

    def test_rejects_excluded_item(self):
        items = [OfferedItem(item="chicken", unit="kg",
                             offered_quantity=10.0, is_essential=True)]
        with pytest.raises(Exception, match="excluded"):
            MatchmakerOfferGuardrail(
                offer=_make_offer(items=items),
                excluded_item_names=["chicken"],
            )

    def test_non_excluded_item_passes(self):
        items = [OfferedItem(item="milk", unit="units",
                             offered_quantity=20.0, is_essential=True)]
        guard = MatchmakerOfferGuardrail(
            offer=_make_offer(items=items),
            excluded_item_names=["chicken"],
        )
        assert guard.offer is not None

    def test_case_insensitive_exclusion(self):
        items = [OfferedItem(item="Chicken", unit="kg",
                             offered_quantity=10.0, is_essential=True)]
        with pytest.raises(Exception, match="excluded"):
            MatchmakerOfferGuardrail(
                offer=_make_offer(items=items),
                excluded_item_names=["chicken"],
            )


# ===========================================================================
# 2. Auto-accept path  (pure Python — no LLM)
# ===========================================================================

class TestAutoAcceptPath:

    @pytest.mark.asyncio
    async def test_produces_valid_offer_without_llm(self):
        from agents.matchmaker_agent import run_matchmaker

        home = _make_care_home(
            care_home_id="home_05",
            negotiates_via_a2a=False,
        )
        eligible = _eligible_items()
        totals = _cross_store_totals()

        offer = await run_matchmaker(
            care_home=home,
            eligible_items=eligible,
            catalog=_catalog(),
            cross_store_totals=totals,
            cross_store_snapshot=dict(totals),
            excluded_item_names=[],
        )

        assert isinstance(offer, MatchmakerOffer)
        assert offer.care_home_id == "home_05"
        assert len(offer.offered_items) > 0
        assert "Auto-accept" in offer.rationale

    @pytest.mark.asyncio
    async def test_auto_accept_quantities_match_remaining(self):
        from agents.matchmaker_agent import run_matchmaker

        home = _make_care_home(negotiates_via_a2a=False)
        totals = {"milk": 25.0, "rice": 15.0}
        eligible = [
            DailyFoodItem(name="milk", days_to_expiry=1, quantity=25.0, unit="units"),
            DailyFoodItem(name="rice", days_to_expiry=5, quantity=15.0, unit="kg"),
        ]

        offer = await run_matchmaker(
            care_home=home,
            eligible_items=eligible,
            catalog=_catalog(),
            cross_store_totals=totals,
            cross_store_snapshot=dict(totals),
            excluded_item_names=[],
        )

        offered_map = {it.item: it.offered_quantity for it in offer.offered_items}
        assert offered_map["milk"] == 25.0
        assert offered_map["rice"] == 15.0

    @pytest.mark.asyncio
    async def test_vegetarian_auto_accept_no_chicken_eggs(self):
        """
        End-to-end: filter through hard_constraint_filter (removes
        chicken and eggs for vegetarian home) then auto-accept.
        """
        from agents.matchmaker_agent import run_matchmaker

        home = _make_care_home(vegetarian=True, negotiates_via_a2a=False)
        all_items = [
            DailyFoodItem(name="milk",    days_to_expiry=1, quantity=50.0,  unit="units"),
            DailyFoodItem(name="chicken", days_to_expiry=1, quantity=20.0,  unit="kg"),
            DailyFoodItem(name="eggs",    days_to_expiry=1, quantity=40.0,  unit="units"),
            DailyFoodItem(name="rice",    days_to_expiry=5, quantity=30.0,  unit="kg"),
        ]

        # Pre-process: filter (caller's responsibility)
        eligible = hard_constraint_filter(home, all_items, _catalog())
        excluded = ["chicken", "eggs"]
        # Cross-store totals only for eligible items
        totals = {it.name.lower(): it.quantity for it in eligible}

        offer = await run_matchmaker(
            care_home=home,
            eligible_items=eligible,
            catalog=_catalog(),
            cross_store_totals=totals,
            cross_store_snapshot=dict(totals),
            excluded_item_names=excluded,
        )

        offered_names = {it.item.lower() for it in offer.offered_items}
        assert "chicken" not in offered_names, "Vegetarian home got chicken"
        assert "eggs" not in offered_names, "Vegetarian home got eggs"
        assert "milk" in offered_names, "Vegetarian home should get milk"
        assert "rice" in offered_names, "Vegetarian home should get rice"

    @pytest.mark.asyncio
    async def test_auto_accept_is_essential_matches_catalog(self):
        from agents.matchmaker_agent import run_matchmaker

        home = _make_care_home(negotiates_via_a2a=False)
        eligible = _eligible_items()
        totals = _cross_store_totals()

        offer = await run_matchmaker(
            care_home=home,
            eligible_items=eligible,
            catalog=_catalog(),
            cross_store_totals=totals,
            cross_store_snapshot=dict(totals),
            excluded_item_names=[],
        )

        catalog_map = {c.name.lower(): c.is_essential for c in _catalog()}
        for item in offer.offered_items:
            expected = catalog_map.get(item.item.lower())
            if expected is not None:
                assert item.is_essential == expected, (
                    f"{item.item}: is_essential={item.is_essential} "
                    f"but catalog says {expected}"
                )

    @pytest.mark.asyncio
    async def test_auto_accept_urgency_request_empty(self):
        from agents.matchmaker_agent import run_matchmaker

        home = _make_care_home(negotiates_via_a2a=False)
        offer = await run_matchmaker(
            care_home=home,
            eligible_items=_eligible_items(),
            catalog=_catalog(),
            cross_store_totals=_cross_store_totals(),
            cross_store_snapshot=dict(_cross_store_totals()),
            excluded_item_names=[],
        )
        assert offer.urgency_request == ""


# ===========================================================================
# 3. LLM integration tests  (Gemini via Vertex AI)
# ===========================================================================

_skip_no_llm = pytest.mark.skipif(
    not _llm_available(),
    reason="No LLM auth available (Vertex AI ADC or API key required)",
)


@_skip_no_llm
class TestMatchmakerLLM:
    """
    Live LLM tests using a small fixed dataset (4 items).
    Kept minimal to reduce API cost and run time.
    """

    @pytest.mark.asyncio
    async def test_offer_contains_dish_framing_placeholder(self):
        from agents.matchmaker_agent import run_matchmaker

        home = _make_care_home(
            care_home_id="home_01",
            name="Test LLM Home",
            negotiates_via_a2a=True,
            resident_count=30,
            storage_capacity_kg=200.0,
        )
        eligible = [
            DailyFoodItem(name="milk", days_to_expiry=1, quantity=50.0, unit="units"),
            DailyFoodItem(name="rice", days_to_expiry=5, quantity=30.0, unit="kg"),
        ]
        totals = {"milk": 50.0, "rice": 30.0}

        offer = await run_matchmaker(
            care_home=home,
            eligible_items=eligible,
            catalog=_catalog(),
            cross_store_totals=totals,
            cross_store_snapshot=dict(totals),
            excluded_item_names=[],
        )

        assert isinstance(offer, MatchmakerOffer)
        assert "[DISH_FRAMING]" in offer.offer_message, (
            f"[DISH_FRAMING] placeholder missing from offer_message:\n"
            f"{offer.offer_message}"
        )

    @pytest.mark.asyncio
    async def test_urgency_request_is_standard_text(self):
        from agents.matchmaker_agent import URGENCY_REQUEST_TEXT, run_matchmaker

        home = _make_care_home(negotiates_via_a2a=True)
        eligible = [
            DailyFoodItem(name="milk", days_to_expiry=1, quantity=50.0, unit="units"),
            DailyFoodItem(name="rice", days_to_expiry=5, quantity=30.0, unit="kg"),
        ]
        totals = {"milk": 50.0, "rice": 30.0}

        offer = await run_matchmaker(
            care_home=home,
            eligible_items=eligible,
            catalog=_catalog(),
            cross_store_totals=totals,
            cross_store_snapshot=dict(totals),
            excluded_item_names=[],
        )

        assert offer.urgency_request == URGENCY_REQUEST_TEXT
        # Standard text only mentions "essential items", not specific names
        assert "essential" in offer.urgency_request.lower()
        # Should NOT mention non-essential items like sugar by name
        assert "sugar" not in offer.urgency_request.lower()

    @pytest.mark.asyncio
    async def test_young_children_home_gets_milk_eggs(self):
        """
        Care home with has_young_children=True should receive milk
        and/or eggs in its offer when they are available.
        """
        from agents.matchmaker_agent import run_matchmaker

        home = _make_care_home(
            has_young_children=True,
            negotiates_via_a2a=True,
            resident_count=25,
            storage_capacity_kg=150.0,
        )
        eligible = [
            DailyFoodItem(name="milk",  days_to_expiry=1, quantity=50.0, unit="units"),
            DailyFoodItem(name="eggs",  days_to_expiry=2, quantity=40.0, unit="units"),
            DailyFoodItem(name="rice",  days_to_expiry=5, quantity=30.0, unit="kg"),
            DailyFoodItem(name="sugar", days_to_expiry=6, quantity=10.0, unit="kg"),
        ]
        totals = {"milk": 50.0, "eggs": 40.0, "rice": 30.0, "sugar": 10.0}

        offer = await run_matchmaker(
            care_home=home,
            eligible_items=eligible,
            catalog=_catalog(),
            cross_store_totals=totals,
            cross_store_snapshot=dict(totals),
            excluded_item_names=[],
        )

        offered_names = {it.item.lower() for it in offer.offered_items}
        # Children priority: milk and/or eggs MUST be present
        has_child_nutrition = "milk" in offered_names or "eggs" in offered_names
        assert has_child_nutrition, (
            f"Care home with young children should get milk and/or eggs, "
            f"but offer only contains: {offered_names}"
        )

    @pytest.mark.asyncio
    async def test_offer_quantities_positive_and_within_bounds(self):
        """Verify LLM output passes guardrail constraints."""
        from agents.matchmaker_agent import run_matchmaker

        home = _make_care_home(negotiates_via_a2a=True)
        totals = {"milk": 50.0, "rice": 30.0}
        eligible = [
            DailyFoodItem(name="milk", days_to_expiry=1, quantity=50.0, unit="units"),
            DailyFoodItem(name="rice", days_to_expiry=5, quantity=30.0, unit="kg"),
        ]

        offer = await run_matchmaker(
            care_home=home,
            eligible_items=eligible,
            catalog=_catalog(),
            cross_store_totals=totals,
            cross_store_snapshot=dict(totals),
            excluded_item_names=[],
        )

        for item in offer.offered_items:
            assert item.offered_quantity > 0, (
                f"{item.item} has non-positive quantity: {item.offered_quantity}"
            )
            available = totals.get(item.item.lower(), float("inf"))
            assert item.offered_quantity <= available + 0.01, (
                f"{item.item} offered {item.offered_quantity} but only "
                f"{available} available"
            )
