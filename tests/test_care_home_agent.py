"""
tests/test_care_home_agent.py

Test suite for the Care Home A2A Negotiation layer.
Covers model validation, mocked negotiation logic, and LLM integration.
"""

from __future__ import annotations

import os
import sys
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.data_model import (
    CareHome,
    DailyFoodItem,
    FoodCatalogItem,
    HardConstraints,
    SimulationDay,
    StoreDailyState,
    VolunteerDailyState,
)
from tools.constraint_tools import StockLedger
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


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_care_home(
    care_home_id: str = "home_01",
    name: str = "Test Care Home",
    negotiates_via_a2a: bool = True,
    vegetarian_only: bool = False,
    has_young_children: bool = False,
    resident_count: int = 30,
    storage_capacity_kg: float = 200.0,
    memory_notes=None,
) -> CareHome:
    return CareHome(
        care_home_id=care_home_id,
        name=name,
        latitude=13.0,
        longitude=80.2,
        hard_constraints=HardConstraints(
            vegetarian_only=vegetarian_only,
            has_young_children=has_young_children,
        ),
        resident_count=resident_count,
        storage_capacity_kg=storage_capacity_kg,
        negotiates_via_a2a=negotiates_via_a2a,
        memory_notes=memory_notes or [],
    )


def _make_offer(
    care_home_id: str = "home_01",
    items: List[OfferedItem] | None = None,
) -> MatchmakerOffer:
    if items is None:
        items = [
            OfferedItem(item="milk", unit="units", offered_quantity=50.0, is_essential=True),
            OfferedItem(item="rice", unit="kg", offered_quantity=30.0, is_essential=True),
            OfferedItem(item="sugar", unit="kg", offered_quantity=10.0, is_essential=False),
        ]
    return MatchmakerOffer(
        care_home_id=care_home_id,
        offered_items=items,
        rationale="Test offer",
        offer_message="We have prepared an offer for you with milk, rice, and sugar.",
        expected_today_statement="Expect delivery today.",
        urgency_request="Please flag any urgent items.",
    )


def _catalog() -> List[FoodCatalogItem]:
    return [
        FoodCatalogItem(name="milk", is_perishable=True, is_essential=True, push_threshold_days=1, unit="units", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="rice", is_perishable=False, is_essential=True, push_threshold_days=7, unit="kg", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="sugar", is_perishable=False, is_essential=False, push_threshold_days=14, unit="kg", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="lentils", is_perishable=False, is_essential=True, push_threshold_days=7, unit="kg", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="chicken", is_perishable=True, is_essential=False, push_threshold_days=1, unit="kg", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="eggs", is_perishable=True, is_essential=True, push_threshold_days=2, unit="units", approx_weight_kg=1.0, cap_category="test"),
    ]


def _make_sim_day() -> SimulationDay:
    """Create a minimal SimulationDay with stock for testing."""
    return SimulationDay(
        run_id="test-run",
        stores=[
            StoreDailyState(
                store_id="store_01",
                full_inventory=[],
                pushed_inventory=[
                    DailyFoodItem(name="milk", days_to_expiry=1, quantity=100.0, unit="units"),
                    DailyFoodItem(name="rice", days_to_expiry=5, quantity=80.0, unit="kg"),
                    DailyFoodItem(name="sugar", days_to_expiry=10, quantity=30.0, unit="kg"),
                    DailyFoodItem(name="lentils", days_to_expiry=5, quantity=20.0, unit="kg"),
                ],
            ),
        ],
        volunteers=[
            VolunteerDailyState(volunteer_id="vol_01", available=True),
        ],
    )


async def _mock_run_negotiation_with_response(
    care_home_response: CareHomeResponse,
    care_home: CareHome | None = None,
    offer: MatchmakerOffer | None = None,
    round_2_response: CareHomeResponse | None = None,
) -> NegotiationResult:
    """
    Run negotiation with a mocked care home agent that returns
    the given response(s).
    """
    from agents.care_home_agent import run_negotiation

    ch = care_home or _make_care_home()
    off = offer or _make_offer(ch.care_home_id)
    sim_day = _make_sim_day()
    ledger = StockLedger(sim_day)

    responses = [care_home_response]
    if round_2_response:
        responses.append(round_2_response)
    call_count = 0

    async def mock_turn(runner, session_id, message_text):
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        return responses[idx]

    with patch("agents.care_home_agent._run_care_home_turn", side_effect=mock_turn):
        return await run_negotiation(
            care_home=ch,
            offer=off,
            ledger=ledger,
            catalog=_catalog(),
            run_id="test-run",
        )


# ===========================================================================
# Model validation tests
# ===========================================================================

class TestNegotiationResultModel:
    """Validate NegotiationResult model constraints."""

    def test_rejected_status_with_empty_items(self):
        result = NegotiationResult(
            care_home_id="home_01",
            status="rejected",
            agreed_items=[],
            urgent_item_names=[],
            rejection_message=_REJECTION_MESSAGE,
        )
        assert result.status == "rejected"
        assert result.agreed_items == []
        assert result.rejection_message == _REJECTION_MESSAGE

    def test_agreed_status_with_items(self):
        result = NegotiationResult(
            care_home_id="home_01",
            status="agreed",
            agreed_items=[
                OrderLineItem(item="milk", unit="units", offered_quantity=50, accepted_quantity=50),
            ],
            urgent_item_names=["milk"],
        )
        assert result.status == "agreed"
        assert len(result.agreed_items) == 1
        assert result.rejection_message is None

    def test_rejection_message_none_when_agreed(self):
        result = NegotiationResult(
            care_home_id="home_01",
            status="agreed",
            agreed_items=[],
            urgent_item_names=[],
        )
        assert result.rejection_message is None

    def test_urgent_item_names_is_list_not_set(self):
        """Verify List[str] is used for JSON serialization safety."""
        result = NegotiationResult(
            care_home_id="home_01",
            status="agreed",
            urgent_item_names=["milk", "rice"],
        )
        assert isinstance(result.urgent_item_names, list)
        # Should be JSON serializable
        import json
        dumped = json.loads(result.model_dump_json())
        assert isinstance(dumped["urgent_item_names"], list)


class TestCareHomeActionModel:
    """Validate CareHomeAction model."""

    def test_valid_accept_all(self):
        a = CareHomeAction(action="accept_all")
        assert a.action == "accept_all"

    def test_valid_reduce_item(self):
        a = CareHomeAction(action="reduce_item", item="rice", requested_quantity=10.0)
        assert a.item == "rice"
        assert a.requested_quantity == 10.0

    def test_invalid_action_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CareHomeAction(action="invalid_action")

    def test_valid_reject_all(self):
        a = CareHomeAction(action="reject_all")
        assert a.action == "reject_all"


# ===========================================================================
# Deterministic negotiation tests (mocked LLM)
# ===========================================================================

class TestRunNegotiation:
    """Test run_negotiation() with mocked care home agent responses."""

    @pytest.mark.asyncio
    async def test_reject_all_returns_rejected(self):
        response = CareHomeResponse(
            actions=[CareHomeAction(action="reject_all")],
            rationale="Offer not suitable today.",
        )
        result = await _mock_run_negotiation_with_response(response)

        assert result.status == "rejected"
        assert result.agreed_items == []
        assert result.urgent_item_names == []
        assert result.rejection_message == _REJECTION_MESSAGE

    @pytest.mark.asyncio
    async def test_accept_all_returns_all_items(self):
        response = CareHomeResponse(
            actions=[CareHomeAction(action="accept_all")],
            rationale="Everything looks good!",
        )
        result = await _mock_run_negotiation_with_response(response)

        assert result.status == "agreed"
        assert len(result.agreed_items) == 3
        item_names = {i.item.lower() for i in result.agreed_items}
        assert item_names == {"milk", "rice", "sugar"}
        # All accepted at full offered quantity
        for item in result.agreed_items:
            assert item.accepted_quantity == item.offered_quantity

    @pytest.mark.asyncio
    async def test_flag_urgent_essential_item(self):
        """flag_urgent on an essential item → appears in urgent_item_names."""
        response = CareHomeResponse(
            actions=[
                CareHomeAction(action="accept_all"),
                CareHomeAction(action="flag_urgent", item="milk"),
            ],
            rationale="Accept all, milk is urgent.",
        )
        result = await _mock_run_negotiation_with_response(response)

        assert result.status == "agreed"
        assert "milk" in result.urgent_item_names

    @pytest.mark.asyncio
    async def test_flag_urgent_non_essential_ignored(self):
        """flag_urgent on a non-essential item → silently ignored."""
        response = CareHomeResponse(
            actions=[
                CareHomeAction(action="accept_all"),
                CareHomeAction(action="flag_urgent", item="sugar"),
            ],
            rationale="Accept all, flag sugar.",
        )
        result = await _mock_run_negotiation_with_response(response)

        assert result.status == "agreed"
        assert "sugar" not in result.urgent_item_names

    @pytest.mark.asyncio
    async def test_reduce_item_sets_lower_quantity(self):
        """reduce_item correctly sets accepted_quantity < offered_quantity."""
        response = CareHomeResponse(
            actions=[
                CareHomeAction(action="accept_all"),
                CareHomeAction(action="reduce_item", item="rice", requested_quantity=10.0),
            ],
            rationale="We only need 10 kg of rice.",
        )
        result = await _mock_run_negotiation_with_response(response)

        assert result.status == "agreed"
        rice = next(i for i in result.agreed_items if i.item.lower() == "rice")
        assert rice.accepted_quantity == 10.0
        assert rice.offered_quantity == 30.0

    @pytest.mark.asyncio
    async def test_exclude_item_removes_from_agreed(self):
        """exclude_item removes item from agreed_items entirely."""
        response = CareHomeResponse(
            actions=[
                CareHomeAction(action="accept_all"),
                CareHomeAction(action="exclude_item", item="sugar"),
            ],
            rationale="We don't need sugar today.",
        )
        result = await _mock_run_negotiation_with_response(response)

        assert result.status == "agreed"
        item_names = {i.item.lower() for i in result.agreed_items}
        assert "sugar" not in item_names
        assert "milk" in item_names
        assert "rice" in item_names

    @pytest.mark.asyncio
    async def test_request_item_available_adds_to_agreed(self):
        """request_item for available ledger item → added to agreed_items."""
        response = CareHomeResponse(
            actions=[
                CareHomeAction(action="accept_all"),
                CareHomeAction(action="request_item", item="lentils"),
            ],
            rationale="Can we also get lentils?",
        )
        # Round 2: care home accepts the re-offer
        r2_response = CareHomeResponse(
            actions=[CareHomeAction(action="accept_all")],
            rationale="Great, thanks!",
        )
        result = await _mock_run_negotiation_with_response(
            response, round_2_response=r2_response,
        )

        assert result.status == "agreed"
        item_names = {i.item.lower() for i in result.agreed_items}
        assert "lentils" in item_names

    @pytest.mark.asyncio
    async def test_request_item_unavailable_not_added(self):
        """request_item for unavailable item → not added to agreed_items."""
        response = CareHomeResponse(
            actions=[
                CareHomeAction(action="accept_all"),
                CareHomeAction(action="request_item", item="chicken"),
            ],
            rationale="Can we also get chicken?",
        )
        result = await _mock_run_negotiation_with_response(response)

        assert result.status == "agreed"
        item_names = {i.item.lower() for i in result.agreed_items}
        # chicken is not in our test ledger pushed_inventory
        assert "chicken" not in item_names

    @pytest.mark.asyncio
    async def test_reoffer_only_when_request_item_fulfilled(self):
        """Re-offer NOT triggered for accept_all/reduce/exclude/flag_urgent."""
        response = CareHomeResponse(
            actions=[
                CareHomeAction(action="accept_all"),
                CareHomeAction(action="flag_urgent", item="milk"),
                CareHomeAction(action="reduce_item", item="rice", requested_quantity=15.0),
                CareHomeAction(action="exclude_item", item="sugar"),
            ],
            rationale="Accept with modifications.",
        )
        result = await _mock_run_negotiation_with_response(response)

        assert result.status == "agreed"
        # Only 1 system offer in transcript (no re-offer)
        system_offers = [
            t for t in result.negotiation_transcript
            if t.speaker == "system" and t.action == "offer"
        ]
        assert len(system_offers) == 1

    @pytest.mark.asyncio
    async def test_second_request_in_round2_not_fulfilled(self):
        """request_item in round 2 → logged but not fulfilled."""
        # Round 1: request lentils (available → triggers re-offer)
        r1_response = CareHomeResponse(
            actions=[
                CareHomeAction(action="accept_all"),
                CareHomeAction(action="request_item", item="lentils"),
            ],
            rationale="Can we also get lentils?",
        )
        # Round 2: request chicken (should NOT be fulfilled)
        r2_response = CareHomeResponse(
            actions=[
                CareHomeAction(action="accept_all"),
                CareHomeAction(action="request_item", item="eggs"),
            ],
            rationale="Accept lentils, can we also get eggs?",
        )
        result = await _mock_run_negotiation_with_response(
            r1_response, round_2_response=r2_response,
        )

        assert result.status == "agreed"
        item_names = {i.item.lower() for i in result.agreed_items}
        assert "lentils" in item_names
        # eggs should NOT be added (round 2 request not fulfilled)
        assert "eggs" not in item_names

    @pytest.mark.asyncio
    async def test_transcript_correct_sequence(self):
        """Negotiation transcript contains correct speaker/action sequence."""
        response = CareHomeResponse(
            actions=[
                CareHomeAction(action="accept_all"),
                CareHomeAction(action="flag_urgent", item="milk"),
            ],
            rationale="Accept all, milk is urgent.",
        )
        result = await _mock_run_negotiation_with_response(response)

        assert len(result.negotiation_transcript) >= 3
        # First turn: system offer
        assert result.negotiation_transcript[0].speaker == "system"
        assert result.negotiation_transcript[0].action == "offer"
        # Second turn: care home accept_all
        assert result.negotiation_transcript[1].speaker == "care_home"
        assert result.negotiation_transcript[1].action == "accept_all"
        # Third turn: care home flag_urgent
        assert result.negotiation_transcript[2].speaker == "care_home"
        assert result.negotiation_transcript[2].action == "flag_urgent"
        assert result.negotiation_transcript[2].item == "milk"

    @pytest.mark.asyncio
    async def test_combined_actions(self):
        """Multiple actions in a single response are all processed."""
        response = CareHomeResponse(
            actions=[
                CareHomeAction(action="reduce_item", item="rice", requested_quantity=10.0),
                CareHomeAction(action="flag_urgent", item="rice"),
                CareHomeAction(action="exclude_item", item="sugar"),
            ],
            rationale="Reduce rice, flag it urgent, exclude sugar.",
        )
        result = await _mock_run_negotiation_with_response(response)

        assert result.status == "agreed"
        rice = next(i for i in result.agreed_items if i.item.lower() == "rice")
        assert rice.accepted_quantity == 10.0
        assert "rice" in result.urgent_item_names
        item_names = {i.item.lower() for i in result.agreed_items}
        assert "sugar" not in item_names


# ===========================================================================
# LLM integration tests
# ===========================================================================

def _llm_available() -> bool:
    try:
        import google.auth
        _, _ = google.auth.default()
        return True
    except Exception:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return bool(key and not key.startswith("your_"))


_skip_no_llm = pytest.mark.skipif(
    not _llm_available(),
    reason="No LLM auth available (Vertex AI ADC or API key required)",
)


@_skip_no_llm
class TestCareHomeLLM:
    """Live LLM tests for the care home agent."""

    @pytest.mark.asyncio
    async def test_vegetarian_home_never_accepts_chicken(self):
        """A vegetarian care home agent should not accept chicken/eggs."""
        from agents.care_home_agent import run_negotiation

        home = _make_care_home(
            vegetarian_only=True,
            care_home_id="home_veg",
            name="Vegetarian Test Home",
        )
        offer = _make_offer(
            care_home_id="home_veg",
            items=[
                OfferedItem(item="milk", unit="units", offered_quantity=30.0, is_essential=True),
                OfferedItem(item="chicken", unit="kg", offered_quantity=10.0, is_essential=False),
                OfferedItem(item="rice", unit="kg", offered_quantity=20.0, is_essential=True),
            ],
        )
        sim_day = _make_sim_day()
        ledger = StockLedger(sim_day)

        result = await run_negotiation(
            care_home=home,
            offer=offer,
            ledger=ledger,
            catalog=_catalog(),
            run_id="test-veg-run",
        )

        if result.status == "agreed":
            item_names = {i.item.lower() for i in result.agreed_items}
            assert "chicken" not in item_names, (
                f"Vegetarian home accepted chicken! Items: {item_names}"
            )

    @pytest.mark.asyncio
    async def test_full_negotiation_produces_valid_result(self):
        """Full round-trip produces a valid NegotiationResult."""
        from agents.care_home_agent import run_negotiation

        home = _make_care_home()
        offer = _make_offer()
        sim_day = _make_sim_day()
        ledger = StockLedger(sim_day)

        result = await run_negotiation(
            care_home=home,
            offer=offer,
            ledger=ledger,
            catalog=_catalog(),
            run_id="test-full-run",
        )

        assert isinstance(result, NegotiationResult)
        assert result.care_home_id == "home_01"
        assert result.status in ("agreed", "rejected")
        assert isinstance(result.negotiation_transcript, list)
        assert len(result.negotiation_transcript) >= 2  # at least offer + response

        if result.status == "agreed":
            assert len(result.agreed_items) > 0
            for item in result.agreed_items:
                assert item.accepted_quantity >= 0
                assert item.accepted_quantity <= item.offered_quantity
