"""
tests/test_tools.py

pytest test suite for the tools/ layer.
Tests are deterministic — no LLM calls, no network calls.
Run with: .venv/Scripts/pytest tests/test_tools.py -v
"""

from __future__ import annotations

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
    SimulationDay,
    StoreDailyState,
    VolunteerDailyState,
)
from tools.constraint_tools import StockLedger, hard_constraint_filter, single_store_candidate
from tools.guardrails import COMMERCIAL_CAPACITY_SENTINEL, DispatchOutput, OrderOutput
from tools.models import Delivery, NegotiationTurn, Order, OrderLineItem
import tools.logger as logger


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _catalog() -> list[FoodCatalogItem]:
    """Minimal catalog for tests."""
    return [
        FoodCatalogItem(name="milk",    is_perishable=True,  is_essential=True,  push_threshold_days=1, unit="units", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="chicken", is_perishable=True,  is_essential=True,  push_threshold_days=1, unit="kg", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="eggs",    is_perishable=True,  is_essential=True,  push_threshold_days=1, unit="units", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="rice",    is_perishable=False, is_essential=True,  push_threshold_days=7, unit="kg", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="sugar",   is_perishable=False, is_essential=False, push_threshold_days=7, unit="kg", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="curd",    is_perishable=True,  is_essential=False, push_threshold_days=1, unit="units", approx_weight_kg=1.0, cap_category="test"),
    ]


def _make_care_home(
    vegetarian: bool = False,
    has_young_children: bool = False,
    memory_notes: list[MemoryNote] | None = None,
) -> CareHome:
    return CareHome(
        care_home_id="home_01",
        name="Test Home",
        latitude=13.0,
        longitude=80.2,
        hard_constraints=HardConstraints(
            vegetarian_only=vegetarian,
            has_young_children=has_young_children,
        ),
        resident_count=30,
        storage_capacity_kg=500.0,
        negotiates_via_a2a=True,
        memory_notes=memory_notes or [],
    )


def _offered_items() -> list[DailyFoodItem]:
    return [
        DailyFoodItem(name="milk",    days_to_expiry=1, quantity=50.0,  unit="units"),
        DailyFoodItem(name="chicken", days_to_expiry=1, quantity=20.0,  unit="kg"),
        DailyFoodItem(name="eggs",    days_to_expiry=1, quantity=100.0, unit="units"),
        DailyFoodItem(name="rice",    days_to_expiry=5, quantity=30.0,  unit="kg"),
        DailyFoodItem(name="sugar",   days_to_expiry=6, quantity=10.0,  unit="kg"),
        DailyFoodItem(name="curd",    days_to_expiry=1, quantity=40.0,  unit="units"),
    ]


def _make_sim_day(store_inventories: dict[str, list[DailyFoodItem]]) -> SimulationDay:
    stores = [
        StoreDailyState(
            store_id=sid,
            full_inventory=[],          # audit-only — never used by tools
            pushed_inventory=items,
        )
        for sid, items in store_inventories.items()
    ]
    return SimulationDay(run_id=str(uuid.uuid4()), stores=stores, volunteers=[])


def _make_order(
    store_id: str = "store_01",
    accepted_qty: float = 10.0,
    item_name: str = "milk",
) -> Order:
    return Order(
        order_id=str(uuid.uuid4()),
        care_home_id="home_01",
        store_id=store_id,
        items=[OrderLineItem(
            item=item_name,
            unit="units",
            offered_quantity=accepted_qty,
            accepted_quantity=accepted_qty,
        )],
    )


def _make_delivery(order_ids: list[str] | None = None, method: str = "volunteer") -> Delivery:
    return Delivery(
        delivery_id=str(uuid.uuid4()),
        store_id="store_01",
        order_ids=order_ids or ["order_01"],
        method=method,
        volunteer_id="vol_01",
    )


# ===========================================================================
# 1. hard_constraint_filter
# ===========================================================================

class TestHardConstraintFilter:

    def test_vegetarian_home_removes_chicken_and_eggs(self):
        home = _make_care_home(vegetarian=True)
        result = hard_constraint_filter(home, _offered_items(), _catalog())
        names = {item.name for item in result}
        assert "chicken" not in names
        assert "eggs"    not in names

    def test_vegetarian_home_keeps_other_items(self):
        home = _make_care_home(vegetarian=True)
        result = hard_constraint_filter(home, _offered_items(), _catalog())
        names = {item.name for item in result}
        assert "milk"  in names
        assert "rice"  in names
        assert "sugar" in names
        assert "curd"  in names

    def test_non_vegetarian_home_receives_all_items(self):
        home = _make_care_home(vegetarian=False)
        result = hard_constraint_filter(home, _offered_items(), _catalog())
        names = {item.name for item in result}
        assert "chicken" in names
        assert "eggs"    in names
        assert "milk"    in names

    def test_memory_note_exclude_removes_item(self):
        home = _make_care_home(memory_notes=[MemoryNote(item="sugar", type="exclude")])
        result = hard_constraint_filter(home, _offered_items(), _catalog())
        assert "sugar" not in {item.name for item in result}

    def test_memory_note_exclude_case_insensitive(self):
        home = _make_care_home(memory_notes=[MemoryNote(item="Curd", type="exclude")])
        result = hard_constraint_filter(home, _offered_items(), _catalog())
        assert "curd" not in {item.name for item in result}

    def test_memory_note_max_quantity_caps_not_removes(self):
        home = _make_care_home(
            memory_notes=[MemoryNote(item="milk", type="max_quantity", value=15.0)]
        )
        result = hard_constraint_filter(home, _offered_items(), _catalog())
        milk = next(it for it in result if it.name == "milk")
        assert milk.quantity == 15.0
        assert milk in result

    def test_memory_note_max_quantity_does_not_cap_if_already_under(self):
        home = _make_care_home(
            memory_notes=[MemoryNote(item="milk", type="max_quantity", value=100.0)]
        )
        result = hard_constraint_filter(home, _offered_items(), _catalog())
        milk = next(it for it in result if it.name == "milk")
        assert milk.quantity == 50.0

    def test_vegetarian_and_exclude_both_applied(self):
        home = _make_care_home(
            vegetarian=True,
            memory_notes=[MemoryNote(item="sugar", type="exclude")],
        )
        result = hard_constraint_filter(home, _offered_items(), _catalog())
        names = {item.name for item in result}
        assert "chicken" not in names
        assert "eggs"    not in names
        assert "sugar"   not in names


# ===========================================================================
# 2. StockLedger
# ===========================================================================

class TestStockLedger:

    def _two_store_day(self) -> SimulationDay:
        return _make_sim_day({
            "store_01": [
                DailyFoodItem(name="milk",  days_to_expiry=1, quantity=100.0, unit="units"),
                DailyFoodItem(name="rice",  days_to_expiry=5, quantity=50.0,  unit="kg"),
            ],
            "store_02": [
                DailyFoodItem(name="milk",  days_to_expiry=1, quantity=60.0,  unit="units"),
                DailyFoodItem(name="sugar", days_to_expiry=6, quantity=30.0,  unit="kg"),
            ],
        })

    def test_initial_stock_correct(self):
        ledger = StockLedger(self._two_store_day())
        assert ledger.get_available("store_01", "milk")  == 100.0
        assert ledger.get_available("store_01", "rice")  == 50.0
        assert ledger.get_available("store_02", "milk")  == 60.0
        assert ledger.get_available("store_02", "sugar") == 30.0

    def test_deduction_reduces_remaining(self):
        ledger = StockLedger(self._two_store_day())
        ledger.deduct("store_01", "milk", 40.0)
        assert ledger.get_available("store_01", "milk") == 60.0

    def test_sequential_deductions_home1_then_home2(self):
        ledger = StockLedger(self._two_store_day())
        ledger.deduct("store_01", "milk", 80.0)
        assert ledger.get_available("store_01", "milk") == 20.0

    def test_deduction_beyond_remaining_raises(self):
        ledger = StockLedger(self._two_store_day())
        with pytest.raises(ValueError, match="only"):
            ledger.deduct("store_01", "milk", 200.0)

    def test_get_store_totals(self):
        ledger = StockLedger(self._two_store_day())
        ledger.deduct("store_01", "milk", 30.0)
        totals = ledger.get_store_totals("store_01")
        assert totals["milk"] == 70.0
        assert totals["rice"] == 50.0
        assert "sugar" not in totals

    def test_get_cross_store_totals(self):
        ledger = StockLedger(self._two_store_day())
        totals = ledger.get_cross_store_totals()
        assert totals["milk"]  == 160.0
        assert totals["rice"]  == 50.0
        assert totals["sugar"] == 30.0

    def test_get_cross_store_totals_after_deduction(self):
        ledger = StockLedger(self._two_store_day())
        ledger.deduct("store_01", "milk", 40.0)
        totals = ledger.get_cross_store_totals()
        assert totals["milk"] == 120.0

    def test_unknown_item_returns_zero(self):
        ledger = StockLedger(self._two_store_day())
        assert ledger.get_available("store_01", "nonexistent_item") == 0.0

    def test_store_ids_public_method(self):
        """Use store_ids() — no private attribute access."""
        ledger = StockLedger(self._two_store_day())
        ids = ledger.store_ids()
        assert set(ids) == {"store_01", "store_02"}
        assert isinstance(ids, list)

    def test_store_ids_sorted_deterministic(self):
        ledger = StockLedger(self._two_store_day())
        assert ledger.store_ids() == sorted(ledger.store_ids())


# ===========================================================================
# 3. single_store_candidate  (new signature with urgent_item_names)
# ===========================================================================

class TestSingleStoreCandidate:

    def _cat(self) -> list[FoodCatalogItem]:
        return _catalog()

    def _sim_day_two_stores(self) -> SimulationDay:
        return _make_sim_day({
            "store_01": [
                DailyFoodItem(name="milk",    days_to_expiry=1, quantity=100.0, unit="units"),
                DailyFoodItem(name="rice",    days_to_expiry=5, quantity=50.0,  unit="kg"),
                DailyFoodItem(name="sugar",   days_to_expiry=6, quantity=20.0,  unit="kg"),
            ],
            "store_02": [
                DailyFoodItem(name="chicken", days_to_expiry=1, quantity=30.0,  unit="kg"),
                DailyFoodItem(name="rice",    days_to_expiry=5, quantity=40.0,  unit="kg"),
            ],
        })

    def test_prefers_single_store_when_one_covers_all(self):
        ledger = StockLedger(self._sim_day_two_stores())
        requested = [
            OrderLineItem(item="milk",  unit="units", offered_quantity=50.0, accepted_quantity=50.0),
            OrderLineItem(item="rice",  unit="kg",    offered_quantity=30.0, accepted_quantity=30.0),
            OrderLineItem(item="sugar", unit="kg",    offered_quantity=10.0, accepted_quantity=10.0),
        ]
        result = single_store_candidate(requested, set(), ledger, self._cat())
        assert len(result["assignments"]) == 1
        assert result["assignments"][0]["store_id"] == "store_01"
        assert result["deferred"] == []
        assert result["needs_commercial"] == []

    def test_non_essential_piggybacked_or_deferred_when_no_single_store(self):
        """
        store_01 has milk + sugar.  store_02 has chicken.
        No single store covers milk + chicken + sugar (step 1 fails).
        milk + chicken are flagged urgent → split across store_01 and store_02.
        sugar is non-essential + non-urgent → step 6 tries to piggyback on a
        selected store.  store_01 has sugar, so sugar is piggybacked there —
        it must NOT appear in deferred or needs_commercial.
        """
        ledger = StockLedger(self._sim_day_two_stores())
        requested = [
            OrderLineItem(item="milk",    unit="units", offered_quantity=50.0, accepted_quantity=50.0),
            OrderLineItem(item="chicken", unit="kg",    offered_quantity=10.0, accepted_quantity=10.0),
            OrderLineItem(item="sugar",   unit="kg",    offered_quantity=5.0,  accepted_quantity=5.0),
        ]
        result = single_store_candidate(
            requested, {"milk", "chicken"}, ledger, self._cat()
        )
        # sugar should be piggybacked — not deferred
        deferred_names = {it.item for it in result["deferred"]}
        assert "sugar" not in deferred_names, "sugar should be piggybacked, not deferred"
        assert "sugar" not in {it.item for it in result["needs_commercial"]}
        # sugar must appear in some assignment
        assigned_items = {it.item for asgn in result["assignments"] for it in asgn["items"]}
        assert "sugar" in assigned_items

    def test_urgent_essential_never_deferred(self):
        """
        milk (store_01) + chicken (store_02) with both flagged urgent.
        No single store covers both → split into two assignments.
        Neither should appear in deferred or needs_commercial.
        """
        ledger = StockLedger(self._sim_day_two_stores())
        requested = [
            OrderLineItem(item="milk",    unit="units", offered_quantity=50.0, accepted_quantity=50.0),
            OrderLineItem(item="chicken", unit="kg",    offered_quantity=10.0, accepted_quantity=10.0),
        ]
        result = single_store_candidate(
            requested, {"milk", "chicken"}, ledger, self._cat()
        )
        assert result["deferred"] == []
        assert result["needs_commercial"] == []
        assigned_items = {it.item for asgn in result["assignments"] for it in asgn["items"]}
        assert "milk"    in assigned_items
        assert "chicken" in assigned_items

    def test_empty_urgent_item_names_auto_accept_path(self):
        """
        Simulates home_05 (negotiates_via_a2a=False): caller passes empty set.
        All requested items that can fit one store → single assignment.
        """
        ledger = StockLedger(self._sim_day_two_stores())
        requested = [
            OrderLineItem(item="milk",  unit="units", offered_quantity=40.0, accepted_quantity=40.0),
            OrderLineItem(item="rice",  unit="kg",    offered_quantity=20.0, accepted_quantity=20.0),
            OrderLineItem(item="sugar", unit="kg",    offered_quantity=5.0,  accepted_quantity=5.0),
        ]
        result = single_store_candidate(requested, set(), ledger, self._cat())
        # store_01 covers all three
        assert len(result["assignments"]) == 1
        assert result["assignments"][0]["store_id"] == "store_01"
        assert result["deferred"] == []
        assert result["needs_commercial"] == []

    def test_item_needing_4th_store_goes_to_needs_commercial(self):
        """
        4 essential items each in a different store.  3-store cap means
        the 4th item goes to needs_commercial.
        """
        sim_day = _make_sim_day({
            "store_01": [DailyFoodItem(name="milk",    days_to_expiry=1, quantity=50.0, unit="units")],
            "store_02": [DailyFoodItem(name="chicken", days_to_expiry=1, quantity=20.0, unit="kg")],
            "store_03": [DailyFoodItem(name="eggs",    days_to_expiry=1, quantity=80.0, unit="units")],
            "store_04": [DailyFoodItem(name="rice",    days_to_expiry=5, quantity=30.0, unit="kg")],
        })
        ledger = StockLedger(sim_day)
        requested = [
            OrderLineItem(item="milk",    unit="units", offered_quantity=30.0, accepted_quantity=30.0),
            OrderLineItem(item="chicken", unit="kg",    offered_quantity=10.0, accepted_quantity=10.0),
            OrderLineItem(item="eggs",    unit="units", offered_quantity=50.0, accepted_quantity=50.0),
            OrderLineItem(item="rice",    unit="kg",    offered_quantity=20.0, accepted_quantity=20.0),
        ]
        result = single_store_candidate(
            requested, {"milk", "chicken", "eggs", "rice"}, ledger, self._cat()
        )
        # At most 3 distinct stores in assignments
        assignment_stores = {asgn["store_id"] for asgn in result["assignments"]}
        assert len(assignment_stores) <= 3
        # The 4th item must be in needs_commercial, not deferred
        nc_items = {nc["item"].item for nc in result["needs_commercial"]}
        assert "rice" in nc_items or "eggs" in nc_items or "chicken" in nc_items or "milk" in nc_items
        assert result["deferred"] == []

    def test_non_essential_flagged_urgent_silently_ignored(self):
        """
        sugar is non-essential; flagging it as urgent should be silently
        ignored — it must not appear in assignments as an urgent item.
        """
        ledger = StockLedger(self._sim_day_two_stores())
        requested = [
            OrderLineItem(item="milk",  unit="units", offered_quantity=30.0, accepted_quantity=30.0),
            OrderLineItem(item="sugar", unit="kg",    offered_quantity=5.0,  accepted_quantity=5.0),
        ]
        # Flagging sugar (non-essential) as urgent — should be ignored
        result = single_store_candidate(
            requested, {"sugar"}, ledger, self._cat()
        )
        # store_01 covers both; single assignment expected
        assert len(result["assignments"]) == 1
        assert result["needs_commercial"] == []

    def test_urgent_split_across_2_stores_succeeds_in_assignments(self):
        """
        An urgent item needing 13 dozen eggs, where store A has 8 and store B has 5.
        This uses 2 stores (which is <= MAX_STORES of 3), so it should split perfectly
        into `assignments` and NOT go to `needs_commercial`.
        """
        sim_day = _make_sim_day({
            "store_01": [DailyFoodItem(name="eggs", days_to_expiry=1, quantity=8.0, unit="dozen")],
            "store_02": [DailyFoodItem(name="eggs", days_to_expiry=1, quantity=5.0, unit="dozen")],
            "store_03": [DailyFoodItem(name="rice", days_to_expiry=5, quantity=10.0, unit="kg")],
        })
        ledger = StockLedger(sim_day)
        requested = [
            OrderLineItem(item="eggs", unit="dozen", offered_quantity=13.0, accepted_quantity=13.0),
        ]
        result = single_store_candidate(requested, {"eggs"}, ledger, self._cat())
        
        # It should succeed in assignments across the 2 stores.
        assert len(result["assignments"]) == 2
        assigned_stores = {a["store_id"] for a in result["assignments"]}
        assert assigned_stores == {"store_01", "store_02"}
        assert result["needs_commercial"] == []
        assert result["deferred"] == []


# ===========================================================================
# 4. Guardrail models
# ===========================================================================

class TestDispatchOutput:

    def test_valid_dispatch_passes(self):
        d = DispatchOutput(
            delivery=_make_delivery(),
            total_payload_kg=50.0,
            estimated_time_minutes=45.0,
            vehicle_capacity_kg=100.0,
        )
        assert d.total_payload_kg == 50.0

    def test_payload_exceeds_capacity_raises(self):
        with pytest.raises(Exception, match="capacity"):
            DispatchOutput(
                delivery=_make_delivery(),
                total_payload_kg=150.0,
                estimated_time_minutes=30.0,
                vehicle_capacity_kg=100.0,
            )

    def test_time_exceeds_120_raises(self):
        with pytest.raises(Exception, match="2-hour"):
            DispatchOutput(
                delivery=_make_delivery(),
                total_payload_kg=10.0,
                estimated_time_minutes=121.0,
                vehicle_capacity_kg=500.0,
            )

    def test_exactly_120_minutes_passes(self):
        d = DispatchOutput(
            delivery=_make_delivery(),
            total_payload_kg=10.0,
            estimated_time_minutes=120.0,
            vehicle_capacity_kg=500.0,
        )
        assert d.estimated_time_minutes == 120.0

    def test_more_than_two_order_ids_raises_on_delivery(self):
        """Delivery itself now enforces max 2 via Field(max_length=2)."""
        with pytest.raises(Exception):
            _make_delivery(["o1", "o2", "o3"])

    def test_exactly_two_order_ids_passes(self):
        d = DispatchOutput(
            delivery=_make_delivery(["o1", "o2"]),
            total_payload_kg=10.0,
            estimated_time_minutes=30.0,
            vehicle_capacity_kg=500.0,
        )
        assert len(d.delivery.order_ids) == 2

    def test_vehicle_capacity_required_no_default(self):
        """vehicle_capacity_kg has no default — omitting it must raise."""
        with pytest.raises(Exception):
            DispatchOutput(
                delivery=_make_delivery(),
                total_payload_kg=10.0,
                estimated_time_minutes=30.0,
                # vehicle_capacity_kg intentionally omitted
            )

    def test_commercial_sentinel_always_passes_capacity(self):
        """Commercial pickup uses COMMERCIAL_CAPACITY_SENTINEL — any payload fits."""
        d = DispatchOutput(
            delivery=_make_delivery(method="commercial"),
            total_payload_kg=5000.0,
            estimated_time_minutes=60.0,
            vehicle_capacity_kg=COMMERCIAL_CAPACITY_SENTINEL,
        )
        assert d.vehicle_capacity_kg == COMMERCIAL_CAPACITY_SENTINEL


class TestOrderOutput:

    def test_negative_accepted_quantity_raises(self):
        order = Order(
            order_id="o1",
            care_home_id="home_01",
            store_id="store_01",
            items=[OrderLineItem(item="milk", unit="units",
                                 offered_quantity=10.0, accepted_quantity=-1.0)],
        )
        with pytest.raises(Exception, match="negative"):
            OrderOutput(order=order)

    def test_valid_order_passes(self):
        order = _make_order(accepted_qty=10.0)
        out = OrderOutput(order=order)
        assert out.order.order_id == order.order_id

    def test_accepted_quantity_exceeds_string_keyed_ledger_raises(self):
        order = _make_order(accepted_qty=200.0, item_name="milk")
        with pytest.raises(Exception, match="exceeds ledger"):
            OrderOutput(order=order, ledger_snapshot={"milk": 50.0})

    def test_accepted_quantity_within_string_keyed_ledger_passes(self):
        order = _make_order(accepted_qty=30.0, item_name="milk")
        out = OrderOutput(order=order, ledger_snapshot={"milk": 50.0})
        assert out.order is not None

    def test_accepted_quantity_exceeds_tuple_keyed_ledger_raises(self):
        """Primary form: tuple keys from StockLedger.snapshot()."""
        order = _make_order(store_id="store_01", accepted_qty=200.0, item_name="milk")
        with pytest.raises(Exception, match="exceeds ledger"):
            OrderOutput(
                order=order,
                ledger_snapshot={("store_01", "milk"): 50.0},
            )

    def test_accepted_quantity_within_tuple_keyed_ledger_passes(self):
        order = _make_order(store_id="store_01", accepted_qty=30.0, item_name="milk")
        out = OrderOutput(
            order=order,
            ledger_snapshot={("store_01", "milk"): 50.0},
        )
        assert out.order is not None

    def test_excluded_item_present_raises(self):
        order = _make_order(item_name="chicken")
        with pytest.raises(Exception, match="excluded"):
            OrderOutput(order=order, excluded_item_names=["chicken"])

    def test_non_excluded_item_passes(self):
        order = _make_order(item_name="milk")
        out = OrderOutput(order=order, excluded_item_names=["chicken"])
        assert out.order is not None


# ===========================================================================
# 5. NegotiationTurn / Delivery model correctness
# ===========================================================================

class TestModelConstraints:

    def test_reject_all_is_valid_action(self):
        turn = NegotiationTurn(
            turn_number=3,
            speaker="care_home",
            action="reject_all",
        )
        assert turn.action == "reject_all"

    def test_invalid_action_raises(self):
        with pytest.raises(Exception):
            NegotiationTurn(turn_number=1, speaker="system", action="unknown_action")

    def test_invalid_speaker_raises(self):
        with pytest.raises(Exception):
            NegotiationTurn(turn_number=1, speaker="robot", action="offer")

    def test_delivery_max_2_order_ids_enforced_by_field(self):
        """Delivery.order_ids uses Field(max_length=2) — 3 entries must fail."""
        with pytest.raises(Exception):
            Delivery(
                delivery_id="d1",
                store_id="store_01",
                order_ids=["o1", "o2", "o3"],
                method="volunteer",
            )

    def test_delivery_invalid_method_raises(self):
        with pytest.raises(Exception):
            Delivery(
                delivery_id="d1",
                store_id="store_01",
                order_ids=["o1"],
                method="bicycle",   # not in Literal
            )

    def test_all_valid_actions_accepted(self):
        valid_actions = [
            "offer", "accept_all", "reduce_item", "exclude_item",
            "request_item", "flag_urgent", "reject_all",
        ]
        for action in valid_actions:
            turn = NegotiationTurn(turn_number=1, speaker="system", action=action)
            assert turn.action == action

    def test_all_valid_methods_accepted(self):
        for method in ("volunteer", "store_truck", "commercial"):
            d = Delivery(
                delivery_id="d1",
                store_id="store_01",
                order_ids=["o1"],
                method=method,
            )
            assert d.method == method


# ===========================================================================
# 6. log_message
# ===========================================================================

class TestLogger:

    def setup_method(self):
        logger.clear_log()
        logger.set_run_id("test-run-001")

    def test_messages_accumulate(self):
        logger.log_message("vol_01", "whatsapp_simulated", "Hello volunteer")
        logger.log_message("home_01", "a2a_negotiation", "Offer received")
        assert len(logger.get_message_log()) == 2

    def test_clear_log_resets_to_empty(self):
        logger.log_message("vol_01", "whatsapp_simulated", "msg1")
        logger.clear_log()
        assert logger.get_message_log() == []

    def test_entry_has_required_fields(self):
        logger.log_message("store_01", "whatsapp_simulated", "Push confirmed")
        entry = logger.get_message_log()[0]
        assert entry["recipient"] == "store_01"
        assert entry["channel"]   == "whatsapp_simulated"
        assert entry["content"]   == "Push confirmed"
        assert "timestamp"        in entry
        assert entry["run_id"]    == "test-run-001"

    def test_correct_channel_values(self):
        logger.log_message("vol_02",  "whatsapp_simulated", "Pickup at 9am")
        logger.log_message("home_02", "a2a_negotiation",    "Accept offer")
        channels = {e["channel"] for e in logger.get_message_log()}
        assert "whatsapp_simulated" in channels
        assert "a2a_negotiation"    in channels

    def test_run_id_set_per_run(self):
        logger.set_run_id("run-abc")
        logger.log_message("home_01", "a2a_negotiation", "test")
        assert logger.get_message_log()[0]["run_id"] == "run-abc"

    def test_clear_then_new_run_id(self):
        logger.log_message("x", "whatsapp_simulated", "old run")
        logger.clear_log()
        logger.set_run_id("run-new")
        logger.log_message("y", "a2a_negotiation", "new run")
        log = logger.get_message_log()
        assert len(log) == 1
        assert log[0]["run_id"] == "run-new"

    def test_get_message_log_returns_copy(self):
        logger.log_message("x", "whatsapp_simulated", "msg")
        log1 = logger.get_message_log()
        log1.clear()
        assert len(logger.get_message_log()) == 1
