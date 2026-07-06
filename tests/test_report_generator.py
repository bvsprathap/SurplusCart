"""
tests/test_report_generator.py

Test suite for reports/report_generator.py.
Uses fixed dummy data throughout — no real API calls or LLM calls.

Coverage:
  - generate_delivery_table: non-empty string, correct names, urgent ★ marker
  - generate_map: HTML saved to correct path, contains store/care home labels
  - generate_negotiation_report: all 5 homes, auto-accept note, rejection message
  - generate_audit_report: pushed vs held-back counts
  - generate_full_report: dict with all four keys populated
  - reports/output/ directory auto-created if missing
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.data_model import (
    CareHome,
    DailyFoodItem,
    FoodCatalogItem,
    HardConstraints,
    SimulationDay,
    Store,
    StoreDailyState,
    Volunteer,
    VolunteerDailyState,
    WorldConfig,
)
from tools.models import (
    Delivery,
    DispatchStats,
    NegotiationResult,
    NegotiationTurn,
    Order,
    OrderLineItem,
)

# ---------------------------------------------------------------------------
# Dummy world builder
# ---------------------------------------------------------------------------

def _make_world() -> WorldConfig:
    catalog = [
        FoodCatalogItem(name="milk", is_perishable=True, is_essential=True,
                        push_threshold_days=1, unit="units", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="rice", is_perishable=False, is_essential=True,
                        push_threshold_days=7, unit="kg", approx_weight_kg=1.0, cap_category="test"),
        FoodCatalogItem(name="lentils", is_perishable=False, is_essential=False,
                        push_threshold_days=14, unit="kg", approx_weight_kg=1.0, cap_category="test"),
    ]
    stores = [
        Store(store_id="store_01", name="Sri Balaji Supermarket",
              latitude=13.04, longitude=80.23,
              has_own_truck=True, truck_capacity_kg=3000.0),
        Store(store_id="store_02", name="Chennai Organic Plaza",
              latitude=13.02, longitude=80.20,
              has_own_truck=True, truck_capacity_kg=1500.0),
    ]
    care_homes = [
        CareHome(care_home_id="home_01", name="Anbu Illam Home",
                 latitude=13.06, longitude=80.27,
                 hard_constraints=HardConstraints(vegetarian_only=True, has_young_children=True),
                 resident_count=40, storage_capacity_kg=250.0,
                 negotiates_via_a2a=True),
        CareHome(care_home_id="home_02", name="Karuna Trust Home",
                 latitude=13.08, longitude=80.25,
                 hard_constraints=HardConstraints(vegetarian_only=False, has_young_children=False),
                 resident_count=35, storage_capacity_kg=225.0,
                 negotiates_via_a2a=True),
        CareHome(care_home_id="home_03", name="Nethaji Children Home",
                 latitude=13.05, longitude=80.22,
                 hard_constraints=HardConstraints(vegetarian_only=False, has_young_children=True),
                 resident_count=25, storage_capacity_kg=150.0,
                 negotiates_via_a2a=True),
        CareHome(care_home_id="home_04", name="Asha Sadan Home",
                 latitude=13.09, longitude=80.28,
                 hard_constraints=HardConstraints(vegetarian_only=True, has_young_children=False),
                 resident_count=30, storage_capacity_kg=200.0,
                 negotiates_via_a2a=True),
        CareHome(care_home_id="home_05", name="Sneha Care Home",
                 latitude=13.07, longitude=80.24,
                 hard_constraints=HardConstraints(vegetarian_only=False, has_young_children=False),
                 resident_count=50, storage_capacity_kg=300.0,
                 negotiates_via_a2a=False),
    ]
    volunteers = [
        Volunteer(volunteer_id="vol_01", name="Priya Sharma",
                  latitude=13.03, longitude=80.21,
                  vehicle_type="car", capacity_kg=60.0),
        Volunteer(volunteer_id="vol_02", name="Ramesh Kumar",
                  latitude=13.05, longitude=80.24,
                  vehicle_type="two_wheeler", capacity_kg=15.0),
    ]
    return WorldConfig(catalog=catalog, stores=stores, care_homes=care_homes,
                       volunteers=volunteers)


def _make_sim_day(world: WorldConfig) -> SimulationDay:
    # milk pushed (1d ≤ threshold 1d), rice pushed (3d ≤ 7d), lentils held back (10d > 14d)
    pushed = [
        DailyFoodItem(name="milk", days_to_expiry=1, quantity=50.0, unit="units"),
        DailyFoodItem(name="rice", days_to_expiry=3, quantity=20.0, unit="kg"),
    ]
    held_back = [
        DailyFoodItem(name="lentils", days_to_expiry=10, quantity=30.0, unit="kg"),
    ]
    store_states = [
        StoreDailyState(
            store_id="store_01",
            full_inventory=pushed + held_back,
            pushed_inventory=pushed,
        ),
        StoreDailyState(
            store_id="store_02",
            full_inventory=pushed,
            pushed_inventory=pushed,
        ),
    ]
    vol_states = [
        VolunteerDailyState(volunteer_id="vol_01", available=True),
        VolunteerDailyState(volunteer_id="vol_02", available=False),
    ]
    return SimulationDay(run_id="test-run-01", stores=store_states, volunteers=vol_states)


def _make_orders() -> List[Order]:
    return [
        Order(
            order_id="ord_01",
            care_home_id="home_01",
            store_id="store_01",
            items=[
                OrderLineItem(item="milk", unit="units",
                              offered_quantity=30.0, accepted_quantity=25.0),
                OrderLineItem(item="rice", unit="kg",
                              offered_quantity=10.0, accepted_quantity=8.0),
            ],
            urgent_essential_items=["milk"],
            final_notice={"arriving_today": ["milk", "rice"], "deferred": [], "message": ""},
        ),
        Order(
            order_id="ord_02",
            care_home_id="home_02",
            store_id="store_01",
            items=[
                OrderLineItem(item="rice", unit="kg",
                              offered_quantity=5.0, accepted_quantity=5.0),
            ],
            urgent_essential_items=[],
        ),
        Order(
            order_id="ord_03",
            care_home_id="home_05",
            store_id="store_02",
            items=[
                OrderLineItem(item="milk", unit="units",
                              offered_quantity=20.0, accepted_quantity=20.0),
            ],
            urgent_essential_items=[],
        ),
    ]


def _make_deliveries() -> List[Delivery]:
    return [
        Delivery(
            delivery_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            store_id="store_01",
            order_ids=["ord_01", "ord_02"],
            method="volunteer",
            volunteer_id="vol_01",
            pickup_time="Today 2:00 PM",
        ),
        Delivery(
            delivery_id="ffffffff-0000-1111-2222-333333333333",
            store_id="store_02",
            order_ids=["ord_03"],
            method="store_truck",
            volunteer_id=None,
            pickup_time="Today 2:00 PM",
        ),
    ]


def _make_negotiation_results() -> List[NegotiationResult]:
    return [
        NegotiationResult(
            care_home_id="home_01",
            status="agreed",
            agreed_items=[
                OrderLineItem(item="milk", unit="units",
                              offered_quantity=30.0, accepted_quantity=25.0),
                OrderLineItem(item="rice", unit="kg",
                              offered_quantity=10.0, accepted_quantity=8.0),
            ],
            urgent_item_names=["milk"],
            negotiation_transcript=[
                NegotiationTurn(turn_number=1, speaker="system", action="offer",
                                item="milk", quantity=30.0),
                NegotiationTurn(turn_number=2, speaker="care_home", action="reduce_item",
                                item="milk", quantity=25.0),
                NegotiationTurn(turn_number=3, speaker="care_home", action="flag_urgent",
                                item="milk"),
                NegotiationTurn(turn_number=4, speaker="care_home", action="accept_all"),
            ],
        ),
        NegotiationResult(
            care_home_id="home_02",
            status="agreed",
            agreed_items=[
                OrderLineItem(item="rice", unit="kg",
                              offered_quantity=5.0, accepted_quantity=5.0),
            ],
            urgent_item_names=[],
            negotiation_transcript=[
                NegotiationTurn(turn_number=1, speaker="system", action="offer"),
                NegotiationTurn(turn_number=2, speaker="care_home", action="accept_all"),
            ],
        ),
        NegotiationResult(
            care_home_id="home_03",
            status="rejected",
            agreed_items=[],
            urgent_item_names=[],
            negotiation_transcript=[
                NegotiationTurn(turn_number=1, speaker="system", action="offer"),
                NegotiationTurn(turn_number=2, speaker="care_home", action="reject_all"),
            ],
            rejection_message="Noted. Will connect with you on another day.",
        ),
        NegotiationResult(
            care_home_id="home_04",
            status="agreed",
            agreed_items=[
                OrderLineItem(item="rice", unit="kg",
                              offered_quantity=8.0, accepted_quantity=8.0),
            ],
            urgent_item_names=[],
            negotiation_transcript=[
                NegotiationTurn(turn_number=1, speaker="system", action="offer"),
                NegotiationTurn(turn_number=2, speaker="care_home", action="accept_all"),
            ],
        ),
        NegotiationResult(
            care_home_id="home_05",
            status="agreed",
            agreed_items=[
                OrderLineItem(item="milk", unit="units",
                              offered_quantity=20.0, accepted_quantity=20.0),
            ],
            urgent_item_names=[],
            negotiation_transcript=[],
        ),
    ]


def _make_dispatch_stats() -> DispatchStats:
    return DispatchStats(
        total_deliveries=2,
        volunteer_assigned=1,
        store_truck_assigned=1,
        commercial_assigned=0,
        volunteers_unavailable=1,
        urgent_items_forced_fallback=0,
        detours_bundled=0,
    )


# ---------------------------------------------------------------------------
# Import the module under test (deferred to avoid path issues at collection)
# ---------------------------------------------------------------------------

def _import_rg():
    from reports import report_generator as rg
    return rg


# ---------------------------------------------------------------------------
# SECTION 1 — generate_delivery_table
# ---------------------------------------------------------------------------

class TestDeliveryTable:
    def test_returns_non_empty_string(self):
        rg = _import_rg()
        world = _make_world()
        result, _ = rg.generate_delivery_table(_make_deliveries(), _make_orders(), world)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_store_name(self):
        rg = _import_rg()
        world = _make_world()
        result, _ = rg.generate_delivery_table(_make_deliveries(), _make_orders(), world)
        assert "Sri Balaji Supermarket" in result

    def test_contains_care_home_name(self):
        rg = _import_rg()
        world = _make_world()
        result, _ = rg.generate_delivery_table(_make_deliveries(), _make_orders(), world)
        assert "Anbu Illam Home" in result

    def test_urgent_items_marked_with_star(self):
        rg = _import_rg()
        world = _make_world()
        result, _ = rg.generate_delivery_table(_make_deliveries(), _make_orders(), world)
        assert "★" in result
        assert "milk" in result

    def test_no_urgent_row_has_no_star_for_that_delivery(self):
        rg = _import_rg()
        world = _make_world()
        # ord_02 has no urgent items — table should still render without crash
        result, _ = rg.generate_delivery_table(_make_deliveries(), _make_orders(), world)
        assert "Store Truck" in result or "store_truck" in result.lower()

    def test_empty_deliveries_returns_string(self):
        rg = _import_rg()
        world = _make_world()
        result, _ = rg.generate_delivery_table([], [], world)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_volunteer_name_in_method_column(self):
        rg = _import_rg()
        world = _make_world()
        result, _ = rg.generate_delivery_table(_make_deliveries(), _make_orders(), world)
        assert "Priya Sharma" in result


# ---------------------------------------------------------------------------
# SECTION 2 — generate_map
# ---------------------------------------------------------------------------

class TestGenerateMap:
    @pytest.fixture(autouse=True)
    def clean_map(self, tmp_path, monkeypatch):
        """Point _REPORTS_DIR to a temp dir so tests don't pollute real output."""
        import reports.report_generator as rg
        monkeypatch.setattr(rg, "_REPORTS_DIR", tmp_path)
        yield tmp_path

    def test_saves_html_to_correct_path(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        filepath, _ = rg.generate_map(_make_deliveries(), _make_orders(), world, "test-run-01")
        assert filepath.endswith("map_test-run-01.html")
        assert os.path.exists(filepath)

    def test_html_contains_store_name(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        filepath, _ = rg.generate_map(_make_deliveries(), _make_orders(), world, "test-run-01")
        content = Path(filepath).read_text(encoding="utf-8")
        assert "Sri Balaji Supermarket" in content

    def test_html_contains_care_home_name(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        filepath, _ = rg.generate_map(_make_deliveries(), _make_orders(), world, "test-run-01")
        content = Path(filepath).read_text(encoding="utf-8")
        assert "Anbu Illam Home" in content

    def test_html_contains_volunteer_name(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        filepath, _ = rg.generate_map(_make_deliveries(), _make_orders(), world, "test-run-01")
        content = Path(filepath).read_text(encoding="utf-8")
        assert "Priya Sharma" in content

    def test_html_contains_legend(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        filepath, _ = rg.generate_map(_make_deliveries(), _make_orders(), world, "test-run-01")
        content = Path(filepath).read_text(encoding="utf-8")
        assert "Delivery Methods" in content
        assert "Routes" in content

    def test_html_assigned_volunteer_color(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        filepath, _ = rg.generate_map(_make_deliveries(), _make_orders(), world, "test-run-01")
        content = Path(filepath).read_text(encoding="utf-8")
        assert "#A0522D" in content

    def test_filepath_uses_run_id(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        filepath, _ = rg.generate_map([], [], world, "unique-run-xyz")
        assert "unique-run-xyz" in filepath

    def test_no_overwrite_different_run_ids(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        fp1, _ = rg.generate_map([], [], world, "run-alpha")
        fp2, _ = rg.generate_map([], [], world, "run-beta")
        assert fp1 != fp2
        assert os.path.exists(fp1)
        assert os.path.exists(fp2)

    def test_marker_position_on_polyline_midpoint(self, tmp_path):
        import polyline
        rg = _import_rg()
        world = _make_world()
        
        # Define a delivery with a curved polyline
        # Start coordinate: [13.04, 80.23] (store_01)
        # End coordinate: [13.06, 80.27] (home_01)
        # Curved path: [13.04, 80.23] -> [13.04, 80.27] -> [13.06, 80.27]
        # Total distance is 0.04 (in degrees, approx)
        # Halfway point is 0.02, which is exactly at [13.04, 80.25]
        # With latitude + 0.003, it should be [13.043, 80.25]
        coords = [(13.04, 80.23), (13.04, 80.27), (13.06, 80.27)]
        poly_str = polyline.encode(coords)
        
        deliveries = [
            Delivery(
                delivery_id="delivery_with_polyline",
                store_id="store_01",
                order_ids=["ord_01"],
                method="store_truck",
                polyline=poly_str,
            )
        ]
        
        orders = [
            Order(
                order_id="ord_01",
                care_home_id="home_01",
                store_id="store_01",
                items=[],
                urgent_essential_items=[],
            )
        ]
        
        filepath, html = rg.generate_map(deliveries, orders, world, "test-polyline-marker")
        
        # Midpoint of straight-line would have been:
        # ((13.04 + 13.06) / 2) + 0.003 = 13.053
        # ((80.23 + 80.27) / 2) = 80.25
        # Straight-line: [13.053, 80.25]
        # Decoded polyline path-length midpoint: [13.043, 80.25]
        
        assert "13.043" in html
        assert "13.053" not in html

    def test_marker_position_fallback_straight_line(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        
        # No polyline present
        deliveries = [
            Delivery(
                delivery_id="delivery_no_polyline",
                store_id="store_01",
                order_ids=["ord_01"],
                method="store_truck",
                polyline=None,
            )
        ]
        
        orders = [
            Order(
                order_id="ord_01",
                care_home_id="home_01",
                store_id="store_01",
                items=[],
                urgent_essential_items=[],
            )
        ]
        
        filepath, html = rg.generate_map(deliveries, orders, world, "test-fallback-marker")
        
        # Midpoint of straight-line:
        # ((13.04 + 13.06) / 2) + 0.003 = 13.053
        # ((80.23 + 80.27) / 2) = 80.25
        # Straight-line: [13.053, 80.25]
        
        assert "13.053" in html

    def test_maps_api_key_safeguard(self, monkeypatch):
        import main
        import asyncio
        # 1. Newline validation
        monkeypatch.setattr(main, "MAPS_API_KEY", "AIzaSyB_6Nih-jUEp8G13mB6bg2ckC1dZ61Vl5o\nGEMINI_API_KEY=test")
        with pytest.raises(ValueError, match="contains newline characters"):
            asyncio.run(main.run_simulation())

        # 2. Hash char validation
        monkeypatch.setattr(main, "MAPS_API_KEY", "AIzaSyB_#6Nih-jUEp8G13mB6bg2ckC1dZ61Vl5o")
        with pytest.raises(ValueError, match="contains '#' characters"):
            asyncio.run(main.run_simulation())

        # 3. Short key validation
        monkeypatch.setattr(main, "MAPS_API_KEY", "too_short")
        with pytest.raises(ValueError, match="appears malformed"):
            asyncio.run(main.run_simulation())

    def test_overlapping_milestones_sequential_matching(self, tmp_path):
        import polyline
        rg = _import_rg()
        world = _make_world()
        
        # Store (13.04, 80.23), Home 1 (13.06, 80.27), Home 2 (13.08, 80.25)
        # We loop physically close to Home 2 early on (idx 1), visit Home 1 (idx 2), then Home 2 (idx 3).
        # Standard naive matching would match Home 2 to idx 1, creating an out-of-order index loop.
        # Sequential progressive matching restricts Home 2's search space to start after Home 1 (idx 2),
        # correctly matching Home 2 to idx 3.
        coords = [
            (13.04, 80.23),    # idx 0: Store
            (13.0795, 80.25),  # idx 1: near Home 2 (loops near it early on)
            (13.06, 80.27),    # idx 2: Home 1
            (13.081, 80.25),   # idx 3: Home 2
        ]
        poly_str = polyline.encode(coords)
        
        deliveries = [
            Delivery(
                delivery_id="del_overlap",
                store_id="store_01",
                order_ids=["ord_01", "ord_02"],
                method="store_truck",
                polyline=poly_str,
            )
        ]
        
        orders = [
            Order(
                order_id="ord_01",
                care_home_id="home_01",
                store_id="store_01",
                items=[],
                urgent_essential_items=[],
            ),
            Order(
                order_id="ord_02",
                care_home_id="home_02",
                store_id="store_01",
                items=[],
                urgent_essential_items=[],
            )
        ]
        
        filepath, html = rg.generate_map(deliveries, orders, world, "test-progressive-matching")
        
        # Confirm that both delivery leg midpoints are correctly computed and rendered:
        # Leg 2 is from coords[2] to coords[3], which has mathematical midpoint lat 13.0735 (with +0.003 offset)
        # Leg 1 is from coords[0] to coords[2], which has path-length midpoint lat 13.0752 (with +0.003 offset)
        assert "13.0735" in html
        assert "13.0752" in html


# ---------------------------------------------------------------------------
# SECTION 3 — generate_negotiation_report
# ---------------------------------------------------------------------------

class TestNegotiationReport:
    def test_contains_all_five_care_homes(self):
        rg = _import_rg()
        world = _make_world()
        result = rg.generate_negotiation_report(
            _make_negotiation_results(), _make_orders(), world
        )
        for ch in world.care_homes:
            assert ch.name in result, f"{ch.name} not in negotiation report"

    def test_auto_accept_home_shows_note(self):
        rg = _import_rg()
        world = _make_world()
        result = rg.generate_negotiation_report(
            _make_negotiation_results(), _make_orders(), world
        )
        assert "Auto-accepted" in result.upper() or "AUTO-ACCEPTED" in result

    def test_rejected_home_shows_rejection_message(self):
        rg = _import_rg()
        world = _make_world()
        result = rg.generate_negotiation_report(
            _make_negotiation_results(), _make_orders(), world
        )
        assert "REJECTED" in result.upper()
        assert "Noted. Will connect" in result

    def test_agreed_home_shows_transcript(self):
        rg = _import_rg()
        world = _make_world()
        result = rg.generate_negotiation_report(
            _make_negotiation_results(), _make_orders(), world
        )
        assert "REDUCE ITEM" in result or "reduce_item" in result.lower()

    def test_urgent_items_shown_with_star(self):
        rg = _import_rg()
        world = _make_world()
        result = rg.generate_negotiation_report(
            _make_negotiation_results(), _make_orders(), world
        )
        assert "★" in result

    def test_home_05_not_negotiates_label(self):
        rg = _import_rg()
        world = _make_world()
        result = rg.generate_negotiation_report(
            _make_negotiation_results(), _make_orders(), world
        )
        # home_05 is auto-accept — should show "No" for negotiates_via_a2a
        assert "auto-accept" in result.lower()

    def test_returns_non_empty_string(self):
        rg = _import_rg()
        world = _make_world()
        result = rg.generate_negotiation_report(
            _make_negotiation_results(), _make_orders(), world
        )
        assert isinstance(result, str)
        assert len(result) > 100

    def test_empty_results_still_shows_all_homes(self):
        rg = _import_rg()
        world = _make_world()
        # Pass empty negotiation_results — all 5 homes should still appear
        result = rg.generate_negotiation_report([], _make_orders(), world)
        for ch in world.care_homes:
            assert ch.name in result

    def test_final_notice_deduplication(self):
        """Test that multiple orders for the same home do not duplicate the final notice message."""
        rg = _import_rg()
        world = _make_world()
        
        # 3 orders for the same home, all with the exact same final_notice message
        notice = {"message": "We have arranged delivery of milk for you today.\nUnfortunately carrots could not be included."}
        orders = [
            Order(order_id="1", care_home_id="home_01", store_id="store_01", items=[], final_notice=notice),
            Order(order_id="2", care_home_id="home_01", store_id="store_02", items=[], final_notice=notice),
            Order(order_id="3", care_home_id="home_01", store_id="store_03", items=[], final_notice=notice),
        ]
        
        # We need a negotiation result to trigger the html_part branch that prints final notice
        mock_result = NegotiationResult(
            care_home_id="home_01", status="agreed", agreed_items=[], urgent_item_names=[], negotiation_transcript=[]
        )
        results_list = [mock_result]
        
        result = rg.generate_negotiation_report(results_list, orders, world)
        
        # The message should only appear exactly ONCE in the HTML block for this care home
        # Use simple string counting
        msg_count = result.count("We have arranged delivery of milk")
        assert msg_count == 1, f"Expected final notice message to appear exactly once, but found it {msg_count} times."


# ---------------------------------------------------------------------------
# SECTION 4 — generate_audit_report
# ---------------------------------------------------------------------------

class TestAuditReport:
    def test_returns_non_empty_string(self):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        result = rg.generate_audit_report(sim_day, world, _make_dispatch_stats())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_shows_pushed_item_count(self):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        result = rg.generate_audit_report(sim_day, world, _make_dispatch_stats())
        # store_01 has 2 pushed items
        assert "2" in result

    def test_shows_held_back_items(self):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        result = rg.generate_audit_report(sim_day, world, _make_dispatch_stats())
        # lentils are held back in store_01
        assert "lentils" in result.lower() or "HELD BACK" in result

    def test_shows_dispatch_stats(self):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        stats = _make_dispatch_stats()
        result = rg.generate_audit_report(sim_day, world, stats)
        assert "2" in result  # total_deliveries
        assert "1" in result  # volunteer_assigned

    def test_commercial_pickups_noted_when_present(self):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        stats = DispatchStats(
            total_deliveries=3,
            volunteer_assigned=1,
            store_truck_assigned=1,
            commercial_assigned=1,
            volunteers_unavailable=2,
            urgent_items_forced_fallback=1,
            detours_bundled=0,
        )
        result = rg.generate_audit_report(sim_day, world, stats)
        assert "commercial" in result.lower()

    def test_store_names_in_audit(self):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        result = rg.generate_audit_report(sim_day, world, _make_dispatch_stats())
        assert "Sri Balaji Supermarket" in result

    def test_pushed_items_listed_with_expiry(self):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        result = rg.generate_audit_report(sim_day, world, _make_dispatch_stats())
        assert "milk" in result.lower()
        assert "1d" in result or "expires in 1" in result


# ---------------------------------------------------------------------------
# SECTION — generate_summary_page
# ---------------------------------------------------------------------------

class TestGenerateSummaryPage:
    @pytest.fixture(autouse=True)
    def clean_output(self, tmp_path, monkeypatch):
        import reports.report_generator as rg
        monkeypatch.setattr(rg, "_REPORTS_DIR", tmp_path)
        yield tmp_path

    def test_summary_page_generation(self):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        deliveries = _make_deliveries()
        orders = _make_orders()
        negotiation_results = _make_negotiation_results()
        dispatch_stats = _make_dispatch_stats()
        
        filepath, html = rg.generate_summary_page(
            deliveries, orders, negotiation_results, dispatch_stats, world, sim_day, "summary-test-01"
        )
        
        assert "summary_summary-test-01.html" in filepath
        assert "SurplusCart &mdash; Daily Impact Summary" in html
        assert "Food Rescued (kg)" in html
        assert "Meals Served" in html
        assert "Care Homes Served" in html
        assert "CO2e Avoided" in html
        # Buttons check
        assert "View Detailed Report" in html
        assert "View Delivery Map" in html
        

# ---------------------------------------------------------------------------
# SECTION — generate_full_report
# ---------------------------------------------------------------------------

class TestGenerateFullReport:
    @pytest.fixture(autouse=True)
    def clean_map(self, tmp_path, monkeypatch):
        import reports.report_generator as rg
        monkeypatch.setattr(rg, "_REPORTS_DIR", tmp_path)
        yield tmp_path

    @pytest.mark.asyncio
    async def test_returns_dict_with_all_keys(self):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        result = await rg.generate_full_report(
            deliveries=_make_deliveries(),
            orders=_make_orders(),
            negotiation_results=_make_negotiation_results(),
            dispatch_stats=_make_dispatch_stats(),
            world=world,
            sim_day=sim_day,
            run_id="full-test-01",
        )
        assert set(result.keys()) == {
            "report_html",
            "map_html",
            "summary_html",
            "report_filepath",
            "map_filepath",
            "summary_filepath",
            "delivery_table",
            "negotiation_report",
            "audit_report",
            "message_log",
            "stats"
        }

    @pytest.mark.asyncio
    async def test_all_sections_non_empty(self):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        result = await rg.generate_full_report(
            deliveries=_make_deliveries(),
            orders=_make_orders(),
            negotiation_results=_make_negotiation_results(),
            dispatch_stats=_make_dispatch_stats(),
            world=world,
            sim_day=sim_day,
            run_id="full-test-02",
        )
        assert len(result["delivery_table"]) > 0
        assert result["map_filepath"].endswith(".html")
        assert len(result["negotiation_report"]) > 0
        assert len(result["audit_report"]) > 0
        assert isinstance(result["message_log"], list)

    def test_audit_allocation_math(self):
        """Test that Pushed Qty = Sum of Allocated + Unallocated across multiple homes."""
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        stats = _make_dispatch_stats()
        
        # Pushed milk: 50.0 units.
        # Home 01 takes 20, Home 02 takes 15. Total allocated = 35.0. Unallocated = 15.0
        orders = [
            Order(
                order_id="ord_test_01",
                care_home_id="home_01",
                store_id="store_01",
                items=[OrderLineItem(item="milk", unit="units", offered_quantity=20.0, accepted_quantity=20.0)],
                urgent_essential_items=[],
                negotiation_transcript=[],
            ),
            Order(
                order_id="ord_test_02",
                care_home_id="home_02",
                store_id="store_01",
                items=[OrderLineItem(item="milk", unit="units", offered_quantity=15.0, accepted_quantity=15.0)],
                urgent_essential_items=[],
                negotiation_transcript=[],
            )
        ]
        
        # Generate full report to trigger store inventory html generation
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(rg.generate_full_report(
                deliveries=_make_deliveries(),
                orders=orders,
                negotiation_results=_make_negotiation_results(),
                dispatch_stats=stats,
                world=world,
                sim_day=sim_day,
                run_id="audit-math-test"
            ))
        finally:
            loop.close()
            
        # The math inside generate_full_report section 1 html string
        html = result["report_html"]
        # milk row should have Pushed Qty = 50.0, Allocated to Care Home = "Anbu Illam Home (20.0), Karuna Trust Home (15.0)"
        # and Unallocated = 15.0
        assert "50.0" in html
        assert "Anbu Illam Home (20.0)" in html
        assert "Karuna Trust Home (15.0)" in html
        assert "15.0" in html # Unallocated


    @pytest.mark.asyncio
    async def test_map_file_created(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        result = await rg.generate_full_report(
            deliveries=_make_deliveries(),
            orders=_make_orders(),
            negotiation_results=_make_negotiation_results(),
            dispatch_stats=_make_dispatch_stats(),
            world=world,
            sim_day=sim_day,
            run_id="full-test-03",
        )
        assert os.path.exists(result["map_filepath"])

    @pytest.mark.asyncio
    async def test_html_report_elements(self, tmp_path):
        rg = _import_rg()
        world = _make_world()
        sim_day = _make_sim_day(world)
        result = await rg.generate_full_report(
            deliveries=_make_deliveries(),
            orders=_make_orders(),
            negotiation_results=_make_negotiation_results(),
            dispatch_stats=_make_dispatch_stats(),
            world=world,
            sim_day=sim_day,
            run_id="full-test-html",
        )
        report_path = result["report_filepath"]
        assert os.path.exists(report_path)
        
        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
            
        assert "SurplusCart" in html
        assert "Store Inventory Report" in html
        assert "Care Home Allocation Summary" in html
        assert "Volunteer Availability and Dispatch Status" in html
        assert "Delivery Routes and Dispatch Decisions" in html
        assert "View Map" in html
        
        latest_path = os.path.join(os.path.dirname(report_path), "latest_report.html")
        assert os.path.exists(latest_path)
        
        for store in world.stores:
            assert store.name in html
            
        for ch in world.care_homes:
            assert ch.name in html
            
        for vol in world.volunteers:
            assert vol.name in html
            
        assert "highlight-waste" in html or "#01F3F4" in html
        assert "rejected-row" in html or "#087C81" in html
        assert "vol-assigned" in html or "vol-unavailable" in html
        
        map_path = result["map_filepath"]
        assert os.path.exists(map_path)
        with open(map_path, "r", encoding="utf-8") as fm:
            map_html = fm.read()
        assert "View Report" in map_html
        assert "latest_report.html" in map_html or "/report" in map_html


# ---------------------------------------------------------------------------
# Output directory auto-creation
# ---------------------------------------------------------------------------

class TestOutputDirectoryCreation:
    def test_output_dir_created_if_missing(self, tmp_path, monkeypatch):
        import reports.report_generator as rg
        new_output = tmp_path / "brand_new_output"
        assert not new_output.exists()
        monkeypatch.setattr(rg, "_REPORTS_DIR", new_output)
        rg._ensure_output_dir()
        assert new_output.exists()

    def test_output_dir_not_error_if_already_exists(self, tmp_path, monkeypatch):
        import reports.report_generator as rg
        existing = tmp_path / "existing"
        existing.mkdir()
        monkeypatch.setattr(rg, "_REPORTS_DIR", existing)
        # Should not raise
        rg._ensure_output_dir()
        assert existing.exists()
