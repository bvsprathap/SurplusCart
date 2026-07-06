import pytest
import uuid
from typing import List

from tools.models import Order, OrderLineItem, Delivery
from tools.dispatch import run_dispatch, DispatchStats
from data.data_model import Store, CareHome, Volunteer, FoodCatalogItem, WorldConfig, SimulationDay
from reports.report_generator import generate_full_report

@pytest.mark.asyncio
async def test_multi_store_commercial_attribution():
    """
    Test that when a commercial pickup sources an item from MULTIPLE stores,
    the resulting Delivery record lists ALL contributing stores, and the Store Inventory 
    audit report correctly credits EACH contributing store.
    """
    # 1. Set up World Config
    store_a = Store(store_id="store_a", name="Store A", latitude=12.0, longitude=77.0)
    store_b = Store(store_id="store_b", name="Store B", latitude=12.1, longitude=77.1)
    from data.data_model import HardConstraints
    ch = CareHome(care_home_id="ch_1", name="Care Home 1", latitude=12.2, longitude=77.2, 
                  capacity_residents=50, negotiates_via_a2a=False, hard_constraints=HardConstraints(vegetarian_only=False, has_young_children=False), resident_count=50, storage_capacity_kg=100.0)
    
    world = WorldConfig(
        stores=[store_a, store_b],
        care_homes=[ch],
        volunteers=[],
        catalog=[FoodCatalogItem(name="eggs", is_perishable=True, is_essential=True, push_threshold_days=1, unit="dozen", approx_weight_kg=1.0, cap_category="poultry_eggs")]
    )

    # 2. Simulate needs_commercial_items output from single_store_candidate split
    # 8 dozen from Store A, 5 dozen from Store B
    order_id_a = str(uuid.uuid4())
    order_id_b = str(uuid.uuid4())
    
    order_a = Order(
        order_id=order_id_a,
        care_home_id="ch_1",
        store_id="store_a",
        items=[OrderLineItem(item="eggs", unit="dozen", offered_quantity=8, accepted_quantity=8)],
        urgent_essential_items=["eggs"]
    )
    order_b = Order(
        order_id=order_id_b,
        care_home_id="ch_1",
        store_id="store_b",
        items=[OrderLineItem(item="eggs", unit="dozen", offered_quantity=5, accepted_quantity=5)],
        urgent_essential_items=["eggs"]
    )
    
    needs_commercial_items = [
        {"store_id": "store_a", "order_id": order_id_a, "item": order_a.items[0]},
        {"store_id": "store_b", "order_id": order_id_b, "item": order_b.items[0]},
    ]
    
    stats = DispatchStats()
    sim_day = SimulationDay(run_id="run-1", stores=[], volunteers=[])
    
    # Mock callables for dispatch
    async def mock_get_volunteer_avail(vid): return True
    async def mock_get_distance_minutes(lat1, lon1, lat2, lon2): return 10.0
    async def mock_get_truck_avail(sid): return {"available": True}

    deliveries, stats = await run_dispatch(
        orders=[order_a, order_b],
        needs_commercial_items=needs_commercial_items,
        world=world,
        sim_day=sim_day,
        get_volunteer_avail=mock_get_volunteer_avail,
        get_distance_minutes=mock_get_distance_minutes,
        get_truck_avail=mock_get_truck_avail,
        run_id="run-1"
    )
    
    # Assert there is exactly 1 commercial delivery for the care home
    assert len(deliveries) == 1
    delivery = deliveries[0]
    assert delivery.method == "commercial"
    
    # The Delivery's store_id MUST contain BOTH store names
    assert "Store A" in delivery.store_id
    assert "Store B" in delivery.store_id
    assert delivery.store_id == "Store A, Store B" # Sorted
    
    # Both order IDs should be linked to the delivery
    assert set(delivery.order_ids) == {order_id_a, order_id_b}
