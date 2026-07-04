"""
tests/test_store_volunteer_mcp.py

Standalone test script for the Store + Volunteer MCP server.
Calls all three tools directly (no agent, no subprocess) using the
in-process init_server_state() path.

Checks:
  1. get_pushable_inventory returns only pushed items (count matches pushed_inventory)
  2. check_vehicle_availability caches on first call — second call returns same result
  3. get_volunteer_schedule returns correct availability flags
"""

import sys
import os

# ── Add project root to path ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.data_model import setup_world, generate_daily_data
from mcp_servers.store_volunteer_server import (
    init_server_state,
    get_pushable_inventory,
    check_vehicle_availability,
    get_volunteer_schedule,
    _sim_day,
)

# ── Re-import to capture the module-level reference after init ─────────────────
import mcp_servers.store_volunteer_server as _srv


def run_tests() -> None:
    # ── Setup ──────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("MCP Server Test — Store & Volunteer Tools")
    print("=" * 70)

    catalog_path = os.path.join(PROJECT_ROOT, "catalog.json")
    world_path = os.path.join(PROJECT_ROOT, "world_config.json")

    world = setup_world(world_path, catalog_path)
    sim_day = generate_daily_data(world)
    init_server_state(world, sim_day)

    # Refresh our reference after init so we read the module-level state
    sim_day_ref = _srv._sim_day
    assert sim_day_ref is not None, "Server state not initialised"

    print(f"\nRun ID : {sim_day_ref.run_id}")
    print(f"Stores : {[s.store_id for s in sim_day_ref.stores]}")
    print(f"Volunteers: {[v.volunteer_id for v in sim_day_ref.volunteers]}\n")

    all_passed = True

    # -- Test 1: get_pushable_inventory -----------------------------------------
    print("-" * 70)
    print("TEST 1: get_pushable_inventory")
    print("-" * 70)
    for store_state in sim_day_ref.stores:
        sid = store_state.store_id
        result = get_pushable_inventory(sid)

        expected_count = len(store_state.pushed_inventory)
        actual_count = len(result)

        # Verify item names match pushed_inventory exactly
        expected_names = sorted(i.name for i in store_state.pushed_inventory)
        actual_names = sorted(i["name"] for i in result)

        if actual_count == expected_count and actual_names == expected_names:
            print(f"  [PASS] {sid}: {actual_count} pushed items returned correctly")
        else:
            print(f"  [FAIL] {sid}: expected {expected_count} items {expected_names}, "
                  f"got {actual_count} items {actual_names}")
            all_passed = False

        # Confirm full_inventory items NOT in result (audit-only items excluded)
        full_names = {i.name for i in store_state.full_inventory}
        pushed_names_set = {i.name for i in store_state.pushed_inventory}
        audit_only = full_names - pushed_names_set
        leaked = {i["name"] for i in result} & audit_only
        if leaked:
            print(f"  [FAIL] {sid}: full_inventory items leaked into result: {leaked}")
            all_passed = False
        else:
            print(f"  [PASS] {sid}: no full_inventory items leaked")

    # -- Test 2: check_vehicle_availability caching ----------------------------
    print("\n" + "-" * 70)
    print("TEST 2: check_vehicle_availability -- caching on first call")
    print("-" * 70)
    test_store = sim_day_ref.stores[0].store_id

    # Ensure no pre-cached value
    _srv._sim_day.stores[0].truck_status = None
    
    result1 = check_vehicle_availability(test_store)
    cached_status = _srv._sim_day.stores[0].truck_status

    if cached_status is None:
        print(f"  [FAIL] {test_store}: truck_status not cached after first call")
        all_passed = False
    else:
        print(f"  [PASS] {test_store}: truck_status cached after first call: {result1}")

    result2 = check_vehicle_availability(test_store)

    if result1 == result2:
        print(f"  [PASS] {test_store}: second call returned cached value (no re-roll)")
    else:
        print(f"  [FAIL] {test_store}: second call returned different result! "
              f"call1={result1}, call2={result2}")
        all_passed = False

    # Verify the cached value on the model matches what was returned
    if cached_status.model_dump() == result1:
        print(f"  [PASS] {test_store}: StoreDailyState.truck_status matches returned dict")
    else:
        print(f"  [FAIL] {test_store}: StoreDailyState.truck_status mismatch")
        all_passed = False

    # -- Test 3: get_volunteer_schedule ----------------------------------------
    print("\n" + "-" * 70)
    print("TEST 3: get_volunteer_schedule")
    print("-" * 70)
    for vol_state in sim_day_ref.volunteers:
        vid = vol_state.volunteer_id
        result = get_volunteer_schedule(vid)

        if (result["volunteer_id"] == vid
                and result["available"] == vol_state.available):
            status = "available" if vol_state.available else "unavailable"
            print(f"  [PASS] {vid}: correctly reported as {status}")
        else:
            print(f"  [FAIL] {vid}: expected available={vol_state.available}, "
                  f"got {result}")
            all_passed = False

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    run_tests()
