"""
mcp_servers/store_volunteer_server.py

MCP server exposing store inventory and volunteer availability as external tools.
Agents connect via ADK MCPToolset (stdio transport). The server holds a reference
to the current SimulationDay and mutates it for truck_status caching.

Tools exposed:
  - get_pushable_inventory(store_id)
  - check_vehicle_availability(store_id)
  - get_volunteer_schedule(volunteer_id)
"""

import sys
import os
import random
import json
from typing import Any

from mcp.server.fastmcp import FastMCP

# ── Path setup so we can import from project root ──────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.data_model import (
    SimulationDay,
    WorldConfig,
    DailyFoodItem,
    TruckStatus,
    VolunteerDailyState,
    get_pushable_inventory as _passes_threshold,
    setup_world,
    generate_daily_data,
)

# ── Module-level simulation state ──────────────────────────────────────────────
# These are set once by whoever launches the server (tests, orchestrator, etc.)
# They are also set inside _load_state() for subprocess launches.
_sim_day: SimulationDay | None = None
_world: WorldConfig | None = None


def init_server_state(world: WorldConfig, sim_day: SimulationDay) -> None:
    """
    Inject a pre-built WorldConfig and SimulationDay into the server.
    Call this before starting the server in-process (e.g., from tests).
    """
    global _sim_day, _world
    _world = world
    _sim_day = sim_day


def _load_state() -> None:
    """
    Fallback: load state from disk when the server is launched as a subprocess.
    Reads catalog.json and world_config.json from the project root, then
    generates a fresh SimulationDay.
    """
    global _sim_day, _world
    catalog_path = os.path.join(PROJECT_ROOT, "catalog.json")
    world_path = os.path.join(PROJECT_ROOT, "world_config.json")
    _world = setup_world(world_path, catalog_path)
    _sim_day = generate_daily_data(_world)


# ── FastMCP server ─────────────────────────────────────────────────────────────
mcp = FastMCP("store-volunteer-server")


@mcp.tool()
def get_pushable_inventory(store_id: str) -> list[dict[str, Any]]:
    """
    Return the list of items this store is pushing today (near-expiry or near
    long-shelf threshold). This simulates the store's own internal decision.

    Only pushed_inventory items (those that pass the push threshold) are returned.
    full_inventory is NEVER exposed to callers.

    Args:
        store_id: The store identifier (e.g. "store_01").

    Returns:
        List of DailyFoodItem dicts with keys: name, days_to_expiry, quantity, unit.
    """
    if _sim_day is None or _world is None:
        raise RuntimeError("Server state not initialised — call init_server_state() first.")

    store_state = next(
        (s for s in _sim_day.stores if s.store_id == store_id), None
    )
    if store_state is None:
        raise ValueError(f"Unknown store_id: {store_id!r}")

    # Return the pre-computed pushed_inventory — we never leak full_inventory
    return [item.model_dump() for item in store_state.pushed_inventory]


@mcp.tool()
def check_vehicle_availability(store_id: str) -> dict[str, Any]:
    """
    Check whether the store's own truck is available for delivery fallback.

    On the FIRST call for a given store_id in this run, availability is
    randomly decided and cached into StoreDailyState.truck_status.
    Subsequent calls for the same store_id return the cached value — no re-roll.

    Args:
        store_id: The store identifier (e.g. "store_01").

    Returns:
        TruckStatus dict with keys: available (bool), capacity_kg (float).
    """
    if _sim_day is None or _world is None:
        raise RuntimeError("Server state not initialised — call init_server_state() first.")

    store_state = next(
        (s for s in _sim_day.stores if s.store_id == store_id), None
    )
    if store_state is None:
        raise ValueError(f"Unknown store_id: {store_id!r}")

    # Return cached result if already checked this run
    if store_state.truck_status is not None:
        return store_state.truck_status.model_dump()

    # First call: generate and cache
    store_config = next(
        (s for s in _world.stores if s.store_id == store_id), None
    )
    # Use the store's configured truck capacity from WorldConfig
    capacity_kg = store_config.truck_capacity_kg if store_config else 0.0
    available = random.random() < 0.75  # 75% chance the truck is free

    status = TruckStatus(available=available, capacity_kg=capacity_kg)
    store_state.truck_status = status  # mutate in-place to cache

    return status.model_dump()


@mcp.tool()
def get_volunteer_schedule(volunteer_id: str) -> dict[str, Any]:
    """
    Return a volunteer's availability for today, as if reading their phone calendar.

    Args:
        volunteer_id: The volunteer identifier (e.g. "vol_01").

    Returns:
        VolunteerDailyState dict with keys: volunteer_id (str), available (bool).
    """
    if _sim_day is None or _world is None:
        raise RuntimeError("Server state not initialised — call init_server_state() first.")

    vol_state = next(
        (v for v in _sim_day.volunteers if v.volunteer_id == volunteer_id), None
    )
    if vol_state is None:
        raise ValueError(f"Unknown volunteer_id: {volunteer_id!r}")

    return vol_state.model_dump()


# ── Entrypoint for subprocess launch (stdio) ───────────────────────────────────
if __name__ == "__main__":
    _load_state()
    mcp.run(transport="stdio")
