"""
main.py

End-to-end orchestrator for the Food Rescue Multi-Agent simulation.
Single entry point that wires all components (Prompts 1-10) into one
complete pipeline for a simulated day.

Pipeline sequence:
  SETUP -> CARE HOME LOOP (5 homes, sequential) -> DETOUR BUNDLING -> DISPATCH -> REPORTING

Usage:
    python main.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Set
from uuid import uuid4

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from data.data_model import (
    CareHome,
    DailyFoodItem,
    FoodCatalogItem,
    SimulationDay,
    Store,
    WorldConfig,
    generate_daily_data,
    setup_world,
)
from tools.constraint_tools import StockLedger, hard_constraint_filter, single_store_candidate
from tools.logger import clear_log, set_run_id
from tools.models import (
    Delivery,
    DispatchStats,
    MatchmakerOffer,
    NegotiationResult,
    NegotiationTurn,
    Order,
    OrderLineItem,
)
from tools.dispatch import check_detour_bundle, run_dispatch
from agents.matchmaker_agent import run_matchmaker
from agents.culinary_agent import run_culinary
from agents.care_home_agent import run_negotiation
from mcp_servers.store_volunteer_server import init_server_state
from reports.report_generator import generate_full_report
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(PYTHON_EXE):
    # Linux / macOS fallback
    PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if not os.path.exists(PYTHON_EXE):
        PYTHON_EXE = sys.executable

SERVER_MODULE = os.path.join(PROJECT_ROOT, "mcp_servers", "store_volunteer_server.py")
MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")


# ---------------------------------------------------------------------------
# MCP response extraction helper
# ---------------------------------------------------------------------------

def _extract_structured(result: Any) -> Any:
    """
    Pull the actual payload out of ADK's MCP response envelope.
    ADK wraps tool responses as:
      {'content': [...], 'structuredContent': <actual value>, 'isError': bool}
    Always use structuredContent; never parse content text for data.
    """
    if isinstance(result, dict):
        if result.get("isError"):
            raise RuntimeError(f"MCP tool returned an error: {result}")
        sc = result.get("structuredContent")
        if sc is not None:
            # FastMCP wraps list returns under {"result": [...]}
            if isinstance(sc, dict) and list(sc.keys()) == ["result"]:
                return sc["result"]
            return sc
        # Fallback: parse first content text block
        content = result.get("content", [])
        if content and isinstance(content[0], dict):
            return json.loads(content[0]["text"])
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_simulation() -> dict:
    """
    Full end-to-end simulation pipeline. Returns the report dict.

    Pipeline:
      SETUP -> CARE HOME LOOP (5 homes) -> DETOUR BUNDLING -> DISPATCH -> REPORTING
    """
    start_time = time.time()

    # ---------------------------------------------------------------------------
    # SETUP
    # ---------------------------------------------------------------------------

    # Step 1: Load world configuration
    print("=" * 70)
    print("  FOOD RESCUE SIMULATION -- Starting Pipeline")
    print("=" * 70)

    world = setup_world("world_config.json", "catalog.json")
    print(f"  [SETUP] World loaded: {len(world.stores)} stores, "
          f"{len(world.care_homes)} care homes, {len(world.volunteers)} volunteers")

    # Step 2: Generate daily simulation data
    sim_day = generate_daily_data(world)
    print(f"  [SETUP] SimulationDay generated")

    # Step 3: Set run_id
    run_id = sim_day.run_id
    set_run_id(run_id)
    print(f"  [SETUP] Run ID: {run_id}")

    # Step 4: Clear message log
    clear_log()
    print(f"  [SETUP] Message log cleared")

    # Step 5: Initialize StockLedger
    ledger = StockLedger(sim_day)
    print(f"  [SETUP] StockLedger initialized -- {len(ledger.store_ids())} stores tracked")

    # Step 6 & 7: Start MCP connections

    store_toolset = None
    maps_toolset = None

    try:
        # Step 6a: Store MCP server (subprocess)
        print(f"  [SETUP] Starting Store MCP server...")
        store_toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=PYTHON_EXE,
                    args=[SERVER_MODULE],
                    env={**os.environ},
                ),
                timeout=30,
            ),
        )
        store_tools = await store_toolset.get_tools()
        store_tool_map = {t.name: t for t in store_tools}

        required_store_tools = {"get_volunteer_schedule", "check_vehicle_availability", "get_pushable_inventory"}
        if not required_store_tools.issubset(set(store_tool_map.keys())):
            raise RuntimeError(
                f"Store MCP server failed to start - check server file path. "
                f"Expected tools {required_store_tools}, got {set(store_tool_map.keys())}"
            )
        print(f"  [SETUP] Store MCP server started -- {len(store_tools)} tools discovered")

        # Step 6b: Maps MCP server (subprocess)
        print(f"  [SETUP] Starting Maps MCP server...")
        maps_env = {
            **os.environ,
            "GOOGLE_MAPS_API_KEY": MAPS_API_KEY,
            "GOOGLE_MAPS_ENABLED_TOOLS": "maps_distance_matrix,maps_directions",
        }
        import shutil
        maps_toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=shutil.which("npx") or ("npx.cmd" if os.name == "nt" else "/usr/bin/npx"),
                    args=["-y", "@cablate/mcp-google-map", "--stdio"],
                    env=maps_env,
                ),
                timeout=30,
            ),
        )
        maps_tools = await maps_toolset.get_tools()
        maps_tool_map = {t.name: t for t in maps_tools}
        print(f"  [SETUP] Maps MCP server started -- {len(maps_tools)} tools discovered")

        # Step 7: Initialize store server state
        init_server_state(world, sim_day)
        print(f"  [SETUP] Store MCP server state initialized")

        # ---------------------------------------------------------------------------
        # MCP CALLABLE WRAPPERS (for dispatch)
        # ---------------------------------------------------------------------------

        async def get_volunteer_avail(volunteer_id: str) -> dict:
            result = await store_tool_map["get_volunteer_schedule"].run_async(
                args={"volunteer_id": volunteer_id}, tool_context=None
            )
            return _extract_structured(result)

        async def get_distance_minutes(olat: float, olng: float, dlat: float, dlng: float) -> float:
            result = await maps_tool_map["maps_distance_matrix"].run_async(
                args={
                    "origins": [f"{olat},{olng}"],
                    "destinations": [f"{dlat},{dlng}"],
                },
                tool_context=None,
            )
            data = _extract_structured(result)
            seconds = data["durations"][0][0]["value"]
            return float(seconds) / 60.0

        async def get_truck_avail(store_id: str) -> dict:
            result = await store_tool_map["check_vehicle_availability"].run_async(
                args={"store_id": store_id}, tool_context=None
            )
            return _extract_structured(result)

        # ---------------------------------------------------------------------------
        # CARE HOME PROCESSING LOOP
        # ---------------------------------------------------------------------------

        # Sort care homes in fixed order: home_01 -> 02 -> 03 -> 04 -> 05
        sorted_care_homes = sorted(world.care_homes, key=lambda ch: ch.care_home_id)

        all_orders: List[Order] = []
        all_negotiation_results: List[NegotiationResult] = []
        all_needs_commercial: List[OrderLineItem] = []

        catalog = world.catalog

        for care_home in sorted_care_homes:
            print(f"\n{'-' * 70}")
            print(f"  Processing: {care_home.name} ({care_home.care_home_id})")
            print(f"{'-' * 70}")

            # ------------------------------------------------------------
            # PHASE 1A - Matchmaker
            # ------------------------------------------------------------

            # Step a: Hard constraint filter - get all ledger items as DailyFoodItem list
            all_ledger_items: List[DailyFoodItem] = []
            for store_state in sim_day.stores:
                for item in store_state.pushed_inventory:
                    # Only include items that still have ledger stock
                    remaining = ledger.get_available(store_state.store_id, item.name)
                    if remaining > 0:
                        all_ledger_items.append(DailyFoodItem(
                            name=item.name,
                            days_to_expiry=item.days_to_expiry,
                            quantity=remaining,
                            unit=item.unit,
                        ))

            # Deduplicate items across stores (take earliest expiry, sum quantity)
            deduped: Dict[str, DailyFoodItem] = {}
            for item in all_ledger_items:
                key = item.name.lower()
                if key in deduped:
                    existing = deduped[key]
                    deduped[key] = DailyFoodItem(
                        name=item.name,
                        days_to_expiry=min(existing.days_to_expiry, item.days_to_expiry),
                        quantity=existing.quantity + item.quantity,
                        unit=item.unit,
                    )
                else:
                    deduped[key] = item
            all_items_deduped = list(deduped.values())

            eligible_items = hard_constraint_filter(care_home, all_items_deduped, catalog)

            # Track excluded item names for guardrail
            eligible_names = {it.name.lower() for it in eligible_items}
            all_names = {it.name.lower() for it in all_items_deduped}
            excluded_item_names = list(all_names - eligible_names)

            # Step b: Get cross-store totals and snapshot
            cross_store_totals = ledger.get_cross_store_totals()
            snapshot = dict(cross_store_totals)  # frozen copy

            # Step c: Run matchmaker
            print(f"  [1A] Running Matchmaker ({len(eligible_items)} eligible items)...")
            try:
                offer = await run_matchmaker(
                    care_home=care_home,
                    eligible_items=eligible_items,
                    catalog=catalog,
                    cross_store_totals=cross_store_totals,
                    cross_store_snapshot=snapshot,
                    excluded_item_names=excluded_item_names,
                )
                print(f"  [1A] Matchmaker offer: {len(offer.offered_items)} items")
            except RuntimeError as e:
                # Matchmaker validation failed twice - skip this care home
                logger.error(
                    "Matchmaker failed for %s after retry: %s - skipping",
                    care_home.care_home_id, e,
                )
                print(f"  [1A] [X] Matchmaker failed for {care_home.care_home_id}, skipping")
                continue

            # ------------------------------------------------------------
            # PHASE 1B - Culinary enrichment
            # ------------------------------------------------------------

            print(f"  [1B] Running Culinary agent...")
            dish_framing = await run_culinary(offer.offered_items)
            final_message = offer.offer_message.replace("[DISH_FRAMING]", dish_framing)
            offer.offer_message = final_message
            print(f"  [1B] Culinary enrichment complete")

            # ------------------------------------------------------------
            # PHASE 1C - Negotiation
            # ------------------------------------------------------------

            negotiation_result: NegotiationResult

            if care_home.negotiates_via_a2a:
                # A2A negotiation (home_01 through home_04)
                print(f"  [1C] Running A2A negotiation...")
                negotiation_result = await run_negotiation(
                    care_home=care_home,
                    offer=offer,
                    ledger=ledger,
                    catalog=catalog,
                    run_id=run_id,
                )

                if negotiation_result.status == "rejected":
                    print(f"  [1C] [X] Offer REJECTED by {care_home.name}")
                    logger.info(
                        "Care home %s rejected the offer - skipping Phase 2",
                        care_home.care_home_id,
                    )
                    all_negotiation_results.append(negotiation_result)
                    continue  # Skip Phase 2 - do NOT deduct from ledger
                else:
                    print(f"  [1C] [OK] Offer AGREED -- {len(negotiation_result.agreed_items)} items")
            else:
                # Auto-accept (home_05): build NegotiationResult directly
                print(f"  [1C] Auto-accept path (no A2A negotiation)")
                agreed_items = [
                    OrderLineItem(
                        item=oi.item,
                        unit=oi.unit,
                        offered_quantity=oi.offered_quantity,
                        accepted_quantity=oi.offered_quantity,
                    )
                    for oi in offer.offered_items
                ]
                negotiation_result = NegotiationResult(
                    care_home_id=care_home.care_home_id,
                    status="agreed",
                    agreed_items=agreed_items,
                    urgent_item_names=[],
                    negotiation_transcript=[
                        NegotiationTurn(
                            turn_number=1,
                            speaker="system",
                            action="offer",
                        ),
                        NegotiationTurn(
                            turn_number=2,
                            speaker="care_home",
                            action="accept_all",
                        ),
                    ],
                    rejection_message=None,
                )
                print(f"  [1C] [OK] Auto-accepted -- {len(agreed_items)} items")

            all_negotiation_results.append(negotiation_result)

            # ------------------------------------------------------------
            # PHASE 2 - Sourcing & Order creation
            # ------------------------------------------------------------

            print(f"  [2] Running single_store_candidate sourcing...")

            # Step i: Call single_store_candidate
            sourcing = single_store_candidate(
                requested_items=negotiation_result.agreed_items,
                urgent_item_names=set(negotiation_result.urgent_item_names),
                ledger=ledger,
                catalog=catalog,
            )

            assignments = sourcing["assignments"]
            deferred = sourcing["deferred"]
            needs_commercial = sourcing["needs_commercial"]

            print(f"  [2] Sourcing: {len(assignments)} assignment(s), "
                  f"{len(deferred)} deferred, {len(needs_commercial)} needs_commercial")

            # Step j: Deduct from ledger and create Orders
            care_home_orders: List[Order] = []

            for assignment in assignments:
                store_id = assignment["store_id"]
                items: List[OrderLineItem] = assignment["items"]

                # Deduct each item from ledger
                for line_item in items:
                    ledger.deduct(store_id, line_item.item, line_item.accepted_quantity)

                # Determine which urgent items are in this assignment
                assignment_item_names = {li.item.lower() for li in items}
                urgent_essential = [
                    item_name for item_name in negotiation_result.urgent_item_names
                    if item_name.lower() in assignment_item_names
                ]

                order = Order(
                    order_id=str(uuid4()),
                    care_home_id=care_home.care_home_id,
                    store_id=store_id,
                    items=items,
                    urgent_essential_items=urgent_essential,
                    negotiation_transcript=negotiation_result.negotiation_transcript,
                    final_notice={},  # populated in step k
                )
                care_home_orders.append(order)

            # Step k: Construct final_notice
            arriving_today = []
            for order in care_home_orders:
                for li in order.items:
                    if li.accepted_quantity > 0:
                        arriving_today.append(li.item)

            deferred_names = [li.item for li in deferred]
            commercial_names = [li.item for li in needs_commercial]

            # Build message
            shortfall_parts = []
            if deferred_names:
                shortfall_parts.append(
                    f"{', '.join(deferred_names)} (deferred to a future run)"
                )
            if commercial_names:
                shortfall_parts.append(
                    f"{', '.join(commercial_names)} (requires commercial sourcing)"
                )

            if shortfall_parts:
                notice_message = (
                    f"We have arranged delivery of {', '.join(arriving_today)} for you today. "
                    f"Unfortunately {' and '.join(shortfall_parts)} could not be included this time "
                    f"- we apologise for the inconvenience and will try again soon."
                )
            else:
                notice_message = (
                    f"Great news! We have arranged delivery of {', '.join(arriving_today)} "
                    f"for you today. Everything you accepted is on its way."
                )

            final_notice = {
                "arriving_today": arriving_today,
                "deferred": deferred_names,
                "needs_commercial": commercial_names,
                "message": notice_message,
            }

            # Attach final_notice to each Order for this care home
            for order in care_home_orders:
                order.final_notice = final_notice

            all_orders.extend(care_home_orders)

            # Step l: Collect needs_commercial items
            all_needs_commercial.extend(needs_commercial)

            print(f"  [2] Created {len(care_home_orders)} order(s) for {care_home.name}")

        # ---------------------------------------------------------------------------
        # DETOUR BUNDLING (before dispatch)
        # ---------------------------------------------------------------------------

        print(f"\n{'=' * 70}")
        print(f"  DETOUR BUNDLING")
        print(f"{'=' * 70}")

        # Build store -> care_home -> Store/CareHome lookup maps
        store_map: Dict[str, Store] = {s.store_id: s for s in world.stores}
        care_home_map: Dict[str, CareHome] = {ch.care_home_id: ch for ch in world.care_homes}

        # Group orders by store_id
        orders_by_store: Dict[str, List[Order]] = defaultdict(list)
        for order in all_orders:
            orders_by_store[order.store_id].append(order)

        # For stores with 2+ orders, check detour bundling
        bundled_orders: List[Order] = []
        for store_id, store_orders in orders_by_store.items():
            if len(store_orders) >= 2:
                store = store_map.get(store_id)
                if store is None:
                    bundled_orders.extend(store_orders)
                    continue

                # Check pairs for detour bundling
                # Process in pairs
                i = 0
                while i < len(store_orders):
                    if i + 1 < len(store_orders):
                        order_a = store_orders[i]
                        order_b = store_orders[i + 1]
                        ch_a = care_home_map.get(order_a.care_home_id)
                        ch_b = care_home_map.get(order_b.care_home_id)

                        if ch_a and ch_b:
                            try:
                                should_bundle, extra_minutes = await check_detour_bundle(
                                    store=store,
                                    care_home_a=ch_a,
                                    care_home_b=ch_b,
                                    get_distance_minutes=get_distance_minutes,
                                )
                                if should_bundle:
                                    print(f"  [BUNDLE] Store {store_id}: "
                                          f"{order_a.care_home_id} + {order_b.care_home_id} "
                                          f"bundled (detour +{extra_minutes:.1f} min)")
                                    # Keep both orders - dispatch will batch them together
                                    bundled_orders.append(order_a)
                                    bundled_orders.append(order_b)
                                else:
                                    print(f"  [SPLIT]  Store {store_id}: "
                                          f"{order_a.care_home_id} + {order_b.care_home_id} "
                                          f"split (detour +{extra_minutes:.1f} min > 15 min)")
                                    bundled_orders.append(order_a)
                                    bundled_orders.append(order_b)
                            except Exception as e:
                                logger.warning("Detour check failed for store %s: %s", store_id, e)
                                bundled_orders.append(order_a)
                                bundled_orders.append(order_b)
                            i += 2
                        else:
                            bundled_orders.append(store_orders[i])
                            i += 1
                    else:
                        bundled_orders.append(store_orders[i])
                        i += 1
            else:
                bundled_orders.extend(store_orders)

        # Use bundled_orders for dispatch (same objects, bundling decision is
        # captured by dispatch's _group_orders_by_store and _batch_orders)
        all_orders = bundled_orders

        # ---------------------------------------------------------------------------
        # DISPATCH
        # ---------------------------------------------------------------------------

        print(f"\n{'=' * 70}")
        print(f"  DISPATCH -- {len(all_orders)} orders, "
              f"{len(all_needs_commercial)} needs_commercial items")
        print(f"{'=' * 70}")

        deliveries, dispatch_stats = await run_dispatch(
            orders=all_orders,
            needs_commercial_items=all_needs_commercial,
            world=world,
            sim_day=sim_day,
            get_volunteer_avail=get_volunteer_avail,
            get_distance_minutes=get_distance_minutes,
            get_truck_avail=get_truck_avail,
            run_id=run_id,
        )

        print(f"  [DISPATCH] Complete -- {len(deliveries)} deliveries created")
        print(f"  [DISPATCH] Stats: volunteer={dispatch_stats.volunteer_assigned}, "
              f"truck={dispatch_stats.store_truck_assigned}, "
              f"commercial={dispatch_stats.commercial_assigned}")

        # ---------------------------------------------------------------------------
        # REPORTING
        # ---------------------------------------------------------------------------

        print(f"\n{'=' * 70}")
        print(f"  GENERATING REPORTS")
        print(f"{'=' * 70}")

        report_dict = await generate_full_report(
            deliveries=deliveries,
            orders=all_orders,
            negotiation_results=all_negotiation_results,
            dispatch_stats=dispatch_stats,
            world=world,
            sim_day=sim_day,
            run_id=run_id,
        )

        # ---------------------------------------------------------------------------
        # CONFIRMATION
        # ---------------------------------------------------------------------------

        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"  SIMULATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"  Run ID         : {run_id}")
        print(f"  Total deliveries: {len(deliveries)}")
        print(f"  Map filepath   : {report_dict.get('map_filepath', 'N/A')}")
        print(f"  Total run time : {elapsed:.1f}s")
        print(f"{'=' * 70}\n")

        return report_dict

    finally:
        # Ensure MCP connections are always closed
        if store_toolset is not None:
            try:
                await store_toolset.close()
                print("  [CLEANUP] Store MCP connection closed")
            except Exception as e:
                logger.warning("Failed to close store MCP: %s", e)

        if maps_toolset is not None:
            try:
                await maps_toolset.close()
                print("  [CLEANUP] Maps MCP connection closed")
            except Exception as e:
                logger.warning("Failed to close maps MCP: %s", e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Force UTF-8 on Windows console to support Unicode in report output
    import sys
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(run_simulation())
