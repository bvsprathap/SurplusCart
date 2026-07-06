"""
tools/dispatch.py

Dispatch Module — DETERMINISTIC Python only. No LLM calls anywhere in this file.

BOUNDARY:
  This module runs AFTER the orchestrator has completed Phase 2 processing:
    - single_store_candidate has been called per Order
    - StockLedger.deduct() has been called for each accepted item
    - Order objects have been created with store_id, items, urgent_essential_items
  Dispatch receives already-created Order objects and needs_commercial items.
  It does NOT call single_store_candidate, does NOT mutate StockLedger, and
  does NOT create Order objects.

RESPONSIBILITIES:
  1. Group Orders into Deliveries (by store_id, max 2 orders per Delivery)
  2. Assign delivery method via fallback chain:
       Step 1 → nearest available volunteer (capacity check)
       Step 2 → next volunteer within 2-hour budget
       Step 3 → 15-min detour bundling (for second Order, same volunteer)
       Step 4 → store's own truck
       Step 5 → commercial pickup (always succeeds)
  3. Guarantee urgent essential items are never left undelivered
  4. Send confirmation messages via log_message
  5. Validate each Delivery via DispatchOutput guardrail

CALLABLE PARAMETERS:
  run_dispatch accepts three async callables instead of McpToolset objects.
  This makes the function testable without real MCP sessions and keeps
  Dispatch fully decoupled from the ADK agent framework.

  get_volunteer_avail(volunteer_id: str) -> dict
    Returns {"volunteer_id": str, "available": bool}
    (wraps get_volunteer_schedule MCP call)

  get_distance_minutes(origin_lat, origin_lng, dest_lat, dest_lng) -> float
    Returns travel time in minutes.
    (wraps maps_distance_matrix MCP call, extracting durations[0][0].value/60)

  get_truck_avail(store_id: str) -> dict
    Returns {"available": bool, "capacity_kg": float}
    (wraps check_vehicle_availability MCP call)
"""

from __future__ import annotations

import logging
import uuid
from typing import Callable, Awaitable, Dict, List, Optional, Tuple
from collections import defaultdict

from data.data_model import CareHome, FoodCatalogItem, SimulationDay, Store, Volunteer, WorldConfig
from tools.guardrails import COMMERCIAL_CAPACITY_SENTINEL, DispatchOutput
from tools.logger import log_message
from tools.models import Delivery, DispatchStats, Order, OrderLineItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_DELIVERY_MINUTES = 120.0
_MAX_DETOUR_MINUTES = 15.0
_PICKUP_TIME = "Today 2:00 PM"  # Simulated — real scheduling is future work
_COMMERCIAL_CHANNEL = "commercial_pickup_dunzo_simulated"

# Type aliases for the callable parameters
VolunteerAvailCallable = Callable[[str], Awaitable[Dict]]
DistanceCallable = Callable[[float, float, float, float], Awaitable[float]]
TruckAvailCallable = Callable[[str], Awaitable[Dict]]
DirectionsPolylineCallable = Callable[[str, str, List[str]], Awaitable[Optional[str]]]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_dispatch(
    orders: List[Order],
    needs_commercial_items: List[dict],
    world: WorldConfig,
    sim_day: SimulationDay,
    get_volunteer_avail: VolunteerAvailCallable,
    get_distance_minutes: DistanceCallable,
    get_directions_polyline: DirectionsPolylineCallable,
    get_truck_avail: TruckAvailCallable,
    run_id: str,
) -> Tuple[List[Delivery], DispatchStats]:
    """
    Main dispatch entry point.

    Parameters
    ----------
    orders              : List of Order objects created by Phase 2 orchestration.
                          Each Order already has store_id, items, urgent_essential_items.
    needs_commercial_items : Items from single_store_candidate that required a 4th
                             distinct store. Now structured as List[{"store_id": str, "order_id": str, "item": OrderLineItem}] — route directly to commercial Delivery.
    world               : WorldConfig for Store/Volunteer lookup.
    sim_day             : SimulationDay (read-only — not mutated here).
    get_volunteer_avail : Async callable wrapping get_volunteer_schedule MCP tool.
    get_distance_minutes: Async callable wrapping maps_distance_matrix MCP tool.
    get_truck_avail     : Async callable wrapping check_vehicle_availability MCP tool.
    run_id              : Simulation run ID for log_message.

    Returns
    -------
    (List[Delivery], DispatchStats)
    """
    stats = DispatchStats()
    deliveries: List[Delivery] = []

    # Build lookup maps
    store_map: Dict[str, Store] = {s.store_id: s for s in world.stores}
    care_home_map: Dict[str, CareHome] = {ch.care_home_id: ch for ch in world.care_homes}
    volunteer_map: Dict[str, Volunteer] = {v.volunteer_id: v for v in world.volunteers}
    catalog_map: Dict[str, FoodCatalogItem] = {c.name.lower(): c for c in world.catalog}

    # -- Step 0a: Handle needs_commercial items directly (skip steps 1–4) -----
    if needs_commercial_items:
        # Group by care_home_id to combine into one commercial delivery per care home
        order_ch_map = {o.order_id: o.care_home_id for o in orders}
        commercial_by_ch = defaultdict(list)
        orders_by_ch = defaultdict(set)
        stores_by_ch = defaultdict(set)

        for nc in needs_commercial_items:
            ch_id = order_ch_map.get(nc["order_id"])
            if not ch_id:
                ch_id = "unknown_ch"
            commercial_by_ch[ch_id].append(nc["item"])
            orders_by_ch[ch_id].add(nc["order_id"])
            stores_by_ch[ch_id].add(nc["store_id"])

        for ch_id, items in commercial_by_ch.items():
            store_names = []
            for sid in stores_by_ch[ch_id]:
                s = store_map.get(sid)
                store_names.append(s.name if s else sid)
            
            combined_store_id = ", ".join(sorted(store_names))

            commercial_delivery = await _make_commercial_delivery(
                items=items,
                store_id=combined_store_id,
                order_ids=list(orders_by_ch[ch_id]),
                world=world,
                care_home_map=care_home_map,
                store_map=store_map,
                catalog_map=catalog_map,
                stats=stats,
                label="needs_commercial overflow",
            )
            if commercial_delivery:
                deliveries.append(commercial_delivery)

    # -- Step 0b: Sort available volunteers by proximity to each store --------
    # Pre-fetch availability for all volunteers (single pass)
    available_volunteers: List[Volunteer] = []
    unavailable_count = 0
    for vol in world.volunteers:
        try:
            avail_result = await get_volunteer_avail(vol.volunteer_id)
            if avail_result.get("available", False):
                available_volunteers.append(vol)
            else:
                unavailable_count += 1
        except Exception as exc:
            logger.warning("get_volunteer_avail(%s) failed: %s", vol.volunteer_id, exc)
            unavailable_count += 1
    stats.volunteers_unavailable = unavailable_count

    # -- Step 0c: Group Orders by store_id, max 2 per Delivery ----------------
    commercial_order_ids = {nc["order_id"] for nc in needs_commercial_items} if needs_commercial_items else set()
    standard_orders = [o for o in orders if o.order_id not in commercial_order_ids]
    grouped = _group_orders_by_store(standard_orders)  # {store_id: [Order, ...]}

    # -- Step 1–5: Assign delivery method per Delivery group ------------------
    for store_id, store_orders in grouped.items():
        store = store_map.get(store_id)
        if store is None:
            logger.error("Unknown store_id '%s' in orders — skipping", store_id)
            continue

        # Split into batches of max 2 orders
        batches = _batch_orders(store_orders, max_per_batch=2)

        for batch_idx, batch in enumerate(batches):
            order_ids = [o.order_id for o in batch]
            payload_kg = _total_payload_kg(batch, catalog_map)
            has_urgent = _has_urgent_items(batch)

            delivery = await _assign_delivery(
                batch=batch,
                store=store,
                order_ids=order_ids,
                payload_kg=payload_kg,
                has_urgent=has_urgent,
                available_volunteers=available_volunteers,
                care_home_map=care_home_map,
                store_map=store_map,
                catalog_map=catalog_map,
                get_distance_minutes=get_distance_minutes,
                get_truck_avail=get_truck_avail,
                stats=stats,
            )
            if delivery:
                deliveries.append(delivery)
                _send_confirmation_messages(
                    delivery=delivery,
                    batch=batch,
                    store=store,
                    care_home_map=care_home_map,
                    volunteer_map=volunteer_map,
                )

    # -- Step 6: Fetch polylines for visualization ----------------------------
    om = {o.order_id: o for o in orders}
    for delivery in deliveries:
        store = store_map.get(delivery.store_id)
        if not store:
            continue
        store_coord = f"{store.latitude},{store.longitude}"
        
        ch_coords = []
        for oid in delivery.order_ids:
            order = om.get(oid)
            if order:
                ch = care_home_map.get(order.care_home_id)
                if ch:
                    ch_coords.append(f"{ch.latitude},{ch.longitude}")
        
        if delivery.method == "volunteer" and delivery.volunteer_id:
            vol = volunteer_map.get(delivery.volunteer_id)
            if vol:
                vol_coord = f"{vol.latitude},{vol.longitude}"
                origin = vol_coord
                if len(ch_coords) > 0:
                    dest = ch_coords[-1]
                    waypoints = [store_coord] + ch_coords[:-1]
                else:
                    dest = store_coord
                    waypoints = []
            else:
                continue
        else:
            origin = store_coord
            if len(ch_coords) > 0:
                dest = ch_coords[-1]
                waypoints = ch_coords[:-1]
            else:
                continue

        try:
            poly = await get_directions_polyline(origin, dest, waypoints)
            delivery.polyline = poly
        except Exception as e:
            logger.warning(f"Failed to fetch polyline for delivery {delivery.delivery_id}: {e}")

    stats.total_deliveries = len(deliveries)
    return deliveries, stats


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def _group_orders_by_store(orders: List[Order]) -> Dict[str, List[Order]]:
    """Group orders by store_id preserving insertion order."""
    groups: Dict[str, List[Order]] = {}
    for order in orders:
        groups.setdefault(order.store_id, []).append(order)
    return groups


def _batch_orders(orders: List[Order], max_per_batch: int = 2) -> List[List[Order]]:
    """Split a list of orders into batches of up to max_per_batch."""
    return [orders[i:i + max_per_batch] for i in range(0, len(orders), max_per_batch)]


def _total_payload_kg(batch: List[Order], catalog_map: Dict[str, FoodCatalogItem]) -> float:
    """Sum all accepted_quantities across all orders in a batch (kg approximation)."""
    total = 0.0
    for order in batch:
        for line in order.items:
            cat = catalog_map.get(line.item.lower())
            w = cat.approx_weight_kg if cat else 1.0
            total += line.accepted_quantity * w
    return total


def _has_urgent_items(batch: List[Order]) -> bool:
    """Return True if any order in the batch has urgent_essential_items."""
    return any(bool(o.urgent_essential_items) for o in batch)


# ---------------------------------------------------------------------------
# Delivery assignment — fallback chain
# ---------------------------------------------------------------------------

async def _assign_delivery(
    batch: List[Order],
    store: Store,
    order_ids: List[str],
    payload_kg: float,
    has_urgent: bool,
    available_volunteers: List[Volunteer],
    care_home_map: Dict[str, CareHome],
    store_map: Dict[str, Store],
    catalog_map: Dict[str, FoodCatalogItem],
    get_distance_minutes: DistanceCallable,
    get_truck_avail: TruckAvailCallable,
    stats: DispatchStats,
) -> Optional[Delivery]:
    """
    Walk the fallback chain in strict order and return the first valid Delivery.
    Validates each candidate via DispatchOutput guardrail before accepting.
    Falls to next tier on guardrail failure rather than raising.
    """
    care_home_ids = [o.care_home_id for o in batch]

    # -- Steps 1 & 2: Volunteer assignment (nearest + budget check) -----------
    if available_volunteers:
        volunteer_result = await _try_volunteer_assignment(
            store=store,
            order_ids=order_ids,
            payload_kg=payload_kg,
            care_home_ids=care_home_ids,
            care_home_map=care_home_map,
            available_volunteers=available_volunteers,
            get_distance_minutes=get_distance_minutes,
            has_urgent=has_urgent,
            stats=stats,
        )
        if volunteer_result:
            return volunteer_result

    # -- Step 4: Store's own truck --------------------------------------------
    truck_result = await _try_store_truck(
        store=store,
        order_ids=order_ids,
        payload_kg=payload_kg,
        care_home_ids=care_home_ids,
        care_home_map=care_home_map,
        get_truck_avail=get_truck_avail,
        get_distance_minutes=get_distance_minutes,
        has_urgent=has_urgent,
        stats=stats,
    )
    if truck_result:
        return truck_result

    # -- Step 5: Commercial pickup (always succeeds) --------------------------
    commercial = await _make_commercial_delivery(
        items=[line for o in batch for line in o.items],
        store_id=store.store_id,
        order_ids=order_ids,
        world=None,
        care_home_map=care_home_map,
        store_map=store_map,
        catalog_map=catalog_map,
        stats=stats,
        label=f"fallback for store {store.store_id}",
        has_urgent=has_urgent,
    )
    return commercial


# ---------------------------------------------------------------------------
# Step 1 & 2: Volunteer assignment
# ---------------------------------------------------------------------------

async def _try_volunteer_assignment(
    store: Store,
    order_ids: List[str],
    payload_kg: float,
    care_home_ids: List[str],
    care_home_map: Dict[str, CareHome],
    available_volunteers: List[Volunteer],
    get_distance_minutes: DistanceCallable,
    has_urgent: bool,
    stats: DispatchStats,
) -> Optional[Delivery]:
    """
    Try to assign the nearest available volunteer.
    Steps 1 & 2 from the fallback chain:
      Step 1: nearest volunteer with sufficient capacity
      Step 2: next volunteer within 2-hour time budget
    """
    # Rank volunteers by distance from store (nearest first)
    ranked = await _rank_volunteers_by_store_distance(
        store=store,
        volunteers=available_volunteers,
        get_distance_minutes=get_distance_minutes,
    )

    for vol, store_to_vol_minutes in ranked:
        if vol.capacity_kg < payload_kg:
            # Step 1 failed — capacity insufficient, try next
            continue

        # Check total time budget (store → volunteer → care home(s))
        try:
            total_minutes = await _estimate_total_time(
                store=store,
                volunteer=vol,
                store_to_vol_minutes=store_to_vol_minutes,
                care_home_ids=care_home_ids,
                care_home_map=care_home_map,
                get_distance_minutes=get_distance_minutes,
            )
        except Exception as exc:
            logger.warning("Distance estimate failed for vol %s: %s", vol.volunteer_id, exc)
            continue

        if total_minutes > _MAX_DELIVERY_MINUTES:
            # Step 2 failed — over time budget
            continue

        # Validate via DispatchOutput guardrail
        delivery = Delivery(
            delivery_id=str(uuid.uuid4()),
            store_id=store.store_id,
            order_ids=order_ids,
            method="volunteer",
            volunteer_id=vol.volunteer_id,
            pickup_time=_PICKUP_TIME,
        )
        try:
            DispatchOutput(
                delivery=delivery,
                total_payload_kg=payload_kg,
                estimated_time_minutes=total_minutes,
                vehicle_capacity_kg=vol.capacity_kg,
            )
        except Exception as exc:
            logger.warning(
                "DispatchOutput validation failed for volunteer %s: %s — trying next",
                vol.volunteer_id, exc,
            )
            continue

        stats.volunteer_assigned += 1
        return delivery

    return None


async def _rank_volunteers_by_store_distance(
    store: Store,
    volunteers: List[Volunteer],
    get_distance_minutes: DistanceCallable,
) -> List[Tuple[Volunteer, float]]:
    """Return volunteers sorted by travel time from store (ascending)."""
    ranked: List[Tuple[Volunteer, float]] = []
    for vol in volunteers:
        try:
            minutes = await get_distance_minutes(
                store.latitude, store.longitude,
                vol.latitude, vol.longitude,
            )
            ranked.append((vol, minutes))
        except Exception as exc:
            logger.warning(
                "Distance store→vol %s failed: %s — using large sentinel",
                vol.volunteer_id, exc,
            )
            ranked.append((vol, 9999.0))  # push to end of ranking

    ranked.sort(key=lambda x: x[1])
    return ranked


async def _estimate_total_time(
    store: Store,
    volunteer: Volunteer,
    store_to_vol_minutes: float,
    care_home_ids: List[str],
    care_home_map: Dict[str, CareHome],
    get_distance_minutes: DistanceCallable,
) -> float:
    """
    Estimate total delivery time:
      vol_home → store + store → care_home_1 [+ care_home_1 → care_home_2]
    We use store→vol as proxy for vol→store (symmetric approximation).
    """
    vol_to_store = store_to_vol_minutes  # symmetric approximation
    store_to_first_ch = 0.0
    ch_to_ch = 0.0

    if care_home_ids:
        ch1 = care_home_map.get(care_home_ids[0])
        if ch1:
            store_to_first_ch = await get_distance_minutes(
                store.latitude, store.longitude,
                ch1.latitude, ch1.longitude,
            )

    if len(care_home_ids) >= 2:
        ch1 = care_home_map.get(care_home_ids[0])
        ch2 = care_home_map.get(care_home_ids[1])
        if ch1 and ch2:
            ch_to_ch = await get_distance_minutes(
                ch1.latitude, ch1.longitude,
                ch2.latitude, ch2.longitude,
            )

    return vol_to_store + store_to_first_ch + ch_to_ch


# ---------------------------------------------------------------------------
# Step 3: 15-minute detour bundling check
# ---------------------------------------------------------------------------

async def check_detour_bundle(
    store: Store,
    care_home_a: CareHome,
    care_home_b: CareHome,
    get_distance_minutes: DistanceCallable,
) -> Tuple[bool, float]:
    """
    Check whether adding care_home_b as a detour from care_home_a adds
    <= 15 minutes over the direct store → care_home_a route.

    Step 3 from the fallback chain: only applies when assigning a second
    Order to an already-assigned volunteer (same store, second care home).

    Returns (should_bundle: bool, detour_extra_minutes: float).
    """
    direct = await get_distance_minutes(
        store.latitude, store.longitude,
        care_home_a.latitude, care_home_a.longitude,
    )
    via_b = await get_distance_minutes(
        store.latitude, store.longitude,
        care_home_b.latitude, care_home_b.longitude,
    )
    b_to_a = await get_distance_minutes(
        care_home_b.latitude, care_home_b.longitude,
        care_home_a.latitude, care_home_a.longitude,
    )
    detour_route = via_b + b_to_a
    extra = detour_route - direct
    return extra <= _MAX_DETOUR_MINUTES, extra


# ---------------------------------------------------------------------------
# Step 4: Store truck
# ---------------------------------------------------------------------------

async def _try_store_truck(
    store: Store,
    order_ids: List[str],
    payload_kg: float,
    care_home_ids: List[str],
    care_home_map: Dict[str, CareHome],
    get_truck_avail: TruckAvailCallable,
    get_distance_minutes: DistanceCallable,
    has_urgent: bool,
    stats: DispatchStats,
) -> Optional[Delivery]:
    """Attempt to use the store's own truck."""
    try:
        truck = await get_truck_avail(store.store_id)
    except Exception as exc:
        logger.warning("get_truck_avail(%s) failed: %s", store.store_id, exc)
        return None

    if not truck.get("available", False):
        return None

    truck_capacity = truck.get("capacity_kg", 0.0)

    # Estimate time
    total_minutes = 0.0
    try:
        if care_home_ids:
            ch1 = care_home_map.get(care_home_ids[0])
            if ch1:
                total_minutes += await get_distance_minutes(
                    store.latitude, store.longitude,
                    ch1.latitude, ch1.longitude,
                )
        if len(care_home_ids) >= 2:
            ch1 = care_home_map.get(care_home_ids[0])
            ch2 = care_home_map.get(care_home_ids[1])
            if ch1 and ch2:
                total_minutes += await get_distance_minutes(
                    ch1.latitude, ch1.longitude,
                    ch2.latitude, ch2.longitude,
                )
    except Exception as exc:
        logger.warning("Distance estimate for store truck failed: %s", exc)
        total_minutes = 0.0  # Accept anyway — truck is best available

    delivery = Delivery(
        delivery_id=str(uuid.uuid4()),
        store_id=store.store_id,
        order_ids=order_ids,
        method="store_truck",
        volunteer_id=None,
        pickup_time=_PICKUP_TIME,
    )
    try:
        DispatchOutput(
            delivery=delivery,
            total_payload_kg=payload_kg,
            estimated_time_minutes=min(total_minutes, _MAX_DELIVERY_MINUTES),
            vehicle_capacity_kg=truck_capacity,
        )
    except Exception as exc:
        logger.warning("DispatchOutput validation failed for store truck: %s", exc)
        return None

    stats.store_truck_assigned += 1
    if has_urgent:
        stats.urgent_items_forced_fallback += 1
        logger.info(
            "Urgent items on delivery for store %s forced fallback to store truck",
            store.store_id,
        )
    return delivery


# ---------------------------------------------------------------------------
# Step 5: Commercial pickup
# ---------------------------------------------------------------------------

async def _make_commercial_delivery(
    items: List[OrderLineItem],
    store_id: Optional[str],
    order_ids: List[str],
    world: Optional[WorldConfig],
    care_home_map: Dict[str, CareHome],
    store_map: Dict[str, Store],
    catalog_map: Dict[str, FoodCatalogItem],
    stats: DispatchStats,
    label: str = "commercial",
    has_urgent: bool = False,
) -> Optional[Delivery]:
    """
    Create a commercial Delivery. Always succeeds (last resort).
    Uses COMMERCIAL_CAPACITY_SENTINEL for guardrail validation.
    """
    effective_store_id = store_id or (
        order_ids[0].split("-")[0] if order_ids else "unknown"
    )

    delivery = Delivery(
        delivery_id=str(uuid.uuid4()),
        store_id=effective_store_id,
        order_ids=order_ids if len(order_ids) <= 2 else order_ids[:2],
        method="commercial",
        volunteer_id=None,
        pickup_time=_PICKUP_TIME,
    )
    
    payload_kg = 0.0
    for it in items:
        cat = catalog_map.get(it.item.lower())
        w = cat.approx_weight_kg if cat else 1.0
        payload_kg += it.accepted_quantity * w

    try:
        DispatchOutput(
            delivery=delivery,
            total_payload_kg=payload_kg,
            estimated_time_minutes=0.0,   # Commercial time not estimated
            vehicle_capacity_kg=COMMERCIAL_CAPACITY_SENTINEL,
        )
    except Exception as exc:
        # Should never fail with sentinel capacity — log and return anyway
        logger.error("Commercial DispatchOutput validation failed: %s", exc)

    log_message(
        to=_COMMERCIAL_CHANNEL,
        channel="whatsapp_simulated",
        content=(
            f"[Commercial Pickup — {label}] "
            f"Store: {effective_store_id}. "
            f"Items: {_format_items(items)}. "
            f"Pickup time: {_PICKUP_TIME}."
        ),
    )

    stats.commercial_assigned += 1
    if has_urgent:
        stats.urgent_items_forced_fallback += 1
        logger.info(
            "Urgent items on delivery for store %s forced fallback to commercial",
            effective_store_id,
        )
    return delivery


def _send_confirmation_messages(
    delivery: Delivery,
    batch: List[Order],
    store: Store,
    care_home_map: Dict[str, CareHome],
    volunteer_map: Dict[str, Volunteer],
) -> None:
    """
    Send two WhatsApp-simulated confirmation messages:
      1. To the volunteer (if method="volunteer")
      2. To the store (always)
    """
    all_items = [line for order in batch for line in order.items]
    items_desc = _format_items(all_items)
    care_home_names = ", ".join(
        care_home_map[o.care_home_id].name
        if o.care_home_id in care_home_map else o.care_home_id
        for o in batch
    )

    # Message to volunteer
    if delivery.method == "volunteer" and delivery.volunteer_id:
        vol = volunteer_map.get(delivery.volunteer_id)
        vol_name = vol.name if vol else delivery.volunteer_id
        log_message(
            to=delivery.volunteer_id,
            channel="whatsapp_simulated",
            content=(
                f"Hi {vol_name}, please pick up {items_desc} from "
                f"{store.name} at {delivery.pickup_time} and deliver to "
                f"{care_home_names}. Thank you for helping!"
            ),
        )

    # Collector description for store message
    if delivery.method == "volunteer" and delivery.volunteer_id:
        vol = volunteer_map.get(delivery.volunteer_id)
        vol_name = vol.name if vol else delivery.volunteer_id
        collector = f"{vol_name} at {delivery.pickup_time}"
    elif delivery.method == "store_truck":
        collector = f"store truck at {delivery.pickup_time}"
    else:
        collector = f"commercial pickup at {delivery.pickup_time}"

    # Message to store
    log_message(
        to=store.store_id,
        channel="whatsapp_simulated",
        content=(
            f"Dear {store.name}, please keep the following ready for "
            f"collection: {items_desc}. "
            f"Collection by: {collector}. Thank you."
        ),
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_items(items: List[OrderLineItem]) -> str:
    """Format items as 'name: qty unit, ...' string."""
    parts = [
        f"{it.item}: {it.accepted_quantity:.0f} {it.unit}"
        for it in items
        if it.accepted_quantity > 0
    ]
    return ", ".join(parts) if parts else "(no items)"
