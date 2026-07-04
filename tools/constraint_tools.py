"""
tools/constraint_tools.py

Deterministic Python functions — no LLM calls anywhere here.
Three components:
  1. hard_constraint_filter  — strips / caps items per care-home rules
  2. StockLedger             — shared depleting stock across all stores
  3. single_store_candidate  — sourcing assignment logic
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

from data.data_model import (
    CareHome,
    DailyFoodItem,
    FoodCatalogItem,
    SimulationDay,
)
from tools.models import OrderLineItem

# Items that are non-vegetarian — removed for vegetarian-only care homes
_NON_VEGETARIAN_ITEMS: frozenset[str] = frozenset({"chicken", "eggs"})


# ---------------------------------------------------------------------------
# 1. hard_constraint_filter
# ---------------------------------------------------------------------------

def hard_constraint_filter(
    care_home: CareHome,
    offered_items: List[DailyFoodItem],
    catalog: List[FoodCatalogItem],
) -> List[DailyFoodItem]:
    """
    Apply care-home hard constraints and memory notes in this exact order:

    a) Vegetarian filter: if vegetarian_only=True remove chicken and eggs.
       (has_young_children adds priority in Matchmaker — NOT a hard filter here.)
    b) Memory-note exclusions: remove any item whose name matches a note
       with type="exclude".
    c) Memory-note quantity caps: reduce (not remove) quantity for any item
       whose name matches a note with type="max_quantity".

    Returns the filtered/capped list. Original objects are NOT mutated —
    capped items are returned as new DailyFoodItem instances.
    """
    excluded_items: set[str] = {
        note.item.lower()
        for note in care_home.memory_notes
        if note.type == "exclude"
    }
    quantity_caps: Dict[str, float] = {
        note.item.lower(): note.value
        for note in care_home.memory_notes
        if note.type == "max_quantity" and note.value is not None
    }

    result: List[DailyFoodItem] = []
    for item in offered_items:
        name_lower = item.name.lower()

        # Step a: vegetarian filter
        if care_home.hard_constraints.vegetarian_only and name_lower in _NON_VEGETARIAN_ITEMS:
            continue

        # Step b: memory-note exclusion
        if name_lower in excluded_items:
            continue

        # Step c: memory-note quantity cap
        if name_lower in quantity_caps:
            cap = quantity_caps[name_lower]
            if item.quantity > cap:
                item = DailyFoodItem(
                    name=item.name,
                    days_to_expiry=item.days_to_expiry,
                    quantity=cap,
                    unit=item.unit,
                )

        result.append(item)

    return result


# ---------------------------------------------------------------------------
# 2. StockLedger
# ---------------------------------------------------------------------------

class StockLedger:
    """
    Shared depleting stock ledger initialised from pushed_inventory across
    all stores in a SimulationDay.

    Internal state: { (store_id, item_name_lower) -> remaining_quantity }

    Processing order (Home 1 → 2 → 3) is enforced by the CALLER (Matchmaker),
    not inside this class. The ledger simply records what's left.
    """

    def __init__(self, sim_day: SimulationDay) -> None:
        self._stock: Dict[tuple[str, str], float] = {}
        for store_state in sim_day.stores:
            for item in store_state.pushed_inventory:
                key = (store_state.store_id, item.name.lower())
                self._stock[key] = self._stock.get(key, 0.0) + item.quantity

    # -- Public read methods ---------------------------------------------------

    def store_ids(self) -> List[str]:
        """Return all store IDs currently tracked in the ledger."""
        seen: set[str] = set()
        result: List[str] = []
        for (sid, _) in self._stock:
            if sid not in seen:
                seen.add(sid)
                result.append(sid)
        return sorted(result)  # deterministic order

    def get_available(self, store_id: str, item_name: str) -> float:
        """Return remaining quantity for (store_id, item_name). 0.0 if not stocked."""
        return self._stock.get((store_id, item_name.lower()), 0.0)

    def get_store_totals(self, store_id: str) -> Dict[str, float]:
        """Return all remaining items for a store as {item_name: quantity}."""
        return {
            item_name: qty
            for (sid, item_name), qty in self._stock.items()
            if sid == store_id and qty > 0
        }

    def get_cross_store_totals(self) -> Dict[str, float]:
        """
        Return aggregated remaining quantities across ALL stores.
        Used by Matchmaker to build the cross-store offer view.
        """
        totals: Dict[str, float] = defaultdict(float)
        for (_, item_name), qty in self._stock.items():
            if qty > 0:
                totals[item_name] += qty
        return dict(totals)

    # -- Write -----------------------------------------------------------------

    def deduct(self, store_id: str, item_name: str, quantity: float) -> None:
        """
        Deduct quantity from (store_id, item_name).
        Raises ValueError if quantity exceeds what remains.
        """
        key = (store_id, item_name.lower())
        available = self._stock.get(key, 0.0)
        if quantity > available + 1e-9:
            raise ValueError(
                f"Cannot deduct {quantity:.2f} of '{item_name}' from {store_id}: "
                f"only {available:.2f} remaining."
            )
        self._stock[key] = max(0.0, available - quantity)

    # -- Snapshot (for guardrail validation) ----------------------------------

    def snapshot(self) -> Dict[tuple[str, str], float]:
        """Return a shallow copy of the current stock state (tuple keys)."""
        return dict(self._stock)


# ---------------------------------------------------------------------------
# 3. single_store_candidate  (rewritten per Correction 6)
# ---------------------------------------------------------------------------

def single_store_candidate(
    requested_items: List[OrderLineItem],
    urgent_item_names: Set[str],
    ledger: StockLedger,
    catalog: List[FoodCatalogItem],
) -> dict:
    """
    Determine sourcing assignments for a set of requested OrderLineItems.

    Parameters
    ----------
    requested_items   : All items the care home has accepted after negotiation.
    urgent_item_names : Set of item names flagged as urgent by the care home
                        (via flag_urgent action). Only essential items can be
                        urgent; non-essential names are silently ignored.
    ledger            : Current StockLedger state (read-only here — caller deducts).
    catalog           : Full food catalog for is_essential lookups.

    Returns a dict:
    {
        "assignments": List[{"store_id": str, "items": List[OrderLineItem]}],
        "deferred":    List[OrderLineItem],   # non-urgent items that didn't fit
        "needs_commercial": List[OrderLineItem],  # items needing a 4th store
    }

    Logic (strict order):
    1. Try ONE store for all items.  If found → single assignment, done.
    2. Split into urgent-essential vs everything-else (non-urgent).
       Validate urgency: non-essential items are silently dropped from the
       urgent set even if the caller passed them in.
    3. Retry single-store search for urgent items only.
    4. If urgent items still can't fit one store, assign each individually
       from whichever store has stock.  Cap distinct stores at 3.
    5. Items that would require a 4th distinct store → needs_commercial.
    6. For non-urgent items: try to piggyback on stores already selected.
       If a selected store can cover a non-urgent item → add it there.
    7. Remaining non-urgent items → deferred.

    For auto-accepting care homes (negotiates_via_a2a=False):
    caller passes urgent_item_names=set().  Steps 2-5 produce no urgent
    assignments; step 6 tries to piggyback all items on one store.
    """
    catalog_map: Dict[str, FoodCatalogItem] = {
        c.name.lower(): c for c in catalog
    }

    def is_essential(item_name: str) -> bool:
        cat = catalog_map.get(item_name.lower())
        return cat.is_essential if cat else True  # unknown → treat as essential

    # Validate urgency: only essential items can be urgent
    valid_urgent: Set[str] = {
        name for name in urgent_item_names
        if is_essential(name)
    }

    all_store_ids = ledger.store_ids()

    def store_can_cover_all(store_id: str, items: List[OrderLineItem]) -> bool:
        return all(
            ledger.get_available(store_id, it.item) >= it.accepted_quantity
            for it in items
        )

    def best_single_store(items: List[OrderLineItem]) -> Optional[str]:
        if not items:
            return None
        for sid in all_store_ids:
            if store_can_cover_all(sid, items):
                return sid
        return None

    assignments: List[dict] = []
    deferred: List[OrderLineItem] = []
    needs_commercial: List[OrderLineItem] = []
    used_store_ids: List[str] = []          # ordered list of stores already committed
    MAX_STORES = 3

    # ── Step 1: one store for everything ──────────────────────────────────────
    winner = best_single_store(requested_items)
    if winner:
        return {
            "assignments": [{"store_id": winner, "items": requested_items}],
            "deferred": [],
            "needs_commercial": [],
        }

    # ── Step 2: split urgent-essential vs non-urgent ──────────────────────────
    urgent_items = [
        it for it in requested_items
        if it.item.lower() in {n.lower() for n in valid_urgent}
    ]
    non_urgent_items = [
        it for it in requested_items
        if it.item.lower() not in {n.lower() for n in valid_urgent}
    ]

    # ── Step 3: single store for all urgent items ─────────────────────────────
    if urgent_items:
        winner_urgent = best_single_store(urgent_items)
        if winner_urgent:
            assignments.append({"store_id": winner_urgent, "items": list(urgent_items)})
            used_store_ids.append(winner_urgent)
        else:
            # ── Step 4: assign urgent items individually ──────────────────────
            # group by best store so we don't create duplicate assignments
            per_store: Dict[str, List[OrderLineItem]] = defaultdict(list)
            for it in urgent_items:
                best = None
                for sid in all_store_ids:
                    if ledger.get_available(sid, it.item) >= it.accepted_quantity:
                        best = sid
                        break

                if best is None:
                    # No store has this item — goes to needs_commercial
                    needs_commercial.append(it)
                    continue

                # ── Step 5: cap at MAX_STORES distinct stores ─────────────────
                if best not in used_store_ids:
                    if len(used_store_ids) >= MAX_STORES:
                        needs_commercial.append(it)
                        continue
                    used_store_ids.append(best)

                per_store[best].append(it)

            for sid, items in per_store.items():
                assignments.append({"store_id": sid, "items": items})

    # ── Step 6: piggyback non-urgent items on already-selected stores ─────────
    remaining_non_urgent: List[OrderLineItem] = []
    for it in non_urgent_items:
        placed = False
        for sid in used_store_ids:
            if ledger.get_available(sid, it.item) >= it.accepted_quantity:
                # Add to that store's existing assignment
                for asgn in assignments:
                    if asgn["store_id"] == sid:
                        asgn["items"].append(it)
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            # Try any store not yet used (only if we still have room)
            for sid in all_store_ids:
                if sid not in used_store_ids and len(used_store_ids) < MAX_STORES:
                    if ledger.get_available(sid, it.item) >= it.accepted_quantity:
                        used_store_ids.append(sid)
                        assignments.append({"store_id": sid, "items": [it]})
                        placed = True
                        break
        if not placed:
            remaining_non_urgent.append(it)

    # ── Step 7: deferred ──────────────────────────────────────────────────────
    deferred = remaining_non_urgent

    return {
        "assignments": assignments,
        "deferred": deferred,
        "needs_commercial": needs_commercial,
    }
