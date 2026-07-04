"""
tools/guardrails.py

Pydantic guardrail models that validate pipeline outputs before they
are passed to downstream agents.  No LLM calls anywhere here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator, model_validator

from tools.models import Delivery, MatchmakerOffer, Order


# ---------------------------------------------------------------------------
# OrderOutput
# ---------------------------------------------------------------------------

class OrderOutput(BaseModel):
    """
    Wraps a finalized Order and validates:
    - all accepted_quantities are >= 0
    - no accepted_quantity exceeds what the StockLedger had available
      (caller must pass ledger_snapshot using the tuple-keyed form from
       StockLedger.snapshot(): Dict[(store_id, item_name_lower), float],
       or the simplified string-keyed form: Dict[str, float])
    - no item was excluded by hard_constraint_filter
      (caller passes excluded_item_names: List[str])
    """

    order: Order
    # Snapshot keyed by (store_id, item_name_lower) -> qty  [primary]
    # or item_name_lower -> qty  [simplified fallback for tests]
    # Dict[Any, float] supports both tuple and string keys.
    ledger_snapshot: Dict[Any, float] = {}
    # Item names (any case) that were excluded by hard_constraint_filter
    excluded_item_names: List[str] = []

    @field_validator("order")
    @classmethod
    def accepted_quantities_non_negative(cls, order: Order) -> Order:
        for line in order.items:
            if line.accepted_quantity < 0:
                raise ValueError(
                    f"accepted_quantity for '{line.item}' is negative: "
                    f"{line.accepted_quantity}"
                )
        return order

    @model_validator(mode="after")
    def accepted_quantities_within_ledger(self) -> "OrderOutput":
        """
        Check each accepted_quantity against the ledger snapshot.
        Snapshot keys can be either:
          (store_id, item_name_lower)  ->  float   (from StockLedger.snapshot())
          item_name_lower              ->  float   (simplified form for tests)
        """
        snapshot = self.ledger_snapshot
        if not snapshot:
            return self  # no snapshot provided — skip check

        store_id = self.order.store_id
        for line in self.order.items:
            name_lower = line.item.lower()
            # Try tuple key first (full ledger snapshot)
            available = snapshot.get((store_id, name_lower))
            if available is None:
                # Fallback to simple item-name key
                available = snapshot.get(name_lower)
            if available is None:
                continue  # item not in snapshot — skip
            if line.accepted_quantity > available + 1e-9:
                raise ValueError(
                    f"accepted_quantity {line.accepted_quantity:.2f} for "
                    f"'{line.item}' exceeds ledger available {available:.2f}"
                )
        return self

    @model_validator(mode="after")
    def no_excluded_items_present(self) -> "OrderOutput":
        excluded = {name.lower() for name in self.excluded_item_names}
        if not excluded:
            return self
        for line in self.order.items:
            if line.item.lower() in excluded:
                raise ValueError(
                    f"Item '{line.item}' is present in the order but was "
                    f"excluded by hard_constraint_filter."
                )
        return self


# ---------------------------------------------------------------------------
# DispatchOutput
# ---------------------------------------------------------------------------

_MAX_DELIVERY_MINUTES = 120.0
_MAX_ORDER_IDS = 2

# Sentinel capacity value for commercial pickups (no physical constraint)
COMMERCIAL_CAPACITY_SENTINEL = 99999.0


class DispatchOutput(BaseModel):
    """
    Wraps a Delivery with computed logistics metadata and validates:
    - total_payload_kg <= vehicle_capacity_kg  (required, no default)
    - estimated_time_minutes <= 120 (2-hour budget)
    - order_ids has at most 2 entries  (also enforced on Delivery itself)

    vehicle_capacity_kg is a REQUIRED field — no default value.
    Callers must pass the appropriate capacity:
      - Volunteer:   Volunteer.capacity_kg from WorldConfig
      - Store truck: Store.truck_capacity_kg from WorldConfig
      - Commercial:  Pass COMMERCIAL_CAPACITY_SENTINEL (99999.0) — no limit
    """

    delivery: Delivery
    total_payload_kg: float
    estimated_time_minutes: float
    vehicle_capacity_kg: float  # required — no default

    @field_validator("total_payload_kg")
    @classmethod
    def payload_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"total_payload_kg must be >= 0, got {v}")
        return v

    @field_validator("estimated_time_minutes")
    @classmethod
    def time_within_budget(cls, v: float) -> float:
        if v > _MAX_DELIVERY_MINUTES:
            raise ValueError(
                f"estimated_time_minutes {v:.1f} exceeds 2-hour budget "
                f"({_MAX_DELIVERY_MINUTES} min)"
            )
        return v

    @model_validator(mode="after")
    def payload_within_capacity(self) -> "DispatchOutput":
        if self.total_payload_kg > self.vehicle_capacity_kg + 1e-9:
            raise ValueError(
                f"total_payload_kg {self.total_payload_kg:.2f} kg exceeds "
                f"vehicle capacity {self.vehicle_capacity_kg:.2f} kg"
            )
        return self

    @model_validator(mode="after")
    def max_two_orders(self) -> "DispatchOutput":
        if len(self.delivery.order_ids) > _MAX_ORDER_IDS:
            raise ValueError(
                f"A Delivery may carry at most {_MAX_ORDER_IDS} orders; "
                f"got {len(self.delivery.order_ids)}: {self.delivery.order_ids}"
            )
        return self


# ---------------------------------------------------------------------------
# MatchmakerOfferGuardrail
# ---------------------------------------------------------------------------

class MatchmakerOfferGuardrail(BaseModel):
    """
    Validates a MatchmakerOffer before it is sent to negotiation.
    - all offered_quantities > 0
    - no offered_quantity exceeds cross-store aggregate availability
    - no item present that was removed by hard_constraint_filter
    """

    offer: MatchmakerOffer
    # Cross-store aggregate snapshot at time of offer: {item_name_lower: qty}
    cross_store_snapshot: Dict[str, float] = {}
    # Item names excluded by hard_constraint_filter for this care home
    excluded_item_names: List[str] = []

    @field_validator("offer")
    @classmethod
    def all_quantities_positive(cls, offer: MatchmakerOffer) -> MatchmakerOffer:
        for item in offer.offered_items:
            if item.offered_quantity <= 0:
                raise ValueError(
                    f"offered_quantity for '{item.item}' must be > 0, "
                    f"got {item.offered_quantity}"
                )
        return offer

    @model_validator(mode="after")
    def quantities_within_cross_store(self) -> "MatchmakerOfferGuardrail":
        snapshot = self.cross_store_snapshot
        if not snapshot:
            return self
        for item in self.offer.offered_items:
            available = snapshot.get(item.item.lower())
            if available is not None and item.offered_quantity > available + 1e-9:
                raise ValueError(
                    f"offered_quantity {item.offered_quantity:.2f} for "
                    f"'{item.item}' exceeds cross-store available "
                    f"{available:.2f}"
                )
        return self

    @model_validator(mode="after")
    def no_excluded_items_present(self) -> "MatchmakerOfferGuardrail":
        excluded = {name.lower() for name in self.excluded_item_names}
        if not excluded:
            return self
        for item in self.offer.offered_items:
            if item.item.lower() in excluded:
                raise ValueError(
                    f"Item '{item.item}' is present in the offer but was "
                    f"excluded by hard_constraint_filter."
                )
        return self
