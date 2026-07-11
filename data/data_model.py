import os
import json
import uuid
import random
from typing import List, Optional
from pydantic import BaseModel


# --- STATIC CONFIG MODELS ---

class FoodCatalogItem(BaseModel):
    name: str
    is_perishable: bool
    is_essential: bool
    push_threshold_days: int
    unit: str
    approx_weight_kg: float
    cap_category: str

class Store(BaseModel):
    store_id: str
    name: str
    latitude: float
    longitude: float
    has_own_truck: bool = True
    truck_capacity_kg: float = 0.0

class MemoryNote(BaseModel):
    item: str
    type: str  # "max_quantity" | "exclude"
    value: Optional[float] = None

class HardConstraints(BaseModel):
    vegetarian_only: bool
    has_young_children: bool

class CareHome(BaseModel):
    care_home_id: str
    name: str
    latitude: float
    longitude: float
    hard_constraints: HardConstraints
    resident_count: int
    storage_capacity_kg: float
    negotiates_via_a2a: bool = True
    memory_notes: List[MemoryNote] = []

class Volunteer(BaseModel):
    volunteer_id: str
    name: str
    latitude: float
    longitude: float
    vehicle_type: str  # "two_wheeler" | "car"
    capacity_kg: float

class WorldConfig(BaseModel):
    availability_rate: float = 0.7
    catalog: List[FoodCatalogItem]
    stores: List[Store]
    care_homes: List[CareHome]
    volunteers: List[Volunteer]

# --- DAILY STATE MODELS ---

class DailyFoodItem(BaseModel):
    name: str
    days_to_expiry: int
    quantity: float
    unit: str

class TruckStatus(BaseModel):
    available: bool
    capacity_kg: float

class StoreDailyState(BaseModel):
    store_id: str
    full_inventory: List[DailyFoodItem]
    pushed_inventory: List[DailyFoodItem]
    truck_status: Optional[TruckStatus] = None

class VolunteerDailyState(BaseModel):
    volunteer_id: str
    available: bool

class SimulationDay(BaseModel):
    run_id: str
    stores: List[StoreDailyState]
    volunteers: List[VolunteerDailyState]

# --- WORKING STATE MODELS (Not persisted) ---

class StockLedgerEntry(BaseModel):
    store_id: str
    item: str
    remaining_quantity: float


# --- INVENTORY FILTER LOGIC ---

def get_pushable_inventory(daily_item: DailyFoodItem, catalog_item: FoodCatalogItem) -> bool:
    """
    Returns True if the daily_item crosses its category's push threshold.
    """
    return daily_item.days_to_expiry <= catalog_item.push_threshold_days


# --- WORLD CONFIG LOADING ---

def setup_world(config_filepath: str = "world_config.json", catalog_filepath: str = "catalog.json") -> WorldConfig:
    """
    Loads catalog.json and world_config.json, returning the combined WorldConfig.
    Raises FileNotFoundError if either file is missing.
    No generation logic remains here.
    """
    if not os.path.exists(catalog_filepath):
        raise FileNotFoundError(f"Required catalog file is missing: {catalog_filepath}")
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"Required world config file is missing: {config_filepath}")

    # Load and deserialize catalog.json
    with open(catalog_filepath, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)
        catalog = [FoodCatalogItem.model_validate(item) for item in catalog_data]

    # Load and deserialize world_config.json
    with open(config_filepath, "r", encoding="utf-8") as f:
        world_data = json.load(f)
        availability_rate = world_data.get("availability_rate", 0.7)
        stores = [Store.model_validate(s) for s in world_data.get("stores", [])]
        care_homes = [CareHome.model_validate(ch) for ch in world_data.get("care_homes", [])]
        volunteers = [Volunteer.model_validate(v) for v in world_data.get("volunteers", [])]

    return WorldConfig(
        availability_rate=availability_rate,
        catalog=catalog,
        stores=stores,
        care_homes=care_homes,
        volunteers=volunteers
    )


# --- DAILY SIMULATION DATA GENERATION ---

def generate_daily_data(world: WorldConfig) -> SimulationDay:
    """
    Generates a SimulationDay based on the static world configuration.
    For each store, generates full_inventory by iterating over all catalog items.
    Derives pushed_inventory by filtering through get_pushable_inventory.
    Randomizes volunteer availability.
    """
    run_id = str(uuid.uuid4())
    
    # Store daily inventory generation
    stores_daily = []
    for store in world.stores:
        full_inventory = []
        pushed_inventory = []
        
        for catalog_item in world.catalog:
            # Generate quantity
            cat = catalog_item.cap_category
            if cat in ("pulses", "poultry", "dairy"):
                qty = random.uniform(0, 50)
            elif cat == "small_veg":
                qty = random.uniform(0, 5)
            elif cat == "poultry_eggs":
                qty = random.uniform(0, 10)
            elif cat == "spices":
                qty = random.uniform(0, 2)
            elif cat == "staples":
                qty = random.uniform(0, 30)
            elif cat == "bread":
                kg_qty = random.uniform(0, 20)
                qty = kg_qty / catalog_item.approx_weight_kg
            elif cat == "vegetables":
                kg_qty = random.uniform(0, 15)
                qty = kg_qty / catalog_item.approx_weight_kg
            else:
                qty = random.uniform(0, 50)
                
            if catalog_item.unit.lower() not in ("kg", "liter", "liters"):
                quantity = float(round(qty))
            else:
                quantity = float(round(qty))
                
            # Generate days to expiry
            if catalog_item.is_perishable:
                days_to_expiry = random.randint(1, 3)
            else:
                days_to_expiry = random.randint(4, 14)

            daily_item = DailyFoodItem(
                name=catalog_item.name,
                days_to_expiry=days_to_expiry,
                quantity=quantity,
                unit=catalog_item.unit
            )
            full_inventory.append(daily_item)

            if get_pushable_inventory(daily_item, catalog_item):
                pushed_inventory.append(daily_item)
        
        stores_daily.append(StoreDailyState(
            store_id=store.store_id,
            full_inventory=full_inventory,
            pushed_inventory=pushed_inventory,
            truck_status=None  # Explicitly None per requirements
        ))

    # Volunteer availability generation
    volunteers_daily = []
    for vol in world.volunteers:
        available = random.random() < world.availability_rate
        volunteers_daily.append(VolunteerDailyState(
            volunteer_id=vol.volunteer_id,
            available=available
        ))

    return SimulationDay(
        run_id=run_id,
        stores=stores_daily,
        volunteers=volunteers_daily
    )
