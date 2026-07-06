import asyncio
import sys
import os

from data.data_model import Store, CareHome, FoodCatalogItem, WorldConfig, SimulationDay
from tools.models import OrderLineItem
from tools.constraint_tools import single_store_candidate, StockLedger

async def run_bananas_test():
    store_a = Store(store_id="store_a", name="Store A", latitude=12.0, longitude=77.0)
    store_b = Store(store_id="store_b", name="Store B", latitude=12.1, longitude=77.1)
    store_c = Store(store_id="store_c", name="Store C", latitude=12.2, longitude=77.2)
    
    # 8, 6, 3 = 17 total
    from data.data_model import DailyFoodItem, StoreDailyState
    sim_day = SimulationDay(
        run_id="run-1", 
        stores=[
            StoreDailyState(store_id="store_a", pushed_inventory=[DailyFoodItem(name="bananas", days_to_expiry=1, quantity=8, unit="dozen")], full_inventory=[]),
            StoreDailyState(store_id="store_b", pushed_inventory=[DailyFoodItem(name="bananas", days_to_expiry=1, quantity=6, unit="dozen")], full_inventory=[]),
            StoreDailyState(store_id="store_c", pushed_inventory=[DailyFoodItem(name="bananas", days_to_expiry=1, quantity=3, unit="dozen")], full_inventory=[])
        ], 
        volunteers=[]
    )
    ledger = StockLedger(sim_day=sim_day)
    
    requested = [OrderLineItem(item="bananas", unit="dozen", offered_quantity=10, accepted_quantity=10)]
    
    res = single_store_candidate(requested_items=requested, urgent_item_names=set(), ledger=ledger, catalog=[])
    
    print("Result for 10 bananas:")
    print(res)

if __name__ == "__main__":
    asyncio.run(run_bananas_test())
