import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.models import OfferedItem
from agents.culinary_agent import run_culinary

async def main():
    items = [
        OfferedItem(item="toor dal", unit="kg", offered_quantity=5, is_essential=True),
        OfferedItem(item="tomatoes", unit="kg", offered_quantity=10, is_essential=True),
        OfferedItem(item="mustard seeds", unit="kg", offered_quantity=1, is_essential=True),
        OfferedItem(item="sugar", unit="kg", offered_quantity=2, is_essential=True),
        OfferedItem(item="raw rice", unit="kg", offered_quantity=5, is_essential=True),
        OfferedItem(item="bread", unit="units", offered_quantity=10, is_essential=True),
    ]
    res = await run_culinary(items)
    print("----- CULINARY OUTPUT -----")
    print(res)
    print("---------------------------")

if __name__ == "__main__":
    asyncio.run(main())
