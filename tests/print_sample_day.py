import json
import sys
from data.data_model import setup_world, generate_daily_data

def print_world_info(world):
    print("=" * 80)
    print("STATIC WORLD CONFIGURATION INFO")
    print("=" * 80)
    print(f"Catalog Items loaded from catalog.json: {len(world.catalog)}")
    print(f"Stores loaded from world_config.json: {len(world.stores)}")
    for store in world.stores:
        print(f"  - {store.store_id}: {store.name} at ({store.latitude}, {store.longitude}), has_own_truck={store.has_own_truck}")
        
    print(f"Care Homes loaded from world_config.json: {len(world.care_homes)}")
    for home in world.care_homes:
        constraints_str = f"Veg Only: {home.hard_constraints.vegetarian_only}, Kids: {home.hard_constraints.has_young_children}"
        notes_str = ", ".join([f"{note.item} ({note.type}={note.value})" for note in home.memory_notes]) if home.memory_notes else "None"
        print(f"  - {home.care_home_id}: {home.name} at ({home.latitude}, {home.longitude}), Residents: {home.resident_count}, Storage: {home.storage_capacity_kg}kg")
        print(f"    Constraints: [{constraints_str}] | Memory Notes: [{notes_str}]")

    print(f"Volunteers loaded from world_config.json: {len(world.volunteers)}")
    for vol in world.volunteers[:3]:
        print(f"  - {vol.volunteer_id}: {vol.name} at ({vol.latitude}, {vol.longitude}), vehicle={vol.vehicle_type}, cap={vol.capacity_kg}kg")
    print(f"  ... (+ {len(world.volunteers) - 3} more volunteers)")
    print("=" * 80 + "\n")

def print_simulation_run(run_num, sim_day, world):
    print("-" * 80)
    print(f"SIMULATION RUN #{run_num} - Run ID: {sim_day.run_id}")
    print("-" * 80)
    
    # Assertions and Auditing Checks
    print("Running Audit Checks:")
    
    # 1. Check full_inventory item count for all stores equals catalog count
    catalog_count = len(world.catalog)
    all_stores_have_full_catalog = True
    for s_state in sim_day.stores:
        if len(s_state.full_inventory) != catalog_count:
            all_stores_have_full_catalog = False
            print(f"  [ERROR] Store {s_state.store_id} has full_inventory size {len(s_state.full_inventory)} (expected {catalog_count})")
            
    if all_stores_have_full_catalog:
        print(f"  [PASS] All stores have exactly {catalog_count} items in full_inventory.")
    
    # 2. Print pushed inventory count for each store
    pushed_counts = []
    for s_state in sim_day.stores:
        pushed_counts.append(f"{s_state.store_id}: {len(s_state.pushed_inventory)}")
    print(f"  [INFO] Pushed inventory sizes: {', '.join(pushed_counts)}")
    
    # 3. Check days_to_expiry variance in full_inventory
    # We expect some items in full_inventory to not be equal to their push_threshold_days
    # Let's inspect a sample store (store_01)
    sample_store = sim_day.stores[0]
    varying_expiry_detected = False
    for daily_item in sample_store.full_inventory:
        # Find corresponding catalog item to see push threshold
        catalog_item = next(c for c in world.catalog if c.name == daily_item.name)
        if daily_item.days_to_expiry != catalog_item.push_threshold_days:
            varying_expiry_detected = True
            break
            
    if varying_expiry_detected:
        print("  [PASS] days_to_expiry values in full_inventory are randomized and NOT all equal to the push thresholds.")
    else:
        print("  [ERROR] days_to_expiry values in full_inventory are all equal to push thresholds (no variance detected).")
        
    print("\nStores Daily State Details:")
    for store_state in sim_day.stores:
        # Print first 3 full inventory items as representative sample
        full_sample_str = ", ".join([f"{item.name} (qty={item.quantity}, exp={item.days_to_expiry}d)" for item in store_state.full_inventory[:3]])
        pushed_items = [f"{item.name} (qty={item.quantity}, exp={item.days_to_expiry}d)" for item in store_state.pushed_inventory]
        pushed_str = ", ".join(pushed_items) if pushed_items else "None"
        
        print(f"  - Store: {store_state.store_id} | Truck Status: {store_state.truck_status}")
        print(f"    Full Inventory (sample): [{full_sample_str} ...]")
        print(f"    Pushed Inventory: [{pushed_str}]")
        
    available_vols = [v.volunteer_id for v in sim_day.volunteers if v.available]
    total_vols = len(sim_day.volunteers)
    print(f"\nVolunteers Available ({len(available_vols)}/{total_vols}):")
    print(f"  Available IDs: {', '.join(available_vols) if available_vols else 'None'}")
    print("-" * 80 + "\n")

def main():
    print("Audit Step 1: Loading static configuration files...")
    try:
        world = setup_world("world_config.json", "catalog.json")
        print("  [PASS] Both catalog.json and world_config.json loaded without errors.")
    except Exception as e:
        print(f"  [FAIL] Error loading JSON files: {e}")
        sys.exit(1)
        
    print_world_info(world)
    
    # Store world state signature to verify consistency
    # We serialize names and coordinates of stores/homes to check if they stay identical
    static_sig = {
        "stores": [(s.store_id, s.name, s.latitude, s.longitude) for s in world.stores],
        "homes": [(h.care_home_id, h.name, h.latitude, h.longitude, h.hard_constraints.vegetarian_only) for h in world.care_homes]
    }
    
    # Generate 3 daily runs
    for run_num in range(1, 4):
        sim_day = generate_daily_data(world)
        
        # Verify world identity consistency
        # setup_world loaded config remains same, but generate_daily_data shouldn't touch it
        current_stores = [(s.store_id, s.name, s.latitude, s.longitude) for s in world.stores]
        current_homes = [(h.care_home_id, h.name, h.latitude, h.longitude, h.hard_constraints.vegetarian_only) for h in world.care_homes]
        if current_stores == static_sig["stores"] and current_homes == static_sig["homes"]:
            print(f"  [PASS] Run #{run_num}: World identity remains perfectly identical.")
        else:
            print(f"  [ERROR] Run #{run_num}: World identity changed or mutated!")
            
        print_simulation_run(run_num, sim_day, world)

if __name__ == "__main__":
    main()
