# CONTEXT RESTORATION — Food Rescue Multi-Agent System

> **Last updated:** 2026-07-04T09:20 IST
> **Last completed prompt:** Prompt 11 (Orchestrator)
> **Next prompt:** Prompt 12 (Cloud Run Deployment)

---

## PROJECT IDENTITY

- **Course:** Google x Kaggle 5-Day AI Agents Intensive Vibe Coding Course
- **Track:** Agents for Good
- **Project root:** `C:\Users\bvspr\.gemini\Projects\Food to Go`
- **Python venv:** `.venv\` (Python 3.12.10)
- **Key packages:** google-adk, google-genai, pydantic, fastmcp, pytest, pytest-asyncio

## PROJECT SUMMARY

A multi-agent food rescue system routing near-expiry surplus food from 5 Chennai
supermarkets to 5 Care Homes via 20 volunteers. Single simulated day per run.
Built with Google ADK (Python), MCP servers (store/volunteer data + Google Maps),
and A2A protocol (4 negotiating care homes). Fixed world data from JSON config files,
randomized daily inventory and volunteer availability per run.

---

## COMPLETE FILE STRUCTURE

```
C:\Users\bvspr\.gemini\Projects\Food to Go\
│
├── catalog.json                    # 31 food items (static reference, NOT code)
├── world_config.json               # 5 stores, 5 care homes, 20 volunteers (static)
├── .env                            # GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_MAPS_API_KEY
├── .env.template                   # Placeholder template (committed to git)
├── pyproject.toml                  # Project config, pytest settings (asyncio strict mode)
│
├── data/
│   ├── __init__.py
│   └── data_model.py               # Pydantic models, setup_world(), generate_daily_data()
│
├── tools/
│   ├── __init__.py
│   ├── models.py                   # Pipeline Pydantic models (DispatchStats added)
│   ├── constraint_tools.py         # hard_constraint_filter, StockLedger, single_store_candidate
│   ├── guardrails.py               # OrderOutput, DispatchOutput, MatchmakerOfferGuardrail
│   ├── logger.py                   # log_message(), get_message_log(), clear_log()
│   └── dispatch.py                 # Deterministic dispatch engine (Prompt 9)
│
├── reports/
│   ├── __init__.py
│   ├── report_generator.py         # Four-section report generator (Prompt 10)
│   └── output/                     # Generated HTML maps, auto-created
│
├── agents/
│   ├── __init__.py
│   ├── agent.py                    # Root ADK agent (scaffold boilerplate — adk_a2a template)
│   ├── matchmaker_agent.py         # Matchmaker LlmAgent (Prompt 6)
│   ├── culinary_agent.py           # Culinary dish framing agent (Prompt 7)
│   └── care_home_agent.py          # Care Home A2A negotiation (Prompt 8)
│
├── mcp_servers/
│   ├── __init__.py
│   └── store_volunteer_server.py   # FastMCP server: 3 tools (pushable inv, vehicle, volunteer)
│
├── reports/                        # (empty — manifest + map generation, built later)
│   └── __init__.py
│
└── tests/
    ├── __init__.py
    ├── print_sample_day.py         # Data layer verification script (all PASS)
    ├── test_store_volunteer_mcp.py # MCP server in-process test (all PASS)
    ├── test_tools.py               # 54 tool tests (54/54 PASS)
    ├── test_matchmaker_agent.py    # 17 matchmaker tests (17/17 PASS)
    ├── test_culinary_agent.py      # 7 culinary tests (7/7 PASS)
    ├── test_care_home_agent.py     # 22 care home tests (22/22 PASS)
    ├── test_dispatch.py            # 33 dispatch tests (33/33 PASS)
    └── test_report_generator.py    # 34 report tests (34/34 PASS)
```

---

## CLOUD / API CONFIGURATION

| Setting | Value |
|---------|-------|
| GCP Project | `decent-rampart-500008-n6` |
| Vertex AI Region | `us-central1` (NOT `global`) |
| Vertex AI env var | `GOOGLE_GENAI_USE_VERTEXAI=True` (deprecated → `GOOGLE_GENAI_USE_ENTERPRISE`) |
| Gemini model | `gemini-2.5-flash` (available; `gemini-2.0-flash` is NOT available) |
| Auth | Application Default Credentials (ADC) via `google.auth.default()` |
| Google Maps MCP | `@cablate/mcp-google-map` (npm global install) |
| Maps API key env var | `GOOGLE_MAPS_API_KEY` (stored as `Google_API_KEY` in .env) |
| Both `GOOGLE_API_KEY` and `GEMINI_API_KEY` are set | SDK prefers `GOOGLE_API_KEY` |

---

## KEY ARCHITECTURAL DECISIONS

### Agent Architecture
1. **LLM agents only where genuine ambiguity exists** — everything else is deterministic Python.
2. **Matchmaker is Phase 1 only** — builds offer, sends to care home. Phase 2 (process
   negotiation response, call `single_store_candidate`, deduct from `StockLedger`, create
   `Order` objects) is handled by the **ORCHESTRATOR** after negotiation resolves.
3. **4 care homes negotiate via A2A** sequentially (`home_01→02→03→04`), `home_05`
   auto-accepts remainder. Sequencing managed by orchestrator, not Matchmaker.
4. **StockLedger is shared and depleting** — each care home sees what remains after the
   previous home accepted.

### Data Flow
5. `single_store_candidate` returns `needs_commercial` list for items needing a 4th store —
   Dispatch (not this function) decides commercial pickup.
6. `reject_all` response: *"Noted. Will connect with you on another day."* — no revised
   offer, move to next care home.
7. `McpToolset` responses extract from `structuredContent` key (not `content`).
8. Google Maps MCP (`@cablate/mcp-google-map`): `distance_matrix` returns
   `durations[0][0].value` (seconds), `distances[0][0].value` (meters) — nested lists,
   NOT flat `rows/elements` format.

### Source of Truth
9. `world_config.json` is source of truth for all entity data including `availability_rate`,
   `truck_capacity_kg`, `negotiates_via_a2a`.
10. `catalog.json` is source of truth for food items — item definitions never hardcoded.

---

## DATA MODELS (data/data_model.py)

### Static Config Models (from world_config.json)
- `FoodCatalogItem`: name, is_perishable, is_essential, push_threshold_days, unit
- `Store`: store_id, name, lat/lng, has_own_truck, **truck_capacity_kg**
- `CareHome`: care_home_id, name, lat/lng, hard_constraints (vegetarian_only,
  has_young_children), resident_count, storage_capacity_kg, **negotiates_via_a2a**,
  memory_notes (List[MemoryNote])
- `MemoryNote`: item, type ("exclude"|"max_quantity"), value (Optional[float])
- `Volunteer`: volunteer_id, name, lat/lng, capacity_kg
- `WorldConfig`: **availability_rate** (default 0.7), catalog, stores, care_homes, volunteers

### Daily Simulation Models (generated per run)
- `DailyFoodItem`: name, days_to_expiry, quantity, unit
- `StoreDailyState`: store_id, full_inventory (31 items, audit only),
  **pushed_inventory** (filtered subset — ONLY data pipeline sees), truck_status (cached)
- `TruckStatus`: available (bool), capacity_kg (float)
- `VolunteerDailyState`: volunteer_id, available (bool)
- `SimulationDay`: run_id, stores, volunteers

### Key Functions
- `setup_world()` → WorldConfig (loads catalog.json + world_config.json)
- `generate_daily_data(world)` → SimulationDay (randomized inventory + availability)
- `get_pushable_inventory(daily_item, catalog_item)` → bool (days_to_expiry ≤ push_threshold)

---

## PIPELINE OUTPUT MODELS (tools/models.py)

- `NegotiationTurn`: turn_number, speaker (`Literal["system"|"care_home"]`),
  action (`Literal["offer"|"accept_all"|"reduce_item"|"exclude_item"|"request_item"|
  "flag_urgent"|"reject_all"]`), item, quantity
- `OrderLineItem`: item, unit, offered_quantity, accepted_quantity
- `Order`: order_id, care_home_id, store_id, items, urgent_essential_items,
  negotiation_transcript, final_notice
- `Delivery`: delivery_id, store_id, order_ids (`Field(max_length=2)`),
  method (`Literal["volunteer"|"store_truck"|"commercial"]`), volunteer_id, pickup_time
- `OfferedItem`: item, unit, offered_quantity, is_essential
- `MatchmakerOffer`: care_home_id, offered_items, rationale, offer_message
  (contains `[DISH_FRAMING]` placeholder), expected_today_statement, urgency_request
- `CareHomeAction`: action (Literal: accept_all/reduce_item/exclude_item/request_item/
  flag_urgent/reject_all), item (Optional), requested_quantity (Optional)
- `CareHomeResponse`: actions (List[CareHomeAction]), rationale — used as output_schema
- `NegotiationResult`: care_home_id, status ("agreed"|"rejected"), agreed_items
  (List[OrderLineItem]), urgent_item_names (List[str], deduplicated/sorted),
  negotiation_transcript (List[NegotiationTurn]), rejection_message (Optional)
- `DispatchStats`: total_deliveries, volunteer_assigned, store_truck_assigned,
  commercial_assigned, volunteers_unavailable, urgent_items_forced_fallback, detours_bundled

---

## CONSTRAINT TOOLS (tools/constraint_tools.py)

### hard_constraint_filter(care_home, offered_items, catalog) → List[DailyFoodItem]
Applies in order: (a) vegetarian filter, (b) memory-note exclusions, (c) memory-note
quantity caps. Returns filtered/capped list. Original objects NOT mutated.

### StockLedger(sim_day) — shared depleting stock
- Internal: `_stock: Dict[(store_id, item_name_lower), float]`
- Public methods: `store_ids()`, `get_available()`, `get_store_totals()`,
  `get_cross_store_totals()`, `deduct()`, `snapshot()`
- Processing order enforced by CALLER (orchestrator), not inside this class.

### single_store_candidate(requested_items, urgent_item_names, ledger, catalog) → dict
Returns: `{"assignments": [...], "deferred": [...], "needs_commercial": [...]}`.
Logic: (1) try single store, (2) split urgent/non-urgent, (3) retry single for urgent,
(4) assign urgent individually, (5) cap at 3 stores → 4th goes to needs_commercial,
(6) piggyback non-urgent on selected stores, (7) remainder → deferred.
**Store coverage is all-or-nothing** — no partial quantity fulfillment across stores.

---

## GUARDRAILS (tools/guardrails.py)

- `OrderOutput`: validates accepted_qty ≥ 0, within ledger snapshot (tuple or string keys),
  no excluded items present
- `DispatchOutput`: validates payload ≤ capacity, time ≤ 120 min, ≤ 2 order_ids.
  **vehicle_capacity_kg is REQUIRED** (no default). Use `COMMERCIAL_CAPACITY_SENTINEL`
  (99999.0) for commercial pickups.
- `MatchmakerOfferGuardrail`: validates all offered_qty > 0, within cross_store_snapshot,
  no excluded items. Takes `cross_store_snapshot: Dict[str, float]` and
  `excluded_item_names: List[str]`.

---

## MCP SERVERS

### store_volunteer_server.py (mcp_servers/)
FastMCP server exposing 3 tools:
1. `get_pushable_inventory(store_id)` → List[DailyFoodItem dict]
2. `check_vehicle_availability(store_id)` → TruckStatus dict (uses per-store
   `truck_capacity_kg` from WorldConfig, cached on first call)
3. `get_volunteer_schedule(volunteer_id)` → {volunteer_id, available, capacity_kg}

Connected via ADK's `McpToolset` using **stdio subprocess transport** for real MCP
demonstration. In-process path only used by test scripts.

### Google Maps MCP (@cablate/mcp-google-map)
External community MCP server. Connected via stdio transport.
- `maps_distance_matrix` → durations/distances as `[[{value, text}]]`
- `maps_geocode` → lat/lng verification (not core pipeline dependency)
- Extraction helper: `_extract_minutes(result)` handles the nested list format

---

## MATCHMAKER AGENT (agents/matchmaker_agent.py)

### Configuration
- Model: `gemini-2.5-flash` via Vertex AI
- Output: JSON via `output_schema=MatchmakerResponseSchema` (structured output)
- Retry: exactly ONE retry with error feedback in same session

### Two Paths
1. **LLM path** (homes 01-04, `negotiates_via_a2a=True`): Gemini reasons over care home
   profile + eligible items + cross-store totals. Produces `MatchmakerOffer` with reasoning.
2. **Auto-accept path** (home_05, `negotiates_via_a2a=False`): Deterministic, no LLM.
   Full remaining quantity for each eligible item.

### Key Design Points
- `urgency_request` is ALWAYS the standard constant `URGENCY_REQUEST_TEXT` — never
  LLM-generated. Only mentions "essential items" generically.
- `[DISH_FRAMING]` placeholder: instruction tells LLM to include it; parser injects it
  as safety net if missing. Culinary agent replaces it later.
- `is_essential` in output is ALWAYS overridden by catalog truth — never trust LLM.
- `_safe_json_loads()`: 4 repair strategies for Gemini's occasionally malformed JSON
  (direct parse → extract { } → fix newlines → strip code fences).

### Interface
```python
offer = await run_matchmaker(
    care_home, eligible_items, catalog,
    cross_store_totals, cross_store_snapshot, excluded_item_names,
)
```

---

## TEST STATUS (as of Prompt 8 completion)

| Test Suite | Count | Status |
|------------|-------|--------|
| `tests/print_sample_day.py` | ~10 checks | All PASS |
| `tests/test_store_volunteer_mcp.py` | ~28 checks | All PASS |
| `tests/test_tools.py` | 54 tests | **54/54 PASS** |
| `tests/test_matchmaker_agent.py` | 17 tests | **17/17 PASS** |
| `tests/test_culinary_agent.py` | 7 tests | **7/7 PASS** |
| `tests/test_care_home_agent.py` | 22 tests | **22/22 PASS** |
| `tests/test_orchestrator.py` | 12 tests | **12/12 PASS** |
| `tests/test_dispatch.py` | 33 tests | **33/33 PASS** |
| `tests/test_report_generator.py` | 35 tests | **35/35 PASS** |
| **Total pytest** | **180** | **180/180 PASS** ✅ |


---

## WORLD DATA SUMMARY

### 5 Stores (Chennai neighborhoods)
| ID | Name | truck_capacity_kg |
|----|------|-------------------|
| store_01 | Sri Balaji Supermarket, T. Nagar | 3000 |
| store_02 | Chennai Organic Plaza, Velachery | 1500 |
| store_03 | Royal Provision Store, Mylapore | 3000 |
| store_04 | Ganga Departmental Store, Guindy | 2000 |
| store_05 | Metro Food Mart, Anna Nagar | 1500 |

### 5 Care Homes
| ID | Name | negotiates_via_a2a | Key constraints |
|----|------|--------------------|-----------------|
| home_01 | Anbu Illam Home | true | vegetarian_only=true, has_young_children=true |
| home_02 | Karuna Trust Home | true | memory_notes: sugar max 15, paneer excluded |
| home_03 | Nethaji Childrens Home | true | has_young_children=true |
| home_04 | Asha Sadan Home | true | vegetarian_only=true |
| home_05 | Sneha Care Home | **false** | Auto-accepts remainder |

### 20 Volunteers
7 near stores, 7 near care homes, 6 mid-city. Each has capacity_kg (10-25 kg range).

---

## PROMPTS COMPLETED

| # | Topic | Key Output |
|---|-------|------------|
| 1 | Data model + simulation | `data_model.py`, `catalog.json`, `world_config.json` |
| 2 | ADK scaffold (adk_a2a) | Project structure, `agents/agent.py` |
| 3 | Store/Volunteer MCP server | `store_volunteer_server.py`, 3 tools, subprocess transport |
| 4 | Google Maps MCP | `@cablate/mcp-google-map` integration, distance_matrix extraction |
| 5 | Tools + guardrails | `constraint_tools.py`, `guardrails.py`, `models.py`, `logger.py` |
| 6 | Matchmaker agent | `matchmaker_agent.py` (LLM + auto-accept), `MatchmakerOfferGuardrail` |
| 7 | Culinary agent | `culinary_agent.py` — dish framing for Tamil Nadu cuisine |
| 8 | Care Home Negotiation | `care_home_agent.py` — CareHomeAgent + run_negotiation() |
| 9 | Dispatch Module | `tools/dispatch.py` — deterministic, 5-step fallback chain |
| 10 | Report Generator | `reports/report_generator.py` — 4 sections + Folium map |
| 11 | Orchestrator | `main.py` — end-to-end simulation, pipeline wiring |

## REMAINING PROMPTS (estimated)

| # | Topic | Notes |
|---|-------|-------|
| 12 | Cloud Run deployment | Full pipeline deployment via ADK CLI |

---

## CARE HOME NEGOTIATION AGENT (agents/care_home_agent.py)

### Architecture
- Uses in-process ADK Agent with `output_schema=CareHomeResponse` (NOT RemoteA2aAgent HTTP)
- Same `Runner + InMemorySessionService` pattern as Matchmaker
- Deterministic `run_negotiation()` orchestrates the multi-turn exchange

### Two Components
1. **CareHomeAgent** (LLM): System prompt parameterized with care home profile. Responds
   with structured `CareHomeResponse` containing multiple `CareHomeAction` items.
2. **run_negotiation()** (deterministic): Manages offer → response → optional re-offer.
   - Round 1: send offer, parse response, process actions
   - Round 2: only triggered if `request_item` was fulfilled — sends revised offer
   - Returns `NegotiationResult` with agreed_items, urgent_item_names, transcript

### Key Design Points
- `flag_urgent` validation: only essential items (per catalog) can be urgent — non-essential
  silently ignored. Enforced at logic level, not model validator.
- `urgent_item_names` uses `List[str]` (not `Set[str]`) for JSON serialization safety.
  Deduplicated and sorted before constructing NegotiationResult.
- Vegetarian enforcement: logic-level post-processing removes chicken/eggs from agreed_items
  for vegetarian homes. LLM is not reliable for hard dietary constraints.
- `reject_all` → immediate return with `_REJECTION_MESSAGE` constant.
- `request_item` in round 2 → logged but NOT fulfilled (prevents infinite re-offers).
- `reduce_item` quantity clamped to [0, offered_quantity].

### Interface
```python
result = await run_negotiation(
    care_home, offer, ledger, catalog, run_id,
)
# result: NegotiationResult
# result.status: "agreed" | "rejected"
# result.agreed_items: List[OrderLineItem]
# result.urgent_item_names: List[str]  (essential-only, sorted)
```

---

## DISPATCH MODULE (tools/dispatch.py) — Prompt 9

### Architecture
- **Pure deterministic Python** — zero LLM calls
- Accepts **async callable parameters** instead of McpToolset objects
  (allows full testability without MCP sessions or ADK agent context)
- Boundary: runs AFTER orchestrator Phase 2 (single_store_candidate, deduction, Order creation)

### 5-Step Fallback Chain (per Delivery)
1. **Nearest volunteer** — ranked by distance to store; capacity_kg >= payload_kg
2. **Next volunteer within budget** — total route <= 120 minutes
3. **15-min detour bundling** — `check_detour_bundle()` (exposed for orchestrator use)
4. **Store's own truck** — `get_truck_avail` callable; cached after first call
5. **Commercial pickup** — always succeeds; uses `COMMERCIAL_CAPACITY_SENTINEL`

### Interface
```python
from tools.dispatch import run_dispatch

deliveries, stats = await run_dispatch(
    orders=orders,                       # List[Order] from Phase 2
    needs_commercial_items=nc_items,     # List[OrderLineItem] from single_store_candidate
    world=world,
    sim_day=sim_day,
    get_volunteer_avail=vol_avail_fn,    # async (volunteer_id) -> {available: bool}
    get_distance_minutes=dist_fn,        # async (lat1,lng1,lat2,lng2) -> float (minutes)
    get_truck_avail=truck_fn,            # async (store_id) -> {available, capacity_kg}
    run_id=run_id,
)
```

### Callable Wrappers (for orchestrator)
When connecting to real MCP, the orchestrator should wrap:
- `get_volunteer_schedule` MCP tool → `get_volunteer_avail` callable
- `maps_distance_matrix` → extract `durations[0][0].value / 60` for minutes
- `check_vehicle_availability` → `get_truck_avail` callable

---

## IMPORTANT GOTCHAS FOR CONTINUATION

1. **Model availability**: Only `gemini-2.5-flash` works on this project's Vertex AI.
   `gemini-2.0-flash` returns 404.
2. **Region**: Must use `us-central1`, NOT `global`.
3. **Env var deprecation**: `GOOGLE_GENAI_USE_VERTEXAI` is deprecated → should be
   `GOOGLE_GENAI_USE_ENTERPRISE`. Current code still uses the old var.
4. **pushed_inventory vs full_inventory**: Pipeline code must NEVER use `full_inventory`
   — it's audit-only. All tools work with `pushed_inventory`.
5. **AVAILABILITY_RATE constant was removed** from `data_model.py` — now lives in
   `world_config.json` as `availability_rate` and is read via `world.availability_rate`.
6. **Store coverage is all-or-nothing** in `single_store_candidate` — no partial quantity
   fulfillment across stores for the same item.
7. **pytest-asyncio strict mode** — every async test needs `@pytest.mark.asyncio`.
8. **output_schema enforcement** — Use `output_schema=PydanticModel` on ADK Agent for
   guaranteed structured output. More reliable than manual JSON parsing.
9. **Negotiation does NOT modify StockLedger** — Phase 2 (sourcing, deduction, Order
   creation) is the ORCHESTRATOR's job after negotiation resolves.
10. **Dispatch uses callable parameters, not McpToolset** — McpToolset requires agent
    context (ToolContext). For pure Python code, pass async callables that wrap the
    in-process MCP server functions or real McpToolset calls.
