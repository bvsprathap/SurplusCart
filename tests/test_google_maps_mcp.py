"""
tests/test_google_maps_mcp.py

Standalone test for the Google Maps MCP server (@cablate/mcp-google-map).
Connects via McpToolset (stdio/subprocess) -- the same pattern agents will use.

Verifies:
  1. Only maps_distance_matrix and maps_directions are discovered
     (maps_geocode must NOT appear -- GOOGLE_MAPS_ENABLED_TOOLS filters it out)
  2. maps_distance_matrix: store_01 -> home_01, returns travel_time_minutes
  3. maps_directions: store_01 -> home_01, returns route with distance + duration
  4. Detour comparison:
       Direct:  store_01 -> home_01
       Detour:  store_01 -> home_05 -> home_01
     Prints both times so we can manually verify the detour is reasonable.

All assertions extract from structuredContent per the Prompt 3 convention.

Chennai coordinates (from world_config.json):
  store_01 : 13.0418, 80.2341  (Sri Balaji Supermarket, T. Nagar)
  home_01  : 13.0012, 80.2565  (Anbu Illam Home, Adyar)
  home_05  : 13.0139, 80.2530  (Sneha Care Home, Kotturpuram)
"""

import sys
import os
import asyncio
import json
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from mcp import StdioServerParameters

# -- Chennai coordinates from world_config.json --------------------------------
STORE_01 = "13.0418,80.2341"   # Sri Balaji Supermarket, T. Nagar
HOME_01  = "13.0012,80.2565"   # Anbu Illam Home, Adyar
HOME_05  = "13.0139,80.2530"   # Sneha Care Home, Kotturpuram

MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()


def extract(r):
    """
    Pull the actual payload out of ADK's MCP response envelope.
    ADK wraps responses as:
      {'structuredContent': <actual value>, 'content': [...], 'isError': bool}
    Always use structuredContent; never parse content text for data.
    """
    if isinstance(r, dict):
        if r.get("isError"):
            raise RuntimeError(f"MCP tool returned an error: {r}")
        sc = r.get("structuredContent")
        if sc is not None:
            # FastMCP wraps list returns under {"result": [...]}
            if isinstance(sc, dict) and list(sc.keys()) == ["result"]:
                return sc["result"]
            return sc
        # Fallback: parse first content text block
        content = r.get("content", [])
        if content and isinstance(content[0], dict):
            return json.loads(content[0]["text"])
    return r


async def call_tool(tools, name: str, args: dict):
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        raise RuntimeError(f"Tool not found in toolset: {name!r}")
    return await tool.run_async(args=args, tool_context=None)


async def run_tests() -> bool:
    print("=" * 70)
    print("Google Maps MCP Server -- End-to-End Test")
    print("=" * 70)

    if not MAPS_API_KEY or MAPS_API_KEY.startswith("your_"):
        print("\n[SKIP] GOOGLE_MAPS_API_KEY not set in .env -- cannot run live tests.")
        print("       Set a valid key and re-run.")
        return False

    all_passed = True

    server_env = {
        **os.environ,
        "GOOGLE_MAPS_API_KEY": MAPS_API_KEY,
        "GOOGLE_MAPS_ENABLED_TOOLS": "maps_distance_matrix,maps_directions",
    }

    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "@cablate/mcp-google-map", "--stdio"],
                env=server_env,
            ),
            timeout=30,
        ),
    )

    try:
        tools = await toolset.get_tools()
        tool_names = {t.name for t in tools}

        # -- Step 1: Tool discovery ----------------------------------------
        print("\n" + "-" * 70)
        print("STEP 1: Tool discovery -- only 2 tools must be registered")
        print("-" * 70)

        required = {"maps_distance_matrix", "maps_directions"}
        forbidden = {"maps_geocode"}

        missing = required - tool_names
        present_forbidden = forbidden & tool_names

        if not missing and not present_forbidden:
            print(f"  [PASS] Exactly the right tools registered: {sorted(tool_names)}")
        else:
            if missing:
                print(f"  [FAIL] Missing required tools: {missing}")
                all_passed = False
            if present_forbidden:
                print(f"  [FAIL] maps_geocode should NOT be registered (ENABLED_TOOLS filter broken)")
                all_passed = False

        # -- Step 2: maps_distance_matrix store_01 -> home_01 -----------------
        print("\n" + "-" * 70)
        print("STEP 2: maps_distance_matrix store_01 -> home_01 (driving)")
        print("-" * 70)

        dm_result = extract(await call_tool(tools, "maps_distance_matrix", {
            "origins": [STORE_01],
            "destinations": [HOME_01],
            "mode": "driving",
        }))

        print(f"  Raw structuredContent: {json.dumps(dm_result, indent=2)[:500]}")

        # The server returns a distance matrix; extract duration minutes
        # Response structure varies -- walk the tree to find duration
        direct_minutes = _extract_minutes(dm_result, label="store_01->home_01")
        if direct_minutes is not None:
            print(f"  [PASS] Direct travel time: {direct_minutes:.1f} min")
        else:
            print(f"  [FAIL] Could not extract travel_time_minutes from result")
            all_passed = False

        # -- Step 3: maps_directions store_01 -> home_01 ----------------------
        print("\n" + "-" * 70)
        print("STEP 3: maps_directions store_01 -> home_01 (driving)")
        print("-" * 70)

        dir_result = extract(await call_tool(tools, "maps_directions", {
            "origin": STORE_01,
            "destination": HOME_01,
            "travelMode": "driving",
        }))

        print(f"  Raw structuredContent (truncated): {json.dumps(dir_result, indent=2)[:600]}")

        distance_ok, duration_ok = _check_directions(dir_result)
        if distance_ok and duration_ok:
            print(f"  [PASS] Route returned with distance and duration fields")
        else:
            if not distance_ok:
                print(f"  [FAIL] 'distance' field not found in directions result")
                all_passed = False
            if not duration_ok:
                print(f"  [FAIL] 'duration' field not found in directions result")
                all_passed = False

        # -- Step 4: Detour comparison ----------------------------------------
        print("\n" + "-" * 70)
        print("STEP 4: Detour comparison (manual verification)")
        print("  Direct:  store_01 -> home_01")
        print("  Detour:  store_01 -> home_05 -> home_01")
        print("-" * 70)

        # Direct time already obtained above
        # Detour: store_01 -> home_05 leg + home_05 -> home_01 leg
        dm_leg1 = extract(await call_tool(tools, "maps_distance_matrix", {
            "origins": [STORE_01],
            "destinations": [HOME_05],
            "mode": "driving",
        }))
        dm_leg2 = extract(await call_tool(tools, "maps_distance_matrix", {
            "origins": [HOME_05],
            "destinations": [HOME_01],
            "mode": "driving",
        }))

        leg1_minutes = _extract_minutes(dm_leg1, "store_01->home_05")
        leg2_minutes = _extract_minutes(dm_leg2, "home_05->home_01")

        if leg1_minutes is not None and leg2_minutes is not None:
            detour_total = leg1_minutes + leg2_minutes
            detour_added = detour_total - (direct_minutes or 0)
            print(f"  Direct time         : {direct_minutes:.1f} min")
            print(f"  Leg 1 (->home_05)   : {leg1_minutes:.1f} min")
            print(f"  Leg 2 (->home_01)   : {leg2_minutes:.1f} min")
            print(f"  Detour total        : {detour_total:.1f} min")
            print(f"  Extra detour time   : +{detour_added:.1f} min")
            if detour_added >= 0:
                print(f"  [PASS] Detour adds {detour_added:.1f} min -- reasonable for Chennai traffic")
            else:
                print(f"  [WARN] Detour time is less than direct (unusual, check routing)")
        else:
            print(f"  [FAIL] Could not extract detour leg times")
            all_passed = False

    finally:
        await toolset.close()

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL STEPS PASSED")
    else:
        print("SOME STEPS FAILED")
    print("=" * 70)
    return all_passed


def _extract_minutes(data, label: str) -> float | None:
    """
    Extract travel time in minutes from the @cablate distance matrix response.

    Confirmed response shape:
      {
        "durations": [[{"value": <seconds>, "text": "23 mins"}]],
        "distances": [[{"value": <meters>, "text": "9.4 km"}]],
        "origin_addresses": [...],
        "destination_addresses": [...]
      }
    """
    if data is None:
        return None

    # Primary shape (confirmed from live call): durations[0][0]["value"] in seconds
    if isinstance(data, dict) and "durations" in data:
        try:
            secs = data["durations"][0][0]["value"]
            return float(secs) / 60.0
        except (IndexError, KeyError, TypeError):
            pass

    # Fallback: flat travel_time_minutes key
    if isinstance(data, dict) and "travel_time_minutes" in data:
        return float(data["travel_time_minutes"])

    # Fallback: Google Maps standard rows/elements shape
    if isinstance(data, dict) and "rows" in data:
        try:
            elem = data["rows"][0]["elements"][0]
            if elem.get("status") == "OK":
                return float(elem["duration"]["value"]) / 60.0
        except (IndexError, KeyError, TypeError):
            pass

    # Fallback: list wrapping
    if isinstance(data, list) and data:
        return _extract_minutes(data[0], label)

    # Fallback: duration_seconds / duration_minutes keys
    if isinstance(data, dict):
        for key in ("duration_seconds", "durationSeconds"):
            if key in data:
                return float(data[key]) / 60.0
        for key in ("duration_minutes", "durationMinutes", "duration_min"):
            if key in data:
                return float(data[key])

    print(f"  [WARN] {label}: unrecognised response shape, cannot extract minutes")
    print(f"         Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    return None


def _check_directions(data) -> tuple[bool, bool]:
    """
    Check that directions result contains 'distance' and 'duration' somewhere.
    Works recursively over dicts/lists.
    """
    text = json.dumps(data).lower()
    distance_ok = "distance" in text
    duration_ok = "duration" in text
    return distance_ok, duration_ok


if __name__ == "__main__":
    passed = asyncio.run(run_tests())
    sys.exit(0 if passed else 1)
