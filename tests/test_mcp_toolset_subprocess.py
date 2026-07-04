"""
tests/test_mcp_toolset_subprocess.py

Proof-of-concept: verifies that the MCP server can be reached via ADK's
McpToolset using stdio (subprocess) transport -- exactly as downstream
agents (Matchmaker, Dispatch, etc.) will connect to it.

This test does NOT import from mcp_servers/ directly. It spawns the server
as a subprocess and uses McpToolset.get_tools() to list and call tools,
proving the real MCP wire format works end to end.

The pattern used here (McpToolset + StdioConnectionParams pointing at the
server module) is exactly how agents/agent.py will wire up the tools:

    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=sys.executable,
                args=["mcp_servers/store_volunteer_server.py"],
            )
        )
    )
    tools = await toolset.get_tools()
    agent = Agent(..., tools=tools)
"""

import sys
import os
import asyncio
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from google.adk.tools.mcp_tool.mcp_toolset import McpToolset, StdioConnectionParams
from google.adk.tools import ToolContext
from mcp import StdioServerParameters


PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
SERVER_MODULE = os.path.join(PROJECT_ROOT, "mcp_servers", "store_volunteer_server.py")


async def call_tool(tools, name: str, args: dict):
    """Helper: find a tool by name and call it with a mock ToolContext."""
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        raise RuntimeError(f"Tool not found: {name!r}")
    # run_async requires a ToolContext; pass None — MCPTool only uses it for
    # auth/confirmation guards which are disabled by default.
    result = await tool.run_async(args=args, tool_context=None)
    return result


async def run_tests() -> bool:
    print("=" * 70)
    print("McpToolset Subprocess Transport -- End-to-End Verification")
    print("=" * 70)
    print(f"\nPython:  {PYTHON_EXE}")
    print(f"Server:  {SERVER_MODULE}\n")

    all_passed = True

    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=PYTHON_EXE,
                args=[SERVER_MODULE],
                env={**os.environ},
            ),
            timeout=30,
        ),
    )

    try:
        # -- Step 1: Tool discovery --------------------------------------------
        tools = await toolset.get_tools()
        tool_names = [t.name for t in tools]

        print("-" * 70)
        print("STEP 1: Tool discovery via McpToolset")
        print("-" * 70)
        expected = {"get_pushable_inventory", "check_vehicle_availability", "get_volunteer_schedule"}
        if expected.issubset(set(tool_names)):
            print(f"  [PASS] All 3 tools discovered: {sorted(tool_names)}")
        else:
            print(f"  [FAIL] Expected {expected}, got {tool_names}")
            all_passed = False

        # -- Step 2: get_pushable_inventory ------------------------------------
        print("\n" + "-" * 70)
        print("STEP 2: call get_pushable_inventory('store_01') via subprocess")
        print("-" * 70)
        inv_result = await call_tool(tools, "get_pushable_inventory", {"store_id": "store_01"})

        def extract(r):
            """Pull the actual payload out of ADK's MCP response envelope.
            ADK wraps tool responses as:
              {'content': [...], 'structuredContent': <actual value>, 'isError': bool}
            """
            if isinstance(r, dict):
                sc = r.get("structuredContent")
                if sc is not None:
                    # FastMCP wraps list returns under {"result": [...]}
                    if isinstance(sc, dict) and "result" in sc:
                        return sc["result"]
                    return sc
                # Fallback: parse first content text
                content = r.get("content", [])
                if content:
                    return json.loads(content[0]["text"])
            return r

        items = extract(inv_result)
        if isinstance(items, list) and len(items) > 0:
            print(f"  [PASS] Received {len(items)} pushed items for store_01")
            print(f"         Sample: {items[0]}")
        else:
            print(f"  [FAIL] Expected non-empty list, got: {inv_result!r}")
            all_passed = False

        # -- Step 3: check_vehicle_availability (caching) ----------------------
        print("\n" + "-" * 70)
        print("STEP 3: call check_vehicle_availability('store_01') x2 via subprocess")
        print("-" * 70)

        r1 = extract(await call_tool(tools, "check_vehicle_availability", {"store_id": "store_01"}))
        r2 = extract(await call_tool(tools, "check_vehicle_availability", {"store_id": "store_01"}))

        if r1 == r2 and isinstance(r1, dict) and "available" in r1 and "capacity_kg" in r1:
            print(f"  [PASS] Both calls returned same cached value: {r1}")
        else:
            print(f"  [FAIL] Caching broken or missing keys -- call1={r1}, call2={r2}")
            all_passed = False

        # -- Step 4: get_volunteer_schedule ------------------------------------
        print("\n" + "-" * 70)
        print("STEP 4: call get_volunteer_schedule('vol_01') via subprocess")
        print("-" * 70)
        vol = extract(await call_tool(tools, "get_volunteer_schedule", {"volunteer_id": "vol_01"}))
        if isinstance(vol, dict) and vol.get("volunteer_id") == "vol_01" and "available" in vol:
            print(f"  [PASS] vol_01 schedule: {vol}")
        else:
            print(f"  [FAIL] Unexpected result: {vol!r}")
            all_passed = False

    finally:
        await toolset.close()

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL STEPS PASSED -- McpToolset subprocess transport confirmed")
    else:
        print("SOME STEPS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    passed = asyncio.run(run_tests())
    sys.exit(0 if passed else 1)
