"""
tests/test_culinary_agent.py

Test suite for the Culinary Agent.
"""

from __future__ import annotations

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from tools.models import OfferedItem, MatchmakerOffer
from agents.culinary_agent import run_culinary

def _llm_available() -> bool:
    """Return True if Gemini LLM auth is available."""
    try:
        import google.auth
        _, _ = google.auth.default()
        return True
    except Exception:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return bool(key and not key.startswith("your_"))

_skip_no_llm = pytest.mark.skipif(
    not _llm_available(),
    reason="No LLM auth available (Vertex AI ADC or API key required)"
)

# ---------------------------------------------------------------------------
# Unit / Graceful Fallback Tests (No LLM needed if length < 2)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_list_fallback():
    res = await run_culinary([])
    assert "also available: none" in res.lower()

@pytest.mark.asyncio
async def test_single_item_fallback():
    item = OfferedItem(item="Milk", unit="units", offered_quantity=10, is_essential=True)
    res = await run_culinary([item])
    assert "also available: milk" in res.lower()
    assert "sambar" not in res.lower()

# ---------------------------------------------------------------------------
# Live LLM Tests
# ---------------------------------------------------------------------------

@_skip_no_llm
@pytest.mark.asyncio
async def test_run_culinary_non_empty():
    items = [
        OfferedItem(item="Tomato", unit="kg", offered_quantity=5, is_essential=True),
        OfferedItem(item="Rice", unit="kg", offered_quantity=10, is_essential=True),
        OfferedItem(item="Lentils", unit="kg", offered_quantity=3, is_essential=True),
    ]
    res = await run_culinary(items)
    assert isinstance(res, str)
    assert len(res.strip()) > 0

@_skip_no_llm
@pytest.mark.asyncio
async def test_plain_text_validation():
    items = [
        OfferedItem(item="Tomato", unit="kg", offered_quantity=5, is_essential=True),
        OfferedItem(item="Rice", unit="kg", offered_quantity=10, is_essential=True),
        OfferedItem(item="Lentils", unit="kg", offered_quantity=3, is_essential=True),
    ]
    res = await run_culinary(items)
    # Check that it doesn't contain markdown headers
    assert "#" not in res
    # Check that it doesn't contain bold markdown formatting
    assert "**" not in res
    # Check that it doesn't contain JSON structure
    assert not (res.strip().startswith("{") and res.strip().endswith("}"))

@_skip_no_llm
@pytest.mark.asyncio
async def test_veg_only_no_meat_suggested():
    items = [
        OfferedItem(item="Tomato", unit="kg", offered_quantity=5, is_essential=True),
        OfferedItem(item="Rice", unit="kg", offered_quantity=10, is_essential=True),
        OfferedItem(item="Lentils", unit="kg", offered_quantity=3, is_essential=True),
    ]
    res = await run_culinary(items)
    res_lower = res.lower()
    assert "chicken" not in res_lower
    assert "fish" not in res_lower
    assert "egg" not in res_lower
    assert "meat" not in res_lower

@_skip_no_llm
@pytest.mark.asyncio
async def test_no_extra_items_hallucinated():
    items = [
        OfferedItem(item="Tomato", unit="kg", offered_quantity=5, is_essential=True),
        OfferedItem(item="Rice", unit="kg", offered_quantity=10, is_essential=True),
    ]
    res = await run_culinary(items)
    # Let's verify that only Tomato and Rice are named as items used.
    # While it might mention "water" or other generic terms in context,
    # it shouldn't introduce other primary rescue ingredients (like chicken, bread, paneer, apples)
    res_lower = res.lower()
    for ingredient in ["chicken", "paneer", "apple", "bread", "banana"]:
        assert ingredient not in res_lower

@_skip_no_llm
@pytest.mark.asyncio
async def test_placeholder_replacement():
    items = [
        OfferedItem(item="Tomato", unit="kg", offered_quantity=5, is_essential=True),
        OfferedItem(item="Rice", unit="kg", offered_quantity=10, is_essential=True),
    ]
    dish_framing = await run_culinary(items)
    
    offer_message = "We have prepared an offer for you.\n\n[DISH_FRAMING]\n\nBest regards,\nFood Rescue Team"
    final_message = offer_message.replace("[DISH_FRAMING]", dish_framing)
    
    assert "[DISH_FRAMING]" not in final_message
    assert dish_framing in final_message
