# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

from agents.fast_api_app import app

client = TestClient(app)

def test_root_route_loads_cached_summary_when_present():
    with patch("agents.fast_api_app._CACHED_SUMMARY", "Cached Summary Content"):
        response = client.get("/")
        assert response.status_code == 200
        assert "Cached Summary Content" in response.text

def test_root_route_loads_from_disk_fallback(tmp_path):
    fake_reports_dir = tmp_path / "reports" / "output"
    fake_reports_dir.mkdir(parents=True, exist_ok=True)
    fake_summary_file = fake_reports_dir / "latest_summary.html"
    fake_summary_file.write_text("Disk Summary Content", encoding="utf-8")
    
    with patch("agents.fast_api_app._CACHED_SUMMARY", None), \
         patch("os.path.join", return_value=str(fake_summary_file)), \
         patch("os.path.exists", return_value=True):
        response = client.get("/")
        assert response.status_code == 200
        assert "Disk Summary Content" in response.text

def test_root_route_shows_placeholder_when_no_cache_or_disk_file():
    with patch("agents.fast_api_app._CACHED_SUMMARY", None), \
         patch("os.path.exists", return_value=False):
        response = client.get("/")
        assert response.status_code == 200
        assert "No completed food rescue simulation report was found" in response.text
        assert "Trigger Simulation Run" in response.text
        assert 'href="/refresh"' in response.text

def test_refresh_route_returns_loading_spinner_and_clears_cache():
    with patch("agents.fast_api_app._CACHED_SUMMARY", "some cache"), \
         patch("agents.fast_api_app._CACHED_REPORT", "some report"), \
         patch("agents.fast_api_app._CACHED_MAP", "some map"):
        
        response = client.get("/refresh")
        assert response.status_code == 200
        assert "Running Daily Simulation" in response.text
        assert "fetch('/run?force=true')" in response.text
        
        # Verify in-memory cache was cleared
        from agents import fast_api_app
        assert fast_api_app._CACHED_SUMMARY is None
        assert fast_api_app._CACHED_REPORT is None
        assert fast_api_app._CACHED_MAP is None
