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
import datetime
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest

from agents.fast_api_app import app
import agents.app_utils.cooldown as cooldown_mod

client = TestClient(app)

@pytest.fixture(autouse=True)
def clean_cooldown_state(tmp_path):
    """Fixture to ensure a fresh, clean state for each test by resetting local files and memory."""
    # Reset in-memory state
    cooldown_mod._IN_MEMORY_TIMESTAMP = None
    
    # Reset local timestamp file path to a temp path
    temp_local_path = str(tmp_path / "last_run_timestamp.txt")
    
    with patch("agents.app_utils.cooldown._LOCAL_TIMESTAMP_PATH", temp_local_path), \
         patch("agents.app_utils.cooldown.storage.Client") as mock_storage_client:
        
        # Mock GCS entirely to avoid network calls and rely purely on local file/memory in tests
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket.blob.return_value = mock_blob
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        
        yield temp_local_path

def test_cooldown_duration_config():
    """Verify REFRESH_COOLDOWN_MINUTES defaults to 15 when not otherwise set and is configurable."""
    with patch.dict(os.environ, {}, clear=True):
        assert cooldown_mod.get_cooldown_minutes() == 15.0
        
    with patch.dict(os.environ, {"REFRESH_COOLDOWN_MINUTES": "5.5"}):
        assert cooldown_mod.get_cooldown_minutes() == 5.5
        
    with patch.dict(os.environ, {"REFRESH_COOLDOWN_MINUTES": "invalid"}):
        assert cooldown_mod.get_cooldown_minutes() == 15.0


def test_cooldown_blocks_refresh_immediately(clean_cooldown_state):
    """Cooldown check correctly blocks a refresh attempt when called again immediately after a successful run."""
    # Set the last run timestamp to now (so elapsed time = 0 minutes)
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    cooldown_mod.set_last_run_timestamp(now)
    
    assert cooldown_mod.is_in_cooldown() is True
    
    # Visit /refresh
    response = client.get("/refresh", follow_redirects=False)
    # Should silently redirect to root / (HTTP 303 See Other)
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_cooldown_allows_refresh_after_expiry(clean_cooldown_state):
    """Cooldown check correctly allows a refresh attempt after REFRESH_COOLDOWN_MINUTES has elapsed."""
    # Mock cooldown duration to 2 minutes for testing
    with patch("agents.app_utils.cooldown.get_cooldown_minutes", return_value=2.0):
        # Set the last run timestamp to 3 minutes ago
        past_time = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.timedelta(minutes=3)
        cooldown_mod.set_last_run_timestamp(past_time)
        
        assert cooldown_mod.is_in_cooldown() is False
        
        # /refresh should allow the refresh and load the spinner page
        response = client.get("/refresh", follow_redirects=False)
        assert response.status_code == 200
        assert "Running Daily Simulation" in response.text


def test_timestamp_survives_simulated_restart(clean_cooldown_state):
    """Persisted timestamp survives a simulated 'restart' (test reads the timestamp from persistence layer fresh)."""
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    cooldown_mod.set_last_run_timestamp(now)
    
    # Simulate restart by clearing in-memory variable
    cooldown_mod._IN_MEMORY_TIMESTAMP = None
    
    # Reading again should load from disk
    restored = cooldown_mod.get_last_run_timestamp()
    assert restored is not None
    assert abs((restored - now).total_seconds()) < 1.0


def test_no_concurrent_runs(clean_cooldown_state):
    """A request arriving while a run is already in progress does not start a second concurrent run."""
    # Mock run_simulation to block or take time (we don't need a real run)
    mock_run_simulation = MagicMock()
    
    with patch("agents.fast_api_app._IS_RUNNING", True), \
         patch("main.run_simulation", mock_run_simulation):
         
        # When /run?force=true is requested but a run is already active, it should return running page or current cache summary
        response = client.get("/run?force=true")
        assert response.status_code == 200
        # No simulation should be triggered since _IS_RUNNING was already True
        mock_run_simulation.assert_not_called()


def test_refresh_redirects_during_active_run(clean_cooldown_state):
    """Verify that hitting /refresh while a run is in progress silently redirects to /."""
    with patch("agents.fast_api_app._IS_RUNNING", True):
        response = client.get("/refresh", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/"


def test_default_page_behavior_unchanged(clean_cooldown_state):
    """Default page (/) behavior is unchanged — still serves last completed static report regardless of cooldown state."""
    # Scenario A: In cooldown
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    cooldown_mod.set_last_run_timestamp(now)
    with patch("agents.fast_api_app._CACHED_SUMMARY", "Cached Summary Active"):
        response = client.get("/")
        assert response.status_code == 200
        assert "Cached Summary Active" in response.text

    # Scenario B: Not in cooldown
    past_time = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.timedelta(minutes=30)
    cooldown_mod.set_last_run_timestamp(past_time)
    with patch("agents.fast_api_app._CACHED_SUMMARY", "Cached Summary Past"):
        response = client.get("/")
        assert response.status_code == 200
        assert "Cached Summary Past" in response.text
