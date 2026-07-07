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

import datetime
import os
import google.auth
from google.cloud import storage

_LOCAL_TIMESTAMP_PATH = os.path.join("reports", "output", "last_run_timestamp.txt")
_IN_MEMORY_TIMESTAMP = None  # Ultimate fallback for tests/local dev

def get_cooldown_minutes() -> float:
    """Get the cooldown duration from environment variables (default 15.0)."""
    try:
        return float(os.environ.get("REFRESH_COOLDOWN_MINUTES", "15"))
    except ValueError:
        return 15.0

def _get_bucket_name() -> str:
    """Get the GCS bucket name dynamically based on GCP project id."""
    bucket_env = os.environ.get("SURPLUS_COOLDOWN_BUCKET")
    if bucket_env:
        return bucket_env
    try:
        _, project_id = google.auth.default()
        if project_id:
            return f"surpluscart-data-{project_id}"
    except Exception:
        pass
    return "surpluscart-data-decent-rampart-500008-n6"

def get_last_run_timestamp() -> datetime.datetime | None:
    """Retrieve the last run timestamp from GCS, local disk, or in-memory fallback."""
    global _IN_MEMORY_TIMESTAMP
    
    # 1. Try reading from GCS
    bucket_name = _get_bucket_name()
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("last_run_timestamp.txt")
        if blob.exists():
            content = blob.download_as_text(encoding="utf-8").strip()
            if content:
                # Store locally & in-memory to keep them in sync
                _IN_MEMORY_TIMESTAMP = datetime.datetime.fromisoformat(content)
                try:
                    os.makedirs(os.path.dirname(_LOCAL_TIMESTAMP_PATH), exist_ok=True)
                    with open(_LOCAL_TIMESTAMP_PATH, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception:
                    pass
                return _IN_MEMORY_TIMESTAMP
    except Exception as e:
        # Expected if running locally without GCP auth, or bucket doesn't exist yet in local testing
        pass

    # 2. Fallback to Local Disk
    if os.path.exists(_LOCAL_TIMESTAMP_PATH):
        try:
            with open(_LOCAL_TIMESTAMP_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                _IN_MEMORY_TIMESTAMP = datetime.datetime.fromisoformat(content)
                return _IN_MEMORY_TIMESTAMP
        except Exception:
            pass

    # 3. Fallback to In-Memory
    return _IN_MEMORY_TIMESTAMP

def set_last_run_timestamp(dt: datetime.datetime) -> None:
    """Persist the last run timestamp to GCS, local disk, and in-memory cache."""
    global _IN_MEMORY_TIMESTAMP
    content = dt.isoformat()
    _IN_MEMORY_TIMESTAMP = dt

    # 1. Persist to Local Disk
    try:
        os.makedirs(os.path.dirname(_LOCAL_TIMESTAMP_PATH), exist_ok=True)
        with open(_LOCAL_TIMESTAMP_PATH, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"Warning: Failed to save timestamp locally: {e}")

    # 2. Persist to GCS
    bucket_name = _get_bucket_name()
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("last_run_timestamp.txt")
        blob.upload_from_string(content, content_type="text/plain", encoding="utf-8")
    except Exception as e:
        # Graceful if offline or no credentials
        pass

def is_in_cooldown() -> bool:
    """Check if we are currently within the cooldown window."""
    last_run = get_last_run_timestamp()
    if not last_run:
        return False
    
    cooldown_minutes = get_cooldown_minutes()
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    elapsed = (now - last_run).total_seconds() / 60.0
    return elapsed < cooldown_minutes

def read_gcs_report_file(filename: str) -> str | None:
    """Read a report file from GCS."""
    bucket_name = _get_bucket_name()
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        if blob.exists():
            return blob.download_as_text(encoding="utf-8")
    except Exception:
        pass
    return None

def write_gcs_report_file(filename: str, content: str) -> bool:
    """Write a report file to GCS."""
    bucket_name = _get_bucket_name()
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_string(
            content,
            content_type="text/html" if filename.endswith(".html") else "text/plain",
            encoding="utf-8"
        )
        return True
    except Exception:
        return False
