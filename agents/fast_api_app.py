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
os.environ["SERVED_VIA_API"] = "1"
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import google.auth
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.utils.agent_card_builder import AgentCardBuilder
from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.cloud import logging as google_cloud_logging

from agents.food_rescue_agent import root_agent
from google.adk.apps import App
adk_app = App(name="food_rescue", root_agent=root_agent)
from agents.app_utils.telemetry import setup_telemetry
from agents.app_utils.typing import Feedback

setup_telemetry()
_, project_id = google.auth.default()
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")
artifact_service = (
    GcsArtifactService(bucket_name=logs_bucket_name)
    if logs_bucket_name
    else InMemoryArtifactService()
)

runner = Runner(
    app=adk_app,
    artifact_service=artifact_service,
    session_service=InMemorySessionService(),
)

request_handler = DefaultRequestHandler(
    agent_executor=A2aAgentExecutor(runner=runner), task_store=InMemoryTaskStore()
)

A2A_RPC_PATH = f"/a2a/{adk_app.name}"


async def build_dynamic_agent_card() -> AgentCard:
    """Builds the Agent Card dynamically from the root_agent."""
    agent_card_builder = AgentCardBuilder(
        agent=adk_app.root_agent,
        capabilities=AgentCapabilities(streaming=True),
        rpc_url=f"{os.getenv('APP_URL', 'http://0.0.0.0:8000')}{A2A_RPC_PATH}",
        agent_version=os.getenv("AGENT_VERSION", "0.1.0"),
    )
    agent_card = await agent_card_builder.build()
    return agent_card


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    agent_card = await build_dynamic_agent_card()
    a2a_app = A2AFastAPIApplication(agent_card=agent_card, http_handler=request_handler)
    a2a_app.add_routes_to_app(
        app_instance,
        agent_card_url=f"{A2A_RPC_PATH}{AGENT_CARD_WELL_KNOWN_PATH}",
        rpc_url=A2A_RPC_PATH,
        extended_agent_card_url=f"{A2A_RPC_PATH}{EXTENDED_AGENT_CARD_PATH}",
    )
    yield


app = FastAPI(
    title="food-rescue-agent",
    description="API for interacting with the Food Rescue Agent",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def get_root() -> str:
    """Return the cached latest summary html with disk fallback or a clean placeholder with refresh link."""
    global _CACHED_SUMMARY
    if _CACHED_SUMMARY:
        return _CACHED_SUMMARY
        
    # 1. Try reading from GCS
    from agents.app_utils.cooldown import read_gcs_report_file
    gcs_content = read_gcs_report_file("latest_summary.html")
    if gcs_content:
        _CACHED_SUMMARY = gcs_content
        return _CACHED_SUMMARY

    # 2. Try reading from Local Disk fallback
    latest_summary_path = os.path.join("reports", "output", "latest_summary.html")
    if os.path.exists(latest_summary_path):
        try:
            with open(latest_summary_path, "r", encoding="utf-8") as f:
                _CACHED_SUMMARY = f.read()
            return _CACHED_SUMMARY
        except Exception:
            pass
            
    # Clean placeholder page if no summary is found
    placeholder_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SurplusCart: Agentic Food Rescue</title>
        <style>
            body { background: #000517; color: #04D8D9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding-top: 20vh; margin: 0; }
            .container { max-width: 600px; margin: 0 auto; padding: 20px; }
            h1 { font-weight: 300; letter-spacing: 1px; margin-bottom: 20px; }
            p { color: #087C81; font-size: 16px; margin-bottom: 30px; line-height: 1.6; }
            .btn { display: inline-block; background: transparent; color: #04D8D9; border: 2px solid #04D8D9; padding: 12px 30px; font-size: 16px; border-radius: 4px; text-decoration: none; transition: all 0.3s ease; cursor: pointer; }
            .btn:hover { background: #04D8D9; color: #000517; box-shadow: 0 0 15px rgba(4, 216, 217, 0.4); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SurplusCart: Agentic Food Rescue</h1>
            <p>Welcome! No completed food rescue simulation report was found. Press the button below to trigger and view the daily simulation run.</p>
            <a href="/refresh" class="btn">Trigger Simulation Run</a>
        </div>
    </body>
    </html>
    """
    return placeholder_html

# Global cache to prevent re-running the simulation on every refresh
_CACHED_SUMMARY = None
_CACHED_REPORT = None
_CACHED_MAP = None
_IS_RUNNING = False

@app.get("/run", response_class=HTMLResponse)
async def run_sim(force: bool = False) -> str:
    """Run the simulation once and cache the HTML report."""
    global _CACHED_SUMMARY, _CACHED_REPORT, _CACHED_MAP, _IS_RUNNING
    
    # 1. Pre-fetch cached summary from GCS or disk if not in memory
    if not _CACHED_SUMMARY:
        from agents.app_utils.cooldown import read_gcs_report_file
        _CACHED_SUMMARY = read_gcs_report_file("latest_summary.html")
        if not _CACHED_SUMMARY:
            latest_summary_path = os.path.join("reports", "output", "latest_summary.html")
            if os.path.exists(latest_summary_path):
                try:
                    with open(latest_summary_path, "r", encoding="utf-8") as f:
                        _CACHED_SUMMARY = f.read()
                except Exception:
                    pass

    # 2. Handle cooldown and active run concurrency checks
    from agents.app_utils.cooldown import is_in_cooldown
    if force:
        if _IS_RUNNING:
            return _CACHED_SUMMARY or "<h1>Simulation is currently running. Please wait a moment and refresh.</h1>"
        if is_in_cooldown():
            return _CACHED_SUMMARY or "<h1>Simulation is in cooldown. Showing last run.</h1>"
            
    if _CACHED_SUMMARY and not force:
        return _CACHED_SUMMARY
        
    if _IS_RUNNING:
        return "<h1>Simulation is currently running. Please wait a moment and refresh.</h1>"
        
    _IS_RUNNING = True
    try:
        from main import run_simulation
        result = await run_simulation()
        
        summary_content = result.get("summary_html")
        html_content = result.get("report_html")
        map_content = result.get("map_html", "<h1>Simulation complete. Check logs.</h1>")
        
        if not summary_content:
            summary_content = "<h1>Simulation complete.</h1><p>Check logs for details.</p>"
            
        _CACHED_SUMMARY = summary_content
        _CACHED_REPORT = html_content
        _CACHED_MAP = map_content
        
        # 3. Persist the run details
        import datetime
        from agents.app_utils.cooldown import set_last_run_timestamp, write_gcs_report_file
        
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        set_last_run_timestamp(now)
        
        write_gcs_report_file("latest_summary.html", summary_content)
        write_gcs_report_file("latest_report.html", html_content)
        write_gcs_report_file("map.html", map_content)
        
        return summary_content
    finally:
        _IS_RUNNING = False

@app.get("/refresh", response_class=HTMLResponse)
async def refresh_sim():
    """Serve a loading page that triggers a fresh simulation via /run?force=true and redirects to /, with a 15m cooldown."""
    from fastapi.responses import RedirectResponse
    from agents.app_utils.cooldown import is_in_cooldown
    
    # Silent redirect if cooldown is active
    if is_in_cooldown():
        return RedirectResponse(url="/", status_code=303)
        
    global _CACHED_SUMMARY, _CACHED_REPORT, _CACHED_MAP
    _CACHED_SUMMARY = None
    _CACHED_REPORT = None
    _CACHED_MAP = None
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SurplusCart: Agentic Food Rescue</title>
        <style>
            body { background: #000517; color: #04D8D9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding-top: 20vh; margin: 0; }
            .spinner { width: 50px; height: 50px; border: 5px solid rgba(4, 216, 217, 0.3); border-radius: 50%; border-top-color: #04D8D9; animation: spin 1s ease-in-out infinite; margin: 0 auto 20px auto; }
            @keyframes spin { to { transform: rotate(360deg); } }
            h1 { font-weight: 300; letter-spacing: 1px; margin-bottom: 10px; }
            p { color: #087C81; font-size: 14px; }
            #loading { transition: opacity 0.5s ease-out; }
        </style>
        <script>
            window.onload = function() {
                fetch('/run?force=true')
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok: ' + response.statusText);
                        }
                        return response.text();
                    })
                    .then(html => {
                        window.location.href = '/';
                    })
                    .catch(err => {
                        document.getElementById('loading').innerHTML = '<h1>Error running simulation</h1><p>' + err + '</p><p><a href="/refresh" style="color: #04D8D9;">Try again</a></p>';
                    });
            };
        </script>
    </head>
    <body>
        <div id="loading">
            <div class="spinner"></div>
            <h1>SurplusCart: Running Daily Simulation</h1>
            <p>Agents are negotiating and dispatching orders... Please wait (~30s).</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/report.html", response_class=HTMLResponse)
@app.get("/report", response_class=HTMLResponse)
async def get_report() -> str:
    """Return the cached detailed report HTML with GCS and disk fallback."""
    global _CACHED_REPORT
    if _CACHED_REPORT:
        return _CACHED_REPORT
        
    # 1. GCS fallback
    from agents.app_utils.cooldown import read_gcs_report_file
    gcs_content = read_gcs_report_file("latest_report.html")
    if gcs_content:
        _CACHED_REPORT = gcs_content
        return _CACHED_REPORT

    # 2. Disk fallback
    latest_report_path = os.path.join("reports", "output", "latest_report.html")
    if os.path.exists(latest_report_path):
        try:
            with open(latest_report_path, "r", encoding="utf-8") as f:
                _CACHED_REPORT = f.read()
            return _CACHED_REPORT
        except Exception:
            pass
    return "<h1>Report not available.</h1><p>Please run the simulation first at <a href='/'>home</a>.</p>"

@app.get("/map.html", response_class=HTMLResponse)
@app.get("/map", response_class=HTMLResponse)
async def get_map() -> str:
    """Return the cached map HTML with GCS and disk fallback."""
    global _CACHED_MAP
    if _CACHED_MAP:
        return _CACHED_MAP
        
    # 1. GCS fallback
    from agents.app_utils.cooldown import read_gcs_report_file
    gcs_content = read_gcs_report_file("map.html")
    if gcs_content:
        _CACHED_MAP = gcs_content
        return _CACHED_MAP

    # 2. Disk fallback
    latest_map_path = os.path.join("reports", "output", "map.html")
    if os.path.exists(latest_map_path):
        try:
            with open(latest_map_path, "r", encoding="utf-8") as f:
                _CACHED_MAP = f.read()
            return _CACHED_MAP
        except Exception:
            pass
    return "<h1>Map not available.</h1><p>Please run the simulation first at <a href='/'>home</a>.</p>"


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
