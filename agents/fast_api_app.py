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
    """Return a loading page that calls /run in the background."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Food Rescue Simulation</title>
        <style>
            body { background: #000517; color: #04D8D9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding-top: 20vh; margin: 0; }
            .spinner { width: 50px; height: 50px; border: 5px solid rgba(4, 216, 217, 0.3); border-radius: 50%; border-top-color: #04D8D9; animation: spin 1s ease-in-out infinite; margin: 0 auto 20px auto; }
            @keyframes spin { to { transform: rotate(360deg); } }
            h1 { font-weight: 300; letter-spacing: 1px; margin-bottom: 10px; }
            p { color: #087C81; font-size: 14px; }
            #content { opacity: 0; transition: opacity 0.5s ease-in; }
            #loading { transition: opacity 0.5s ease-out; }
        </style>
        <script>
            window.onload = function() {
                fetch('/run')
                    .then(response => response.text())
                    .then(html => {
                        document.open();
                        document.write(html);
                        document.close();
                    })
                    .catch(err => {
                        document.getElementById('loading').innerHTML = '<h1>Error running simulation</h1><p>' + err + '</p>';
                    });
            };
        </script>
    </head>
    <body>
        <div id="loading">
            <div class="spinner"></div>
            <h1>Running Daily Simulation</h1>
            <p>Agents are negotiating and dispatching orders... Please wait (~30s).</p>
        </div>
    </body>
    </html>
    """
    return html_content

# Global cache to prevent re-running the simulation on every refresh
_CACHED_REPORT = None
_CACHED_MAP = None
_IS_RUNNING = False

@app.get("/run", response_class=HTMLResponse)
async def run_sim() -> str:
    """Run the simulation once and cache the HTML report."""
    global _CACHED_REPORT, _CACHED_MAP, _IS_RUNNING
    
    if _CACHED_REPORT:
        return _CACHED_REPORT
        
    if _IS_RUNNING:
        return "<h1>Simulation is currently running. Please wait a moment and refresh.</h1>"
        
    _IS_RUNNING = True
    try:
        from main import run_simulation
        result = await run_simulation()
        
        html_content = result.get("report_html")
        map_content = result.get("map_html", "<h1>Simulation complete. Check logs.</h1>")
        
        if not html_content:
            html_content = "<h1>Simulation complete.</h1><p>Check logs for details.</p>"
            
        _CACHED_REPORT = html_content
        _CACHED_MAP = map_content
        return html_content
    finally:
        _IS_RUNNING = False

@app.get("/map", response_class=HTMLResponse)
async def get_map() -> str:
    """Return the cached map HTML."""
    if _CACHED_MAP:
        return _CACHED_MAP
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
