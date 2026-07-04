import asyncio
import json
import requests
from typing import List, Dict, Any, Optional

async def get_pending_interrupts(
    session_service: Any, 
    app_name: str, 
    interrupt_name: str = "adk_request_input"
) -> List[Dict[str, Any]]:
    """
    Scans recent Vertex AI Agent Engine sessions and extracts those actively 
    waiting for human-in-the-loop input.
    
    Args:
        session_service: An instance of VertexAiSessionService
        app_name: The Agent Runtime ID (projects/.../reasoningEngines/...)
        interrupt_name: The name of the ADK RequestInput interrupt (default: adk_request_input)
    """
    response = await session_service.list_sessions(app_name=app_name)
    session_list = getattr(response, "sessions", [])
    if not session_list and isinstance(response, list):
        session_list = response
        
    pending = []
    
    for session_info in session_list:
        session_id = getattr(session_info, "id", getattr(session_info, "session_id", None))
        user_id = getattr(session_info, "user_id", "default-user")
        if not session_id:
            continue
            
        session = await session_service.get_session(app_name=app_name, session_id=session_id, user_id=user_id)
        events = getattr(session, "events", getattr(session, "history", []))
        
        unresolved_call = None
        for event in events:
            event_dict = event.model_dump() if hasattr(event, "model_dump") else event
            parts = []
            
            if "content" in event_dict and isinstance(event_dict["content"], dict):
                parts = event_dict["content"].get("parts", [])
            elif "actions" in event_dict and event_dict["actions"]:
                parts = event_dict["actions"].get("parts", [])
            elif "output" in event_dict and event_dict["output"]:
                parts = event_dict["output"].get("parts", [])
            elif "parts" in event_dict:
                parts = event_dict.get("parts", [])
                
            for part in parts:
                if "function_call" in part and part["function_call"]:
                    fc = part["function_call"]
                    if fc.get("name") == interrupt_name:
                        unresolved_call = {"id": fc.get("id"), "args": fc.get("args", {})}
                elif "function_response" in part and part["function_response"]:
                    fr = part["function_response"]
                    if fr.get("name") == interrupt_name:
                        if unresolved_call and unresolved_call["id"] == fr.get("id"):
                            unresolved_call = None
                            
        if unresolved_call:
            state = getattr(session, "state", {})
            pending.append({
                "session_id": session_id,
                "user_id": user_id,
                "interrupt_id": unresolved_call["id"],
                "args": unresolved_call["args"],
                "state": state
            })
            
    return pending

async def resume_workflow(
    session_service: Any,
    app_name: str,
    session_id: str,
    user_id: str,
    interrupt_id: str,
    decision_payload: Dict[str, Any]
) -> Any:
    """
    Resumes a paused Vertex AI Agent Engine workflow with a human decision.
    Note: The ADK agent node MUST parse this as a dict (see safe_parse_resume_input).
    """
    response_payload = {
        interrupt_id: decision_payload
    }
    return await session_service.resume_session(
        app_name=app_name,
        session_id=session_id,
        user_id=user_id,
        interrupt_response=response_payload
    )

def safe_parse_resume_input(ctx: Any, interrupt_id: str) -> str:
    """
    Helper function to use INSIDE an ADK @node to safely parse the resume input
    regardless of whether the frontend sent a string or a JSON dictionary.
    """
    decision_input = ctx.resume_inputs.get(interrupt_id, "")
    if isinstance(decision_input, dict):
        return str(decision_input.get(interrupt_id, "")).strip().lower()
    return str(decision_input).strip().lower()

def forward_to_relay(relay_url: str, payload_dict: Dict[str, Any]) -> requests.Response:
    """
    Generic wrapper to push a JSON payload to a Cloud Run relay
    that forwards to Vertex AI streamQuery.
    """
    import base64
    import json
    
    encoded_data = base64.b64encode(json.dumps(payload_dict).encode("utf-8")).decode("utf-8")
    pubsub_message = {
        "message": {
            "data": encoded_data
        }
    }
    return requests.post(relay_url, json=pubsub_message)
