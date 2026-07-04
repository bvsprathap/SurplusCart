---
name: GCP ADK Workflow Best Practices
description: Best practices, lessons learned, and utility scripts for building Google ADK 2.0 graphs, Vertex AI Agent Engines, and human-in-the-loop (HITL) ambient workflows.
---

# GCP ADK Workflow Best Practices

This skill provides guidelines for deploying robust, ambient, event-driven agent workflows using the Google ADK and Vertex AI Agent Engine.

## 1. What Works Well
- **Human-in-the-Loop (HITL)**: Yielding `RequestInput(interrupt_id="...")` to pause workflows, and using `VertexAiSessionService.list_sessions()` in a frontend to fetch pending tasks.
- **State Management**: Using `ctx.state` to store cross-node data (e.g., LLM Risk Assessments) so that external frontends can parse the session state and display rich contextual data.
- **Architectural Ordering**: Running security constraints (PII redaction, prompt-injection heuristics) as the very first node in a `Workflow` graph, before any business routing (e.g., auto-approvals).
- **Pub/Sub to REST Relays**: Using a lightweight Cloud Run relay to map incoming Pub/Sub push events to the Vertex AI `streamQuery` API.

## 2. What Did NOT Work & Lessons Learned (What NOT to use)
- **Avoid Custom Session IDs on Creation**: Do NOT pass a custom string `session_id` in the `streamQuery` payload for a new session. Vertex AI Agent Engine will attempt to fetch it, resulting in a `SessionNotFoundError`. Instead, omit `session_id` and allow the engine to auto-generate its own numeric ID.
- **Handle Dictionary Resumption Payloads**: Do NOT assume `ctx.resume_inputs.get(interrupt_id)` returns a plain string. Frontends and REST APIs typically send a JSON dictionary (e.g., `{"human_decision": "approve"}`). Use `isinstance(input, dict)` to parse it safely, otherwise `.strip()` will throw an `AttributeError` and permanently crash the resumed workflow.
- **Security Checkpoint Ordering Vulnerability**: Do NOT place auto-approval or threshold routing before security checks. If a malicious input (prompt injection) meets the auto-approval threshold (e.g., < $100), it will bypass the security checkpoint entirely if routed directly to approval.
- **Avoid `InMemoryRunner` for Prod**: Despite documentation, `InMemoryRunner` wipes sessions on restart. Always deploy to the actual Agent Engine or use `DatabaseSessionService` for ambient workflows.
- **Avoid Subprocess CLI on Windows**: When invoking `gcloud pubsub` in test scripts on Windows, avoid `subprocess.run` due to nested quote escaping issues. Prefer hitting the REST relay directly via `requests` or using the native Python SDK.

## 3. Utility Scripts
The `scripts/gcp_adk_utils.py` file contains reusable generic functions for extracting pending human-in-the-loop tasks from Vertex AI, and safely resuming them.
