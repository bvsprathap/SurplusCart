import asyncio
import os
import google.auth
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel
from typing import List

_, project_id = google.auth.default()
os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
os.environ['GOOGLE_CLOUD_LOCATION'] = 'us-central1'
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'

class Item(BaseModel):
    name: str

class ResponseSchema(BaseModel):
    items: List[Item]
    rationale: str

async def main():
    agent = Agent(
        name='matchmaker',
        model='gemini-2.5-flash',
        instruction='You are matchmaker. output JSON schema.',
        output_schema=ResponseSchema
    )
    ss = InMemorySessionService()
    runner = Runner(agent=agent, session_service=ss, app_name='test')
    session = await ss.create_session(app_name='test', user_id='system')
    user_content = types.Content(role='user', parts=[types.Part.from_text(text='hello')])
    
    print('=== START RUN_ASYNC ===')
    async for event in runner.run_async(user_id='system', session_id=session.id, new_message=user_content):
        print('--- EVENT ---')
        print('Type:', type(event))
        print('Is final:', getattr(event, 'is_final_response', None))
        if hasattr(event, 'content') and event.content:
            parts = getattr(event.content, 'parts', None) or []
            for i, p in enumerate(parts):
                print(f'  Part {i} text: {getattr(p, "text", None)!r}')
        else:
            print('  No content')

asyncio.run(main())
