from google.adk.agents import Agent
from main import run_simulation

async def run_simulation_tool() -> dict:
    """Run the complete food rescue simulation for today.
    
    This runs the full orchestration pipeline:
    - Scrapes pushed inventory via Store MCP
    - Connects to Volunteer schedules
    - Uses Matchmaker agent to generate food offers
    - Simulates A2A Care Home negotiations for surplus food
    - Dispatches delivery volunteers and fallback trucks
    - Generates full operational HTML report and maps
    """
    result = await run_simulation()
    
    # We only return the strings necessary for the UI or next step
    # Pydantic models need to be serialized 
    if "stats" in result and hasattr(result["stats"], "model_dump"):
         result["stats"] = result["stats"].model_dump()
         
    return result

root_agent = Agent(
    name="food_rescue_agent",
    model="gemini-2.5-flash",
    description="SurplusCart: Agentic Food Rescue System",
    instruction="""
    You are the Food to Go orchestrator agent for Chennai.
    When triggered, you run a complete food rescue simulation for
    the day: collecting surplus food from stores, matching it to
    care homes via negotiation, dispatching volunteers, and
    generating a full operations report.
    Respond with the simulation results including the delivery
    summary, negotiation outcomes, and a link to the full report.
    """,
    tools=[run_simulation_tool]
)
