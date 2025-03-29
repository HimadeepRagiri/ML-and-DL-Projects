from pydantic import BaseModel

## Defining Pydantic Models

# Orchestrator Agent Output
class OrchestratorOutput(BaseModel):
    status: str
    message: str
    target_agent: str
    parameters: dict
