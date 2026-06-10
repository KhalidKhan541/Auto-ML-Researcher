"""
FastAPI backend — wraps the LangGraph ML agent
Deploy on: Modal / Railway / Fly.io / any Python host

The Worker calls POST /run  →  we run the graph  →  PATCH updates back to Worker KV
"""

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.graph import ExperimentState, build_graph, run_experiment

# ── Config ────────────────────────────────────────────────────────────────────

WORKER_BASE_URL = os.getenv("WORKER_BASE_URL", "")  # e.g. https://ml-agent.workers.dev
API_KEY = os.getenv("AGENT_API_KEY", "dev-secret")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("ML Agent backend starting…")
    yield
    print("ML Agent backend shutting down.")

app = FastAPI(title="ML Experiment Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────

def check_api_key(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ── Models ────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    experiment_id: str
    goal: str
    max_iterations: int = 3

# ── Background worker ─────────────────────────────────────────────────────────

async def push_update(experiment_id: str, patch: dict):
    """Push a state update back to the Cloudflare Worker KV store."""
    if not WORKER_BASE_URL:
        return  # dev mode — skip
    async with httpx.AsyncClient() as client:
        try:
            await client.patch(
                f"{WORKER_BASE_URL}/api/experiment/{experiment_id}/update",
                json=patch,
                headers={"X-API-Key": API_KEY},
                timeout=10,
            )
        except Exception as e:
            print(f"[push_update] failed: {e}")


async def run_agent(experiment_id: str, goal: str, max_iterations: int):
    """Run the LangGraph agent and stream progress back to Worker."""
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0.2,
        anthropic_api_key=ANTHROPIC_API_KEY,
    )

    from langgraph.graph.message import add_messages
    from langchain_core.messages import HumanMessage

    graph = build_graph(llm)

    initial_state: ExperimentState = {
        "messages": [HumanMessage(content=goal)],
        "experiment_id": experiment_id,
        "iteration": 0,
        "max_iterations": max_iterations,
        "hypothesis": {},
        "dataset_plan": {},
        "architecture": {},
        "training_config": {},
        "eval_results": {},
        "iteration_log": [],
        "status": "running",
        "next_action": "dataset",
    }

    await push_update(experiment_id, {"status": "running", "current_node": "hypothesize"})

    try:
        # Stream node-by-node updates
        async for event in graph.astream(initial_state):
            for node_name, node_state in event.items():
                patch: dict[str, Any] = {
                    "current_node": node_name,
                    "status": node_state.get("status", "running"),
                    "iteration": node_state.get("iteration", 0),
                }
                if node_state.get("iteration_log"):
                    patch["iteration_log"] = node_state["iteration_log"]
                await push_update(experiment_id, patch)

        # Final update
        await push_update(experiment_id, {
            "status": "converged",
            "current_node": "done",
        })

    except Exception as e:
        await push_update(experiment_id, {"status": "failed", "error": str(e)})
        raise


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/run")
async def start_run(req: RunRequest, background_tasks: BackgroundTasks, x_api_key: str = Header(default="")):
    check_api_key(x_api_key)
    background_tasks.add_task(
        lambda: asyncio.run(run_agent(req.experiment_id, req.goal, req.max_iterations))
    )
    return {"queued": True, "experiment_id": req.experiment_id}


# ── Local dev shortcut ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
