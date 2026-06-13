# ML Experiment Agent
[![Hire me on Fiverr](https://img.shields.io/badge/Hire%20me%20on-Fiverr-1dbf73?style=for-the-badge&logo=fiverr)](https://www.fiverr.com/khalid_khan55)

An autonomous LangGraph research scientist that runs the full ML loop:
**Hypothesis → Dataset → Architecture → Train → Evaluate → Iterate**

Deployed on **Cloudflare Workers + Pages** with a Python LangGraph backend.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Cloudflare Edge                             │
│                                                                 │
│  ┌──────────────────┐        ┌──────────────────────────────┐  │
│  │  Cloudflare Pages│        │    Cloudflare Worker         │  │
│  │  (frontend/)     │◄──────►│    (worker/index.ts)         │  │
│  │  index.html      │  REST  │    POST /api/experiment       │  │
│  └──────────────────┘        │    GET  /api/experiment/:id  │  │
│                               │    PATCH .../update          │  │
│                               │    KV: experiment store      │  │
│                               └──────────────┬───────────────┘  │
└──────────────────────────────────────────────┼──────────────────┘
                                               │ HTTP
                              ┌────────────────▼──────────────────┐
                              │   Python Backend (Modal/Railway)  │
                              │   server.py  (FastAPI)            │
                              │                                   │
                              │  ┌─────────────────────────────┐ │
                              │  │     LangGraph Agent          │ │
                              │  │                             │ │
                              │  │  Hypothesize                │ │
                              │  │      ↓                      │ │
                              │  │  Dataset Prep               │ │
                              │  │      ↓                      │ │
                              │  │  Architecture Search        │ │
                              │  │      ↓                      │ │
                              │  │  Train (generates script)   │ │
                              │  │      ↓                      │ │
                              │  │  Evaluate                   │ │
                              │  │      ↓                      │ │
                              │  │  Iterate?  ──yes──► loop   │ │
                              │  │      ↓ no                   │ │
                              │  │  Done                       │ │
                              │  └─────────────────────────────┘ │
                              │   Anthropic Claude API           │
                              └───────────────────────────────────┘
```

---

## Project Structure

```
ml-agent/
├── agent/
│   └── graph.py          # LangGraph agent — all nodes, prompts, state
├── worker/
│   └── index.ts          # Cloudflare Worker — REST API + KV store
├── frontend/
│   └── index.html        # Single-page dashboard (deploy to CF Pages)
├── server.py             # FastAPI server wrapping the agent
├── requirements.txt
└── wrangler.toml         # (create this — see worker/index.ts bottom)
```

---

## Setup

### 1. Python Backend

```bash
pip install -r requirements.txt

# Set env vars
export ANTHROPIC_API_KEY=sk-ant-...
export AGENT_API_KEY=your-secret-key
export WORKER_BASE_URL=https://ml-agent.YOUR-SUBDOMAIN.workers.dev

# Run locally
python server.py
```

Deploy to **Modal** (recommended for on-demand GPU):
```python
# modal_deploy.py
import modal
app = modal.App("ml-agent")

@app.function(
    image=modal.Image.debian_slim().pip_install_from_requirements("requirements.txt"),
    secrets=[modal.Secret.from_name("ml-agent-secrets")],
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    from server import app
    return app
```

Or deploy to **Railway**:
```bash
railway init
railway up
```

### 2. Cloudflare Worker

```bash
npm install -g wrangler
wrangler login

# Create KV namespace
wrangler kv:namespace create EXPERIMENTS_KV
# Copy the ID into wrangler.toml

# Set secrets
wrangler secret put AGENT_API_KEY      # same key as AGENT_API_KEY above
wrangler secret put AGENT_BACKEND_URL  # your deployed Python service URL

# Deploy
wrangler deploy
```

**wrangler.toml** (create in project root):
```toml
name = "ml-agent-worker"
main = "worker/index.ts"
compatibility_date = "2024-09-01"

[[kv_namespaces]]
binding = "EXPERIMENTS_KV"
id = "YOUR_KV_NAMESPACE_ID_HERE"
```

### 3. Cloudflare Pages (Frontend)

```bash
# From project root
wrangler pages deploy frontend/ --project-name ml-agent-ui
```

Or connect your GitHub repo in the Cloudflare dashboard → Pages → point build output to `frontend/`.

---

## Requirements

```
langgraph>=0.2.0
langchain-anthropic>=0.2.0
langchain-core>=0.3.0
fastapi>=0.115.0
uvicorn>=0.32.0
httpx>=0.28.0
pydantic>=2.0.0
```

---

## Extending the Agent

### Add a real training sandbox
Replace the `simulated_output` in `node_train()` with actual subprocess execution:

```python
import subprocess, tempfile, os

script = state["training_config"]["script"]
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
    f.write(script)
    script_path = f.name

result = subprocess.run(
    ["python", script_path],
    capture_output=True, text=True, timeout=300
)
training_output = result.stdout + result.stderr
```

### Add memory / RAG
Store `iteration_log` in a vector store and retrieve similar past experiments:

```python
from langchain_community.vectorstores import Chroma
# embed each iteration_log entry and retrieve k=3 most relevant
# pass as context into build_hypothesis_prompt()
```

### Add architecture search with Optuna
Replace the LLM architecture node with an Optuna trial:

```python
import optuna
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    # ... build and train model
    return val_accuracy
```

---

## Agent Flow Details

| Node | Input | Output | LLM calls |
|------|-------|--------|-----------|
| `hypothesize` | goal + prev eval | hypothesis JSON | 1 |
| `dataset` | hypothesis | dataset plan JSON | 1 |
| `architecture_search` | hypothesis + dataset + prev arch | architecture JSON | 1 |
| `train` | architecture + dataset | training script + config | 1 |
| `evaluate` | training output | metrics + verdict + suggestions | 1 |
| `iterate` | hypothesis + eval | updated hypothesis | 1 |
| `done` | full state | final report | 0 |

**Per iteration: ~5 LLM calls. 3 iterations = ~15 calls.**
