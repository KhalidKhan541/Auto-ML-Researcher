"""
ML Experiment Agent - LangGraph implementation
Autonomous research loop: Hypothesis → Data → Architecture → Train → Evaluate → Iterate
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

# ── State ─────────────────────────────────────────────────────────────────────

class ExperimentState(TypedDict):
    # Conversation / messages
    messages: Annotated[list, add_messages]

    # Experiment metadata
    experiment_id: str
    iteration: int
    max_iterations: int

    # Research artifacts (each node writes its section)
    hypothesis: dict          # {goal, rationale, success_criteria, constraints}
    dataset_plan: dict        # {name, source, splits, preprocessing_steps, shape}
    architecture: dict        # {model_type, layers, hyperparameters, rationale}
    training_config: dict     # {optimizer, lr, epochs, batch_size, callbacks}
    eval_results: dict        # {metrics, confusion_matrix, sample_predictions}
    iteration_log: list[dict] # history across iterations

    # Control flow
    status: Literal["running", "converged", "max_iter", "failed"]
    next_action: str          # which node to route to


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an autonomous ML research scientist.
Your job is to design, run, and iteratively improve machine-learning experiments.
Think step-by-step. Be precise about numbers. Output ONLY valid JSON.
Never hallucinate results — mark uncomputed fields as null.
"""

def build_hypothesis_prompt(goal: str, prev_results: dict | None) -> str:
    base = f"""
Task: Formulate a concrete ML research hypothesis.

Research goal: {goal}
Previous evaluation results: {json.dumps(prev_results, indent=2) if prev_results else "None (first iteration)"}

Respond with JSON matching this schema:
{{
  "goal": "one-sentence research goal",
  "hypothesis": "specific, falsifiable hypothesis",
  "rationale": "why this approach should work",
  "success_criteria": {{"metric": "accuracy", "threshold": 0.90}},
  "constraints": ["list", "of", "constraints"],
  "dataset_requirements": "what kind of data is needed"
}}
"""
    return base


def build_dataset_prompt(hypothesis: dict) -> str:
    return f"""
Task: Design the dataset preparation plan for this hypothesis.

Hypothesis: {json.dumps(hypothesis, indent=2)}

Respond with JSON:
{{
  "dataset_name": "name",
  "source": "where to get it (HuggingFace/Kaggle/synthetic/etc)",
  "download_snippet": "Python code to load the dataset (2-5 lines)",
  "target_column": "label column name",
  "feature_columns": ["list"],
  "preprocessing_steps": [
    {{"step": "normalize", "detail": "StandardScaler on numerical cols"}}
  ],
  "train_size": 0.7,
  "val_size": 0.15,
  "test_size": 0.15,
  "expected_shape": {{"train": [null, null], "val": [null, null], "test": [null, null]}}
}}
"""


def build_architecture_prompt(hypothesis: dict, dataset_plan: dict, prev_arch: dict | None) -> str:
    return f"""
Task: Design the model architecture.

Hypothesis: {json.dumps(hypothesis, indent=2)}
Dataset plan: {json.dumps(dataset_plan, indent=2)}
Previous architecture (if iterating): {json.dumps(prev_arch, indent=2) if prev_arch else "None"}

Respond with JSON:
{{
  "model_type": "e.g. feedforward / CNN / transformer / gradient-boosting",
  "framework": "pytorch | sklearn | xgboost",
  "architecture_description": "plain English summary",
  "layers": [
    {{"type": "Linear", "in": 128, "out": 64, "activation": "ReLU"}},
    {{"type": "Dropout", "p": 0.3}},
    {{"type": "Linear", "in": 64, "out": 1, "activation": "Sigmoid"}}
  ],
  "total_params_estimate": "~50K",
  "hyperparameters": {{
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 20,
    "weight_decay": 1e-4
  }},
  "rationale": "why this architecture for this problem"
}}
"""


def build_training_prompt(architecture: dict, dataset_plan: dict) -> str:
    return f"""
Task: Write the complete training script (Python).

Architecture: {json.dumps(architecture, indent=2)}
Dataset: {json.dumps(dataset_plan, indent=2)}

Respond with JSON:
{{
  "training_script": "FULL Python training script as a single string",
  "expected_runtime_minutes": 5,
  "callbacks": ["early_stopping", "lr_scheduler"],
  "checkpointing": "save best model by val_loss"
}}

The training_script must:
- Load data using the download_snippet from dataset_plan
- Build the model per architecture spec
- Train with proper train/val split
- Print epoch metrics as JSON lines: {{"epoch":1,"train_loss":0.5,"val_loss":0.4,"val_acc":0.82}}
- Save the final model to ./checkpoints/model_best.pt
"""


def build_evaluation_prompt(architecture: dict, eval_raw: str, iteration: int) -> str:
    return f"""
Task: Analyze training results and generate evaluation report.

Architecture: {json.dumps(architecture, indent=2)}
Raw training output: {eval_raw}
Iteration: {iteration}

Respond with JSON:
{{
  "metrics": {{
    "train_loss_final": null,
    "val_loss_final": null,
    "val_accuracy": null,
    "test_accuracy": null,
    "f1_score": null
  }},
  "convergence_assessment": "converged | overfitting | underfitting | diverged",
  "key_observations": ["list of observations"],
  "success_criteria_met": true,
  "suggested_improvements": [
    {{"change": "increase dropout", "reason": "signs of overfitting", "priority": "high"}}
  ],
  "should_iterate": true,
  "iteration_verdict": "brief summary of what happened and what to try next"
}}
"""


def build_iteration_prompt(hypothesis: dict, eval_results: dict, iteration_log: list) -> str:
    return f"""
Task: Decide how to update the hypothesis for the next iteration.

Current hypothesis: {json.dumps(hypothesis, indent=2)}
Evaluation results: {json.dumps(eval_results, indent=2)}
Full iteration history: {json.dumps(iteration_log, indent=2)}

Respond with JSON — an UPDATED hypothesis that addresses the evaluation findings:
{{
  "goal": "...",
  "hypothesis": "UPDATED specific hypothesis based on findings",
  "rationale": "what changed and why",
  "success_criteria": {{"metric": "accuracy", "threshold": 0.90}},
  "constraints": ["..."],
  "dataset_requirements": "...",
  "changes_from_prev": ["bullet list of what changed"]
}}
"""


# ── LLM call helper ──────────────────────────────────────────────────────────

def call_llm(prompt: str, llm) -> dict:
    """Call the LLM, parse JSON, retry once on failure."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    for attempt in range(2):
        try:
            response = llm.invoke(messages)
            text = response.content.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except (json.JSONDecodeError, Exception) as e:
            if attempt == 1:
                return {"error": str(e), "raw": text if "text" in dir() else ""}
            time.sleep(1)


# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_hypothesize(state: ExperimentState, llm) -> ExperimentState:
    """Generate or update research hypothesis."""
    goal = state["messages"][-1].content if state["iteration"] == 0 else state["hypothesis"].get("goal", "")
    prev_results = state.get("eval_results") if state["iteration"] > 0 else None

    result = call_llm(build_hypothesis_prompt(goal, prev_results), llm)
    return {
        "hypothesis": result,
        "messages": [AIMessage(content=f"[Iteration {state['iteration']+1}] Hypothesis: {result.get('hypothesis', '')}")],
        "next_action": "dataset",
        "status": "running",
    }


def node_dataset(state: ExperimentState, llm) -> ExperimentState:
    result = call_llm(build_dataset_prompt(state["hypothesis"]), llm)
    return {
        "dataset_plan": result,
        "messages": [AIMessage(content=f"Dataset plan: {result.get('dataset_name', '')} — {result.get('source', '')}")],
        "next_action": "architecture",
    }


def node_architecture_search(state: ExperimentState, llm) -> ExperimentState:
    prev_arch = state.get("architecture") if state["iteration"] > 0 else None
    result = call_llm(
        build_architecture_prompt(state["hypothesis"], state["dataset_plan"], prev_arch),
        llm,
    )
    return {
        "architecture": result,
        "messages": [AIMessage(content=f"Architecture: {result.get('model_type','')} — {result.get('total_params_estimate','')}")],
        "next_action": "train",
    }


def node_train(state: ExperimentState, llm) -> ExperimentState:
    result = call_llm(
        build_training_prompt(state["architecture"], state["dataset_plan"]),
        llm,
    )
    # In production this would actually execute the script in a sandbox
    training_config = {
        "script": result.get("training_script", ""),
        "callbacks": result.get("callbacks", []),
        "expected_runtime_minutes": result.get("expected_runtime_minutes", "unknown"),
        # Simulated output for demo — replace with real execution
        "simulated_output": json.dumps({
            "epoch": 10, "train_loss": 0.31, "val_loss": 0.38, "val_acc": 0.87
        }),
    }
    return {
        "training_config": training_config,
        "messages": [AIMessage(content=f"Training script generated. Expected runtime: {result.get('expected_runtime_minutes')}min")],
        "next_action": "evaluate",
    }


def node_evaluate(state: ExperimentState, llm) -> ExperimentState:
    raw_output = state["training_config"].get("simulated_output", "")
    result = call_llm(
        build_evaluation_prompt(state["architecture"], raw_output, state["iteration"]),
        llm,
    )
    log_entry = {
        "iteration": state["iteration"],
        "hypothesis": state["hypothesis"].get("hypothesis", ""),
        "architecture": state["architecture"].get("model_type", ""),
        "metrics": result.get("metrics", {}),
        "verdict": result.get("iteration_verdict", ""),
        "should_iterate": result.get("should_iterate", False),
    }
    return {
        "eval_results": result,
        "iteration_log": state.get("iteration_log", []) + [log_entry],
        "messages": [AIMessage(content=f"Eval: {result.get('convergence_assessment','')} — {result.get('iteration_verdict','')}")],
        "next_action": "iterate" if result.get("should_iterate") else "done",
        "iteration": state["iteration"] + 1,
    }


def node_iterate(state: ExperimentState, llm) -> ExperimentState:
    if state["iteration"] >= state["max_iterations"]:
        return {"status": "max_iter", "next_action": "done"}

    result = call_llm(
        build_iteration_prompt(
            state["hypothesis"],
            state["eval_results"],
            state.get("iteration_log", []),
        ),
        llm,
    )
    return {
        "hypothesis": result,
        "messages": [AIMessage(content=f"Iterating → {result.get('hypothesis','')} | Changes: {result.get('changes_from_prev', [])}")],
        "next_action": "dataset",
        "status": "running",
    }


def node_done(state: ExperimentState) -> ExperimentState:
    best = max(
        state.get("iteration_log", [{}]),
        key=lambda x: x.get("metrics", {}).get("val_accuracy", 0) or 0,
    )
    return {
        "status": "converged" if state.get("eval_results", {}).get("success_criteria_met") else "max_iter",
        "messages": [AIMessage(content=f"Experiment complete after {state['iteration']} iterations. Best: {best.get('metrics', {})}")],
    }


# ── Router ─────────────────────────────────────────────────────────────────────

def route(state: ExperimentState) -> str:
    action = state.get("next_action", "done")
    mapping = {
        "dataset": "dataset",
        "architecture": "architecture_search",
        "train": "train",
        "evaluate": "evaluate",
        "iterate": "iterate",
        "done": "done",
    }
    return mapping.get(action, "done")


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(llm):
    """Build and compile the LangGraph experiment graph."""
    from functools import partial

    g = StateGraph(ExperimentState)

    g.add_node("hypothesize",        partial(node_hypothesize, llm=llm))
    g.add_node("dataset",            partial(node_dataset, llm=llm))
    g.add_node("architecture_search",partial(node_architecture_search, llm=llm))
    g.add_node("train",              partial(node_train, llm=llm))
    g.add_node("evaluate",           partial(node_evaluate, llm=llm))
    g.add_node("iterate",            partial(node_iterate, llm=llm))
    g.add_node("done",               node_done)

    g.set_entry_point("hypothesize")

    # hypothesize → conditional
    g.add_conditional_edges("hypothesize", route)
    g.add_conditional_edges("dataset",     route)
    g.add_conditional_edges("architecture_search", route)
    g.add_conditional_edges("train",       route)
    g.add_conditional_edges("evaluate",    route)
    g.add_conditional_edges("iterate",     route)
    g.add_edge("done", END)

    return g.compile()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_experiment(goal: str, max_iterations: int = 3, llm=None):
    """Run a full ML experiment given a research goal."""
    import uuid

    if llm is None:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.2)

    graph = build_graph(llm)

    initial_state: ExperimentState = {
        "messages": [HumanMessage(content=goal)],
        "experiment_id": str(uuid.uuid4())[:8],
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

    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    result = run_experiment(
        goal="Build a classifier to predict customer churn on tabular data with >90% accuracy",
        max_iterations=2,
    )
    print(json.dumps(result.get("iteration_log"), indent=2))
