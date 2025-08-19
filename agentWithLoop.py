from typing import TypedDict, Dict, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
import json
from dotenv import load_dotenv
import os
from langsmith import traceable
from langchain_core.tracers import LangChainTracer
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import Client
from langsmith.run_helpers import trace


load_dotenv()
# Print debug info to check environment configuration
print("LangChain Projekt:", os.getenv("LANGCHAIN_PROJECT"))
print("LangChain API Key gesetzt:", bool(os.getenv("LANGCHAIN_API_KEY")))

# Wrap the OpenAI client with LangSmith integration
client = wrap_openai(OpenAI())

# LangSmith tracer to send execution traces
tracer = LangChainTracer()

# Explicit LangSmith client (optional if multiple clients or workspaces are needed)
langSmithClient = Client()

# ---- State definition ----
class AgentState(TypedDict, total=False):
    messages: List[str]       # Conversation messages
    answers: Dict[str, str]   # Collected answers from the user
    missing: List[str]        # Still missing required fields
    asked_defaults: bool      # Flag if default questions were already asked
    awaiting_input: bool      # Flag if agent is waiting for user input

# Default questions to ask the user
DEFAULT_QUESTIONS = [
    "Welche Workloads erwartest du (Requests/Sekunde)? -> rps",
    "Durchschnittliche Antwortgröße in KB? -> resp_size_kb",
    "Ziel-Latenz p95 in Millisekunden? -> latency_p95_ms",
    "Anzahl Services im kritischen Pfad? -> service_count",
    "DB-Typ und erwartete QPS? -> db_type_qps",
]
# Required keys for validation
REQUIRED_KEYS = ["rps", "resp_size_kb", "latency_p95_ms", "service_count", "db_type_qps"]

# ---- Structured model definition (validation with Pydantic) ----
class PerfModel(BaseModel):
    workload_rps: int = Field(..., ge=1)
    response_size_kb: int = Field(..., ge=1)
    latency_p95_ms: int = Field(..., ge=1)
    service_count: int = Field(..., ge=1)
    db_type_qps: str
    # Resource estimates as output from the model
    resources_estimate: Dict[str, float]  # Example: {"cpu_cores": 0.5, "ram_mb": 256, "net_mbps": 5.0}

# Node: ask the user the default set of questions
@traceable
def ask_defaults(state: AgentState) -> AgentState:
    if state.get("asked_defaults"):
        return state
    prompt = (
        "Um ein Performance-Modell zu erstellen, beantworte bitte kurz diese Basisfragen.\n"
        + "\n".join(f"- {q}" for q in DEFAULT_QUESTIONS)
        + "\n\nBitte antworte im Format: key=value, durch Komma getrennt.\n"
        + "Beispiel: rps=200, resp_size_kb=50, latency_p95_ms=180, service_count=5, db_type_qps=Postgres 120"
    )
    print(f"\n[Agent] {prompt}")
    state["asked_defaults"] = True
    return state

# Node: evaluate current answers and check for missing fields
@traceable
def evaluate(state: AgentState) -> AgentState:
    answers = dict(state.get("answers", {}))
    missing = [k for k in REQUIRED_KEYS if k not in answers]
    state["answers"] = answers
    state["missing"] = missing
    return state

# Node: ask the next missing value from the user
@traceable
def ask_next(state: AgentState) -> AgentState:
    state["awaiting_input"] = True  # signals a pause/stop until user answers
    return state

# Node: create the performance model by calling the LLM
@traceable
def create_model(state: AgentState) -> AgentState:
    a = state.get("answers", {})
    # Prompt LLM to generate a JSON object representing the model
    prompt = (
        "Erzeuge ein JSON-Objekt für ein kompaktes Performance-Modell mit Feldern:\n"
        "workload_rps, response_size_kb, latency_p95_ms, service_count, db_type_qps, resources_estimate\n"
        "Die resources_estimate ist eine grobe Schätzung (cpu_cores,total processor utilization in all cores, ram_mb, net_mbps) basierend auf den Parametern.\n"
        f"Parameter: {a}\n"
        "Antworte NUR mit JSON.\n" 
        "db_type_qps sollte ein String wie 'Postgres 120' oder 'MongoDB 500' sein.\n"
    )
    # Call OpenAI model via LangSmith wrapper
    resp = client.responses.create(model="gpt-4o-mini", input=prompt).output_text
    # Clean up code block markers if present
    if resp.strip().startswith("```"):
        resp = resp.strip().strip("`")
        if resp.lower().startswith("json"):
            resp = resp[4:].lstrip()
    try:
        data = json.loads(resp)   # Parse JSON response
        model = PerfModel(**data) # Validate with Pydantic
        print("\n[Agent] Vorschlag Performance-Modell:\n", json.dumps(model.model_dump(), indent=2, ensure_ascii=False))
        state["model"] = model.model_dump()
    except Exception as e:
        print("\n[Agent] Fehler beim Parsen/Validieren:", e, "\nRohantwort:", resp)
    return state

# Decide whether more answers are needed or enough info is collected
def need_more(state: AgentState) -> Literal["more", "enough"]:
    return "more" if state.get("missing") else "enough"

# ---- Graph definition ----
graph = StateGraph(AgentState)
graph.add_node("ask_defaults", ask_defaults)
graph.add_node("evaluate", evaluate)
graph.add_node("ask_next", ask_next)
graph.add_node("create_model", create_model)
graph.set_entry_point("ask_defaults")
graph.add_edge("ask_defaults", "evaluate")
graph.add_conditional_edges("evaluate", need_more, {"more": "ask_next", "enough": "create_model"})
graph.add_edge("ask_next", "evaluate")
graph.add_edge("create_model", END)

# Compile the graph with MemorySaver and allow interruptions before "ask_next"
app = graph.compile(checkpointer=MemorySaver(),
                    interrupt_before=["ask_next"])

# Helper function to parse user input (key=value pairs separated by commas)
def parse_user_kv(text: str) -> Dict[str, str]:
    out = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out

# ---- Main execution ----
if __name__ == "__main__":
    cfg = {
        "configurable": {"thread_id": "gemini-loop-1"}, # unique session ID for checkpointing
        "callbacks": [tracer],                          # send traces to LangSmith
        "run_name": "PerfModel Graph",                  # trace run name
    }

    # Root run context groups the whole execution as a single session in LangSmith
    with trace(
        name="llm-agent-demo",
        project_name=os.getenv("LANGCHAIN_PROJECT"),
        inputs={"entry": "start"},
        tags=["perf-model", "langgraph"],
        metadata={"thread_id": "gemini-loop-1"},
        client=langSmithClient,       # explicit LangSmith client (optional)
    ) as root:
        state: AgentState = {"answers": {}, "messages": []}

        # Start the graph execution
        state = app.invoke(state, config=cfg)

        # Loop until all required inputs are collected or max rounds reached
        max_rounds = 10
        rounds = 0
        while rounds < max_rounds:
            missing = state.get("missing", [])
            if not missing:
                break

            # Ask user for the next missing input
            key = missing[0]
            questions = {
                "rps": "Wie viele Requests/s (rps)?",
                "resp_size_kb": "Durchschnittliche Antwortgröße (resp_size_kb)?",
                "latency_p95_ms": "Ziel-Latenz p95 (latency_p95_ms)?",
                "service_count": "Wieviele Services im kritischen Pfad (service_count)?",
                "db_type_qps": "DB-Typ und erwartete QPS (db_type_qps)?",
            }
            print(f"[Agent] {questions[key]} Bitte im Format key=value antworten.")
            user = input("\nDu: ").strip()
            if user.lower() in {"exit", "quit", "stop"}:
                break

            # Parse and update user answers
            kv = parse_user_kv(user)
            state["answers"].update(kv)
            state["awaiting_input"] = False

            # Update graph state and continue execution
            app.update_state(cfg, state)
            state = app.invoke(None, config=cfg)
            rounds += 1

        # Optionally attach final outputs explicitly to the root run
        root.end(outputs={"model": state.get("model")})
