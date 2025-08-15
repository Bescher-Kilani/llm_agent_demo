from typing import TypedDict, Dict, List, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import json

# ---- LLM (Gemini) ----
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# ---- State ----
class AgentState(TypedDict, total=False):
    messages: List[str]
    answers: Dict[str, str]
    missing: List[str]
    asked_defaults: bool
    awaiting_input: bool

DEFAULT_QUESTIONS = [
    "Welche Workloads erwartest du (Requests/Sekunde)? -> rps",
    "Durchschnittliche Antwortgröße in KB? -> resp_size_kb",
    "Ziel-Latenz p95 in Millisekunden? -> latency_p95_ms",
    "Anzahl Services im kritischen Pfad? -> service_count",
    "DB-Typ und erwartete QPS? -> db_type_qps",
]
REQUIRED_KEYS = ["rps", "resp_size_kb", "latency_p95_ms", "service_count", "db_type_qps"]

# ---- Strukturiertes Modell (Validierung) ----
class PerfModel(BaseModel):
    workload_rps: int = Field(..., ge=1)
    response_size_kb: int = Field(..., ge=1)
    latency_p95_ms: int = Field(..., ge=1)
    service_count: int = Field(..., ge=1)
    db_type_qps: str
    resources_estimate: Dict[str, float]  # z.B. {"cpu_cores": 0.5, "ram_mb": 256, "net_mbps": 5.0}

# ---- Nodes ----
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

def evaluate(state: AgentState) -> AgentState:
    answers = dict(state.get("answers", {}))
    # Nimm die letzte Nutzereingabe aus der Konsole ab (wir lesen gleich in main)
    # Hier keine Aktion – Parsing passiert separat in main, wir berechnen nur missing:
    missing = [k for k in REQUIRED_KEYS if k not in answers]
    state["answers"] = answers
    state["missing"] = missing
    return state

def ask_next(state: AgentState) -> AgentState:
    missing = state.get("missing", [])
    if not missing:
        return state
    key = missing[0]
    questions = {
        "rps": "Wie viele Requests/s (rps)?",
        "resp_size_kb": "Durchschnittliche Antwortgröße (resp_size_kb)?",
        "latency_p95_ms": "Ziel-Latenz p95 (latency_p95_ms)?",
        "service_count": "Wieviele Services im kritischen Pfad (service_count)?",
        "db_type_qps": "DB-Typ und erwartete QPS (db_type_qps)?",
    }
    print(f"[Agent] {questions[key]} Bitte im Format key=value antworten.")
    state["awaiting_input"] = True  # signalisiert: Stop/Turn-Ende
    return state

def create_model(state: AgentState) -> AgentState:
    a = state.get("answers", {})
    # LLM um JSON bitten
    prompt = (
        "Erzeuge ein JSON-Objekt für ein kompaktes Performance-Modell mit Feldern:\n"
        "workload_rps, response_size_kb, latency_p95_ms, service_count, db_type_qps, resources_estimate\n"
        "Die resources_estimate ist eine grobe Schätzung (cpu_cores, ram_mb, net_mbps) basierend auf den Parametern.\n"
        f"Parameter: {a}\n"
        "Antworte NUR mit JSON."
        "Antworte NUR mit gültigem JSON. Kein Text, keine Codeblöcke, keine ```-Markierungen."
    )
    resp = llm.invoke(prompt).content
    # Falls Codeblock-Markup vorhanden ist, entfernen:
    if resp.strip().startswith("```"):
        resp = resp.strip().strip("`")
        # evtl. auch ersten Zeilenumbruch + "json" entfernen
        if resp.lower().startswith("json"):
            resp = resp[4:].lstrip()
    try:
        data = json.loads(resp)
        model = PerfModel(**data)
        print("\n[Agent] Vorschlag Performance-Modell:\n", model.model_dump_json(indent=2, ensure_ascii=False))
        state["model"] = model.model_dump()
    except Exception as e:
        print("\n[Agent] Fehler beim Parsen/Validieren:", e, "\nRohantwort:", resp)
    return state

def need_more(state: AgentState) -> Literal["more", "enough"]:
    return "more" if state.get("missing") else "enough"

# ---- Graph ----
graph = StateGraph(AgentState)
graph.add_node("ask_defaults", ask_defaults)
graph.add_node("evaluate", evaluate)
graph.add_node("ask_next", ask_next)
graph.add_node("create_model", create_model)
graph.set_entry_point("ask_defaults")
graph.add_edge("ask_defaults", "evaluate")
graph.add_conditional_edges("evaluate", need_more, {"more": "ask_next", "enough": "create_model"})
graph.add_edge("ask_next", END)
app = graph.compile(checkpointer=MemorySaver())

# ---- Mini-CLI (Konsole) ----
def parse_user_kv(text: str) -> Dict[str, str]:
    out = {}
    for part in text.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out

if __name__ == "__main__":
    state: AgentState = {"answers": {}, "messages": []}
    # Start
    state = app.invoke(state, config={"configurable": {"thread_id": "gemini-loop-1"}})

    # Loop: bis alle REQUIRED_KEYS vorhanden sind
    max_rounds = 6
    rounds = 0
    while rounds < max_rounds:
        user = input("\nDu: ").strip()
        if user.lower() in {"exit", "quit", "stop"}:
            break
        # parse key=value,merge
        kv = parse_user_kv(user)
        state["answers"].update(kv)
        state["awaiting_input"] = False

        # ein Schritt im Graph
        state = app.invoke(state, config={"configurable": {"thread_id": "gemini-loop-1"}})
        if not state.get("missing"):
            break
        rounds += 1

    # finales Modell erzeugen
    state = app.invoke(state, config={"configurable": {"thread_id": "gemini-loop-1"}})
