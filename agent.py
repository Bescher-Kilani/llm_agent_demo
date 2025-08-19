from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LLM (Gemini) via LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# LangSmith
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langsmith.run_helpers import trace  # Root-Run (Session) kapseln

# ------------------ Setup ------------------
load_dotenv()

# Optional: sanity prints
print("LangSmith Projekt:", os.getenv("LANGCHAIN_PROJECT"))
print("LANGSMITH_API_KEY gesetzt:", bool(os.getenv("LANGSMITH_API_KEY")))
print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))

# LangSmith Client (optional, aber praktisch für benannte Projekte/Sessions)
ls_client = Client()

# LangChain -> LangSmith Tracer (als Callback ins Graph-Invoke)
tracer = LangChainTracer()

# LLM (Gemini 1.5 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

prompt = "Erkläre kurz und freundlich, was Kubernetes ist."

# ------------------ Graph ------------------
def node(state: dict) -> dict:
    # Mini-Testprompt
    resp = llm.invoke(prompt)
    state["reply"] = resp.content
    return state

graph = StateGraph(dict)
graph.add_node("hello", node)
graph.set_entry_point("hello")
graph.add_edge("hello", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ------------------ Run ------------------
if __name__ == "__main__":
    # Jeder Checkpointer-Run braucht eine thread_id
    cfg = {
        "configurable": {"thread_id": "gemini-demo-run-1"},
        "callbacks": [tracer],                 # <— wichtig: schickt Runs zu LangSmith
        "run_name": "Gemini Hello Graph",      # sprechender Name im Trace
    }

    # (Optional, aber empfohlen)
    # Root-Run um die gesamte Session sauber zu gruppieren
    with trace(
        name="agent",                    # Session/Root Name
        project_name=os.getenv("LANGCHAIN_PROJECT"),   # dein Projektname
        inputs={"entry": prompt},
        tags=["demo", "langgraph", "gemini"],
        client=ls_client,
        metadata={"thread_id": "gemini-demo-run-1"},
    ) as root:
        out = app.invoke({}, config=cfg)
        print(out["reply"])
        # Optional: am Ende explizit Outputs an den Root hängen
        root.end(outputs={"reply": out.get("reply")})
