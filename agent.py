from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LLM (Gemini) via LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# LangSmith
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langsmith.run_helpers import trace  # Root-Run (session) context manager

# ------------------ Setup ------------------
load_dotenv()

# Debug prints to confirm environment variables are loaded correctly
print("LangSmith Projekt:", os.getenv("LANGCHAIN_PROJECT"))
print("LANGSMITH_API_KEY gesetzt:", bool(os.getenv("LANGSMITH_API_KEY")))
print("LANGCHAIN_TRACING_V2:", os.getenv("LANGCHAIN_TRACING_V2"))

# LangSmith client (optional, but useful for naming projects/sessions explicitly)
ls_client = Client()

# LangChain -> LangSmith tracer (used to send traces to LangSmith)
tracer = LangChainTracer()

# Define LLM (Gemini 1.5 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
)

# Prompt text
prompt = "Erkläre kurz und freundlich, was Kubernetes ist."

# ------------------ Graph ------------------
def node(state: dict) -> dict:
    # Simple test node that queries the LLM
    resp = llm.invoke(prompt)
    state["reply"] = resp.content
    return state

# Define LangGraph workflow
graph = StateGraph(dict)
graph.add_node("hello", node)   # add a node
graph.set_entry_point("hello")  # set entry point
graph.add_edge("hello", END)    # connect node to END

# MemorySaver for checkpointing
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ------------------ Run ------------------
if __name__ == "__main__":
    # Every run with a checkpointer requires a unique thread_id
    cfg = {
        "configurable": {"thread_id": "gemini-demo-run-1"},
        "callbacks": [tracer],                 # <— important: send traces to LangSmith
        "run_name": "Gemini Hello Graph",      # descriptive name for trace
    }

    # Root run to group the entire session in LangSmith
    with trace(
        name="agent",                          # root run/session name
        project_name=os.getenv("LANGCHAIN_PROJECT"),   # project name from env
        inputs={"entry": prompt},              # log initial input
        tags=["demo", "langgraph", "gemini"],  # custom tags for filtering
        client=ls_client,
        metadata={"thread_id": "gemini-demo-run-1"},   # attach metadata
    ) as root:
        out = app.invoke({}, config=cfg)       # run the graph
        print(out["reply"])                    # print LLM reply
        # Optionally attach outputs explicitly to the root run
        root.end(outputs={"reply": out.get("reply")})
