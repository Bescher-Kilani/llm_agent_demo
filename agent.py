from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# WICHTIG: Gemini-Adapter für LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# LLM (Gemini 1.5 Flash ist schnell + gutes Free Tier)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # alternativ: "gemini-1.5-pro"
    temperature=0.2
    # Der GOOGLE_API_KEY wird automatisch aus der Env-Variable gelesen.
)

def node(state: dict) -> dict:
    # Mini-Testprompt
    resp = llm.invoke("erkläre aus fühlrich das")
    state["reply"] = resp.content
    return state

# LangGraph: einfacher Ein-Schritt-Workflow
graph = StateGraph(dict)
graph.add_node("hello", node)
graph.set_entry_point("hello")
graph.add_edge("hello", END)

# Checkpointing bleibt gleich (MemorySaver)
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    # Neuere LangGraph-Versionen brauchen eine thread_id, wenn ein Checkpointer aktiv ist
    out = app.invoke({}, config={"configurable": {"thread_id": "gemini-demo-run-1"}})
    print(out["reply"])
