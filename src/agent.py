"""LangGraph agent: the LLM decides which tools to call, the graph runs them
and loops back until the LLM gives a final answer.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END

A checkpointer stores each conversation (thread) so follow-up questions like
"what about the extension?" remember what was discussed.
"""
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src import config
from src.rag import to_text
from src.tools import TOOLS

SYSTEM_PROMPT = """You are a tax form assistant for US state tax forms.

Rules:
- For any question about form rules, dates, lines, forms or credits, call
  search_tax_instructions first. Never answer these from your own knowledge.
- Never say you couldn't find something without searching first. If the first
  search misses, try one more search with different words.
- Only call list_loaded_forms when the user asks what forms are available.
- Cite every fact from a search using the label shown above the passage,
  copied exactly, e.g. [UT TC-40-instructions, p.4-5]. Cite the passage the
  fact actually came from.
- Answer only for the tax year covered by the loaded forms. If asked about a
  different year, say the loaded forms only cover that year and do not
  predict other years.
- Name specific form numbers (e.g. TC-547) whenever the passage gives one.
- For any arithmetic, call calculate. Never do math yourself.
- To check a scanline or check digit, call validate_check_digit.
- If the search does not contain the answer, say "I couldn't find that in the
  loaded forms" instead of guessing.
- Use earlier messages in the conversation to understand follow-up questions.
- Keep answers short and clear."""

# Stop runaway tool loops (each loop step uses an LLM request).
MAX_STEPS = 10


def build_graph(llm=None, checkpointer=None):
    """Build the agent graph. `llm` can be swapped for testing."""
    llm = llm or init_chat_model(config.LLM_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: MessagesState):
        messages = [SystemMessage(SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # tools, or END
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or MemorySaver())


@lru_cache(maxsize=1)
def get_agent():
    return build_graph()


def run(question: str, thread_id: str = "default", agent=None) -> dict:
    """Ask the agent a question within a conversation thread.
    Returns the answer and the tools it used, in order."""
    agent = agent or get_agent()
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_STEPS * 2}

    before = len(agent.get_state(cfg).values.get("messages", []))
    result = agent.invoke({"messages": [("user", question)]}, cfg)
    new_messages = result["messages"][before:]

    tools_used = [
        f"{call['name']}({', '.join(f'{k}={v!r}' for k, v in call['args'].items())})"
        for msg in new_messages
        for call in getattr(msg, "tool_calls", []) or []
    ]
    return {"answer": to_text(result["messages"][-1].content), "tools_used": tools_used}