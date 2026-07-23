"""LangGraph node for Qdrant-backed uploaded-document retrieval."""

from langchain_core.messages import HumanMessage

from .document_store import retrieve_document_context
from .multi_agent import AgentState


def run_docs_agent(state: AgentState) -> str:
    """Return relevant uploaded PDF chunks for the latest user query."""
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("run_docs_agent: messages list is empty")

    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("run_docs_agent: last message is not a HumanMessage")

    uploaded_files = state.get("uploaded_files", [])
    if not uploaded_files:
        raise ValueError("run_docs_agent: no uploaded files found in state")

    session_id = state.get("session_id", "")
    if not session_id:
        raise ValueError("run_docs_agent: session_id is missing from state")

    return retrieve_document_context(
        query=last_message.content,
        uploaded_files=uploaded_files,
        session_id=session_id,
    )

