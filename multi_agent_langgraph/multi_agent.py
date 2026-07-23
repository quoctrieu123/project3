import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# During development, expose trace inputs and outputs so the complete pipeline
# can be inspected in LangSmith. Production can opt back into masking by setting
# both values to "true" in .env.
os.environ.setdefault("LANGSMITH_HIDE_INPUTS", "false")
os.environ.setdefault("LANGSMITH_HIDE_OUTPUTS", "false")

import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()
logging.disable_progress_bar()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ACCELERATE_DISABLE_LOGGING"] = "1"


from typing import TypedDict, Annotated, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import time
import uuid

from .config import (
    GOOGLE_GENERATIVE_MODEL,
    GOOGLE_GENERATIVE_TEMPERATURE,
    PATH_TO_EMBEDDING,
    QDRANT_DOCUMENT_COLLECTION,
    QDRANT_LEGAL_COLLECTION,
)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    docs_context: str
    laws_context: str
    uploaded_files: List[str]
    route: str
    generated_subqueries: List[str]
    session_id: str


<<<<<<< HEAD:multi_agent_system/multi_agent.py
llm = ChatGoogleGenerativeAI(
    model=GOOGLE_GENERATIVE_MODEL,
    temperature=GOOGLE_GENERATIVE_TEMPERATURE,
)


def extract_text_content(content: object) -> str:
    """Return only user-visible text and discard model thinking blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        return "\n\n".join(text_parts)
    return ""
=======
# Setup path for .env file relative to the script location
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", temperature = 0.2)
>>>>>>> d04d76698cd41bf38741665b5a9a466d01239e30:multi_agent_langgraph/multi_agent.py

def setup(state: AgentState) -> AgentState:
    """Initialize the agent state"""
    print("======================== System Message ==========================")
    print(f" Hệ thống đã sẵn sàng, hiện ghi nhận {len(state.get('uploaded_files', []))} file được tải lên.")
    ai_message = AIMessage(content= "Chào bạn! Tôi là trợ lý AI, tôi sẵn sàng trả lời các câu hỏi của bạn về luật pháp hoặc tài liệu bạn cung cấp.")
    ai_message.pretty_print()
    print("======================== Human Message ==========================")
    user_input = input("Your query: ")
    user_message = HumanMessage(content = user_input)
    return {"messages": [ai_message, user_message]}


def router_agent(state: AgentState) -> str:
    """ Route the query to the appropriate agent based on the content of the query """
    a = time.time()
    from .router_agent import run_router_agent
    print("======================== Router Agent ==========================")
    route = run_router_agent(state)
    b = time.time()
    #print(f"Router agent took {b - a:.2f} seconds.")
    return {"route": route}

def route_path(state: AgentState) -> str:
    """
    Determine the path based on the route decided by the router agent.
    Args:
        state (AgentState): The current state of the agent.
    Returns:
        str: The path to take ("documents" or "extract_laws").
    """
    route = state.get("route", "")
    if route not in ["documents", "extract_laws"]:
        raise ValueError(f"route_path: Invalid route '{route}'")
    if route == "documents":
        print("Routing to documents agent.")
        return "documents"
    elif route == "extract_laws":
        print("Routing to extract laws agent.")
        return "extract_laws"
    

def generate_subqueries_agent(state: AgentState) -> AgentState:
    """Generate sub-queries from the main query for retrieval purposes."""
    a = time.time()
    from .generate_subqueries_agent import run_generate_subqueries_agent
    print("======================== Generate Subqueries Agent ==========================")
    sub_queries = run_generate_subqueries_agent(state)
    b = time.time()
    print("Generated Sub-queries:")
    for i, sub_query in enumerate(sub_queries):
        print(f"- Sub-query {i+1}: {sub_query}")
    #print(f"Generate subqueries agent took {b - a:.2f} seconds.")
    return {"generated_subqueries": sub_queries}

def retrieve_laws_agent(state: AgentState) -> AgentState:
    """Retrieve laws context based on the generated sub-queries."""
    a = time.time()
    from .retrieve_laws_agent import run_retrieve_laws_agent
    print("======================== Retrieve Laws Agent ==========================")
    laws_content = run_retrieve_laws_agent(state)
    print("Retrieved laws context sucessfully.")
    b = time.time()
    #print(f"Retrieve laws agent took {b - a:.2f} seconds.")
    return {"laws_context": laws_content}

def laws_agent(state: AgentState) -> AgentState:
    """Process the laws context and generate a response to the user's query."""
    a = time.time()
    from .laws_agent import run_laws_agent
    print("======================== Laws Agent ==========================")
    response = run_laws_agent(state)
    answer_text = extract_text_content(response.content)
    if not answer_text:
        raise ValueError("Laws agent returned no user-visible text")
    b = time.time()
<<<<<<< HEAD:multi_agent_system/multi_agent.py
    print(f"Laws agent took {b - a:.2f} seconds.")
    return {"messages": [AIMessage(content=answer_text)]}
=======
    print(response.content)
    #print(f"Laws agent took {b - a:.2f} seconds.")
    return {"messages": [response]}
>>>>>>> d04d76698cd41bf38741665b5a9a466d01239e30:multi_agent_langgraph/multi_agent.py

def extract_docs_agent(state: AgentState) -> AgentState:
    """Extract docs context related to the query from the uploaded files"""
    a = time.time()
    from .extract_docs_agent import run_docs_agent
    print("======================== Extract Documents Agent ==========================")
    docs_context = run_docs_agent(state)
    b = time.time()
    print("Extracted documents context successfully.")
    #print(f"Extract documents agent took {b - a:.2f} seconds.")
    return {"docs_context": docs_context}

def documents_agent(state: AgentState) -> AgentState:
    """Generate a response to the user's query based on the extracted documents context."""
    a = time.time()
    from .documens_agent import run_document_agent
    print("======================== Documents Agent ==========================")
    response = run_document_agent(state)
    answer_text = extract_text_content(response.content)
    if not answer_text:
        raise ValueError("Documents agent returned no user-visible text")
    b = time.time()
<<<<<<< HEAD:multi_agent_system/multi_agent.py
    print(f"Documents agent took {b - a:.2f} seconds.")
    return {"messages": [AIMessage(content=answer_text)]}
=======
    print(response.content)
    #print(f"Documents agent took {b - a:.2f} seconds.")
    return {"messages": [response]}
>>>>>>> d04d76698cd41bf38741665b5a9a466d01239e30:multi_agent_langgraph/multi_agent.py

def verifier_agent(state: AgentState) -> AgentState:
    """ Run the verifier agent to verify the answers provided by the other agents """
    a = time.time()
    from .verifier_agent import run_verifier_agent
    print("======================== Verifier Agent ==========================")
    fact_check, relevance_check, clarity_check, policy_check = run_verifier_agent(state)
    b = time.time()
<<<<<<< HEAD:multi_agent_system/multi_agent.py
    print(final_answer)
    print(f"Verifier agent took {b - a:.2f} seconds.")
    return {"messages": [AIMessage(content=final_answer)]}
=======
    print("Verification Results:")
    print(f"- Fact Check: {fact_check}")
    print(f"- Relevance Check: {relevance_check}")
    print(f"- Clarity Check: {clarity_check}")
    print(f"- Policy Check: {policy_check}")
    return {}
>>>>>>> d04d76698cd41bf38741665b5a9a466d01239e30:multi_agent_langgraph/multi_agent.py

def reasoning_agent(state: AgentState) -> AgentState:
    """ Run the reasoning agent to generate a detailed explaination about the process of generating answer for the user's query """
    a = time.time()
    from .reasoning_agent import run_reasoning_agent
    print("======================== Reasoning Agent ==========================")
    reasoning = run_reasoning_agent(state)
    b = time.time()
    print(reasoning)
    #print(f"Reasoning agent took {b - a:.2f} seconds.")
    return {}

def human_response(state: AgentState) -> AgentState:
    """User response to AI's answer (can ask further questions or end the conversation)"""
    print("======================== Human Message ==========================")
    user_input = input("Your response: ")
    user_message = HumanMessage(content=user_input)
    return {"messages": [user_message]}

def should_continue(state: AgentState) -> str:
    """ 
    Decide whether to continue the conversation or end it based on user input. 
    Args:
        state (AgentState): The current state of the agent.
    Returns:
        str: "continue" to continue the conversation, "end" to terminate it.
    """
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("should_continue : messages list is empty")
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("should_continue: last message is not a HumanMessage")
    user_input = last_message.content.lower()
    if user_input == "exit" or user_input == "quit":
        print("======================== System Message ==========================")
        print("Conversation ended.")
        return "end"
    print("======================== System Message ==========================")
    print("Conversation continues.")
    return "continue"

graph = StateGraph(state_schema= AgentState)
graph.add_node(node = "setup", action = setup)
graph.add_node(node = "router_agent", action = router_agent)
graph.add_node(node = "extract_docs_agent", action = extract_docs_agent)
graph.add_node(node = "documents_agent", action = documents_agent)
graph.add_node(node = "generate_subqueries_agent", action = generate_subqueries_agent)
graph.add_node(node = "retrieve_laws_agent", action = retrieve_laws_agent)
graph.add_node(node = "laws_agent", action = laws_agent)
graph.add_node(node = "verifier_agent", action = verifier_agent)
graph.add_node(node = "human_response", action = human_response)
graph.add_edge(START, "setup")
graph.add_edge("setup", "router_agent")
graph.add_conditional_edges(source = "router_agent", path= route_path, path_map = {
    "documents": "extract_docs_agent",
    "extract_laws": "generate_subqueries_agent"
})
graph.add_edge("extract_docs_agent", "documents_agent")
graph.add_edge("generate_subqueries_agent", "retrieve_laws_agent")
graph.add_edge("retrieve_laws_agent", "laws_agent")
graph.add_edge("documents_agent", "verifier_agent")
graph.add_edge("laws_agent", "verifier_agent")
graph.add_edge("verifier_agent", "human_response")

graph.add_conditional_edges(source = "human_response", path = should_continue, path_map = {
    "continue": "router_agent",
    "end": END
})
app = graph.compile()

<<<<<<< HEAD:multi_agent_system/multi_agent.py
def run_multi_agent_system(
    uploaded_files: list | None = None,
    *,
    cleanup_documents: bool = True,
) -> AgentState:
=======
def run_multi_agent_system() -> AgentState:
>>>>>>> d04d76698cd41bf38741665b5a9a466d01239e30:multi_agent_langgraph/multi_agent.py
    """
    Run the multi-agent system graph.
    Args:
        uploaded_files (list): List of uploaded file paths (if any)
    """
<<<<<<< HEAD:multi_agent_system/multi_agent.py
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    session_id = str(uuid.uuid4())
=======
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    tartget_folder = os.path.join(project_root, "uploaded_files")
    uploaded_files = []
    for filename in os.listdir(tartget_folder):
        file_path =  os.path.join(tartget_folder, filename)
        uploaded_files.append(file_path)
>>>>>>> d04d76698cd41bf38741665b5a9a466d01239e30:multi_agent_langgraph/multi_agent.py
    initial_state: AgentState = {
        "messages": [],
        "docs_context": "",
        "laws_context": "",
        "uploaded_files": uploaded_files or [],
        "route": "",
        "generated_subqueries": [],
        "session_id": session_id,
    }
    try:
        return app.invoke(
            initial_state,
            config={
                "recursion_limit": 1000,
                "run_name": "legal-chat-session",
                "tags": [
                    "legal-chatbot",
                    "multi-agent",
                    "qdrant",
                    "local",
                ],
                "metadata": {
                    "session_id": session_id,
                    "environment": "development",
                    "llm_model": GOOGLE_GENERATIVE_MODEL,
                    "embedding_model": PATH_TO_EMBEDDING,
                    "legal_collection": QDRANT_LEGAL_COLLECTION,
                    "document_collection": QDRANT_DOCUMENT_COLLECTION,
                    "uploaded_file_count": len(uploaded_files or []),
                },
            },
        )
    finally:
        if cleanup_documents:
            from .document_store import cleanup_session_documents

            cleanup_session_documents(session_id)

if __name__ == "__main__":
<<<<<<< HEAD:multi_agent_system/multi_agent.py
    run_multi_agent_system(
        uploaded_files=[
            r"C:\Users\Admin\Downloads\Project 3\Project code\pdf files\Cristiano Ronaldo.pdf"
        ]
    )
=======
    run_multi_agent_system()
>>>>>>> d04d76698cd41bf38741665b5a9a466d01239e30:multi_agent_langgraph/multi_agent.py
