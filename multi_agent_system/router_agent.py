# nâng cấp thư mục lên một bậc để import các modul trong thư mục cha
import os
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from .multi_agent import AgentState
from langchain_core.messages import HumanMessage
from classifier_based_router_training import embedder, label_map, PathClassifier
import torch
from langsmith import traceable

from .tracing import trace_router_inputs, trace_router_outputs


DOCUMENT_QUERY_SIGNALS = (
    "file",
    "pdf",
    "tài liệu",
    "văn bản",
    "đính kèm",
    "tải lên",
    "upload",
    "trong tài liệu",
    "trong văn bản",
    "tài liệu này",
    "văn bản này",
)

_router_model = None
_router_device = None


def get_router_model():
    """Load the classifier once instead of once per conversation turn."""
    global _router_model, _router_device
    if _router_model is None:
        _router_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / "path_classifier_model.pth"
        )
        state_dict = torch.load(
            state_dict_path,
            map_location=_router_device,
            weights_only=True,
        )
        _router_model = PathClassifier()
        _router_model.load_state_dict(state_dict)
        _router_model.to(_router_device)
        _router_model.eval()
    return _router_model, _router_device


@traceable(
    name="route-query",
    run_type="chain",
    tags=["router", "multi-agent"],
    process_inputs=trace_router_inputs,
    process_outputs=trace_router_outputs,
)
def run_router_agent(state: AgentState) -> str:
    """ 
    Run the path_classifier model to decide which path to take based on the query content.
    Args:
        state (AgentState): The current state of the agent containing messages.
    Returns:
        route (str): The decided route ("documents" or "extract_laws").
    """
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("run_router_agent: messages list is empty")
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("run_router_agent: last message is not a HumanMessage")
    query = last_message.content.lower()

    uploaded_files = state.get("uploaded_files", [])
    if not uploaded_files:
        return "extract_laws"

    if any(signal in query for signal in DOCUMENT_QUERY_SIGNALS):
        return "documents"

    model, device = get_router_model()
    embedding = embedder.encode([query])
    input_tensor = torch.tensor(embedding, dtype=torch.float32)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output,dim=1).item()
        if predicted_class == label_map.get("documents"):
            route = "documents"
        elif predicted_class == label_map.get("extract_laws"):
            route = "extract_laws"
    return route
