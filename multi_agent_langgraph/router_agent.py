# nâng cấp thư mục lên một bậc để import các modul trong thư mục cha
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multi_agent_langgraph.multi_agent import AgentState
from langchain_core.messages import HumanMessage
from classifier_based_path_classifier_training import embedder, label_map, PathClassifier
import torch
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
    law_keywords = ["luật", "điều", "mức phạt", "theo pháp luật", "quốc hội", "hành vi", "nghị định", "thông tư"]
    documents_keywords = ["văn bản", "tài liệu", "file", "đoạn văn", "bài viết"]
    if any(keyword in query for keyword in documents_keywords) and not any(keyword in query for keyword in law_keywords):
        route = "documents"
    elif any(keyword in query for keyword in law_keywords) and not any(keyword in query for keyword in documents_keywords):
        route = "extract_laws"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use relative path for model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        state_dict_path = os.path.join(project_root, "models", "path_classifier_model.pth")
        
        state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
        model = PathClassifier()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
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
