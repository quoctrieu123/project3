from multi_agent import AgentState
from langchain_core.messages import AIMessage, SystemMessage

def run_document_agent(state: AgentState) -> AIMessage:
    """
    Run the agent that answers based on retrieved document context
    Args:
        state (AgentState): The current state of the agent containing messages and document context.
    Returns:
        response (AIMessage): The response from the document agent.
    """
    from multi_agent import llm
    docs_context = state.get("docs_context", "")
    if not docs_context:
        raise ValueError("run_document_agent: docs_context is empty")
    """Run the agent that answers based on retrieved document context"""
    system_message_content = f"""
    Bạn là một trợ lý AI chỉ được phép dựa trên văn bản cung cấp (context). TUYỆT ĐỐI không thêm kiến thức ngoài.
    Chế độ trả lời câu hỏi:
    - Nếu đủ thông tin: Bắt đầu bằng: "Dựa trên nội dung văn bản nhập vào," rồi trả lời ngắn gọn, trích dẫn câu hoặc đoạn liên quan nếu cần.
    - Nếu KHÔNG đủ: trả đúng chuỗi: "Dựa trên nội dung văn bản nhập vào, tôi không đủ dữ kiện để đưa ra câu trả lời"
    Chế độ tóm tắt:
    - Tạo một tóm tắt súc tích, không thêm đánh giá chủ quan.
    - Định dạng Markdown: bắt đầu với: "# Văn bản được tóm tắt".
    CẤM: Suy luận ngoài phạm vi, tạo ví dụ không có trong context, thêm dữ liệu nền.
    Nội dung văn bản: {docs_context}
    """
    system_message = SystemMessage(content = system_message_content)
    messages = state.get("messages", [])
    all_messages = [system_message] + messages
    response = llm.invoke(all_messages)
    return response
    