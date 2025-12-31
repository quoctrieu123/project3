from multi_agent import AgentState
from langchain_core.messages import AIMessage, SystemMessage
def run_laws_agent(state: AgentState) -> AIMessage:
    """
    Run the agent that answers based on retrieved laws context
    Args:
        state (AgentState): The current state of the agent containing messages and laws context.
    Returns:
        response (AIMessage): The response from the laws agent.
    """
    laws_context = state.get("laws_context", "")
    system_message_content = f"""
    Bạn là một chuyên gia về luật pháp trả lời các câu hỏi của người dùng.
    Bạn bắt buộc phải trả lời dưới cấu trúc Markdown CHỈ dựa trên dữ liệu luật (context) được cung cấp, tuyệt đối KHÔNG suy đoán.
    Nếu câu trả lời có thể dựa trên nhiều điều luật, ưu tiên liệt kê từng điều rõ ràng.
    Nếu context KHÔNG đủ dữ liệu: trả lời đúng chuỗi: "Tôi không đủ dữ kiện để đưa ra câu trả lời".
    CẤM: thêm kiến thức ngoài, ví dụ mở rộng khái niệm, suy luận xã hội, ví dụ hư cấu.
    Định dạng Markdown:
        - Tiêu đề lớn: cỡ lớn, in đậm, có thể kèm emoji.
        - Phần mục chính: in đậm.
        - Các đề mục nhỏ: in nghiêng hoặc in đậm nhẹ.
        - Nội dung: trình bày bằng đoạn văn, có thể xuống dòng giữa các ý.
        - Giữ bố cục rõ ràng, dễ đọc
        - Có thể in đậm các ý quan trọng
        Luôn mở đầu phần trả lời bằng: "Dựa trên các luật được trích dẫn:" (trừ trường hợp không đủ dữ kiện).
    Dữ liệu luật: {laws_context}
    """
    messages = state.get("messages", [])
    system_message = SystemMessage(content = system_message_content)
    all_messages = [system_message] + messages
    from multi_agent import llm
    response = llm.invoke(all_messages)
    return response