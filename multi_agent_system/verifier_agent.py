from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel
from multi_agent import llm, AgentState

class VerifierModle(BaseModel):
    final_answer: str
    explaination: str

llm_verifier = llm.with_structured_output(VerifierModle)

def run_verifier_agent(state: AgentState) -> tuple:
    """ 
    Run the verifier agent to verify the answers provided by the other agents
    Args:
        state (AgentState): The current state of the agent containing messages and context.
    Returns:
        Tuple[str,str]:
            - final_answer (str): The verified final answer.
            - explanation (str): Explanation of the verification process.
    """
    route = state.get("route","")
    if route == "documents":
        context = state.get("docs_context","")
    else:
        context = state.get("laws_context","")
    last_message = state.get("messages",[])[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError("run_verifier_agent: last message is not an AIMessage")
    last_user_message = state.get("messages",[])[-2]
    if not isinstance(last_user_message, HumanMessage):
        raise ValueError("run_verifier_agent: second last message is not a HumanMessage")
    system_prompt = (
    f"""
    Bạn là một trợ lý kiểm định câu trả lời. Nhiệm vụ của bạn là rà soát câu trả lời của AI để đảm bảo rằng nó chính xác và tuyệt đối so với ngữ cảnh được cung cấp.
    THÔNG TIN ĐẦU VÀO:
    - Ngữ cảnh: {context}

    YÊU CẦU QUÁ TRÌNH KIỂM ĐỊNH:
    - FACT-CHECK: câu trả lời của AI có phù hợp với ngữ cảnh không? Nếu không, hãy chỉ ra những điểm sai lệch cụ thể.
    - RELEVENCE-CHECK: Câu trả lời có liên quan trực tiếp đến câu hỏi không?
    - CLARITY-CHECK: Câu trả lời có rõ ràng và dễ hiểu không?
    - POLICY-CHECK: Câu trả lời có phù hợp với các quy chuẩn đạo đức, không chứa các thông tin nhạy cảm, gây tranh cãi, ngôn từ không phù hợp không?
    
    HÃY TẠO RA CÂU TRẢ LỜI CUỐI CÙNG:
    - Nếu câu trả lời của AI là chính xác và phù hợp: Giữ nguyên câu trả lời và xác nhận tính chính xác của nó.
    - Nếu câu trả lời của AI không chính xác hoặc không phù hợp: Cung cấp câu trả lời đúng dựa trên ngữ cảnh đã cho.
    - Nếu nguồn dữ liệu không chứa thống tin câu hỏi: Thông báo rằng không thể trả lời câu hỏi dựa trên ngữ cảnh đã cho.

    KẾT QUẢ CẦN TRẢ VỀ:
    - Final Answer: câu trả lời cuối cùng
    - Explanation: giải thích ngắn gọn về quá trình kiểm định và lý do tại sao câu trả lời cuối cùng là chính xác hoặc đã được sửa đổi.
    """
    )
    system_message = AIMessage(content= system_prompt)
    messages = state.get("messages",[])
    all_messages = [system_message] + messages
    response = llm_verifier.invoke(all_messages)
    final_answer = response.final_answer
    explaination = response.explaination
    return final_answer, explaination