from langchain_core.messages import AIMessage, HumanMessage, SystemMessage # Import SystemMessage
from pydantic import BaseModel
from multi_agent import llm, AgentState

class VerifierModle(BaseModel):
    fact_check: str
    relevance_check: str
    clarity_check: str
    policy_check: str

llm_verifier = llm.with_structured_output(VerifierModle)

def run_verifier_agent(state: AgentState) -> tuple:
    """ 
    Run the verifier agent to verify the answers provided by the other agents
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
        for msg in reversed(state.get("messages", [])[:-1]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg
                break
        else:
             raise ValueError("run_verifier_agent: No HumanMessage found")

    system_prompt = (
    f"""
    Bạn là một trợ lý kiểm định câu trả lời. Nhiệm vụ của bạn là rà soát câu trả lời của AI để đảm bảo rằng nó chính xác và tuyệt đối so với ngữ cảnh được cung cấp.
    
    THÔNG TIN ĐẦU VÀO:
    - Ngữ cảnh (Context): {context}
    - Câu hỏi của người dùng (User Query): {last_user_message.content}
    - Câu trả lời của AI (AI Answer): {last_message.content}

    YÊU CẦU QUÁ TRÌNH KIỂM ĐỊNH:
    - FACT-CHECK: Câu trả lời của AI có dựa trên ngữ cảnh không? Nếu không, hãy chỉ ra điểm sai.
    - RELEVANCE-CHECK: Câu trả lời có trả lời đúng trọng tâm câu hỏi của người dùng không?
    - CLARITY-CHECK: Câu trả lời có rõ ràng, dễ hiểu không?
    - POLICY-CHECK: Câu trả lời có vi phạm đạo đức hay chứa nội dung nhạy cảm không?
    Lưu ý: 
    - Với mỗi mục kiểm định, hãy trả lời dưới dạng giải thích ngắn gọn.
    - Đối với các câu hỏi hướng đến lịch sử hội thoại, câu trả lời sinh ra có thể dựa trên lịch sử hội thoại mà không cần dựa trên ngữ cảnh được cung cấp.
    """
    )
    
    system_message = SystemMessage(content=system_prompt)
    trigger_message = HumanMessage(content="Hãy thực hiện kiểm định dựa trên các thông tin đã cung cấp.")
    
    all_messages = [system_message, trigger_message]
    
    response = llm_verifier.invoke(all_messages)
    
    fact_check = response.fact_check
    relevance_check = response.relevance_check
    clarity_check = response.clarity_check
    policy_check = response.policy_check
    return (fact_check, relevance_check, clarity_check, policy_check)