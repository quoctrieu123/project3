from .multi_agent import AgentState, llm
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from langsmith import traceable

from .tracing import trace_agent_state_inputs, trace_subquery_outputs

class SubQueryModel(BaseModel):
    sub_query_1: str
    sub_query_2: str
    sub_query_3: str
    sub_query_4: str
    sub_query_5: str

llm_subquery = llm.with_structured_output(SubQueryModel)

@traceable(
    name="generate-legal-subqueries",
    run_type="chain",
    tags=["legal", "subqueries"],
    process_inputs=trace_agent_state_inputs,
    process_outputs=trace_subquery_outputs,
)
def run_generate_subqueries_agent(state: AgentState) -> list:
    """
    Generate sub-queries from the main query for retrieval purposes.
    Args:
        state (AgentState): The current state of the agent containing messages.
    Returns:
        subqueries (list): The list of generated sub-queries.
    """
    system_message_content = """
    Bạn là một trợ lý luật sư thông minh. Nhiệm vụ của bạn là phân tích câu hỏi của khách hàng và ĐẶT LẠI VẤN ĐỀ dưới 5 góc độ pháp lý khác nhau để tra cứu luật.

    YÊU CẦU BẮT BUỘC:
    1. KHÔNG được viết lại câu hỏi gốc (Paraphrase).
    2. Phải tách câu hỏi thành các khía cạnh nhỏ hơn (Decomposition).
    3. Mỗi câu hỏi con phải tập trung vào một từ khóa pháp lý riêng biệt, ví dụ:
    - Câu 1: Hỏi về "Điều kiện" / "Đối tượng áp dụng".
    - Câu 2: Hỏi về "Hồ sơ" / "Giấy tờ cần thiết".
    - Câu 3: Hỏi về "Trình tự" / "Thủ tục thực hiện".
    - Câu 4: Hỏi về "Thẩm quyền" / "Cơ quan giải quyết".
    - Câu 5: Hỏi về "Thời hạn" / "Chi phí" / "Chế tài xử phạt".

    Ví dụ:
    Query: "Mở quán karaoke cần gì?"
    Sub-queries Tốt:
    1. Điều kiện cấp giấy phép kinh doanh dịch vụ karaoke?
    2. Hồ sơ đề nghị cấp giấy phép đủ điều kiện an ninh trật tự cho quán karaoke?
    3. Quy định về phòng cháy chữa cháy đối với cơ sở kinh doanh karaoke?
    4. Lệ phí cấp giấy phép kinh doanh karaoke là bao nhiêu?
    5. Cơ quan nào có thẩm quyền cấp phép kinh doanh karaoke?
    """
    system_prompt = SystemMessage(content = system_message_content)
    last_message = state.get("messages", [])[-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("sub_queries_agent: last message is not a HumanMessage")
    all_message = [system_prompt, last_message]
    response = llm_subquery.invoke(all_message)
    try: 
        subqueries = [
            last_message.content,
            response.sub_query_1,
            response.sub_query_2,
            response.sub_query_3,
            response.sub_query_4,
            response.sub_query_5]
    except Exception as e:
        raise ValueError(f"sub_queries_agent: Error extracting sub-queries - {e}")
    return subqueries
