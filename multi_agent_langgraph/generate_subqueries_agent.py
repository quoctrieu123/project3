from multi_agent import AgentState
from langchain_core.messages import SystemMessage, HumanMessage
from multi_agent import llm
from pydantic import BaseModel

class SubQueryModel(BaseModel):
    sub_query_1: str
    sub_query_2: str
    sub_query_3: str
    sub_query_4: str
    sub_query_5: str

llm_subquery = llm.with_structured_output(SubQueryModel)

def run_generate_subqueries_agent(state: AgentState) -> list:
    """
    Generate sub-queries from the main query for retrieval purposes.
    Args:
        state (AgentState): The current state of the agent containing messages.
    Returns:
        subqueries (list): The list of generated sub-queries.
    """
    system_message_content = """
    Bạn là một trợ lý luật sư thông minh. Nhiệm vụ của bạn là phân tích câu hỏi của người dùng và tạo ra năm (5) câu hỏi phụ (sub-queries) liên quan đến các khía cạnh khác nhau của câu hỏi gốc. Mục đích của việc tạo các câu hỏi phụ này là để tìm kiếm thông tin pháp luật liên quan một cách hiệu quả hơn.

    YÊU CẦU BẮT BUỘC:
    1. Nếu câu hỏi gốc đã rất cụ thể, hãy trả lời năm câu hỏi phụ Y HỆT NHƯ CÂU HỎI GỐC.
    2. Nếu câu hỏi gốc rộng hoặc mơ hồ, hãy chia nó thành năm câu hỏi phụ cụ thể hơn, mỗi câu hỏi tập trung vào một khía cạnh khác nhau của chủ đề.
    Ví dụ:
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