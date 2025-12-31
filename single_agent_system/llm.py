import os
import warnings
warnings.filterwarnings("ignore")

# Giảm tối đa log từ TF/Transformers/Accelerate và chặn TF hoàn toàn
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all,1=warn,2=error,3=fatal
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")    # buộc Transformers không dùng TensorFlow
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("ACCELERATE_LOG_LEVEL", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

from path_classifier import PathClassifier, label_map
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
from langchain_huggingface import HuggingFacePipeline
from config import examples, examples_qa, PATH_TO_LLM, example_subqueries, example_router
import copy
import torch
import numpy as np

# Thiết lập yên lặng các log không cần thiết
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Cấu hình tải model với cơ chế lượng tử hoá (nếu khả dụng) và fallback an toàn trên CPU
_use_4bit = False
_quantization_config = None
try:
    from transformers import BitsAndBytesConfig  # tuỳ chọn, chỉ dùng khi có bitsandbytes
    if torch.cuda.is_available():
        _quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        _use_4bit = True
except Exception:
    _use_4bit = False
    _quantization_config = None

_common_kwargs = {
    "trust_remote_code": True,
}

if _use_4bit and _quantization_config is not None:
    model = AutoModelForCausalLM.from_pretrained(
        PATH_TO_LLM,
        quantization_config=_quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        **_common_kwargs,
    )
else:
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        PATH_TO_LLM,
        device_map=device_map,
        torch_dtype=dtype,
        **_common_kwargs,
    )

tokenizer = AutoTokenizer.from_pretrained(PATH_TO_LLM, trust_remote_code=True)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.4,
    max_new_tokens=1024,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)
llm_router = copy.deepcopy(llm)
llm_subquery = copy.deepcopy(llm)
def format_prompt(input_dict: dict) -> str:
    """Format the prompt for QWEN2.5 including the 2.5B with two shot."""
    context = input_dict.get("context", "")
    query = input_dict.get("query", "")
    messages = [
        {
            "role": "system",
            "content": (
                """
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
                """           
            ),
        },
    ]

    for example in examples:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Các luật được trích dẫn: {example['context']}\n Câu hỏi: {example['query']}"
                },
                {
                    "role": "assistant",
                    "content": example["answer"]
                }
            ]
        )

    messages.append(
        {
            "role": "user",
            "content": f"Các luật được trích dẫn: {context}\n Câu hỏi: {query}",
        },
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,             
        add_generation_prompt=True  
    )

    return prompt

def format_prompt_qa(input_dict: dict) -> str:
    """Format the prompt for QWEN2.5"""
    query = input_dict.get("query", "")
    context = input_dict.get("context","")
    messages = [
        {
            "role": "system",
            "content": ("""Bạn là một trợ lý AI chỉ được phép dựa trên văn bản cung cấp (context). TUYỆT ĐỐI không thêm kiến thức ngoài.
Chế độ trả lời câu hỏi:
- Nếu đủ thông tin: Bắt đầu bằng: "Dựa trên nội dung văn bản nhập vào," rồi trả lời ngắn gọn, trích dẫn câu hoặc đoạn liên quan nếu cần.
- Nếu KHÔNG đủ: trả đúng chuỗi: "Dựa trên nội dung văn bản nhập vào, tôi không đủ dữ kiện để đưa ra câu trả lời"
Chế độ tóm tắt:
- Tạo một tóm tắt súc tích, không thêm đánh giá chủ quan.
- Định dạng Markdown: bắt đầu với: "# Văn bản được tóm tắt".
CẤM: Suy luận ngoài phạm vi, tạo ví dụ không có trong context, thêm dữ liệu nền.
"""),
        }
    ]
    for example in examples_qa:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Nội dung văn bản: {example['context']}\nCâu hỏi: {example['query']}"
                },
                {
                    "role": "assistant",
                    "content": example["answer"]
                },
            ]
        )
    messages.extend([
        {"role": "user", "content": f"Nội dung văn bản: {context}\n Câu hỏi: {query}."},
    ])
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True
    )
    return prompt

def generate_subquestion(input_dict: dict) -> list:
    """Generate up to 5 subqueries to improve retrieval robustness.

    Fallback: if parsing fails, return only the original query.
    """
    import re
    query = input_dict.get("query", "").strip()
    if not query:
        return [""]
    messages = [
        {
            "role": "system",
            "content": (
                """
Bạn là một chuyên gia về luật pháp. Bạn sẽ giúp chia nhỏ nội dung query phức tạp thành 5 query con đơn giản hơn để tìm kiếm thông tin liên quan trong cơ sở dữ liệu luật pháp.
Bạn BẮT BUỘC trả lời DƯỚI DẠNG XML theo đúng mẫu sau. Chỉ trả lại định dạng này, không được sinh ra thêm bất kỳ cái khác.
<subquery>query con 1</subquery>
<subquery>query con 2</subquery>
<subquery>query con 3</subquery>
<subquery>query con 4</subquery>
<subquery>query con 5</subquery>
"""
            ),
        }
    ]
    for example in example_subqueries:
        messages.extend(
            [
                {"role": "user", "content": example["query"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        )
    messages.append({"role": "user", "content": query})
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    response = llm_subquery.invoke(prompt)
    print(response)
    # Robust XML extraction
    matches = re.findall(r"<subquery>(.*?)</subquery>", str(response), flags=re.DOTALL)
    cleaned = [m.strip() for m in matches if m.strip()]
    if not cleaned:
        return [query]
    return [query] + cleaned[:5]


def router(input_dict: dict) -> dict:
    """Route the query to the appropriate agent based on the query content"""
    query = input_dict.get("query", "")
    messages = [
        {
            "role": "system",
            "content": ("""
Bạn là bộ định tuyến tác vụ. Dựa trên nội dung query, hãy chọn loại hành động phù hợp.
Bạn bắt buộc phải trả kết quả là 1 trong 2 từ "extract_laws" hoặc "documents". Chỉ trả lại 1 trong 2 từ này, không được trả lời thêm bất kỳ điều gì khác.
Các ví dụ được cung cấp dưới đây minh họa cách chọn loại hành động dựa trên nội dung query.

Các lựa chọn:
- extract_laws → nếu nội dung hỏi trong query có liên quan đến trích xuất luật pháp, điều luật, quy định pháp lý
- documents → nếu query không liên quan đến luật pháp hoặc câu hỏi yêu cầu dựa trên nội dung tài liệu, văn bản, file nhập vào
"""
        ),
        }
    ]
    for example in example_router:
        messages.extend(
            [
                {
                    "role":"user",
                    "content": f"Nội dung query: {example['query']}"
                },
                {
                    "role":"assistant",
                    "content": example["answer"]
                }
            ]
        )
    messages.extend(
    [
        {
            "role":"user",
            "content": f"Nội dung query: {query}"
        }
    ]
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,           
        add_generation_prompt=True  
    )
    response = llm_router.invoke(prompt)
    text = str(response).lower().strip()
    if "extract_laws" in text and "documents" not in text:
        route = "extract_laws"
    elif "documents" in text and "extract_laws" not in text:
        route = "documents"
    else:
        route = "documents" if input_dict.get("uploaded_files") else "extract_laws"
    print(f"Router chọn: {route}")
    return {"route": route, "uploaded_files": input_dict.get("uploaded_files", []), "query": input_dict.get("query","")}


def router_classifier_based(input_dict:dict) -> dict:
    """Route the query using the path classifier model"""
    from path_classifier import embedder, label_map, PathClassifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict_path = "path_classifier_model.pth"
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    model = PathClassifier()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    query = input_dict.get("query","")
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
    print(f"Router chọn (bằng classifier): {route}")
    return {"route": route, "uploaded_files": input_dict.get("uploaded_files",[]), "query": input_dict.get("query","")}

def router_matching_based(input_dict:dict) -> dict:
    """Route the query using keywork matching"""
    if input_dict.get("uploaded_files") == []:
        route = "extract_laws"
    else:
        query = input_dict.get("query","")
        query_lower = query.lower()
        law_keywords = ["luật", "điều", "mức phạt", "theo pháp luật", "quốc hội", "hành vi"]
        documents_keywords = ["văn bản", "tài liệu", "file", "đoạn văn", "bài viết"]
        if any(keyword in query_lower for keyword in documents_keywords) and not any(keyword in query_lower for keyword in law_keywords):
            route = "documents"
        elif any(keyword in query_lower for keyword in law_keywords) and not any(keyword in query_lower for keyword in documents_keywords):
            route = "extract_laws"
        else:
            route = "documents"
    print(f"Router chọn (bằng matching-based router): {route}")
    return {"route": route, "uploaded_files": input_dict.get("uploaded_files",[]), "query": input_dict.get("query","")}

def distance_to_sim(index, distance):
    metric_type = index.metric_type
    if metric_type == 1:  # faiss.METRIC_L2
        return -distance
    elif metric_type == 0:  # faiss.METRIC_INNER_PRODUCT
        return distance
    else:
        print("Không xác định được kiểu metric của index, mặc định dùng L2")
        return -distance

def router_embedding_similarity(input_dict:dict, top_k:int = 1) -> dict:
    """Route the query by comparing embedding-similarity against the laws index and uploaded-files index."""
    from index import get_laws_index_and_json, get_faiss_index_for_uploads, get_embedder
    laws_index, laws = get_laws_index_and_json()
    uploaded_files = input_dict.get("uploaded_files", [])
    if uploaded_files:
        keywords = ["tóm tắt", "tổng hợp", "tóm tắt văn bản", "tóm tắt nội dung", "ngắn gọn nội dung"]
        if any(keyword in input_dict.get("query", "").lower() for keyword in keywords):
            route = "documents"
        else:
            docs_index, _, _ = get_faiss_index_for_uploads(uploaded_files)
            emb = get_embedder()
            query_embedding = emb.encode([input_dict.get("query","")])
            D_laws, I_laws = laws_index.search(query_embedding, k = top_k)
            D_docs, I_docs = docs_index.search(query_embedding, k = top_k)
            D_docs_avg = np.mean(D_docs)
            D_laws_avg = np.mean(D_laws)
            D_docs_sim = distance_to_sim(docs_index, D_docs_avg)
            D_laws_sim = distance_to_sim(laws_index, D_laws_avg)
            print(f"Similarity với luật: {D_laws_sim}, similarity với tài liệu: {D_docs_sim}")
            if D_docs_sim > D_laws_sim:
                route = "documents"
            elif abs(D_docs_sim - D_laws_sim) < 0.01:
                route = "documents"
            else:
                route = "extract_laws"
    else:
        route = "extract_laws"
    print(f"Router chọn (bằng embedding-similarity): {route}")
    return {"route": route, "uploaded_files": input_dict.get("uploaded_files",[]), "query": input_dict.get("query","")}