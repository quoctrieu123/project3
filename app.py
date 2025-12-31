import os
import warnings
import logging

# Thiết lập môi trường thật sớm để tắt log rác từ TF/Transformers/Accelerate
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 0=all,1=warn,2=error,3=fatal
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("ACCELERATE_LOG_LEVEL", "error")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import uuid



@st.cache_resource
def load_agent():
    from agent import agent
    return agent

agent = load_agent()


@st.cache_data
def get_agent_response(query: str, docs_context:str):
    response = agent.invoke({"query": query, "docs_context": docs_context})
    return response

# --- Cấu hình giao diện ---
st.set_page_config(page_title="Chatbot", layout="wide")

st.markdown(
    """
    <h2 style='text-align: center;'>Chatbot</h2>
    <p style='text-align: center; color: gray;'>Nhập câu hỏi của bạn, Chatbot sẽ trả lời dựa trên các luật được trích dẫn hoặc văn bản tải lên</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# --- CSS tuỳ chỉnh ---
st.markdown(
    """
    <style>
        [data-testid="stChatMessageAvatar"] {
            display: none !important;
        }

        .user-msg {
            background-color: #3a3a3a;
            color: white;
            padding: 0.8em 1em;
            border-radius: 20px;
            width: fit-content;
            margin-left: auto;
            margin-top: 0.5em;
            margin-bottom: 0.5em;
            word-wrap: break-word;
            white-space: normal;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Khởi tạo session state ---
if "chats" not in st.session_state:
    st.session_state.chats = {}  # Lưu nhiều cuộc hội thoại
if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = [
        {"role": "assistant", "content": "Xin chào bạn. Tôi sẵn sàng trả lời các câu hỏi dựa trên văn bản nhập vào hoặc các luật tôi biết."}
    ]


st.sidebar.header("🗪 Chats List")
st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader(
    "Thêm file vào hệ thống",
    type=["pdf"],
    accept_multiple_files=True
)
st.sidebar.write(f"Trong hệ thống đang lưu trữ {len(uploaded_files)} file được upload.")

# Xử lý file upload và trích xuất nội dung đưa lên session state
if uploaded_files:
    st.session_state.uploaded_files = uploaded_files
else:
    st.session_state.docs_context = ""


st.sidebar.markdown("---")
# Đảm bảo mỗi cuộc hội thoại có tên riêng
if "chat_names" not in st.session_state:
    st.session_state.chat_names = {}

# Gán tên mặc định
for cid, msgs in st.session_state.chats.items():
    if cid not in st.session_state.chat_names:
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        default_name = user_msgs[0][:30] + "..." if user_msgs else "New chat"
        st.session_state.chat_names[cid] = default_name

# --- Hiển thị danh sách hội thoại ---
selected = st.sidebar.radio(
    "Choose a chat:",
    options=list(st.session_state.chats.keys()),
    format_func=lambda cid: st.session_state.chat_names.get(cid, "New chat"),
    index=list(st.session_state.chats.keys()).index(st.session_state.current_chat),
)

# Khi chọn cuộc hội thoại khác
if selected != st.session_state.current_chat:
    st.session_state.current_chat = selected
    st.rerun()

# --- Nút tạo hội thoại mới ---
if st.sidebar.button("New chat"):
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = [
        {"role": "assistant", "content": "Xin chào bạn. Tôi sẵn sàng trả lời các câu hỏi dựa trên văn bản nhập vào hoặc các luật tôi biết."}
    ]
    st.session_state.chat_names[new_id] = "New chat"
    st.session_state.current_chat = new_id
    st.rerun()

# --- Dropdown "Tùy chọn" ---
st.sidebar.markdown("---")
st.sidebar.subheader("Options")
option = st.sidebar.selectbox(
    "Choose options",
    ["Rename", "Delete"],
    label_visibility="collapsed",
)

if option == "Rename":
    new_name = st.sidebar.text_input(
        "Input the new name:",
        value=st.session_state.chat_names[st.session_state.current_chat],
        key="rename_input",
    )
    # Đổi tên ngay lập tức khi nhập xong
    if new_name and new_name.strip() != st.session_state.chat_names[st.session_state.current_chat]:
        st.session_state.chat_names[st.session_state.current_chat] = new_name.strip()
        st.rerun()

elif option == "Delete":
    if st.sidebar.button("Delete"):
        if len(st.session_state.chats) <=1:
            st.sidebar.warning("Không thể xóa vì đây là cuộc hội thoại duy nhất")
        else:
            current = st.session_state.current_chat
            st.session_state.chats.pop(current, None)
            st.session_state.chat_names.pop(current, None)
            # Nếu xóa hết thì tạo mới
            if not st.session_state.chats:
                new_id = str(uuid.uuid4())
                st.session_state.chats[new_id] = [
                    {"role": "assistant", "content": "Xin chào bạn. Tôi sẵn sàng trả lời các câu hỏi dựa trên văn bản nhập vào hoặc các luật tôi biết."}
                ]
                st.session_state.chat_names[new_id] = "New chat"
                st.session_state.current_chat = new_id
            else:
                st.session_state.current_chat = list(st.session_state.chats.keys())[0]
            st.rerun()

# --- Hiển thị lịch sử hội thoại ---
messages = st.session_state.chats[st.session_state.current_chat]
for msg in messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(msg["content"])

# --- Ô nhập tin nhắn ---
query = st.chat_input("Ask anything")

if query:
    # Hiển thị tin nhắn người dùng
    messages.append({"role": "user", "content": query})
    st.markdown(f"<div class='user-msg'>{query}</div>", unsafe_allow_html=True)

    placeholder = st.empty()
    streamed_response = ""


    with st.spinner("Đang sinh câu trả lời..."):
        try:
            for chunk in agent.stream({"query": query, "uploaded_files": st.session_state.get("uploaded_files", [])}):
                if isinstance(chunk, str):
                    streamed_response += chunk
                elif isinstance(chunk, dict) and "content" in chunk:
                    streamed_response += chunk["content"]
                else:
                    streamed_response += str(chunk)
                placeholder.markdown(streamed_response, unsafe_allow_html=True)
        except Exception as e:
            try:
                streamed_response = agent.invoke({"query": query, "docs_context": st.session_state.get("docs_context", "")})
            except Exception as ie:
                streamed_response = f"Lỗi khi gọi mô hình: {ie}"

    # Lưu vào lịch sử hội thoại
    messages.append({"role": "assistant", "content": streamed_response})
    placeholder.markdown(streamed_response, unsafe_allow_html=True)