#Viết hàm nhận vào là FileuploadedObject của Streamlit và trả về nội dung file dưới dạng chuỗi
#Biết rằng file chỉ có thể là pdf. FileUploadedObject là list các file được upload
import streamlit as st
from pypdf import PdfReader



@st.cache_data
def extract_text_from_pdfs(uploaded_files) -> str:
    """Extract text from a list of uploaded PDF files"""
    docs_context = ""
    for file in uploaded_files:
        docs_context += f"==================File: {file.name}====================" + "\n"
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            docs_context += f"------------------Page {i}--------------------" + "\n" + page.extract_text() + "\n"
    return docs_context

#thử với streamlit file uploader
"""
import streamlit as st
st.title("📂 File Uploader and Extractor")
uploaded_files = st.file_uploader(
    "Chọn một hoặc nhiều file PDF để upload",
    type=["pdf"],
    accept_multiple_files=True
)
if uploaded_files:
    extracted_text = extract_text_from_pdfs(uploaded_files)
    st.text_area("Nội dung trích xuất từ file PDF:", extracted_text, height=300)
    st.session_state.docs_context = extracted_text

#trả về văn bản tóm tắt bằng agent
from agent import agent
response = agent.invoke({"query": "Tóm tắt nội dung các file đã upload", "docs_context": st.session_state.docs_context})
st.markdown(response)    

"""