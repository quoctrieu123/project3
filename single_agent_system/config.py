PATH_TO_LLM = "Qwen/Qwen2.5-3B-Instruct"
PATH_TO_EMBEDDING = "intfloat/multilingual-e5-large"

"""Các ví dụ few-shot dùng cho luồng luật. Thêm context_1, context_2 để mô hình học TRẢ LỜI CHỈ DỰA TRÊN LUẬT.
Lưu ý: Nội dung dưới đây chỉ là trích lược / mô phỏng súc tích, không thay thế văn bản pháp lý đầy đủ.
"""

# Ví dụ 1: xử lý khi điều khiển xe máy đã uống rượu
query_1 = "Các mức phạt khi điều khiển xe máy mà đã uống rượu"
context_1 = (
    "Điều 13 BLHS 2015: Người phạm tội do mất khả năng nhận thức hoặc điều khiển hành vi vì rượu bia vẫn phải chịu trách nhiệm hình sự.\n"
    "Điều 260 BLHS 2015: Vi phạm quy định về tham gia giao thông đường bộ gây thiệt hại cho người khác có thể bị phạt tù.\n"
    "Nghị định 100/2019/NĐ-CP Điều 6: Mức phạt hành chính đối với người điều khiển xe mô tô có nồng độ cồn vượt quá quy định."
)
answer_1 = """
# Mức xử lý khi điều khiển xe máy đã uống rượu 🍺

**Căn cứ pháp lý trích dẫn:**
- Điều 13 BLHS 2015: Người uống rượu gây mất khả năng nhận thức vẫn chịu trách nhiệm hình sự.
- Điều 260 BLHS 2015: Vi phạm quy định giao thông gây hậu quả nghiêm trọng có thể bị truy cứu hình sự.
- Nghị định 100/2019/NĐ-CP Điều 6: Quy định mức phạt hành chính theo nồng độ cồn.

**Tóm tắt xử lý:**
1. Phạt hành chính (phạt tiền, tước GPLX) nếu chỉ vi phạm nồng độ cồn.
2. Truy cứu hình sự nếu gây hậu quả nghiêm trọng (thiệt hại về người/tài sản) theo Điều 260.
3. Trách nhiệm hình sự không được miễn trừ do đã uống rượu (Điều 13).

**Kết luận:** Người điều khiển xe máy đã uống rượu vẫn phải chịu trách nhiệm; mức độ xử lý phụ thuộc mức độ vi phạm và hậu quả gây ra.
"""

# Ví dụ 2: giải thích về Quốc hội Việt Nam
query_2 = "Hãy giải thích chi tiết về Quốc Hội Việt Nam?"
context_2 = (
    "Hiến pháp 2013 Điều 69: Quốc hội là cơ quan đại biểu cao nhất của nhân dân, cơ quan quyền lực nhà nước cao nhất.\n"
    "Hiến pháp 2013 Điều 70: Quốc hội có thẩm quyền lập hiến, lập pháp, quyết định các vấn đề quan trọng của đất nước và giám sát tối cao hoạt động của Nhà nước."
)
answer_2 = """
# Quốc hội Việt Nam 🇻🇳

**Căn cứ pháp lý trích dẫn:**
- Điều 69 Hiến pháp 2013: Quốc hội là cơ quan đại biểu cao nhất của nhân dân, quyền lực nhà nước cao nhất.
- Điều 70 Hiến pháp 2013: Quyền lập hiến, lập pháp; quyết định chính sách cơ bản; giám sát tối cao.

**Vai trò chính:** Quyết định đường lối phát triển, tổ chức bộ máy nhà nước, phê duyệt ngân sách, giám sát Chính phủ và các thiết chế tư pháp.

**Chức năng cốt lõi:** Lập hiến – lập pháp; giám sát tối cao; quyết định nhân sự chủ chốt (Chủ tịch nước, Thủ tướng, Chánh án, Viện trưởng). 

**Kết luận:** Quốc hội giữ vị trí trung tâm trong hệ thống chính trị, bảo đảm pháp luật và chính sách phản ánh ý chí nhân dân.
"""

examples = []
for q, a, c in [
    (query_1, answer_1, context_1),
    (query_2, answer_2, context_2),
]:
    examples.append({"query": q, "answer": a, "context": c})



query_qa_1 = "Nền kinh tế của nước nào bị ảnh hưởng mạnh bởi Mỹ vào năm 2025?"
context_qa_1 = "Vào năm 2025, Mỹ áp đặt một nền ảnh hưởng lớn lên nền kinh tế Trung Quốc."
answer_qa_1 = "Dựa trên nội dung văn bản nhập vào, nền kinh tế Trung Quốc bị ảnh hưởng bởi Mỹ vào năm 2025."

query_qa_3 = "Trái đất màu gì?"
context_qa_3 = "Vào năm 2025, nền kinh tế có biến động mạnh với sự ảnh hưởng của Mỹ."
answer_qa_3 = "Dựa trên nội dung văn bản nhập vào, tôi không đủ dữ kiện để đưa ra câu trả lời"

query_qa = [query_qa_1, query_qa_3]
answer_qa = [answer_qa_1, answer_qa_3]
context_qa = [context_qa_1, context_qa_3]
examples_qa = []
for answer,query, context in zip(answer_qa, query_qa,context_qa):
    example = {
        "query": query,
        "context": context,
        "answer": answer
    }
    examples_qa.append(example)


query_to_split_1 = "săn bắt trái phép"
answer_subqueries = """
<subquery>Quy định về hành vi săn bắt động vật trái phép</subquery> 
<subquery>Mức xử phạt hành chính đối với hành vi săn bắt trái phép</subquery> 
<subquery>Trách nhiệm hình sự tội săn bắt động vật trái phép</subquery> 
<subquery>Săn bắt động vật hoang dã nguy cấp quý hiếm bị phạt như thế nào</subquery> 
<subquery>Các trường hợp nào bị coi là săn bắt trái phép</subquery>
"""

query_to_split_2 = "Các yếu tố chính ảnh hưởng đến biến đổi khí hậu và các biện pháp mà các quốc gia có thể thực hiện để giảm thiểu tác động của nó là gì?"
answer_subqueries_2 = """
<subquery>Những yếu tố chính nào ảnh hưởng đến biến đổi khí hậu?</subquery>
<subquery>Làm thế nào các quốc gia có thể giảm thiểu tác động của biến đổi khí hậu?</subquery>
<subquery>Vai trò của năng lượng tái tạo trong việc giảm thiểu biến đổi khí hậu là gì?</subquery>
<subquery>Chính sách quốc tế nào hỗ trợ các nỗ lực chống biến đổi khí hậu?</subquery>
<subquery>Tác động kinh tế của các biện pháp giảm thiểu biến đổi khí hậu là gì?</subquery>
"""
example_subqueries = []
for query, answer_subqueries in zip([query_to_split_1, query_to_split_2], [answer_subqueries, answer_subqueries_2]):
    example = {
        "query": query,
        "answer": answer_subqueries
    }
    example_subqueries.append(example)

query_router_1 = "Hành vi vi phạm hợp đồng được quy định như thế nào?"
answer_router_1 = "extract_laws"

query_router_2 = "Thủ tục cấp giấy phép mang vũ khí thể thao để luyện tập, thi đấu thể thao; triển lãm, trưng bày, chào hàng, giới thiệu sản phẩm; làm đạo cụ trong hoạt động văn hóa, nghệ thuật thì thực hiện như thế nào?"
answer_router_2 = "extract_laws"

query_router_3 = "Em của tôi đã mất, nay liên quan đến việc phân chia di sản thừa kế của gia đình cần phải có giấy xác nhận việc em tôi còn độc thân. Vậy UBND cấp xã có thẩm quyền cấp giấy này cho em tôi không?"
answer_router_3 = "extract_laws"

query_router_4 = "Tóm tắt văn bản"
answer_router_4 = "documents"

query_router_5 = "Hoa hậu Việt Nam 2023 được tổ chức ở đâu?"
answer_router_5 = "documents"

example_router = []
for query, answer in zip([query_router_4, query_router_5, query_router_1, query_router_2, query_router_3], [answer_router_4, answer_router_5, answer_router_1, answer_router_2, answer_router_3]):
    example = {
        "query": query,
        "answer": answer
    }
    example_router.append(example)