# Demo Notes

## Câu hỏi data — ví dụ
"Zone 1 trung bình tháng 7 là bao nhiêu?" → gọi tool aggregate_metric(column="zone_1_consumption",
aggregation="mean", start_date="2017-07-01", end_date="2017-07-31") → trả lời: khoảng **35.831,55**
(đo thật từ case `data_001`, `evals/results/eval_detail_20260901_011250.json`, dataset đầy đủ
`data/processed/tetouan_clean.csv`, tháng 7/2017).

## Câu hỏi KB — ví dụ
"Tỷ lệ hộ dân nông thôn có điện của EVN là bao nhiêu?" → gọi search_knowledge_base → trả lời
99.74% kèm trích dẫn [EVN Annual Report 2024-2025, trang 9].

## Câu hỏi ngoài phạm vi KB — ví dụ
"Zone 5 trung bình bao nhiêu?" → agent từ chối, giải thích rõ Zone 5 không tồn tại trong dataset,
liệt kê các cột thật có sẵn — không tự bịa số.

## Cách demo dynamic KB upload (Phần B)
1. Mở sidebar Streamlit, phần "Thêm tài liệu mới vào KB"
2. Upload 1 file PDF/TXT/MD bất kỳ (mentor tự chọn), đặt tên hiển thị
3. Bấm "Nạp vào Knowledge Base" — thấy số đoạn (chunks) mới xuất hiện ngay trong danh sách
4. Hỏi 1 câu liên quan tới nội dung vừa nạp — agent trả lời có trích dẫn đúng nguồn mới
