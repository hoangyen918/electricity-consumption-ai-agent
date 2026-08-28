import requests

API_KEY = "753aeb73adced5ec7bd7c76f61fba6bf77454abd049c9f42d32d72e5fbf408cc"  # giữ nguyên key thật của bạn

# Test embedding — đã xác nhận hoạt động, thử thêm câu tiếng Việt dài hơn
resp = requests.post(
    "http://74bt.ats.com.vn:8880/embedding/embed",
    headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
    json={"inputs": "Tỷ lệ hộ dân nông thôn có điện là bao nhiêu?"},
)
print("Embedding:", resp.status_code)
if resp.status_code == 200:
    vec = resp.json()[0]
    print(f"  Độ dài vector: {len(vec)}")

# Test reranker — theo đúng format Swagger đã xem (query + texts)
resp2 = requests.post(
    "http://74bt.ats.com.vn:8880/reranker/rerank",
    headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
    json={
        "query": "Tỷ lệ hộ dân nông thôn có điện là bao nhiêu?",
        "texts": [
            "99.83% of households nationwide and 99.74% of rural households have access to electricity.",
            "The World Energy Outlook projects electricity demand growth of 3% per year.",
            "Chỉ thị 20 quy định các cơ sở sản xuất phải tiết kiệm điện.",
        ],
    },
)
print("Reranker:", resp2.status_code, resp2.text[:500])