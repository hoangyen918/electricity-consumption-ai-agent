"""
Test retrieval bằng các câu hỏi có ground truth đã verify thủ công.
Các test này GỌI API THẬT (embedding) — chi phí rất nhỏ (~5 query x vài trăm token).
"""
import pytest
from rag.retriever import retrieve, search_knowledge_base


# Mỗi case: (câu hỏi tiếng Việt, source phải chứa, page phải đúng)
VERIFIED_CASES = [
    (
        "Tỷ lệ hộ dân nông thôn có điện là bao nhiêu?",
        "2024-2025",
        9,
    ),
    (
        "Tổng sản lượng điện thương phẩm năm 2023 là bao nhiêu?",
        "2022-2023",
        10,
    ),
    (
        "Tỷ lệ tổn thất điện năng của EVN là bao nhiêu?",
        "2022-2023",
        10,
    ),
]


@pytest.mark.parametrize("question,expected_source_fragment,expected_page", VERIFIED_CASES)
def test_verified_fact_appears_in_topk(question, expected_source_fragment, expected_page):
    """
    Với câu hỏi có ground truth đã verify, nguồn đúng phải xuất hiện
    trong top-k kết quả (dùng default SETTINGS.rag_top_k, không bắt buộc top-1,
    vì đã biết có cross-lingual gap).
    """
    results = retrieve(question)
    assert len(results) > 0, f"No results returned for: {question}"

    found = any(
        expected_source_fragment in r["source"] and r["page"] == expected_page
        for r in results
    )
    assert found, (
        f"Expected source containing '{expected_source_fragment}' page {expected_page} "
        f"not found in top-{len(results)} for question: {question}\n"
        f"Got: {[(r['source'], r['page'], r['distance']) for r in results]}"
    )


def test_empty_query_returns_empty_list():
    assert retrieve("") == []
    assert retrieve("   ") == []


def test_search_knowledge_base_returns_ok_format():
    result = search_knowledge_base("Tỷ lệ tổn thất điện năng là bao nhiêu?")
    assert result["ok"] is True
    assert "results" in result["data"]
    assert len(result["data"]["results"]) > 0
    # Mỗi result phải có citation
    for r in result["data"]["results"]:
        assert "citation" in r
        assert r["citation"].startswith("[")


def test_search_knowledge_base_nonsense_query_still_returns_something_or_fails_gracefully():
    """Query vô nghĩa vẫn phải trả về structure hợp lệ, không crash."""
    result = search_knowledge_base("xyzabc123 completely unrelated nonsense query")
    assert result["ok"] in (True, False)  # không crash, dù có tìm ra gì hay không
    if not result["ok"]:
        assert "error" in result or result.get("error") is not None
