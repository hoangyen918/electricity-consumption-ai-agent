from unittest.mock import MagicMock, patch

from services.openrouter_client import (
    chat_completion,
    _looks_like_html,
    FRIENDLY_CONNECTION_ERROR_MESSAGE,
)

# Mau HTML loi Cloudflare THAT (rut gon) da gap qua browser test 2026-09-15 khi tunnel
# down — dung DUNG dang that de test khong giả định sai cấu trúc.
REAL_CLOUDFLARE_ERROR_HTML = (
    "<!DOCTYPE html><html><head><title>colony-rain-stud-assume.trycloudflare.com | "
    "522: Connection timed out</title></head><body><div id=\"cf-wrapper\">"
    "<div id=\"cf-error-details\"><h1>Error <span>1016</span></h1>"
    "<h2>Origin DNS error</h2><p>You've requested a page on a website "
    "(colony-rain-stud-assume.trycloudflare.com) that is on the Cloudflare network.</p>"
    "</div></div></body></html>"
)


def test_looks_like_html_detects_real_cloudflare_error_page():
    assert _looks_like_html(REAL_CLOUDFLARE_ERROR_HTML) is True


def test_looks_like_html_detects_html_with_leading_whitespace():
    assert _looks_like_html("   \n<html><body>lỗi</body></html>") is True


def test_looks_like_html_rejects_normal_error_text():
    assert _looks_like_html("Connection timed out after 60s") is False
    assert _looks_like_html("Error code: 502 - {'error': 'Bad Gateway'}") is False


def test_chat_completion_replaces_html_error_with_friendly_message(capsys):
    """Khi loi cuoi cung (sau het retry) la HTML -> RuntimeError phai la thong bao
    THAN THIEN, KHONG chua HTML goc — nhung HTML goc VAN phai in ra console (log)."""
    with patch("services.openrouter_client._client") as mock_client, \
         patch("services.openrouter_client.time.sleep"):  # bo qua backoff that trong test
        mock_client.chat.completions.create.side_effect = RuntimeError(REAL_CLOUDFLARE_ERROR_HTML)

        try:
            chat_completion(messages=[{"role": "user", "content": "hi"}])
            assert False, "Ky vong RuntimeError nhung khong co exception nao duoc raise"
        except RuntimeError as e:
            assert str(e) == FRIENDLY_CONNECTION_ERROR_MESSAGE, (
                f"UI-facing message PHAI la thong bao than thien, thuc te: {e}"
            )
            assert "<!DOCTYPE" not in str(e) and "<html" not in str(e), (
                "HTML goc KHONG duoc lot vao message hien thi UI"
            )

    captured = capsys.readouterr()
    assert "Origin DNS error" in captured.out, (
        "Chi tiet ky thuat (bao gom HTML goc) PHAI con trong console/log de debug sau nay"
    )
    assert REAL_CLOUDFLARE_ERROR_HTML in captured.out


def test_chat_completion_keeps_original_message_for_non_html_error(capsys):
    """Loi KHONG phai HTML (vd timeout binh thuong) van giu nguyen message goc nhu
    truoc — KHONG bi anh huong boi thay doi nay (dung yeu cau 'khong dung logic
    retry/timeout da co')."""
    with patch("services.openrouter_client._client") as mock_client, \
         patch("services.openrouter_client.time.sleep"):
        mock_client.chat.completions.create.side_effect = TimeoutError("Connection timed out after 60s")

        try:
            chat_completion(messages=[{"role": "user", "content": "hi"}])
            assert False, "Ky vong RuntimeError"
        except RuntimeError as e:
            assert "Connection timed out after 60s" in str(e), (
                f"Loi khong phai HTML phai giu nguyen message goc de de chan doan, thuc te: {e}"
            )
            assert str(e) != FRIENDLY_CONNECTION_ERROR_MESSAGE
