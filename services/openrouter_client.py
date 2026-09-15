"""Chat completion client qua CHAT_BASE_URL/CHAT_API_KEY (mặc định fallback về
openrouter_base_url/openrouter_api_key cũ nếu 2 biến này chưa set — xem
config/settings.py::_fallback_chat_endpoint), có retry cho lỗi tạm thời."""
import time
from openai import OpenAI
from config.settings import SETTINGS

# timeout NGẮN hơn default SDK (mặc định openai-python là 600s) — endpoint chat mới
# (proxy tunnel) có lúc trả lỗi/hang chậm hơn nhiều so với OpenRouter thật; không để
# UI treo cả 10 phút chờ 1 request duy nhất trước khi báo lỗi rõ ràng cho người dùng.
CHAT_CLIENT_TIMEOUT_SECONDS = 60.0

_client = OpenAI(
    base_url=SETTINGS.chat_base_url,
    api_key=SETTINGS.chat_api_key,
    timeout=CHAT_CLIENT_TIMEOUT_SECONDS,
)

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

# Thong bao THAN THIEN thay the khi loi cuoi cung la HTML (vd trang loi Cloudflare khi
# tunnel down/DNS error) thay vi JSON binh thuong — phat hien THAT qua browser test
# (2026-09-15): tunnel down tra ve nguyen 1 trang HTML loi Cloudflare (hang nghin ky
# tu CSS/JS), bi nhet nguyen van vao str(last_error) roi hien thang len UI chat, KHONG
# phai traceback Python nhung van la noi dung ky thuat xau, khong danh cho nguoi dung
# cuoi doc.
FRIENDLY_CONNECTION_ERROR_MESSAGE = (
    "Không thể kết nối tới máy chủ AI (lỗi mạng/hạ tầng). Vui lòng thử lại sau."
)


def _looks_like_html(text: str) -> bool:
    """Heuristic nhan dien response loi la HTML thay vi JSON/text loi binh thuong —
    KHONG dua vao content-type header (khong luon con giu lai trong message exception
    cua SDK, da xac nhan qua truong hop that: openai-python nhet nguyen response body
    vao str(exception) khong kem header). Chi can 1 trong 3 dau hieu dac trung."""
    stripped = text.strip().lower()
    return stripped.startswith("<") or "<!doctype" in stripped or "<html" in stripped


def chat_completion(
    messages: list,
    tools: list | None = None,
    tool_choice: str = "auto",
    model: str | None = None,
    temperature: float | None = None,
):
    """
    Gọi chat completion qua CHAT_BASE_URL, có retry với exponential backoff.
    Trả về response object nguyên bản của OpenAI SDK (có .choices, .usage).

    model: override model dùng cho lần gọi này (vd LLM-as-a-Judge dùng model khác
    agent chính). Mặc định None -> dùng SETTINGS.chat_model như cũ, không đổi hành vi
    hiện có ở bất kỳ nơi nào khác đang gọi hàm này.

    temperature: override nhiệt độ sampling cho lần gọi này (vd LLM-as-a-Judge dùng
    0.0 để chấm điểm nhất quán). Mặc định None -> KHÔNG truyền tham số này lên API,
    dùng default của model/provider như cũ, không đổi hành vi agent chính.

    stream=False LUÔN được truyền tường minh — endpoint tunnel đã verify thật: nếu
    thiếu field "stream" trong request body, server mặc định trả SSE streaming (data:
    {...}\\n\\n từng dòng) thay vì 1 JSON object hoàn chỉnh, khiến OpenAI SDK parse
    response sai/lỗi. OpenRouter (endpoint cũ) không có vấn đề này, nhưng set tường
    minh ở đây không đổi hành vi OpenRouter (stream=False vốn đã là default ngầm của
    SDK khi không truyền gì).
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {
                "model": model or SETTINGS.chat_model,
                "messages": messages,
                "stream": False,
                "extra_headers": {
                    "HTTP-Referer": SETTINGS.app_referer,
                    "X-Title": SETTINGS.app_title,
                },
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            return _client.chat.completions.create(**kwargs)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                print(f"  Chat completion failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    # Lỗi rõ ràng, không im lặng treo — RuntimeError này propagate lên tới
    # agent/loop.py -> Agent.ask() -> app.py's try/except (st.error(...)), không crash
    # toàn bộ Streamlit app. last_error giữ nguyên message gốc (timeout/5xx/... đọc
    # được ngay) để dễ chẩn đoán chứ không che giấu nguyên nhân thật — TRỪ trường hợp
    # response là HTML (vd trang lỗi Cloudflare khi tunnel down/DNS error): message đó
    # dài hàng nghìn ký tự CSS/JS, không đọc được gì hữu ích, KHÔNG phù hợp hiện thẳng
    # cho người dùng cuối — vẫn in đầy đủ nguyên văn ra console (docker logs bắt được)
    # để debug, chỉ THAY THẾ message trong RuntimeError (cái thực sự lên UI qua
    # app.py's except Exception as e: f"...{e}") bằng câu ngắn gọn, thân thiện.
    last_error_text = str(last_error)
    if _looks_like_html(last_error_text):
        print(
            f"  Chat completion failed after {MAX_RETRIES} attempts calling "
            f"{SETTINGS.chat_base_url} — response trông giống HTML (lỗi hạ tầng/tunnel), "
            f"chi tiết kỹ thuật đầy đủ:\n{last_error_text}"
        )
        raise RuntimeError(FRIENDLY_CONNECTION_ERROR_MESSAGE)
    raise RuntimeError(
        f"Chat completion failed after {MAX_RETRIES} attempts calling {SETTINGS.chat_base_url}: {last_error}"
    )
