"""Agent loop: LLM <-> tool execution, với MAX_ITERATIONS chống vòng lặp vô hạn."""
import json
import re
from services.openrouter_client import chat_completion
from agent.tool_schemas import ALL_TOOL_SCHEMAS
from agent.tool_registry import execute_tool
from tools.dataset_manager import DatasetManager
from config.settings import SETTINGS

EMPTY_CONTENT_MAX_RETRIES = 2

# Quan sát thật: gpt-oss (qua tunnel mới) đôi khi lộ token định dạng nội bộ của
# chính nó ra message.content thay vì được proxy/SDK parse đúng thành tool_calls —
# vd '<|start|>assistant<|channel|>commentary to=functions.search_knowledge_base
# <|constrain|>json<|message|>{"query": "..."}<|call|>'. Tái hiện được cả khi guard
# search-limit (MAX_SEARCH_KB_CALLS ở trên) CHƯA kích hoạt (chỉ 1 lần search đã gọi)
# — xác nhận đây là lỗi định dạng độc lập của model/proxy, KHÔNG phải do guard đó
# gây ra. Regex tổng quát <|...|> (không liệt kê cứng 4 token đã thấy) để bắt được
# cả biến thể token khác chưa từng quan sát.
_LEAKED_TOKEN_PATTERN = re.compile(r"<\|[a-z_]+\|>")


def _looks_like_leaked_internal_tokens(content: str) -> bool:
    """Phát hiện token nội bộ model bị lộ ra content (thay vì được parse đúng thành
    tool_calls) — dùng regex tổng quát <|...|> thay vì liệt kê cứng token đã quan
    sát, để bắt được cả biến thể chưa từng thấy."""
    return bool(_LEAKED_TOKEN_PATTERN.search(content))


# Guard CỨNG chống "search-until-exhausted" — quan sát thật: model gpt-oss (qua
# tunnel mới) lặp gọi search_knowledge_base 4-5 lần với query reformulate khác nhau
# rồi hit MAX_AGENT_ITERATIONS mà KHÔNG BAO GIỜ trả lời, dù đã tìm được vài kết quả
# từ những lần gọi trước đó (2/3 lần tái hiện thật). Chỉ sửa system prompt (nhắc mềm
# "không gọi quá 3 lần") KHÔNG đủ — vẫn fail 2/3 lần verify thật. Guard này chặn ở
# CẤU TRÚC: sau khi đã gọi search_knowledge_base đủ MAX_SEARCH_KB_CALLS lần, loại
# hẳn tool này khỏi danh sách "tools" gửi cho LLM ở vòng tiếp theo — LLM không còn
# lựa chọn nào khác ngoài dùng tool KHÁC (vd calculate cho câu hỏi "both") hoặc trả
# lời/từ chối bằng text. KHÔNG dùng tool_choice="none" (sẽ chặn LUÔN cả các tool khác
# như calculate, phá vỡ câu hỏi "both" cần search KB rồi tính toán tiếp).
MAX_SEARCH_KB_CALLS = 3


def _tools_for_next_call(trace: list[dict]) -> list[dict]:
    """Trả về ALL_TOOL_SCHEMAS như bình thường, TRỪ search_knowledge_base nếu đã bị
    gọi >= MAX_SEARCH_KB_CALLS lần trong trace TÍNH TỚI THỜI ĐIỂM HIỆN TẠI (không
    phải tổng cuối cùng — trace được truyền vào ngay trước mỗi lần gọi LLM, nên luôn
    phản ánh đúng số lần đã gọi CHO TỚI LÚC NÀY)."""
    search_kb_count = sum(1 for t in trace if t.get("tool_name") == "search_knowledge_base")
    if search_kb_count >= MAX_SEARCH_KB_CALLS:
        return [s for s in ALL_TOOL_SCHEMAS if s["function"]["name"] != "search_knowledge_base"]
    return ALL_TOOL_SCHEMAS


# Quan sat that (eval group_006, run 20260909_160757 — "So sánh độ ẩm trung bình giữa
# tháng 6 và tháng 12?"): model goi get_dataset_summary({}) 3 LAN LIEN TIEP giong het
# nhau (khong tien trien gi), roi moi goi aggregate_metric 2 lan (thang 6 + thang 12,
# THANH CONG, du du lieu de tra loi) — nhung da het max_agent_iterations (5) ngay sau
# do, roi vao cau xin loi co dinh, VUT BO 2 ket qua aggregate_metric da co that. Day
# la NHAC MEM (qua prompt) chu KHONG chan cung nhu MAX_SEARCH_KB_CALLS — hien tuong
# nay moi quan sat DUNG 1 LAN, chua du bang chung de chan cung nhu search-loop (da
# quan sat lap lai nhieu lan, du co so xay dung guard cung).
_REDUNDANT_CALL_NOTICE_TEMPLATE = (
    "Lưu ý: Bạn vừa gọi lại {tool_name} với argument giống hệt lần trước — nếu không "
    "có lý do mới, hãy dùng kết quả đã có hoặc chuyển sang bước khác."
)


def _is_redundant_call(tool_name: str, arguments: dict, trace: list[dict]) -> bool:
    """
    Trả True nếu (tool_name, arguments) NÀY đã được gọi THÀNH CÔNG (ok=True) ít nhất 1
    lần TRƯỚC ĐÓ trong trace của CÙNG lượt hỏi hiện tại (trace reset mỗi lần gọi
    run_agent_loop() mới, xem bên dưới — không rò rỉ qua các câu hỏi khác trong cùng
    session). So khớp arguments bằng dict equality (Python so sánh dict bỏ qua thứ tự
    key) — CHỈ bắt đúng trường hợp lặp Y HỆT đã quan sát thật (group_006:
    get_dataset_summary({}) x3 liên tiếp), KHÔNG cố xử lý các biến thể "tương đương
    nhưng khác kiểu" (vd "5" so với 5) — chưa có bằng chứng cần thiết cho việc đó.
    """
    return any(
        t["tool_name"] == tool_name and t["arguments"] == arguments and t["result"]["ok"]
        for t in trace
    )


# Quan sat that (case group_006 nhu tren): het max_agent_iterations sau khi DA CO du
# lieu that tu tool call thanh cong (2x aggregate_metric) van tra ve cau xin loi co
# dinh, vut bo toan bo cong da lam. FINAL_SYNTHESIS_PROMPT chi dung 1 LAN, NGOAI ngan
# sach max_agent_iterations (khong lam vong lap chinh chay them, khong doi
# MAX_AGENT_ITERATIONS hien co) — ep model tong hop tu nhung gi DA CO thay vi tiep tuc
# goi tool (chat_completion() goi KHONG kem "tools" -> model khong the goi tool nao
# nua, bat buoc tra loi bang text).
FINAL_SYNTHESIS_PROMPT = (
    "Bạn đã hết ngân sách để gọi thêm tool. Dựa vào TOÀN BỘ kết quả tool đã có trong "
    "lịch sử, hãy tổng hợp câu trả lời TỐT NHẤT CÓ THỂ, và nói rõ phần nào chưa hoàn "
    "thành nếu có."
)


def _call_with_empty_content_retry(messages: list, tools: list[dict]) -> tuple:
    """
    Gọi chat_completion(), nếu response không có tool_calls VÀ content rỗng/None
    HOẶC chứa token nội bộ bị lộ (provider trả 'thành công' nhưng nội dung vô dụng),
    tự động thử lại tối đa EMPTY_CONTENT_MAX_RETRIES lần TRƯỚC KHI trả về cho caller
    — tách biệt hoàn toàn khỏi MAX_AGENT_ITERATIONS (ngân sách đó dành cho vòng lặp
    gọi tool, không phải cho việc phục hồi từ lỗi provider).

    tools: danh sách tool schema truyền vào lần gọi này — do caller (run_agent_loop)
    tính toán qua _tools_for_next_call(), có thể đã loại search_knowledge_base nếu
    vượt ngưỡng MAX_SEARCH_KB_CALLS.

    Trả về (response, message, empty_retries_used).
    """
    empty_retries = 0
    while True:
        response = chat_completion(messages, tools=tools, tool_choice="auto")
        message = response.choices[0].message
        has_content = (
            message.content
            and message.content.strip()
            and not _looks_like_leaked_internal_tokens(message.content)
        )
        if message.tool_calls or has_content:
            return response, message, empty_retries
        if empty_retries >= EMPTY_CONTENT_MAX_RETRIES:
            return response, message, empty_retries
        empty_retries += 1
        print(f"  [empty-content-retry] Response rỗng/lộ token nội bộ, thử lại ({empty_retries}/{EMPTY_CONTENT_MAX_RETRIES})...")


def run_agent_loop(messages: list, dataset_manager: DatasetManager) -> dict:
    """
    Chạy vòng lặp agent trên messages (list đã có system + user messages).
    MUTATES messages list trực tiếp (append tool calls + results + final answer).

    Returns:
        {
            "answer": str,
            "trace": list[dict],   # mỗi entry là 1 tool call: name, arguments, result
            "iterations": int,
            "hit_max_iterations": bool,
            "usage_per_call": list[dict],  # input/output tokens mỗi lần gọi LLM, cho M8 dùng
            "empty_content_retries": int,  # số lần phải retry do provider trả content rỗng
            "graceful_termination": bool,  # True nếu hit_max_iterations NHƯNG vẫn tổng
                                            # hợp được câu trả lời thật từ tool call đã
                                            # thành công (xem FINAL_SYNTHESIS_PROMPT) —
                                            # hit_max_iterations vẫn giữ nguyên =True
                                            # trong trường hợp này (ngân sách tool-call
                                            # THẬT SỰ đã hết), chỉ thêm field này để
                                            # phân biệt "hết ngân sách nhưng có trả lời
                                            # thật" với "hết ngân sách, không trả lời
                                            # được gì" (fallback cũ)
        }
    """
    trace = []
    usage_per_call = []
    iteration = 0
    max_iter = SETTINGS.max_agent_iterations
    total_empty_retries = 0

    while iteration < max_iter:
        iteration += 1
        tools_for_this_call = _tools_for_next_call(trace)
        response, message, empty_retries = _call_with_empty_content_retry(messages, tools_for_this_call)
        total_empty_retries += empty_retries

        if response.usage:
            usage_per_call.append({
                "iteration": iteration,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            })

        if message.tool_calls:
            messages.append(message.model_dump(exclude_unset=True))

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                # Kiem tra TRUOC khi goi + TRUOC khi tu them entry nay vao trace (neu
                # khong se luon tu-khop voi chinh no).
                is_redundant = _is_redundant_call(tool_name, arguments, trace)

                result = execute_tool(tool_name, arguments, dataset_manager)

                trace.append({
                    "iteration": iteration,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

                if is_redundant:
                    messages.append({
                        "role": "user",
                        "content": _REDUNDANT_CALL_NOTICE_TEMPLATE.format(tool_name=tool_name),
                    })
            continue

        else:
            # Kiểm tra LẦN NỮA ở đây (không chỉ trong _call_with_empty_content_retry())
            # — nếu retry đã hết lượt (EMPTY_CONTENT_MAX_RETRIES) mà content VẪN rỗng
            # hoặc VẪN lộ token nội bộ, _call_with_empty_content_retry() trả message đó
            # thẳng ra cho đây, KHÔNG raise lỗi. Nếu chỉ kiểm tra content rỗng ở đây
            # (như code cũ) thì token rác (không rỗng, chỉ vô nghĩa) sẽ lọt qua nhánh
            # này và bị trả thẳng cho user ở dưới — phải kiểm tra CẢ HAI điều kiện.
            if (
                not message.content
                or not message.content.strip()
                or _looks_like_leaked_internal_tokens(message.content)
            ):
                fallback_answer = (
                    "Xin lỗi, hệ thống gặp sự cố tạm thời khi tạo câu trả lời. "
                    "Vui lòng thử hỏi lại câu hỏi này."
                )
                messages.append({"role": "assistant", "content": fallback_answer})
                return {
                    "answer": fallback_answer,
                    "trace": trace,
                    "iterations": iteration,
                    "hit_max_iterations": False,
                    "usage_per_call": usage_per_call,
                    "empty_content_retries": total_empty_retries,
                    "graceful_termination": False,
                }
            messages.append({"role": "assistant", "content": message.content})
            return {
                "answer": message.content,
                "trace": trace,
                "iterations": iteration,
                "hit_max_iterations": False,
                "usage_per_call": usage_per_call,
                "empty_content_retries": total_empty_retries,
                "graceful_termination": False,
            }

    # Het ngan sach max_agent_iterations (khong doi so nay). TRUOC KHI tra cau xin loi
    # co dinh (vut bo toan bo cong da lam — dung bug that quan sat o group_006), kiem
    # tra: da co it nhat 1 tool call THANH CONG voi du lieu that chua? Neu co, thu tong
    # hop cau tra loi tu du lieu do bang 1 lan goi LLM CUOI CUNG, NGOAI ngan sach
    # max_agent_iterations (khong lam vong lap chinh chay them buoc nao) — xem
    # FINAL_SYNTHESIS_PROMPT o tren.
    has_successful_tool_call = any(t["result"]["ok"] for t in trace)
    if has_successful_tool_call:
        messages.append({"role": "user", "content": FINAL_SYNTHESIS_PROMPT})
        synthesis_content = None
        synthesis_response = None
        try:
            # KHONG truyen "tools" -> model khong the goi tool nao nua, bat buoc tra
            # loi bang text (thay vi tool_choice="none", vi cach nay loai tool khoi
            # request hoan toan, khong phu thuoc hanh vi tool_choice cua tung provider).
            synthesis_response = chat_completion(messages)
            synthesis_message = synthesis_response.choices[0].message
            if (
                synthesis_message.content
                and synthesis_message.content.strip()
                and not _looks_like_leaked_internal_tokens(synthesis_message.content)
            ):
                synthesis_content = synthesis_message.content
        except Exception:
            # Loi khi goi lan tong hop cuoi cung (vd het retry API) -> roi xuong
            # fallback cu ben duoi, KHONG de loi nay lam crash ca Agent.ask().
            synthesis_content = None

        if synthesis_content:
            if synthesis_response.usage:
                usage_per_call.append({
                    "iteration": iteration + 1,
                    "input_tokens": synthesis_response.usage.prompt_tokens,
                    "output_tokens": synthesis_response.usage.completion_tokens,
                })
            messages.append({"role": "assistant", "content": synthesis_content})
            return {
                "answer": synthesis_content,
                "trace": trace,
                "iterations": iteration,
                "hit_max_iterations": True,
                "usage_per_call": usage_per_call,
                "empty_content_retries": total_empty_retries,
                "graceful_termination": True,
            }
        # Tong hop that bai (content rong/leaked/exception) -> bo FINAL_SYNTHESIS_PROMPT
        # vua them, khong de rac trong lich su hoi thoai, dung fallback cu ben duoi.
        messages.pop()

    fallback_answer = (
        "Xin lỗi, yêu cầu này cần nhiều bước xử lý hơn giới hạn cho phép "
        f"({max_iter} bước). Vui lòng thử chia nhỏ câu hỏi."
    )
    messages.append({"role": "assistant", "content": fallback_answer})
    return {
        "answer": fallback_answer,
        "trace": trace,
        "iterations": iteration,
        "hit_max_iterations": True,
        "usage_per_call": usage_per_call,
        "empty_content_retries": total_empty_retries,
        "graceful_termination": False,
    }
