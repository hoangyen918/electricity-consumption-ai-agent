"""Đề xuất ánh xạ cột của 1 file LẠ (chưa từng thấy) sang 4 khái niệm chuẩn turbine
gió (wind_speed/power/pitch/yaw_or_direction) — THUẦN rule-based (tên cột + giá trị),
KHÔNG gọi LLM/API nào. Lý do chọn rule-based thay vì để agent tự suy luận: nhanh,
miễn phí, tái lập được (deterministic), không phụ thuộc tunnel/credit (đã gây gián
đoạn nhiều lần) — đây là bước RIÊNG, độc lập, chạy xong mới tới bước con người xác
nhận qua UI, KHÁC với cách agent tự suy luận khi trả lời câu hỏi.

PHẠM VI: chỉ detect + đề xuất + lưu ánh xạ đã xác nhận. KHÔNG nối ánh xạ đã xác nhận
vào bất kỳ tool/tab nào trong số 22 tool/13 tab hiện có — đó là việc tích hợp LỚN
RIÊNG, để dành làm sau (xem ghi chú cuối commit)."""
import hashlib
import json
from pathlib import Path

from tools.dataset_manager import DatasetManager
from tools.dataframe_tools import get_dataset_summary

MAPPINGS_DIR = Path("data/column_mappings")

# Không cần đầy đủ tuyệt đối — nếu tên KHÔNG khớp gì cả, vẫn xử lý được qua bước
# value-based (_value_match()), đây chỉ là gợi ý đầu vào.
KNOWN_COLUMN_SYNONYMS = {
    "wind_speed": ["wind_speed", "windspeed", "wind speed", "v_wind", "ws", "anemometer"],
    "power": ["power", "active_power", "power_output", "grid_power", "totw", "kw", "output"],
    "pitch": ["pitch", "blade_pitch", "pitch_angle"],
    "yaw_or_direction": ["yaw", "direction", "nacelle", "wind_dir"],
}

# Ngưỡng vật lý HEURISTIC TỰ ĐỊNH NGHĨA — KHÔNG có nguồn tài liệu chính thức xác nhận
# (tương tự cách PITCH_MISMATCH_DEG/CUT_IN_SPEED_MS trong turbine_diagnostic_tools.py
# đã làm), cần domain expert review trước khi dùng cho quyết định quan trọng. Chỉ
# dùng min/max (field "numeric_ranges" của get_dataset_summary(), đã CÓ SẴN) — không
# tính lại percentile/mean riêng cho module này.
#   min_floor: giá trị MIN của cột phải >= số này (loại cột âm quá sâu).
#   max_range: (thấp, cao) — giá trị MAX của cột phải nằm trong khoảng này (cao=None
#     nghĩa là không có trần trên, vd "power" vì turbine to/nhỏ khác nhau rất nhiều).
_VALUE_RANGE_RULES = {
    "wind_speed": {"min_floor": -2.0, "max_range": (10.0, 70.0)},
    "power": {"min_floor": -2000.0, "max_range": (200.0, None)},
    "pitch": {"min_floor": -30.0, "max_range": (5.0, 100.0)},
    "yaw_or_direction": {"min_floor": -1.0, "max_range": (100.0, 370.0)},
}

# Nếu ứng viên "gần đúng nhất còn lại" (LOW) vẫn lệch quá xa khoảng hợp lý, báo
# UNKNOWN thay vì đề xuất 1 cột vô nghĩa — ngưỡng này cũng là HEURISTIC tự chọn.
_UNKNOWN_DISTANCE_CUTOFF = {
    "wind_speed": 500.0,
    "power": 5000.0,
    "pitch": 300.0,
    "yaw_or_direction": 500.0,
}


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _name_match_score(column: str, concept: str) -> int:
    """Số lượng synonym KHỚP (substring, sau khi normalize) — dùng làm tín hiệu tên
    VÀ làm tie-break khi nhiều cột cùng khớp 1 concept (ưu tiên cột khớp CÀNG NHIỀU
    synonym càng cụ thể/mạnh, vd "DIRECTION_NACELLE" khớp cả "direction" lẫn
    "nacelle" nên được ưu tiên hơn "DIRECTION_WIND" chỉ khớp "direction")."""
    norm_col = _normalize(column)
    return sum(1 for syn in KNOWN_COLUMN_SYNONYMS[concept] if _normalize(syn) in norm_col)


def _value_match(concept: str, col_min: float | None, col_max: float | None) -> bool:
    if col_min is None or col_max is None:
        return False
    rule = _VALUE_RANGE_RULES[concept]
    if col_min < rule["min_floor"]:
        return False
    lo, hi = rule["max_range"]
    if col_max < lo:
        return False
    if hi is not None and col_max > hi:
        return False
    return True


def _value_distance(concept: str, col_min: float | None, col_max: float | None) -> float:
    """Độ 'gần đúng' khi KHÔNG khớp hoàn toàn — tổng độ lệch (vượt ngưỡng) ở min và
    max, dùng để chọn ứng viên LOW tốt nhất trong các lựa chọn còn lại. 0 = khớp hoàn
    toàn, càng lớn càng xa (vô cực nếu thiếu dữ liệu range)."""
    if col_min is None or col_max is None:
        return float("inf")
    rule = _VALUE_RANGE_RULES[concept]
    dist = 0.0
    if col_min < rule["min_floor"]:
        dist += rule["min_floor"] - col_min
    lo, hi = rule["max_range"]
    if col_max < lo:
        dist += lo - col_max
    if hi is not None and col_max > hi:
        dist += col_max - hi
    return dist


def looks_like_turbine_data(dataset_name: str, manager: DatasetManager) -> bool:
    """Heuristic MỀM (không cần hoàn hảo, không cần chính xác 100%) — CHỈ để tránh
    hiện UI đề xuất cho dataset HOÀN TOÀN không liên quan turbine (vd Tetouan — điện
    tiêu thụ + thời tiết). True nếu CÓ ÍT NHẤT 1 cột numeric (không tính cột derive)
    khớp CẢ tên LẪN giá trị hợp lý cho wind_speed hoặc power (tương đương điều kiện
    tier "HIGH" trong suggest_column_mapping()).

    ĐÃ ĐO THẬT (2026-09-17) trước khi chốt thiết kế này: bản đầu chỉ dùng giá trị
    (không cần tên) — Tetouan VẪN kích hoạt gate vì trùng hợp giá trị thuần túy:
    temperature (max=40.01) lọt khoảng hợp lý wind_speed (10-70); general_diffuse_flows/
    diffuse_flows/zone_1-3_consumption đều lọt khoảng power (không có trần trên). Thêm
    điều kiện TÊN (như HIGH) mới thật sự loại được Tetouan — xác nhận qua test thật,
    không suy đoán.

    ĐÁNH ĐỔI đã biết (đã trao đổi rõ, chấp nhận): nếu 1 file turbine thật bị đổi tên
    TOÀN BỘ cột liên quan (không còn gợi ý tên nào — không có cột nào đạt HIGH), gate
    này cũng ẩn UI theo — suggest_column_mapping() vẫn gọi được bằng Python trực tiếp,
    nhưng KHÔNG có đường vào qua chat/agent (hàm này CHƯA đăng ký vào
    agent/tool_registry.py hay agent/tool_schemas.py — đã grep xác nhận, không đoán)."""
    summary = get_dataset_summary(manager, dataset_name)
    if not summary["ok"]:
        return False
    data = summary["data"]
    numeric_ranges = data["numeric_ranges"]
    derived_columns = set(data["derived_columns"])
    for col, rng in numeric_ranges.items():
        if col in derived_columns:
            continue
        col_min, col_max = rng.get("min"), rng.get("max")
        for concept in ("wind_speed", "power"):
            if _name_match_score(col, concept) > 0 and _value_match(concept, col_min, col_max):
                return True
    return False


def suggest_column_mapping(dataset_name: str, manager: DatasetManager) -> dict:
    """
    Trả về dict {canonical_field: {"candidate": col_name_or_None, "confidence":
    "HIGH"/"MEDIUM"/"LOW"/"UNKNOWN", "reason": str}}. CHỈ đề xuất, KHÔNG áp dụng gì
    vào dataset/tool nào.

    TÁI DÙNG get_dataset_summary() (numeric_columns/numeric_ranges/derived_columns)
    THAY VÌ tự đọc lại DataFrame — tránh tính trùng lặp logic, dùng ĐÚNG cùng 1 nguồn
    dữ liệu mà UI/agent đã thấy.

    Thuật toán (chốt) — xử lý TỪNG canonical_field theo THỨ TỰ trong
    KNOWN_COLUMN_SYNONYMS (wind_speed -> power -> pitch -> yaw_or_direction), cột nào
    ĐÃ được gán cho field trước thì LOẠI khỏi vùng ứng viên của field sau (tránh 1 cột
    vật lý bị đề xuất 2 lần cho 2 khái niệm khác nhau). Đây là GREEDY đơn giản, KHÔNG
    phải bài toán gán ghép tối ưu — đủ dùng cho phạm vi "đề xuất", đúng tinh thần
    "không cần đầy đủ tuyệt đối" của yêu cầu.

    Confidence:
      HIGH    = tên khớp DANH SÁCH đã biết VÀ giá trị hợp lý.
      MEDIUM  = chỉ 1 trong 2 (tên khớp nhưng giá trị lạ, hoặc giá trị hợp lý nhưng
                tên không gợi ý gì).
      LOW     = không khớp gì rõ ràng, chỉ là "gần đúng nhất trong các lựa chọn còn
                lại" (và độ lệch chưa vượt _UNKNOWN_DISTANCE_CUTOFF).
      UNKNOWN = không có cột nào phù hợp (hết ứng viên, hoặc ứng viên gần nhất vẫn
                lệch quá xa để coi là gợi ý có ý nghĩa).
    """
    summary = get_dataset_summary(manager, dataset_name)
    if not summary["ok"]:
        return {
            field: {"candidate": None, "confidence": "UNKNOWN", "reason": summary["error"]}
            for field in KNOWN_COLUMN_SYNONYMS
        }

    data = summary["data"]
    numeric_columns = data["numeric_columns"]
    numeric_ranges = data["numeric_ranges"]
    derived_columns = set(data["derived_columns"])

    # Loại cột derive (year/month/day/hour/weekday/date, và "datetime" nếu mới thêm)
    # khỏi vùng ứng viên — KHÔNG BAO GIỜ là 1 cảm biến turbine thật.
    candidate_pool = [c for c in numeric_columns if c not in derived_columns]

    result = {}
    for concept in KNOWN_COLUMN_SYNONYMS:
        scored = []
        for col in candidate_pool:
            rng = numeric_ranges.get(col, {})
            col_min, col_max = rng.get("min"), rng.get("max")
            name_hits = _name_match_score(col, concept)
            value_ok = _value_match(concept, col_min, col_max)
            if name_hits > 0 and value_ok:
                tier = "HIGH"
            elif name_hits > 0 or value_ok:
                tier = "MEDIUM"
            else:
                tier = "LOW"
            distance = _value_distance(concept, col_min, col_max)
            scored.append((col, tier, name_hits, value_ok, distance, col_min, col_max))

        if not scored:
            result[concept] = {
                "candidate": None, "confidence": "UNKNOWN",
                "reason": "Không còn cột số nào chưa được gán cho khái niệm khác.",
            }
            continue

        tier_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        # Ưu tiên: tier tốt nhất -> nhiều synonym hit hơn -> distance nhỏ hơn (gần
        # đúng hơn) — thứ tự này quyết định khi nhiều cột cùng tier.
        scored.sort(key=lambda t: (tier_rank[t[1]], -t[2], t[4]))
        best_col, best_tier, best_hits, best_value_ok, best_dist, best_min, best_max = scored[0]

        if best_tier == "LOW" and best_dist > _UNKNOWN_DISTANCE_CUTOFF[concept]:
            result[concept] = {
                "candidate": None, "confidence": "UNKNOWN",
                "reason": (
                    f"Ứng viên gần nhất ('{best_col}', min={best_min}, max={best_max}) "
                    f"vẫn lệch quá xa khoảng hợp lý kỳ vọng — không đủ tin cậy để đề xuất."
                ),
            }
            continue

        others_same_tier = [s[0] for s in scored[1:] if s[1] == best_tier][:2]
        reason_parts = []
        if best_hits > 0:
            reason_parts.append(f"tên cột khớp {best_hits} từ khóa liên quan '{concept}'")
        if best_value_ok:
            reason_parts.append(f"giá trị (min={best_min}, max={best_max}) nằm trong khoảng hợp lý")
        elif best_min is not None:
            reason_parts.append(
                f"giá trị (min={best_min}, max={best_max}) KHÔNG khớp khoảng hợp lý kỳ vọng "
                f"— chỉ là lựa chọn gần đúng nhất còn lại"
            )
        reason = "; ".join(reason_parts) if reason_parts else "Không có tín hiệu tên hay giá trị nào hỗ trợ."
        if others_same_tier:
            reason += f" Cột khác cũng cùng mức tin cậy: {others_same_tier}."

        result[concept] = {"candidate": best_col, "confidence": best_tier, "reason": reason}
        candidate_pool = [c for c in candidate_pool if c != best_col]

    return result


def compute_columns_hash(columns: list[str]) -> str:
    """Hash từ sorted(tuple(columns)) — nhận diện ĐÚNG file có CÙNG cấu trúc cột dù
    tên file khác nhau (đúng yêu cầu). sha256 rút gọn 16 ký tự đủ tránh đụng độ cho
    quy mô số file thực tế của use-case này (vài chục-trăm dataset cục bộ)."""
    key = "|".join(sorted(columns))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def save_column_mapping(columns: list[str], mapping: dict) -> Path:
    """Lưu ánh xạ ĐÃ XÁC NHẬN (qua UI) vào data/column_mappings/{hash}.json. Ghi kèm
    danh sách cột gốc (không chỉ hash) để dễ debug/đọc lại bằng mắt."""
    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = MAPPINGS_DIR / f"{compute_columns_hash(columns)}.json"
    path.write_text(
        json.dumps({"columns": sorted(columns), "mapping": mapping}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_column_mapping(columns: list[str]) -> dict | None:
    """None nếu chưa từng xác nhận ánh xạ cho ĐÚNG bộ cột này (theo hash)."""
    path = MAPPINGS_DIR / f"{compute_columns_hash(columns)}.json"
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    return saved.get("mapping")
