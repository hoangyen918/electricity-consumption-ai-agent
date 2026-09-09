"""
Wrapper cho các tool turbine gió cần "chuỗi" kết quả (estimate_power_curve ->
classify_turbine_status -> group_anomaly_windows/detect_anomalies_ml/predict_power)
khi gọi qua agent (LLM), KHÔNG phải khi gọi trực tiếp bằng Python (tools/*.py không đổi).

VẤN ĐỀ THẬT cần giải quyết: classify_turbine_status() trả về "classification" là 1
dict ĐẦY ĐỦ TỪNG DÒNG (233.953-262.796 entries cho GT1/WT1 — hàng MB), thiết kế để
tiêu thụ TRỰC TIẾP trong Python (group_anomaly_windows() cần đủ dữ liệu để gộp cửa sổ
thời gian), KHÔNG BAO GIỜ định để LLM tự copy nguyên văn vào tool-call argument tiếp
theo (bất khả thi cả về token limit lẫn chi phí). Tương tự, power_curve (output
estimate_power_curve()) tuy nhỏ hơn nhiều nhưng vẫn cần "truyền" giữa các lệnh gọi.

Giải pháp: cache power_curve/classification NGAY TRÊN instance DatasetManager của
session đó (attribute riêng, không phải module-level — tránh rò rỉ cache giữa các
session/user khác nhau khi nhiều người dùng chung app). Các tool wrapper LLM-facing
(estimate_power_curve/classify_turbine_status/group_anomaly_windows/detect_anomalies_ml/
predict_power) tự tính toán lại NẾU CHƯA CÓ trong cache, tái dùng NẾU ĐÃ CÓ (đúng
tham số) — LLM chỉ cần gọi tuần tự theo tên tool, không cần tự truyền dict khổng lồ.

classify_turbine_status wrapper CHỈ trả về "summary" cho LLM (bỏ "classification" đầy
đủ khỏi output — giữ đúng nguyên tắc "không hiển thị nguyên văn cho agent/LLM" đã ghi
trong docstring gốc của tools/turbine_diagnostic_tools.py::classify_turbine_status()).
"""
import pandas as pd

from tools.dataset_manager import DatasetManager
from tools.tool_results import failure, success
from tools.turbine_diagnostic_tools import (
    _detect_site,
    _power_wind_columns,
    estimate_power_curve,
    classify_turbine_status,
    group_anomaly_windows,
    detect_anomalies_ml,
    predict_power,
)

# Dùng bin_width=1.0 làm mặc định cho LLM (KHÁC mặc định nội bộ của estimate_power_curve()
# là 0.5) — vì TOÀN BỘ số liệu đã đo thật và validate (Gate 4a/4b/4c, cross-tab OpSt...)
# trong suốt Phase 3-4 đều dùng bin_width=1.0, chưa từng validate 0.5 trên dữ liệu thật.
TURBINE_DEFAULT_BIN_WIDTH = 1.0

# Feature set MẶC ĐỊNH cho predict_power khi LLM không chỉ định feature_columns — đúng
# bộ feature ĐÃ ĐO THẬT vượt Gate 4b/4c (xem docstring predict_power(), đo 2026-09-08):
# GT1 RMSE=269.11<305.15, WT1 RMSE=135.12<151.58, wind_speed rank #1 cả 2 site.
DEFAULT_PREDICT_POWER_FEATURES = {
    "site1": ["GT1.Pitch.Angle1", "GT1.Common.WindTrackDevAng"],
    "site2": ["PITCH", "WIND_SPEED_STD", "TEMPERATURE_STATOR"],
}

_POWER_CURVE_CACHE_ATTR = "_turbine_power_curve_cache"
_CLASSIFICATION_CACHE_ATTR = "_turbine_classification_cache"


def resolve_power_wind_columns(
    dataset_manager: DatasetManager, dataset_name: str | None, power_column: str | None, wind_speed_column: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Tự nhận diện power_column/wind_speed_column theo site (GT1.* vs ACTIVE_POWER/
    WIND_SPEED) nếu LLM không chỉ định — LLM không cần thuộc lòng tên cột chính xác của
    từng dataset. Nếu LLM CÓ chỉ định, ưu tiên dùng đúng giá trị đó (linh hoạt hơn).

    GUARD CỨNG (phát hiện thật khi test E2E — LLM đôi khi bỏ qua hướng dẫn trong system
    prompt và gọi tool KHÔNG chỉ định dataset_name dù có >=2 dataset đã tải, khiến tool
    ngầm chạy trên dataset "gần nhất" — 1 lần quan sát được: predict_power({}) chạy nhầm
    trên WindFarm04_WT1 nhưng LLM trình bày kết quả NHƯ THỂ là his_data_GT1, dữ liệu sai
    bị gắn nhãn site sai. KHÔNG thể chỉ dựa vào prompt engineering (xác suất) cho lỗi
    nghiêm trọng này — chặn cứng ở code khi dataset_name=None VÀ có nhiều hơn 1 dataset.
    """
    if dataset_name is None and len(dataset_manager.list_datasets()) > 1:
        return None, None, (
            "Có nhiều hơn 1 dataset đang được tải — PHẢI chỉ định rõ dataset_name (không "
            f"để trống) khi gọi tool này. Các dataset hiện có: {dataset_manager.list_datasets()}"
        )
    if power_column and wind_speed_column:
        return power_column, wind_speed_column, None
    df = dataset_manager.get_dataframe(dataset_name)
    if df is None:
        return None, None, (
            f"Dataset '{dataset_name}' không tồn tại. Các dataset hiện có: {dataset_manager.list_datasets()}"
        )
    site = _detect_site(df)
    if site is None:
        return None, None, (
            "Không tự nhận diện được cấu trúc cột turbine gió cho dataset này (cần cột "
            "bắt đầu bằng 'GT1.' hoặc cả 'ACTIVE_POWER' và 'WIND_SPEED') — cần chỉ định "
            "rõ power_column và wind_speed_column."
        )
    detected_power, detected_wind = _power_wind_columns(site)
    return power_column or detected_power, wind_speed_column or detected_wind, None


def _cache_dict(dataset_manager: DatasetManager, attr: str) -> dict:
    cache = getattr(dataset_manager, attr, None)
    if cache is None:
        cache = {}
        setattr(dataset_manager, attr, cache)
    return cache


def get_or_compute_power_curve(
    dataset_manager: DatasetManager, dataset_name: str | None, power_column: str, wind_speed_column: str, bin_width: float,
) -> dict:
    cache = _cache_dict(dataset_manager, _POWER_CURVE_CACHE_ATTR)
    key = (dataset_name, power_column, wind_speed_column, bin_width)
    if key in cache:
        return success(cache[key])
    result = estimate_power_curve(
        dataset_manager, power_column=power_column, wind_speed_column=wind_speed_column,
        dataset_name=dataset_name, bin_width=bin_width,
    )
    if result["ok"]:
        cache[key] = result["data"]
    return result


def get_or_compute_classification(
    dataset_manager: DatasetManager, dataset_name: str | None, power_column: str, wind_speed_column: str,
    setpoint_column: str | None, bin_width: float,
) -> dict:
    curve_result = get_or_compute_power_curve(dataset_manager, dataset_name, power_column, wind_speed_column, bin_width)
    if not curve_result["ok"]:
        return curve_result
    cache = _cache_dict(dataset_manager, _CLASSIFICATION_CACHE_ATTR)
    key = (dataset_name, power_column, wind_speed_column, setpoint_column, bin_width)
    if key in cache:
        return success(cache[key])
    result = classify_turbine_status(
        dataset_manager, power_column=power_column, wind_speed_column=wind_speed_column,
        power_curve=curve_result["data"], dataset_name=dataset_name, setpoint_column=setpoint_column,
    )
    if result["ok"]:
        cache[key] = result["data"]
    return result


def estimate_power_curve_for_agent(
    dataset_manager: DatasetManager, dataset_name: str | None, power_column: str | None,
    wind_speed_column: str | None, bin_width: float,
) -> dict:
    power_column, wind_speed_column, err = resolve_power_wind_columns(dataset_manager, dataset_name, power_column, wind_speed_column)
    if err:
        return failure(err)
    return get_or_compute_power_curve(dataset_manager, dataset_name, power_column, wind_speed_column, bin_width)


def classify_turbine_status_for_agent(
    dataset_manager: DatasetManager, dataset_name: str | None, power_column: str | None,
    wind_speed_column: str | None, setpoint_column: str | None, bin_width: float,
) -> dict:
    power_column, wind_speed_column, err = resolve_power_wind_columns(dataset_manager, dataset_name, power_column, wind_speed_column)
    if err:
        return failure(err)
    result = get_or_compute_classification(dataset_manager, dataset_name, power_column, wind_speed_column, setpoint_column, bin_width)
    if not result["ok"]:
        return result
    # KHONG tra "classification" day du (hang tram nghin dong) cho LLM — chi tom tat.
    visible_data = {k: v for k, v in result["data"].items() if k != "classification"}
    return success(visible_data)


# Gioi han so window tra ve cho LLM — PHAT HIEN THAT khi test E2E: goi
# group_anomaly_windows KHONG thu hep exclude_labels (mac dinh chi loai "Normal") tren
# GT1 tra ve HANG NGHIN window nho le (Under Production/Partial Stop/Measurement Error...
# deu la nhan pho bien) -> 1 tool result ~700.000 ky tu, VUOT context limit cua model
# (262.144 token) ngay ca khi CHI 1 tool result nay trong hoi thoai. group_anomaly_windows()
# goc (tools/turbine_diagnostic_tools.py) KHONG doi — chi cat bot o day (wrapper) truoc
# khi tra cho LLM, uu tien giu cac window DAI NHAT (thong tin huu ich nhat cho cau hoi
# dieu tra "khoang thoi gian nao bi keo dai").
#
# PHAT HIEN THAT #2 (2026-09-08, xem docs bao cao Phase 5): cap 30-dai-nhat-GLOBAL ap
# dung TRUOC khi biet cau hoi can khoang thoi gian nao -> neu user hoi ve 1 thang cu
# the (vd "curtailment thang 4"), cac window that trong thang do co the KHONG lot top-30
# toan cuc va bi xoa hoan toan khoi ket qua, du that su ton tai. Da verify tren GT1 that:
# 5 thang (4,5,6,7,10) bi xoa SACH du co tong 60 window that. FIX: them start_date/
# end_date de filter TRUOC khi cap — neu co filter thoi gian, nguoi dung da chu dong thu
# hep pham vi nen KHONG can cap chat (nang len 200, chi de phong truong hop cuc doan);
# neu KHONG co filter (cau hoi tong quat toan bo dataset), GIU NGUYEN cap 30 nhu cu.
MAX_WINDOWS_FOR_AGENT = 30
MAX_WINDOWS_FOR_AGENT_WITH_DATE_FILTER = 200


def group_anomaly_windows_for_agent(
    dataset_manager: DatasetManager, dataset_name: str | None, timestamp_column: str,
    power_column: str | None, wind_speed_column: str | None, setpoint_column: str | None,
    bin_width: float, min_consecutive_points: int, exclude_labels: list[str] | None,
    start_date: str | None = None, end_date: str | None = None,
) -> dict:
    power_column, wind_speed_column, err = resolve_power_wind_columns(dataset_manager, dataset_name, power_column, wind_speed_column)
    if err:
        return failure(err)
    classification_result = get_or_compute_classification(dataset_manager, dataset_name, power_column, wind_speed_column, setpoint_column, bin_width)
    if not classification_result["ok"]:
        return classification_result
    result = group_anomaly_windows(
        dataset_manager, classification=classification_result["data"]["classification"],
        timestamp_column=timestamp_column, dataset_name=dataset_name,
        min_consecutive_points=min_consecutive_points, exclude_labels=exclude_labels,
    )
    if not result["ok"]:
        return result
    windows = result["data"]["windows"]

    date_filter_applied = start_date is not None or end_date is not None
    if date_filter_applied:
        # Dong bo dung convention da co o tools/dataframe_tools.py::aggregate_metric()/
        # compare_periods(): start_date so sanh >=, end_date la CA ngay do (+1 ngay, so
        # sanh <) — khong tu suy dien convention khac.
        start_ts = end_ts = None
        if start_date is not None:
            try:
                start_ts = pd.to_datetime(start_date)
            except (ValueError, TypeError):
                return failure(f"Invalid start_date format: '{start_date}'")
        if end_date is not None:
            try:
                end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            except (ValueError, TypeError):
                return failure(f"Invalid end_date format: '{end_date}'")

        def _in_date_range(w):
            w_start = pd.Timestamp(w["start_time"])
            if start_ts is not None and w_start < start_ts:
                return False
            if end_ts is not None and w_start >= end_ts:
                return False
            return True

        windows = [w for w in windows if _in_date_range(w)]

    max_cap = MAX_WINDOWS_FOR_AGENT_WITH_DATE_FILTER if date_filter_applied else MAX_WINDOWS_FOR_AGENT

    def _duration_seconds(w):
        # duration la string kieu "X days HH:MM:SS" hoac "HH:MM:SS" tu str(pd.Timedelta)
        try:
            return pd.Timedelta(w["duration"]).total_seconds()
        except Exception:
            return 0

    result["data"]["date_filter_applied"] = {"start_date": start_date, "end_date": end_date} if date_filter_applied else None
    result["data"]["windows_total_found"] = len(windows)

    if len(windows) > max_cap:
        sorted_windows = sorted(windows, key=_duration_seconds, reverse=True)
        result["data"]["windows"] = sorted_windows[:max_cap]
        result["data"]["windows_truncated"] = True
        result["data"]["windows_truncation_note"] = (
            f"Tìm được {len(windows)} khoảng thời gian, chỉ hiển thị {max_cap} khoảng DÀI "
            "NHẤT (sắp theo duration giảm dần) để tránh kết quả quá lớn. Nếu cần thu hẹp "
            "phạm vi thật (không chỉ hiển thị ít hơn), gọi lại với exclude_labels cụ thể "
            "hơn, tăng min_consecutive_points, hoặc thu hẹp start_date/end_date hơn nữa."
        )
    else:
        result["data"]["windows"] = windows
        result["data"]["windows_truncated"] = False
    return result


def detect_anomalies_ml_for_agent(
    dataset_manager: DatasetManager, dataset_name: str | None, power_column: str | None,
    wind_speed_column: str | None, setpoint_column: str | None, pitch_column: str | None,
    wind_speed_std_column: str | None, bin_width: float,
) -> dict:
    power_column, wind_speed_column, err = resolve_power_wind_columns(dataset_manager, dataset_name, power_column, wind_speed_column)
    if err:
        return failure(err)
    classification_result = get_or_compute_classification(dataset_manager, dataset_name, power_column, wind_speed_column, setpoint_column, bin_width)
    if not classification_result["ok"]:
        return classification_result
    return detect_anomalies_ml(
        dataset_manager, power_column=power_column, wind_speed_column=wind_speed_column,
        classification=classification_result["data"]["classification"], dataset_name=dataset_name,
        pitch_column=pitch_column, wind_speed_std_column=wind_speed_std_column,
    )


def predict_power_for_agent(
    dataset_manager: DatasetManager, dataset_name: str | None, power_column: str | None,
    wind_speed_column: str | None, setpoint_column: str | None, feature_columns: list[str] | None,
    bin_width: float,
) -> dict:
    power_column, wind_speed_column, err = resolve_power_wind_columns(dataset_manager, dataset_name, power_column, wind_speed_column)
    if err:
        return failure(err)
    if feature_columns is None:
        df = dataset_manager.get_dataframe(dataset_name)
        site = _detect_site(df) if df is not None else None
        candidate = DEFAULT_PREDICT_POWER_FEATURES.get(site, [])
        # chi giu feature THAT SU ton tai trong dataset (phong truong hop bien the site2
        # thieu 1 vai cot nhiet do/PITCH)
        feature_columns = [c for c in candidate if df is not None and c in df.columns] or None
    curve_result = get_or_compute_power_curve(dataset_manager, dataset_name, power_column, wind_speed_column, bin_width)
    if not curve_result["ok"]:
        return curve_result
    classification_result = get_or_compute_classification(dataset_manager, dataset_name, power_column, wind_speed_column, setpoint_column, bin_width)
    if not classification_result["ok"]:
        return classification_result
    return predict_power(
        dataset_manager, power_column=power_column, wind_speed_column=wind_speed_column,
        power_curve=curve_result["data"], classification=classification_result["data"]["classification"],
        dataset_name=dataset_name, feature_columns=feature_columns,
    )
