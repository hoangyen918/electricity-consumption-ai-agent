"""Test guard chống ghi đè evals/baselines/v1_baseline.json khi phần lớn case fail vì
lỗi hạ tầng — sự cố thật 2026-09-24: 1 lần chạy 14/14 case fail vì CHAT_BASE_URL trả
530 đã ghi đè mất baseline thật (58 case, 89.66%), phải revert tay bằng git checkout.
Không mock toàn bộ Agent/eval loop — _save_baseline_with_guard()/_is_infra_failure()
là pure function nhận input trực tiếp, xem evals/run_evals.py."""
import json

from evals.run_evals import _is_infra_failure, _save_baseline_with_guard, INFRA_FAIL_THRESHOLD


def _fake_result(exception_text: str | None) -> dict:
    """1 entry tối giản trong all_results — chỉ cần field checks.exception mà
    _is_infra_failure() đọc, không cần các field khác của case thật."""
    return {"checks": {"exception": exception_text} if exception_text else {}}


def test_is_infra_failure_matches_known_patterns_not_logic_fail():
    """Khớp đúng các message lỗi hạ tầng ĐÃ THẬT gặp (xem INFRA_ERROR_PATTERNS), và
    KHÔNG khớp nhầm 1 lỗi logic bình thường (không phải hạ tầng)."""
    assert _is_infra_failure(
        "Không thể kết nối tới máy chủ AI (lỗi mạng/hạ tầng). Vui lòng thử lại sau."
    )
    assert _is_infra_failure("Error code: 401 - {'error': {'message': 'API key expired'}}")
    assert _is_infra_failure("Error code: 530 - Cloudflare tunnel error")
    assert _is_infra_failure(
        "HTTPConnectionPool(host='192.168.244.208', port=11434): Read timed out."
    )
    assert _is_infra_failure(
        "('Connection aborted.', ConnectionResetError(10054, "
        "'An existing connection was forcibly closed by the remote host'))"
    )
    # Lỗi LOGIC (recall/MRR thấp, sai câu trả lời) KHÔNG phải hạ tầng — không được
    # khớp nhầm, nếu không guard sẽ chặn nhầm cả những lần chạy hợp lệ có vài case
    # fail thật.
    assert not _is_infra_failure("Expected 'recall_10' to be True, got False")
    assert not _is_infra_failure("Answer did not contain expected refusal phrase")


def test_infra_heavy_run_blocks_baseline_overwrite_and_writes_failed_file(tmp_path):
    """100% case fail vì lỗi hạ tầng (vượt xa ngưỡng 30%) -> baseline chính GIỮ
    NGUYÊN, kết quả lần chạy này lưu riêng vào v1_baseline_FAILED_<ts>.json."""
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir()
    original_baseline = {"marker": "ORIGINAL_BASELINE", "total_cases": 58, "overall_accuracy": 0.8966}
    (baselines_dir / "v1_baseline.json").write_text(
        json.dumps(original_baseline), encoding="utf-8"
    )

    all_results = [
        _fake_result("Không thể kết nối tới máy chủ AI (lỗi mạng/hạ tầng). Vui lòng thử lại sau.")
        for _ in range(3)
    ]
    summary = {"marker": "NEW_FAILED_RUN", "total_cases": 3, "overall_accuracy": 0.0}

    saved_path = _save_baseline_with_guard(
        summary, all_results, total=3, ts="20260924_999999",
        force_save_baseline=False, baselines_dir=baselines_dir,
    )

    assert saved_path is None

    # Baseline chính KHÔNG bị đổi.
    current_main = json.loads((baselines_dir / "v1_baseline.json").read_text(encoding="utf-8"))
    assert current_main == original_baseline

    # Có file _FAILED riêng chứa kết quả của lần chạy fail này.
    failed_file = baselines_dir / "v1_baseline_FAILED_20260924_999999.json"
    assert failed_file.exists()
    assert json.loads(failed_file.read_text(encoding="utf-8")) == summary

    # Không có backup timestamp nào được tạo (không ghi đè nên không cần backup).
    backups = list(baselines_dir.glob("v1_baseline.2026*.json"))
    assert backups == []


def test_normal_run_overwrites_baseline_and_creates_timestamped_backup(tmp_path):
    """0% case fail hạ tầng (dưới ngưỡng) VÀ total_cases >= baseline hiện tại -> ghi đè
    baseline chính như cũ, có backup file cũ theo timestamp trước khi ghi đè."""
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir()
    # total_cases=5, KHÔNG phải 58 — cố tình bằng đúng số case của lần chạy mới bên
    # dưới, để không tự vướng guard scope mới (test riêng cho guard đó ở dưới).
    original_baseline = {"marker": "ORIGINAL_BASELINE", "total_cases": 5, "overall_accuracy": 0.8966}
    (baselines_dir / "v1_baseline.json").write_text(
        json.dumps(original_baseline), encoding="utf-8"
    )

    # 5 case: 4 pass (không có exception), 1 fail vì LOGIC (không phải hạ tầng) ->
    # infra_fail_ratio = 0/5 = 0%, dưới ngưỡng 30%.
    all_results = [_fake_result(None) for _ in range(4)] + [
        _fake_result("Expected 'recall_10' to be True, got False")
    ]
    summary = {"marker": "NEW_NORMAL_RUN", "total_cases": 5, "overall_accuracy": 0.8}

    saved_path = _save_baseline_with_guard(
        summary, all_results, total=5, ts="20260924_888888",
        force_save_baseline=False, baselines_dir=baselines_dir,
    )

    assert saved_path == baselines_dir / "v1_baseline.json"

    # Baseline chính đã được ghi đè bằng kết quả MỚI.
    current_main = json.loads((baselines_dir / "v1_baseline.json").read_text(encoding="utf-8"))
    assert current_main == summary

    # Bản cũ đã được backup theo timestamp TRƯỚC khi ghi đè.
    backup_file = baselines_dir / "v1_baseline.20260924_888888.json"
    assert backup_file.exists()
    assert json.loads(backup_file.read_text(encoding="utf-8")) == original_baseline


def test_partial_scope_run_blocks_baseline_overwrite_and_writes_partial_file(tmp_path):
    """Sự cố thật khác cùng ngày 2026-09-24: chạy eval CHỈ subset kb_qa (14/58 case,
    HỢP LỆ, KHÔNG lỗi hạ tầng) đã ghi đè mất baseline full-suite (58 case) bằng kết
    quả 14-case, phải revert tay lần 2 trong ngày — guard hạ tầng không bắt được vì
    tỷ lệ fail hạ tầng = 0%. Guard scope mới phải chặn đúng trường hợp này: 14 case
    (toàn bộ PASS, không exception nào) < 58 case của baseline hiện tại -> KHÔNG ghi
    đè, lưu riêng vào v1_baseline_PARTIAL_14cases_<ts>.json."""
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir()
    original_baseline = {"marker": "ORIGINAL_BASELINE_58", "total_cases": 58, "overall_accuracy": 0.8966}
    (baselines_dir / "v1_baseline.json").write_text(
        json.dumps(original_baseline), encoding="utf-8"
    )

    all_results = [_fake_result(None) for _ in range(14)]  # 14/14 pass, không case nào lỗi hạ tầng
    summary = {"marker": "PARTIAL_KB_QA_RUN", "total_cases": 14, "overall_accuracy": 1.0}

    saved_path = _save_baseline_with_guard(
        summary, all_results, total=14, ts="20260924_666666",
        force_save_baseline=False, baselines_dir=baselines_dir,
    )

    assert saved_path is None

    # Baseline chính (58 case) KHÔNG bị đổi.
    current_main = json.loads((baselines_dir / "v1_baseline.json").read_text(encoding="utf-8"))
    assert current_main == original_baseline

    # Có file _PARTIAL riêng chứa kết quả 14-case của lần chạy này.
    partial_file = baselines_dir / "v1_baseline_PARTIAL_14cases_20260924_666666.json"
    assert partial_file.exists()
    assert json.loads(partial_file.read_text(encoding="utf-8")) == summary

    # Không có backup timestamp nào được tạo (không ghi đè nên không cần backup).
    backups = list(baselines_dir.glob("v1_baseline.2026*.json"))
    assert backups == []


def test_full_scope_run_overwrites_baseline_normally(tmp_path):
    """Chạy ĐỦ 58/58 case (bằng đúng total_cases của baseline hiện tại, không phải
    subset) -> KHÔNG vướng guard scope (total >= existing, không phải nhỏ hơn), ghi
    đè bình thường như trước khi có guard này, có backup timestamp."""
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir()
    original_baseline = {"marker": "ORIGINAL_BASELINE_58", "total_cases": 58, "overall_accuracy": 0.8966}
    (baselines_dir / "v1_baseline.json").write_text(
        json.dumps(original_baseline), encoding="utf-8"
    )

    all_results = [_fake_result(None) for _ in range(58)]
    summary = {"marker": "FULL_58_RUN", "total_cases": 58, "overall_accuracy": 0.9138}

    saved_path = _save_baseline_with_guard(
        summary, all_results, total=58, ts="20260924_555555",
        force_save_baseline=False, baselines_dir=baselines_dir,
    )

    assert saved_path == baselines_dir / "v1_baseline.json"

    current_main = json.loads((baselines_dir / "v1_baseline.json").read_text(encoding="utf-8"))
    assert current_main == summary

    backup_file = baselines_dir / "v1_baseline.20260924_555555.json"
    assert backup_file.exists()
    assert json.loads(backup_file.read_text(encoding="utf-8")) == original_baseline


def test_force_save_baseline_overwrites_even_when_infra_heavy(tmp_path):
    """force_save_baseline=True phải ghi đè được dù tỷ lệ fail hạ tầng vượt ngưỡng —
    đây là lối thoát tường minh người dùng tự chọn, không phải hành vi mặc định."""
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir()

    all_results = [_fake_result("Error code: 530 - Cloudflare tunnel error") for _ in range(3)]
    summary = {"marker": "FORCED_SAVE", "total_cases": 3, "overall_accuracy": 0.0}

    saved_path = _save_baseline_with_guard(
        summary, all_results, total=3, ts="20260924_777777",
        force_save_baseline=True, baselines_dir=baselines_dir,
    )

    assert saved_path == baselines_dir / "v1_baseline.json"
    current_main = json.loads((baselines_dir / "v1_baseline.json").read_text(encoding="utf-8"))
    assert current_main == summary


def test_force_save_baseline_overwrites_even_when_partial_scope(tmp_path):
    """force_save_baseline=True cũng phải bypass được guard SCOPE (total_cases nhỏ
    hơn baseline hiện tại), không chỉ guard hạ tầng — cùng 1 flag cho cả 2 guard."""
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir()
    original_baseline = {"marker": "ORIGINAL_BASELINE_58", "total_cases": 58, "overall_accuracy": 0.8966}
    (baselines_dir / "v1_baseline.json").write_text(
        json.dumps(original_baseline), encoding="utf-8"
    )

    all_results = [_fake_result(None) for _ in range(14)]
    summary = {"marker": "FORCED_PARTIAL_SAVE", "total_cases": 14, "overall_accuracy": 1.0}

    saved_path = _save_baseline_with_guard(
        summary, all_results, total=14, ts="20260924_444444",
        force_save_baseline=True, baselines_dir=baselines_dir,
    )

    assert saved_path == baselines_dir / "v1_baseline.json"
    current_main = json.loads((baselines_dir / "v1_baseline.json").read_text(encoding="utf-8"))
    assert current_main == summary
