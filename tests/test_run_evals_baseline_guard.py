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
    """0% case fail hạ tầng (dưới ngưỡng) -> ghi đè baseline chính như cũ, có backup
    file cũ theo timestamp trước khi ghi đè."""
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir()
    original_baseline = {"marker": "ORIGINAL_BASELINE", "total_cases": 58, "overall_accuracy": 0.8966}
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
