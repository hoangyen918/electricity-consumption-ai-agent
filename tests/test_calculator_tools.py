from tools.calculator_tools import calculate


def test_add():
    r = calculate("add", 5, 3)
    assert r["ok"] is True
    assert r["data"]["result"] == 8


def test_percentage_of():
    # 2% của 1,200,000 = 24,000 (test case từ M0b)
    r = calculate("percentage_of", 1200000, 2)
    assert r["ok"] is True
    assert r["data"]["result"] == 24000


def test_percentage_change():
    r = calculate("percentage_change", 100, 150)
    assert r["ok"] is True
    assert r["data"]["result"] == 50.0


def test_divide_by_zero():
    r = calculate("divide", 10, 0)
    assert r["ok"] is False
    assert "zero" in r["error"].lower()


def test_invalid_operation():
    r = calculate("run_python", 1, 2)
    assert r["ok"] is False
    assert "not allowed" in r["error"].lower()


def test_percentage_change_zero_base():
    r = calculate("percentage_change", 0, 50)
    assert r["ok"] is False
