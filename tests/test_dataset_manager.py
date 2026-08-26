from pathlib import Path

import pandas as pd
import pytest

from tools.dataset_manager import DatasetManager

SAMPLE_PATH = str(Path(__file__).resolve().parent.parent / "data" / "sample" / "tetouan_sample.csv")


def test_load_csv_success():
    manager = DatasetManager()
    manager.load_csv(SAMPLE_PATH)

    assert manager.is_loaded() is True
    df = manager.get_dataframe()
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])


def test_get_dataframe_raises_when_not_loaded():
    manager = DatasetManager()

    with pytest.raises(RuntimeError):
        manager.get_dataframe()


def test_get_columns_returns_correct_list():
    manager = DatasetManager()
    manager.load_csv(SAMPLE_PATH)

    columns = manager.get_columns()

    assert "datetime" in columns
    assert "zone_1_consumption" in columns
    assert "temperature" in columns
    assert columns == list(manager.get_dataframe().columns)


def test_get_numeric_columns_excludes_datetime():
    manager = DatasetManager()
    manager.load_csv(SAMPLE_PATH)

    numeric_columns = manager.get_numeric_columns()

    assert "datetime" not in numeric_columns
    assert "zone_1_consumption" in numeric_columns
    assert "temperature" in numeric_columns
