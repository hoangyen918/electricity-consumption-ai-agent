import pandas as pd


class DatasetManager:
    """Giữ 1 dataframe duy nhất trong memory cho toàn bộ session."""

    def __init__(self):
        self._df: pd.DataFrame | None = None
        self._source_path: str | None = None

    def load_csv(self, path: str) -> None:
        df = pd.read_csv(path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        self._df = df
        self._source_path = path

    def is_loaded(self) -> bool:
        return self._df is not None

    def get_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("No dataset loaded. Call load_csv() first.")
        return self._df

    def get_columns(self) -> list[str]:
        return list(self.get_dataframe().columns)

    def get_numeric_columns(self) -> list[str]:
        df = self.get_dataframe()
        return df.select_dtypes(include="number").columns.tolist()
