from pathlib import Path

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

    def load_excel(self, path: str, sheet_name=0) -> None:
        """
        Load dữ liệu từ file Excel (.xlsx/.xls). Mặc định đọc sheet đầu tiên.
        sheet_name có thể là tên sheet (str) hoặc index (int).
        """
        df = pd.read_excel(path, sheet_name=sheet_name)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        self._df = df
        self._source_path = path

    def load_file(self, path: str, sheet_name=0) -> None:
        """
        Điểm vào chung — tự nhận diện .csv hay .xlsx/.xls theo phần mở rộng,
        gọi đúng method tương ứng. Dùng hàm này thay vì gọi trực tiếp
        load_csv()/load_excel() khi không biết trước định dạng file (ví dụ user upload).
        """
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            self.load_csv(path)
        elif ext in (".xlsx", ".xls"):
            self.load_excel(path, sheet_name=sheet_name)
        else:
            raise ValueError(f"Định dạng '{ext}' không được hỗ trợ. Chỉ hỗ trợ .csv, .xlsx, .xls")

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
