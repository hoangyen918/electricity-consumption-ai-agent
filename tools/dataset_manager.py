from pathlib import Path

import pandas as pd

DATETIME_DERIVED_COLUMNS = ["year", "month", "day", "hour", "weekday", "date"]
MIN_DATETIME_PARSE_RATE = 0.95


def _parse_datetime_best_effort(series: pd.Series) -> tuple[pd.Series, float]:
    """
    Thử pd.to_datetime với CẢ dayfirst=False (mặc định) và dayfirst=True, chọn kết
    quả có tỷ lệ parse thành công (không NaT) cao hơn. KHÔNG hardcode dayfirst cho
    toàn bộ derive_time_columns — mỗi cột tự chọn hướng phù hợp với dữ liệu của nó
    (vd cột kiểu VN "15/03/2024" cần dayfirst=True mới parse được vì ngày=15>12 khiến
    cách hiểu MM/DD mặc định coerce thành NaT; cột ISO "2017-01-05 13:20:00" không bị
    ảnh hưởng bởi dayfirst nên vẫn ra kết quả như cũ).

    Lưu ý: nếu MỌI giá trị trong cột đều có phần "ngày" ≤ 12 (vd toàn "01/03/2024"),
    cả 2 cách hiểu đều "parse được" (tỷ lệ bằng nhau) nhưng có thể cho ý nghĩa khác
    nhau (tháng 1 ngày 3 vs tháng 3 ngày 1) — trường hợp này không thể phân biệt chỉ
    bằng tỷ lệ parse, giữ nguyên dayfirst=False (mặc định pandas) khi tỷ lệ bằng nhau.
    """
    non_null = series.notna().sum()
    if non_null == 0:
        return pd.to_datetime(series, errors="coerce"), 0.0

    parsed_default = pd.to_datetime(series, errors="coerce", dayfirst=False)
    rate_default = parsed_default.notna().sum() / non_null

    parsed_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True)
    rate_dayfirst = parsed_dayfirst.notna().sum() / non_null

    if rate_dayfirst > rate_default:
        return parsed_dayfirst, rate_dayfirst
    return parsed_default, rate_default


def derive_time_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """
    Tự động phát hiện cột datetime trong dataframe (KHÔNG hard-code tên cột cụ thể
    của Tetouan — phải chạy được trên CSV lạ với tên cột bất kỳ) và derive các cột
    year/month/day/hour/weekday/date mà group_metric() cần có sẵn (group_metric
    không tự tính lại từ datetime tại runtime).

    Không rename/xóa bất kỳ cột giá trị gốc nào. Nếu không cột nào đạt ngưỡng
    MIN_DATETIME_PARSE_RATE, trả về df nguyên vẹn, không raise — dataset không có
    timestamp là hợp lệ, các câu hỏi theo group_by/start_date sẽ tự fail qua
    failure() sẵn có trong dataframe_tools.py.

    Trả về (df đã (có thể) thêm cột, tên cột gốc được chọn làm nguồn datetime hoặc None).
    """
    df = df.copy()
    source_column: str | None = None

    if "datetime" in df.columns:
        parsed, rate = _parse_datetime_best_effort(df["datetime"])
        if rate >= MIN_DATETIME_PARSE_RATE:
            source_column = "datetime"
            df["datetime"] = parsed

    if source_column is None:
        best_col = None
        best_rate = 0.0
        best_parsed = None
        for col in df.columns:
            if col == "datetime" or pd.api.types.is_numeric_dtype(df[col]):
                continue
            if df[col].notna().sum() == 0:
                continue
            parsed, rate = _parse_datetime_best_effort(df[col])
            if rate > best_rate:
                best_rate = rate
                best_col = col
                best_parsed = parsed
        if best_col is not None and best_rate >= MIN_DATETIME_PARSE_RATE:
            source_column = best_col
            df["datetime"] = best_parsed

    if source_column is not None:
        dt = df["datetime"]
        if "year" not in df.columns:
            df["year"] = dt.dt.year
        if "month" not in df.columns:
            df["month"] = dt.dt.month
        if "day" not in df.columns:
            df["day"] = dt.dt.day
        if "hour" not in df.columns:
            df["hour"] = dt.dt.hour
        if "weekday" not in df.columns:
            df["weekday"] = dt.dt.weekday
        if "date" not in df.columns:
            df["date"] = dt.dt.date

    return df, source_column


class DatasetManager:
    """Giữ 1 dataframe duy nhất trong memory cho toàn bộ session."""

    def __init__(self):
        self._df: pd.DataFrame | None = None
        self._source_path: str | None = None
        self._datetime_source_column: str | None = None

    def load_csv(self, path: str) -> None:
        df = pd.read_csv(path)
        df, source_column = derive_time_columns(df)
        self._df = df
        self._source_path = path
        self._datetime_source_column = source_column

    def load_excel(self, path: str, sheet_name=0) -> None:
        """
        Load dữ liệu từ file Excel (.xlsx/.xls). Mặc định đọc sheet đầu tiên.
        sheet_name có thể là tên sheet (str) hoặc index (int).
        """
        df = pd.read_excel(path, sheet_name=sheet_name)
        df, source_column = derive_time_columns(df)
        self._df = df
        self._source_path = path
        self._datetime_source_column = source_column

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

    def get_datetime_source_column(self) -> str | None:
        """
        Tên cột gốc được tự động chọn làm nguồn datetime (xem derive_time_columns).
        None nếu không có cột nào đạt ngưỡng parse hoặc chưa load dataset nào.
        """
        return self._datetime_source_column
