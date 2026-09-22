import csv
from pathlib import Path

import pandas as pd

DATETIME_DERIVED_COLUMNS = ["year", "month", "day", "hour", "weekday", "date"]
MIN_DATETIME_PARSE_RATE = 0.95
CSV_DELIMITER_CANDIDATES = [",", ";", "\t", "|"]


def _detect_delimiter(path: str) -> str:
    """
    Tự phát hiện delimiter thật của file CSV thay vì hardcode dấu phẩy — cần thiết
    vì có dataset dùng dấu chấm phẩy (vd file wind turbine WindFarm04_WT*.csv, xuất
    từ hệ thống châu Âu). Đọc dòng đầu (header) bằng encoding utf-8-sig (tự bỏ BOM
    nếu có) rồi dùng csv.Sniffer() giới hạn trong CSV_DELIMITER_CANDIDATES.

    Nếu Sniffer không nhận diện được (vd file chỉ có 1 cột, không có delimiter nào
    xuất hiện), mặc định về "," để giữ nguyên hành vi cũ cho các dataset chỉ có 1 cột.
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        header_line = f.readline()
    try:
        dialect = csv.Sniffer().sniff(header_line, delimiters="".join(CSV_DELIMITER_CANDIDATES))
        return dialect.delimiter
    except csv.Error:
        return ","


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
    """
    Giữ NHIỀU dataset trong memory cho 1 session (key = tên dataset, mặc định là
    basename file, có thể truyền tên khác qua tham số `name` — dùng khi path vật lý
    là file tạm nhưng muốn hiển thị đúng tên gốc user upload, xem app.py::handle_upload).
    "Active" luôn là dataset được load() gần nhất — tool nào không truyền dataset_name
    sẽ thao tác trên active, giữ nguyên hành vi single-dataset cũ.
    """

    def __init__(self):
        self._datasets: dict[str, pd.DataFrame] = {}
        self._source_paths: dict[str, str | None] = {}
        self._datetime_source_columns: dict[str, str | None] = {}
        self._active_name: str | None = None

    @property
    def df(self) -> pd.DataFrame:
        """Backward-compat property, tương đương get_dataframe() không tham số (raise
        nếu chưa load dataset nào)."""
        return self.get_dataframe()

    def load_csv(self, path: str, name: str | None = None, set_active: bool = True) -> None:
        """set_active=True (mặc định, giữ NGUYÊN hành vi cũ cho mọi call site hiện có):
        dataset vừa nạp trở thành active, tool nào không truyền dataset_name sẽ thao
        tác trên đúng dataset này. set_active=False: thêm vào self._datasets như bình
        thường nhưng KHÔNG đổi self._active_name — dùng khi nạp 1 dataset "nền" (vd
        bridge từ Dashboard sang Chat) mà KHÔNG được phép cướp active của dataset user
        đang thao tác (xem _bridge_wind_turbine_datasets_to_chat() trong app.py).
        Edge case CỐ Ý: nếu manager đang RỖNG (_active_name=None) và set_active=False,
        _active_name vẫn giữ None sau khi nạp (không tự động "nhận" active dù chưa có
        active nào) — dataset đó tồn tại trong self._datasets nhưng get_dataframe(None)
        sẽ raise RuntimeError cho tới khi có 1 lần nạp/switch với set_active=True."""
        delimiter = _detect_delimiter(path)
        df = pd.read_csv(path, sep=delimiter, encoding="utf-8-sig")
        df, source_column = derive_time_columns(df)
        key = name or Path(path).name
        self._datasets[key] = df
        self._source_paths[key] = path
        self._datetime_source_columns[key] = source_column
        if set_active:
            self._active_name = key

    def load_excel(self, path: str, sheet_name=0, name: str | None = None, set_active: bool = True) -> None:
        """
        Load dữ liệu từ file Excel (.xlsx/.xls). Mặc định đọc sheet đầu tiên.
        sheet_name có thể là tên sheet (str) hoặc index (int).

        set_active — xem docstring load_csv() (cùng ý nghĩa/edge case, KHÔNG lặp lại).
        """
        df = pd.read_excel(path, sheet_name=sheet_name)
        df, source_column = derive_time_columns(df)
        key = name or Path(path).name
        self._datasets[key] = df
        self._source_paths[key] = path
        self._datetime_source_columns[key] = source_column
        if set_active:
            self._active_name = key

    def add_dataframe(self, df: pd.DataFrame, name: str, set_active: bool = True) -> None:
        """Đăng ký 1 DataFrame ĐÃ CÓ SẴN trong bộ nhớ (vd đã load ở 1 DatasetManager
        khác) vào manager này, KHÔNG đọc lại từ file — dùng khi 1 dataset (vd người
        dùng tự upload qua sidebar, hoặc dataset turbine bridge từ Dashboard) cần dùng
        chung được ở nhiều DatasetManager instance khác nhau trong cùng 1 session (xem
        app.py: st.session_state.dataset_manager cho Chat vs st.session_state.
        wind_turbine_manager cho Dashboard — 2 instance TÁCH RIÊNG có chủ đích).

        Vẫn gọi derive_time_columns() — AN TOÀN/idempotent (chỉ thêm cột còn thiếu,
        không ghi đè cột đã có, xem docstring derive_time_columns) — để đảm bảo dataset
        được đăng ký ĐÚNG CHUẨN như load_csv()/load_excel(), kể cả khi df truyền vào
        chưa từng qua derive_time_columns() ở manager nguồn.

        source_path lưu None (dataset không xuất phát từ 1 file vật lý cụ thể TRONG
        manager này — get_datetime_source_column() vẫn hoạt động bình thường, chỉ
        get_source_path()-kiểu-method nào đó trong tương lai mới bị ảnh hưởng, hiện
        chưa có method như vậy).

        set_active — xem docstring load_csv() (cùng ý nghĩa/edge case, KHÔNG lặp lại)."""
        df, source_column = derive_time_columns(df)
        self._datasets[name] = df
        self._source_paths[name] = None
        self._datetime_source_columns[name] = source_column
        if set_active:
            self._active_name = name

    def load_file(self, path: str, sheet_name=0, name: str | None = None) -> None:
        """
        Điểm vào chung — tự nhận diện .csv hay .xlsx/.xls theo phần mở rộng,
        gọi đúng method tương ứng. Dùng hàm này thay vì gọi trực tiếp
        load_csv()/load_excel() khi không biết trước định dạng file (ví dụ user upload).
        """
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            self.load_csv(path, name=name)
        elif ext in (".xlsx", ".xls"):
            self.load_excel(path, sheet_name=sheet_name, name=name)
        else:
            raise ValueError(f"Định dạng '{ext}' không được hỗ trợ. Chỉ hỗ trợ .csv, .xlsx, .xls")

    def is_loaded(self) -> bool:
        return len(self._datasets) > 0

    def get_dataframe(self, name: str | None = None) -> pd.DataFrame | None:
        """
        name=None -> active dataset. name=None VÀ chưa từng load gì -> raise
        RuntimeError (giữ đúng hành vi cũ trước khi có multi-dataset, xem
        test_get_dataframe_raises_when_not_loaded). name=chuỗi cụ thể không tồn tại
        trong dict -> trả None (KHÔNG raise) — dùng để tool tự phát hiện dataset_name
        sai và trả lỗi rõ ràng, xem tools/dataframe_tools.py.
        """
        if name is None:
            if self._active_name is None:
                raise RuntimeError("No dataset loaded. Call load_csv() first.")
            return self._datasets[self._active_name]
        return self._datasets.get(name)

    def get_columns(self, name: str | None = None) -> list[str]:
        df = self.get_dataframe(name)
        return list(df.columns) if df is not None else []

    def get_numeric_columns(self, name: str | None = None) -> list[str]:
        df = self.get_dataframe(name)
        if df is None:
            return []
        return df.select_dtypes(include="number").columns.tolist()

    def get_datetime_source_column(self, name: str | None = None) -> str | None:
        """
        Tên cột gốc được tự động chọn làm nguồn datetime (xem derive_time_columns).
        None nếu không có cột nào đạt ngưỡng parse, tên không tồn tại, hoặc chưa load
        dataset nào.
        """
        key = name or self._active_name
        if key is None:
            return None
        return self._datetime_source_columns.get(key)

    def list_datasets(self) -> list[str]:
        return list(self._datasets.keys())

    def get_active_name(self) -> str | None:
        return self._active_name

    def switch_dataset(self, name: str) -> bool:
        """Đổi active sang dataset đã tồn tại trong dict, KHÔNG load lại file. Trả
        False nếu tên không tồn tại (không raise), giữ active nguyên trạng."""
        if name not in self._datasets:
            return False
        self._active_name = name
        return True

    def remove_dataset(self, name: str) -> bool:
        """Xóa 1 dataset. Nếu xóa đúng active, tự chuyển active sang dataset còn lại
        gần nhất (theo thứ tự chèn) hoặc None nếu không còn gì. Trả False nếu tên
        không tồn tại (không raise)."""
        if name not in self._datasets:
            return False
        del self._datasets[name]
        self._source_paths.pop(name, None)
        self._datetime_source_columns.pop(name, None)
        if self._active_name == name:
            remaining = list(self._datasets.keys())
            self._active_name = remaining[-1] if remaining else None
        return True
