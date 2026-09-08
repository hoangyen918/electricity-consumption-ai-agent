"""
Function schemas theo chuẩn OpenAI-compatible tool calling.
Model qua OpenRouter chỉ nhìn thấy các schema này — không biết Pandas/Chroma bên trong.
"""

DATASET_NAME_PROPERTY = {
    "type": "string",
    "description": (
        "Tên file dataset cần thao tác. Nếu không chỉ định, dùng dataset được upload "
        "gần nhất. Gọi get_dataset_summary để xem danh sách dataset hiện có."
    ),
}

GET_DATASET_SUMMARY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_dataset_summary",
        "description": "Lấy thông tin tổng quan về dataset đang tải: số dòng, danh sách cột, khoảng thời gian, các cột số. Dùng khi cần biết dataset có gì trước khi phân tích sâu hơn. Nếu không chỉ định dataset_name, kết quả còn kèm danh sách TẤT CẢ dataset đã upload để chọn cho các lần gọi tool tiếp theo.",
        "parameters": {
            "type": "object",
            "properties": {
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

AGGREGATE_METRIC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "aggregate_metric",
        "description": "Tính giá trị tổng hợp (trung bình, tổng, min, max, median, độ lệch chuẩn, đếm) của 1 cột số, có thể lọc theo khoảng thời gian.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {
                    "type": "string",
                    "description": "Tên cột cần tính (phải là cột có thật trong dataset, ví dụ: zone_1_consumption, temperature)",
                },
                "aggregation": {
                    "type": "string",
                    "enum": ["mean", "sum", "min", "max", "median", "std", "count"],
                },
                "start_date": {
                    "type": "string",
                    "description": "Ngày bắt đầu định dạng YYYY-MM-DD, để trống nếu không lọc thời gian",
                },
                "end_date": {
                    "type": "string",
                    "description": "Ngày kết thúc định dạng YYYY-MM-DD, để trống nếu không lọc thời gian",
                },
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": ["column", "aggregation"],
        },
    },
}

GROUP_METRIC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "group_metric",
        "description": "Gom nhóm dữ liệu theo giờ/ngày/tháng/thứ trong tuần/năm và tính giá trị tổng hợp cho mỗi nhóm. Dùng khi câu hỏi cần so sánh theo thời gian, ví dụ: tháng nào cao nhất.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "aggregation": {
                    "type": "string",
                    "enum": ["mean", "sum", "min", "max", "median", "std", "count"],
                },
                "group_by": {
                    "type": "string",
                    "enum": ["hour", "day", "month", "weekday", "year", "date"],
                },
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": ["column", "aggregation", "group_by"],
        },
    },
}

FIND_PEAK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_peak",
        "description": "Tìm giá trị cao nhất hoặc thấp nhất của 1 cột, kèm thời điểm xảy ra. Dùng cho câu hỏi kiểu 'đạt đỉnh khi nào', 'thấp nhất bao nhiêu'.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "mode": {"type": "string", "enum": ["max", "min"]},
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": ["column"],
        },
    },
}

COMPARE_PERIODS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compare_periods",
        "description": "So sánh giá trị tổng hợp của 1 cột giữa 2 khoảng thời gian khác nhau, trả về chênh lệch tuyệt đối và phần trăm thay đổi.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "period_a_start": {"type": "string", "description": "YYYY-MM-DD"},
                "period_a_end": {"type": "string", "description": "YYYY-MM-DD"},
                "period_b_start": {"type": "string", "description": "YYYY-MM-DD"},
                "period_b_end": {"type": "string", "description": "YYYY-MM-DD"},
                "aggregation": {
                    "type": "string",
                    "enum": ["mean", "sum", "min", "max", "median", "std", "count"],
                    "default": "mean",
                },
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": ["column", "period_a_start", "period_a_end", "period_b_start", "period_b_end"],
        },
    },
}

CALCULATE_CORRELATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate_correlation",
        "description": "Tính hệ số tương quan Pearson giữa 2 cột số, ví dụ giữa nhiệt độ và tiêu thụ điện.",
        "parameters": {
            "type": "object",
            "properties": {
                "column_a": {"type": "string"},
                "column_b": {"type": "string"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD, để trống nếu không lọc"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD, để trống nếu không lọc"},
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": ["column_a", "column_b"],
        },
    },
}

CALCULATE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Thực hiện phép tính đơn giản (cộng, trừ, nhân, chia, phần trăm của, phần trăm thay đổi). Dùng cho tính toán không liên quan trực tiếp tới dataset, ví dụ tính X% của 1 con số cụ thể.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "percentage_of", "percentage_change"],
                },
                "value_a": {"type": "number"},
                "value_b": {"type": "number"},
            },
            "required": ["operation", "value_a", "value_b"],
        },
    },
}

SEARCH_KNOWLEDGE_BASE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Tìm kiếm thông tin trong kho tài liệu (báo cáo thường niên EVN). Dùng khi câu hỏi cần tra cứu thông tin từ tài liệu chính thức, không phải tính toán từ dataset CSV.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Câu hỏi hoặc từ khóa cần tìm"},
                "top_k": {"type": "integer", "description": "Số kết quả muốn lấy, mặc định 8"},
                "language": {
                    "type": "string",
                    "description": (
                        "Lọc theo ngôn ngữ gốc của tài liệu, ví dụ 'en' (báo cáo thường niên EVN, "
                        "tiếng Anh) hoặc 'vi' (văn bản pháp luật tiếng Việt như Chỉ thị). Để trống "
                        "nếu câu hỏi không cần giới hạn theo ngôn ngữ tài liệu."
                    ),
                },
                "document_type": {
                    "type": "string",
                    "description": (
                        "Lọc theo loại tài liệu. Giá trị hợp lệ: 'annual_report' (báo cáo thường niên "
                        "EVN), 'policy_directive' (văn bản quy phạm pháp luật, ví dụ Chỉ thị của Thủ "
                        "tướng Chính phủ). Dùng khi câu hỏi rõ ràng chỉ hỏi về quy định/chính sách "
                        "(nên đặt 'policy_directive') hoặc chỉ về số liệu kinh doanh của EVN (nên đặt "
                        "'annual_report'). Để trống nếu câu hỏi không rõ ràng thuộc loại nào."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

TURBINE_POWER_COLUMN_PROPERTY = {
    "type": "string",
    "description": (
        "Tên cột công suất. Để trống để TỰ NHẬN DIỆN theo cấu trúc dataset (GT1.Grid.totW "
        "cho Site 1/his_data_GT1, ACTIVE_POWER cho Site 2/WindFarm04_WT*) — chỉ cần chỉ định "
        "nếu muốn dùng cột khác."
    ),
}
TURBINE_WIND_SPEED_COLUMN_PROPERTY = {
    "type": "string",
    "description": (
        "Tên cột tốc độ gió. Để trống để TỰ NHẬN DIỆN (GT1.Common.WindSpd cho Site 1, "
        "WIND_SPEED cho Site 2) — chỉ cần chỉ định nếu muốn dùng cột khác."
    ),
}
TURBINE_BIN_WIDTH_PROPERTY = {
    "type": "number",
    "description": "Độ rộng bin tốc độ gió (m/s) khi ước lượng power curve. Mặc định 1.0 (đã validate thật trên dữ liệu turbine gió) — thường không cần đổi.",
}
TURBINE_SETPOINT_COLUMN_PROPERTY = {
    "type": "string",
    "description": (
        "Tên cột setpoint công suất (GT1.SP.ActualSPP) — CHỈ Site 1/his_data_GT1 có cột "
        "này. Cần thiết để phát hiện đúng trạng thái Curtailment/Partial Curtailment; bỏ "
        "trống hoàn toàn nếu đang phân tích Site 2 (WindFarm04_WT*, không có setpoint)."
    ),
}

# --- Nhóm generic ---

CLEAN_MISSING_VALUES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "clean_missing_values",
        "description": "Kiểm tra/xử lý dữ liệu thiếu (missing values) trong dataset — dùng cho MỌI dataset, không riêng turbine gió. strategy='report' (mặc định) chỉ báo cáo % thiếu theo từng cột, KHÔNG đổi dữ liệu. Các strategy khác (drop/ffill/interpolate) chỉ trả về preview kết quả nếu áp dụng, KHÔNG ghi đè dataset thật.",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Cột cụ thể cần kiểm tra/xử lý. Để trống để áp dụng cho toàn bộ dataset (chỉ có ý nghĩa với strategy='report')."},
                "strategy": {"type": "string", "enum": ["report", "drop", "ffill", "interpolate"], "description": "Mặc định 'report' — chỉ báo cáo, không đổi dữ liệu."},
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

DETECT_DUPLICATES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_duplicates",
        "description": "Đếm số dòng trùng lặp trong dataset — dùng cho MỌI dataset. Có thể giới hạn so sánh theo 1 nhóm cột cụ thể (subset_cols) thay vì toàn bộ dòng.",
        "parameters": {
            "type": "object",
            "properties": {
                "subset_cols": {"type": "array", "items": {"type": "string"}, "description": "Chỉ so sánh trùng lặp theo các cột này (vd chỉ theo timestamp). Để trống để so sánh toàn bộ dòng."},
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

DETECT_STATISTICAL_ANOMALIES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_statistical_anomalies",
        "description": "Phát hiện giá trị bất thường thống kê (outlier) trong 1 cột số bất kỳ — dùng cho MỌI dataset, phương pháp z-score hoặc IQR thuần thống kê (không hiểu ngữ nghĩa vật lý của dữ liệu). Dùng khi câu hỏi chung chung về 'giá trị bất thường/outlier' trong 1 cột cụ thể; với dataset turbine gió, ưu tiên detect_physical_violations/detect_sensor_issues nếu câu hỏi về bất thường VẬT LÝ (công suất âm, cảm biến lỗi...).",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Cột số cần kiểm tra"},
                "method": {"type": "string", "enum": ["zscore", "iqr"], "description": "Mặc định 'zscore'"},
                "threshold": {"type": "number", "description": "Ngưỡng z-score (mặc định 3.0). Không áp dụng cho method='iqr' (luôn dùng hệ số IQR chuẩn 1.5)."},
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": ["column"],
        },
    },
}

CORRELATION_ANALYSIS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "correlation_analysis",
        "description": "Tính correlation của 1 cột với TẤT CẢ cột số khác trong dataset, sắp theo độ mạnh — dùng cho MỌI dataset khi câu hỏi muốn biết 'cột nào liên quan/ảnh hưởng nhiều nhất tới X', khác calculate_correlation (chỉ giữa đúng 2 cột được chỉ định trước).",
        "parameters": {
            "type": "object",
            "properties": {
                "target_column": {"type": "string", "description": "Cột muốn tìm các cột khác liên quan tới nó"},
                "top_n": {"type": "integer", "description": "Số cột liên quan mạnh nhất muốn lấy (mặc định 10)"},
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": ["target_column"],
        },
    },
}

EXPLAIN_ANOMALY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "explain_anomaly",
        "description": "So sánh đặc điểm (mean/median/std hoặc phân bố %) giữa 1 nhóm dòng cụ thể (bất thường) và phần còn lại của dataset — dùng cho MỌI dataset. Thường dùng làm bước THEO SAU 1 tool phát hiện bất thường khác (vd detect_statistical_anomalies, detect_physical_violations) đã cho ra danh sách row_indices cụ thể, để tìm hiểu điểm bất thường đó khác biệt ở đâu. KHÔNG tự kết luận nguyên nhân, chỉ mô tả.",
        "parameters": {
            "type": "object",
            "properties": {
                "row_indices": {"type": "array", "items": {"type": "integer"}, "description": "Danh sách row_index cụ thể coi là 'bất thường' (thường lấy từ kết quả 1 tool khác trước đó)"},
                "anomaly_mask_column": {"type": "string", "description": "HOẶC dùng tên 1 cột boolean có sẵn trong dataset đánh dấu bất thường, thay vì row_indices (chỉ dùng ĐÚNG 1 trong 2)"},
                "compare_columns": {"type": "array", "items": {"type": "string"}, "description": "Các cột muốn so sánh. Để trống để so sánh tất cả cột số."},
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

# --- Nhóm turbine-specific ---

DETECT_PHYSICAL_VIOLATIONS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_physical_violations",
        "description": "CHỈ dùng cho dataset turbine gió (his_data_GT1 hoặc WindFarm04_WT*). Kiểm tra 6 quy tắc vật lý: công suất âm bất thường (magnitude-aware, không tính house-load bình thường), công suất khi gió dưới/trên ngưỡng cắt, lệch pitch 3 lưỡi (Site 1), lệch 2 cảm biến gió (Site 2), nhiệt độ vượt ngưỡng báo động (Site 2). Tự nhận diện site, tự bỏ qua rule không áp dụng.",
        "parameters": {
            "type": "object",
            "properties": {"dataset_name": DATASET_NAME_PROPERTY},
            "required": [],
        },
    },
}

DETECT_SENSOR_ISSUES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_sensor_issues",
        "description": "CHỈ dùng cho dataset turbine gió. Phát hiện 3 loại lỗi cảm biến/thu thập dữ liệu: flatline (cảm biến đứng, giá trị y hệt nhiều bước liên tiếp), khoảng trống thời gian (timestamp gap), và giá trị placeholder bất khả thi vật lý (vd -273.2°C, nhiệt độ 0 đồng thời nhiều cột — chỉ Site 2 có cột nhiệt độ).",
        "parameters": {
            "type": "object",
            "properties": {
                "flatline_min_length": {"type": "integer", "description": "Số bước liên tiếp tối thiểu để coi là flatline (mặc định 6)"},
                "gap_multiplier": {"type": "number", "description": "Hệ số nhân với timestep kỳ vọng để xác định gap bất thường (mặc định 3.0)"},
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

DIAGNOSE_THEORETICAL_POWER_DEVIATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "diagnose_theoretical_power_deviation",
        "description": "CHỈ dùng cho dataset his_data_GT1 (Site 1) — dataset DUY NHẤT có cột GT1.Common.TheoActiveW (công suất lý thuyết). So sánh nhóm dòng có công suất thực tế vượt lý thuyết hơn threshold_pct% với nhóm còn lại, trên các field liên quan (trạng thái vận hành, pitch, yaw, setpoint, thời gian). Dùng khi câu hỏi hỏi TẠI SAO công suất vượt lý thuyết. KHÔNG tự kết luận nguyên nhân, chỉ mô tả dữ liệu để agent/user tự diễn giải.",
        "parameters": {
            "type": "object",
            "properties": {
                "threshold_pct": {"type": "number", "description": "Ngưỡng % vượt lý thuyết để coi là vi phạm (mặc định 10.0)"},
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

ESTIMATE_POWER_CURVE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "estimate_power_curve",
        "description": "CHỈ dùng cho dataset turbine gió. Ước lượng power curve (công suất kỳ vọng theo từng dải tốc độ gió) bằng phương pháp iterative median binning, robust với dữ liệu nhiễu/bất thường. Dùng khi câu hỏi muốn biết 'power curve', 'đường cong công suất', hoặc làm BƯỚC ĐẦU TIÊN bắt buộc trước khi gọi classify_turbine_status (classify_turbine_status tự động tái sử dụng kết quả này, không cần truyền tay).",
        "parameters": {
            "type": "object",
            "properties": {
                "power_column": TURBINE_POWER_COLUMN_PROPERTY,
                "wind_speed_column": TURBINE_WIND_SPEED_COLUMN_PROPERTY,
                "bin_width": TURBINE_BIN_WIDTH_PROPERTY,
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

CLASSIFY_TURBINE_STATUS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "classify_turbine_status",
        "description": (
            "CHỈ dùng cho dataset turbine gió. Phân loại từng thời điểm vận hành theo 8 nhãn "
            "Meteodyn: Measurement Error, Stop, Curtailment (chỉ Site 1, cần setpoint_column), "
            "Under/Over Production, Partial Stop, Partial Curtailment (chỉ Site 1), Normal. "
            "Dùng khi câu hỏi hỏi về PHÂN LOẠI TRẠNG THÁI vận hành. Tool này TỰ ĐỘNG gọi "
            "estimate_power_curve() nội bộ nếu chưa có (không cần gọi tay trước, dù gọi trước "
            "cũng không sao — kết quả được cache lại, không tính 2 lần). Kết quả trả về CHỈ "
            "gồm số liệu tổng hợp theo nhãn (không phải toàn bộ dữ liệu từng dòng)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "power_column": TURBINE_POWER_COLUMN_PROPERTY,
                "wind_speed_column": TURBINE_WIND_SPEED_COLUMN_PROPERTY,
                "setpoint_column": TURBINE_SETPOINT_COLUMN_PROPERTY,
                "bin_width": TURBINE_BIN_WIDTH_PROPERTY,
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

GROUP_ANOMALY_WINDOWS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "group_anomaly_windows",
        "description": (
            "CHỈ dùng cho dataset turbine gió. Gộp các thời điểm LIÊN TIẾP cùng nhãn bất "
            "thường (từ classify_turbine_status, tự động gọi nội bộ nếu chưa có) thành các "
            "KHOẢNG THỜI GIAN cụ thể (start/end/duration) — chỉ giữ khoảng đủ dài "
            "(min_consecutive_points điểm liên tiếp trở lên), loại nhiễu ngắn hạn lẻ tẻ. Dùng "
            "khi câu hỏi hỏi về khoảng thời gian/giai đoạn bị 1 vấn đề cụ thể kéo dài (vd "
            "'có curtailment kéo dài không', 'khoảng nào bị dừng lâu')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timestamp_column": {"type": "string", "description": "Tên cột thời gian, mặc định 'datetime' (cột chuẩn tự động derive cho mọi dataset)"},
                "power_column": TURBINE_POWER_COLUMN_PROPERTY,
                "wind_speed_column": TURBINE_WIND_SPEED_COLUMN_PROPERTY,
                "setpoint_column": TURBINE_SETPOINT_COLUMN_PROPERTY,
                "bin_width": TURBINE_BIN_WIDTH_PROPERTY,
                "min_consecutive_points": {"type": "integer", "description": "Số điểm liên tiếp tối thiểu để tính là 1 khoảng thời gian (mặc định 3)"},
                "exclude_labels": {"type": "array", "items": {"type": "string"}, "description": "Nhãn muốn loại khỏi kết quả, mặc định ['Normal'] (chỉ quan tâm khoảng bất thường)"},
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

DETECT_ANOMALIES_ML_SCHEMA = {
    "type": "function",
    "function": {
        "name": "detect_anomalies_ml",
        "description": (
            "CHỈ dùng cho dataset turbine gió. Thử phát hiện bất thường bằng Isolation Forest "
            "(machine learning), nhưng CHỈ thực sự dùng nếu vượt gate kiểm định so với "
            "detect_statistical_anomalies — nếu không đạt gate, tự động fallback về "
            "detect_statistical_anomalies (rule-based) và báo rõ lý do trong kết quả. LƯU Ý: "
            "đã đo thật trên GT1/WT1 — Gate KHÔNG đạt ở cả 2 site (xem field method_used/"
            "ml_gate_reason trong kết quả để biết đang dùng ML hay fallback), nên hiện tại tool "
            "này trong thực tế luôn trả kết quả z-score, không phải ML thật."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "power_column": TURBINE_POWER_COLUMN_PROPERTY,
                "wind_speed_column": TURBINE_WIND_SPEED_COLUMN_PROPERTY,
                "setpoint_column": TURBINE_SETPOINT_COLUMN_PROPERTY,
                "pitch_column": {"type": "string", "description": "Cột pitch dùng làm feature cho Isolation Forest (nếu có), vd GT1.Pitch.Angle1"},
                "wind_speed_std_column": {"type": "string", "description": "Cột độ lệch chuẩn tốc độ gió dùng làm feature (Site 2 có sẵn WIND_SPEED_STD)"},
                "bin_width": TURBINE_BIN_WIDTH_PROPERTY,
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

PREDICT_POWER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "predict_power",
        "description": (
            "CHỈ dùng cho dataset turbine gió. Dự đoán công suất từ tốc độ gió + feature khác "
            "(pitch/yaw/nhiệt độ) bằng XGBoost, so với baseline nội suy power curve — CHỈ dùng "
            "XGBoost nếu vượt cả 2 gate kiểm định (RMSE thấp hơn baseline VÀ wind_speed là "
            "feature quan trọng nhất/nhì), nếu không tự fallback về nội suy power curve. Đã đo "
            "thật: cả GT1 và WT1 đều vượt gate, dùng XGBoost thật. KHÔNG bao giờ dùng "
            "GT1.Common.TheoActiveW hay GT1.SP.ActualSPP làm feature (data leakage)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "power_column": TURBINE_POWER_COLUMN_PROPERTY,
                "wind_speed_column": TURBINE_WIND_SPEED_COLUMN_PROPERTY,
                "setpoint_column": TURBINE_SETPOINT_COLUMN_PROPERTY,
                "feature_columns": {"type": "array", "items": {"type": "string"}, "description": "Cột feature bổ sung ngoài wind_speed. Để trống để dùng bộ feature mặc định ĐÃ VALIDATE THẬT theo từng site."},
                "bin_width": TURBINE_BIN_WIDTH_PROPERTY,
                "dataset_name": DATASET_NAME_PROPERTY,
            },
            "required": [],
        },
    },
}

ALL_TOOL_SCHEMAS = [
    GET_DATASET_SUMMARY_SCHEMA,
    AGGREGATE_METRIC_SCHEMA,
    GROUP_METRIC_SCHEMA,
    FIND_PEAK_SCHEMA,
    COMPARE_PERIODS_SCHEMA,
    CALCULATE_CORRELATION_SCHEMA,
    CALCULATE_SCHEMA,
    SEARCH_KNOWLEDGE_BASE_SCHEMA,
    CLEAN_MISSING_VALUES_SCHEMA,
    DETECT_DUPLICATES_SCHEMA,
    DETECT_STATISTICAL_ANOMALIES_SCHEMA,
    CORRELATION_ANALYSIS_SCHEMA,
    EXPLAIN_ANOMALY_SCHEMA,
    DETECT_PHYSICAL_VIOLATIONS_SCHEMA,
    DETECT_SENSOR_ISSUES_SCHEMA,
    DIAGNOSE_THEORETICAL_POWER_DEVIATION_SCHEMA,
    ESTIMATE_POWER_CURVE_SCHEMA,
    CLASSIFY_TURBINE_STATUS_SCHEMA,
    GROUP_ANOMALY_WINDOWS_SCHEMA,
    DETECT_ANOMALIES_ML_SCHEMA,
    PREDICT_POWER_SCHEMA,
]
