"""
Function schemas theo chuẩn OpenAI-compatible tool calling.
Model qua OpenRouter chỉ nhìn thấy các schema này — không biết Pandas/Chroma bên trong.
"""

GET_DATASET_SUMMARY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_dataset_summary",
        "description": "Lấy thông tin tổng quan về dataset đang tải: số dòng, danh sách cột, khoảng thời gian, các cột số. Dùng khi cần biết dataset có gì trước khi phân tích sâu hơn.",
        "parameters": {
            "type": "object",
            "properties": {},
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
            },
            "required": ["query"],
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
]
