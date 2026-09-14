# Finding 02: Unsupervised ML Không Tái Tạo Được 8 Nhãn Rule-Based Meteodyn Cho Phát Hiện Bất Thường

**Dataset:** his_data_GT1.csv (Site 1, 233.953 dòng) + WindFarm04_WT1.csv (Site 2, 261.525-262.796 dòng tùy bước lọc valid_mask)
**Ngày đo lần đầu (Isolation Forest lần 1):** 2026-09-08
**Ngày đo lại chính thức (dùng cho finding này):** 2026-09-11 — Isolation Forest lần 1 đo lại + lần 2 + K-Means lần 3, tất cả chạy trên `tools/turbine_diagnostic_tools.py` hiện tại, scikit-learn 1.9.1
**Trạng thái:** Kết luận định tính vững (re-verify độc lập, 2 lần đo hội tụ cùng hướng); số liệu định lượng chính xác nhạy nhẹ với phiên bản thư viện — xem Limitation.

---

## 1. Problem

`classify_turbine_status()` gán 8 nhãn vận hành (theo Meteodyn PO024/WPA) cho từng dòng bằng **rule-based thuần** (ngưỡng % công suất, độ lệch so với power curve theo dải gió). Câu hỏi: có nên bổ sung/thay thế một phần bằng ML không giám sát (không biết trước nhãn) để (a) validate độc lập xem rule-based có phản ánh đúng cấu trúc tự nhiên của dữ liệu không, và/hoặc (b) tự động phát hiện bất thường mà rule hiện tại có thể bỏ sót?

## 2. Expected Behavior

Nếu 8 nhãn Meteodyn phản ánh đúng các trạng thái vận hành vật lý thực sự khác biệt (không phải ranh giới tùy ý), một thuật toán ML không giám sát đủ tốt — dù KHÔNG được biết trước nhãn — vẫn nên tự nhiên tách dữ liệu thành các nhóm/điểm bất thường khớp cao với 8 nhãn này:
- **Isolation Forest:** điểm nó tự flag "anomaly" nên trùng phần lớn (Gate 4a: overlap ≥80%) với baseline độc lập (z-score) VÀ mang ý nghĩa vật lý thật (đa số điểm flag riêng có classification ≠ Normal).
- **K-Means:** cụm tự nhiên (không biết nhãn) nên có Adjusted Rand Index cao (>0,5) so với 8 nhãn Meteodyn.

## 3. Finding — 3 lần thử, đều không đạt kỳ vọng, 3 nguyên nhân khác nhau

### 3.1 Isolation Forest lần 1 — feature (power, wind_speed) thô [+ pitch cho GT1]

**Method:** gọi trực tiếp `detect_anomalies_ml()` (`tools/turbine_diagnostic_tools.py`) — logic Gate 4a thật, không giả lập. Feature = `[power_column, wind_speed_column]` (+ `pitch_column="GT1.Pitch.Angle1"` cho GT1). Gate: overlap ≥80% với baseline z-score (threshold=3.0) trên power thô, VÀ ≥3/10 điểm "IF-only" (flag riêng, z-score bỏ sót) có classification ≠ Normal.

**Kết quả đo lại 2026-09-11** (gọi hàm production thật, không phải script khảo sát):

| Site | overlap_pct_with_zscore | Gate overlap (≥80%) | meaningful_count/10 | Gate ý nghĩa (≥3) | Kết luận Gate 4a |
|---|---|---|---|---|---|
| GT1 | 0,00% | ❌ FAIL | 7/10 | ✅ PASS | FAIL |
| WT1 | 4,51% | ❌ FAIL | 5/10 | ✅ PASS | FAIL |

GT1 overlap=0,00% **không phải** do IF flag 0 điểm, mà do baseline z-score bị **suy biến**: std(`GT1.Grid.totW`) ≈ 1530, quá lớn so với range thực tế (-44,5 → 4525,2) → `|z|>3` không bao giờ xảy ra trên toàn bộ 233.953 dòng, baseline flag đúng 0 điểm. Đây là bằng chứng baseline hỏng cho chính cột/ngưỡng này, không phải bằng chứng IF kém.

**Nguyên nhân thất bại (phần overlap thật — không suy biến, cả 2 site):** feature (power, wind_speed) thô có tương quan mạnh (power phụ thuộc wind_speed theo đường cong phi tuyến) → Isolation Forest khó phân biệt "biến thiên bình thường dọc power curve" với "bất thường thật", nên tập điểm nó flag gần như độc lập ngẫu nhiên so với baseline z-score (vốn chỉ nhìn 1 chiều power).

**Cách tái tạo:** gọi `detect_anomalies_ml(manager, power_column, wind_speed_column, classification, dataset_name=..., pitch_column=...)` trên dataset thật — cùng tham số với `tests/test_turbine_diagnostic_tools.py::test_integration_detect_anomalies_ml_on_real_gt1_falls_back_gate_fails_as_measured` (test đó chỉ assert `ok is True`, không khóa số; bảng trên đo bằng cách gọi hàm y hệt rồi in field `gate_4a` của kết quả).

### 3.2 Isolation Forest lần 2 — feature deviation_from_curve (1D / 2D)

**Method:** script khảo sát (KHÔNG commit vào repo, KHÔNG tích hợp vào `detect_anomalies_ml()`) — train `IsolationForest(contamination=%-không-Normal, random_state=42, n_jobs=-1)` trên `deviation_from_curve` (= `power - median_power_tại_bin`, công thức TÁI DÙNG y hệt `classify_turbine_status()`), thử 2 biến thể: 1D (chỉ deviation) và 2D (deviation + wind_speed, chuẩn hóa `StandardScaler`). So với 2 baseline z-score **độc lập**: baseline A = z-score trên power thô (như lần 1), baseline B = z-score mới trên chính `deviation_from_curve`.

**Kết quả đo lại 2026-09-11:**

| Site | Variant | contamination | % điểm IF flag | overlap vs baseline A (power) | overlap vs baseline B (deviation) |
|---|---|---|---|---|---|
| GT1 | 1D (deviation) | 0,2285 | 22,83% | 0,00% | 6,28% |
| GT1 | 2D (deviation+wind_speed) | 0,2285 | 22,84% | 0,00% | 6,28% |
| WT1 | 1D (deviation) | 0,2810 | 27,94% | 2,48% | 3,49% |
| WT1 | 2D (deviation+wind_speed) | 0,2810 | 27,96% | 7,01% | 3,48% |

Cả 4 tổ hợp đều **xa dưới 80%** (0,00%-7,01%). GT1 baseline A (power) vẫn suy biến như lần 1 (cùng lý do cột `GT1.Grid.totW`); baseline B (deviation) KHÔNG suy biến (flag 1,43% điểm ở GT1, 0,97% ở WT1) nhưng overlap với IF vẫn chỉ 3,48%-6,28%.

**Nguyên nhân thất bại (cấu trúc, không chỉ do chọn sai feature):** contamination ước lượng từ %-không-Normal trong classification (22,85% GT1, 28,10% WT1) — gồm CẢ trạng thái vận hành hợp lệ (Curtailment/Stop/Partial Stop...), không chỉ điểm bất thường thật. Isolation Forest bị buộc flag 22,8%-28,0% tổng số điểm, trong khi cả 2 baseline z-score chỉ flag <2% (đuôi cực trị thống kê thật). Hai phương pháp đo 2 khái niệm khác bản chất ("~1/4-1/3 dữ liệu lệch nhẹ" vs "điểm cực trị hiếm") — overlap thấp gần như là hệ quả tất yếu của cách ước lượng contamination hiện tại, không chỉ do chọn sai không gian feature.

**Cách tái tạo:** `deviation = power_column - median_power_tại_bin` (tra `median_power` từ `power_curve` trả về bởi `estimate_power_curve()`, bin theo đúng `bin_width`) → `StandardScaler` (chỉ biến thể 2D) → `IsolationForest(contamination=round(clip(%-không-Normal, 0.001, 0.5), 4), random_state=42, n_jobs=-1)`. `%-không-Normal` lấy từ `classify_turbine_status()["data"]["summary"]` (= 1 − % nhãn "Normal").

### 3.3 K-Means (n_clusters=8) — feature deviation_from_curve + wind_speed [+ pitch/yaw cho GT1]

**Method:** script khảo sát (KHÔNG commit, KHÔNG tích hợp) — `KMeans(n_clusters=8, random_state=42, n_init=10)` trên feature đã `StandardScaler`, **không dùng gate đạt/không đạt** như Isolation Forest — chỉ so sánh khách quan bằng Adjusted Rand Index (`sklearn.metrics.adjusted_rand_score`) + confusion matrix (8 cụm × 8 nhãn).

Bộ nhớ đỉnh đo được (theo dõi bằng watchdog riêng, poll RSS mỗi giây): **~0,18-0,19 GB** cho cả 2 site (huấn luyện 2,4s GT1 / 6,5s WT1) — xác nhận đúng dự đoán O(n×k) nhẹ hơn hẳn DBSCAN, vốn từng phình tới 36,8GB trước khi bị kill (xem Limitation).

**Kết quả đo 2026-09-11:**

| Site | n_points | Feature | ARI | Đánh giá |
|---|---|---|---|---|
| GT1 | 233.953 | deviation, wind_speed, pitch, yaw | **0,0104** | < 0,2 — giới hạn đáng ghi nhận |
| WT1 | 261.525 | deviation, wind_speed | **0,0455** | < 0,2 — giới hạn đáng ghi nhận |

GT1: 6/8 cụm bị "Normal" chiếm ưu thế (purity 61-93%), không cụm nào bắt riêng 1 nhãn bất thường cụ thể. Cụm lẫn lộn nhất (2.514 điểm, cụm 6): Curtailment/Measurement Error/Stop/Under Production trộn lẫn, purity chỉ 41,65%.

**Nguyên nhân thất bại:** hình học cụm K-Means dựa trên khoảng cách Euclid (xu hướng cụm hình cầu, đối xứng) không khớp logic ngưỡng/độ lệch-theo-dải-gió (bất đối xứng, biên rule cứng theo bin) của rule-based Meteodyn.

**Cách tái tạo:** feature giống mục 3.2 biến thể 2D, cộng `pitch_column="GT1.Pitch.Angle1"` + `get_yaw_error(df, "site1")` (chỉ cho riêng GT1) → `StandardScaler` → `KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict()` → `adjusted_rand_score(nhãn_Meteodyn, cluster_labels)`.

### 3.4 Điểm sáng — không bỏ qua: WT1, trạng thái "Stop" đủ tách biệt vật lý

Trong K-Means WT1 (mục 3.3), 2 cụm nhỏ khớp **rất tốt** với đúng 1 nhãn "Stop":
- Cụm 2: 696 điểm, **666 điểm = Stop (95,69% purity)**
- Cụm 7: 1.284 điểm, **1.150 điểm = Stop (89,56% purity)**

Trong khi 6/8 cụm còn lại đều lẫn lộn Normal với các nhãn khác (purity 48-98% nhưng không riêng biệt 1 nhãn bất thường nào). → Trạng thái "Stop" (công suất ≈0) đủ tách biệt về vật lý để **một phương pháp không giám sát, không biết trước nhãn, tự nhận ra nó như một cụm riêng** — củng cố độ tin cậy của nhãn "Stop" qua một cách đo hoàn toàn độc lập với rule-based.

## 4. Hypotheses (đã kiểm định tuần tự qua 3 lần thử)

| # | Giả thuyết | Kỳ vọng nếu ĐÚNG |
|---|---|---|
| H0 | 8 nhãn Meteodyn phản ánh đúng cấu trúc vật lý tự nhiên của dữ liệu | ML không giám sát (không biết nhãn) sẽ tự nhiên khớp cao với 8 nhãn |
| H1 | (Sau IF lần 1 fail) Nguyên nhân là feature (power, wind_speed) thô tương quan mạnh theo curve | Đổi sang `deviation_from_curve` (đã loại tương quan curve) sẽ cải thiện overlap rõ rệt |
| H2 | (Sau IF lần 2 vẫn fail dù đổi feature) Nguyên nhân nằm ở cách ước lượng contamination, không phải feature | Đổi sang thuật toán KHÔNG cần contamination (K-Means) sẽ cho kết quả tốt hơn hẳn |
| H3 | (Sau cả 2 lần IF đều fail vì lý do "thống kê") Hình học cụm không khớp bản chất rule-based | K-Means (centroid-based, không cần contamination) vẫn cho ARI thấp; nhưng trạng thái vật lý tách biệt rõ (Stop) vẫn được nhận ra |

## 5. Evidence Matrix

| Giả thuyết | Evidence cần | Kết quả đo | Kết luận |
|---|---|---|---|
| H0 (ML không giám sát khớp rule-based) | Overlap ≥80% (IF) hoặc ARI >0,5 (K-Means) ở ít nhất 1 trong 3 lần thử | Cả 3 lần: overlap 0,00%-7,01% / ARI 0,0104-0,0455 | ❌ **Bác bỏ mạnh** — không lần nào tới gần ngưỡng |
| H1 (feature thô là nguyên nhân) | Đổi feature (`deviation_from_curve`) cải thiện overlap rõ rệt | Lần 2 vẫn 0,00%-7,01% — không cải thiện đáng kể so với lần 1 (0,00%-4,51%) | ❌ **Bác bỏ** — không phải chỉ do feature |
| H2 (contamination là nguyên nhân duy nhất) | K-Means (không cần contamination) cho kết quả (ARI) cao hơn hẳn overlap của IF | ARI (0,01-0,05) còn THẤP HƠN cả overlap IF (0,00-7,01%) | ❌ **Bác bỏ** — không phải chỉ do contamination |
| H3 (hình học cụm không khớp rule-based) | Cả 3 lần thử (2 cơ chế khác nhau: isolation, centroid) đều fail tổng thể, NHƯNG trạng thái vật lý tách biệt rõ (Stop) vẫn được nhận ra | Đúng — 3/3 fail tổng thể, nhưng Stop được K-Means WT1 bắt tốt (89,56%-95,69% purity, mục 3.4) | ✅ **Ủng hộ** — thất bại có chọn lọc, không phải nhiễu ngẫu nhiên hoàn toàn |

## 6. Root Cause

3 lần thử thất bại vì **3 nguyên nhân kỹ thuật khác nhau** (tương quan feature theo curve → contamination ước lượng sai bản chất → hình học cụm Euclid không khớp) nhưng hội tụ về **cùng một kết luận**: rule-based Meteodyn định nghĩa "bất thường" bằng **ngưỡng/độ lệch theo dải gió** (wind-speed-bucketed threshold) — một khái niệm không tương đương với "khoảng cách" hay "mật độ" trong không gian feature phẳng mà cả isolation-based (Isolation Forest) lẫn centroid-based (K-Means) đều ngầm giả định. Điểm sáng ở mục 3.4 (Stop) củng cố thêm luận điểm này: khi một trạng thái vận hành đủ tách biệt vật lý (công suất ≈0, không phụ thuộc dải gió), ranh giới rule-based VÀ ranh giới hình học/mật độ trùng nhau tự nhiên — chứng tỏ thất bại ở các nhãn còn lại không phải do ML "kém" nói chung, mà do đúng bản chất hình học không khớp với ranh giới rule cụ thể của các nhãn đó (Under/Over Production, Partial Stop...).

## 7. Limitation

- **Chỉ thử 2 họ thuật toán hoàn chỉnh** (isolation-based: Isolation Forest × 2 lần; centroid-based: K-Means × 1 lần). **DBSCAN (density-based) đã thử nhưng KHÔNG cho ra bằng chứng khoa học** — dừng do `MemoryError` thật (RSS phình tới ~36,8GB trước khi bị chủ động kill), nguyên nhân là độ phức tạp CẤU TRÚC O(n²) của thuật toán trên n>233K dòng (không phải lỗi cấu hình `n_jobs`). Đây là **giới hạn hạ tầng**, khác bản chất với 3 lần thử ở mục 3 (vốn chạy xong hoàn chỉnh, cho ra số liệu so sánh được) — không nên coi DBSCAN là "lần thử thứ 4 fail", nó đơn giản chưa từng chạy xong để có bằng chứng.
- **Chưa thử feature engineering sâu hơn** — ví dụ chuyển power curve sang tọa độ cực (polar coordinates) hoặc chuẩn hóa theo bin gió trước khi đưa vào thuật toán, để biến "khoảng cách theo đường cong phi tuyến" thành khoảng cách Euclid thông thường mà K-Means/Isolation Forest ngầm giả định.
- **Số liệu định lượng nhạy nhẹ với phiên bản thư viện:** lần đo lại 2026-09-11 (scikit-learn 1.9.1) không khớp bit-for-bit với lần đo gốc 2026-09-08 của Isolation Forest lần 1 — GT1 khớp tuyệt đối (0,00%, meaningful 7/10), nhưng WT1 lệch (4,51% đo lại vs 5,38% ghi nhận ban đầu). Nguyên nhân nhiều khả năng nhất: venv dự án không hề cài scikit-learn cho tới thời điểm đo lại (2026-09-11), nên lần đo gốc 2026-09-08 chắc chắn chạy trên một môi trường/phiên bản sklearn khác — `IsolationForest` không đảm bảo tái lập bit-for-bit giữa các phiên bản dù cùng `random_state=42`. **Kết luận định tính hoàn toàn không đổi** qua cả 2 lần đo (Gate 4a fail rất xa ngưỡng 80% ở mọi trường hợp) — chỉ phần trăm chính xác dao động nhẹ. Finding này dùng bản đo lại 2026-09-11 làm số liệu **chính thức** (có script/công thức tái tạo ở mục 3.1-3.3); số 2026-09-08 vẫn giữ nguyên trong docstring `detect_anomalies_ml()` dưới dạng ghi chú lịch sử, không bị xóa.
- Isolation Forest lần 2 và K-Means chạy bằng script khảo sát KHÔNG commit vào repo (đúng ràng buộc đặt ra cho mỗi lần thử) — công thức tái tạo đầy đủ được mô tả ở mục 3.2/3.3 để bất kỳ ai cũng viết lại được script tương đương và tự verify.

## 8. Recommendation

1. **Giữ nguyên rule-based (`classify_turbine_status()`) làm phương pháp chính cho phân loại/phát hiện bất thường** — đã validate 85-98% khớp `OpSt` cho 5/8 nhãn (xem `docs/turbine_data_audit.md`), và 3 lần thử ML không giám sát độc lập đều không tìm ra cấu trúc khớp tốt hơn.
2. **`predict_power()` (XGBoost) vẫn là bằng chứng ML được áp dụng thành công** trên chính 2 dataset này — đạt Gate 4b/4c rõ ràng (RMSE thấp hơn baseline, wind_speed rank #1 importance cả 2 site). Anomaly detection không dùng được ML không có nghĩa "ML không hợp với turbine data" nói chung — chỉ có nghĩa bài toán CỤ THỂ này (phát hiện bất thường không giám sát, so với ranh giới rule-based hiện có) không phù hợp với 2 họ thuật toán đã thử.
3. Phần anomaly detection tiếp tục dùng rule-based đã validate, **không phải vì "chưa thử ML"** mà vì đã thử nghiêm túc 3 lần, có bằng chứng số liệu rõ ràng, tái lập được (xem mục 3).
4. Nếu muốn thử lại lần 4 trong tương lai: ưu tiên feature engineering (polar coordinates cho power curve) trước khi đổi thuật toán lần nữa — đã loại trừ 2 nguyên nhân "dễ sửa" (feature thô, contamination), nguyên nhân còn lại (hình học cụm) cần thay đổi cách biểu diễn dữ liệu, không chỉ đổi tham số.
