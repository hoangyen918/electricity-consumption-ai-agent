# Finding 01: GT1 Actual Power Exceeds Theoretical Power (TheoActiveW) at Low-to-Mid Wind Speed

**Dataset:** his_data_GT1.csv (Site 1, single turbine, 5-min interval, 233.953 dòng)
**Ngày phát hiện:** trong quá trình Wind Turbine Phase 2 (data audit + diagnostic tools)
**Trạng thái:** Root cause có bằng chứng vừa phải (correlational), chưa xác nhận tuyệt đối

---

## 1. Problem

`GT1.Grid.totW` (công suất thực tế đo được) vượt `GT1.Common.TheoActiveW` (công suất lý thuyết, tính từ model nhà sản xuất theo điều kiện gió) quá 10% trên 1 tỷ lệ đáng kể số dòng — mức vượt này cần được hiểu rõ trước khi dùng `TheoActiveW` làm baseline cho các phép tính Loss/KPI ở phase sau.

## 2. Expected Behavior

`TheoActiveW` đại diện cho công suất lý thuyết turbine "nên" đạt được ở 1 tốc độ gió cho trước, theo model/power curve chuẩn của nhà sản xuất. Trong vận hành bình thường, công suất thực tế (`totW`) không nên **hệ thống** vượt xa giá trị này — sai lệch nhỏ ngẫu nhiên là bình thường, nhưng sai lệch lớn, tập trung theo pattern rõ ràng (không phải noise ngẫu nhiên) gợi ý có vấn đề cần điều tra: hoặc turbine hoạt động khác thường, hoặc chính `TheoActiveW`/phép đo có vấn đề.

## 3. Finding

**21.302 / 233.953 dòng (9.11%)** có `totW > TheoActiveW × 1.1` (vượt lý thuyết hơn 10%).

Sau khi loại 8 dòng suy biến (`TheoActiveW < 1.0`, gây deviation → vô cực), phân tích trên 21.294 dòng còn lại:
- Deviation trung bình: 18.88% | trung vị: 14.98% | std: 14.79%
- Khoảng: 10.0% (đúng ngưỡng) → 427% (một số điểm cực đoan)

## 4. Impact

9.11% dữ liệu Site 1 nằm trong nhóm này — đủ lớn để ảnh hưởng đáng kể tới bất kỳ phép tính Loss (`TheoActiveW - totW`) hoặc KPI dựa trên `TheoActiveW` làm baseline, đặc biệt ở dải gió thấp nơi mức vượt lớn nhất.

## 5. Localization

- **Turbine:** không áp dụng — Site 1 chỉ có 1 turbine (GT1), không có turbine khác để so sánh chéo
- **Wind-speed range:** tập trung rõ rệt ở gió **thấp-vừa** (xem Evidence)
- **Thời gian:** lệch nhẹ theo giờ (đỉnh giờ 16, ~60% cao hơn mức nền) và theo tháng (Aug/Oct/Sep/Feb/May cao hơn Jan/Dec/Nov) — nhưng **không đủ mạnh** để coi là driver chính (tỷ lệ max/min chỉ 2.6 lần, không có giai đoạn liên tục rõ ràng kiểu bảo trì)
- **Operating status:** 95.96% vi phạm rơi vào `OpSt=5` (Power Production) — hợp lý cơ học (lúc không phát điện thì `totW`≈0 nên khó vượt lý thuyết), không phải insight độc lập

## 6. Hypotheses

1. **Turbine hiệu suất vượt trội** khi điều kiện gió/vận hành tối ưu (pitch thấp, yaw canh chuẩn) — kỳ vọng: deviation % tăng theo wind speed
2. **Sai số đo lường/model bị khuếch đại ở gió thấp** — do power curve có dạng phi tuyến (gần bậc 3 dưới rated power), 1 sai số tuyệt đối cố định (đo gió, đồng bộ thời gian) sẽ tạo ra %deviation lớn hơn nhiều ở gió thấp (mẫu số nhỏ) so với gió cao (mẫu số lớn)
3. Pitch control bất thường
4. Yaw misalignment
5. Setpoint (`ActualSPP`) bị điều chỉnh bất thường

## 7. Evidence

| Hypothesis | Evidence cần | Output | Kết quả |
|---|---|---|---|
| (1) Turbine vượt trội khi gió mạnh | Correlation deviation vs WindSpd nên **dương** | Tercile: 23.77% (gió thấp) → 17.74% → 15.12% (gió cao) | ❌ **Bác bỏ** — deviation giảm dần, ngược chiều kỳ vọng |
| (2) Sai số khuếch đại ở gió thấp | Correlation deviation vs WindSpd âm, mạnh hơn hẳn các biến khác | r(WindSpd) = -0.29 (vừa phải) vs r(Pitch) = 0.04 (~0) | ✅ **Ủng hộ** |
| (3) Pitch control bất thường | Correlation deviation vs Pitch, độc lập sau khi kiểm soát WindSpd | Cross-tercile: deviation theo pitch gần như phẳng trong từng dải gió (chênh 1-2 điểm %, không đơn điệu) | ❌ **Bác bỏ** |
| (4) Yaw misalignment | Nhóm vi phạm có `WindTrackDevAng` bất thường (lệch lớn) | Vi phạm: std 4.68° (khoảng hẹp) vs bình thường: std 13.98° (toàn miền) — vi phạm canh gió CHUẨN hơn, không lệch hơn | ❌ **Bác bỏ** (ngược hướng giả thuyết) |
| (5) Setpoint bất thường | `ActualSPP` khác biệt rõ giữa 2 nhóm | Vi phạm: mean 3190.47 vs bình thường: mean 3257.98 (chênh ~2%) | ❌ **Bác bỏ** — không khác biệt đáng kể |

## 8. Validation

**Cross-check độc lập bằng phương pháp hoàn toàn khác:** `estimate_power_curve()` (iterative median binning trên **toàn bộ** 233.953 dòng, không chỉ tập con 21.302 dòng vi phạm) cho ra pattern **cùng hướng**:

| Wind speed (m/s) | SCADA vs TheoActiveW diff% |
|---|---|
| 3.5 | -15.0% |
| 5.5 | -5.5% |
| 7.5 | -1.7% |
| 9.5 | +0.8% |
| 11.5 – 17.5 | ±0.1 – 2.0% (khớp rất sát) |

Đường median toàn bộ dữ liệu hội tụ về gần 0 đúng ở dải gió trung-cao — khớp chính xác với tercile analysis trên tập con vi phạm. Hai phương pháp độc lập (subset-based deviation analysis và full-curve median estimation) hội tụ về cùng 1 kết luận → tăng đáng kể độ tin cậy.

## 9. Root Cause

**Nhiều khả năng nhất:** sai số đo lường hoặc mô hình hóa (không phải "turbine vượt trội"), khuếch đại thành %deviation lớn ở dải gió thấp do bản chất phi tuyến của power curve (dạng gần bậc 3 dưới rated power — đường dốc ở gió thấp-vừa, phẳng dần khi gần rated). Cùng 1 sai số tuyệt đối nhỏ (đo tốc độ gió, độ trễ đồng bộ giữa lúc đo gió và lúc turbine phản hồi công suất do quán tính rotor) sẽ biến thành %deviation lớn ở gió thấp và nhỏ ở gió cao — đúng khớp pattern quan sát được ở cả 2 phương pháp phân tích độc lập.

## 10. Limitation

- Đây là **bằng chứng tương quan (correlational)**, không phải chứng minh nhân quả trực tiếp — chưa có log calibration cảm biến, log hệ thống điều khiển turbine, hay tài liệu chính xác cách nhà sản xuất tính `TheoActiveW` để xác nhận dứt điểm cơ chế.
- Không loại trừ hoàn toàn khả năng chính `TheoActiveW` được tính từ 1 power curve chung (generic manufacturer curve) chưa hiệu chỉnh riêng cho điều kiện site cụ thể (mật độ không khí, độ cao, v.v.) — nếu vậy, phần "sai số" nằm ở baseline lý thuyết, không phải phép đo thực tế.
- Không có tài liệu kỹ thuật turbine để xác nhận liệu phần mềm điều khiển có cố ý "boost" công suất vượt nameplate curve trong 1 số điều kiện cụ thể hay không.

## 11. Recommendation

1. **Phase 3 classification (chiều nay):** dùng ngưỡng deviation theo **dải gió** (wind-speed-bucketed), không dùng % cố định toàn cục — vì baseline deviation tự nhiên khác nhau theo dải gió.
2. Nếu có cơ hội hỏi sếp/tài liệu OEM: xác nhận chính xác công thức tính `TheoActiveW` để xác nhận hoặc bác bỏ giả thuyết measurement-artifact.
3. Cân nhắc dùng power curve ước lượng từ SCADA thực tế (`estimate_power_curve()`, empirical) làm baseline bổ sung hoặc thay thế cho `TheoActiveW` (theoretical) ở dải gió thấp — nơi 2 đường phân kỳ nhiều nhất.
