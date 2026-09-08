# PO.024 — An estimation of key metrics from SCADA data: a third-party's view

**Tác giả:** Julien Berthaut-Gerentes, Minh-Thang DO
**Tổ chức:** METEODYN, Nantes, France
**Nguồn:** WindEurope Poster (windeurope.org/op18, #OpWF18)

## Abstract

SCADA-based monitoring là một cách tiếp cận hiệu quả về chi phí để giảm ngân
sách O&M trong năng lượng gió. Từ dữ liệu lịch sử của máy, có thể tính được
nhiều chỉ số hiệu suất quan trọng (key indicators). Các chỉ số này cho cái
nhìn tổng quan về sức khỏe của turbine gió, cho phép so sánh hiệu suất giữa
các turbine trong cùng farm, và giữa hiệu suất thực tế với hiệu suất lý
thuyết do nhà sản xuất công bố.

Tuy nhiên, việc ước tính các chỉ số quan trọng từ dữ liệu SCADA đòi hỏi 1
tín hiệu status-code sạch và đầy đủ, thường được tái tạo từ error log. Quy
trình lấy tín hiệu này rất phức tạp và đôi khi không khả thi do error log
bị thiếu hoặc không đầy đủ. Ngay cả khi có error log, không dễ để xác định
lỗi nào liên quan tới hiệu suất kém của máy.

## Objectives

Trong nghiên cứu này, nhóm tác giả giới thiệu 1 thuật toán phân loại chi
tiết và chuyên biệt, tái tạo lại 1 status code đầy đủ bao gồm cả partial
stops và curtailments. Thuật toán chỉ cần wind speed đo tại nacelle và
công suất đầu ra từ hệ thống SCADA. Khi có sẵn, các tham số SCADA khác
(như pitch control) được tự động thêm vào để tăng độ chính xác phân loại.

Power curve vận hành sau đó được ước tính từ tập dữ liệu các điểm "normal".
Từ status code đã phát hiện và power curve đã ước tính, các chỉ số quan
trọng của mỗi máy trong 1 khoảng thời gian cho trước được xác định: tổng
sản lượng, tổn thất (losses), time-based availability và energy-based
availability.

Ở bước thứ hai, chỉ số hiệu suất công suất (power performance index) của
mỗi máy được tính theo tháng. Chỉ số này không nhạy với thay đổi điều kiện
gió, cho phép đo lường đơn giản và nhanh chóng tác động của: bảo trì (vd
sửa 1 bộ phận cơ khí), thay đổi cài đặt (vd hiệu chỉnh lại anemometer),
và phát hiện bất thường chức năng (functional anomaly).

## Methods

Các điểm SCADA của mỗi turbine gió sẽ được phân loại.

Quy trình phân loại bắt đầu bằng việc **phát hiện các điểm đo lường bị lỗi**
(measurement failure). Các điểm này bị loại khỏi phân tích tiếp theo.

Tiếp theo, **power curve SCADA của turbine gió được ước tính** bằng kỹ
thuật ước lượng median lặp (iterative median estimation technique). Phương
pháp phát hiện này có độ bền vững cao — cho ra ước tính tốt về power curve
ngay cả trong trường hợp trạng thái "normal" chiếm dưới 50% tổng dữ liệu
SCADA.

Dùng power curve này, các phân loại khác của điểm dữ liệu được thực hiện:
normal, stop, curtailment, under production... Power curve cuối cùng sau
đó được ước tính lại chỉ dùng các điểm dữ liệu "normal".

Khi đã phân loại xong, việc tính toán các chỉ số quan trọng có thể thực
hiện mà không tốn nhiều công sức.

**Sơ đồ quy trình:**
```
SCADA Data
    ↓
Detection of points with measurement failure
    ↓
Estimation of the SCADA Power Curve
    ↓
Classify others data points
```

## Results

Cách tiếp cận được đề xuất được kiểm thử trên 1 wind farm gồm 4 turbine
gió MM82. Dữ liệu SCADA của 2 năm được thu thập và phân tích.

Biểu đồ Power Curve Estimation cho từng turbine (WT01-WT04) hiển thị các
điểm được phân loại theo màu: **Normal Points - Stops - Partial Stops -
Curtailment - Under Production - Over Production**.

### Bảng chỉ số chính (theo turbine)

| Turbine | Production (MWh) | Losses (%) | Capacity Factor (%) | Avg. Prod. (MWh/day) | Avg. Wind (m/s) | TBA (%) | PBA (%) |
|---|---|---|---|---|---|---|---|
| WT01 | 10119 | 1.86 | 28.58 | 14.05 | 6.58 | 96.45 | 98.33 |
| WT02 | 9928 | 5.84 | 28.00 | 13.77 | 6.45 | 94.00 | 94.54 |
| WT03 | 8327 | 18.94 | 23.58 | 11.55 | 6.57 | 86.87 | 81.29 |
| WT04 | 10570 | 4.22 | 29.82 | 14.64 | 7.08 | 96.26 | 95.92 |

(TBA = Time Based Availability, PBA = Production Based Availability)

### Bảng độ tin cậy (theo turbine)

| Turbine | Số lần failure | MTTR (ngày) | MTTF (ngày) | MTBF (ngày) |
|---|---|---|---|---|
| WT01 | 3 | 3.7 | 277.1 | 281.5 |
| WT02 | 5 | 5.1 | 111.7 | 117.4 |
| WT03 | 5 | 24.5 | 17.9 | 52.6 |
| WT04 | 3 | 6.1 | 46.1 | 54.1 |

(MTTR = Mean Time To Repair, MTTF = Mean Time To Failure, MTBF = Mean Time
Between Failures)

Nghiên cứu cũng có biểu đồ sản lượng và tổn thất theo tháng (monthly
production and losses) cho từng turbine, và biểu đồ trạng thái vận hành
(Operation State) của toàn bộ turbine trong farm theo thời gian.

## Conclusions

Trong nghiên cứu này, 1 phương pháp ước tính các chỉ số quan trọng chỉ từ
dữ liệu SCADA đã được đề xuất. Phương pháp này dựa trên 1 thuật toán phân
loại vững chắc (solid classification algorithm) cho các điểm vận hành
SCADA.

Một wind farm với 4 turbine đã được dùng làm test case. Nhiều chỉ số quan
trọng đã được tính cho các turbine này: sản lượng, tổn thất, capacity
factor, sản lượng trung bình ngày, tốc độ gió trung bình, time-based
availability, production-based ability, MTTR, MTTF, MTBF. Sản lượng theo
tháng của từng turbine cũng đã được nghiên cứu.

Kết quả chỉ ra rằng hiệu suất thấp của WT03 là do 1 giai đoạn bảo trì kéo
dài 2 tháng. Đây cũng chính là turbine có MTTR cao nhất và MTTF thấp nhất.

## References

1. K. Kim, G. Parthasarathy, O. Uluyol, W. Foslien, S. Sheng and P. Fleming,
   "Use of SCADA Data for Failure Detection in Wind Turbines", Energy
   Sustainability Conference and Fuel Cell Conference, Washington, D.C.,
   2011
2. Wang Ke-Sheng, Sharma Vishal S. and Zhang Zhen-You, "SCADA data based
   condition monitoring of wind turbines", Advances in Manufacturing,
   vol. 2, no. 1, p. 61-69, Mar 2014
3. Elena Gonzalez, Emmanouil M. Nanos, Helene Seyr, Laura Valldecabres,
   Nursedа Y. Yürüşen, Ursula Smolka, Michael Muskulus, Julio J. Melero,
   "Key Performance Indicators for Wind Farm Operation and Maintenance",
   Energy Procedia, vol. 137, p. 559-570, 2017
4. Dienst Steffen and Beseler Jonas, "Automatic Anomaly Detection in
   Offshore Wind SCADA Data", Proceedings of WindEurope Summit 2016,
   Hamburg Germany, 2016
