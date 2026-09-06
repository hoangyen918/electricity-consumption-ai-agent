# Wind Turbine Data Audit — Phase 2 Ngày 1 (Baseline)

Đo thật qua Pandas trên 5 file đã nạp ở Phase 1 (`DatasetManager`, dùng đúng logic
delimiter/BOM/dayfirst đã fix, không đọc `pd.read_csv` trực tiếp):

| Dataset | Hàng | Cột gốc | Datetime source |
|---|---|---|---|
| `data/wind_turbine/data1/his_data_GT1.csv` (GT1) | 233.953 | 13 | `TimeStamp` (M/D/YYYY) |
| `data/wind_turbine/data2/WindFarm04_WT1.csv` (WT1) | 262.796 | 17 | `DATE_TIME` (DD/MM/YYYY) |
| `data/wind_turbine/data2/WindFarm04_WT2.csv` (WT2) | 262.796 | 17 | `DATE_TIME` |
| `data/wind_turbine/data2/WindFarm04_WT3.csv` (WT3) | 262.796 | 17 | `DATE_TIME` |
| `data/wind_turbine/data2/WindFarm04_WT4.csv` (WT4) | 262.796 | 17 | `DATE_TIME` |

Không có dòng trùng lặp hoàn toàn (`duplicated().sum() == 0`) ở cả 5 file.

Timestep chuẩn: GT1 = 5 phút (233.952/233.952 khoảng cách đúng 5 phút, không lệch).
WT1-4 = 10 phút (262.754/262.795 khoảng cách đúng 10 phút; phần còn lại là các gap
bất thường, xem Vấn đề 9).

> **Cập nhật (verify lại — xem mục "Sửa đổi sau verify" cuối file):** phiên bản đầu
> xếp "ACTIVE_POWER âm" là vấn đề severity Cao chỉ dựa trên số dòng/tỷ lệ. Phân tích
> lại theo magnitude (percentile, so với rated power) cho thấy đây là hiện tượng
> house-load bình thường (biên độ luôn <50kW, <1% rated), KHÔNG phải anomaly — đã hạ
> xuống severity Thấp và dời xuống Vấn đề 10. Rule 10 (nhiệt độ báo động) verify lại
> đúng số 0 vi phạm nhưng phần giải thích cũ bị nhầm cột — đã sửa.

## 10 vấn đề nổi bật (severity cao → thấp)

---

**Vấn đề 1: GT1 sản lượng thực tế vượt sản lượng lý thuyết (TheoActiveW) hơn 10%**
File ảnh hưởng: GT1
Số dòng/tỷ lệ: 21.302 dòng (9,11%)
Rule/check phát hiện: Rule 6 (Bước 2)
Ví dụ 1-2 dòng data thật: `7/30/2023 16:35`: totW=2920.76, TheoActiveW=2619.21
(vượt 11,5%)
Đánh giá mức độ: **Cao** — xảy ra ở gần 1/10 số dòng, cho thấy lệch hệ thống giữa số
đo thật (Grid.totW) và mô hình lý thuyết (TheoActiveW), ảnh hưởng trực tiếp mọi phân
tích hiệu suất/curtailment dùng cột TheoActiveW làm baseline ở Phase 2. Đây hiện là
vấn đề severity cao nhất còn lại sau khi verify lại (xem "Vấn đề 10" — không có cách
giải thích tương tự qua trạng thái vận hành như trường hợp power âm).

---

**Vấn đề 2: Giá trị nhiệt độ placeholder bất khả thi vật lý (mã lỗi cảm biến ẩn dưới dạng số thật)**
File ảnh hưởng: WT2 (giá trị -273,2°C — đúng độ không tuyệt đối theo Kelvin, chắc
chắn là mã lỗi, không phải nhiệt độ thật), WT4 (giá trị 0,0°C đồng thời trên cả 4 cột
nhiệt độ — cùng dấu hiệu cảm biến rớt tín hiệu)
Số dòng/tỷ lệ: WT2: 33 dòng (0,0126%) trên đồng thời cả 4 cột
(TEMPERATURE_EXTERNAL/STATOR/GEARBOX_OIL/MAIN_BEARING); WT4: 10 dòng (0,0038%)
đồng thời cả 4 cột tương tự, cộng thêm 174 dòng (0,066%) riêng TEMPERATURE_EXTERNAL=0.0
Rule/check phát hiện: phát hiện qua `describe()` min ở Bước 1 (không nằm trong 11 rule
định nghĩa sẵn ở Bước 2 vì rule 10 chỉ kiểm tra ngưỡng trên — xem thêm phần verify Rule
10 ở cuối file, hai phát hiện này KHÔNG liên quan tới nhau dù cùng nói về nhiệt độ)
Ví dụ 1-2 dòng data thật: WT2, `08/06/2014 22:40`: TEMPERATURE_EXTERNAL=-273.2,
TEMPERATURE_STATOR=-273.2, TEMPERATURE_GEARBOX_OIL=-273.2, TEMPERATURE_MAIN_BEARING=-273.2
Đánh giá mức độ: **Cao (khối lượng nhỏ)** — số dòng rất ít nhưng nếu không lọc riêng,
giá trị -273,2 sẽ kéo lệch nghiêm trọng bất kỳ mean/std nào tính trên các cột nhiệt độ
này; đồng thời trên 4 cột cùng lúc là bằng chứng rõ ràng đây là mã lỗi hệ thống, không
phải nhiễu ngẫu nhiên.

---

**Vấn đề 3: Cột GT1.App.StartedFlag thiếu dữ liệu nghiêm trọng**
File ảnh hưởng: GT1
Số dòng/tỷ lệ: 81.147/233.953 dòng thiếu (34,69%)
Rule/check phát hiện: `isnull().sum()` (Bước 1)
Ví dụ 1-2 dòng data thật: NaN chiếm hơn 1/3 toàn bộ cột, không tập trung ở 1 khoảng
thời gian cụ thể (đã kiểm tra rải rác theo describe/null tổng, không tách theo thời
gian ở phase này)
Đánh giá mức độ: **Vừa-cao** — thiếu tới hơn 1/3 dữ liệu khiến cột này không đủ tin
cậy để dùng làm feature chính ở Phase 2 nếu không impute hoặc loại bỏ có chủ đích.

---

**Vấn đề 4: 2 cảm biến gió (WIND_SPEED và WIND_SPEED_2) lệch nhau hơn 20%**
File ảnh hưởng: WT1, WT2, WT3, WT4
Số dòng/tỷ lệ: WT1 10.183 (3,87%), WT2 13.289 (5,06%), WT3 16.379 (6,23%), WT4 11.715
(4,46%)
Rule/check phát hiện: Rule 11 (Bước 2)
Ví dụ 1-2 dòng data thật: WT3, `10/01/2012 15:10`: WIND_SPEED=0.58, WIND_SPEED_2=0.11
(lệch hơn 5 lần)
Đánh giá mức độ: **Vừa** — WT3 có tỷ lệ cao nhất (6,23%), gợi ý 1 trong 2 anemometer
kém tin cậy hơn ở turbine này so với 3 turbine còn lại; cần đối chiếu thêm ở Phase 2
trước khi chọn cột nào làm nguồn wind speed chính.

---

**Vấn đề 5: Công suất dương khi tốc độ gió dưới ngưỡng cut-in (~3 m/s)**
File ảnh hưởng: GT1, WT1, WT2, WT3, WT4 (tất cả)
Số dòng/tỷ lệ: GT1 7.071 (3,02%); WT1 995 (0,38%), WT2 1.451 (0,55%), WT3 2.826
(1,08%), WT4 1.185 (0,45%)
Rule/check phát hiện: Rule 2 (GT1) và Rule 8 (WT1-4) ở Bước 2
Ví dụ 1-2 dòng data thật: GT1, `8/14/2023 18:05`: totW=12.49, WindSpd=2.96 m/s
Đánh giá mức độ: **Vừa** — GT1 có tỷ lệ cao gấp 3-8 lần so với các turbine Site 2,
đáng chú ý nhất ở GT1; hợp lý về vật lý (ngưỡng cut-in không phải hằng số cứng tuyệt
đối, có quán tính quay), nhưng tỷ lệ 3% ở GT1 cao hơn dự kiến, cần kiểm tra thêm liệu
đồng hồ đo gió và đồng hồ đo công suất có lệch thời gian ghi nhận hay không.

---

**Vấn đề 6: Góc pitch 3 lưỡi turbine GT1 lệch nhau hơn 5°**
File ảnh hưởng: GT1
Số dòng/tỷ lệ: 3.875 dòng (1,66%)
Rule/check phát hiện: Rule 5 (Bước 2)
Ví dụ 1-2 dòng data thật: `8/1/2023 0:25`: Angle1=62.59°, Angle2=88.76°, Angle3=88.75°
(lệch 26,17° giữa lưỡi 1 và 2 lưỡi còn lại)
Đánh giá mức độ: **Vừa** — 3 lưỡi lệch nhau đáng kể về mặt cơ khí là bất thường, dù
tỷ lệ dòng ảnh hưởng không lớn (1,66%), cần xác nhận đây là giai đoạn khởi động
(pitch chưa đồng bộ hoá xong) hay lỗi cảm biến thật.

---

**Vấn đề 7: Cột DIRECTION_WIND/DIRECTION_NACELLE ở Site 2 thiếu dữ liệu nhiều hơn hẳn các cột khác, nhất quán trên cả 4 turbine**
File ảnh hưởng: WT1, WT2, WT3, WT4
Số dòng/tỷ lệ: phần thiếu THÊM (ngoài phần thiếu chung với các cột khác):
WT1 +7.370, WT2 +7.372, WT3 +7.372, WT4 +7.372 dòng (~2,8% mỗi file) — con số gần như
giống hệt nhau trên cả 4 turbine
Rule/check phát hiện: so sánh `isnull().sum()` giữa các cột (Bước 1), không nằm trong
11 rule Bước 2
Ví dụ 1-2 dòng data thật: WT1: ACTIVE_POWER null=1.271 nhưng DIRECTION_WIND
null=8.641 (chênh 7.370 dòng)
Đánh giá mức độ: **Vừa-thấp** — độ chênh gần như bằng nhau tuyệt đối trên cả 4
turbine (7.370-7.372) gợi ý đây là lỗi từ 1 nguồn đo hướng gió DÙNG CHUNG cho cả
farm (vd cột mốc khí tượng chung), tách biệt hẳn với lỗi mất dữ liệu ngẫu nhiên theo
từng turbine.

---

**Vấn đề 8: Cảm biến công suất bị "đứng" (flatline) nhiều timestep liên tiếp, nổi bật nhất ở WT3**
File ảnh hưởng: GT1, WT1, WT2, WT3 (đáng chú ý), WT4
Số dòng/tỷ lệ: WT3 163 chuỗi, 1.406 dòng (0,535%) — cao nhất; WT1 54 chuỗi/492 dòng
(0,187%); WT2 12 chuỗi/85 dòng (0,032%); WT4 1 chuỗi/10 dòng (0,004%); GT1 4 chuỗi/534
dòng (0,228%, có 1 chuỗi dài bất thường 38 bước ~3,2 giờ)
Rule/check phát hiện: đếm chuỗi ≥6 timestep liên tiếp giá trị y hệt nhau (Bước 3)
Ví dụ 1-2 dòng data thật: GT1: chuỗi 38 bước liên tiếp giá trị totW=-35.85 (~3,2 giờ
đứng yên); WT3: chuỗi 6 bước giá trị ACTIVE_POWER=-0.01
Đánh giá mức độ: **Thấp-vừa** — WT3 có số chuỗi đứng cao gấp 3-13 lần các turbine
còn lại cùng farm, đáng nghi vấn cảm biến turbine này kém ổn định hơn; GT1's chuỗi 38
bước liên tiếp là dấu hiệu cảm biến "đóng băng" rõ ràng cần lọc ở Phase 2.

---

**Vấn đề 9: Khoảng trống timestamp >1 giờ, giống hệt nhau trên cả 4 turbine Site 2, KHÔNG xuất hiện ở GT1**
File ảnh hưởng: WT1, WT2, WT3, WT4 (giống hệt nhau); GT1 = 0 gap
Số dòng/tỷ lệ: mỗi file WT1-4 đều có đúng 7 gap >1 giờ (bao gồm 1 gap tròn 24 giờ 10
phút); GT1: 0 gap trên toàn bộ 233.952 khoảng cách
Rule/check phát hiện: `datetime.diff() > 1 giờ` (Bước 3)
Ví dụ 1-2 dòng data thật: cả 4 file WT cùng có gap kết thúc tại `2013-03-05 00:10:00`
(gap dài 1 ngày 10 phút) và gap kết thúc tại `2012-03-25 03:00:00` (gap 1 giờ 10 phút)
Đánh giá mức độ: **Thấp** — các gap trùng khớp tuyệt đối về thời điểm trên cả 4
turbine chứng tỏ đây là lỗi hệ thống thu thập dữ liệu dùng chung (không phải lỗi
riêng từng turbine); GT1 dùng hệ thống SCADA khác, ổn định hơn hẳn về mặt này (0 gap).

---

**Vấn đề 10: Công suất âm — biên độ nhỏ ở CẢ 2 site, khả năng cao là house/parasitic load bình thường, KHÔNG phải anomaly (đã hạ severity sau verify magnitude)**
File ảnh hưởng: GT1, WT1, WT2, WT3, WT4 (tất cả)
Số dòng/tỷ lệ: GT1 21.805 dòng (9,32%); Site 2: WT1 43.847 (16,68%), WT2 55.016
(20,93%), WT3 48.635 (18,51%), WT4 78.562 (29,89%)
Rule/check phát hiện: Rule 1 (GT1) và Rule 7 (WT1-4) ở Bước 2; **verify magnitude bổ
sung** (percentile 10/25/50/75/90/99 của các giá trị âm, so với rated power ước tính
từ max ACTIVE_POWER/Grid.totW quan sát được — Site 2 ~2.051 kW/turbine tương đồng dữ
liệu công khai turbine dòng MM82 (~2,05 MW), GT1 ~4.525 kW)
Kết quả verify: **100% số dòng âm ở cả 2 site có |giá trị| < 50 kW** — không có dòng
nào rơi vào nhóm ">50kW" (nhóm khả năng cao là anomaly thật). Độ lớn tối đa quan sát
được: GT1 -44,52 kW (0,98% rated), Site 2 -18,49 kW (dưới 0,90% rated ở mọi turbine).
Percentile p99 (Site 2 gộp cả 4 turbine) chỉ -0,01 kW — tức phần lớn giá trị âm gần
như bằng 0, chỉ 1 số ít chạm mức chục kW. Ở GT1, đối chiếu thêm với `GT1.Common.OpSt`
cho thấy 55% số dòng âm rơi vào trạng thái OpSt=3 (Starting) và 20,5% vào OpSt=2
(Standstill) — giải thích được bằng tiêu thụ điện phụ trợ lúc không phát điện.
Ví dụ 1-2 dòng data thật: GT1 `8/9/2023 12:40`: totW=-17.59 (lúc OpSt=3, Starting);
WT4 `05/01/2012 04:00`: ACTIVE_POWER=-5.21
Đánh giá mức độ: **Thấp** (hạ từ Cao xuống Thấp sau verify magnitude) — đây là 1 hiện
tượng DUY NHẤT (house/parasitic load), KHÔNG cần tách thành 2 nhóm vì nhóm "anomaly
thật >50kW" trống rỗng ở cả 2 site. KHÔNG phải data-quality issue cần Phase 2 xử lý
(clip/loại bỏ) — là vấn đề gắn nhãn/diễn giải đúng ngữ nghĩa cho Phase 3 (vd thêm cờ
`is_house_load` thay vì coi là lỗi dữ liệu).

---

## Rule không tìm ra vi phạm nào (dataset "sạch" hơn dự kiến)

- **Rule 3 / Rule 9** (công suất dương khi gió vượt cut-out ~25 m/s): **0 dòng vi phạm**
  ở cả 5 file — kể cả khi WIND_SPEED đạt tới 49,66 m/s (WT1, giá trị cực đại bất
  thường về mặt khí tượng nhưng KHÔNG đi kèm công suất dương) — cho thấy logic
  shutdown khi quá tốc độ gió hoạt động đúng và nhất quán trên toàn bộ dữ liệu.
- **Rule 4** (yaw misalignment ngoài [-180, 180]): **0 dòng vi phạm** ở GT1 — cột
  `GT1.Common.WindTrackDevAng` luôn nằm trong biên vật lý hợp lệ (min=-180.0000,
  max=179.6319).
- **Rule 10** (nhiệt độ báo động: `TEMPERATURE_GEARBOX_OIL > 90` HOẶC
  `TEMPERATURE_STATOR > 150`): verify lại chính xác cho cả 4 file WT — **đúng 0 dòng
  vi phạm** ở MỌI file (WT1: gearbox>90 = 0, stator>150 = 0; WT2/WT3/WT4 tương tự,
  xem bảng dưới). Số 0 ban đầu là ĐÚNG, nhưng giải thích trong bản audit trước bị
  **nhầm cột**: câu cũ viết giá trị 90,57°C "rất gần ngưỡng báo động" — giá trị này
  đúng là max thật của **TEMPERATURE_STATOR** (WT1), nhưng ngưỡng báo động của CHÍNH
  CỘT NÀY là 150°C, tức còn cách tới 59,43°C, hoàn toàn KHÔNG gần. Bài viết trước đã
  nhầm lẫn so sánh giá trị max của STATOR với ngưỡng 90°C — vốn là ngưỡng của cột khác
  (GEARBOX_OIL). Thực tế TEMPERATURE_GEARBOX_OIL (WT1) chỉ đạt max 70,72°C, cách
  ngưỡng 90°C của chính nó 19,28°C — cũng không sát ngưỡng. Đây là lỗi diễn giải văn
  bản, KHÔNG phải lỗi số liệu (số 0 dòng vi phạm không đổi).

  | File | GEARBOX_OIL>90 | STATOR>150 | max GEARBOX_OIL | max STATOR |
  |---|---|---|---|---|
  | WT1 | 0 | 0 | 70,72°C | 90,57°C |
  | WT2 | 0 | 0 | 64,01°C | 88,40°C |
  | WT3 | 0 | 0 | 64,22°C | 90,28°C |
  | WT4 | 0 | 0 | 64,96°C | 88,50°C |

  Lưu ý: điều này KHÔNG mâu thuẫn với Vấn đề 2 (giá trị -273,2°C/0,0°C là mã lỗi cảm
  biến khác hẳn về bản chất — giá trị sentinel bất khả thi, không phải nhiệt độ cao
  thật).

## Rule tìm ra nhiều vi phạm nhất (cần chú ý đặc biệt ở Phase 2)

1. **Rule 6 (công suất vượt lý thuyết >10%, GT1)** — 9,11% số dòng. Sau khi hạ severity
   của Rule 1/Rule 7 (xem dưới), đây là rule "đáng lo ngại nhất còn lại" vì KHÔNG có
   cách giải thích tương tự qua OpSt hay magnitude nhỏ — cần điều tra sâu hơn ở Phase 2.
2. **Rule 7 (ACTIVE_POWER âm, Site 2)** — tỷ lệ dòng cao nhất toàn bộ audit (tới 29,89%
   ở WT4), nhưng verify magnitude (Vấn đề 10) cho thấy 100% các dòng này có |giá trị|
   <50kW (<0,9% rated) — kết luận: đây là rule "ồn ào" nhất về SỐ LƯỢNG nhưng KHÔNG
   phải rule nghiêm trọng nhất về mức độ ảnh hưởng thực tế.
3. **Rule 1 (Grid.totW âm, GT1)** — 9,32% số dòng, cùng kết luận như Rule 7: magnitude
   nhỏ (max -44,52kW, <1% rated), 75% số dòng giải thích được qua OpSt=Starting/
   Standstill — đã gộp vào Vấn đề 10, không còn xếp riêng là vấn đề severity cao.

## Ghi chú phương pháp

- Toàn bộ số liệu trong file này đo trực tiếp qua Pandas trên dữ liệu thật, không suy
  đoán. Script phân tích: `turbine_audit.py` (chạy tạm qua scratchpad, không commit
  vào repo theo đúng ràng buộc "không viết tool permanent" của task này).
- 3 plot minh hoạ (histogram, power curve, time series 1 tháng) cho WT1 được lưu tại
  thư mục scratchpad của phiên làm việc (không phải `/tmp/audit_plots/` như mô tả gốc
  vì môi trường thực tế là Windows, không có `/tmp` — đã báo cáo lệch này cho người
  dùng).
- Rule 11 (lệch 2 anemometer) tính theo % so với WIND_SPEED (cảm biến chính):
  `|WIND_SPEED - WIND_SPEED_2| / WIND_SPEED > 20%`, loại trừ các dòng WIND_SPEED = 0
  (chia cho 0) khỏi phép tính — không tính các dòng đó là vi phạm.
- Correlation Site 2 (Bước 4): ma trận tương quan ACTIVE_POWER giữa WT1-4 trên
  263.216 dòng chung (inner join theo `datetime`) — tất cả cặp đều >0.85 (thấp nhất
  WT3-WT4 = 0,852), KHÔNG có turbine nào rơi vào ngưỡng nghi vấn <0.7 → không phát
  sinh thêm vấn đề từ Bước 4 để đưa vào danh sách 10 vấn đề trên.
- Rated power dùng để tính % magnitude ở Vấn đề 10 là ƯỚC TÍNH từ max quan sát được
  trong data (không phải số liệu nameplate chính thức từ nhà sản xuất) — với Site 2,
  con số ~2.051 kW khớp gần đúng với công suất định mức phổ biến của dòng turbine
  MM82 (~2,05 MW theo thông tin công khai), nhưng chưa đối chiếu trực tiếp với tài
  liệu kỹ thuật MM82 đã ingest trong KB — nếu cần số chính xác tuyệt đối cho Phase 2,
  nên verify lại qua tài liệu đó.

## Sửa đổi sau verify (phiên làm việc tiếp theo)

1. **Magnitude power âm**: percentile 10/25/50/75/90/99 tính trên toàn bộ giá trị âm
   mỗi file (Site 2: xem Vấn đề 10 cho số liệu p10-p99 từng turbine — dao động từ
   khoảng -4,71kW (p10, WT4) tới -0,01kW (p99, gộp cả 4 turbine); GT1: p10=-24,45kW
   tới p99=-0,98kW). 100% số dòng âm ở CẢ 2 site có |giá trị| <50kW — 0% thuộc nhóm
   ">50kW". Kết luận: 1 hiện tượng duy nhất (house load), không tách 2 nhóm, hạ severity
   Cao → Thấp, dời từ vị trí Vấn đề 1 xuống Vấn đề 10.
2. **Rule 10 verify lại**: đúng 0 dòng vi phạm ở cả 4 file WT (không làm tròn — số
   nguyên 0 chính xác cho cả 2 điều kiện GEARBOX_OIL>90 và STATOR>150). Giải thích cũ
   bị nhầm cột (so sánh max STATOR với ngưỡng của GEARBOX_OIL) — đã sửa trong mục
   "Rule không tìm ra vi phạm nào" ở trên, kèm bảng số liệu 4 file.
