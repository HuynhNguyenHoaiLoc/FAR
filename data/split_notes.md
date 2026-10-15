
# FAR-POLYP-SEG – Patient Split Notes

## 1. Mục đích

Chia dữ liệu theo bệnh nhân (patient-level split) để tránh
data leakage giữa Train, Validation và Test.

Sử dụng cách chia 5-fold chính thức của tác giả
FAR-POLYP-SEG, không tự chia ngẫu nhiên.

Nguồn:
https://github.com/FAR-POLYP-SEG/Codes

Official protocol:
- Scheme: grouped_cv5
- Seed: 2026
- Number of folds: 5

## 2. Thống kê dữ liệu

Tổng cộng 455 bệnh nhân:

- POLYP_COHORT: 187 bệnh nhân
- NEVER_POLYP_COHORT: 268 bệnh nhân

Có 432 ảnh Polyp được sử dụng trong official CV5.

## 3. Phân chia fold0

| Split | Patients | Polyp Images |
|---|---:|---:|
| TRAIN | 130 | 301 |
| VAL | 19 | 44 |
| TEST | 38 | 87 |
| NEVER_POLYP_TEST | 268 | Không thuộc bộ 432 ảnh Polyp |

Train, Val và Test không có bệnh nhân trùng nhau.

NEVER_POLYP_TEST được giữ riêng để phục vụ
đánh giá false positive / stress-test.

Không đưa NEVER_POLYP_TEST vào huấn luyện,
tuning hoặc calibration.

## 4. Các file dữ liệu

### data/splits.csv

File bàn giao theo kế hoạch A02.

Chứa patient-level split của fold0:
- patient_id
- fold
- cohort
- split

### data/patient_splits_official_cv5.csv

Lưu patient-level splits đầy đủ của cả 5 folds.

### data/official_polyp_image_splits_cv5.csv

Lưu danh sách image_id chính thức của Train,
Val và Test trong cả 5 folds.

Phải sử dụng file này khi reproduce segmentation
baseline để chọn đúng ảnh theo JSON của tác giả.

### data/dataset_manifest.csv

Danh sách ảnh, Patient ID, nhãn Polyp và
đường dẫn ảnh/mask.

## 5. Quy tắc sử dụng

1. Chia dữ liệu theo patient_id, không chia ngẫu nhiên
   theo từng ảnh.

2. Khi chạy fold0, sử dụng data/splits.csv.

3. Khi chạy cross-validation đầy đủ, sử dụng
   data/patient_splits_official_cv5.csv.

4. Đối với PraNet baseline, lọc chính xác image_id
   theo data/official_polyp_image_splits_cv5.csv.

5. Augmentation chỉ áp dụng cho TRAIN.

6. Không dùng TEST để lựa chọn hyperparameters
   hoặc checkpoint.

7. Không tự xem VAL là CALIBRATION hay META_FIT.
   Các tập này cần protocol riêng nếu pipeline mở rộng
   sử dụng chúng.

8. NEVER_POLYP_TEST chỉ dùng cho đánh giá cuối cùng.

## 6. Kết quả kiểm tra

- Official 5-fold patient split: PASS
- Official image split consistency: PASS
- Patient leakage fold0: NONE
- Fold0 Dataset class: PASS
- Fold0 DataLoader: PASS
- All 432 polyp images and masks: PASS
- Horizontal flip synchronization: PASS

## 7. Ghi chú tái lập

Hai script:
- code/module1/import_official_splits.py
- code/module1/build_official_image_splits.py

đọc file JSON chính thức từ bản clone
FAR-POLYP-SEG-Official nằm cạnh thư mục FAR.

File splits.csv là bản bàn giao fold0.
Nó không thay thế bộ official CV5 đầy đủ.
