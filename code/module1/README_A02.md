# A02 · FAR Clean V2 Dataset / augmentation

Mã mới được thêm riêng để không sửa hỏng `code/module1/dataset.py` / pipeline M1 cũ.

- `clean_v2_dataset.py`: dataset cho Inner CV5 đã duyệt, kiểm patient/group leakage, hỗ trợ Polyp GT hoặc Healthy mask 0.
- `clean_v2_transforms.py`: resize 256, chuẩn hóa ImageNet; hflip/vflip/rotate cùng hình học ảnh/mask; augmentation chỉ SEG_TRAIN.
- `a02_verify.py`: kiểm tra 10 cặp RGB/GT khác bệnh nhân trên dữ liệu thật và tạo contact sheet.
- `tests/test_module1_a02_clean_v2.py`: kiểm thử không cần ảnh thật.

Xem `docs/A02_CLEAN_V2_HANDOFF.md` để biết lệnh chạy trên VS Code và Kaggle.

Với các module mới: **chỉ** dùng CSV ở `data/splits/inner_clean_v2_v1/`; KHÔNG dùng `data/patient_splits_official_cv5.csv` cho giao thức Clean V2.
