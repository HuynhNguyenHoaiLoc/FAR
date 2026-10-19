# A02 · Patient Splits, Dataset, augmentation — FAR Clean V2

**Phạm vi:** Module 1, Fold 0; task A02 trong sheet `02_VIEC_CAN_LAM`, hạn 20/10/2026.

## Nội dung và bằng chứng

1. **Patient split:** Chỉ **đọc** `data/splits/inner_clean_v2_v1/image_inner_splits_cv5_candidate.csv` đã duyệt (PR #7). Không tái chia bệnh nhân hoặc ghi đè file cũ. Mỗi `split_group_id` / `patient_id` chỉ thuộc đúng một `inner_split` cho mỗi fold. 6 nhãn hợp lệ: SEG_TRAIN, SEG_VAL, META_FIT, CALIBRATION, OUTER_TEST, NEVER_POLYP_TEST.
2. **Dataset:** `code/module1/clean_v2_dataset.py` đọc ảnh RGB, mask GT Polyp và tùy chọn Healthy zero mask; kiểm tra đường dẫn an toàn, yêu cầu kích thước ảnh–mask gốc khớp, giữ metadata khi cần. `make_loader` trả PyTorch DataLoader.
3. **Augmentation:** `code/module1/clean_v2_transforms.py` dùng flips đồng bộ ảnh/mask và rotation đồng bộ tối đa ±15°; **chỉ** `SEG_TRAIN` được augmentation. RGB bilinear, GT nearest, GT binary {0,1}; 256×256, RGB ImageNet normalized.
4. **Minh chứng 10 cặp thật:** `code/module1/a02_verify.py` mở 10 ảnh Polyp **khác patient** trong SEG_TRAIN, kiểm GT gốc, chạy DataLoader và xuất `ten_real_pairs.csv`, `ten_real_pairs.png` và `a02_audit.json`. Không dùng ảnh đánh giá ngoài SEG_TRAIN cho minh chứng.

## Chạy thử trên VS Code (không cần dataset thật)

Tại repo root:

```powershell
python -m pytest -q tests/test_module1_a02_clean_v2.py
```

Sáu unit tests kiểm tra dataset, mask Healthy, augmentation, patient-group leakage, outer leakage và công cụ xuất bằng chứng bằng **fixture giả lập**. PASS ở đây **không** được coi là xác minh 10 cặp dữ liệu thật.

## Kiểm tra bắt buộc trên Kaggle (dataset thật)

Notebook phải attach:

- `eoquiither/far-polyp-seg-clean-v1-candidate`
- `eoquiither/datasetok` (phần approved Inner CV5)

Sau khi nhánh A02 được push lên GitHub, chạy trong Kaggle notebook có Internet enabled (hoặc thêm code thủ công vào notebook đã có):

```python
import subprocess
from pathlib import Path
repo = Path('/kaggle/working/FAR_A02')
if not repo.exists():
    subprocess.run(['git','clone','--depth','1','--branch',
                    'feature/clean-v2-inner-m1-handoff',
                    'https://github.com/HuynhNguyenHoaiLoc/FAR.git',
                    str(repo)], check=True)
subprocess.run([
    'python', str(repo/'code/module1/a02_verify.py'),
    '--dataset', '/kaggle/input/datasets/eoquiither/far-polyp-seg-clean-v1-candidate/far_polyp_seg_clean_v1_candidate/Dataset',
    '--splits', '/kaggle/input/datasets/eoquiither/datasetok/data/splits/inner_clean_v2_v1/image_inner_splits_cv5_candidate.csv',
    '--fold','0',
    '--out','/kaggle/working/A02_ACCEPTANCE_FOLD0',
], check=True)
```

**Kết quả cần đạt:** `STATUS: A02 REAL DATA PASS`. Trước khi đánh dấu `Hoàn thành` trong sheet, cung cấp `a02_audit.json`, `ten_real_pairs.csv` và đường dẫn Kaggle output chứa contact sheet `ten_real_pairs.png` để nhóm review. **Không push ảnh nội soi gốc/ảnh ghép lên GitHub public nếu chưa kiểm tra quyền tái phân phối của dataset.** Không thay các bằng chứng này bằng sản phẩm từ dữ liệu giả lập.

## Dùng Dataset trong code M1

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('code/module1').resolve()))
from clean_v2_dataset import CleanV2SegDataset, make_loader
train = CleanV2SegDataset(dataset_root, image_inner_csv, 0, 'SEG_TRAIN')  # (image, gt); flip/rotate on
val = CleanV2SegDataset(dataset_root, image_inner_csv, 0, 'SEG_VAL')      # no augmentation
loader = make_loader(dataset_root, image_inner_csv, 0, 'SEG_TRAIN', batch_size=8)
```

Đoạn import trên chạy khi terminal/notebook có working directory là repo root. Không sửa mô hình/training hiện có bằng cách tự động thay import.

## Trạng thái

- Code implementation: **DONE**.
- Synthetic tests: **6/6 PASS** (chạy trong môi trường đóng gói).
- Real 10-pair Kaggle acceptance: **PENDING user run**.
- GitHub reviewer acceptance: **PENDING**.

**Không** kết luận A02 hoàn thành nghiệm thu trước khi có bằng chứng từ dữ liệu thật.
