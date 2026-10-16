# Module 1 – PraNet Polyp Segmentation Backbone

## 1. Giới thiệu

Module 1 xây dựng mô hình PraNet để phân đoạn polyp
trong ảnh nội soi tiêu hóa.

Đây là baseline segmentation backbone của hệ thống.
Module cung cấp mask xác suất và mask nhị phân cho
các module xử lý tiếp theo.

**Phạm vi bàn giao:** FAR-POLYP-SEG, official grouped
5-fold cross-validation, **fold0**.

## 2. Dữ liệu và cấu hình

| Thành phần | Giá trị |
|---|---|
| Model | PraNet |
| Backbone | ResNet-50 pretrained ImageNet |
| Experiment | E1 |
| Split scheme | grouped_cv5 |
| Fold | 0 |
| Train images | 301 |
| Validation images | 44 |
| Test images | 87 |
| Input size | 256 x 256 |
| Batch size | 8 |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Scheduler | Cosine Annealing |
| Maximum epochs | 150 |
| Early stopping patience | 30 |
| Loss | StructureLoss trên 4 đầu ra |
| Checkpoint selection | Best Validation Dice |

Official patient-grouped splits được giữ nguyên.
Test không được sử dụng để lựa chọn checkpoint.

Cấu hình gốc được lưu trong
`configs/train_config.yaml`.

## 3. Kết quả huấn luyện

- Epoch đã chạy: **45**, đánh số 0–44.
- Best checkpoint: **epoch 14**.
- Best Validation Dice: **0.749832**.
- Early stopping: epoch 44.

### Kết quả trên tập Test (87 ảnh)

| Metric | Giá trị |
|---|---:|
| Dice | 0.758565 |
| IoU | 0.653502 |
| Precision | 0.781914 |
| Recall | 0.802542 |
| Accuracy | 0.965110 |
| Specificity | 0.987293 |

Các chỉ số trên là kết quả official macro evaluation
ở kích thước ảnh gốc, không phải kết quả trung bình
của toàn bộ 5 folds.

Thông tin chi tiết:
- `metrics/metrics.json`
- `metrics/scores.csv`
- `metrics/official_summary.json`

## 4. Cấu trúc file bàn giao

```text
fold0/
├── checkpoints/
│   ├── best_model.pth
│   └── last_checkpoint.pth
├── predictions/
│   ├── probability/
│   │   └── PatientXXX/*.npy
│   ├── binary_mask/
│   │   └── PatientXXX/*.png
│   └── predictions.csv
├── metrics/
│   ├── metrics.json
│   ├── scores.csv
│   └── official_summary.json
├── configs/
│   └── train_config.yaml
├── logs/
│   ├── train.log
│   └── history.csv
├── visualizations/
│   └── comparison/
└── README_module1.md
```

## 5. Định dạng đầu ra

### Probability Mask (.npy)

- NumPy array `float32`
- Shape: `(H, W)` theo kích thước ảnh gốc
- Giá trị: `[0, 1]`
- Tính từ `sigmoid(main_logits)`
- Resize bằng bilinear, `align_corners=False`

### Binary Mask (.png)

- Ảnh grayscale, kích thước `(H, W)`
- Giá trị pixel: `0` hoặc `255`
- Ngưỡng nhị phân: `probability > 0.5`

**Quan trọng:** Resize xác suất về kích thước ảnh gốc
trước khi threshold.

### predictions.csv

Mỗi dòng tương ứng với một ảnh Test.

Các trường chính:

- `frame_id`: mã ảnh, có kèm Patient ID
- `patient_id`: mã bệnh nhân
- `split`: OUTER_TEST
- `image_path`: đường dẫn ảnh gốc theo official dataset
- `probability_path`: đường dẫn tương đối đến file .npy
- `binary_mask_path`: đường dẫn tương đối đến file .png
- `height`, `width`: kích thước ảnh gốc
- `pred_pos`: số pixel được dự đoán là polyp

Đường dẫn prediction trong CSV được tính tương đối
từ thư mục `fold0/`.

## 6. Cách đọc Probability Mask

Ví dụ Python:

```python
from pathlib import Path
import numpy as np
import pandas as pd

root = Path("results/module1/fold0")

df = pd.read_csv(
    root / "predictions/predictions.csv"
)

row = df.iloc[0]

prob = np.load(
    root / row["probability_path"]
)

binary = (prob > 0.5).astype(np.uint8)

print("Image ID:", row["frame_id"])
print("Shape:", prob.shape)
print("Range:", prob.min(), prob.max())
```

Các module sau nên sử dụng trực tiếp Probability Mask
nếu cần thông tin độ tin cậy theo pixel.

## 7. Checkpoint

### best_model.pth

Chứa model `state_dict` của epoch có
Validation Dice cao nhất.

Sử dụng cho inference:

```python
import torch
from farseg.models.pranet import PraNet

model = PraNet(channel=32, pretrained=False)

state = torch.load(
    "checkpoints/best_model.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state)
model.eval()
```

Cần có mã nguồn chính thức FAR-POLYP-SEG
để import lớp PraNet.

### last_checkpoint.pth

Chứa trạng thái huấn luyện cuối cùng:
model, optimizer, scheduler, epoch, RNG và
thông tin early stopping.

Dùng để resume thông qua runner chính thức;
không nạp trực tiếp như một model state_dict.

## 8. Kiểm chứng kết quả xuất

- Probability Mask: **87/87**
- Binary Mask: **87/87**
- Dòng trong predictions.csv: **87**
- So khớp `pred_pos` với official scores.csv: **87/87**
- Ảnh trực quan hóa: **10**

Việc khớp `pred_pos` xác nhận số pixel dương
giống official scores cho tất cả ảnh Test;
không tương đương với kiểm tra mask từng pixel.

## 9. Visualization

Thư mục `visualizations/comparison/` gồm:

- 3 ảnh có Dice cao nhất
- 3 ảnh có Dice gần trung vị
- 3 ảnh có Dice thấp nhất
- 1 ảnh tổng hợp: `overview_9_cases.png`

Chú thích vùng màu trên overlay:

- Vàng: True Positive
- Đỏ: False Positive
- Xanh lá: False Negative

## 10. Giới hạn và lưu ý

1. Đây là kết quả của **fold0**, chưa phải kết quả
   tổng hợp toàn bộ 5 folds.
2. Probability và Binary Mask hiện chỉ xuất cho
   87 ảnh thuộc official OUTER_TEST của fold0.
3. Tập Test có thể dùng để báo cáo chất lượng,
   không dùng để điều chỉnh mô hình.
4. Muốn chạy trên ảnh mới, cần tải best checkpoint
   và áp dụng đúng preprocessing/inference.
5. Dữ liệu gốc không được đóng gói trong thư mục này.
6. Các file Kaggle trong `/kaggle/working` cần được
   sao lưu trước khi phiên làm việc kết thúc.

## 11. Nguồn tham khảo

Official FAR-POLYP-SEG repository:
https://github.com/FAR-POLYP-SEG/Codes
