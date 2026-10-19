# Báo Cáo Hoàn Thành Hạng Mục M4: Metrics

- **Người thực hiện:** Huỳnh Hữu Lộc (Thành viên D – Phụ trách Module 4)
- **Hạng mục:** `M4: Metrics` – Viết code đo `Dice`, `IoU`, `Recall`, `FPR` và kiểm thử quy tắc Connected Component $\ge 100\text{ px}$
- **Thời gian kế hoạch:** 13/10/2026 – 20/10/2026
- **Ngày hoàn thành:** 19/10/2026
- **Trạng thái:** Hoàn thành (Đạt 7/7 bài kiểm thử tính tay)

---

## 1. Mục tiêu và Phạm vi Công việc

Hạng mục `M4: Metrics` đóng vai trò xây dựng bộ khung đánh giá chuẩn (evaluation protocol) dùng chung cho toàn bộ dự án **FAR-Polyp-Seg**:

1. **Cài đặt các độ đo chuẩn:** `Dice`, `IoU` (Jaccard), `Recall` (Sensitivity), `Precision` và `FPR` (False Positive Rate).
2. **Cài đặt quy tắc Connected Component $\ge 100\text{ px}$:** Loại bỏ các vùng dự đoán nhiễu có diện tích nhỏ hơn $100\text{ pixels}$ trước khi tính toán chỉ số và xác định báo động giả (False Alarm) trên ảnh Normal/Healthy.
3. **Kiểm thử đối chiếu tính tay (Hand-calculated verification):** Thiết kế các mask giả với tọa độ/kích thước xác định, tự tính tay từng phân số và kiểm chứng kết quả xuất ra từ code trùng khớp 100%.
4. **Sẵn sàng bàn giao cho Module 1, 2, 3:** Cung cấp API rõ ràng để thành viên A (đánh giá PraNet Baseline), thành viên B (so sánh Normal-only vs Hard-negative) và Module 4 (Reranker & Ablation cuối) gọi trực tiếp.

---

## 2. Danh mục File Bàn giao

| STT | Đường dẫn File | Mô tả |
| :---: | :--- | :--- |
| 1 | `code/eval/metrics.py` | Mã nguồn chính chứa toàn bộ hàm tiền xử lý nhị phân, tìm miền liên thông (Connected Components), lọc vùng $\ge 100\text{ px}$, tính độ đo mức pixel/mức ảnh và bộ 7 bài test đối chiếu tính tay. |
| 2 | `code/eval/__init__.py` | File khởi tạo package giúp các module khác import ngắn gọn qua `from code.eval import ...`. |
| 3 | `results/M4/tests.txt` | Biên bản log kết quả chạy thực tế 7 kịch bản kiểm thử đối chiếu giữa tính tay và kết quả code. |
| 4 | `docs/HuynhHuuLoc/M4_metrics_report.md` | Tài liệu báo cáo kỹ thuật và hướng dẫn sử dụng chi tiết cho các thành viên trong nhóm. |

---

## 3. Quy tắc Đánh giá và Công thức Toán học

### 3.1. Nhị phân hóa Mask (`binarize_mask`)
- Nếu đầu vào là bản đồ xác suất (probability map $\in [0, 1]$), điểm ảnh được gán nhãn dương tính ($1$) nếu $p \ge \text{threshold}$ (mặc định $\text{threshold} = 0.5$).
- Nếu đầu vào là ảnh mask nguyên `uint8` ($\in [0, 255]$), điểm ảnh được gán nhãn dương tính nếu giá trị $\ge 128$ (hoặc $\ge \text{threshold} \times 255$).

### 3.2. Quy tắc Miền Liên thông $\ge 100\text{ px}$ (`extract_connected_components`)
- Trên mask nhị phân dự đoán $P$, thuật toán tìm tất cả các miền liên thông (mặc định **8-connectivity**, hỗ trợ cấu hình **4-connectivity**).
- Với mỗi miền liên thông $C_k$:
  - Nếu diện tích $|C_k| < 100\text{ px}$: vùng đó bị xem là nhiễu và bị loại bỏ (`rejected_components`).
  - Nếu diện tích $|C_k| \ge 100\text{ px}$: vùng đó được giữ lại (`valid_components`) để tạo thành mask dự đoán sau lọc $\tilde{P}$.

### 3.3. Độ đo mức Điểm ảnh (Pixel-level Metrics)
Gọi $\text{TP}, \text{FP}, \text{FN}, \text{TN}$ lần lượt là số điểm ảnh True Positive, False Positive, False Negative và True Negative giữa dự đoán và Ground Truth ($G$):

$$\text{Dice} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}, \quad \text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Pixel FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

- **Quy ước trường hợp biên (Edge cases):** Khi mẫu số bằng $0$ (ví dụ cả $P$ và $G$ đều rỗng), `Dice`, `IoU`, `Recall`, `Precision` trả về `1.0` (và `Pixel FPR` trả về `0.0`).

### 3.4. Độ đo mức Ảnh và Phân loại TP / FP / TN / FN theo Protocol
- **Đối với ảnh Normal / Healthy ($|G| = 0$):**
  - Ảnh bị tính là **Báo nhầm (`FP`)** nếu và chỉ nếu sau khi lọc miền liên thông $\ge 100\text{ px}$, ảnh còn **ít nhất 1 vùng liên thông hợp lệ** ($|\tilde{P}| > 0$).
  - Ngược lại, nếu mọi đốm dự đoán đều $< 100\text{ px}$ (dù tổng diện tích các đốm rời rạc có thể $\ge 100\text{ px}$), ảnh vẫn được tính là **Đúng âm tính (`TN`)**.
  - **Image-level FPR** trên tập $N_{\text{normal}}$ ảnh Normal:
    $$\text{FPR}_{\text{image}} = \frac{\text{Số ảnh Normal có } \ge 1 \text{ miền liên thông } \ge 100\text{ px}}{N_{\text{normal}}}$$
- **Đối với ảnh Polyp ($|G| > 0$):**
  - Ảnh được tính là **Phát hiện đúng (`TP`)** nếu vùng dự đoán sau lọc ($\ge 100\text{ px}$) có giao với vùng Polyp thật ($\text{TP}_{\text{pixel}} > 0$).
  - Ngược lại là **Bỏ sót (`FN`)**.

---

## 4. Kiến trúc Code và Danh sách Hàm API

Module `code/eval/metrics.py` được thiết kế linh hoạt:
- Hỗ trợ trực tiếp `numpy.ndarray`, `torch.Tensor` (tự động `.detach().cpu().numpy()`) và danh sách 2 chiều thuần Python.
- Tích hợp sẵn thuật toán **BFS Connected Components** bên cạnh `scipy.ndimage.label`, giúp code chạy độc lập mà không bị lỗi thiếu thư viện trên mọi môi trường.

| Tên hàm | Chức năng chính |
| :--- | :--- |
| `binarize_mask(mask, threshold=0.5)` | Chuyển probability map hoặc ảnh `uint8` về mask nhị phân 2D. |
| `extract_connected_components(mask, threshold=0.5, min_area=100, connectivity=8)` | Tách miền liên thông, trả về `(filtered_mask, valid_components, rejected_components)` kèm `area` và `bbox`. |
| `filter_small_components(mask, threshold=0.5, min_area=100, connectivity=8)` | Trả về mask nhị phân chỉ giữ lại các vùng liên thông $\ge 100\text{ px}$. |
| `dice_score`, `iou_score`, `recall_score`, `precision_score`, `pixel_fpr_score` | Các hàm tính từng độ đo đơn lẻ giữa `pred` và `gt`. |
| `image_level_fpr(normal_preds, threshold=0.5, min_area=100)` | Tính tỷ lệ báo nhầm mức ảnh trên danh sách các ảnh Normal. |
| `image_level_recall(polyp_preds, polyp_gts, threshold=0.5, min_area=100)` | Tính tỷ lệ phát hiện đúng mức ảnh trên danh sách các ảnh Polyp. |
| `compute_mask_metrics(pred, gt, threshold=0.5, min_area=100)` | Tính trọn bộ chỉ số (cả trước và sau khi lọc $\ge 100\text{ px}$) và nhãn kết quả (`TP`/`FP`/`TN`/`FN`) cho 1 cặp ảnh. |
| `evaluate_dataset(samples, threshold=0.5, min_area=100)` | Đánh giá toàn bộ tập dữ liệu, tách riêng thống kê nhóm Polyp, nhóm Normal và trả về danh sách `image_id` theo từng nhóm `TP`, `FP`, `TN`, `FN`. |

---

## 5. Kết quả Kiểm thử Đối chiếu Tính tay (`results/M4/tests.txt`)

Toàn bộ 7 bài kiểm thử trong `run_hand_calculated_tests()` đã được chạy và xác nhận khớp tuyệt đối với kết quả tính tay:

| Bài Test | Kịch bản Kiểm thử | Kết quả Tính tay (Hand-calculated) | Kết quả Code (`metrics.py`) | Trạng thái |
| :--- | :--- | :--- | :--- | :---: |
| **Test 1** | Lưới $10\times 10$, GT $20\text{ px}$, Pred $20\text{ px}$, giao nhau $10\text{ px}$ | $\text{TP}=10, \text{FP}=10, \text{FN}=10, \text{TN}=70$<br>$\text{Dice}=0.5, \text{IoU}=1/3, \text{Recall}=0.5, \text{FPR}=0.125$ | `Dice=0.500000`, `IoU=0.333333`, `Recall=0.500000`, `Pixel FPR=0.125000` | **PASS** |
| **Test 2** | Lưới $20\times 20$ bất đối xứng ($\text{GT}=120\text{ px}, \text{Pred}=150\text{ px}, \text{TP}=90$) | $\text{Dice}=2/3, \text{IoU}=0.5, \text{Recall}=0.75,$<br>$\text{Precision}=0.6, \text{Pixel FPR}=3/14$ | `Dice=0.666667`, `IoU=0.500000`, `Recall=0.750000`, `Precision=0.600000`, `FPR=0.214286` | **PASS** |
| **Test 3** | Các trường hợp biên: Trùng khít 100%, Lệch hoàn toàn, Ảnh Normal rỗng | Trùng khít: $\text{Dice}=1, \text{FPR}=0$<br>Lệch: $\text{Dice}=0, \text{FPR}=4/80=0.05$<br>Rỗng–Rỗng: $\text{Dice}=1, \text{FPR}=0$ | Khớp chính xác cả 3 trường hợp (`3a`, `3b`, `3c`) | **PASS** |
| **Test 4** | Kiểm tra biên Connected Component $\ge 100\text{ px}$ với 4 đốm: `45`, `99`, `100`, `150 px` | Loại bỏ `[45, 99]`, giữ lại `[100, 150]`<br>Tổng pixel còn lại $= 250\text{ px}$ | `Rejected=[45, 99]`, `Kept=[100, 150]`, `Filtered pixels=250` | **PASS** |
| **Test 5** | Ảnh Polyp có đốm nhiễu FP `60 px` ($<100\text{ px}$) đi kèm vùng đúng `100 px` | Trước lọc: $\text{Dice}=5/7, \text{IoU}=5/9$<br>Sau lọc $\ge 100\text{ px}$: $\text{Dice}=10/11, \text{IoU}=5/6$ | Trước lọc: `Dice=0.714286`, `IoU=0.555556`<br>Sau lọc: `Dice=0.909091`, `IoU=0.833333` | **PASS** |
| **Test 6** | Tính `Image-level FPR` trên 5 ảnh Normal (có ảnh chứa 2 đốm rời `30+85=115 px` và biên `99 px`, `100 px`) | Chỉ ảnh #2 (`120 px`) và #5 (`100 px`) tính là FP $\Rightarrow \text{FPR} = 2/5 = 0.40$ (Chưa lọc: $4/5 = 0.80$) | `Image-level FPR (CC>=100px) = 0.400000`<br>`Raw FPR = 0.800000` | **PASS** |
| **Test 7** | Đánh giá tập dữ liệu hỗn hợp (2 ảnh Polyp + 5 ảnh Normal) qua `evaluate_dataset` | $\text{Mean Dice}=5/11, \text{Mean IoU}=5/12,$<br>$\text{Image Recall}=0.5, \text{Normal FPR}=0.4$ | Khớp toàn bộ số đo và phân nhóm đúng `TP=['polyp_A']`, `FN=['polyp_B']`, `FP=['norm_2', 'norm_5']` | **PASS** |

---

## 6. Hướng dẫn Sử dụng cho các Thành viên Nhóm

### 6.1. Chạy lại bộ kiểm thử từ Terminal
```bash
python3 code/eval/metrics.py
```
Kết quả chi tiết sẽ được in ra màn hình và cập nhật vào `results/M4/tests.txt`.

### 6.2. Dành cho Thành viên A (Đánh giá M1 Baseline & Xuất ảnh TP/FP/FN)
```python
from code.eval import compute_mask_metrics, evaluate_dataset

# Đánh giá toàn bộ danh sách ảnh trong tập Validation / Test
samples = [
    {
        "image_id": "Patient100__Polyp__1403-09-251048237-7307",
        "patient_id": "Patient100",
        "pred": prob_map_or_binary_mask,  # .npy (float32) hoặc .png (uint8)
        "gt": gt_mask,                    # Mask thật cho ảnh Polyp
    },
    {
        "image_id": "Patient100__Healthy__1402-06-121049271-5126",
        "patient_id": "Patient100",
        "pred": normal_prob_map,          # Ảnh Healthy không cần truyền 'gt'
    },
]

summary = evaluate_dataset(samples, threshold=0.5, min_area=100)

# 1. Lấy bảng số đo chuẩn theo protocol
print("Polyp Segmentation Metrics:", summary["polyp_segmentation"])
print("Normal False Alarm (FPR):", summary["normal_false_alarm"])

# 2. Lấy danh sách image_id theo từng nhóm để lưu ảnh minh họa lỗi (TP / FP / FN)
tp_ids = summary["outcome_image_ids"]["TP"]
fp_ids = summary["outcome_image_ids"]["FP"]
fn_ids = summary["outcome_image_ids"]["FN"]
```

### 6.3. Dành cho Thành viên B & C (Lọc vùng nhiễu và Trích xuất Connected Components)
```python
from code.eval import extract_connected_components

filtered_mask, valid_rois, rejected_rois = extract_connected_components(
    prob_map, threshold=0.5, min_area=100, connectivity=8
)
# valid_rois chứa danh sách các vùng >= 100 px kèm diện tích (area) và tọa độ khung (bbox)
```
