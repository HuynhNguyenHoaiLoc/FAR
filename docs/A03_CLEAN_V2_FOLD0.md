# A03 · PraNet baseline Fold 0 / Clean V2 · acceptance

## 1. Trạng thái thực tế

- Giao thức: `inner_clean_v2_v1`; `SEG_TRAIN` 215 Polyp, `SEG_VAL` 44 Polyp.
- **Checkpoint lịch sử:** best epoch **71**, native validation Dice **0.748697476988594**; SHA-256 `8c75a4fb5675d0dc7985704dcbf2a44e2545f603645dec31ff38c61e9f91561a`.
- Best checkpoint và `history.csv` gốc thuộc notebook Kaggle phiên bản https://www.kaggle.com/code/eoquiither/xulyanh?scriptVersionId=359730609
- **AMP hiện chưa được chứng minh bằng metadata/log hiện có**. Không tự gắn nhãn AMP=true cho lần train cũ.
- File `results/module1/inner_clean_v2_v1/fold0/configs/train_config.yaml` là **bản phục dựng** tham số đã được metadata xác nhận, **không phải cấu hình gốc được ghi lúc chạy**; loss/augmentation/AMP chưa thể chứng minh đầy đủ.
- Toàn bộ trọng số mới và 2.717 cặp prediction vẫn lưu trên Kaggle, không commit `.pth` vào GitHub.

## 2. Những điều kiện nghiệm thu A03

| Điều kiện | Bằng chứng cần có |
|---|---|
| Đúng dữ liệu, không leakage | SHA256 split CSV, patient/group disjoint, 215/44 Polyp |
| Seed, config | `run_metadata.json`, `configs/train_config.yaml` có phân biệt phục dựng và gốc |
| Train/validation loss & Dice mỗi epoch | `history.csv` gốc (không tự bịa `val_loss`) |
| Checkpoint best validation | max `val_dice` trùng `best_epoch`; checkpoint SHA256 |
| Nạp lại mô hình | `PraNet(channel=32, pretrained=False)` + `load_state_dict(strict=True)` |
| **CUDA AMP** | metadata AMP + bằng chứng `autocast`/`GradScaler` thực thi trong history |

`a03_verify.py` xuất `a03_audit.json`: **PASS** khi có đủ chứng cứ, **PENDING** nếu thiếu AMP/loss/checkpoint, **FAIL** nếu lệch hash, split, lịch sử. Không xem `PENDING` là hoàn thành.

## 3. Kaggle · Kiểm chứng checkpoint cũ trước, không train lại ngay

1. Clone branch của PR #7 sau khi thêm các file A03.
2. Tải/truy cập notebook output bản `359730609` và xác định file gốc `best_model.pth`, `history.csv`, `run_metadata.json`. Nếu notebook mới không thấy `/kaggle/working/M1_CLEAN_V2_INNER_FOLD0/`, hãy gắn output đã lưu thành **Input** hoặc mở notebook cũ bằng đúng version; không giả định rằng output Kaggle tồn tại trong session mới.
3. Clone upstream FAR: `git clone https://github.com/FAR-POLYP-SEG/Codes.git /kaggle/working/FAR_UPSTREAM` và checkout commit `ad3bf1c51529bda6de50badedba124e1e15a653c` (nếu repository vẫn có commit).
4. Chạy lệnh (chỉnh **đường dẫn thực tế**):

```bash
python code/module1/a03_verify.py \
  --run results/module1/inner_clean_v2_v1/fold0 \
  --splits data/splits/inner_clean_v2_v1/image_inner_splits_cv5_candidate.csv \
  --checkpoint /kaggle/working/M1_CLEAN_V2_INNER_FOLD0/best_model.pth \
  --history /kaggle/working/M1_CLEAN_V2_INNER_FOLD0/history.csv \
  --upstream /kaggle/working/FAR_UPSTREAM \
  --out /kaggle/working/A03_FOLD0_ACCEPTANCE
```

**Quan trọng:** `a03_verify.py` cố tình chặn PASS nếu `run_metadata.json` của checkpoint lịch sử không ghi `amp_enabled: true` và history thiếu các trường `amp_enabled`, `grad_scaler_enabled` cho mọi epoch. Điều này là thiếu *bằng chứng*, chưa khẳng định lần train cũ đã tắt AMP. Muốn xác nhận lần train cũ có AMP, cần đối chiếu **cell training gốc đã thực thi** hoặc log lúc train, rồi bổ sung chứng cứ có thể kiểm toán; không tự sửa metadata cũ.

## 4. Chỉ nếu AMP thực sự thiếu: train lại Fold 0 với AMP

`code/module1/a03_train_amp.py` tạo **thí nghiệm mới**, dùng `torch.autocast(cuda,float16)` và `torch.amp.GradScaler(cuda)`, train Polyp SEG_TRAIN, chọn best native-Dice trên SEG_VAL, ghi per-epoch train/val loss, validation Dice, `amp_enabled`, seed, checkpoint hash; reload strict sau train. Lệnh Kaggle:

```bash
python code/module1/a03_train_amp.py \
  --dataset /kaggle/input/datasets/eoquiither/far-polyp-seg-clean-v1-candidate/far_polyp_seg_clean_v1_candidate/Dataset \
  --splits data/splits/inner_clean_v2_v1/image_inner_splits_cv5_candidate.csv \
  --upstream /kaggle/working/FAR_UPSTREAM \
  --out /kaggle/working/M1_CLEAN_V2_INNER_FOLD0_AMP_A03
```

Lệnh tự từ chối ghi đè vào thư mục output đã tồn tại và có file. **Không** tự động thay checkpoint lịch sử. Nếu nhóm chọn checkpoint AMP mới, cần **chạy lại A04 inference** toàn bộ 2.717 ảnh bằng checkpoint SHA256 mới trước khi M2–M4 tiếp tục; không trộn prediction/checkpoint của hai run. Kết quả Dice mới có thể khác 0.748697.

Xác minh sau train bằng cách trỏ `--run`, `--checkpoint`, `--history` vào thư mục **mới** và dùng `--out /kaggle/working/A03_AMP_ACCEPTANCE`. Nếu status=PASS, mới được ghi `A03 technically complete, pending team review`.

## 5. GitHub PR #7

Commit script và tài liệu **trước**, sau khi chạy Kaggle thêm `a03_audit.json` (và nếu có, `history.csv` bản AMP mới, `run_metadata.json` có provenance) vào đường dẫn `results/module1/inner_clean_v2_v1/fold0/a03_acceptance/` hoặc thư mục versioned cho run mới. **Không** commit `.pth` nặng; chia sẻ link Kaggle notebook output và SHA256.

**Không thay đổi các file:** `results/module1/fold0/` (legacy), `results/module1/inner_clean_v2_v1/fold0/predictions.csv` (A04 bản cũ) cho đến khi team khóa checkpoint chính thức.

### Lưu ý newline giữa Windows và Kaggle

Git checkout Windows có thể chuyển line endings của `image_inner_splits_cv5_candidate.csv` từ LF sang CRLF. Để không báo giả lỗi SHA256, `a03_verify.py` so sánh **SHA256 raw** và SHA256 sau khi chuẩn hóa CRLF → LF. Chỉ chấp nhận nếu một trong hai khớp **đúng digest của metadata**; không bỏ qua checksum.
