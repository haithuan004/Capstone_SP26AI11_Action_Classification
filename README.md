# Skeleton-Based Action Recognition — ST-GCN / AGCN / SGCN

Dự án Capstone nhận diện **4 hành động** (Standing, Walking, Sitting, Falling) dựa trên dữ liệu **khung xương (Skeleton)**. Dữ liệu được thu thập, gán nhãn qua **CVAT**, xử lý qua pipeline 3 bước (Calibration → Temporal Interpolation → Kinematic Correction), sau đó dùng để huấn luyện và so sánh 3 kiến trúc mạng đồ thị thời gian-không gian: **ST-GCN**, **AGCN**, và **SGCN**.

---

## 📁 Cấu Trúc Thư Mục

```
Labeled_data/
│
├── Data_origin/                  # ⚠️ Dữ liệu gốc từ CVAT — KHÔNG được xóa
│   └── *.zip                     # Annotation thủ công (CVAT format)
│
├── _test_enhanced/               # Zips đã qua pipeline xử lý (tái tạo được)
│   └── *_pipe.zip
│
├── stgcn_augmented/              # Dataset augment & balance cho training
│   ├── train_data.npy
│   ├── train_label.pkl
│   ├── val_data.npy
│   └── val_label.pkl
│
├── stgcn_split_all/              # Dataset split từ toàn bộ dữ liệu (không augment)
│   ├── train_data.npy / train_label.pkl
│   └── val_data.npy  / val_label.pkl
│
├── Models/                       # Định nghĩa kiến trúc mạng & dataset
│   ├── stgcn_model.py            # ST-GCN (pretrained MMAction2)
│   ├── agcn_model.py             # AGCN (Adaptive GCN)
│   ├── sgcn_model.py             # SGCN (Semantics-Guided GCN)
│   ├── stgcn_dataset.py          # Dataset loader, sliding window, augment
│   └── stgcn_8xb16-bone-u100-80e_ntu60-xsub-keypoint-2d_20221129-c4b44488.pth
│                                 # Pretrained weights ST-GCN (MMAction2)
│
├── Finetuning/                   # Scripts huấn luyện từng model
│   ├── finetune_v2.py            # Train ST-GCN
│   ├── finetune_agcn.py          # Train AGCN
│   └── finetune_sgcn.py          # Train SGCN
│
├── Data_preprocessing/           # Augment & balance dữ liệu sau pipeline
│   ├── augment_and_balance.py
│   └── offline_balancing.py
│
├── Visuallising/                 # Phân tích & trực quan hóa kết quả
│   ├── generate_comparison_chart.py
│   ├── plot_best_metrics_table.py
│   ├── plot_model_comparison.py
│   ├── plot_track_distribution.py
│   ├── visualize_augmented_sample.py
│   └── visualize_preprocessing_steps.py
│
├── checkpoints_v4/               # Model weights & metrics tốt nhất được lưu
│   ├── best_stgcn_augmented.pth
│   ├── best_agcn_augmented.pth   # (sau khi train xong)
│   ├── best_sgcn_augmented.pth   # (sau khi train xong)
│   └── best_metrics_*.json
│
├── pipeline_calibration_temporal_kinematic.py   # Pipeline xử lý dữ liệu chính
├── interpolate_keypoints_in_zips.py             # Bước temporal interpolation
├── kinematic_spatial_impute_zips.py             # Bước kinematic imputation
├── export_cvat_zips_to_stgcn.py                 # Chuyển CVAT → numpy array
└── export_best_model.py                         # Xuất model tốt nhất ra file
```

---

## ⚙️ Pipeline Xử Lý Dữ Liệu

```
Data_origin/*.zip
      │
      ▼  pipeline_calibration_temporal_kinematic.py
_test_enhanced/*_pipe.zip
  (1. Calibration: đo L_max từng đoạn xương)
  (2. Temporal Interpolation: nội suy frame bị thiếu)
  (3. Kinematic Correction: chuẩn hóa tỷ lệ tay/chân)
      │
      ▼  export_cvat_zips_to_stgcn.py
raw numpy arrays (N, C, T, V, M)
      │
      ▼  Data_preprocessing/augment_and_balance.py
stgcn_augmented/   (train_data.npy, train_label.pkl, val_data.npy, val_label.pkl)
      │
      ▼  Finetuning/finetune_*.py
checkpoints_v4/best_*.pth
```

> **Lưu ý quan trọng:** `Data_origin/` là **ground truth duy nhất**, không thể tái tạo. `_test_enhanced/` là output trung gian, có thể tái tạo bằng cách chạy lại pipeline.

---

## 🚀 Hướng Dẫn Chạy

### Bước 1 — Xử lý dữ liệu từ CVAT zips gốc

```bash
python pipeline_calibration_temporal_kinematic.py \
    --input-dir Data_origin \
    --output-dir _test_enhanced \
    --suffix _pipe \
    --verbose
```

### Bước 2 — Chuyển annotations sang numpy (STGCN format)

```bash
python export_cvat_zips_to_stgcn.py
```

### Bước 3 — Augment & cân bằng dữ liệu

```bash
python Data_preprocessing/augment_and_balance.py
```

### Bước 4 — Huấn luyện model

```bash
# ST-GCN (có pretrained weights từ MMAction2)
python Finetuning/finetune_v2.py

# AGCN (train from scratch)
python Finetuning/finetune_agcn.py

# SGCN (train from scratch)
python Finetuning/finetune_sgcn.py
```

Model tốt nhất (theo Macro F1) sẽ được tự động lưu vào `checkpoints_v4/`.

---

## 🧠 Các Mô Hình

| Model | File | Pretrained | Đặc điểm |
|-------|------|-----------|----------|
| **ST-GCN** | `Models/stgcn_model.py` | ✅ MMAction2 (NTU60) | Graph tĩnh, nhanh, ổn định |
| **AGCN** | `Models/agcn_model.py` | ❌ From scratch | Graph thích nghi theo dữ liệu |
| **SGCN** | `Models/sgcn_model.py` | ❌ From scratch | Graph hướng dẫn theo ngữ nghĩa |

Tất cả model đều dùng:
- **Focal Loss** (gamma=2) + Label Smoothing 0.2
- **MixUp** augmentation (alpha=0.2)
- **TTA** (Test-Time Augmentation — Flip LR)
- **WandB** logging
- **Cosine Annealing LR** scheduler

---

## 📊 Dữ Liệu

- **4 lớp hành động:** `standing`, `walking`, `sitting`, `falling`
- **17 keypoints COCO** per skeleton
- **Định dạng tensor:** `(N, C=3, T=100, V=17, M=1)` — (sample, XYZ, frames, joints, người)
- **Tập train (augmented):** ~180 tracks, cân bằng 45 samples/class
- **Tập validation:** giữ nguyên từ dữ liệu gốc (không augment)

---

## 📈 Theo Dõi Kết Quả

Kết quả huấn luyện được đẩy tự động lên **Weights & Biases**:

```bash
# Xem dashboard
wandb login
# Sau đó chạy training như bình thường
```

Các metrics được track: `val/accuracy`, `val/macro_f1`, `val/fall_recall`, `val/fall_f1`.

---

## 🗂️ Git & Dọn Dẹp

Các thư mục nên có trong `.gitignore`:

```
Capstone/          # Virtual environment
__pycache__/
wandb/
_test_enhanced/    # Output trung gian, tái tạo được
*.log
```

`Data_origin/`, `stgcn_augmented/`, `checkpoints_v4/` nên được **backup riêng** (cloud hoặc external drive).
