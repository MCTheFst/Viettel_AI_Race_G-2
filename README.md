# 3D Parcel Localization - Viettel AI Race

## Thông tin
- **Tên đội:** DataScight
- **Thành viên thực hiện nhiệm vụ:** Phạm Mạnh Cường
- **Nhiệm vụ:** 3D Parcel Localization with Orientation Estimation

## Tổng quan giải pháp

### Kiến trúc mô hình
- **Backbone:** Vision Transformer (ViT-Base) với patch size 16x16
- **Input:** RGB + Depth (4 channels)
- **Outputs:**
  - Segmentation mask (2 classes)
  - 3D center coordinates (x, y, z)
  - Surface normal vector (Rx, Ry, Rz)
  - Auxiliary quaternion (4D)

### Đổi mới chính
1. **Orientation-focused training:**
   - Sign-invariant angular loss: θ = arccos(|n · n̂|)
   - Mask-weighted per-pixel depth normal supervision
   - Quaternion consistency regularization

2. **Multi-modal fusion:**
   - Depth token encoder kết hợp với CLS token
   - Per-pixel normal guidance từ depth gradients
   
3. **Training strategy:**
   - OneCycleLR scheduler với warm-up
   - Mixed precision training (AMP)
   - Early stopping based on Orientation Error

## Cài đặt môi trường

### 1. Tạo môi trường ảo (khuyến nghị)
```bash
conda create -n viettel_cv python=3.10
conda activate viettel_cv
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Kiểm tra GPU
```python
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))
```

## Cấu trúc dữ liệu

Đảm bảo dữ liệu được tổ chức theo cấu trúc sau:

```
data/
├── train/
│   ├── rgb/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   ├── depth/
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   └── ...
│   ├── ply/
│   │   ├── 0000.ply
│   │   ├── 0001.ply
│   │   └── ...
│   └── Public train.csv
└── private/
    ├── rgb/
    ├── depth/
    └── ply/
```

## Huấn luyện mô hình

### 1. Cấu hình đường dẫn

Sửa đường dẫn trong file `task3.py`:

```python
class Config:
    BASE_DIR = Path('/path/to/your/data')  # Thay đổi đường dẫn này
    TRAIN_DIR = BASE_DIR / 'train'
    # ...
```

### 2. Chạy training

```bash
python task3.py
```

### 3. Các tham số quan trọng

Có thể điều chỉnh trong class `Config`:

- `IMAGE_SIZE`: 224 (resolution đầu vào)
- `BATCH_SIZE`: 8 (giảm nếu thiếu GPU memory)
- `NUM_EPOCHS`: 40 
- `LR`: 3e-4 (learning rate khởi đầu)
- `MAX_LR`: 1e-3 (max learning rate cho OneCycle)
- `WEIGHT_NORMAL`: 2.0 (trọng số cho orientation loss)

### 4. Theo dõi training

Training sẽ in ra các metrics sau mỗi epoch:
- **Loss:** Tổng loss và breakdown (seg, center, normal)
- **IoU:** Intersection over Union cho segmentation
- **OE (deg):** Orientation Error trung bình (độ)
- **OE (norm):** Normalized OE (0-1, dùng cho early stopping)

### 5. Checkpointing

- Model tốt nhất (theo OE normalized thấp nhất) được lưu tại: `checkpoints/best_model.pth`
- Đường dẫn tới file best_model.pth: https://drive.google.com/file/d/1craYpqEtinCoXuwpvsuYUnX-NiGKm3Ji/view?usp=sharing
## Inference và tạo submission

### 1. Chạy inference

```bash
python task3.py  # Hàm inference() sẽ tự động chạy sau training
```

Hoặc chạy riêng inference:

```python
from train import inference, Config

inference(
    model_path='checkpoints/best_model.pth',
    test_dir='/path/to/private', # Thay đổi đường dẫn này
    save_csv='Submission_3D.csv'
)
```

### 2. Output

File `Submission_3D.csv` với format:

```csv
image_filename,x,y,z,Rx,Ry,Rz
image_0000.png,0.123,-0.045,1.234,0.001,0.002,0.999
...
```
## Tái tạo kết quả

### Để tái tạo kết quả Private test tốt nhất:

1. **Sử dụng checkpoint đã cung cấp:**
```bash
# Download checkpoint (nếu lưu trên cloud)
# Chạy inference với checkpoint
python -c "from train import inference; inference(model_path='checkpoints/best_model.pth')"
```

2. **Training lại từ đầu:**
```bash
# Set seed để đảm bảo reproducibility
python task3.py
```

**Lưu ý:** Do tính chất ngẫu nhiên của training, kết quả có thể dao động nhẹ (±1-2%).

## Yêu cầu hệ thống

### Tối thiểu:
- GPU: 8GB VRAM (GTX 1080, RTX 2070)
- RAM: 16GB
- Thời gian training: ~2-3 giờ (40 epochs)

### Khuyến nghị:
Bài làm được train bằng A100 trên colab mất 5-10 phút, BTC có thể cân nhắc ạ.

## Troubleshooting

### 1. Out of Memory
```python
# Giảm batch size trong Config
BATCH_SIZE = 4  # hoặc 2
```

### 2. Slow training
```python
# Giảm số workers
num_workers=2  # trong DataLoader
```

### 3. Model không converge
- Kiểm tra data augmentation (có thể quá mạnh)
- Điều chỉnh learning rate (giảm MAX_LR)
- Tăng GRAD_ACCUM_STEPS nếu batch size nhỏ

## Tài liệu tham khảo

1. Vision Transformer: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
2. timm library: https://github.com/huggingface/pytorch-image-models
3. Orientation estimation methods

## Liên hệ

- **Email:** windowcuong100@gmail.com
- **GitHub:** MCTheFst

**Lưu ý cho BTC:** Repository này được tạo cho mục đích hậu kiểm kết quả Private test của cuộc thi Viettel AI Race. File CSV và code đảm bảo tái tạo kết quả đã nộp trên hệ thống chấm điểm.