# DACN — Phát hiện bất thường MRI bằng diffusion

Đồ án chuyên ngành 3 người, kế hoạch 6 tuần: **ANDi thích nghi cho BMAD FLAIR 2D**,
đối chứng DAE, nghiên cứu giảm timestep để cân bằng chất lượng và thời gian suy luận.

Hiện có pipeline kiểm tra/chuẩn bị dữ liệu, kiểm thử và kế hoạch nghiên cứu.
**Chưa huấn luyện DAE/ANDi; chưa có số liệu hiệu năng mô hình.** Demo cũ chỉ dùng số
giả lập để kiểm tra metric. Sản phẩm phục vụ nghiên cứu, không dùng chẩn đoán.

## Bắt đầu ở đây

- [Kế hoạch 6 tuần và việc cần làm ngay](docs/PROJECT_PLAN.md)
- [Audit: leakage trong bản BMAD tải về](docs/DATA_AUDIT.md)
- [Protocol data và lệnh chạy](docs/DATA_PROTOCOL.md)
- [Câu hỏi nghiên cứu, adaptation, thực nghiệm](docs/RESEARCH.md)
- [Paper và thứ tự đọc](docs/REFERENCES.md)
- [Vai trò tạm thời](docs/TEAM.md)

## Cài đặt và kiểm tra

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[data,dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m dacn.cli --help
```

Không cần GPU cho pipeline data. Cài PyTorch phù hợp thiết bị ở giai đoạn training.
Từ raw root đã giải nén theo DATA_PROTOCOL:

```powershell
.\.venv\Scripts\python.exe -m dacn.cli data prepare `
  --root data/raw/bmad/BraTS2021_slice --encoding bmad-bone `
  --split-policy patient-disjoint-v1 `
  --output data/processed/bmad-flair-patient-disjoint-v1
```

Lệnh chỉ tạo output mới. `original` là policy mặc định và sẽ phát hiện leakage trong
bản nguồn đã tải; `patient-disjoint-v1` loại các slice test có patient ID trong val,
ghi exclusion ledger, rồi kiểm tra lại. Đây là split nghiên cứu sửa từ BMAD, không
so trực tiếp với số paper gốc. Không chọn tham số dựa trên test.

## Thư mục và cộng tác

`src/dacn/`: CLI/data/metric; `tests/`: kiểm thử không cần ảnh y tế hoặc GPU;
`configs/`: hợp đồng nghiên cứu; `docs/`: protocol và kế hoạch.
`configs/bmad_flair.yaml` hiện là cấu hình mô tả, CLI **chưa đọc YAML**.
`data/`, `runs/`, `checkpoints/`, `.venv/` bị Git ignore.

Giữ `Paper_Reference.xlsx` làm tài liệu nguồn của nhóm. Danh mục bổ sung nằm trong
REFERENCES. Không commit ảnh, manifest patient, credentials hoặc checkpoint.
Làm việc trên feature branch, review trong nhóm trước khi tích hợp; xem CONTRIBUTING.
