# DACN — Phát hiện bất thường MRI bằng diffusion

Đồ án chuyên ngành 3 người: **DDPM partial reconstruction cho BMAD FLAIR 2D**,
khảo sát timestep bắt đầu tái tạo và score sai khác pixel/local SSIM; DAE là đối chứng khi khả thi.

Hiện có pipeline kiểm tra/chuẩn bị dữ liệu, export dataset riêng tư lên Hugging Face,
trainer baseline DAE coarse-noise, prototype AnoDDPM-BMAD tách biệt, trainer
DDPM partial reconstruction và runner lưu kết quả Modal vào Hugging Face.
Đến 06/11/2026, nhóm đã có checkpoint Gaussian/Simplex và kết quả validation;
nhánh C bổ sung DDIM, mask/stitch/resample và perceptual scoring LPIPS.
Cập nhật C đã tích hợp vào nhánh làm việc; replay metric từ cache hoàn tất.
Cần hoàn thiện runner và freeze protocol trước held-out test. Xem review để phân biệt kết quả
thực với kế hoạch cũ. Demo giả lập chỉ kiểm tra metric, không phải thí nghiệm.
Sản phẩm phục vụ nghiên cứu, không dùng chẩn đoán.

## Bắt đầu ở đây

- [Review toàn dự án 07/11 — trạng thái và việc cần hoàn thành](docs/PROJECT_REVIEW_2026-11-07.md)
- [Audit hiện trạng 06/11 và hướng tiếp theo — đọc trước các snapshot cũ](docs/PROJECT_AUDIT_2026-11-06.md)
- [Kế hoạch 6 tuần và việc cần làm ngay](docs/PROJECT_PLAN.md)
- [Đề bài và yêu cầu dự án](docs/PROJECT_REQUIREMENTS.md)
- [Project Linear, milestone và backlog công việc](docs/PROJECT_TRACKING.md)
- [Worklog](docs/WORKLOG.md)
- [Lessons learned](docs/LESSONS_LEARNED.md)
- [Theo dõi thí nghiệm và kết quả](docs/RESULT_TRACKING.md)
- [Audit: leakage trong bản BMAD tải về](docs/DATA_AUDIT.md)
- [Protocol data và lệnh chạy](docs/DATA_PROTOCOL.md)
- [Hugging Face + Modal: lưu dataset và chạy experiment](docs/HUGGINGFACE_MODAL.md)
- [Google Colab notebook: AnoDDPM smoke test và pilot](notebooks/anoddpm_bmad_colab.ipynb)
- [Local Jupyter notebook: train trên Modal, xem output local](notebooks/anoddpm_modal_local.ipynb)
- [Dataset đã xác minh: revision và cách tải cho nhóm](docs/DATASET_HANDOFF.md)
- [Câu hỏi nghiên cứu, adaptation, thực nghiệm](docs/RESEARCH.md)
- [Paper và thứ tự đọc](docs/REFERENCES.md)
- [Vai trò tạm thời](docs/TEAM.md)

## Cài đặt và kiểm tra

macOS/Linux (zsh hoặc bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[data,dev]"
python -m pytest -q
python -m dacn.cli demo --output runs/demo
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[data,dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m dacn.cli --help
```

Lệnh `demo` tạo dữ liệu tổng hợp, tính pixel difference/local SSIM, chọn ngưỡng
trên validation và dùng lại ngưỡng đã khóa trên test. Kết quả aggregate được ghi
vào `runs/demo/metrics.json`; đây chỉ là kiểm tra pipeline, không phải hiệu năng
mô hình hay kết quả trên dữ liệu y tế.

Không cần GPU cho pipeline data. Cài PyTorch phù hợp thiết bị ở giai đoạn training.
Notebook Colab mặc định chạy smoke test trên 128 ảnh train và 16 ảnh validation;
đổi sang `pilot` chỉ sau khi smoke thành công. Smoke metrics không phải kết quả nghiên cứu.
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

`src/dacn/`: CLI/data/model/training/metric; `tests/`: kiểm thử không cần ảnh y tế;
`configs/`: hợp đồng nghiên cứu; `docs/`: protocol và kế hoạch.
`configs/bmad_flair.yaml` hiện là cấu hình mô tả, CLI **chưa đọc YAML**.
`data/`, `runs/`, `checkpoints/`, `.venv/`, `.env` bị Git ignore. Hub export mặc định
private, bỏ patient ID nguồn và source manifest, chỉ giữ UUID nhóm ẩn danh để
bootstrap theo patient; kiểm tra điều kiện BMAD/BraTS trước khi chia sẻ ra ngoài.
Modal gọi DAE qua `dacn.training.dae:train` hoặc prototype AnoDDPM qua
`dacn.training.anoddpm:train`; candidate DDPM chính chạy qua
`dacn.training.ddpm_partial:train` sau DAC-8 A/B review và hoàn tất data gate DAC-9.
Lệnh bounded smoke và resume nằm trong `docs/HUGGINGFACE_MODAL.md`. Không bắt đầu
timestep/score sweep trước evaluator và review gate DAC-15. Không báo cáo prototype
AnoDDPM như phương pháp chính đã hoàn tất; xem `configs/anoddpm_bmad.yaml` và
third-party notices.
DAE mặc định chạy pilot 3 epoch, chỉ chọn checkpoint bằng validation pixel AP; chưa
đánh giá test và chưa tạo kết quả hiệu năng cho báo cáo.

Giữ `Paper_Reference.xlsx` làm tài liệu nguồn của nhóm. Danh mục bổ sung nằm trong
REFERENCES. Không commit ảnh, manifest patient, credentials hoặc checkpoint.
Làm việc trên feature branch, review trong nhóm trước khi tích hợp; xem CONTRIBUTING.
