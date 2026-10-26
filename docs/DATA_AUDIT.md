# Audit dữ liệu ngày 24/10/2026

Nguồn: `Brain_AD.zip` tải qua link Drive của [BMAD](https://github.com/DorisBao/BMAD).
Archive 401.031.573 byte; SHA256
`eb7a5aae9ea1752b16d3a76b47595ffd956c0b210444da76853455d4d7be61c8`.
Chi tiết source receipt nằm ở `data/raw/source_receipt.json` trên máy, không trong Git.

## Giới hạn provenance của split nguồn

Artifact được audit là đúng file Drive `Brain_AD.zip` (ID `1xuapqiL1s18eMeKMd-Vi1Cqp2wp7Pghp`, SHA256 ở trên). Phiên bản script, lệnh chạy và lịch sử thư mục output đã tạo ra archive này **chưa xác minh được** từ artifact/repository; không có căn cứ để gắn nó với một commit preprocessing cụ thể.

Để giải thích hành vi của mã nguồn công khai, tại ngày 26/10/2026 đã kiểm tra [`bras.py` ở commit `5b6d583898b0a53c72aaebfa9a3f075c126e1bd5`](https://github.com/DorisBao/BMAD/blob/5b6d583898b0a53c72aaebfa9a3f075c126e1bd5/data_processing/brain/bras.py). Script chọn cohort abnormal validation từ `list(set(...))` rồi tạo test bằng phần bù: một lần chạy sạch cho hai cohort rời nhau, nhưng lựa chọn validation không ổn định giữa lần chạy. Script cũng ghi vào thư mục output hiện có mà không dọn file cũ. File sót từ lần tạo trước là một lời giải thích **có thể có**, chưa được xác nhận cho overlap trong archive này.

Vì vậy, phát hiện overlap bên dưới áp dụng cho đúng archive đã pin, không chứng minh split chính thức trong paper BMAD hoặc mọi bản phát hành BMAD đều bị leakage. Archive cùng SHA256 là đầu vào nguồn duy nhất có thể tái lập ở đây; `patient-disjoint-v1` là split dẫn xuất từ artifact này.

## Phát hiện trên dữ liệu thật

1. Layout train là `train/good/*.png`, mask evaluation nằm trong `label/`.
2. Ảnh/mask PNG RGBA, 240×240, bảng màu bone; alpha255. Mã preprocessing tác giả
   xác nhận cách lưu ảnh. Không được đọc RGBA mask như bốn kênh nhị phân.
3. Bản gốc có đúng 7.500/83/3.715 ảnh nhưng **10 patient ID giao val/test**.
   Có 10 sample ID trùng, đồng thời trùng pixel đã decode; 50 test slice thuộc
   những patient này. Không thấy patient giao train với val/test trong lần kiểm tra.
4. Sau decode đúng protocol, tất cả mask evaluation khớp label nhóm và shape ảnh.

Report kiểm tra nguyên bản: `runs/data-qc/bone-report.json`, `ok=false`, 70 lỗi
gồm 50 lượt patient overlap, 10 sample overlap, 10 pixel overlap. Các con số này
không phải 70 bệnh nhân; nhiều lỗi cùng phát sinh từ một nguồn giao nhau.
Report thử nghiệm `raw-report.json` trước đó dùng giả định grayscale/layout cũ,
không dùng làm kết quả khoa học.

## Quyết định sửa có thể truy vết

Chính sách `patient-disjoint-v1`: giữ train/val, loại mọi test slice có patient ID
thuộc val. Quy tắc chốt từ cấu trúc/ID trước khi train, không xem model score.
Raw archive và ảnh nguồn không thay đổi. Không đưa patient IDs cụ thể vào tài liệu
Git; danh sách đầy đủ cùng hash/lý do nằm trong `qc_report.json` tại processed root.

| Tập | Số ảnh gốc | Số ảnh giữ | Patient IDs giữ |
|---|---:|---:|---:|
| Train | 7.500 | 7.500 | 424 |
| Validation | 83 | 83 | 50 |
| Test | 3.715 | 3.665 | 767 |
| Tổng | 11.298 | 11.248 | 1.241 |

Đầu ra: `data/processed/bmad-flair-patient-disjoint-v1/`.
QC sau sửa: `ok=true`, 0 errors, 50 exclusions. Có cảnh báo chủ ý: split dẫn xuất,
số lượng khác paper, và provenance patient dựa trên tên file/mã preprocessing nguồn.
Không khẳng định đã kiểm tra độc lập danh tính người bệnh hay near-duplicate.

SHA256 manifest:
`6266a6200475643da9f5c4130e04079ce070994e3b2fcdbd8178a0bc84d5a90b`.

Môi trường đã dùng: Python trong `.venv`, NumPy 2.5.4, Pillow 13.0.0,
scikit-learn 1.9.1, pytest 9.1.1. Không cần GPU cho công đoạn này.
Chưa có mô hình được train hay số liệu phát hiện bất thường từ bộ này.

Kiểm chứng lại ngày 25/10/2026: đọc độc lập toàn bộ **11.248 NPZ**, xác nhận
shape/dtype/range, padding, mask nhị phân, số pixel lesion khớp manifest, label
khớp mask, đủ file và 1.241 patient ID không giao split. Log tại
`runs/data-qc/output-verification.log`. Package đã cài editable vào `.venv` và
CLI chạy được trực tiếp. Full unit/integration suite: 32 passed; Ruff: passed
trước lượt review độc lập cuối.

Sau review: sửa CI để cài extra `data,dev`, từ chối PNG palette/tRNS ngoài protocol.
Sáu regression cases tái hiện lỗi trước sửa và đạt sau sửa; **38 tests passed**,
Ruff check và format check đạt (25 Python files). Đây là kiểm tra local; chưa chạy
workflow trên GitHub. Các file BMAD RGBA chuẩn không thay đổi cách chuyển đổi.

Xem [DATA_PROTOCOL](DATA_PROTOCOL.md) để tái tạo và [RESEARCH](RESEARCH.md) để dùng
đúng split/metric. Không so trực tiếp kết quả trên bản sửa với bảng BMAD chính thức.
