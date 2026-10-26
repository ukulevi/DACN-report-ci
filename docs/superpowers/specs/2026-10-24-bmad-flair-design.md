# Thiết kế nghiên cứu: ANDi trên MRI FLAIR 2D

Ngày quyết định: 24/10/2026. Người dùng giao quyền lựa chọn và thực hiện phương án đề xuất.

## Mục tiêu và phạm vi

Đồ án chuyên ngành 3 người, còn 6 tuần. Tái triển khai phương pháp ANDi trên BMAD Brain
FLAIR một kênh và nghiên cứu đánh đổi giữa chất lượng phát hiện/định vị với chi phí suy luận.
Người dùng phụ trách nguồn và xử lý dữ liệu; các vai trò khác còn có thể đổi.
Cho phép tái lập và nghiên cứu cải tiến, không bắt buộc phát minh thuật toán mới.

Deliverable của đợt chuẩn bị này: kế hoạch 6 tuần, protocol dữ liệu, danh mục đọc bổ sung,
CLI kiểm tra/chuyển đổi BMAD và kiểm thử. Huấn luyện mô hình và số liệu nghiên cứu là các
mốc tiếp theo; đợt này không báo cáo kết quả mô hình từ dữ liệu giả lập.

## Các phương án đã cân nhắc

1. BMAD Brain FLAIR 2D: chọn vì có ảnh, mask và split sẵn, giảm thời gian khởi động.
2. BraTS đa chuỗi: gần ANDi gốc hơn nhưng tăng tải dữ liệu, xử lý và huấn luyện.
3. IXI + bộ bệnh lý: để nghiên cứu ngoài phạm vi chính vì khác nguồn chụp và chuỗi ảnh.

Đây là quyết định do trợ lý thực hiện theo quyền được giao. Nếu sau 48 giờ chưa lấy được
BMAD, thử nguồn tải chính thức được tác giả liên kết; báo rõ trở ngại, không tự thay bộ
dữ liệu hoặc dùng một mirror không rõ phiên bản.

## Thiết lập khoa học

- ANDi gốc dùng 4 chuỗi MRI; bản FLAIR và hậu xử lý 2D là adaptation có ghi chép.
- DAE với coarse noise là baseline; một ANDi-FLAIR đầy đủ là đối chứng cho giảm timestep.
- Thử tập timestep thưa K=8,16,32 trong khoảng của đối chứng. Cố định cách chọn trên val.
- Hiệu chỉnh điểm bằng dữ liệu normal là hướng mở rộng, chỉ thử sau khi đối chứng ổn định.
- Kết hợp hai thay đổi phải có bốn hàng so sánh: gốc, A, B, A+B.
- Primary localization metric: pixel average precision (AP, không đánh đồng với PR-AUC
  tính bằng trapezoid). Báo thêm image AUROC/AP, Dice ở ngưỡng khóa trên validation,
  thời gian và peak GPU memory trên cùng thiết bị/batch. Dùng patient-bootstrap khi ID
  được xác minh; không gọi nhãn slice là nhãn bệnh nhân khỏe mạnh.
- Test chỉ dùng đánh giá cuối sau khi khóa cấu hình. Không chọn threshold trên test.
- Ưu tiên 3 seed cho cấu hình cuối; nếu thiếu ngân sách phải công bố số seed thực chạy.
- Thành công không phụ thuộc cải tiến thắng: thí nghiệm tái lập, số liệu đúng và phân tích
  thất bại vẫn là đầu ra hợp lệ. Không hứa đạt số liệu paper hoặc ứng dụng lâm sàng.

## Protocol dữ liệu

Nguồn: https://github.com/DorisBao/BMAD và https://arxiv.org/html/2306.11876v3.
Paper mô tả 7.500 train, 83 validation, 3.715 test. Đây là số tham chiếu, CLI phải đo lại.
Normal = lát cắt không có tumor annotation; không suy ra người bệnh khỏe mạnh.

Giữ split gốc làm nguồn bất biến (xem ngoại lệ D2 bên dưới): thư mục `valid` được ánh xạ thành `val`. Kiểm tra patient ID từ stem
`<patient_id>_<slice_index>.png`; xác minh quy ước bằng provenance của bản tải trước khi
tuyên bố độc lập bệnh nhân. Không tự chia lại từng ảnh. Manifest lưu ID dạng chuỗi.

CLI chấp nhận layout archive thực: train/good/*.png; valid và test có
{good,Ungood}/img và /label. Chế độ mặc định grayscale vẫn từ chối màu; nguồn thật
phải chọn encoding=bmad-bone. Ảnh RGB/RGBA opaque chuyển Pillow L; mask foreground
là any RGB>0, bỏ alpha; đây là độ sáng PNG, không phải intensity MRI nguyên thủy.
Mask eval bắt buộc, cùng kích thước và khớp nhãn; lỗi input luôn chặn xuất.

Audit 24/10 tìm thấy 10 patient ID giao val/test (10 ảnh trùng, 50 test slice).
Quyết định D3 cập nhật yêu cầu giữ split: giữ raw bất biến; mặc định original phải
báo lỗi; chỉ khi chọn patient-disjoint-v1 mới giữ train/val và loại toàn bộ test
slice của patient thuộc val. Lưu exclusion ledger, không sửa raw, kiểm tra leakage
lại sau loại trừ. Đây là split dẫn xuất, không so trực tiếp với paper BMAD.
Metadata gồm source, modality, sample_id, patient_id, slice_index, split, label,
image_path, mask_path, width, height, file_sha256, pixel_sha256, lesion_pixels.
Đường dẫn tương đối với root. Không ghi ảnh y tế vào Git. Metadata nguồn giữ nguyên.

Preprocessing chính: padding đối xứng ảnh 240x240 vào 256x256, không resize tổn thương.
Cho phép kích thước ảnh vuông nhỏ hơn target (fixture/pilot), từ chối target nhỏ hơn ảnh.
Ảnh float32 shape (1,H,W), chuẩn hóa cố định x/127.5-1; mask uint8 (H,W) với 0/1.
Không histogram matching/equalization trong baseline. Không học thống kê trên test.
Một file npz/mẫu; manifest đầu ra và QC lưu riêng với dữ liệu. Xuất vào thư mục mới,
không ghi đè. Kiểm tra toàn bộ input trước khi tạo output; lỗi khi xuất không để lại
một bộ dữ liệu hoàn chỉnh giả. Báo cáo QC ghi nguồn là chưa xác minh ID độc lập.

## Giao diện kỹ thuật cho đợt chuẩn bị

- `inspect_bmad(root: Path) -> tuple[list[dict], dict]`: đọc input, không ghi file;
  trả manifest và report có `ok`, `errors`, `warnings`, `counts`, `patients`.
- `prepare_bmad(root: Path, output: Path, image_size: int = 256) -> dict`:
  validate trước, xuất atomically qua thư mục tạm cạnh output.
- `python -m dacn.cli data inspect --root PATH --output REPORT.json`:
  exit 0 nếu QC đạt, 1 nếu QC không đạt, báo lỗi dễ hiểu cho đường dẫn sai.
- `python -m dacn.cli data prepare --root PATH --output PATH --image-size 256`:
  exit 0 sau khi xuất đầy đủ, 1 nếu dữ liệu không hợp lệ hoặc output đã tồn tại.
- Pillow là optional extra `data`; tính metric/demo cũ vẫn dùng được khi chưa cài Pillow.
- Python >=3.10. Giữ API chấm điểm hiện có; không thêm torch cho công việc dữ liệu này.

## Rủi ro, giới hạn và quyết định thực hiện

- BMAD validation nhỏ: giới hạn số thử nghiệm, ghi search budget và không mở test để tune.
- Phần ANDi 3D filtering của paper không áp dụng nguyên xi lên các slice rời. Bản FLAIR
  dùng cấu hình 2D chung được khai báo; không dựng giả một volume đủ lát cắt từ BMAD.
- Patient grouping từ tên file là kiểm tra tính nhất quán, không tự chứng minh provenance.
- Cloud account không chứng minh hạn mức GPU. Pilot tuần 1 đo thời gian/bộ nhớ trước khi
  nhóm phân bổ giờ GPU; chưa khởi tạo job tính phí trong đợt chuẩn bị này.
- Workbook người dùng là tài liệu nguồn, không thay đổi. Danh mục bổ sung được lưu ở
  docs/REFERENCES.md để nhóm đọc và cập nhật trạng thái tập trung.
