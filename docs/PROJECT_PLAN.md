# Kế hoạch đồ án 6 tuần — BMAD Brain / ANDi-FLAIR

Theo dõi việc thực thi và issue theo tuần trong [PROJECT_TRACKING](PROJECT_TRACKING.md).

**Quyết định ngày 24/10/2026:** tái triển khai ANDi cho MRI FLAIR 2D, so sánh với
denoising autoencoder (DAE), rồi nghiên cứu đánh đổi chất lượng–thời gian khi giảm
số timestep dùng để chấm điểm. Nhóm 3 người; người dùng phụ trách nguồn và xử lý data.
Đây là bản thích nghi có kiểm chứng, chưa có kết quả huấn luyện hoặc tuyên bố mới.

## Vì sao chọn hướng này

BMAD cung cấp split và mask sẵn, phù hợp thời hạn 6 tuần. ANDi có mã nguồn và cơ chế
chấm điểm theo timestep cho phép thực nghiệm cải tiến mà không cần huấn luyện lại
mỗi biến thể. DAE là đối chứng đủ mạnh, rẻ hơn diffusion. Không mở đồng thời AE,
VAE, GAN, Isolation Forest, DDPM và DDIM thành sáu nhánh nghiên cứu.

Tên đề tài gợi ý: **Phát hiện và định vị bất thường trên MRI não bằng mô hình
khuếch tán: tái triển khai và khảo sát đánh đổi độ chính xác–chi phí suy luận.**

## Lộ trình và điều kiện hoàn thành

Mốc ngày dưới đây tính từ 24/10, cần điều chỉnh theo lịch nộp chính thức của trường.

| Tuần | Thời gian dự kiến | Kết quả bắt buộc | Điều kiện qua mốc |
|---|---|---|---|
| 1 | 24–30/10 | Chốt nguồn, manifest, QC; đọc BMAD/ANDi; chạy pilot DAE và diffusion | Data version cố định; kiểm tra leakage; đo thời gian/VRAM trên GPU thực tế; cả 3 người chạy được smoke test |
| 2 | 31/10–06/11 | DAE hoàn chỉnh; ANDi-FLAIR có train/checkpoint/scoring | Có heatmap trên validation; metric có kiểm thử; job resume được; ước lượng đủ ngân sách cho baseline |
| 3 | 07–13/11 | Đối chứng ANDi và sweep K=8,16,32 | Cùng checkpoint, noise seeds, tập validation và thiết bị; ghi AP + latency + VRAM; chọn tối đa 1 biến thể chính |
| 4 | 14–20/11 | Lặp lại cấu hình cuối; tùy chọn calibration normal-only | Chốt công thức, ngưỡng, hậu xử lý và danh sách run trước khi mở test; bỏ nhánh phụ nếu chưa ổn định |
| 5 | 21–27/11 | Đánh giá test, bảng ablation, phân tích lỗi | Kết quả đủ seed thực chạy; CI theo patient; lưu config/checkpoint hash; mọi thay đổi sau test phải công khai |
| 6 | 28/11–04/12 | Báo cáo, slide, demo, tập bảo vệ | Chạy lại từ hướng dẫn; bảng số khớp log; giải thích được adaptation, giới hạn và kết quả âm |

**Mức hoàn thành tối thiểu:** data chuẩn + DAE + ANDi-FLAIR + một ablation giảm
timestep + báo cáo tái lập. **Mục tiêu tốt:** 3 seed cho cấu hình cuối, patient-bootstrap
CI, phân tích theo kích thước tổn thương và đồ thị chất lượng–latency. **Mở rộng:**
normal-only calibration, chỉ khi kết quả chính đã ổn định.

## Phân công để chạy song song

| Vai trò tạm thời | Sở hữu chính | Bàn giao |
|---|---|---|
| Bạn — Data / nguồn tham khảo | Nguồn, phiên bản, quyền sử dụng, decode PNG, manifest, QC, thống kê | Dataset contract + audit + ghi chú giới hạn; không tự thay split |
| Thành viên B — Baseline / đánh giá | DAE, metric, ngưỡng validation, logging, benchmark thời gian | Một lệnh train/evaluate; bảng kết quả cùng format với diffusion |
| Thành viên C — Diffusion / tích hợp | ANDi adaptation, checkpoint, scoring timestep, demo | Mô tả sai khác paper; full-vs-sparse scoring cùng checkpoint |

Phân vai theo việc, không gắn cứng với tên phân công cũ. B review data; C review
metric; bạn review nguồn và protocol của C. Mỗi người phải hiểu pipeline chung.

## Việc của bạn trong 72 giờ đầu

1. Đọc [DATA_PROTOCOL](DATA_PROTOCOL.md) và [DATA_AUDIT](DATA_AUDIT.md), chạy lại inspect.
2. Đọc BMAD phần Brain và mã `bras.py`; điền bảng nguồn: modality, split, labels,
   điều kiện sử dụng, liên kết, giới hạn. Đọc ANDi phần data để liệt kê khác biệt.
3. Mở một số ảnh **train/validation** cùng mask; kiểm tra đen nền, alpha, orientation,
   lesion nhỏ. Chỉ dùng test cho QC cấu trúc/nhãn, không chọn preprocessing theo heatmap test.
4. Bàn giao `manifest.csv`, `qc_report.json`, source receipt và một mẫu NPZ cho B/C.
5. Cùng B chốt cách bỏ padding khi tính metric; cùng C chốt shape/range đầu vào.

## Ngân sách và điểm dừng

Tài khoản Modal/Colab/Hugging Face là hạ tầng có thể dùng, chưa xác định GPU, quota
hay số giờ. Pilot tuần 1 ghi GPU/VRAM, batch, ảnh/giây, epoch/giờ và chi phí nếu có.
Thời gian dự kiến = số bước × thời gian/bước + validation + checkpoint. Chỉ phân bổ
run lớn sau số đo này; không hứa huấn luyện đủ số epoch của paper trong 6 tuần.

- Cuối tuần 2 nếu diffusion chưa ổn: giữ DAE, sửa pipeline diffusion tối thiểu,
  bỏ calibration/kiến trúc mới; dành thời gian cho kết quả tái lập đáng tin.
- Cuối tuần 4 khóa phạm vi. Kết quả cải tiến không thắng vẫn hợp lệ nếu đối chứng
  công bằng và phân tích được giới hạn.
- Mỗi ngày cập nhật 3 ý: đã làm, vướng gì, đầu ra kế tiếp. Mỗi tuần có run/demo
  kiểm chứng, không chỉ báo phần trăm công việc.

Chi tiết câu hỏi nghiên cứu và ma trận thực nghiệm: [RESEARCH](RESEARCH.md).
Danh mục đọc đã đối chiếu workbook: [REFERENCES](REFERENCES.md).
