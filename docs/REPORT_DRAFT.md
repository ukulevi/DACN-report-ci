# Diffusion Models for Anomaly Detection — bản thảo nội dung

Trạng thái 07/11/2026: nội dung nghiên cứu đã có bằng chứng; chưa phải báo cáo
hoàn chỉnh hoặc kết quả held-out test. References dùng danh mục REFERENCES.md.

## Mục tiêu và dữ liệu

Nhóm nghiên cứu phát hiện bất thường trên ảnh BMAD Brain FLAIR 2D bằng diffusion
normal-only. Normal ở đây là slice không có tumor annotation, không đồng nghĩa
bệnh nhân khỏe mạnh. Dữ liệu PNG render từ BraTS khác intensity NIfTI gốc.
Preprocessing bảo toàn ảnh, padding và chuẩn hóa theo DATA_PROTOCOL.md.
Patient-disjoint nhằm ngăn các slice của cùng bệnh nhân rò giữa các split.
Dataset revision: `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.

## Phương pháp

DDPM học dự đoán epsilon trên normal slices. Epsilon-MSE đánh giá noise prediction;
không trực tiếp đo tumor segmentation. Partial reconstruction thêm noise đến
timestep đã chọn rồi denoise. Discrepancy giữa input và reconstruction tạo anomaly
map: absolute difference, local SSIM và perceptual LPIPS là các hướng đã khảo sát.

Ứng viên hiện tại dùng Gaussian EMA30, phase 1 DDIM t=300 với 20 calls và phase 2
DDIM t=50 với 30 calls, eta=0. Pseudo-mask, stitch và một lượt resampling được dùng
để hạn chế vùng sửa đổi; LPIPS spatial SqueezeNet tạo score cuối. Chi tiết và hashes
trong FINAL_EVALUATION_CANDIDATE.md. 50 U-Net calls/ảnh là bộ đếm compute;
không thay thế đo latency hoặc peak VRAM.

## Thiết kế đánh giá và kết quả có bằng chứng

Validation gồm 40 calibration và 43 holdout slices; thresholds chỉ fit calibration.
Holdout đã được xem qua nhiều thí nghiệm, nên vẫn có selection bias. B đã replay
metrics từ cached scores và khớp C đến tolerance 1e-10; chưa phải tái chạy U-Net/
LPIPS độc lập. Tất cả số dưới đây là validation.

| Phương án trên phép thử DDIM20+30 | Pixel AP | Pixel F1/Dice |
|---|---:|---:|
| Final LPIPS | 0.312330 | 0.371215 |
| Initial LPIPS | 0.157876 | 0.246871 |
| Naive-stitch LPIPS | 0.180593 | 0.296849 |
| Brightness control | 0.389506 | 0.268051 |

Paired bootstrap theo 25 patient groups trên 43 holdout slices: 999/1000 draws
hợp lệ, seed bootstrap 17. Delta AP final trừ naive stitch có CI95
[0.026164, 0.177945]; final trừ brightness có CI95 [-0.287385, 0.232914].
Chưa có bằng chứng final vượt brightness về AP trên dữ liệu mới. Không dùng kết
quả LPIPS-only của ancestral sampler như đối chứng compute-matched DDIM.
Các image/pixel ROC-AUC, Precision, Recall, F1 và provenance đầy đủ phải lấy từ
aggregate receipt đã replay, xem RESULT_TRACKING.md; không tự điền giá trị thiếu.

## Giới hạn và việc cần hoàn thành trước nộp

Một model/sampling seed, một contrast MRI 2D, validation đã khảo sát lặp và thiếu
scanner/site metadata giới hạn khả năng suy rộng. Brightness control mạnh gợi ý
cần thảo luận bias của dữ liệu và chất lượng anomaly ranking. Không có kết luận
hiệu quả chẩn đoán lâm sàng từ các thí nghiệm này.

Cần hoàn thiện adapter frozen inference, kiểm provenance/environment, chạy bounded
validation để nghiệm thu, đăng ký protocol final rồi mới mở test. Báo cáo cuối cần
thêm bảng metrics image/pixel đầy đủ, artifact-derived figures, runtime/VRAM,
trích dẫn BMAD/BraTS/AnoDDPM/LPIPS và các phương pháp thực sự dùng; cuối cùng
định dạng theo mẫu trường, slides và demo có hướng dẫn tái chạy.
