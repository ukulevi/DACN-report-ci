# Hồ sơ ứng viên đánh giá cuối — 06/11/2026

Trạng thái: **candidate, chưa frozen, chưa cho phép mở test**. Nhóm làm chung,
thứ tự công việc dựa vào prerequisite thay vì chờ người mang role A/B/C.

Manifest máy đọc: [final_evaluation_candidate.json](../configs/final_evaluation_candidate.json).
Các pin chưa xác minh để `null`; không điền hash suy đoán. Gate
`dacn.evaluation.frozen_inputs.capture_frozen_inputs(manifest, artifact_paths)`
kiểm status/schema, immutable dataset revision, artifact hashes và finite
thresholds, rồi trả lại captured bytes. Runner tương lai phải dùng đúng bytes
này, không đọc lại file paths. Gate chưa nối với inference runner và không tự
chạy test. Đây chỉ là contract final LPIPS; controls cần contract riêng.

Gate yêu cầu source snapshot, config, checkpoint, LPIPS backbone/calibration,
dependency lock, validation index và mask parameters. Các file private ở runtime;
manifest chỉ chứa hash và thông số tổng hợp. LPIPS hiện chưa cài tại local và
không có weights trong cache được kiểm tra; cần chụp hashes từ môi trường thực thi.

## Cập nhật pins 07/11/2026

Đã export source snapshot commit cf7de8f và native LPIPS trunk/head cùng dependency
freeze từ môi trường Windows CPU mới; hash điền vào candidate JSON. LPIPS trên
83 cached initial/final pairs khớp cả166maps, maxabs5.2154e-7. Đây là môi trường
mới, không tuyên bố giống runtime lịch sử. Candidate vẫn chưa frozen: còn nghiệm
thu trained DDIM/resume, runtime/source/dataset binding, benchmark và controls.
Các đoạn mô tả thiếu LPIPS ở phía trên là trạng thái trước cập nhật này.

## Bằng chứng đã hoàn thành

Replay pixel/image AUROC, AP, Precision, Recall, F1, threshold, macro lesion Dice
và normal FPR của final LPIPS, initial LPIPS, naive-stitch LPIPS và brightness:
đều khớp kết quả C. Threshold được refit đúng 40 calibration slices; 43 holdout
không dùng chọn lại threshold. Bootstrap và limitations xem RESULT_TRACKING.
Đây là tái tính từ cached scores, không phải chạy lại LPIPS/U-Net độc lập.

## Candidate chính và đối chứng

- Gaussian EMA30 checkpoint SHA256:
  `d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946`.
- Dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.
- Artifact revision `fae9b50a4b78114fb51d87e803c77d019668f523`.
- DDIM phase 1: t=300, 20 calls; phase 2: t=50, 30 calls; eta=0;
  một lượt phase 2; 50 U-Net calls/ảnh, ghi LPIPS calls riêng.
- Mask percentile 95 trên normal calibration, dilation 3, epsilon 1e-8;
  threshold từ run đã đo `0.13344676792621613`.
- LPIPS SqueezeNet version 0.1, spatial=True, grayscale lặp RGB, crop 240,
  input [-1,1], normalize=False; giữ nguyên input adaptation đã đo.
- Final LPIPS pixel threshold `0.26617634296417236`, image threshold
  `0.24719371760470998`; image score là mean top 576 pixels.
- Giữ initial LPIPS, naive-stitch LPIPS và brightness làm controls cùng input.
  Bảng baseline ancestral pixel/local SSIM và LPIPS-only giữ nhãn khác sampler;
  không ghép như đối chứng compute-matched.
- Sampling seed policy hiện tại: base 17 + UUID-derived seed; UUID/derived seeds
  ở private runtime, không xuất vào báo cáo. Những seed bổ sung cần đăng ký trước.

Các thông số trên là snapshot candidate đã đo, không phải lựa chọn lại dựa trên
test. LPIPS package/backbone/calibration weight checksums và source snapshot
thực thi cuối vẫn cần pin. Khi sửa code guard, phải phân biệt source mới với
source đã sinh kết quả lịch sử.

## Thứ tự thực hiện chung

Tiến độ sau review: helper continuation đã có test CPU và prefix đã gắn checksum
NPZ/source/donor-index; checkpoint load dùng snapshot bytes. Còn nghiệm thu source
tích hợp và kiểm manifest thực tế; chưa có remote continuation job mới.

1. Hoàn thiện guard và regression tests đã nêu trong PROGRESS_REVIEW; xác minh
   thật sự path continuation dùng đúng artifact schema. Không yêu cầu quyền
   workspace C để kiểm logic filesystem/mocks tại local.
2. Pin hashes LPIPS/backbone, source/config/dependencies; lập run manifest cho
   test và các controls. Kiểm runner bằng dữ liệu synthetic hoặc validation.
3. Runner phải kiểm trạng thái frozen, hashes và finite thresholds; xử lý theo
   shard/sample, có receipt để resume mà không đổi source/config. Tính metric
   pixel global micro đúng ties; không lấy trung bình AUROC/AP của từng shard.
4. Ghi benchmark cùng GPU/batch/startup scope, peak VRAM và số model calls.
   Chỉ chạy GPU theo authorization/budget, không suy tốc độ từ số calls.
5. Freeze file/commit khi các prerequisite đã đủ. Mọi quyết định thay shortlist,
   seed hoặc threshold phải trước khi đọc test. Nếu không có thêm GPU, ghi rõ
   hạn chế một seed; không điền kết quả dự kiến vào bảng.
6. Chạy test một lần theo protocol frozen, lưu cả interrupted/failed run; giữ
   artifact riêng tư. Tổng hợp kết quả và viết thảo luận/demonstration.

## Viết báo cáo ngay từ bằng chứng đã có

Phần data mô tả patient-disjoint và train normal-only; phần phương pháp phân biệt
training epsilon-MSE, reconstruction, pseudo-mask và anomaly score. Phần kết quả
giữ controls, negative results, budget và provenance. Phần giới hạn nêu validation
đã khảo sát lặp, một model/sampling seed, MRI 2D một contrast và brightness control.
Không dùng chỉ số validation hiện tại làm số test cuối hoặc khẳng định hiệu quả
chẩn đoán. Deadline vẫn 04/12/2026.
