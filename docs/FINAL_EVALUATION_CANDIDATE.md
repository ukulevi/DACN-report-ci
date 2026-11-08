# Hồ sơ ứng viên đánh giá cuối — 06/11/2026

Trạng thái: **candidate, chưa frozen, chưa cho phép mở test**. Nhóm làm chung,
thứ tự công việc dựa vào prerequisite thay vì chờ người mang role A/B/C.

Quyết định08Nov: người dùng giữ candidate, chờ CI hoặc nhóm review. Hồ sơ cụ thể:
[freeze review](FREEZE_REVIEW_2026-11-08.md), [runbook](FROZEN_EVALUATION_RUNBOOK.md).
Cached4arm runner/resume validation đã đạt; chưa có turnkey final wrapper4arms.

Update08Nov: T4 smoke2/full83 và private artifact handoff đã nghiệm thu;
[evidence](COLAB_ACCEPTANCE_2026-11-08.md). Bốn controls khớp cache C. Candidate
JSON đã repin đúng source/archive và môi trường T4 đã nghiệm thu, loại nhãn CPU
cũ. [Static preflight](evidence/2026-11-08-final-candidate-preflight.json) xác minh
9artifacts và95file Python trong archive. Đây là kiểm offline trên bytes đã tải,
không chứng minh một runtime Colab mới có cùng packages. Còn review freeze và
live runtime binding trước final; candidate vẫn bị gate từ chối chạy final.

Manifest máy đọc: [final_evaluation_candidate.json](../configs/final_evaluation_candidate.json).
Các pin hiện có provenance được điền trong JSON; không điền hash suy đoán. Gate
`dacn.evaluation.frozen_inputs.capture_frozen_inputs(manifest, artifact_paths)`
kiểm status/schema, immutable dataset revision, artifact hashes và finite
thresholds, rồi trả lại captured bytes. Runner tương lai phải dùng đúng bytes
này, không đọc lại file paths. Gate đã nối với DDIM adapter và có bounded CPU
acceptance. Validation-only candidate execution có scope riêng; final yêu cầu
frozen. Controls dùng contract riêng trong final_control_candidates.json.

Gate yêu cầu source snapshot, config, checkpoint, LPIPS backbone/calibration,
dependency lock, validation index, registered validation partition và mask parameters. Các file private ở runtime;
manifest chỉ chứa hash và thông số tổng hợp. LPIPS đã export và xác minh output
trên83cachedpairs tại local. Runtime final vẫn phải đối chiếu đúng các artifact
đã pin, không tự suy rằng cùng tên package thì cùng môi trường.

## Pins hiện hành 08/11/2026

Source archive d5889d51a1a19213c6e533412b4582d645354c88,
SHA256a3c5345eb494b88cc75da95e7a7a588e58c0e864d7c039d77239f1f2affefd8b;
LPIPS weights/dependency inventory lấy từ ZIP T4 đã tải và kiểm10file hashes.
Source thực thi final phải được extract từ đúng archive này; checkout HEAD đã
có thay đổi generator nên không tự coi là cùng source tree. Candidate/control
JSON là hồ sơ bên ngoài archive, không sửa ngược ZIP lịch sử. Giữ nguyên4arms,
thresholds40calibration và seed17; việc nâng lên frozen cần review riêng.

## Lịch sử pins 07/11/2026 — đã thay bằng T4 pins ở trên

Nghiệm thu trained DDIM/resume giới hạn hai mẫu CPU đã đạt; arrays khớp lượt
liên tục tuyệt đối, tổng200 U-Net calls. Xem ACCEPTANCE_2026-11-07.md.
Source pin mới là archive commit84382a8 có sửa config mapping; không dùng archive
cf7de8f cho adapter cũ. ZIP source của lượt nghiệm thu được lưu riêng theo receipt.
Trạng thái vẫn candidate vì còn acceptance controls/runtime final/benchmark.

Đã export source snapshot commit cf7de8f và native LPIPS trunk/head cùng dependency
freeze từ môi trường Windows CPU mới; hash điền vào candidate JSON. LPIPS trên
83 cached initial/final pairs khớp cả166maps, maxabs5.2154e-7. Đây là môi trường
mới, không tuyên bố giống runtime lịch sử. Nghiệm thu trained DDIM/resume đã đạt
ở hai mẫu; runtime final và benchmark/controls vẫn phải có bằng chứng riêng.

## Bằng chứng đã hoàn thành

Replay pixel/image AUROC, AP, Precision, Recall, F1, threshold, macro lesion Dice
và normal FPR của final LPIPS, initial LPIPS, naive-stitch LPIPS và brightness:
đều khớp kết quả C. Threshold được refit đúng 40 calibration slices; 43 holdout
không dùng chọn lại threshold. Bootstrap và limitations xem RESULT_TRACKING.
Lượt08Nov đã tái chạy LPIPS/U-Net trên T4, tải private maps và tính lại metric CPU;
map khớp C trong1e-6. Chi tiết/timing/limitations trong acceptance08Nov.

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
test. LPIPS/package inventory/source đã pin từ lượt T4 nghiệm thu. Khi runtime
mới khác bytes hoặc packages, phải nghiệm thu lại thay vì tự thay hash để qua gate.

## Việc hiện còn cần thực hiện — 08/11/2026

1. Review/freeze candidate và cả4control contracts: checkpoint/sampler, thresholds,
   registered partition, seed17 và giới hạn một seed/validation đã quan sát.
2. Xử lý CI account/billing hoặc ghi quyết định nghiệm thu local evidence của
   nhóm. Không suy local checks là CI xanh.
3. Trước khi mở test, xác minh live final runtime/source/import tree/dependencies
   đúng pins. Dùng source archive đã nghiệm thu và manifests bên ngoài archive;
   không chạy trực tiếp HEAD rồi thay hash cho qua gate.
4. Sau freeze mới thực hiện final evaluation theo protocol, rồi hoàn thiện
   figures, report/slides/demo. Chưa có final-test run trong hồ sơ này.

## Thứ tự thực hiện chung trước nghiệm thu — lịch sử06–07/11

Tiến độ tại thời điểm06–07Nov: helper continuation đã có test CPU và prefix đã gắn checksum
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
