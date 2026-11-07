# Frozen evaluation orchestration — 07/11/2026

Entry point: `dacn.evaluation.frozen_runner.run_frozen_evaluation`.

## Đã thực hiện

- Capture tám artifact qua gate trước khi gọi dataset factory.
- Backend nhận captured bytes; thresholds lấy trực tiếp từ frozen manifest.
- Mỗi sample tạo private NPZ shard, receipt gắn checksum và hash stable ID.
- Resume kiểm contract, thứ tự ID và checksum; shard chưa commit được tính lại.
- Gộp labels/scores để tính global image/pixel AUROC, AP, Precision, Recall, F1.
  Không lấy trung bình metric của shard; AUROC một lớp hoặc rỗng bị từ chối.

9 tests runner/gate đạt: từ chối candidate trước dataset access, gián đoạn/resume,
đổi thứ tự và sửa shard. Dữ liệu chỉ là synthetic trong unit tests.

## Adapter DDIM+LPIPS — 07/11 follow-up

`dacn.evaluation.frozen_ddim.run_frozen_ddim` nối gate/orchestrator với adapter
DDIM20+30. Load EMA checkpoint từ captured bytes; LPIPS trunk/head export được
load offline, strict keys. Mask threshold và indices lấy từ artifact đã pin,
không fit lại. Seed UUID theo policy cũ; crop240, LPIPS grayscale-repeatRGB,
mean top576 image score. Kiểm đúng 50 U-Net calls/sample. Runner bổ sung micro
Dice, macro lesion Dice và normal pixel FPR.

Script `scripts/export_lpips_environment.py --output <private-dir>` export đúng
native trunk state, các keys còn lại của LPIPS state và dependency freeze.
Chạy trong runtime gốc nếu còn; chạy runtime mới phải ghi là môi trường mới.
Đã thêm optional extra `perceptual`, LPIPS==0.1.4. Artifact export không đưa lên Git.

## Phần nghiệm thu còn thiếu

Đã có adapter và API Python; chưa có executable MRI final-test CLI được nghiệm thu.
Adapter phải load DDIM/LPIPS từ captured bytes, kiểm source/environment thực thi,
dataset revision/order/content và preprocessing/score đúng protocol.
Factory trả `(stable_id, sample)`; ID hash chỉ kiểm thứ tự, không kiểm nội dung.
Receipt và NPZ phải giữ riêng tư. Output một writer; chưa hỗ trợ concurrent jobs.

Runner concatenate arrays để tính exact global metrics; cần benchmark RAM trên
validation. Chưa có out-of-core ranking, latency/VRAM hoặc adapter controls.
Candidate vẫn chưa frozen khi còn null pins. Tests dùng ZeroModel chứng minh
orchestration/NFE, chưa chứng minh U-Net trained replay trên môi trường mới.

## Điều kiện tích hợp

Nghiệm thu CPU trained G30 ngày 07/11: xem ACCEPTANCE_2026-11-07.md.
`scripts/accept_ddim_validation.py` chạy hai mẫu val, kiểm dữ liệu83/hash/deps,
ngắt chủ động và đối chiếu resume với uninterrupted. `validation_only=True`
cho phép candidate đầy đủ pins ở phạm vi nghiệm thu; final vẫn yêu cầu frozen.
Receipt binds execution_scope nên không thể resume validation thành final.
Không sửa status candidate thành frozen chỉ để vượt gate.

1. Pin source/config, dependency và LPIPS weights từ môi trường thật.
2. Nối bounded validation adapter đúng crop/normalization/seed policy.
3. Đối chiếu cache replay, kiểm interruption/resume; bổ sung Dice/macro Dice/FPR.
4. Benchmark RAM/GPU; freeze controls riêng và protocol trước khi mở test.
