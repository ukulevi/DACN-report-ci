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

## Phần tích hợp còn thiếu

Orchestration dùng callback, chưa phải executable MRI final-test runner.
Adapter phải load DDIM/LPIPS từ captured bytes, kiểm source/environment thực thi,
dataset revision/order/content và preprocessing/score đúng protocol.
Factory trả `(stable_id, sample)`; ID hash chỉ kiểm thứ tự, không kiểm nội dung.
Receipt và NPZ phải giữ riêng tư. Output một writer; chưa hỗ trợ concurrent jobs.

Runner concatenate arrays để tính exact global metrics; cần benchmark RAM trên
validation. Chưa có out-of-core ranking, latency/VRAM, macro lesion Dice hoặc
adapter controls. Candidate vẫn chưa frozen khi còn null pins.

## Điều kiện tích hợp

1. Pin source/config, dependency và LPIPS weights từ môi trường thật.
2. Nối bounded validation adapter đúng crop/normalization/seed policy.
3. Đối chiếu cache replay, kiểm interruption/resume; bổ sung Dice/macro Dice/FPR.
4. Benchmark RAM/GPU; freeze controls riêng và protocol trước khi mở test.
