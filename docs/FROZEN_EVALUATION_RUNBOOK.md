# Vận hành và review trước final evaluation

Quyết định người dùng08/11/2026: **giữ candidate, chờ CI hoặc nhóm review**.
Không đổi status thành frozen để vượt gate, không tải/chạy test trong bước này.

## Những đường chạy đã có

- `run_frozen_evaluation`: generic orchestration, kiểm9artifact hashes trước
  dataset factory, private per-sample NPZ shards, receipt/contract và resume.
- `FrozenDDIMAdapter`: load checkpoint/LPIPS từ captured bytes, DDIM20+30,
 50U-Net calls/mẫu; include_controls=True tạo cả4score maps trên cùng trajectory.
- `run_frozen_ddim`: nối runtime/source/import binding và adapter cho model chính.
  Caller phải cung cấp dataset factory đã kiểm revision/content. API này chưa
  đóng gói4arm outputs vào một lệnh final hoàn chỉnh.

## Trình tự trước một lượt fresh-model

1. Đọc candidate/control manifests và freeze decision. Hiện chỉ được validation
   acceptance với candidate; final cần hồ sơ frozen được review riêng.
2. Kiểm private source ZIP/LPIPS/checkpoint/config/index/partition/mask/lock theo
   đúng9pins. Dùng source archive d5889d5 đã nghiệm thu; manifests hiện hành nằm
   ngoài archive. Không tự dùng checkout HEAD hoặc đổi hash cho vừa runtime.
3. Trong subprocess sạch của runtime đã chọn, bảo đảm `dacn.__file__` nằm ở
   `<source_root>/src/dacn/__init__.py`; kiểm95source files và dependency inventory.
   Local Windows CPU hiện khác T4 pins nên bị từ chối là đúng. Offline preflight
   đã đạt không có nghĩa một runtime Colab mới tự đạt.
4. Dataset factory phải kiểm nguồn revision/content rồi mới trả samples đúng
   thứ tự, stable ID trong private runtime. Không ghi UUID, private masks/maps
   hoặc per-sample seed vào Git/Linear. Tập test tiếp tục kín khi chưa freeze.
5. Mỗi attempt dùng output riêng. Resume chỉ cùng manifest/scope, source/config,
   thresholds và sample order. Không dùng validation receipt làm final receipt;
   không có nhiều writer đồng thời trong cùng output directory.
6. Ghi actual start/terminal status vào RESULT_TRACKING, tải private artifacts
   trước khi runtime hết hạn, kiểm ZIP và member hashes ở local. Chỉ đẩy số tổng
   hợp an toàn. Không infer metrics từ kế hoạch/log progress.

## Kiểm tra local không cần GPU/test

Từ worktree, có thể kiểm phần orchestration và gate bằng:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_frozen_runner.py tests/test_frozen_inputs.py tests/test_runtime_binding.py tests/test_registered_partition.py -q --basetemp runs/pytest-gate-review-new
```

Dùng thư mục basetemp mới để giữ lịch sử. Tests dùng dữ liệu tổng hợp; không là
model-quality experiment. Actual real-validation acceptance08Nov dùng cached
T4 scores:4arms×43slices, ngắt1shard/resume42 và completed replay0backend calls;
mọi score shard khớp input,15numeric metric fields/arm khớp receipt tại1e-10.
[Evidence](evidence/2026-11-08-controls-cache-resume.json).

Full suite08Nov: sandbox273pass/1WinError5 khi rename staging ở BMAD CLI;
test đó chạy riêng đạt, ngoài sandbox toàn suite274pass38.57s. Chưa xác định
nguyên nhân; không thêm retry vào data publisher và không gọi đây là đã sửa lỗi.
CI account/billing vẫn là blocker riêng, không suy local pass thành CI success.

## API final — chỉ dùng sau freeze, chưa là CLI turnkey

Ví dụ cấu trúc caller, **không phải script sẵn chạy**:

```python
from dacn.evaluation.frozen_ddim import run_frozen_ddim

# manifest: hồ sơ đã frozen/review; artifact_paths:9private paths đúng pins.
# verified_samples: factory đã kiểm revision/content; không đọc dataset trước gate.
receipt = run_frozen_ddim(
    manifest=manifest,
    artifact_paths=artifact_paths,
    samples=verified_samples,
    output_dir=private_output,
    device="cuda",
    source_root=pinned_source_root,
)
```

Trước final còn cần launcher cho verified dataset factory và durable4arm output,
kiểm validation-only trên đúng runtime, rồi mới dùng protocol frozen cho test.
Không chạy4lần U-Net riêng chỉ để đổi scoring arm; cần giữ chung reconstruction.
Không nâng API model chính thành tuyên bố đã nghiệm thu four-arm final wrapper.
