# Blockers and team handoff

File này theo dõi các blocker chưa giải quyết cần thành viên khác hỗ trợ hoặc
cần quyết định chung. Linear vẫn là nguồn chính cho owner/status của task;
`RESULT_TRACKING.md` vẫn là nguồn chính cho provenance và metric của các run.

## Cách sử dụng

- Chỉ tạo blocker khi công việc không thể tiếp tục an toàn do thiếu quyền truy
  cập, artifact, prerequisite, quyết định của nhóm hoặc thay đổi từ hệ thống
  bên ngoài. TODO thông thường và experiment đang chạy đúng dự kiến không phải
  blocker.
- Dùng ID `BLK-YYYYMMDD-NN`. Cập nhật entry hiện có thay vì tạo trùng.
- Status hợp lệ: `Open`, `Investigating`, `Waiting`, `Resolved`.
- Khi handoff, ghi rõ người/role tiếp theo, hành động cụ thể và điều kiện đóng.
- Khi giải quyết xong, đổi status thành `Resolved`, thêm bằng chứng và ngày đóng;
  không xóa lịch sử.
- Không ghi token, credential, patient ID, đường dẫn nguồn bệnh nhân, ảnh y tế
  riêng tư hoặc seed suy ra cho từng sample.

## Bảng theo dõi

| ID | Blocker | Status | Owner / next handoff | Affects | Last updated |
|---|---|---|---|---|---|
| `BLK-20261101-01` | Primary DDPM run mới hoàn thành epoch 10/30 | Open | C — Hai Tuan | Checkpoint 20/30, kết luận convergence | 2026-11-01 |
| `BLK-20261101-02` | Pixel anomaly map có nhiều false positive | Investigating | B — Chí Vĩ; C — Hai Tuan | Score/model freeze trước held-out test | 2026-11-02 |
| `BLK-20261102-03` | Modal client chưa deploy được T4 batch-8 pilot | Resolved | C — Hai Tuan | T4 one-epoch throughput pilot | 2026-11-02 |

## Blocker details

### BLK-20261101-01 — Primary DDPM run chưa hoàn thành

- **Status:** Open
- **Reported / updated:** 2026-11-01 / 2026-11-01
- **Owner / next handoff:** C — Hai Tuan xác minh provenance và resume; người có
  quyền shared Modal Workspace hỗ trợ nếu access thay đổi.
- **Affected work:** Chưa có checkpoint epoch 20/30, chưa thể phân biệt rõ
  undertraining với hạn chế của reconstruction/score, và chưa đủ điều kiện chốt
  model cuối.
- **Evidence:** Run
  `ddpm-partial-a100-s17-e30-milestones-rngfix-20261031T041542Z` dừng trong
  epoch 11. Epoch 10 đã hoàn thành 75.000 optimizer steps; normal-validation
  epsilon-MSE tốt nhất là `0.0101424818`. Modal Volume giữ `ddpm_last.pt`,
  `ddpm_best.pt` và milestone epoch 5/10. Checkpoint epoch 10 cùng manifest,
  history và source snapshot đã được backup vào private Hugging Face model
  revision `07ec8b455a3318a267df5568f7af3fa4f15d7a3c`. Chi tiết provenance nằm
  trong [RESULT_TRACKING.md](RESULT_TRACKING.md).
- **Attempts / constraints:** Resume chỉ hợp lệ khi cùng run ID, config, trainer,
  dataset revision và source snapshot SHA256 đã lưu. Phần epoch 11 chạy dở không
  có trong checkpoint và sẽ chạy lại từ đầu.
- **Action needed:** Xác minh `ddpm_last.pt`, manifest và exact source snapshot
  còn khớp; resume cùng run từ epoch 11; giữ milestone epoch 20/30; cập nhật
  `RESULT_TRACKING.md` từ artifact thực tế. Không dùng held-out test.
- **Close when:** Epoch 30 hoàn thành hoặc nhóm ghi quyết định có bằng chứng rằng
  run sẽ không tiếp tục; mọi checkpoint/status cuối đã được lưu và track.
- **Resolution:** Chưa có.

### BLK-20261101-02 — Pixel anomaly map có nhiều false positive

- **Status:** Investigating
- **Reported / updated:** 2026-11-01 / 2026-11-02
- **Owner / next handoff:** B — Chí Vĩ dẫn evaluation; C — Hai Tuan cung cấp
  checkpoint/reconstruction và cùng review inference contract.
- **Affected work:** Chưa nên freeze checkpoint/score/threshold để chạy held-out
  test hoặc đưa ra kết luận nghiên cứu cuối.
- **Evidence:** Trên 83 validation slices, candidate epoch 10, `t*=300`, local
  one-minus-SSIM đạt pixel AUROC `0.892010` nhưng Precision `0.101571`, Recall
  `0.395303`, micro F1/Dice `0.161616` và pixel AP `0.095895`. Đây là một
  inference seed; test chưa được dùng. Heatmap định tính cho thấy score cao ở
  nhiều mô bình thường, phù hợp với giả thuyết false positive nhưng không tự nó
  chứng minh bug. Audit CPU ngày 02/11 xác nhận checksum/config/EMA, crop,
  one-based timestep, seed và score/threshold path khớp contract; 18 test liên
  quan đều pass. Trên một validation lesion slice được chọn theo lesion lớn nhất,
  2.982/3.087 local-SSIM false positives nằm trong brain ROI; foreground mask
  chỉ tăng F1 của slice từ 0,3476 lên 0,3515. Trung bình ba reconstruction có
  stochastic pixel standard deviation trung bình 0,0406 nhưng chỉ cải thiện nhẹ
  AP trong brain ROI; posterior-mean reverse cho kết quả AP/AUROC trái chiều.
  Các số một-slice này chỉ dùng diagnosis, không phải validation result. Full
  83-slice T4 ablation sau đó xác nhận posterior mean giảm empty-normal FPR từ
  0,077585 xuống 0,054391 nhưng local-SSIM AP giảm từ 0,095895 xuống 0,081721,
  F1 giảm từ 0,161616 xuống 0,140292 và Recall giảm từ 0,395303 xuống 0,254533.
  Vì vậy posterior mean không thay thế ancestral sampler.
- **Attempts / constraints:** Đã so checkpoint epoch 5/10, timestep 100/200/300
  và pixel-difference/local-SSIM theo validation protocol. Noise-MSE thấp đo
  epsilon prediction trên normal validation, không đo segmentation quality.
- **Action needed:** Giữ ancestral sampler làm baseline hiện tại; không ưu tiên
  posterior mean hoặc foreground mask đơn lẻ. Bước nghiên cứu tiếp theo cần được
  review riêng là một run Simplex-noise có cùng data, kiến trúc và optimizer-step
  budget để kiểm tra giả thuyết noise distribution; không thể đổi checkpoint
  Gaussian hiện tại sang Simplex chỉ ở inference. Không tune bằng test. Epoch
  20/30 chỉ đánh giá sau khi checkpoint tồn tại.
- **Close when:** Nhóm có diagnosis dựa trên artifact, chọn/freeze protocol hoặc
  phê duyệt ablation mới, và ghi rõ giới hạn cùng số seed thực chạy.
- **Resolution:** Chưa có.

### BLK-20261102-03 — Modal client chưa deploy được T4 batch-8 pilot

- **Status:** Resolved
- **Reported / updated:** 2026-11-02 / 2026-11-02
- **Owner / next handoff:** C — Hai Tuan hoặc thành viên đã đăng nhập shared
  Modal Workspace retry lệnh documented khi kết nối function deployment ổn định.
- **Affected work:** Fresh Gaussian DDPM one-epoch pilot trên toàn bộ 7.500 ảnh,
  T4, batch size 8 chưa bắt đầu; chưa có runtime/loss/checkpoint từ cấu hình này.
- **Evidence:** Config `ddpm_partial_t4_batch8_epoch1.json` và remote entrypoint
  `train_remote_t4` đã validate; 77 test pass, Ruff/format/compile/diff check pass.
  Bốn lần submit trả `Could not connect to the Modal server` trước khi tạo app.
  `modal app list` rỗng và Volume không có run directory
  `ddpm-partial-t4-b8-e1-20261102T032740Z`, nên không có GPU training hay quota
  run nào đã bắt đầu. Modal status page lúc kiểm tra báo dịch vụ operational.
- **Attempts / constraints:** Đã thử local entrypoint dynamic T4, fixed T4 remote
  entrypoint, có/không `--detach`; profile và Volume read vẫn hoạt động. Không
  re-authenticate hoặc thay shared Secret vì chưa có bằng chứng credential hỏng.
- **Action needed:** Không còn; theo dõi run trong `RESULT_TRACKING` và chỉ báo
  runtime/loss/checkpoint từ durable artifacts sau khi hoàn tất.
- **Close when:** Run manifest được tạo và pilot bắt đầu, hoặc nhóm quyết định
  không chạy cấu hình này.
- **Resolution:** Đã retry fixed `train_remote_t4` ngoài sandbox với quyền mạng;
  Modal tạo app `ap-vIFbYSY0oC1V74Tj4DFJvV`, durable start manifest và source
  snapshot. Pilot bắt đầu ngày 2026-11-02; blocker kết nối được đóng.

## Mẫu blocker mới

```markdown
### BLK-YYYYMMDD-NN — Tiêu đề ngắn

- **Status:** Open | Investigating | Waiting | Resolved
- **Reported / updated:** YYYY-MM-DD / YYYY-MM-DD
- **Owner / next handoff:**
- **Affected work:**
- **Evidence:**
- **Attempts / constraints:**
- **Action needed:**
- **Close when:**
- **Resolution:** Chưa có hoặc bằng chứng đóng blocker và ngày đóng.
```

## 2026-11-05 — B handoff update

BLK-20261101-02 remains Investigating. B reproduced both scores on full83 Gaussian30/Simplex20 caches;34 numeric fields per model exactly match C JSON. Producer provenance corrected to00d199f+dirty source snapshot, evaluator84088f9; see RESULT_TRACKING and docs/evidence. This validates handoff, not anomaly improvement. New branch codex/bc-evaluation-colab integrates B evaluation/provenance code with C7134b10;37 CPU checks pass. Reusable legacy adapter, Torch/GPU integration check, Colab Gaussian20 and grouped comparison remain pending. Do not ask C to regenerate existing full83 caches.

2026-11-05 B update for BLK-20261101-02: the FG-v1 analysis and patient-group
holdout comparison are recorded in RESULT_TRACKING. The brightness control
still leads pixel AP. B will run the prepared Gaussian20 notebook for the
matched-budget comparison and report uncertainty before recommending a
configuration freeze. Status remains Investigating.

2026-11-05 review update for BLK-20261101-02: notebook lifecycle/interrupt and
verified-snapshot bugs have been fixed with regression coverage. Full Torch CPU
suite:164 passed; data-only environment:113 passed/10 skipped. Aggregate metric
consistency checks passed; these are software/report checks, not new model
results. B can run the separate smoke gate, then explicitly enable full83 on
Colab. Gaussian20 real inference, matched-budget comparison and paired group
bootstrap remain pending. Closure still requires research evidence/protocol
decision, not code tests. See B_REVIEW_2026-11-05.md. No Linear status changed.

### BLK-20261105-02 — Colab HF credential rejected before Gaussian20 inference

- **Status:** Waiting
- **Reported / updated:** 2026-11-05 / 2026-11-05
- **Owner / next handoff:** B(account holder) enters a valid HF read credential
  directly into masked Colab getpass; agent continues smoke/inference afterwards.
- **Affected work:** Gaussian20 full83 inference and budget-matched comparison.
- **Evidence:** T4 connected and pinned source7a0f024 installed. HF whoami
  returned401/invalid user token before downloading checkpoint or dataset.
  No model inference started. CPU local HF index verification and existing-cache
  bootstrap succeeded independently.
- **Attempts / constraints:** Source clone access failure bypassed with authorized
  SHA-verified source archive. Supplied read/fine-grained token creation guide.
  Do not send credentials through Git, Linear or chat.
- **Action needed:** Account holder reruns notebook cell2 and enters a valid
  read token with access to dacn-bmad-team's dataset/model repos in getpass.
- **Close when:** HF authentication and pinned checkpoint/validation checks pass.
  Closing this access blocker does not mean smoke/full evaluation is completed.
- **Resolution:** Pending. The separate anomaly-score blocker remains Investigating.

2026-11-05 update for BLK-20261101-02: paired group bootstrap completed999valid
of1000draws on43validation rows/25groups. Aggregate evidence in RESULT_TRACKING;
fusion delta-AP CI is positive conditional on the examined cohort, but differs
in training budget and does not change calibration selection. Uncertainty for
Gaussian20/Simplex20 still needs the pending Gaussian20 cache. No Linear status
changed and no model-quality blocker closed.
