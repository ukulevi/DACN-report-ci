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
| `BLK-20261107-02` | Colab mới chờ HF read authentication | Resolved | Người dùng / agent tiếp tục smoke | GPU smoke/full validation | 2026-11-07 |
| `BLK-20261107-01` | GitHub Actions bị chặn do tài khoản/billing | Waiting | Chủ repo kiểm Billing/Actions | CI head hiện tại | 2026-11-07 |
| `BLK-20261101-01` | Chưa chốt terminal status của run batch-1 cũ | Open | C — Hai Tuan | Provenance run lịch sử; G30 run khác đã có | 2026-11-06 |
| `BLK-20261101-02` | Xác nhận chất lượng score trước freeze | Investigating | B — Chí Vĩ; C — Hai Tuan | Score/model freeze trước held-out test | 2026-11-06 |
| `BLK-20261102-03` | Modal client chưa deploy được T4 batch-8 pilot | Resolved | C — Hai Tuan | T4 one-epoch throughput pilot | 2026-11-02 |
| `BLK-20261106-01` | Nghiệm thu resume/cache sau tích hợp | Investigating | Nhóm review freeze; CI repo owner | DAC-11/12/14/15/30 | 2026-11-08 |

## Blocker details

### BLK-20261107-02 — Private Colab Hugging Face read access

- **Status:** Resolved, 07/11/2026.
- **Owner/action:** Người dùng tạo token Read của tài khoản có quyền đọc hai repo
  DACN, nhập trực tiếp vào ô getpass trong notebook controls mới và Enter.
- **Evidence:** Colab cấp Tesla T4, source bootstrap thành công; acquisition cell
  đang chờ input. Chưa có GPU inference hoặc metrics từ notebook mới.
- **Attempts:** Chuẩn bị/upload notebook riêng, cài source, mở getpass; hướng dẫn
  tạo read token bằng tài liệu HF chính thức. Không lấy token từ chat/log.
- **Close when:** Checkpoint/validation downloads hoàn tất với checksum đúng;
  sau đó chạy smoke và đồng bộ cell full43holdout reporting trước full83.
- **Notebook:** Xem COLAB_CONTROLS_VALIDATION.md. Không công khai notebook/maps.
- **Resolution:** Người dùng nhập token trực tiếp; checkpoint, metadata, val index/
  shard và mask parameters đã tải, checksum assertions qua. Cell dừng sau đó ở
  subprocess export môi trường, đang chẩn đoán riêng; không còn thiếu HF read auth.

### BLK-20261107-01 — CI không khởi động job

- **Status:** Waiting; reported 07/11/2026, nguyên nhân cấp tài khoản xác nhận qua ảnh người dùng.
- **Owner / next handoff:** Nhóm cùng kiểm; repo admin hỗ trợ nếu nguyên nhân cần quyền quản trị.
- **Evidence:** ecad491 và b986080 có Actions conclusion startup_failure.
  [Run ecad491](https://github.com/tanh1c/DACN/actions/runs/42240463952).
  Jobs API rỗng; check-runs không cung cấp output.
- **Affected work:** Chưa có CI evidence cho head, không dùng run xanh cũ để nghiệm thu.
- **Attempts:** Đọc live run/job/check-suite API. Local full suite 249 pass/1 fail
  WinError 5 rename staging; Ruff pass. Hai hiện tượng chưa chứng minh cùng nguyên nhân.
- **Follow-up 07/11:** Chạy riêng rename test đạt; chạy lại toàn suite 250 passed
  trong 38.88s. Không sửa code rename vì chưa tái hiện nguyên nhân. CI startup
  vẫn cần thông báo phía Actions; local green không đóng blocker này.
- **Evidence mới:** Ảnh Summary run a6ccf41 do người dùng cung cấp hiển thị:
  "The job was not started because recent account payments have failed or …".
  Phần tiếp theo bị thu gọn, chưa có đầy đủ thông báo. Đây là lỗi chặn tài khoản
  trước khi khởi động job, không phải failure của test. API xác nhận check-runs=0.
- **Action needed:** Chủ repo mở Show more và Billing/Actions để xử lý thông báo
  payment/account hoặc hạn mức được nêu đầy đủ. Không đổi phương thức thanh toán,
  tăng budget hay retry hàng loạt từ agent. Sau khi xử lý, chạy lại CI trên head.
- **08Nov live check:** Commit edff8c8 vẫn startup_failure tại
  [run42451751296](https://github.com/tanh1c/DACN/actions/runs/42451751296),
  jobs=0/check-runs=0 qua API. Không có test job bắt đầu; giữ Waiting,
  không suy rằng lỗi mới nằm trong code và không retry tự động.
- **Close when:** Nguyên nhân được ghi và một run trên source liên quan thực sự
  khởi động/kết thúc với bằng chứng; trạng thái test failure nếu có được tách riêng.

### BLK-20261106-01 — Tích hợp và kiểm chứng branch C mới trước freeze

- **Status:** Investigating
- **08Nov evidence:** T4 smoke2/full83 hoàn tất, ZIP tải và10file checksum đạt;
  toàn bộ43holdout metric replay khớp1e-10,4scoremaps khớp cache C tại1e-6.
  Bounded trained CPU resume đã đạt trước đó. Xem COLAB_ACCEPTANCE_2026-11-08.md.
  Sửa nhãn provenance CPU kế thừa cho generator mới,24related CPU checks đạt.
  Chưa tự đóng gate: runtime cuối cần repin/review freeze và CI head còn bị
  chặn tài khoản ở BLK-20261107-01; chưa chạy remote Modal continuation.
- **08Nov repin:** Candidate đã pin9artifacts từ T4 receipt; static archive
  preflight95files đạt, final candidate rejection và CPU-runtime mismatch gate
  đều hoạt động. Thresholds không đổi. Còn review freeze/live final runtime,
  không còn thiếu checksum/export LPIPS hoặc artifact tải về.
- **Reported / updated:** 2026-11-06
- **Owner / next handoff:** B tích hợp/evaluator; C cache/continuation; A review freeze.
- **Affected work:** DAC-11/12/14/15/30 và final reproducibility.
- **Evidence:** B `a556a85` và C `c02addb` chưa tích hợp; CI chỉ push main/PR.
  C d282f02 có 8 collection errors trong data-only env; c02addb có 181 tests pass,
  1 float32 equality failure khi có Torch. Prefix cache và hash-before-reopen/parent-byte guards còn thiếu.
  B full suite có Windows rename lỗi lặp ở hai test khác nhau, cả ngoài sandbox.
- **Attempts:** fetch/isolated source review, CPU suites, minimal dependency
  collection, HF JSON/LFS checksum audit; không GPU hay mở test.
- **Progress 06/11:** C `c02addb` đã tích hợp vào B; conflict hai log giữ cả
  hai phía. Đã sửa test optional Torch và sai khác rounding; full suite 237 pass,
  data/dev 121 pass / 22 skip với basetemp riêng ngoài sandbox. Branch push đã
  bật CI. Các guard cache/checkpoint và team acceptance vẫn cần review.
- **Progress tiếp:** feature-condition và condition-intervention giờ parse JSON/
  NPZ từ đúng bytes được hash; 2 regression tests tái hiện lỗi cũ và 14 tests liên
  quan pass. Hash của checkpoint trong condition-intervention vẫn được xác minh
  riêng trước khi caller tải checkpoint; chưa tuyên bố đường đó đã an toàn.
- **Progress 06/11:** prefix resume yêu cầu contract sidecar gắn checkpoint/index
  hash và mode inference; Modal continuation kiểm hash last/best trên bytes thực
  và chép đúng bytes đã xác minh vào run mới. 19 test liên quan đạt, Ruff đạt.
  Prefix lịch sử thiếu sidecar sẽ bị từ chối resume và cần tạo lại.
- **Progress toàn bộ nhóm:** B đã đối chiếu cả image/pixel AUROC/AP/P/R/F1,
  thresholds, macro lesion Dice và normal FPR từ 4 cache với C; tolerance 1e-10,
  tất cả khớp. Không cần bàn giao theo role để tiếp tục các việc A/C. Remaining
  acceptance tập trung source/cache guard và runner/freeze, không còn thiếu metric replay.
- **Review mới:** GitHub CI tại `9aa42c5` đã success trên môi trường data/dev.
  Còn thiếu donor/source identity của prefix, binding hash cache/sidecar,
  checkpoint load/hash cùng snapshot, và test trực tiếp orchestration continuation.
  Xem [review sau tích hợp](PROGRESS_REVIEW_2026-11-06.md). Không coi 19 test đã chạy
  là kiểm chứng trực tiếp nhánh Modal continuation mới.
- **Tiến độ sau review:** helper continuation dùng trong Modal có 5 test local
  (missing file/hash, byte mismatch, parent mutation, interrupted write/restart).
  Prefix gắn checksum NPZ + source digest + donor index; evaluator load checkpoint
  từ cùng captured bytes được hash. 24 test liên quan đạt. Chưa chạy remote Modal;
  helper verification không tương đương kiểm chứng toàn bộ orchestration.
- **07/11 adapter follow-up:** Frozen DDIM adapter và 11 tests đạt; LPIPS83 replay
  từ runtime CPU mới khớp cache. Full suite 254 pass/1 WinError5 rename ở RGBA
  preparation test. Lỗi filesystem vẫn tái diễn tại test khác nhau; chưa có bằng
  chứng thuộc inference code, cần chẩn đoán handle/locking trước khi thêm retry.
- **Chuẩn bị runner cuối:** frozen-input gate có test CPU cho candidate, thiếu
  pin, revision mutable, threshold NaN và artifact bị thay. Candidate JSON giữ
  source/LPIPS/dependency pins chưa xác minh ở null. Gate chưa nối inference;
  cần lấy weights hashes từ môi trường thực tế, không coi candidate là freeze.
- **Action needed:** Review tích hợp và xác nhận manifest/schema trên artifact
  thực tế, pin dependency/weights cho runner cuối. Donor index/source digest không
  thay dataset revision/hash shard: nếu dùng normal-donor trong shortlist phải
  pin donor dataset riêng. Replay metric và bootstrap đã hoàn thành.
- **Close when:** source chung có verification xanh, private-cache reproduction
  có receipt, B/C ký acceptance; C xác nhận manifest checkpoint trên Volume dùng
  đúng tên artifact/schema; không đòi model phải đạt một AP tùy ý.
- **Resolution:** Chưa có; xem PROJECT_AUDIT_2026-11-06.md.

### BLK-20261101-01 — Terminal status của primary run batch-1 cũ

- **Status:** Open
- **Reported / updated:** 2026-11-01 / 2026-11-06
- **Owner / next handoff:** C — Hai Tuan xác minh provenance và chốt trạng thái
  run cũ; người có quyền shared Modal Workspace hỗ trợ nếu cần.
- **Affected work:** Lifecycle/provenance lịch sử chưa khép kín. Gaussian30 đã
  hoàn tất ở run khác; blocker này không ngăn B evaluation checkpoint mới.
- **Current evidence (06/11):** HF snapshot af0bf400 vẫn ghi run batch-1 cũ
  running và thiếu 3 checkpoint được manifest tham chiếu; không xác minh live
  Modal C. Xem [audit](PROJECT_AUDIT_2026-11-06.md).
- **Historical evidence (01/11):** Run
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
- **Action needed:** C xác minh trạng thái cuối trên Modal, ghi completed /
  interrupted / abandoned theo bằng chứng và lý do; cập nhật manifest và
  `RESULT_TRACKING.md`. Không tự resume run cũ chỉ để đóng blocker.
- **Close when:** Epoch 30 hoàn thành hoặc nhóm ghi quyết định có bằng chứng rằng
  run sẽ không tiếp tục; mọi checkpoint/status cuối đã được lưu và track.
- **Resolution:** Chưa có.

### BLK-20261101-02 — Pixel anomaly map có nhiều false positive

- **Status:** Investigating
- **Reported / updated:** 2026-11-01 / 2026-11-06
- **Owner / next handoff:** B — Chí Vĩ dẫn evaluation; C — Hai Tuan cung cấp
  checkpoint/reconstruction và cùng review inference contract.
- **Affected work:** Chưa nên freeze checkpoint/score/threshold để chạy held-out
  test hoặc đưa ra kết luận nghiên cứu cuối.
- **Current evidence (06/11):** LPIPS/DDIM20+30 có AP .312330 / Dice .371215,
  naive-stitch LPIPS AP .180593 trên 43 internal holdout slices. Giữ Investigating
  cho freeze. B đã replay score maps và paired bootstrap 999/1000 draws thành công;
  CI delta AP final minus naive stitch [.0262, .1779], final minus brightness
  [-.2874, .2329], conditional on one selected run. Xem RESULT_TRACKING. Không yêu cầu chạy lại đối
  chứng đã có hoặc tăng epoch. Xem [audit](PROJECT_AUDIT_2026-11-06.md), HF fae9b50.
- **Historical evidence:** Trên 83 validation slices, candidate epoch 10, `t*=300`, local
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
- **Action needed:** B/C review replay và contract source chung; A/B/C freeze
  shortlist và quy tắc test, cân nhắc repeat sampling seed theo budget; không tune bằng test.
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

- **Status:** Resolved
- **Reported / updated:** 2026-11-05 / 2026-11-06
- **Owner / next handoff:** B; access action completed.
- **Affected work:** Gaussian20 full83 inference and budget-matched comparison.
- **Evidence:** T4 connected and pinned source7a0f024 installed. HF whoami
  returned401/invalid user token before downloading checkpoint or dataset.
  No model inference started. CPU local HF index verification and existing-cache
  bootstrap succeeded independently.
- **Attempts / constraints:** Source clone access failure bypassed with authorized
  SHA-verified source archive. Supplied read/fine-grained token creation guide.
  Do not send credentials through Git, Linear or chat.
- **Action needed:** None for this access blocker.
- **Close when:** HF authentication and pinned checkpoint/validation checks pass.
  Closing this access blocker does not mean smoke/full evaluation is completed.
- **Resolution:** User re-entered credential; whoami and pinned dataset/checkpoint
  checks passed. Smoke4/full83 completed2026-11-05; downloaded private artifact
  and checkpoint hashes verified. Recorded2026-11-06 with manifest evidence in
  RESULT_TRACKING. The separate anomaly-score blocker remains Investigating.

2026-11-05 update for BLK-20261101-02: paired group bootstrap completed999valid
of1000draws on43validation rows/25groups. Aggregate evidence in RESULT_TRACKING;
fusion delta-AP CI is positive conditional on the examined cohort, but differs
in training budget and does not change calibration selection. Uncertainty for
Gaussian20/Simplex20 still needs the pending Gaussian20 cache. No Linear status
changed and no model-quality blocker closed.

2026-11-06 update for BLK-20261101-02: Gaussian20/Simplex20 same-budget CPU
comparison and paired patient-group CI completed. AP .099777/.087416,deltaCI
[-.030666,.011189]; no stable noise-family advantage demonstrated. C feature A/B
HF review added:small AP gains, no demonstrated pretrained/self-conditioning
benefit, image classification/FP remain weak. Keep Investigating; A/B/C need to
review the aggregate evidence and choose the next bounded scoring/control
experiment before final freeze. No new training requested or started.
