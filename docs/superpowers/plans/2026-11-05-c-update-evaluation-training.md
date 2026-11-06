# Kế hoạch tiếp nhận cập nhật C, đánh giá và hỗ trợ train qua Colab

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Các checkbox chưa đánh dấu là công việc đề xuất, không phải kết quả đã chạy.

**Goal:** Tận dụng checkpoint/cache C đã tạo, tái lập evaluation độc lập, rồi chọn một thí nghiệm train có đối chứng nếu bằng chứng và tài nguyên cho phép.

**Architecture:** CPU kiểm provenance, cache và metric; Colab của B chạy inference/checkpoint có sẵn. C giữ training chính và recovery; B hỗ trợ một nhánh train độc lập sau khi tích hợp nguồn và đăng ký protocol.

**Tech Stack:** Python, NumPy, scikit-learn, PyTorch, Hugging Face private artifacts, Colab GPU; Modal là lựa chọn của C, không là điều kiện để B làm việc.

**Spec:** PROJECT_REQUIREMENTS.md, RESEARCH.md và phạm vi/điểm quyết định trong tài liệu này. Ngày kiểm tra: 05/11/2026, Asia/Saigon. User xác nhận có Colab GPU cá nhân; chưa khởi chạy GPU trong lần kiểm tra này.

## 1. Nguồn đã kiểm tra và kết quả mới

- Git C: `7134b105c0c4d4fb620715908f100103b1d0b0ab`, nhánh `feat/anoddpm-notebook-training`; thêm/sửa 46 file so với `00d199f`.
- HF model: `dacn-bmad-team/dacn-runs` revision `1788b1232aa19b3c9ae96aa95a74e2975f4b592a`.
- Dataset vẫn là `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`: 7.500 train normal, 83 validation, test giữ kín.
- 75 đối chiếu manifest: 26 file JSON kiểm từ bytes tải về, 49 đối chiếu SHA với metadata LFS. Tất cả khớp. Chưa tải/deserialise các checkpoint mới.
- Sau đó tải riêng hai cache Gaussian30/Simplex20, kiểm SHA từ bytes, xác nhận ảnh/mask/nhãn khớp chính xác dataset, đủ 39 normal/44 abnormal; reconstruction hữu hạn và thuộc [-1,1]. Không in UUID hoặc ảnh bệnh nhân.
- Evidence: `docs/evidence/2026-11-05-hf-manifest-verification.json` và `docs/evidence/2026-11-05-cache-alignment.json`. Đây là kiểm chứng artifact, chưa phải tái tính metric độc lập.

Số dưới đây đọc từ JSON kết quả C trên HF: cùng 83 validation, crop 240, t300, sampling seed17, local SSIM, threshold max-F1 trên toàn validation. Là descriptive validation, không phải test hoặc grouped holdout.

| Model | Optimizer steps | Pixel AP | Pixel AUROC | Micro Dice | Normal pixel FPR |
|---|---:|---:|---:|---:|---:|
| Gaussian batch8 epoch30 | 28.140 | 0,100741 | 0,895647 | 0,167483 | 0,071126 |
| Simplex batch8 epoch10 | 9.380 | 0,073910 | — | 0,165806 | 0,133712 |
| Simplex batch8 epoch20 | 18.760 | 0,074791 | 0,743631 | 0,166414 | 0,131908 |

Gaussian30 là run batch8 mới tiếp nối pilot epoch1, KHÔNG phải resume run batch1 epoch10 cũ có 75.000 updates. Simplex20 đã recover sau preemption và hoàn thành attempt2; history ghi đủ 20 epochs. Cả hai mới có train seed17.

Simplex noise-val MSE giảm 0,00297178 -> 0,00254602 (14,33%), nhưng AP tăng khoảng 1,19%. Không so trực tiếp raw noise-MSE giữa Gaussian và Simplex để kết luận model tốt hơn: target noise khác phân phối/scale. Simplex hiện dùng Simplex forward, Gaussian fixed-large reverse, là adaptation đã mô tả chứ không phải mọi đường inference của upstream AnoDDPM.

Các hướng C đã thử, không nên làm lại nguyên xi:

- Foreground gating chỉ cải thiện nhẹ; ở Gaussian30 SSIM, 83,47% FP nằm trong foreground interior theo heuristic của C. Đây không phải bằng chứng lỗi chỉ ở nền đen. Heuristic `input > -0.95` của C khác FG-v1 `input > -1` + largest-component/fill-holes của B.
- One-step t300 có all83 SSIM AP 0,105883 so với ancestral 0,100741, nhưng cùng grouped holdout43 đạt 0,098530 so với 0,102402. Có lợi về tốc độ, chưa có lợi AP ổn định.
- DDIM10, giảm t về100/200, patch-context hiện tại, denoising score và AutoDDPM-style pilot chưa tạo lợi ích AP đáng tin cậy. Không mở rộng đồng thời các nhánh này.
- Brightness control trên cùng holdout43 AP0,389506, cao hơn reconstruction SSIM0,102402. Fusion chọn bằng calibration phải giữ nguyên; không chọn lại tỷ trọng theo bảng holdout.
- Image F1 Gaussian30/Simplex20 =0,692913 là all-positive, không phải classifier tốt. Phải báo confusion/specificity cùng AUROC.

## 2. Quyết định ưu tiên

### Review thứ tự thực hiện — 05/11, sau yêu cầu dùng branch riêng

Phần này thay thế thứ tự phụ thuộc cứng Task1 -> Task2 -> Task3 ở bản đầu.
CPU reproduction chạy ngay trên source B đã pin và hai cache đã xác minh;
không cần chờ merge cả hai nhánh, adapter tổng quát hay PR được duyệt.
Branch tích hợp mới phục vụ code dùng chung và GPU runner sau đó.

| Ưu tiên / ID local | Việc và owner đề xuất | Phụ thuộc | Điều kiện hoàn thành |
|---|---|---|---|
| P0 / NEXT-01 | B: tái tính hai score Gaussian30/Simplex20 từ cache, kiểm image confusion | Cache/data đã QC, hiện có | Báo cáo metric/threshold đối chiếu HF, sai lệch được giải thích, run ghi RESULT_TRACKING |
| P0 / NEXT-02 | B thực hiện, C review: tạo branch `codex/bc-evaluation-colab` từ C7134b10 và tích hợp phần B cần thiết | Bảo toàn thay đổi local trước khi tạo worktree mới | Branch có evaluator CPU, provenance guards, sampler C và notebook; kiểm tra cần thiết pass; push branch để nhóm dùng |
| P0 / NEXT-03 | C: kiểm checksum checkpoint thực trước load/copy; A/C quyết định trạng thái run batch1 cũ | Source/manifest/checkpoint liên quan | Fail nếu bytes sai; run cũ có quyết định interrupted/superseded rõ ràng, không ghi completed |
| P1 / NEXT-04 | B: Colab Gaussian20/t300/seed17/full83, C review config | NEXT-01 baseline khớp, NEXT-02 runner sẵn sàng, checkpoint hash pass | Smoke thành công, đủ83 cache+manifest durable; tối đa45 phút GPU cho full run |
| P1 / NEXT-05 | B: so cùng budget Gaussian20/Simplex20, brightness control; A/C review | NEXT-04 | Bảng cùng cohort/protocol/budget; AP chính, FPR/confusion đi kèm; quyết định nghiên cứu được ghi |
| P1 / NEXT-06 | B: uncertainty/paired bootstrap theo patient group | Cache/threshold đã cố định | CI có số replicate/seed/missing-class rõ ràng; không dùng CI để đổi chọn trên holdout |
| P2 / NEXT-07 | B/C: pilot loss weighted và uniform control | NEXT-05, code loss/recovery được kiểm chứng và đủ compute | Cùng parent/budget; pilot1epoch/arm; review AP/FPR trước khi tăng đến5epoch/arm |

NEXT-* là ID kế hoạch local, chưa phải issue Linear. DAC-12/DAC-30 là mapping
dự kiến cho evaluation/score; DAC-11 cho checkpoint lifecycle; DAC-14 cho
trainer/reconstruction. Cần đọc acceptance criteria trên DACN-KHMT trước khi đổi
status hoặc kết luận Done. Connector hiện trả workspace traffic-agent-assistant;
không suy ra trạng thái DACN-KHMT từ connector đó.

**Hai ngày làm việc đầu:** B làm NEXT-01 ngay và NEXT-02 trong worktree riêng;
C xử lý NEXT-03, kiểm source/config và cache interface. A chốt nguồn chuẩn,
quy tắc handoff và terminal decision run cũ. Sau đó B chạy NEXT-04 khi runner
sẵn sàng; NEXT-06 không chặn lượt inference ngắn này. Các mốc ngày ở Task1–5
là mục tiêu, không phải công việc đã bắt đầu hoặc cam kết GPU luôn sẵn.

**Cách dùng branch:** checkpoint riêng các thay đổi local hiện có trước; giữ
worktree cũ nguyên trạng; branch mới lấy C7134b10 làm base, đưa code B cần thiết
vào bằng diff/cherry-pick có review. Không bê toàn bộ doc lịch sử rồi ghi đè
log C. Nếu chỉ cần reproduction CPU thì dùng snapshot hiện tại ngay. PR20 có
thể giữ để tham chiếu; không đặt merge PR20 làm prerequisite cho NEXT-01/04.
Việc vào main vẫn cần quy trình review của nhóm, nhưng công việc tiếp diễn trên
branch mới mà không chờ bước đó.

**Điểm review code cần xử lý:** cache C không có UUID; adapter B phải ràng buộc
identity bằng dataset đã checksum. `noise_fusion.py` đang pin Simplex10 SHA và
method name; không đổi mỗi tên file sang Simplex20 rồi chạy. Cần manifest/config
experiment riêng và giữ nguyên regression của kết quả Simplex10. Continuation
ghi hash parent trong manifest chưa đủ để chứng minh bytes được kiểm trước load.

**Điều kiện train:** đề xuất foreground-weighted là hypothesis. Nếu AP thấp do
normal-structure error thì có lý do thử, nhưng không tự suy ra train sẽ sửa được.
Pilot1epoch chủ yếu kiểm ổn định, memory và recovery; một pilot không đủ kết luận
hiệu quả. Chọn checkpoint cuối epoch25 mỗi arm cho so sánh5epoch; không bỏ
control hoặc thay threshold/primary metric sau khi nhìn holdout.

**Tạm hoãn:** Simplex30, tăng base128, mở grid timestep/sampler/fusion mới, và
chạy test trước freeze. Giữ các thử nghiệm đã có làm bằng chứng nghiên cứu,
không tái chạy chỉ để tìm một số đẹp hơn.

Ba phương án: (a) train Simplex lên30 ngay để bằng số epoch Gaussian30; (b) đánh giá Gaussian20 đã có để bằng budget Simplex20; (c) mở kiến trúc/noise/loss mới ngay. Chọn **(b) sau kiểm chứng CPU**, vì cần một lượt inference thay vì khoảng 10 epoch train. (a) chỉ dùng khi cần đường convergence hoặc kết quả mới hỗ trợ; (c) giữ đúng một loss ablation có đối chứng, không thử hàng loạt.

Không hứa tăng metric. Kết quả âm có kiểm soát vẫn là đầu ra nghiên cứu hợp lệ.

## Global constraints

- Train normal-only; labelled validation dùng selection; test không dùng trong các task dưới đây.
- Pin Git/HF/config/checkpoint SHA; lưu nguồn thực thi nếu khác source tạo checkpoint. Không gọi nguồn mới là exact resume của nguồn cũ.
- Metric chính global micro pixel AP trên crop240; báo AUROC, Precision, Recall, Dice, normal pixel FPR và image confusion. Không đổi metric chính vì Dice của nhánh khác đẹp hơn.
- Tách all83 threshold-in-sample với calibration40/holdout43; validation đã được xem nhiều lần, internal holdout vẫn exploratory.
- Patient UUID, arrays và seeds từng mẫu chỉ ở private/ignored artifacts. Chỉ aggregate vào Git/Linear.
- Mỗi job mới có run ID riêng, budget/time cap trước submit, checkpoint durable và manifest started/completed/failed. Không ghi planned run là result.

## Review focus

1. Cache legacy không có UUID: chỉ gắn ID sau khi toàn bộ input/mask/label khớp dataset, tuyệt đối không chỉ dựa vào số83 hay label order.
2. Hai nhánh sửa cùng runner: giữ cả cache legacy lẫn UUID handoff; không làm mất sampler/continuation của C.
3. Hash trong manifest không thay thế việc kiểm bytes trước load: parent best/last cần checksum thực trước copy/deserialise.
4. Colab ngắt runtime: checkpoint local không đủ; phải có recovery artifact ngoài VM và không tự khởi động trùng run.
5. Chọn seed/threshold theo holdout hoặc đảo score sau khi nhìn AUROC sẽ làm sai đánh giá; các quy tắc phải chốt trước.

## Task 1 — Tích hợp B/C và tiếp nhận cache (B chủ trì, C review; 05–06/11)

Files: `src/dacn/cloud/modal_evaluate.py`, `src/dacn/evaluation/ddpm_validation.py`, `handoff.py`, `artifacts.py`; tests `test_handoff_guards.py`, `test_validation_artifacts.py`, `test_ddpm_validation.py`; docs ISSUES/RESEARCH/RESULT_TRACKING/WORKLOG.

- [ ] Tạo branch `codex/bc-evaluation-colab` từ C7134b10 trong worktree riêng; tích hợp phần B cần thiết, giữ thay đổi local và lịch sử thực nghiệm. Không chờ PR20 merge để làm CPU reproduction.
- [ ] Thêm adapter `convert_legacy_cache(cache, dataset_dir, expected_cache_sha256, expected_metadata_sha256, output)` trong `evaluation/handoff.py`. Đọc/verify cùng bytes, đối chiếu input/mask/labels với index+shards đã checksum, xuất `sample_ids` + reconstruction[N,1,256,256] và receipt liên kết hash nguồn/đích. Không sửa cache gốc; từ chối output tồn tại, mismatch hoặc mapping mơ hồ.
- [ ] Regression trước implementation: đảo thứ tự hai ảnh có cùng label phải reject nếu không khớp index; hash sai/mất mẫu/NaN/overwrite reject; cache hợp lệ roundtrip giữ tensor và metric.
- [ ] Future producer lưu UUID và dataset metadata SHA ở cùng lần export, giữ lựa chọn cache đầy đủ cho script C.
- [ ] Chạy targeted tests trên CPU và Torch tests trên Colab; giữ test bị skip trong báo cáo. CI pass và C review trước merge.

Đầu ra: hai cache đã nhận trở thành artifact hợp lệ của evaluator B; không cần rerun GPU. Việc đổi format không tạo metric mới.

## Task 2 — Báo cáo evaluation độc lập (B; 06–07/11, CPU)

Files: existing `evaluation/artifacts.py`, `diagnostics.py`, `noise_fusion.py`; proposed `scripts/evaluate_cached_comparison.py`, `tests/test_cached_comparison.py`; safe aggregate docs/evidence và RESULT_TRACKING.

- [ ] Tái tính Gaussian30 và Simplex20 cùng all83 bằng evaluator B; so mỗi metric/threshold với HF JSON với tolerance1e-6, điều tra sai lệch trước nghiên cứu mới.
- [ ] Kiểm riêng image confusion, top576 nằm ở đâu, FP/FN theo vùng/lesion size. Không đảo chiều score tự động vì AUROC<0,5.
- [ ] Dùng group split40/43 hiện có và cùng controls, calibration-only thresholds để so Simplex20 với Simplex10/Gaussian30. Refactor checksum hard-code của noise_fusion nếu cần, giữ regression của experiment cũ; chưa mở grid weight mới.
- [ ] Proposed uncertainty report: paired patient-group bootstrap trên holdout25groups, 1000 replicates, seed17, cùng group draw cho mọi candidate; giữ threshold/normalizer đã fit, báo percentile95% CI của delta AP/Dice, số replicate thiếu lớp bị loại. Test group integrity/shared draws/fixed thresholds. CI này không sửa được selection bias từ validation đã xem và không thay train seeds.
- [ ] Ghi actual runtime/provenance/metrics vào RESULT_TRACKING sau khi thật sự chạy; lập bảng giữ nguyên cùng cohort, không ghép AP full83 với AP holdout43.

Đầu ra: báo cáo B tái lập được số liệu, đánh giá bất định, và đề xuất giữ/bỏ từng hướng dựa trên evidence.

## Task 3 — Colab inference cùng budget (B chạy; C kiểm checkpoint; 07–08/11)

Files: update `notebooks/ddpm_validation_colab.ipynb`, proposed `scripts/run_ddpm_colab.py`, tests cho manifest/config/checksum. Notebook B cũ pin9566683/epoch10 nên không dùng nguyên trạng cho nghiên cứu mới.

- [ ] Pin source đã tích hợp, model1788b123..., dataset350c74...; private login bằng Colab Secrets/getpass, không in token. Download val-only cho inference; không đọc test.
- [ ] First run: 4 validation samples/t3 chỉ kiểm plumbing, không metric nghiên cứu. Kiểm SHA checkpoint, CUDA, disk/RAM và cache upload/download trước full83.
- [ ] Full83: **Gaussian batch8 epoch20**, checkpoint `runs/ddpm-partial-t4-b8-e30-cont-20261102T034644Z/ddpm_epoch_020.pt`, SHA `b4fc177db493fce2143a6412161068c7af73a25e8ae7ef9d88c117863c7e37f1`; t300, EMA, ancestral batch1, seed17, hai score cũ. So Simplex20, cùng18.760 optimizer steps/150.000 image presentations. Không phải đối chứng Gaussian epoch10 batch1.
- [ ] Cap đề xuất: một job full83, tối đa45 phút GPU; lịch sử C khoảng12–13 phút trên T4 cho inference+scoring, không bảo đảm Colab. Nếu smoke không đủ tài nguyên thì dừng và ghi blocker, không mua compute tự động.
- [ ] So cùng budget vẫn phải kiểm config/initialization/noise-path/source; một train seed không đủ kết luận nhân quả tổng quát về Gaussian vs Simplex.

Đầu ra: cache83 và manifest riêng; B chạy đánh giá CPU. Không train mới để có bảng cùng budget.

## Task 4 — Một thí nghiệm train hỗ trợ C, có điều kiện (B/C; 09–13/11)

Chỉ làm sau Task1–3 nếu nhóm muốn thử cải tiến objective. Đây là đề xuất, chưa implement/chạy. Đóng băng Simplex20 hiện tại; không auto Simplex30 hay tăng base128.

Phương án ưu tiên: **Gaussian foreground-weighted epsilon loss với control tiếp tục loss cũ**. Lý do: normal-foreground distortion còn lớn; noise objective toàn ảnh có nhiều pixel nền. Đây là giả thuyết, không phải bug đã chứng minh hoặc cải tiến đã đo.

- C và B dùng cùng parent Gaussian20, khởi tạo cùng model/EMA/optimizer/RNG/sampler; fresh experiment IDs, ghi parent SHA. Control uniform và candidate weighted cùng5epochs bổ sung (4.690 updates,37.500 image presentations), batch8,lr1e-4,EMA0.9992,T1000,base32. Không đổi noise/sampler/timestep cùng lúc.
- Candidate duy nhất đề xuất: foreground=input>-0.95 từ ảnh train normal; weight1 foreground/0.1 background, loss mỗi ảnh = sum(w*epsilon_error²)/sum(w), rồi mean batch. Không tumor-mask training, không morphology search. Cả hai ghi thêm unweighted/full/foreground/background loss để so sánh.
- Chọn checkpoint cuối cùng của mỗi arm cho comparison đã đăng ký; giữ best/last để recovery nhưng không chọn arm bằng noise-MSE khác objective. Eval cùng t300/seed17/protocol sau pilot.
- Pilot1epoch/arm trước; checkpoint durable mỗi epoch. Cap10epoch-GPU tổng cho hai arm; dự trù khoảng70–90 phút training từ lịch sử C, cộng scoring/upload, Colab có thể khác. Nếu chỉ đủ một GPU thì chạy tuần tự, không bỏ control.
- Không mở rộng khi chỉ noise-MSE giảm mà AP/normal-FPR không có trade-off đáng chấp nhận. Dừng sau pilot/bounded comparison nếu tín hiệu không rõ; không điều chỉnh tiếp trên holdout.

Files dự kiến: `src/dacn/models/diffusion/process.py`, `training/ddpm_partial.py`, Colab runner, hai config riêng, tests loss/resume. Trước code: test weight1 tái lập loss cũ, mask chỉ từ input, all-background finite, normalized loss theo từng sample, resume mismatch bị reject, checkpoint/optimizer/RNG restore. Đây là fine-tune/branch experiment, không bypass exact-resume guard.

Prerequisite kỹ thuật: `modal_train.py` của C ghi parent best/last hashes trong continuation metadata nhưng đoạn copy/load chưa đối chiếu bytes với các hash đó. Bổ sung kiểm hash thực và source compatibility trước load; test sửa1byte phải fail trước trainer. Chưa có bằng chứng checkpoint thực tế đã sai. Không áp exact whole-tree hash equality cho một experiment cố ý đổi loss; phải ghi rõ parent/new source và phần thay đổi.

Colab runtime mất là dự kiến: lưu full-state checkpoint + config + source snapshot + history + manifest vào private HF/Drive mỗi epoch, upload path mới; không publish arrays. Khi ngắt, dùng checkpoint durable gần nhất và ghi lost partial epoch, không ghi toàn run là completed.

## Task 5 — Chốt phạm vi và báo cáo (A điều phối; B/C; 14/11–04/12)

- 14–20/11: chọn một candidate diffusion có căn cứ; nếu có tài nguyên, thêm train seeds độc lập cho baseline/candidate thay vì mở thêm thuật toán. Sampling repeats phải ghi riêng, không gọi là independent model seeds.
- 21–24/11: freeze data/source/checkpoint selection/sampler/score/threshold/preprocessing. Chốt DAE comparison nếu artifact/config đủ tương thích; brightness là sanity control, không thay chủ đề diffusion.
- 25–28/11: một đợt held-out test sau freeze; report CI và kết quả âm, không tune lại theo test.
- 29/11–04/12: báo cáo, demo, reproducibility appendix; điều chỉnh nếu deadline nhóm thay đổi.

## Blocker handoff

- BLK-20261101-01: run batch1 cũ vẫn chưa có terminal decision. Gaussian batch8 epoch30 đã hoàn thành ở run khác; A/C cần ghi run cũ stopped/superseded có lý do, không sửa thành completed.
- BLK-20261101-02: giữ Investigating; thiếu cache đã được giải quyết cho Gaussian30/Simplex20, nhưng false positives/image all-positive chưa được giải quyết. Handoff hiện tại là alignment/replication/controlled comparison, không yêu cầu C export lại83.
- BLK-20261101-03: Modal cá nhân vẫn chưa xác minh thanh toán; Colab là đường thay thế, không tự đánh Resolved.
- BLK-20261105-01: integration PR20 và cache schema cần B/C xử lý trước notebook chung/research run. Không đóng issue nghiên cứu chỉ vì code/CI pass.

## Thực hiện trong lượt kiểm tra này

Đã fetch Git, đọc source/docs mới, đọc HF manifest/config/results, đối chiếu checksum và kiểm hai cache với validation dataset. Đã viết kế hoạch và evidence tổng hợp. Chưa merge/push, chưa sửa thuật toán, chưa chạy unit test mới, chưa tái tính metric, chưa chạy Colab/GPU/train và chưa cập nhật Linear.

## Cập nhật sau implementation và review — 05/11/2026

Phần trên giữ như bản kế hoạch/lịch sử ban đầu. Trạng thái hiện tại:

- NEXT-01: đã tái lập Gaussian30/Simplex20, kiểm confusion và provenance; xem
  RESULT_TRACKING và aggregate evidence, không yêu cầu C tạo lại cache sẵn có.
- NEXT-02: branch tích hợp và adapter đã có; notebook được sửa lifecycle,
  tách smoke/full và kiểm thử CPU. Full suite 164 pass, môi trường data-only
  113 pass/10 skip. Runtime GPU/Colab với checkpoint thật chưa được xác nhận.
- NEXT-03: terminal decision cho run batch1 cũ vẫn thuộc A/C; review B này
  không cập nhật trạng thái run hoặc tự kết luận recovery đã được giải quyết.
- NEXT-04: notebook Gaussian20 sẵn cho smoke riêng trước full83. Chưa chạy thật;
  giới hạn45phút là cooperative guard giữa các calls, không hard GPU shutdown.
- NEXT-05/06: so cùng budget Gaussian20/Simplex20 và bootstrap chưa hoàn tất.
  Grouped Gaussian30/Simplex20 đã có nhưng khác budget, không thay thế NEXT-05.
- NEXT-07: vẫn là đề xuất có điều kiện; chưa bắt đầu fine-tune/training mới.

Chi tiết review: [B_REVIEW_2026-11-05.md](../../B_REVIEW_2026-11-05.md).
Giữ nguyên test sealed và các quyết định chọn bằng calibration; không chọn lại
fusion hoặc hyperparameter bằng holdout. Không lấy CI/test pass làm Done cho
blocker chất lượng reconstruction/anomaly score.

### Tiến độ NEXT-04/06 sau review

NEXT-06 đã thực hiện cho Gaussian30/Simplex20:1000 paired group bootstrap,
999valid/1missing-class, giữ threshold/normalizer. Xem RESULT_TRACKING và
evidence/2026-11-05-b-group-bootstrap-g30-s20.json. Đây là uncertainty cho
comparison khác budget, chưa thay uncertainty của NEXT-05.
NEXT-04: T4 Colab/source đã sẵn sàng; HF token bị401 trước download, đang chờ
người dùng nhập credential read hợp lệ trong getpass(BLK-20261105-02).

### Gaussian20 inference và đối chứng CPU — 05/11, trước khi nhận output

HF xác thực/checksum đã pass; smoke4/t3 pass và full83/t300/seed17 bắt đầu.
Phần CPU dùng UUID join, giữ fold đã đăng ký bằng thứ tự index CSV rồi ánh xạ
vào vị trí artifact; không dựa vào row order hoặc labels để ghép patient group.
Các candidate raw theo thứ tự: Gaussian20 difference, Gaussian20 localSSIM,
Simplex20 difference, Simplex20 localSSIM; brightness là control loại khỏi
selection. Chọn AP cao nhất trên calibration40; tie ưu tiên candidate đứng trước.
Fit threshold image/pixel trên calibration, lưu selection trước holdout43.
Ghi riêng full83 descriptive với threshold fit trên full83; không ghép hai bảng.
Bootstrap1000/seed17 trên cùng25holdoutgroups, reference Gaussian20 localSSIM,
candidates Simplex20 localSSIM và brightness; mọi threshold giữ nguyên.
Fusion khác budget đã báo riêng, không đưa vào raw matched-budget selection này.
Kiểm full83 Simplex outputs với evidence đã tái lập làm regression thực dữ liệu.
Không dùng kết quả để mở test hoặc tự chọn/freeze protocol cuối.

### Handoff hoàn tất — 06/11/2026

NEXT-04/05 và uncertainty cho matched comparison đã hoàn tất:Gaussian20 full83
receipt được kiểm SHA, đối chứng cùng18.760updates với Simplex20, UUID join và
calibration/holdout40/43;1000bootstrap/999valid. Gaussian20localSSIM được chọn
bằng calibration;holdout AP .099777 so với Simplex .087416;deltaCI bao gồm0.
Artifact/checkpoint private đã chuyển từ Downloads vào runs/checkpoints;Colab
ngắt runtime. Không cần train lại để bổ sung bảng cùng budget này.
Review feature A/B mới của C ở C_HF_REVIEW_2026-11-05.md. Nhiệm vụ tiếp theo là
nhóm review kết quả/giả thuyết cải tiến và chọn một control có budget rõ ràng;
NEXT-07 vẫn có điều kiện, không tự bắt đầu training từ các số validation này.
