# Thiết kế nghiên cứu: DDPM partial reconstruction

## Quyết định và câu hỏi

Ngày 28/10/2026 nhóm chọn đề xuất DDPM reconstruction thay kế hoạch ANDi-FLAIR trước đây. ANDi là related work/extension, không là đường găng. Đây là kế hoạch, không phải kết quả hay tuyên bố phương pháp mới.

**RQ1:** Mức nhiễu khởi đầu `t*` ảnh hưởng thế nào tới anomaly detection và localization, thời gian và VRAM trên BMAD Brain FLAIR 2D?
**RQ2:** Trên cùng ảnh tái tạo, pixel absolute difference và local SSIM difference thay đổi kết quả như thế nào? Khi đủ nguồn lực, so với DAE coarse-noise.

Giả thuyết: nhiễu ít có thể giữ cả tổn thương; nhiễu nhiều có thể xóa tổn thương nhưng cũng làm sai mô bình thường. Chỉ xác nhận hoặc bác bỏ bằng artifact thực tế. Partial diffusion đã có tiền lệ trong [AnoDDPM](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf).

## Hợp đồng mô hình và thực nghiệm

### K4 uncertainty scoring — authorized2026-11-05

User explicitly requests inference-only trial after proposing uncertainty-aware
scoring. Freeze original GaussianEMA30,t300,K4 bases17/1017/2017/3017,
reuse immutable first draw,three new draws per83validation. Six fixedscores:
single absolute/SSIM,mean4 absolute/SSIM,variance-normalized squared residual,
pooled normal-calibration3x3patch Mahalanobis. Parameters normal-calibration only,
thresholds40calibration,freeze43group holdout,no test/training/sweep. Bounded
2image K4smoke before full. Exact formulas/cache/reproducibility and limits in
UNCERTAINTY_SCORING.md;do not infer uncertainty guarantees tumor localization.

### Spatial ImageNet conditioning, grayscale — authorized2026-11-05

User requests Spatial ResNet18 ImageNet pretrained +original U-Net EMA30,then
explicitly keeps1channel grayscale. Follow SPATIAL_IMAGENET_GRAYSCALE.md:
meanRGB conv1 adaptation,MRI[-1,1],noRGB repeat/norm/crop,4spatial maps,
zero1x1 residual fusion at64/32/16/8. Joint10epoch/9380update pilot after bounded
T4 smoke,encoderLR1e-5,U-Net1e-5,fusion1e-4,normal-only checkpoint selection,
automatic same83slice t300 evaluation with baseline/A/B/unconditional control.
No test access or claim that this isolates spatial fusion from pretraining.

### Unconditional control validation — authorized2026-11-05

User requests validation after completed control training. Freeze normal-val
best36(FT6),9380updates,EMA,noencoder. Same full83,ancestral t300,batch1,seed17,
40calibration/43group holdout/calibration-only thresholds as A/B. Bounded
replay/source-noise smoke before full. Baseline/A/B private artifacts immutable,
aligned input/noise checks,quality/latency/VRAM/private examples. No extra
training,sweep,test access or selection using AP. Exact protocol appended to
UNCONDITIONAL_FINETUNE_CONTROL.md. Earlier training-only scope is superseded by
this explicit validation request.

### Matched unconditional fine-tune — authorized2026-11-05

User requests original Gaussian DDPM epoch30 plus10fine-tune epochs without
encoder, following A/B conclusion. See UNCONDITIONAL_FINETUNE_CONTROL.md:
EMA30 warm start,freshAdamW/U-NetLR1e-5,batch8,seed17,9380new updates,epoch31..40,
same39normal-val selection and EMA as A/B. Preserve existing full-state resume
contract; do not confuse this control with LR1e-4 native continuation.
Bounded smoke precedes full job; training/private preview only,no automatic
AP sweep/full83 inference or held-out-test access.

### B condition intervention — authorized2026-11-05

User requests inference-only diagnostic after B's AP approximately matches A.
Freeze B normal-val-selected EMA fine-tuneepoch6,checkpoint SHA
0c4f21705830f5dd6054174fc6b2c1591263779413737ce87b4407e3ca8b01b7,
ancestral t300,batch1,seed17 and all83 validation samples. Three cases:
original-image condition(reuse immutable B cache), zero entire injected embedding
(bypass projection including bias), and condition from a deterministic normal
training donor(UUID-derived seed77 modulo7500,never chosen using masks/metrics).
Donor affects encoder input only; forward image/noise/reverse RNG are paired.
No checkpoint changes, fine-tuning, timestep sweep or held-out-test access.

Before full evaluation, bounded T4 smoke replays one normal/one abnormal image
through original300steps and requires identical initial noise/max reconstruction
error<=1e-6 against registered B cache, then finite3step zero/donor controls.
Full run reuses self-condition cache, computes two83-image interventions and
checks exact input/mask/label/noisy-image alignment. Separate prefix caches and
source-checked manifests support resume. Metrics follow existing40calibration/
43group holdout with calibration-only thresholds; all83 metrics descriptive.
Report AP,ROC-AUC,Precision/Recall/F1,normal pixel FPR and reconstruction changes
versus self on240crop scaled[0,1], separately normal/abnormal. Thresholds are
independently calibrated per intervention; AP does not depend on threshold.
One donor per image/one exploratory seed; similar results would show weak
sensitivity under these interventions, not proof the encoder is always ignored.
Private caches/figures and model artifacts remain on Modal/private HF.

### ResNet-18 feature conditioning B — authorized2026-11-05

After completed A, user requests B. Follow FEATURE_CONDITION_B.md:5epochs
normal-only masked ResNet-18 pretraining, then independently initialize the same
Gaussian EMA30 U-Net and fine-tune10epochs/9380joint updates with A's exact
settings. Additional encoder pretraining compute is separate. Bounded smoke
precedes full T4 chain with automatic fixed t300 validation/A/baseline comparison;
no timestep search, external encoder weights or held-out-test use.

### ResNet-18 feature conditioning A — authorized2026-11-05

User supersedes the initial pretraining preference in docs/newmethod.md and
requests experiment A first: random ResNet-18, zero condition projection,
original Gaussian epoch30 EMA U-Net, joint10-epoch fine-tune then Pixel AP
validation. Follow FEATURE_CONDITION_A.md for exact architecture/learning rates,
EMA buffers, update budget, normal-only selection, bounded smoke and automatic
t300 validation with existing grouped calibration/holdout and baseline cache.
This entry records A's original scope; B is subsequently authorized above.

### Simplex epoch20 validation — authorized2026-11-04

After completed Simplex continuation, user requests validation. Evaluate explicit
ddpm_epoch_020.pt EMA from ddpm-simplex-t4-b8-e20-cont-20261104T070341Z on all83
validation slices (39 normal/44 abnormal), T4, ancestral batch1,t300,seed17,
same Simplex-forward/Gaussian-fixed-large reverse adaptation as epoch10.
Verify checkpoint against completed parent manifest, same pinned dataset.
Full240crop absolute difference/local1-SSIM, existing all-validation max-F1
threshold rule and top576 image score; compare same-protocol epoch10 metrics.
Save private reconstruction cache for possible later group/fusion analysis.
No new training, no timestep sweep or held-out-test access. Fresh evaluation ID,
detached execution, status checked on user request; do not continuously monitor.

### Simplex continuation to20 total epochs — authorized2026-11-04

User requests continuing Simplex training to20 total epochs. Use completed
ddpm-simplex-t4-b8-e10-20261102T074257Z/ddpm_last.pt (epoch10/9,380steps),
with original model/EMA/optimizer/sampler/RNG states; fresh output run ID via
continue-from-run-id. Preserve source checkpoints. Config
configs/ddpm_partial_simplex_t4_batch8_epoch20.json differs only in epochs20
and milestones[1,5,10,15,20]. T4,batch8,LR1e-4,EMA0.9992,seed17,7500 normal
train and39 normal-validation slices, same pinned dataset and Simplex settings.
Train epochs11..20,9,380 additional steps,18,760 cumulative. Normal-validation
epsilon MSE still selects best; persist last each epoch and new milestones15/20,
private Hub upload on completion. Approximately75min incremental training from
prior timings, not a guarantee. No automatic GPU evaluation/sweep or test access;
check results when requested, do not continuously monitor.

### Gaussian/Simplex SSIM fusion — authorized2026-11-04

CPU-only fusion of registered ancestral t300 caches: Gaussian EMA epoch30 and
Simplex EMA epoch10. Verify immutable cache/index hashes and identical full83
input/mask/label arrays. Reuse first StratifiedGroupKFold2/shuffle/seed17 split:
40 calibration and43 grouped holdout. Compute local1-SSIM; fit each empirical
midrank CDF on all pixels of normal calibration slices only. No per-image or
holdout normalization fitting. Three fixed Gaussian weights0.25,0.5,0.75;
controls raw Gaussian/raw Simplex and each CDF-normalized map. Brightness is an
independent control, excluded from diffusion-method selection. Select highest
calibration pixel AP among diffusion candidates (exact ties prefer raw Gaussian
by fixed order); also freeze highest-calibration-AP fusion before holdout.
Fit pixel/image max-F1 thresholds only on calibration. Write frozen selection
before evaluating any holdout candidate; no weight/normalizer retuning from
holdout outcomes. Exploratory previously inspected validation, not final test.
This tests an ensemble with different training budgets, not causal noise effects.
No new GPU inference, no training or test-set access. Outputs private/ignored.

### Fixed t300 sampler comparison — authorized2026-11-04

User approves original Gaussian EMA epoch30 on all83 validation images: one-step
x0 versus DDIM10 eta0 versus existing ancestral300 cache. Same base17 UUID seed
and exact initial noisy tensor for new samplers, t300=index299. DDIM model-call
indices299,269,239,209,179,149,119,89,59,29, then clean index-1(alpha1);
10 calls, not11. Use clipped x0 and epsilon re-derived after clipping as in
[guided-diffusion DDIM](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py).
No reverse random draws. Reuse immutable ancestral cache after SHA/content/order
checks and recompute first sample as a seed/sampler control. Same40 calibration
/43 grouped holdout, independently fit each score's thresholds on calibration
only; report AP/F1 and normal pixel FPR, plus descriptive all83 metrics. Both
pixel difference and localSSIM; brightness control. Time new reconstruction
calls with CUDA synchronization separately from scoring/cache work. Historical
ancestral runtime is not a matching pure model timing. No training/test access,
no extra timestep/step-count sweep. Save private reconstructions/results.

### One-step patch/context validation — authorized 2026-11-04

Follow-up user request: evaluate original Gaussian DDPM epoch30 EMA one-step on
all83 validation images, t100. Use full-image Gaussian noise from the same
base17 UUID seed as original ancestral baseline, one model call, direct x0
formula and clamp[-1,1]. No patch averaging/context channels/fine-tuning. Keep
the same scores, grouped calibration/holdout, descriptive all83 metrics and
private artifacts. This isolates inference change versus original epoch30
t100 ancestral reconstruction; context comparison additionally changes weights
and patch averaging. Test sealed, no timestep sweep.

User requests T4 one-step inference on all83 validation slices. Use EMA best
context checkpoint from fine-tune epoch5, t100, Gaussian forward draw, same
private UUID-derived base17 seeds and patch64/stride48 coverage as the reverse
diagnostic. Each patch calls epsilon model once at index99, computes x0 using
the schedule, clamps to[-1,1], and retains original context outside that patch.
Average overlapping patch outputs;25 model calls/image. No fine-tuning.
Reuse the full-validation evaluator: pixel difference/localSSIM plus brightness,
fixed40-slice calibration /43-slice grouped holdout, calibration-only thresholds;
separate descriptive all83 metrics. Test sealed. This is one-step clean-image
estimation from noised input, not distilled one-step generation. Prior full
reverse validation remains user-stopped; this is a fresh evaluation run.

### Approved patch/context fine-tune pilot — 2026-11-02

User approved5-epoch T4 batch8 fine-tuning from Gaussian epoch30 EMA at LR1e-5.
This is the team's3-channel adaptation, not an exact pDDPM reproduction. Keep
the source DDPM checkpoint immutable; verify SHA256
d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
Load its EMA into the original base32 U-Net, then replace only the first
convolution: copy original image-channel kernel/bias exactly, initialize the
two context/mask channel kernels to zero; output remains1 epsilon channel.
New optimizer (AdamW LR1e-5,betas0.9/0.999,weight_decay0,clip1), all parameters
trainable; new EMA starts equal to expanded model, decay0.9992. No source
optimizer/RNG resume. Fine-tune epochs1..5 are separate from the30 pretraining
epochs; report both compute budgets, planned4,690 additional optimizer steps.

Data: all7,500 normal training slices, unchanged256 input normalization and
Gaussian linear T1000. One64x64 patch per image, centered on a uniformly drawn
foreground pixel(input>-0.95), clipped inside original240 crop. First channel
uses noised image inside patch and clean context outside; second channel is
clean input outside patch and zero inside; third is binary patch mask. This
prevents clean target pixels inside patch leaking through conditioning.
Uniform0..999 timestep; loss mean squared epsilon error only in the patch,
normalized per image by its mask area. No abnormal annotations in training.

Selection: fixed-RNG patch/noise/timestep MSE over all39 normal validation slices,
seed17+1,000,000, batch8, EMA. Save full optimizer/RNG/history/model/EMA states
each epoch as patch_context_last.pt, normal-val-selected patch_context_best.pt,
and milestones1/3/5. Patch loss is a new objective and not directly comparable
to old full-image noise MSE. Private final-EMA one-step t100 normal preview is a
sanity artifact, not an AP evaluation. No training epochs beyond5 authorized.

Inference helper covers all240x240 pixels using64 patches/stride48 and equal
overlap averaging, independent per-patch seeds; no tumor mask needed. Full
sliding reverse inference can be expensive (25 patches per image), so measure
a bounded pilot before any full83-image AP evaluation. Keep the established
group calibration/holdout and brightness baseline for later evaluation; no
automatic additional GPU inference sweep in this training run. Test sealed.

Follow-up authorized2026-11-02: after the four-example reconstruction pilot,
user requested evaluation on all83 validation slices (39 normal,44 abnormal).
Freeze EMA fine-tune epoch5, t100, patch64/stride48,25 patches, same per-sample
and per-patch seed derivation, fixed-large stochastic reverse; no new training.
Use full240crop pixel difference/local SSIM and brightness control; thresholds
fit only on the established40-slice group calibration, applied unchanged to
43-slice group holdout (StratifiedGroupKFold2,shuffle,seed17,first fold).
Also report all83 validation metrics as descriptive with locally fit thresholds,
separate from group holdout. Validation has been inspected; neither is final test.
T4 detached,3-hour ceiling based on ~100-minute pilot extrapolation; persist
private prefix reconstructions after first sample and every5 samples, upload
final artifacts privately. No additional timestep sweep or sealed-test access.

- Input: một kênh PNG FLAIR đã render, pad 240→256, normalized [-1, 1] theo [DATA_PROTOCOL](DATA_PROTOCOL.md). Train chỉ trên normal slice; normal ở đây là slice không có tumor annotation, không đồng nghĩa bệnh nhân khỏe mạnh.
- **Contract triển khai DAC-8 — bản đề xuất, chờ A/B review:** train U-Net dự đoán Gaussian noise với MSE (`epsilon-prediction`), `T=1000`, linear β từ 0.0001 đến 0.02 theo [DDPM](REFERENCES.md); timestep huấn luyện lấy đều trên `[0,999]`. Kiến trúc một kênh: U-Net hiện có, base channels 32, multipliers `(1,1,2,2,4,4)`, hai residual blocks/level, attention ở resolution 16 và 8, hai heads, dropout 0. Tối ưu AdamW (`lr=1e-4`, weight decay 0), gradient norm clip 1, EMA 0.9999. Đây là cấu hình triển khai được đề xuất, chưa phải cấu hình đã được nhóm phê duyệt qua review hay đã có kết quả.
- **Quy ước partial reconstruction:** `t*` bên ngoài là one-based và chỉ mức forward-noise: `t*=100` ánh xạ tới timestep array index 99. Tạo một noisy input bằng forward process, rồi chạy toàn bộ ancestral DDPM reverse transitions từ index `t*-1` về 0 (`t*` reverse steps); fixed-large reverse variance theo `GaussianDiffusion`, dự đoán clean image được clip trong `[-1,1]`, đầu ra cuối cũng clip trong `[-1,1]`. Phân biệt `t*`, tổng schedule 1000 và số reverse steps. Ghi rõ khác biệt với nguồn tham khảo nếu contract review đổi lựa chọn nào.
- **Checkpoint proposal:** chọn epoch có epsilon-MSE thấp nhất trên normal validation slices với timestep/noise cố định, dùng EMA weights. Labels dùng để stratify validation cap và lấy normal subset; anomaly labels/masks không tham gia loss hay tiêu chí checkpoint. Held-out test không được trainer nhận. A/B cần xem xét checkpoint rule trước nghiên cứu chính.
- `t*` ứng viên nghiên cứu vẫn là 100, 200, 300; 50 và 500 chỉ mở rộng sau quyết định budget. Trainer cung cấp `load_ema_checkpoint`, `build_diffusion`, `derive_sampling_seed` và `partial_reconstruct`; B dùng cùng reconstruction cho pixel absolute difference và local SSIM. Không train lại theo score/timestep. Đo latency reconstruction và score riêng, tổng end-to-end, VRAM và số ảnh thực đo.
- **Bounded smoke proposal:** seed 17, một epoch, batch 1, tối đa 128 normal train slices và 16 validation slices, preview `t*=100`, tối đa 4 ảnh validation. Đây chỉ là pipeline/sanity smoke, không phải model comparison hay nghiên cứu. Chưa được chạy trong phần DDPM này; DAC-8/A-B review và DAC-9 data gate phải hoàn tất trước khi bắt đầu GPU job.
- **Training trước sweep:** sau smoke/pilot và review, đặt `max_train_samples=null`, `max_validation_samples=null` để dùng cả 7.500 normal train slices và 39 normal validation slices cho checkpoint loss; các validation abnormal slices không vào loss. Cố định số epoch và mọi tham số training theo budget pilot **trước** run nghiên cứu, ghi config/seed/checkpoint; không mang checkpoint của smoke 128/16 vào bảng kết quả. Epsilon-MSE validation dùng seed `train_seed + 1000000`, batch size và thứ tự validation cố định trong config.
- **Score contract đề xuất cho DAC-8:** crop padding trước khi tính score; hai ảnh float trong `[-1,1]`, ROI là toàn vùng 240×240, không foreground mask hay post-processing. Pixel map là `abs(input - reconstruction)`. Local SSIM map là `1 - SSIM`, dùng Gaussian window 11×11, sigma 1.5, chuẩn hóa kernel về tổng 1, weighted population variance/covariance, padding `reflect` trên ảnh đã crop, `data_range=2`, `C1=(0.01*2)^2`, `C2=(0.03*2)^2`. Không clip hay chuẩn hóa score map; score cao hơn là bất thường. Image score là mean của 576 pixel có score cao nhất (1% của 240×240). B cần review contract và implementation ở DAC-30 trước sweep.
- **Seed policy đề xuất:** train seed và sampling base seed ghi riêng; sweep đầu dùng mỗi loại seed 17. Cho mỗi sample UUID của export HF, canonicalize UUID và tính 4 byte đầu SHA-256 của UTF-8 `dacn-ddpm-seed-v1:<base_seed>:<canonical_uuid>` theo big-endian làm seed 32-bit. Gọi reconstruction với batch 1 và dùng cùng seed của sample cho mọi `t*`, rồi dùng chính tensor reconstruction đó cho cả hai score. UUID chỉ dùng trong bộ nhớ/private data, không ghi vào Git, Linear hoặc manifest công khai. Quy tắc này không biến seed thành dữ liệu ẩn danh. Các lần lặp sau phải đăng ký sampling seed và train seed trước khi chạy; train seeds 42/2026 là mục tiêu khi đủ budget, không tự coi là đã chạy.
- **Sweep validation đề xuất:** một checkpoint EMA đã chọn theo normal-validation epsilon-MSE, một train seed 17 và một sampling base seed 17, toàn bộ 83 validation slices theo revision đã pin, `t*` = 100/200/300 và hai score trên cùng reconstruction: 249 reconstruction, sáu tổ hợp score/timestep. Không dùng 50/500, feature score, nhiều checkpoint hay train lại trong sweep này. Chọn tổ hợp bằng global micro pixel AP trên validation; nếu AP bằng nhau ở độ chính xác đầy đủ, ưu tiên `t*` nhỏ hơn, sau đó pixel difference. Chọn riêng image và pixel threshold trên validation bằng F1 cao nhất với rule dự đoán `score >= threshold`, xét các giá trị score phân biệt; nếu F1 bằng nhau, chọn threshold lớn hơn. Pixel micro F1 và Dice tương đương trong phép chọn này. Chốt checkpoint, thresholds và toàn protocol trước held-out test. Ước lượng runtime/quota từ pilot trước khi chạy sweep; nếu không đủ, ghi lại thay đổi và review trước khi chạy, không chọn bằng test.
- **Sai khác nguồn tham khảo:** phần forward dùng Gaussian noise và epsilon-MSE theo DDPM; backbone/process được tái sử dụng từ adaptation AnoDDPM, nhưng cấu hình này dùng Gaussian thay simplex, BMAD FLAIR 2D đã preprocess, fixed-large ancestral reverse, clipping và EMA checkpoint chọn theo normal-validation epsilon-MSE. Đây là adaptation của nhóm, không tuyên bố tái lập nguyên thí nghiệm DDPM/AnoDDPM. Checkpoint rule đo dự đoán noise, không bảo đảm checkpoint tốt nhất cho anomaly detection; A/B cần review hạn chế này.
- DAE là baseline khi khả thi, không chặn diffusion MVP. Feature difference chỉ làm khi còn thời gian và đã chốt encoder, nguồn/training, layer, map, license và chi phí. ANDi, VAE và Isolation Forest là related work/extension tùy chọn.

## Prototype AnoDDPM tách biệt

### Ablation Simplex được người dùng yêu cầu — 2026-11-02

Thử nghiệm mới dùng chính trainer partial DDPM với `noise_type=simplex`;
giữ Gaussian là mặc định cho checkpoint cũ. Config đăng ký trước chạy:
`configs/ddpm_partial_simplex_t4_batch8_epoch10.json`, seed 17, 10 epoch,
7.500 normal slices, T4 batch 8, LR 1e-4, EMA 0.9992, cùng U-Net base32 và
schedule linear T=1000. Simplex: 6 octave, persistence 0.8, frequency divisor 64.
Train từ đầu; checkpoint Gaussian không được resume thành Simplex. Selection
vẫn dùng normal-validation noise MSE, lưu milestone 1/5/10. Forward corruption
train và reconstruction dùng Simplex; reverse vẫn Gaussian fixed-large ancestral
theo adaptation AnoDDPM trong code, không phải posterior-mean hay exact Gaussian
posterior cho Simplex. Không so trực tiếp trị MSE của hai noise distribution như
thước đo chất lượng anomaly detection. Kế hoạch đánh giá sau train: epoch 10,
t*=300, full validation, sampling seed 17, hai score/threshold/ROI như baseline;
không đổi test. Chỉ báo cải thiện khi có kết quả thực, ưu tiên đối chứng Gaussian
epoch 10 batch8 cùng training budget, không chỉ đối chiếu Gaussian epoch30.

Repo có một trainer `dacn.training.anoddpm:train` được port có chọn lọc từ AnoDDPM
để kiểm tra sớm pipeline diffusion một kênh trên BMAD. Nó dùng U-Net dự đoán noise,
simplex hoặc Gaussian noise, full-shard epochs và squared reconstruction error. Đây
là **AnoDDPM-style BMAD adaptation** để kiểm tra data/model/run boundary, không thay
thế protocol DDPM partial reconstruction đã chốt: nó chưa có pixel absolute/local
SSIM comparison và chưa freeze `t*`/sampler theo DAC-8. Không suy diễn smoke test
hoặc validation subset thành kết quả nghiên cứu.

## Protocol đánh giá

### User-requested train/inference contract audit — 2026-11-02

Fixed diagnostic, not an AP sweep: use Gaussian checkpoint epoch30, verify
every loaded EMA tensor exactly equals checkpoint EMA, inspect online/EMA
relative L2 difference and unchanged source modules in training snapshot.
Check4 train tensors and4 selected normal-val tensors for1x256x256 float32,
[-1,1], black padding8px and uint8 normalization grid. Choose4 of39 normal
validation slices uniformly without replacement with NumPy seed17, then sort
indices. Do not choose by reconstruction quality or use held-out test.

At t*=100/300, derive the same sample seed17 as prior inference. Compare
clipped noisy input, one-step x0 prediction with EMA, one-step with online
weights, EMA full reverse with fixed-large variance, fixed-small variance and
deterministic posterior mean. All share identical initial noise; stochastic
chains share reverse Gaussian draws. No production sampler is changed.
Report mean per-slice MAE/RMSE on[0,1] and localSSIM in foreground>-0.95 and
interior distance>5px, plus private paired images. Test oracle-noise inversion,
direct posterior-mean formula, exact linear schedule and both variance arrays.
Unit test ensures diagnostic large chain exactly matches production sampler.

This4-normal-image diagnostic can identify weight-loading mismatches or
sampling contributions, not prove generalization or estimate AP improvement.
Preserving normal anatomy alone is not sufficient evidence of tumor removal.

User-requested reconstruction-map multi-timestep fusion (2026-11-02): fix
Gaussian EMA epoch30 cached t100/200/300, existing83 validation samples, same
grouped40-calibration/43-holdout split seed17. Verify cache hashes and exact
image/mask/label alignment. For each of absolute difference and localSSIM,
fit separate empirical midrank CDFs at each timestep on calibration pixels only,
then average all three normalized maps with equal1/3 weights. Compare these
two fixed ensembles with all six original single-timestep scores and brightness
on the same holdout. Calibration-only thresholds, no weight search, ROI240x240,
no foreground gating, no GPU or new inference. Exploratory validation only,
not a final untouched test. Do not combine this experiment with averaged
epsilon-prediction error, which is a different score already evaluated.

### User-requested AutoDDPM-style pilot — 2026-11-02

This is an independent implementation of mask/stitch/resample ideas from
[AutoDDPM](https://arxiv.org/abs/2305.19643), NOT an exact reproduction of its
LPIPS mask, training dataset/backbone or posterior variance. User was informed
before execution that this pilot substitutes local SSIM for LPIPS, retaining
the current checkpoint and fixed-large Gaussian reverse transitions.

Fixed before run: Gaussian EMA epoch30; reuse checksum-verified initial full-val
t*=200 reconstructions. On each input/reconstruction crop: heatmap=
clip(abs residual / its95th percentile,0,1)*clip(1-localSSIM,0,1).
Predicted binary mask=heatmap>=0.5, dilation3x3; no ground-truth mask used.
Pad mask with zeros, stitch original outside mask and initial reconstruction
inside. Forward to one-based50, reverse50 steps with5 resamples per step;
stitch fresh noised original context at the matching next timestep, with exact
clean original context at the final step. Re-noise x[k-1] to x[k] using beta[k]
before repeating. Seed17, sample-UUID derived, isolated RNG, batch1, T4.

All83 validation images; same40-calibration/43-holdout grouped split as previous
experiments. Fixed outputs: initial raw absolute/SSIM residuals; initial heatmap; naive stitch-only absolute residual;
final absolute residual; final localSSIM; final absolute residual times initial
heatmap; brightness control. Fit thresholds on calibration only, no weight or
mask threshold tuning. Report all outputs without holdout-based reselection,
mask lesion coverage/precision/ROI fraction on both partitions. Coverage is
diagnostic only, not used to generate masks. No foreground ROI filtering.

Outside-mask equality is imposed by construction, so lower outside-mask error
alone is not evidence of better restoration. Naive stitching is the required
control for score gating; mask coverage reveals missed lesions. Private maps
enable later error review. Completed pilot: holdout resampled-SSIM AP0.093603
versus initial t200 SSIM0.090168 and earlier t300 SSIM0.102402; brightness0.389506.
Predicted mask covers63.05% of holdout lesion pixels with9.20% precision.
No overall AP gain established; resampling difference AP0.070007 is essentially
the naive-stitch AP0.069851. Full record in RESULT_TRACKING, private revision
2c8646c1. No retraining or held-out test use.

### Planned denoising-score branch — requested 2026-11-02

Implemented at `dacn.evaluation.denoising`; evaluated on real validation MRI
in run `ddpm-denoising-e30-3t4r-20261102T103159Z`. Holdout AP0.085848 is below
reconstruction AP~0.10 and brightness0.389506 on the same43 images; this fixed
score is not adopted as an improvement. See RESULT_TRACKING for full results.
Reuse Gaussian EMA epoch30; input images/weights/preprocessing remain fixed.
For each sample, derive a seed from sampling base17 and its private UUID; use
an isolated device generator. Draw4 independent Gaussian noise tensors. For each
draw reuse the same tensor at one-based t*=100/200/300 (indices99/199/299).
Average squared epsilon-prediction errors equally over all12 forward passes;
crop to240x240, no reverse chain or smoothing. This is a simple noise-error
ablation, not full ANDi, and high error is only hypothesized to mark lesions.

Evaluate against rendered brightness on the same patient-group split used for
score fusion: first StratifiedGroupKFold(2,shuffle=True,random_state=17) split,
calibration thresholds frozen before internal-holdout metrics. Existing
reconstruction holdout metrics remain the comparison. Fixed score parameters,
no additional weighting/timestep search. The user requested pipeline changes;
implementation/tests and the user-authorized T4 run are complete. Any further
run must be separately registered in RESULT_TRACKING after start.

Usage in an environment with the private checkpoint and validation shards:

```bash
python -m dacn.evaluation.denoising \
  --validation-dir PRIVATE_VALIDATION_SHARDS \
  --checkpoint PRIVATE_DDPM_EPOCH_030.pt \
  --output-dir NEW_PRIVATE_OUTPUT_DIRECTORY \
  --device cuda
```

Results contain aggregate pixel/image metrics, checkpoint/index hashes and
partition sizes. Score maps and masks are private derived medical data; keep
output within ignored runs/ or the private Modal/Hub artifact storage.

Score-fusion diagnostic registered 2026-11-02 at the user's request: use only
Gaussian epoch30 t*=300 full-validation cache, same original ROI. Fix the first
split of StratifiedGroupKFold(2, shuffle=True, random_state=17) by Hub patient
group; first training indices are calibration and first validation indices are
internal holdout. Evaluate raw brightness, absolute difference and local SSIM,
plus w*CDF(brightness)+(1-w)*CDF(reconstruction score), w in {0.25,0.5,0.75}
for each of the two reconstruction scores. Fit empirical midrank CDFs and
max-F1 pixel/image thresholds on calibration only; select one of nine candidates
by calibration pixel AP, exact ties in predefined order (brightness first).
Persist selection before reading holdout metrics. Report all predefined
candidates and the calibration-selected candidate, without selecting weights
again on holdout. This is exploratory internal validation: full validation has
already influenced prior investigations, so holdout is not an untouched final
test. No new model training/GPU inference or held-out test access.

Các hàm nền cho DAC-12/DAC-30 nằm trong `dacn.evaluation.scores` và
`dacn.evaluation.validation`: `paired_score_maps` nhận một cặp 256×256 và trả
hai score map đã crop; `evaluate_reconstructions` nối hai score với evaluator
trên cùng các cặp reconstruction; `evaluate_maps` nhận map và mask đã crop 240×240, trả
metric tổng hợp và chọn hai threshold riêng khi `split="validation"`. Khi
`split="test"`, caller phải truyền threshold validation đã freeze và chịu
trách nhiệm ghi provenance của threshold. Đây là API nền đã kiểm tra trên dữ
liệu tổng hợp, chưa phải pipeline chạy dataset, sweep hay kết quả thực nghiệm.
Caller phải dùng cùng tensor reconstruction cho hai map, không công bố UUID,
ảnh hoặc mask; ghi checkpoint, seed, split và số seed theo protocol khi chạy thật.

- Split `patient-disjoint-v1` và revision Hub cố định theo DATA_PROTOCOL. Validation có nhãn dùng chọn checkpoint/config/threshold: mô tả là normal-only training với labeled validation selection. Không dùng held-out test để lựa chọn hoặc tuning.
- Pixel AP chính: global micro average precision trên vùng ảnh gốc 240×240, crop padding, gồm ảnh normal/abnormal. Report pixel/image ROC-AUC, Precision, Recall, F1; thêm pixel AP, Dice, latency, VRAM khi phù hợp. Image score là mean top 1% pixel score. Chốt threshold image và pixel riêng trên validation; pixel Dice micro-optimal trên validation là quy tắc dự kiến, phải freeze trước test.
- Sweep và threshold đề xuất ở trên cần A/B review, rồi được đăng ký trước khi
  chạy validation; ghi thresholds thực tế và freeze trước test.
- Chốt foreground/post-processing cho từng method, công bố khác biệt; cùng vùng metric cho mọi phương pháp. Kiểm tra trên validation tỷ lệ lesion pixel bị loại bởi foreground erosion của DAE trước so sánh; không thay protocol dựa trên test.
- Ghi macro Dice trên slice lesion và false-positive pixel rate ở mask rỗng; báo prevalence. Dùng cùng tập validation và quy tắc ROI cho đối chứng. Pixel metrics dựa trên cùng output tái tạo, score và threshold đã đăng ký.
- Validation chỉ 83 slice/50 patient groups: hạn chế sweep, không xem một lần chọn trên validation như kết luận chắc chắn. Cố gắng 3 train seeds khi ngân sách cho phép; ghi số thực chạy. Bootstrap theo patient group, ghi số replicate/seed và xử lý replicate thiếu lớp; không coi slice/pixel là bệnh nhân độc lập.
- Latency: cùng GPU, batch, precision và resolution; đề xuất warm-up 20 batch,
  đo ít nhất 100 slice khi khả thi và GPU synchronize, báo median/p95 ms/ảnh và
  peak VRAM. Nêu rõ nếu không đạt số lượng này và tách scoring khỏi I/O.
- Sau review, freeze model, seed policy, `t*`, sampler, scores, preprocessing, thresholds và metric trước đánh giá held-out test. Mọi thay đổi sau test phải công khai; không tuning lại trên test.

## Kết quả và giới hạn

Theo dõi mọi run thực bắt đầu, cả lỗi/gián đoạn, trong RESULT_TRACKING với manifest khi có. Hình validation: ảnh đầu vào/noisy/reconstruction/map và trường hợp lỗi; test chỉ xem sau freeze. PNG không giữ intensity MRI gốc; train normal slice có thể thuộc bệnh nhân u; chưa chứng minh generalization hoặc tính lâm sàng. Dùng shared private artifacts và chỉ đưa số aggregate an toàn vào Linear.
