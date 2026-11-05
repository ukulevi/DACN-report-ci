# Gaussian DDPM EMA30 → unconditional fine-tune 10 epoch

User yêu cầu chạy đối chứng không thêm encoder ngày05/11/2026, sau A/B và
condition intervention. Đây là đối chứng cho phần cải thiện có thể đến từ thêm
training, không phải phương pháp conditioning mới.

## Khởi tạo và ngân sách

- Workspace/profile `phutungcogioihaitan`, T4,batch8,seed17.
- U-Net một kênh giữ nguyên; không khởi tạo encoder/projection/condition.
- Nguồn `ddpm-partial-t4-b8-e30-cont-20261102T034644Z/ddpm_epoch_030.pt`.
  SHA256 `d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946`.
- Nạp EMA epoch30 vào cả model và EMA. Fresh AdamW, LR1e-5,betas0.9/0.999,
  weight_decay0,clipnorm1,EMA0.9992,Gaussian epsilonMSE,linearT1000.
- Seed/RNG và sampler khởi tạo lại cho fine-tune, như A/B. Không restore source
  optimizer/history/best. Cùng preprocessing và train7500normal slices.
- Train epoch31..40:10epoch/9380updates mới. Optimizer-step counter trong run
  mới bắt đầu0,không phải cumulative counter28140 của nguồn.
- Chọn best mới chỉ trong10epoch fine-tune bằng EMA epsilonMSE trên39normal
  validation, fixed seed1000017,batch8. Không chọn bằng AP/tumor annotation.
- Milestones31/35/40,ddpm_best.pt,ddpm_last.pt,initialization.json,history;
  initial recovery checkpoint ở epoch30/0updates. Chỉ đặt tên epoch để thể hiện
  lineage; best epoch36 tương ứng fine-tuneepoch6 của A/B.

Đây là **EMA warm-start fine-tuning**, khác native full-state continuation từ
ddpm_last.pt với LR1e-4. Native continuation vẫn chỉ cho thay horizon/milestones;
không nới lỏng contract đó để âm thầm đổi LR/optimizer. Checkpoint fine-tune giữ
schema partial-DDPM không condition và config của chính run, thêm provenance
ema_initialization; loader gốc dùng được khi truyền config tương ứng.

## Triển khai và kiểm tra

Trainer `dacn.training.ddpm_unconditional_control:train` gọi training loop gốc
với EMA initialization có checksum, kiểm tra source Gaussian epoch30 và toàn bộ
architecture/diffusion contract. Không instantiate feature-conditioned model.
Native same-run resume khôi phục model,EMA,optimizer,RNG,sampler/history và
giữ provenance. Không ghi đè source/A/B.

Entry `src/dacn/cloud/modal_unconditional_control.py::run_remote` hỗ trợ fresh
run hoặc same-source/config snapshot resume trên cùng T4,timeout4h,private
HF upload/shared Volumes qua Secret sẵn có. Checkpoint commit mỗi epoch.
Để tránh đụng runID, dùng fresh ID cho mỗi attempt; source/config phải giữ nguyên
khi resume.

Chạy smoke riêng trước full:epoch31 duy nhất,batch2,16normal train,at most8val
được chọn gồm2lớp nhưng loss chỉ trên normal,preview2ảnh/3reverse steps.
Smoke weights không dùng khởi tạo full. Full khởi tạo độc lập từ EMA30.
Configs `ddpm_unconditional_control_smoke.json` và
`ddpm_unconditional_control_t4_epoch40.json`.

Yêu cầu training ban đầu không tự chạy AP sweep hay83-image inference.
Trainer giữ private preview2ảnh/t100 để kiểm tra reconstruction theo contract
gốc; đó không phải full AP evaluation. Khi được yêu cầu so kết quả, dùng cùng
t300,ancestral,seed17,40calibration/43group holdout/scorers như A/B,đánh giá
best normal-val và tách milestone40 nếu cần. Test giữ kín.

So sánh này kiểm tra A/B so với thêm fine-tune U-Net cùng hyperparameters và
update budget. Một seed/validation đã xem nhiều lần vẫn chưa đủ khẳng định
significance hoặc hiệu quả tổng quát.

## Validation được yêu cầu sau training

Ngày05/11/2026 user yêu cầu validation. Training đã hoàn tất40epoch/9380updates
mới; best normal-val epoch36(FT6),noiseMSE0.007353029596. Dùng best36 để so với
best A/B(FT6),không chọn checkpoint theo AP hay tự đổi sang milestone40.
Checksum350791c48fd99babc38c329af5644a4c53a70f6628b6d751e85e3a6cab8cf0f7.

Chạy T4,batch1,ancestral t300,seed17,full83validation và cùng40calibration/
43group holdout/calibration-only thresholds. Chung evaluator/scorers với A/B,
absolute difference/local1-SSIM,brightness control,normal tissue quality và
private figures. So baseline30/A/B qua immutable cache/JSON và kiểm tra protocol
alignment. Giữ noisy arrays để xác minh initial noise giống hệt A/B.

Bounded smoke trước full:replay1normal+1abnormal của Gaussian30 qua300steps,
maxerror<=1e-6 so cached baseline và exact noisy arrays với B;target36 chạy
3reverse steps trên2ảnh phải hữu hạn. Full dùng cùng helper sau smoke,cache
prefix mỗi5ảnh,resume cùng source/config,snapshot và checkpoint. Không training,
timestep sweep,milestone40 AP search hay held-out test. Entrypoint mới
`src/dacn/cloud/modal_unconditional_validation.py::evaluate_remote`,mode smoke/full.
