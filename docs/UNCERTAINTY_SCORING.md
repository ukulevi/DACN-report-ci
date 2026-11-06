# Gaussian EMA30: K4 reconstruction uncertainty và normal-calibrated scoring

User yêu cầu thử hướng uncertainty-aware inference sau spatial conditioning
pilot có AP thấp. Không training lại,không đổi encoder hay checkpoint. Mục tiêu
đo xem averaging/normalization của residual có tăng AP và giảm false positive
so với một reconstruction không. Không giả định variance cao đồng nghĩa tumor.

## Contract cố định trước chạy

- Gaussian original EMAepoch30,t300,ancestral300,batch1,T4.
  Source ddpm-partial-t4-b8-e30-cont-20261102T034644Z/ddpm_epoch_030.pt,
  SHA d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946.
- K=4,base seeds[17,1017,2017,3017],mỗi draw seed riêng qua UUIDhash như baseline.
  Resample cả forward và reverse randomness. Không xuất UUID/derived seeds.
- Draw1/base17 reuse immutable Gaussian cache đã có,hash
  8278ad18543dd87f698c36b510564b9fcd4445d4a1b0ae5d7a859093e2a4e959.
  Chỉ tính thêm3×83=249reconstruction mới;replay2ảnh để kiểm tra cache.
- Dataset350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7,83validation39normal/44abnormal,
  fixed40calibration/43group holdout,25groups mỗi bên,split seed17 như A/B.
  Model selection nguồn đã dùng normal-validation;holdout nội bộ không phải test.
- MRI[-1,1],padding256,crop240 trước scoring. Fullcrop gồm cả nền,không masking
  ROI hay postprocessing riêng cho một phương pháp. Higher score=more anomalous.
- Bốn draws thuộc **một model và một ensemble**,không phải4independent training
  seeds. Variance là stochastic reconstruction variation,chưa phải uncertainty
  epistemic/aleatoric đã được hiệu chỉnh.

## Sáu score,không sweep hyperparameter

Với reconstruction xhat_k,mu=mean_K(xhat),v=sample variance_K(xhat,ddof1),
r=x-mu (signed residual):

| Score | Định nghĩa / vai trò |
|---|---|
| single_absolute | abs(x-xhat_1),baseline1draw |
| single_ssim | 1-localSSIM(x,xhat_1),baseline1draw |
| mean4_absolute | abs(x-mu),control averaging4draws |
| mean4_ssim | 1-localSSIM(x,mu),control averaging4draws |
| variance_normalized | r²/(v+lambda),lambda từ normal calibration |
| patch_mahalanobis | (r_patch-m)^T Sigma_reg^-1 (r_patch-m),signed3×3 patch9D |

Nếu normalization không vượt averaging control thì chưa có lợi ích riêng từ
uncertainty. Mahalanobis scalar1D chỉ là standardized squared residual;nên dùng
9Dpatch để kiểm tra cấu trúc covariance thay vì gọi trùng score bằng tên khác.

**Fit chỉ trên normal calibration** (không normal holdout,không abnormal
calibration và không mask tổn thương):

- lambda=max(1e-6,median pixel variance của normal calibration trong240crop).
  Giữ nguyên lambda khi chấm holdout;không chọn lambda bằng AP.
- Pooled signed residual patch3×3 dùng reflect padding sau crop. Học mean9D và
  population covariance9×9 từ toàn pixel normal calibration. Không ước lượng
  9×9 covariance riêng cho từng pixel từK4(vốn rank<=3).
- Sigma_reg=0.9Sigma+0.1diag(Sigma)+ridgeI,
  ridge=max(1e-8,1e-4trace(Sigma)/9). Eigenvalue floor bằng ridge để ổn định
  numerical inversion. Không search shrinkage/patchsize/ridge theo holdout.
- Mahalanobis này là model thống kê **normal residual**,không phải predictive
  covariance riêng của4samples. Hai normalized scores kiểm tra hai cơ chế khác.
- calibration_parameters.json ghi/freeze mọi fitted parameter trước đo holdout.

Threshold pixel/image maxF1 chỉ học trên40calibration(normal+abnormal),giữ trên
43holdout. All83optimized thresholds là descriptive. Pixelglobalmicro240crop;
image score mean top576pixel. Report ROC-AUC,AP,Precision,Recall,F1/Dice,
normal pixel FPR,latency/VRAM,normal tissue quality và score/variance means trong
lesion/nonlesion. Diagnostics trên toànvalidation không dùng để retune tham số.

## Kiểm tra và chi phí

Bounded smoke trước full:1normal+1abnormal,K4,t300,replay draw1phải khớp
cached baseline(maxerror<=1e-6),6new reconstructions và finite variance/
normalized/Mahalanobis. Fit normal smoke riêng để kiểm tra toán,**không AP**.
Smoke không được dùng để chọnK/lambda/score hay chuyển fitted parameters sangfull.

Full dùng all83,cachedfirstdraw+249new,t300. Dự kiến khoảng45–60phútT4 tùy
throughput,hardtimeout2h. Prefixcache độc lập cho mỗi new draw,save/commit mỗi
5images và đầu/cuối;metadata checkpoint/index/base_seed xác thực khi resume.
Repeated run requires same source snapshot/config/mode. PrivateHF uploads
gồm prefix arrays,calibration parameters,ensemble summary và aggregate results.
Single-draw AP phải tái lập registered baseline theo đúnggrouped protocol.

Entrypoint `src/dacn/cloud/modal_uncertainty.py::run_uncertainty_remote`,mode
smoke/full. Workspace phutungcogioihaitan,shared Secret sẵn có. Không tải dữ liệu
MRI về local,không inference Spatial/encoder trong trial này. Không test-set
access,timestep sweep,checkpoint search hay training. Full detached sau start
confirmation;check khi user yêu cầu.

## Giới hạn

K4variance có thể nhiễu. Tumor variance cao có thể bị normalization giảm score;
mahalanobis có thể mô hình hóa nền nhiều hơn mô não. Chỉ xác nhận lợi ích bằng
AP/FPR thực tế so với mean4controls;có thể kết quả âm. One previously inspected
validation/ensemble không đủ significance hay clinical generalization. Đây là
thử scoring trên model hiện tại,chưa đảm bảo model không tái tạo tumor.

## Kết quả thực tế — kiểm tra06/11/2026

Đã hoàn tất83ảnh/K4 với249new reconstructions và cachedfirstdraw. Replay
maxerrors[0,0],single-draw AP tái lập đúng baseline. Normal calibration20ảnh,
1152000patchvectors;variance floor0.00005194913791110404,shrinkage0.1,
ridge0.000000835616497628855. Các tham số được giữ cố định trước holdout.

| Score | Pixel AP,holdout43 | Pixel F1 | Normal pixel FPR | Image AP |
|---|---:|---:|---:|---:|
| Single absolute |0.101066|0.173678|5.8079%|0.430006|
| Single SSIM |0.102402|0.154654|5.5234%|0.505258|
| Mean4 absolute |0.103810|0.173276|5.2835%|0.453941|
| Mean4 SSIM |0.099472|0.141579|3.8697%|0.539785|
| Variance-normalized |0.045729|0.123433|4.0329%|0.720711|
| Patch Mahalanobis |0.077657|0.148199|13.7883%|0.542902|

**Kết luận cho tumor segmentation:hai score mới không vượt baseline hoặc
averaging controls.** Mean4absolute chỉ tăng nhẹ AP;mean4SSIM AP giảm. Normal
FPR giảm ở một số score đi kèm Recall thấp hơn,không phải so ở cùng sensitivity.
Mahalanobis pooled-normal patch này cũng chưa cải thiện Pixel AP.

Variance-normalized cải thiện **image-level AP** lên0.720711/AUROC0.594298,
nhưng Pixel AP chỉ0.045729/AUROC0.534862. Không dùng image-level cải thiện để
kết luận định vị lesion tốt hơn. Brightness-only control trên cùng holdout có
image AP0.770309/AUROC0.679825,còn cao hơn;không chỉ chọn comparator thuận lợi.
Một fixedensemble/validation đã khảo sát chưa xác nhận significance.

Normal foreground reconstruction mean4 tốt hơn single:MAE[0,1]
0.063042679 vs0.082674084;SSIM0.408378352 vs0.292438059. Lesion variance mean
0.041919858,normal foreground variance0.032296945. Trong44abnormal images,
normalized-score macro mean lesion5.556999 thấp hơn nonlesion foreground
11.326831,gợi ý score này đã không giữ độ phân biệt lesion trong dữ liệu hiện
tại. Chưa gán một cơ chế duy nhất từ mean thống kê hoặc kết luận uncertainty
nói chung vô ích.

Chi tiết đầy đủ pixel/image ROC-AUC,Precision,Recall,F1,thresholds,all83
descriptive metrics và timing ở [RESULT_TRACKING.md](RESULT_TRACKING.md).
Walltime từ start đến completed khoảng41.80phút. Progress elapsed reset giữa
draws,phù hợp chạy tiếp prefix cache;finalattempt367.99s inference phase và
469.90s total không đại diện toàn bộ job. Peak allocated VRAM101438464B.

[Private artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/af0bf4006db4c948292b9bc50748073e1b421bb4/evaluations/ddpm-uncertainty-k4-val83-t4-20261105T163529Z),
revisionaf0bf4006db4c948292b9bc50748073e1b421bb4. Result SHA
fc12e2093045f9b6f7c0850e683a7c0258c05529d6a92580419101bf77f8ce16
khớp manifest;completed2026-11-05T17:17:40UTC=06/11 00:17VN. Không training
lại hoặc dùng held-out test. Không tự tăngK,tune normalization hay chạy tiếp
GPU sau khi đọc kết quả này.
