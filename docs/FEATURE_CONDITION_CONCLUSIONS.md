# Kết luận DDPM conditioning: baseline, thí nghiệm A/B và kiểm tra condition

Cập nhật: **06/11/2026**. Tài liệu tổng hợp kết quả thực tế đã hoàn tất trên
BMAD-derived Brain FLAIR 2D: Gaussian DDPM epoch 30, ResNet-18 condition A,
encoder pretraining B, phép thử tắt/đổi condition trên B và đối chứng fine-tune
U-Net thuần thêm 10 epoch đã hoàn tất validation.
Kết quả spatial ResNet-18 ImageNet grayscale mới được bổ sung ở mục14.
Uncertainty K4 trên GaussianEMA30 được bổ sung ở mục15.

Các thí nghiệm trước đó về Simplex, fusion, one-step, DDIM và patch/context được
lưu riêng trong [RESULT_TRACKING.md](RESULT_TRACKING.md). Tài liệu này tập trung
vào chuỗi A/B và chẩn đoán condition vừa thực hiện; không gộp điểm từ các protocol
khác nhau thành một bảng xếp hạng.

## 1. Kết luận chính

**Condition đúng ảnh chưa chứng minh được lợi ích rõ ràng cho anomaly segmentation
trong cấu hình đang thử. Chưa có cơ sở tốt để tiếp tục tăng epoch với nguyên thiết
kế này chỉ vì muốn tăng AP.**

- A và B nhỉnh hơn baseline về Pixel AP trên nhóm validation holdout, nhưng mức
  tăng nhỏ. Đối chứng fine-tune U-Net thuần cùng 10 epoch đạt AP gần tương đương
  A/B: chưa có bằng chứng rõ ràng cho lợi ích tăng thêm từ encoder/conditioning.
- B có thêm 5 epoch pretrain encoder nhưng gần như bằng A. Pretraining đã thử
  chưa cho thấy lợi ích rõ ràng ở cùng ngân sách fine-tune.
- Trên B, tắt condition hoặc thay bằng condition của ảnh normal khác không làm
  AP holdout giảm rõ rệt; điểm còn nhích lên rất nhẹ.
- Condition có ảnh hưởng tới reconstruction, nhưng ảnh hưởng đó chưa chuyển
  thành khả năng phân biệt tumor/mô không có tumor tốt hơn.
- Spatial ImageNet grayscale làm normal reconstruction tốt hơn rõ rệt nhưng
  holdout AP giảm còn0.063110/0.065459,thấp hơn baseline/control/A/B. Không
  dùng reconstruction đẹp hơn hay noiseMSE thấp hơn để kết luận detection tốt.
- K4variance-normalized và pooled patch Mahalanobis cũng chưa cải thiện Pixel AP;
  averaging absolute chỉ tăng nhẹ. Normalized image AP có tín hiệu khác với
  localization,cần báo hai bài toán riêng;chi tiết mục15.
- Đây là kết quả một seed trên validation đã được xem nhiều lần. Không kết luận
  encoder hoàn toàn bị bỏ qua, conditioning luôn vô ích, hay diffusion không phù
  hợp cho MRI nói chung.

## 2. Dữ liệu và giới hạn đánh giá

| Nội dung | Thiết lập thực tế |
|---|---|
| Dataset | BMAD-derived Brain FLAIR 2D, split `patient-disjoint-v1` |
| Train | 7.500 lát cắt normal, không dùng tumor mask để training |
| Validation | 83 ảnh: 39 normal, 44 abnormal; 50 patient groups |
| Calibration | 40 ảnh, 25 groups |
| Holdout nội bộ validation | 43 ảnh, 25 groups, không trùng group với calibration |
| Chia calibration/holdout | `StratifiedGroupKFold(2, shuffle=True, random_state=17)`, fold đầu |
| Held-out test | 3.665 ảnh, chưa sử dụng trong chuỗi thí nghiệm này |
| Số seed | 1; seed sampling 17 |
| Ảnh đầu vào | Grayscale 240×240, chuẩn hóa `uint8 / 127.5 - 1`, padding 8 px mỗi cạnh thành 256×256 |
| Vùng tính pixel metrics | Crop gốc 240×240, gồm cả nền; không chỉ vùng mô |

“Normal” nghĩa là lát cắt không có tumor annotation, không đồng nghĩa bệnh nhân
khỏe mạnh. Dữ liệu là ảnh PNG đã render, không phải intensity MRI NIfTI gốc.
Split đã xử lý patient overlap; không so trực tiếp điểm này với điểm paper BMAD
trên split hoặc protocol khác. Xem [DATA_PROTOCOL.md](DATA_PROTOCOL.md).

Dataset private: `dacn-bmad-team/bmad-flair-patient-disjoint-v1`.
Revision cố định:
`350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.

## 3. Pipeline và ý nghĩa các phép đo

```text
Ảnh x trong [-1,1]
  -> thêm Gaussian noise tại t*=300 (index mạng 299)
  -> ancestral reverse 300 bước, dùng EMA checkpoint đã chọn
  -> reconstruction x_recon
  -> crop 240×240
  -> |x - x_recon| hoặc 1 - local SSIM(x, x_recon)
  -> anomaly map
  -> ROC-AUC, AP, Precision, Recall, F1/Dice và normal false positive rate
```

Diffusion có `T=1000`, beta schedule linear, dự đoán epsilon bằng U-Net. Noise
MSE là loss giữa noise được thêm vào và noise mạng dự đoán. Nó **không phải** AP
của tumor segmentation. Denoising MSE thấp không đảm bảo mô hình xóa tumor mà
vẫn giữ nguyên mô bình thường trong chuỗi reverse.

Absolute difference tính trực tiếp trên thang `[-1,1]`, không chia 2 trong anomaly
map. Local SSIM dùng cửa sổ Gaussian 11×11, sigma 1,5, reflect padding, data range
2 và hằng số `(0.01×2)^2`, `(0.03×2)^2`. Score là `1-SSIM`, không clipping; score
có thể lớn hơn 1. Cả hai map có score càng cao càng bất thường.

Pixel metrics gộp tất cả pixel trong crop theo global micro. Pixel AP đánh giá
xếp hạng score qua các ngưỡng, không cần chọn một threshold. Precision, Recall,
F1 và normal FPR phụ thuộc threshold. Mỗi phương pháp/score chọn threshold
max-F1 trên 40 ảnh calibration rồi giữ cố định trên 43 ảnh holdout; khi hòa chọn
threshold lớn nhất. Image score là trung bình top 576 pixel.

Normal FPR trong báo cáo là tỷ lệ pixel bị báo bất thường trên các ảnh có mask
rỗng; **không phải** tỷ lệ toàn bộ ảnh normal bị phân loại sai. F1/Dice micro
tương đương khi áp dụng trên cùng binary pixel prediction. Macro Dice trên các
ảnh có lesion được lưu riêng trong JSON gốc.

## 4. Thiết kế baseline, A và B

### Baseline Gaussian DDPM

U-Net một kênh, base channels 32, channel multipliers `[1,1,2,2,4,4]`,
2 residual blocks, attention resolutions `16,8`, 2 heads, dropout 0.
Checkpoint nguồn là **Gaussian EMA epoch 30**, không phải checkpoint AnoDDPM
khác chỉ có cùng tên file. Inference ancestral tại t*=300, batch 1.

### A: joint training, encoder khởi tạo mới

U-Net khởi tạo strict từ baseline EMA epoch 30. Encoder là ResNet-18 grayscale
chuẩn `[2,2,2,2]`, 11.170.240 tham số, global pooling ra vector 512 chiều.
`Linear(512,128) -> SiLU -> Linear(128,time_dim)` tạo condition. Linear cuối có
weight và bias khởi tạo bằng 0; condition được cộng vào timestep embedding dùng
trong các residual block của U-Net.

Fine-tune cả U-Net, encoder và projections. Encoder nhìn ảnh gốc một lần; vector
condition được giữ cố định trong toàn bộ reverse chain. Đây là **global feature
conditioning**, không phải spatial feature map, cross-attention hay patch mask.
Không dùng ImageNet/external pretrained encoder.

### B: pretrain encoder, rồi joint fine-tune độc lập

Pretrain cùng ResNet-18 trong **5 epoch trên 7.500 normal train slices**:

- Che ngẫu nhiên 32/64 ô lưới, kích thước patch 32×32; vùng che được điền 0 trên
  thang `[-1,1]`. Padding 8 px giữ nguyên.
- Bắt buộc có ít nhất một patch che giao foreground `input > -0.95`.
- Encoder + decoder upsampling 5 tầng học masked reconstruction; loss MSE chỉ
  tính pixel foreground bị che, chuẩn hóa theo từng ảnh.
- Foreground là heuristic intensity, không phải anatomical segmentation.
- AdamW LR `1e-4`, batch 8, seed 17, weight decay 0, clip norm 1.
- Chọn best encoder bằng masked MSE trên 39 normal validation, mask cố định qua
  seed 1.000.017. Đây là dense masked reconstruction đơn giản, không tái lập SparK.

Bỏ decoder và optimizer pretrain. Khởi tạo U-Net lại từ **cùng baseline EMA epoch
30**, chỉ nạp encoder đã pretrain. Projection khởi tạo như A, injection vẫn bằng
0, optimizer fine-tune mới. B không tiếp tục từ A hay từ smoke checkpoint.
Fine-tune cả encoder, projection và U-Net, không đóng băng encoder.

### Đối chứng unconditional fine-tune

Sau A/B, đã train U-Net thuần từ **cùng Gaussian EMA30**, không thêm encoder hay
projection. Fresh AdamW/LR1e-5, cùng seed17,batch8,EMA0.9992 và 9.380 updates như
A/B. Train epoch31..40, best normal-validation là **epoch36 = fine-tune epoch6**.
Đây là EMA warm-start control, không phải native continuation LR1e-4 giữ source
optimizer. Full-state checkpoint/resume và source checksum được kiểm tra riêng.
Validation sau đó dùng best36/cùng t300/protocol A/B, không chọn theo AP.

### Cấu hình fine-tune chung A/B và đối chứng U-Net

| Tham số | Giá trị |
|---|---|
| GPU / batch / seed | T4 / 8 / 17 |
| Epoch / optimizer updates | 10 / 9.380 mỗi thí nghiệm |
| U-Net LR | `1e-5` |
| Encoder / projection LR | `1e-4` / `1e-4` cho A/B; đối chứng U-Net không có hai thành phần này |
| Optimizer | AdamW, betas `(0.9,0.999)`, weight decay 0 |
| Gradient clipping | Norm 1 |
| EMA decay | 0,9992; copy BatchNorm buffers từ online model |
| Loss | Gaussian epsilon MSE, timestep sampling trên 0…999 |
| Chọn checkpoint | EMA noise MSE trên 39 normal-validation, noise seed 1.000.017 |
| Milestones | Fine-tune epoch 1, 5, 10; thêm best, last và epoch 0 để recovery |

Injection bằng 0 khiến gradient encoder ở update đầu bằng 0; gradient bắt đầu
đi vào encoder khi projection mở. Đây là hành vi dự kiến, không phải đóng băng
encoder. B có thêm **4.690 pretrain updates**, nên A/B có cùng ngân sách joint
fine-tune nhưng không có cùng tổng compute.

## 5. Training đã hoàn tất

| Giai đoạn | Epoch hoàn tất | Updates | Best epoch | Normal-validation loss | Thời gian train + normal-val | Peak allocated VRAM |
|---|---:|---:|---:|---:|---:|---:|
| A joint fine-tune | 10 | 9.380 | 6 | Epsilon MSE 0.007343507 | 4.449,683 s | 5.884.708.864 B |
| B encoder pretrain | 5 | 4.690 | 5 | Masked MSE 0.055597169 | 313,874 s | 910.336.512 B |
| B joint fine-tune | 10 | 9.380 | 6 | Epsilon MSE 0.007341489 | 4.436,785 s | 5.883.660.288 B |
| U-Net thuần fine-tune | 10 (31–40) | 9.380 | 36 = FT6 | Epsilon MSE 0.007353030 | 4.144,205 s | 5.465.677.824 B |

Masked MSE và epsilon MSE đo hai mục tiêu khác nhau, không so trực tiếp độ lớn.
Thời gian trên không gồm toàn bộ setup, checkpoint writes hay upload.
Best fine-tune epoch 6 được chọn bằng normal loss, không dựa vào Pixel AP.
Có best epoch sớm chưa đủ để khẳng định train đã hội tụ hoặc bị overfit.

## 6. Baseline so với A/B: 43 ảnh validation holdout

Threshold chỉ học từ calibration. FPR ghi dưới dạng tỷ lệ 0…1.

| Mô hình | Pixel score | ROC-AUC | Precision | Recall | F1/micro Dice | AP | Normal FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline EMA30 | Difference | 0.876854 | 0.119934 | 0.314702 | 0.173678 | 0.101066 | 0.058079 |
| U-Net thuần EMA36 | Difference | 0.879392 | 0.125207 | 0.317036 | 0.179518 | 0.104391 | 0.056382 |
| A EMA6 | Difference | 0.878721 | 0.126465 | 0.316731 | 0.180757 | 0.104645 | 0.056641 |
| B EMA6 | Difference | 0.878343 | 0.126436 | 0.310393 | 0.179681 | 0.104464 | 0.055678 |
| Baseline EMA30 | Local 1-SSIM | 0.888870 | 0.110476 | 0.257712 | 0.154654 | 0.102402 | 0.055234 |
| U-Net thuần EMA36 | Local 1-SSIM | 0.889144 | 0.113874 | 0.260477 | 0.158469 | 0.104004 | 0.054682 |
| A EMA6 | Local 1-SSIM | 0.888060 | 0.115090 | 0.252128 | 0.158039 | 0.104247 | 0.053224 |
| B EMA6 | Local 1-SSIM | 0.887708 | 0.115097 | 0.251140 | 0.157851 | 0.104084 | 0.053144 |

A/B và đối chứng đều chỉ tăng AP nhẹ so với baseline. B không vượt A trên holdout.
A chỉ hơn đối chứng AP 0.000253351 (difference) và 0.000243256 (SSIM); B hơn
0.000072805 và 0.000079918 tương ứng. Đối chứng cho thấy chỉ thêm fine-tune U-Net
đã đạt mức tăng gần tương đương. Không suy ra phần trăm đóng góp nhân quả từ
những chênh lệch nhỏ này; một seed chưa đủ xác nhận lợi ích riêng của conditioning.

## 7. Kiểm tra condition trên checkpoint B

Giữ nguyên **B EMA fine-tune epoch 6**, t*=300, ancestral 300 bước, batch 1,
sampling seed 17 theo từng sample và cùng ảnh validation. Ba trường hợp:

| Trường hợp | Thao tác |
|---|---|
| Self / đúng ảnh | Encoder nhận chính ảnh cần đánh giá; tái sử dụng cache B đã xác thực |
| Zero | Bypass toàn bộ condition injection, gồm cả learned bias; không chỉ đặt feature trước projection bằng 0 |
| Normal donor | Encoder nhận một ảnh normal khác từ train; ảnh được thêm noise/reconstruct vẫn là ảnh đánh giá |

Donor được ánh xạ cố định bằng UUID-derived seed 77 modulo 7.500, không chọn
theo mask hoặc kết quả. Không đưa tumor mask vào model; mask chỉ dùng đánh giá.
Không có training trong phép thử này. Zero vẫn dùng U-Net đã fine-tune của B,
**không phải** quay lại baseline epoch 30.

Trước lượt đầy đủ, smoke trên T4 tái chạy 1 normal + 1 abnormal qua 300 bước:
reconstruction khớp cache cũ hoàn toàn, max absolute errors `[0,0]`; initial noise
khớp chính xác. Các control zero/donor chạy hữu hạn 3 bước. Lượt đầy đủ kiểm tra
lại preflight và exact alignment của input, mask, label, initial noisy image.
Hai arm mới đều hoàn tất **83/83 ảnh**.

### Kết quả pixel trên 43 ảnh holdout

| Condition | Score | ROC-AUC | Precision | Recall | F1/micro Dice | AP | Normal FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| Self | Difference | 0.878343 | 0.126436 | 0.310393 | 0.179681 | 0.104464 | 0.055678 |
| Zero | Difference | 0.879854 | 0.125348 | 0.312906 | 0.178993 | 0.104579 | 0.055271 |
| Normal donor | Difference | 0.880107 | 0.126232 | 0.319065 | 0.180896 | 0.105132 | 0.056058 |
| Self | Local 1-SSIM | 0.887708 | 0.115097 | 0.251140 | 0.157851 | 0.104084 | 0.053144 |
| Zero | Local 1-SSIM | 0.890025 | 0.114561 | 0.253636 | 0.157832 | 0.104416 | 0.052590 |
| Normal donor | Local 1-SSIM | 0.889813 | 0.115234 | 0.254911 | 0.158719 | 0.104699 | 0.052691 |

AP thay đổi rất nhỏ; condition đúng ảnh không chiếm ưu thế. Threshold được fit
riêng trên calibration cho từng arm, nên FPR/F1 giữa các arm bao gồm tác động của
calibration; đây không phải phép so với một threshold số học chung.

### Image-level trên cùng 43 ảnh holdout

| Mô hình / condition | Difference ROC-AUC | Difference AP | SSIM ROC-AUC | SSIM AP |
|---|---:|---:|---:|---:|
| Baseline | 0.263158 | 0.430006 | 0.372807 | 0.505258 |
| U-Net thuần EMA36 | 0.243421 | 0.422203 | 0.405702 | 0.523079 |
| A | 0.221491 | 0.415001 | 0.412281 | 0.527520 |
| B self | 0.228070 | 0.417369 | 0.414474 | 0.528226 |
| B zero | 0.254386 | 0.426584 | 0.405702 | 0.523079 |
| B normal donor | 0.243421 | 0.421661 | 0.405702 | 0.523708 |

Với các threshold image-level đã chọn, mọi hàng/score có Precision 0.558140,
Recall 1 và F1 0.716418: toàn bộ holdout được dự đoán positive. F1 cao hơn pixel
F1 ở đây không có nghĩa mô hình phân loại ảnh tốt.

### Kết quả mô tả trên toàn bộ 83 ảnh

| Mô hình / condition | Difference AP | Difference F1 | SSIM AP | SSIM F1 |
|---|---:|---:|---:|---:|
| A | 0.104001 | 0.191032 | 0.102805 | 0.169561 |
| U-Net thuần EMA36 | 0.103114 | 0.189007 | 0.102278 | 0.169098 |
| B self | 0.104162 | 0.191531 | 0.103002 | 0.169885 |
| B zero | 0.102956 | 0.188329 | 0.102292 | 0.169577 |
| B normal donor | 0.103748 | 0.189807 | 0.102624 | 0.169933 |

Threshold F1 của bảng này tối ưu trên chính 83 ảnh, chỉ để mô tả. Trên toàn 83
ảnh, self có AP nhỉnh hơn zero/donor, khác với holdout. Điều này củng cố rằng
chênh lệch nhỏ và chưa ổn định; không chọn kết luận chỉ từ bảng thuận lợi hơn.

## 8. Reconstruction thay đổi thế nào?

### Chất lượng trên 39 ảnh normal

MAE dưới đây dùng thang `[0,1]`; SSIM trung bình chỉ trong foreground heuristic
`input > -0.95` của crop 240×240. Chỉ số này không cùng miền tổng hợp với pixel
AP/FPR trên cả crop.

| Mô hình | Normal foreground MAE ↓ | Normal foreground SSIM ↑ |
|---|---:|---:|
| Baseline | 0.082674084 | 0.292438059 |
| U-Net thuần EMA36 | 0.082240741 | 0.295440689 |
| A | 0.081020562 | 0.302187646 |
| B | 0.081059648 | 0.302334383 |

A/B cải thiện nhẹ chất lượng normal so với baseline; B gần như bằng A.
Đối chứng cũng cải thiện nhẹ, còn kém A/B về quality normal. Reconstruction normal
tốt hơn không tự động dẫn đến tumor AP cao hơn.

Trên 44 ảnh abnormal, macro mean score trong lesion / foreground ngoài lesion:

| Mô hình | Difference lesion / nonlesion | 1-SSIM lesion / nonlesion |
|---|---|---|
| Baseline | 0.229169273 / 0.150152315 | 0.798852441 / 0.662142919 |
| U-Net thuần EMA36 | 0.232020910 / 0.147787429 | 0.797625873 / 0.653556285 |
| A | 0.230797590 / 0.144901834 | 0.791948763 / 0.642770893 |
| B | 0.230870375 / 0.144413748 | 0.791558521 / 0.640803810 |

Các trung bình này không thể hiện toàn bộ phân bố/overlap của score. Có lesion
mean cao hơn nonlesion mean vẫn chưa đảm bảo Precision/AP tốt.

### Thay đổi so với B self khi can thiệp condition

Tính `|reconstruction_can_thiep - reconstruction_self| / 2` trên crop 240×240.

| Can thiệp | Mean tất cả ảnh | Mean normal | Mean abnormal | Max pixel |
|---|---:|---:|---:|---:|
| Zero | 0.001952624 | 0.001514973 | 0.002340543 | 0.535652041 |
| Normal donor | 0.001710228 | 0.001585713 | 0.001820594 | 0.506980836 |

Mean nhỏ nhưng có pixel thay đổi lớn. Các mean gồm cả nền, không chỉ mô não;
không dùng chúng để kết luận mọi vùng mô đều bất biến. Dữ liệu cho thấy condition
có tác động, nhưng chưa tạo lợi ích segmentation rõ ràng.

## 9. Control brightness và chi phí

Brightness-only là control riêng, không phải phương pháp diffusion được cải tiến.
Trên 43 ảnh holdout: pixel ROC-AUC 0.950197, AP 0.389506, Precision 0.161469,
Recall 0.788577, F1 0.268051, normal FPR 0.134935. Control này có AP cao hơn các
diffusion score hiện tại nhưng cũng FPR cao hơn tại threshold được chọn.
Không trộn brightness vào score rồi tuyên bố diffusion tự cải thiện.

| Lượt evaluation | Thời gian loop được ghi | Peak allocated VRAM |
|---|---:|---:|
| B self, 83 ảnh | 806,260 s | 147.551.744 B |
| U-Net thuần EMA36, 83 ảnh | 1.075,185 s | 101.438.464 B |
| B zero, 83 ảnh | 813,079 s | 146.502.656 B |
| B normal donor, 83 ảnh | 818,889 s | 146.765.312 B |

Toàn lượt can thiệp condition mất 1.739,939 s (khoảng 29 phút), gồm preflight,
scoring và tạo artifact trước upload. Loop timing gồm phần xử lý trong evaluator,
không phải benchmark riêng U-Net latency. VRAM là allocated peak được đo trong
evaluator, không phải toàn bộ memory GPU hiển thị bởi hệ thống.

## 10. Diễn giải: biết gì và chưa biết gì?

**Đã quan sát:** A/B tương đương về AP; pretraining B chưa tạo cải thiện rõ;
đối chứng không encoder cùng ngân sách cho AP gần A/B; đúng condition không
thắng zero/donor trên holdout; reconstruction có thay đổi.

**Giả thuyết cần kiểm chứng:** global pooled condition có thể cung cấp ít thông
tin không gian; U-Net có thể chủ yếu dựa vào noisy input/timestep; encoder nhìn
ảnh gốc có thể truyền cả thông tin tumor. Các giả thuyết này chưa được tách riêng
bằng thí nghiệm hiện tại.

**Chưa được kết luận:** code hoàn toàn không có lỗi; encoder luôn bị bỏ qua;
mọi conditioning đều thất bại; train dài hơn chắc chắn không giúp; diffusion nói
chung kém cho MRI; best epoch 6 đồng nghĩa hội tụ; AP thấp chỉ do lesion nhỏ.

Lesion chiếm ít pixel có thể làm bài toán Precision/AP khó, nhưng chưa có phân
tích prevalence đủ để quy toàn bộ điểm thấp cho mất cân bằng. AP phụ thuộc khả
năng xếp hạng lesion cao hơn nền/mô bình thường; không suy ra AP từ noise MSE.

Các giới hạn chính: một seed, validation đã được dùng nhiều lần, 83 ảnh/50groups
khá nhỏ, donor chỉ một lựa chọn mỗi ảnh, không có khoảng tin cậy, matched control
mới một lần chạy, pretrain encoder mới 5 epoch và không có kết quả test.

## 11. Hướng tiếp theo đề xuất — chưa chạy

Ưu tiên giữ DDPM/A/B làm baseline đã ghi nhận, không tăng epoch ngay chỉ dựa vào
noise MSE. Trước khi mua thêm compute, kiểm tra mức đóng góp của conditioning:
độ lớn/độ biến thiên injected embedding so với timestep embedding, và thay đổi
epsilon prediction khi tắt/đổi condition tại cùng noisy input.

Nếu tiếp tục nghiên cứu conditioning, chỉ thay một yếu tố mỗi thí nghiệm, ví dụ
spatial condition thay global pooling hoặc đưa feature vào nhiều resolution.
Đây là hướng thử mới, không phải phương án đã chứng minh tăng AP. Cần xác định
trước giả thuyết, ngân sách updates và tiêu chí đánh giá, rồi chạy pilot nhỏ.

Matched control đã có: không nên tiếp tục dùng mức tăng so với baseline30 làm
bằng chứng riêng cho encoder. Khi có cải thiện đủ lớn, lặp control/A/B nhiều seed và dùng
patient-group bootstrap để báo độ bất định. Chốt protocol trước khi đánh giá
held-out test. Không tiếp tục tối ưu trên test.

## 12. Artifact, checkpoint và khả năng tái lập

Các repo HF dưới đây private; cần quyền nhóm. Không đưa checkpoint, MRI, mask,
cache ảnh hay sample identifiers vào Git. Manifest và source snapshot ghi cấu
hình/code chính xác, kể cả working tree dirty.

| Artifact | Run / evaluation ID | HF revision |
|---|---|---|
| Gaussian nguồn | `ddpm-partial-t4-b8-e30-cont-20261102T034644Z`, `ddpm_epoch_030.pt` | Xem manifest nguồn trong RESULT_TRACKING |
| A training | `feature-A-joint-t4-b8-e10-20261104T174552Z` | `10833f849f9a93bbfd5364336a2b920350b278b1` |
| A validation | `feature-A-joint-t4-b8-e10-20261104T174552Z-val` | `56f13bc33a470858475058cb961b1e904f680946` |
| B encoder | `feature-B-pretrain5-joint10-t4-20261105T053132Z-enc` | `626ec3ec07acdfc38274300c374045d09212f1c7` |
| B joint training | `feature-B-pretrain5-joint10-t4-20261105T053132Z` | `352d96a26a3b9b056274bbeb7bc5864eb8e25562` |
| B validation | `feature-B-pretrain5-joint10-t4-20261105T053132Z-val` | `348c47aa2528fcd48b1dc62f378e7078c7592571` |
| Unconditional control training | `ddpm-uncond-control-t4-b8-e40-20261105T082502Z` | `375bc6ba5578f5c27953f3306cecdabb531f073b` |
| Unconditional control validation | `ddpm-uncond-val-best36-t300-20261105T094934Z` | `38491cba20168ac41a038e98cfaffe1bc3fb51b5` |
| Condition smoke | `B-condition-smoke-t4-20261105T072319Z` | `92df7e4ef60628e019d7a2b85146b393f2d28450` |
| Condition full83 | `B-condition-full83-t4-20261105T072451Z` | `c22ee10da5f0193e99d3f3c097ab37e120cb7737` |

[A evaluation artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/56f13bc33a470858475058cb961b1e904f680946/evaluations/feature-A-joint-t4-b8-e10-20261104T174552Z-val),
[B evaluation artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/348c47aa2528fcd48b1dc62f378e7078c7592571/evaluations/feature-B-pretrain5-joint10-t4-20261105T053132Z-val),
[condition intervention artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/c22ee10da5f0193e99d3f3c097ab37e120cb7737/evaluations/B-condition-full83-t4-20261105T072451Z).

[Unconditional control validation artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/38491cba20168ac41a038e98cfaffe1bc3fb51b5/evaluations/ddpm-uncond-val-best36-t300-20261105T094934Z).

SHA256 checkpoint và JSON kết quả đã đối chiếu:

| Đối tượng | SHA256 |
|---|---|
| Gaussian EMA30 source file | `d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946` |
| A best EMA6 checkpoint file | `0a5939b2b9e1df31f635c411c67fb01ef46df42ed5dad9599992948d6b0ddd29` |
| B best EMA6 checkpoint file | `0c4f21705830f5dd6054174fc6b2c1591263779413737ce87b4407e3ca8b01b7` |
| U-Net control best EMA36 checkpoint file | `350791c48fd99babc38c329af5644a4c53a70f6628b6d751e85e3a6cab8cf0f7` |
| A results JSON | `a8e57ed824c9e6a00404d82296f15fb208eabb9d8d911c06fcd983dfb8b5a962` |
| B results JSON | `17cea7ce87afc15332835ffdafa514b2758883d92510339f86bb45b521ac247d` |
| Condition results JSON | `17300018158205e715e2c186c6fa1ea713fe13abeebbdaec21ff2300f08df0b0` |
| Unconditional control results JSON | `babbd335d6a9c40e64d23b2e7dded4dd8f195e3522669586713488ced3f62c7e` |

Code base revision: `7134b105c0c4d4fb620715908f100103b1d0b0ab`, có thay đổi chưa
commit tại thời điểm chạy; chỉ Git SHA chưa đủ tái lập. Package source snapshot:

- A: `72e43a42c623d123c29ec5ebb707de3de3b22c63a859148e662a7e4ceaef9ee1`.
- B: `bcc65e702e23a2ca97179cf066c7d38c9a40d1bd1c3f3c58b8f726c9d67aadf2`.
- Condition diagnostic: `b5e4fccffe407d80914106884b49a41567a1fab134bfa025a89f77799d1f41c0`.
- Unconditional training: `03f0be1c2c2991db35952c492372b51e9cbad3afcc05307a3b60e80c0034ffd2`.
- Unconditional validation: `68092ef7fe5e470d833a1ce0909c84011438e7b41414e91f1761f5dc4968c93a`.

A/B lưu `feature_condition_best.pt`, `feature_condition_last.pt` và milestones.
Encoder pretrain lưu `encoder_pretrain_*.pt`. Checkpoint chứa method/config khác
baseline; dùng `load_feature_checkpoint`, không nạp vào unconditional loader.
Resume chỉ cùng run, source snapshot, config và dataset; không dùng tên file
hoặc số epoch để suy ra checkpoint tương đương.

Chi tiết implementation/config và lịch sử kiểm thử:

- [FEATURE_CONDITION_A.md](FEATURE_CONDITION_A.md), [FEATURE_CONDITION_B.md](FEATURE_CONDITION_B.md).
- [UNCONDITIONAL_FINETUNE_CONTROL.md](UNCONDITIONAL_FINETUNE_CONTROL.md).
- [RESEARCH.md](RESEARCH.md), [RESULT_TRACKING.md](RESULT_TRACKING.md).
- A implementation: 119 tests passed; sau B: 124 tests; sau intervention: 129
  tests. Tests kiểm tra zero injection, gradient, EMA/BatchNorm, checkpoint
  transfer/resume, normal-only input, calibration, paired noise và donor control.
- Unconditional fine-tune:133tests; shared unconditional validation:135tests,
  gồm kiểm tra helper khớp original chain và resume prefix đúng selected epoch36.
- Smoke thật trên T4 đã qua trước các lượt đầy đủ. Unit tests không thay thế
  bằng chứng cải thiện AP hoặc chứng minh toàn bộ code không còn lỗi.
- Việc viết tài liệu tổng hợp này không chạy thêm training/inference và không
  truy cập held-out test.

## 13. Đoạn kết luận có thể dùng cho báo cáo

> Trên tập BMAD-derived Brain FLAIR 2D với patient-disjoint split, bổ sung global
> ResNet-18 conditioning vào DDPM và fine-tune 10 epoch chỉ cải thiện nhẹ Pixel AP
> so với checkpoint Gaussian ban đầu. Đối chứng U-Net thuần fine-tune cùng ngân
> sách cho AP gần tương đương A/B, chưa cho thấy lợi ích riêng rõ ràng từ encoder.
> Pretraining encoder bằng masked
> reconstruction trong 5 epoch chưa đem lại lợi ích rõ ràng so với encoder khởi
> tạo ngẫu nhiên ở cùng ngân sách joint fine-tuning. Can thiệp inference bằng
> cách tắt condition hoặc thay condition bằng ảnh normal khác cho kết quả AP
> holdout gần tương đương, cho thấy condition đúng ảnh chưa cung cấp lợi ích
> segmentation rõ rệt trong cấu hình này. Condition vẫn làm thay đổi ảnh tái tạo;
> do đó không thể kết luận mô hình hoàn toàn bỏ qua condition. Các kết quả mới
> mang tính thăm dò trên một seed và validation đã được khảo sát nhiều lần;
> cần lặp matched control/A/B trên nhiều seed và đánh giá độc lập trước khi
> khẳng định hiệu quả hoặc hạn chế tổng quát của phương pháp.

## 14. Bổ sung: spatial ImageNet grayscale — kết quả âm

Thử thêm ResNet18 ImageNet,conv1=meanRGB->1channel,không pooling,4spatial maps
và zero1x1 fusion vào U-Net EMA30. Giữ MRI[-1,1],không repeat RGB hayRGBnorm.
Joint10epochs/9380updates,T4/batch8;encoderLR1e-5,U-Net1e-5,fusion1e-4.
Best theo39normal-val là EMAepoch10,noiseMSE0.004498155053. Đánh giá đủ83
validation,t300,cùng40calibration/43group holdout vàscorers như các bảng trên.

| Mô hình | Difference AP,holdout43 | Local SSIM AP,holdout43 |
|---|---:|---:|
| Baseline30 |0.101066|0.102402|
| U-Net thuần +10epoch |0.104391|0.104004|
| A global |0.104645|0.104247|
| B global normal-pretrained |0.104464|0.104084|
| Spatial ImageNet grayscale |0.063110|0.065459|

Spatial SSIMpixelF10.136377,normal pixelFPR26.0876% tại threshold học trên
calibration;control SSIMF10.158469,FPR5.4682%. Normal foreground MAE/SSIM
tốt lên0.039324712/0.688031178 so control0.082240741/0.295440689.
Tuy nhiên abnormal44mean1-SSIM lesion0.276705021 vànonlesion0.271836988 gần
nhau. Gợi ý truyền spatial feature của ảnh gốc có thể giúp giữ lại/tái tạo cả
lesion,làm mất độ phân biệt residual;chưa chứng minh cơ chế từ những mean này.
AP giảm với cảdifference/SSIM nên không chỉ do chọn threshold.

**Không nên tăng epoch cấu hình này chỉ dựa vào noiseMSE tốt hơn.** Negative
result không chứng minh grayscale/ImageNet/spatial nói chung sai;trial đổi cả
pretraining/fusion/LR và mới một seed trên validation đã khảo sát. Không có
held-out-test result. Cần ảnh/ablation để kiểm chứng giả thuyết giữ lại tổn thương.

Chi tiết metric,checkpoint,source/provenance ở
[SPATIAL_IMAGENET_GRAYSCALE.md](SPATIAL_IMAGENET_GRAYSCALE.md) và
[RESULT_TRACKING.md](RESULT_TRACKING.md). Private eval revision
`a6e21e5c2553e683fda3e51496fdc52a31662b60`,bestcheckpoint SHA
`6a4103bb14937421fc79e595fb7fdd7d460774dc68e2db1ef3e8c75b04d307cb`,
resultSHA`0c9b75267a8d6f26a54c071ab26e263ea3fc0b292a438ff112acd2a15dd5299f`
khớp manifest. Kết quả này bổ sung,các bảng A/B/global ở trên giữ đúng phạm vi cũ.

## 15. Bổ sung: uncertainty-aware scoring K4 — chưa cải thiện localization

Giữ Gaussian original EMA30,t300,seedbases17/1017/2017/3017. Reuse draw1,
thêm249reconstruction,khôngtraining. Sáu score fixed trước chạy;lambda và
covariance chỉ fit20normal calibration,thresholds40calibration,đo43group holdout.

| Score | Holdout Pixel AP | Pixel F1 | Normal pixel FPR | Holdout Image AP |
|---|---:|---:|---:|---:|
| Single absolute |0.101066|0.173678|5.8079%|0.430006|
| Single SSIM |0.102402|0.154654|5.5234%|0.505258|
| Mean4 absolute |0.103810|0.173276|5.2835%|0.453941|
| Mean4 SSIM |0.099472|0.141579|3.8697%|0.539785|
| Variance-normalized |0.045729|0.123433|4.0329%|0.720711|
| Patch Mahalanobis |0.077657|0.148199|13.7883%|0.542902|

Variance-normalized=r²/(v+lambda);Mahalanobis là signed3x3patch9D so với normal
calibration residual distribution,covariance shrinkage0.1. Không phải per-pixel
9Dpredictive covariance suy ra từK4. Hai phương án cụ thể này không vượt
averaging controls cho Pixel AP. Mean4absolute tăng nhỏ,không đủ khẳng định
significance hay hợp lý hóa tăngK không giới hạn.

Mean4normal reconstruction SSIM0.408378 so single0.292438,nhưng improvement
reconstruction vẫn không đảm bảo localization. Normalized score mean lesion
5.556999 vsnonlesionforeground11.326831 cho thấy separation kém trong cấu hình
này. ImageAP0.720711 là kết quả phân loại ảnh,không phải segmentation;brightness
control imageAP0.770309 vẫn cao hơn trên cùngholdout. Giảm normal FPR cũng phải
đọc cùng Recall,không tuyên bố cải thiện tại matched sensitivity.

Các bước nghiên cứu tiếp chỉ là đề xuất:phân tích phân bố residual/variance theo
vùng mô và mục tiêu image-vs-pixel,tránh tiếp tục tăngepoch/K hoặc tuneholdout
để đuổi điểm. Không kết luận mọi uncertainty scoring/Mahalanobis đều thất bại;
one model/ensemble,validation đã được xem nhiều lần,held-out test giữ kín.
Nguồn đầy đủ ở [UNCERTAINTY_SCORING.md](UNCERTAINTY_SCORING.md),
private revisionaf0bf4006db4c948292b9bc50748073e1b421bb4,resultSHA
fc12e2093045f9b6f7c0850e683a7c0258c05529d6a92580419101bf77f8ce16
đã đối chiếu manifest. Đây là inference-only,không có training mới.

## 16. Normal memory bank fusion: full validation đã hoàn tất

Thí nghiệm inference riêng,không đưa feature vào U-Net:ResNet18 ImageNet chính
thức đóng băng,grayscale mean-conv1,layer2/layer3 tạo descriptor384D. Dùng7500
normal train slices,foreground heuristic,random seed17 giữ10000vectors từ
1929258eligible descriptors;không phải PatchCore coreset. Tái dùng GaussianEMA30
ancestral t300 cache. Median/IQR fit20normal calibration,alpha0.5 cố định cho
hai fusion arms,threshold fit40calibration và đóng băng khi đánh giá43holdout.

| Method | Pixel AP | Micro Dice | Normal FPR | Image AP |
|---|---:|---:|---:|---:|
| DDPM absolute | 0.101066 | 0.173678 | 0.058079 | 0.430006 |
| DDPM SSIM | 0.102402 | 0.154654 | 0.055234 | 0.505258 |
| Memory only | 0.064212 | 0.139943 | 0.134794 | 0.822745 |
| Absolute + memory | 0.102374 | 0.180837 | 0.061204 | 0.434041 |
| SSIM + memory | 0.087033 | 0.151198 | 0.079000 | 0.638276 |
| Brightness | 0.389506 | 0.268051 | 0.134935 | 0.770309 |

Absolute fusion tăngAP rất ít(+0.001308),SSIM fusion giảmAP(-0.015370). Chưa có
cải thiện segmentation đáng kể;brightness còn mạnh hơn nhiều. Absolute fusion
tăngRecall/F1 nhưng cũng tăngnormal FPR,không kết luận giảm false positive tại
cùng độ nhạy. Các operating points fit theo calibration recall không tạo cùng
recall thực tế trên holdout,không được tune lại để khẳng định matched sensitivity.

Memory-only có imageAP0.822745 và imageAUROC0.703947,cao hơn brightness trong
fold này;đây là tín hiệu image ranking,không chứng minh định vị tumor tốt. Tại
threshold image maxF1 chọn ở calibration,nó dự đoán toàn bộ holdout positive:
precision0.558140,recall1,F1=0.716418. Fusion cũng chưa vượt memory-only ở
imageAP. Không đồng nhất image-level detection với pixel-level segmentation.

Full run `memory-fusion-full83-t4-20261106T055134Z` completed/uploaded private
revision `01c5995f6ec19092df7bb4ac302861399f2de868`. Root resultSHA
`bfeb1f4573a6a0cef68cc68df6e8d86f0a3d7824080674cfb39e39547faaf476`
đã đối chiếu manifest. Tổng145.25s trước upload,không train mới,không dùng final
test hay alpha search. One model/bank seed trên validation đã xem nhiều lần;
kết quả exploratory. Xem [DDPM_MEMORY_FUSION.md](DDPM_MEMORY_FUSION.md) và
[RESULT_TRACKING.md](RESULT_TRACKING.md) cho protocol và đầy đủ chỉ số.

## 17. LPIPS pseudo-mask -> re-sampling: có tiến bộ so với DDPM baseline

GaussianEMA30 không train lại,tái dùng t300cache. SpatialLPIPS SqueezeNet kết hợp
residual chuẩn hóa95percentile tạo pseudo-mask;threshold fit20normalcalibration,
dilation3,reverse50steps x5repeats,seed17. Tất cả83validation hoàn tất,đánh giá
43group holdout sau40calibration,không dùng finaltest. Đây là AutoDDPM adaptation,
không tái lập chính xác paper hoặc cô lập LPIPS khỏi mọi thay đổi của pilot cũ.

| Phương án | Pixel AP | Micro Dice | Normal FPR | Image AP |
|---|---:|---:|---:|---:|
| Initial DDPM absolute | 0.101066 | 0.173678 | 0.058079 | 0.430006 |
| Initial DDPM SSIM | 0.102402 | 0.154654 | 0.055234 | 0.505258 |
| Initial LPIPS x normalized residual | 0.204298 | 0.274354 | 0.010727 | 0.887665 |
| Chỉ stitch,absolute | 0.118781 | 0.204641 | 0.047041 | 0.481000 |
| Re-sampling absolute | 0.129407 | 0.227416 | 0.044342 | 0.534048 |
| Re-sampling SSIM | 0.149446 | 0.223469 | 0.045201 | 0.581114 |
| Final residual x initialheatmap (primary) | 0.154785 | 0.226256 | 0.026640 | 0.814398 |
| Brightness | 0.389506 | 0.268051 | 0.134935 | 0.770309 |

Có tăngAP so baseline,nhưng **heatmap trước re-sampling tốt hơn các score sau
re-sampling**. Khi giữ scoreabsolute để so với naive stitch,re-sampling tăngAP
0.010626;chưa có bằng chứng rằng thêm chuỗi inpainting là lựa chọn tốt nhất.
Heatmapcontrol đã được đăng ký trước,không đổi primaryscore theoholdout.
Chưa đo LPIPS-only nên không quy tất cả cải thiện riêng cho perceptual distance.

Pseudo-mask holdout lesionrecall83.7541%,precision13.3407%;còn16.2459% lesion
pixels ngoài vùng sửa. Normal ROI sau dilation11.7060%;không nhầm với target5%
trước dilation hoặc FPR ởscorethreshold. Outside-mask error0 là điều kiện stitch,
không chứng minh tái tạo mô lành tốt hơn. Các FPR ởthresholdcalibration córecall
khác nhau;không tuyên bố cải thiện ởmatched sensitivity. Brightness vẫn tốt hơn
vềpixelAP,LPIPSheatmapimageAP tốt hơn nhưng đây là nhiệm vụ khác.

Run `autoddpm-lpips-full83-t4-20261106T061833Z`,private revision
`bab00576bb8271ddb191c8fcd13840eff40f1919`,root resultSHA
`6fb277491308c63b2cc52fa614fc912486266a9c3322f7f099aff850cb51a7e0`
đã đối chiếu manifest,total715.98s trước upload. One model/seed,validation đã
được xem nhiều lần nên chưa có independent confirmation. Hướng tiếp theo hợp lý
là kiểm chứng khả năng bổ sung của **LPIPS x residual scoring** trước khi tăng
chi phí re-sampling;đây là đề xuất,chưa có run mới. Xem
[AUTODDPM_LPIPS.md](AUTODDPM_LPIPS.md) vàRESULT_TRACKING cho đủmetrics/provenance.

## 18. Tách LPIPS-only: đóng góp scoring rõ hơn đóng góp re-sampling

Audit theo lộ trình người dùng đã hoàn tất trên83cặp ảnh đã cache,không train,
không chạy lại DDPM. Giữ nguyên LPIPS state,240crop và40/43group partition.
Tích hai term tái lập heatmap cũ sai khác0,pseudo-mask cũng khớp hoàn toàn.

| Score | Pixel AP | Micro Dice | Recall | Normal FPR | Image AP |
|---|---:|---:|---:|---:|---:|
| Gaussian absolute | 0.101066 | 0.173678 | 0.314702 | 0.058079 | 0.430006 |
| Normalized absolute only | 0.102264 | 0.167640 | 0.299978 | 0.057595 | 0.558140 |
| LPIPS only | 0.266899 | 0.339668 | 0.422469 | 0.007686 | 0.868242 |
| LPIPS x normalized absolute | 0.204298 | 0.274354 | 0.280623 | 0.010727 | 0.887665 |
| Weighted final re-sampled residual | 0.154785 | 0.226256 | 0.290408 | 0.026640 | 0.814398 |
| Brightness | 0.389506 | 0.268051 | 0.788577 | 0.134935 | 0.770309 |

Trong checkpoint/cohort này,nhân thêm normalized residual làm giảm pixelAP so
với LPIPS-only. CalibrationAP cũng theo cùng thứ tựLPIPS0.316582 >product0.285192
>weightedfinal0.196476. Đây là bằng chứng về đóng góp scoring trên ảnh tái tạo
cố định,không chứng minh một cải tiến mask-guided diffusion độc lập đã thành công.
LPIPS-onlyimageAP thấp hơn product;brightnesspixelAP vẫn cao hơn. HigherDice/
lowernormalFPR tại thresholdcalibration córecall khác nhau,không phải kết luận
matched sensitivity. Normalized residual cóimageAUROC0.5 vì cliptop5percent
thành1 trong khi imageaggregation chỉ lấytop1percent:scoreảnh bị bão hòa1.

Pseudo-mask bỏ sót9048/55694lesionpixels(16.2459percent). Boundary<=3pixels
missrate18.2356percent,core>3pixels15.5972percent;72.4027percent tổngmiss nằm
trongcore do diện tích lớn hơn. Vì thế không thể chỉ giải thích bằng thiếu dilation
ởbiên. Mask bỏ sót hoàn toàn3/29components1–25pixels và3/6components26–100pixels;
đó là thành phần liên thông2D,không phải số tumor hay bệnh nhân. LPIPS final
threshold vẫn bỏ sót20/29components nhỏ nhất,dù cóAP tốt hơn.

Pseudo-maskFP303007pixels:background5.7504percent,foregroundboundary17.5772percent,
interiorhighgradient1.8524percent,interiorother74.8201percent. Highgradient córate
cao trên diện tích nhỏ;đa sốFPcount không nằm ởproxy đó. Các vùng là thống kê
ảnh,không có nhãn giải phẫu để kết luận não thất/vỏ não. Bốn panel kiểm tra được
lưu private cho nhóm;chưa được review giải phẫu trong lượt check aggregate.

Run`perceptual-audit-full83-t4-20261106T065328Z`,private revision
`3b82d8b2bbfa62d887dbb2f687a790905c351113`,rootresultSHA
`224e17448d671bf96de619efe4196a1d0a2992bc07e29915a7c0499df63ba0fa`
đã đối chiếu manifest,total25.922s trước upload. Không chọn thiết kếmask mới bằng
thứ hạng trênholdout đã xem. Cần chốt tiêu chí development/calibration và method
trước xác nhận độc lập;finaltest vẫn kín. Xem
[PERCEPTUAL_MASK_ROADMAP.md](PERCEPTUAL_MASK_ROADMAP.md) cho số liệu và lộ trình.

## 19. DDIM20+30 một pass: ablation dùng LPIPS trên ảnh final

Thực hiện đúngdocs/testing_Lpips.md:giữGaussianEMA30,phase1DDIM20t300,
phase2DDIM30t50,eta0,mộtpass,tổng50U-Netcalls/ảnh. Dùng cùnginitial/final vàmask
cho bốnscoringarms;LPIPS vànormalizedresidual củafinal đều tính lại với ảnhfinal,
không thay bằnginitialheatmap nhưscoreprimary thí nghiệmAutoDDPM cũ.
Full83đã hoàn tất,40calibration/43groupholdout,thresholdcalibration-only.

| Required method | Pixel AP | Micro Dice | Normal FPR |
|---|---:|---:|---:|
| MatchedDDPM-DDIM20 + L1 | 0.090936 | 0.156053 | 0.074325 |
| AutoDDPM-DDIM20+30 + L1 | 0.099569 | 0.200077 | 0.069091 |
| AutoDDPM-DDIM20+30 + finalLPIPS | 0.312330 | 0.371215 | 0.005772 |
| AutoDDPM-DDIM20+30 + finalLPIPS x finalresidual | 0.196812 | 0.272453 | 0.006499 |

MatchedinitialDDIM20LPIPS-onlyAP0.157876/Dice0.246871 ->finalLPIPS0.312330/
0.371215;calibrationAP0.238079->0.312952. FinalLPIPSimageAUROC0.914474,
imageAP0.944386. Đây là tín hiệu tích cực cho pipeline cóphase2 khi giữ cùng
LPIPSscoring vàinitial;chưa phải independentconfirmation hay mộtmaskdesign được
chọn từholdout. Phải giữ mọiarm trongbáocáo,không chỉrow cóAP cao nhất.

Naive-stitchL1AP0.099308 vàfinalL1AP0.099569 gầnnhư nhau. Chưa đo naive-stitch
LPIPS để tách lợi ích learnedinpainting khỏi tác động stitching trongperceptual
space. Vì vậy không kết luận đã chứng minh tumor được xóa hoặc thay bằng môlành.
Historicalancestral300LPIPS-onlyAP0.266899 thuộc samplerkhác;oldAutoDDPMscore
0.154785 còn khác địnhnghĩascoring/repeatbudget,không quy toàn bộ chênh lệchchoDDIM.

Pseudo-maskholdout lesionrecall75.9256percent,precision12.3192percent,
normalcoverage12.0276percent;miss24.0744percent lesionpixels. FinalLPIPSnormalFPR
0.5772percent ởscorethresholdkhác. BrightnesspixelAP0.389506 vẫn cao hơn;
Recallbrightness0.788577 vsfinalLPIPS0.493572 nên Dice/FPR không phải so tại cùng
độnhạy. Finaloutside-mask error0 là điều kiện sampler,không chứngminh restoration.

Actual4150U-Net/166LPIPScalls,mỗi83output đúng50,mộtpass,nguồn/configkhớpsmoke.
Meanphase1/phase2denoising0.588042/0.874035s,sum1.462077s/ảnh;job186.571349s
trước upload gồmsetup/calibration/metrics/I/O. Speedup9.187524x chỉ từ2ảnh đối
chiếuT4 lịch sử cùngscope denoising,không phải full83end-to-end speedup.
Private2panel/maps retainedremote,không tải ảnhy tế vềlocal.

Run`autoddpm-ddim2030-full83-t4-20261106T074200Z`,revision
`ceba5af6b7cdc46b5404554575add3c5cb09d9c7`,rootSHA
`a8bae24bd7270b078d49f923461e43fb51aea3163ede5a2fe71c90a70b127730`
đã kiểm với manifest. Không train/testfinal/tune thêm trênlượtcheck này.
Xem [AUTODDPM_DDIM20_30.md](AUTODDPM_DDIM20_30.md) cho four-armtable,timing,
diagnostics,historicalbaseline vàgiới hạn xác nhận độc lập.
