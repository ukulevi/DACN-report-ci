# Spatial ResNet-18 ImageNet + Gaussian DDPM EMA30: grayscale pilot

User yêu cầu thử spatial feature conditioning, sau đó chỉ định giữ grayscale,
ngày05/11/2026. Mục tiêu: xem feature map giữ thông tin vị trí có cải thiện
segmentation so với global-pooled A/B và matched unconditional control không.
Đây là giả thuyết, chưa có kết quả AP cho đến khi artifact thực tế hoàn tất.

## Kiến trúc và grayscale

Input MRI giữ nguyên `1×256×256`, miền `[-1,1]` và padding8px như baseline.
**Không repeat RGB, không ImageNet RGB mean/std, không resize hay center crop.**
ResNet18 lấy trọng số chính thức `ResNet18_Weights.IMAGENET1K_V1`; thay conv1
3kênh bằng1kênh: `W_gray = mean(W_rgb, dim=1, keepdim=True)`.
Các convolution/BatchNorm còn lại nạp strict từ ImageNet. Bỏ global pooling và
classifier; đây là thích nghi grayscale/MRI, không phải nguyên transform RGB
phân loại ImageNet, cũng không phải encoder đã pretrain trên MRI.

Nguồn chính thức: [PyTorch ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
[architecture/weight URL](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).
Weights URL cố định `https://download.pytorch.org/models/resnet18-f37072fd.pth`,
46.830.571bytes,SHA256
`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
Hash được kiểm tra khi bake image CPU và khi khởi tạo training. Không đưa weights
vào Git; checkpoint inference nạp encoder từ EMA đã lưu, không download ImageNet.

```text
MRI 1×256×256
  -> ResNet18 grayscale, ImageNet initialization, no pooling
  -> C×H×W: 64×64×64;128×32×32;256×16×16;512×8×8
  -> 4fusion1×1 zero weight/bias
  -> cộng residual vào last encoder block tại mỗi resolution của U-Net
  -> thông tin đi tiếp qua bottleneck và skip connections sang decoder

Noisy x_t + timestep t -> U-Net initialized from Gaussian EMA30 -> epsilon_hat
```

U-Net baseline32channels,multipliers[1,1,2,2,4,4],2resblocks được giữ nguyên.
Fusion tại down-block indices[8,11,14,17],output channels[64,64,128,128].
Không sửa checkpoint keys của U-Net; forward thêm optional spatial_residuals.
Zero fusion phải cho output bằng chính xác original EMA30 trước update đầu.
Encoder gradient bằng0 ở update đầu do zero gate; fusion học trước rồi encoder
nhận gradient. Đây là dự kiến, không đóng băng encoder.

## Fine-tune và dữ liệu

- Khởi tạo độc lập từ Gaussian EMA30 có SHA
  `d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946`.
  Không dùng A/B hay unconditional epoch36 làm U-Net nguồn.
- T4,seed17,batch8,10fine-tune epochs/9380updates như A/B.
- U-NetLR1e-5,pretrained encoderLR1e-5,fusionLR1e-4. Fresh AdamW,
  betas0.9/0.999,weight_decay0,clipnorm1,EMA0.9992.
- Joint train encoder/fusion/U-Net. BatchNorm running stats cập nhật online và
  copy sang EMA như A/B; không đóng băng pretrained BatchNorm.
- Gaussian epsilonMSE,linearT1000,7500normal train slices; mask không vào model.
- Best chọn bằng EMA epsilonMSE trên39normal validation,fixedseed1000017.
  Milestones1/5/10,last each epoch,best và epoch0recovery.
- Dataset `dacn-bmad-team/bmad-flair-patient-disjoint-v1`,immutable revision
  `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.
- “Normal” là slice không có annotation tumor, không đảm bảo bệnh nhân khỏe.

Config `spatial_imagenet_grayscale_smoke.json`:1epoch,batch2,16normal train,
4normal-val,3step reconstruction check. Smoke không là AP result,weights không
chuyển sang full. Full config `spatial_imagenet_grayscale_t4_epoch10.json`.

## Validation tự động sau full training

Best EMA theo normal-loss ->83validation(39normal/44tumor),ancestral t300,batch1,
same UUID-derived seed17 và paired difference/local1-SSIM từ cùng reconstruction.
Giữ40calibration/43group holdout,25groups mỗi bên,calibration-only thresholds,
globalmicro240crop và top576 image score. All83thresholds chỉ mô tả.

So immutable caches/results của baseline30,A,B và matched unconditional best36;
kiểm tra SHA/index/protocol alignment. Report pixel/image ROC-AUC,Precision,
Recall,F1/AP,normal FPR,tissue MAE/SSIM,lesion/nonlesion means,latency/VRAM và
private examples. Test3665 giữ kín; không timestep sweep/chọn checkpoint bằng AP.

## Checkpoint và chạy Modal

Method `ddpm_spatial_resnet18_imagenet_v1`,namespace feature_condition_*.pt trong
run riêng. State chứa U-Net+grayscale encoder+fusion,EMA/optimizer/RNG/sampler,
backbone/config; imagenet_provenance.json ghi weights/hash/input adaptation và
fusion indices. `load_feature_checkpoint` chọn đúng architecture theo method;
unconditional loader tiếp tục từ chối loại checkpoint này.

Entrypoint `src/dacn/cloud/modal_spatial_condition.py::run_spatial_remote`,mode smoke
trước train_eval. Image bake official weights trước GPU allocation. Wrapper dùng
cùng lifecycle training/evaluation đã kiểm tra,source-checked resume,prefix
cache và private uploads. Workspace phutungcogioihaitan,shared Secret hiện có,
T4 timeout4h,detached full run; chỉ xác nhận start rồi check khi user yêu cầu.

## Giới hạn cần giữ trong báo cáo

Trial đổi cả ImageNet initialization lẫn spatial fusion; encoderLR1e-5 cũng khác
A/B1e-4. Nếu AP tăng, chưa quy riêng cho spatial information mà cần ablation
matched-pretraining/fusion. Encoder vẫn nhìn ảnh gốc có thể truyền tumor vào
reconstruction; spatial detail không đảm bảo xóa lesion. MRI[-1,1] khác ImageNet
RGB transforms. Một seed trên validation đã khảo sát chưa có significance hay
khả năng khái quát. Giữ baseline/A/B/control để báo cáo cả kết quả âm.

## Kết quả thực tế — hoàn tất05/11/2026

Training10epochs/9380updates hoàn tất;best normal-val EMAepoch10,MSE
0.004498155053,train+val4507.506611s,peak5883574272B. Validation83/83
hoàn tất với cùng40calibration/43group holdout,t300 như trên.

| Mô hình | Difference Pixel AP,holdout43 | Local SSIM Pixel AP,holdout43 |
|---|---:|---:|
| Original Gaussian EMA30 |0.101066|0.102402|
| Unconditional +10epochs,best36 |0.104391|0.104004|
| A global random encoder |0.104645|0.104247|
| B global normal-pretrained encoder |0.104464|0.104084|
| Spatial ImageNet grayscale,best10 |0.063110|0.065459|

Spatial pixel F1 difference0.119789/SSIM0.136377;normal pixel FPR
12.0666%/26.0876% tại thresholds học trên calibration. SSIM Recall86.1924%
nhưng Precision7.4047%. AP giảm ở cả hai score nên không chỉ là chọn threshold.
Full pixel/image ROC-AUC,Precision,Recall,F1 và ngưỡng xem
[RESULT_TRACKING.md](RESULT_TRACKING.md). All83 AP mô tả difference0.062166,
SSIM0.066047 cũng thấp.

Normal foreground reconstruction tốt lên:MAE[0,1]0.039324712,SSIM0.688031178
so unconditional0.082240741/0.295440689. Trên44abnormal slices,
mean1-SSIM lesion0.276705021 và foreground ngoài lesion0.271836988 gần nhau;
difference lesion0.073481523/nonlesion0.063075532. Gợi ý original-image condition
có thể giúp tái tạo cả lesion,làm residual khó phân biệt. Đây là diễn giải từ
thống kê,chưa chứng minh cơ chế bằng ảnh/ablation.

**Kết luận pilot:cấu hình này cải thiện reconstruction normal nhưng anomaly
segmentation kém baseline. Không lấy noiseMSE thấp làm lý do tự tăng epoch.**
Chưa quy lỗi riêng cho grayscale,ImageNet,spatial fusion hay mọi diffusion model;
trial thay nhiều yếu tố và chỉ một seed/validation đã khảo sát.

Training private revision29777bf14204bdc7b4450f2b9a0746b1a537911c;
[evaluation artifacts](https://huggingface.co/dacn-bmad-team/dacn-runs/tree/a6e21e5c2553e683fda3e51496fdc52a31662b60/evaluations/spatial-imagenet-gray-t4-b8-e10-20261105T141939Z-val).
BestcheckpointSHA6a4103bb14937421fc79e595fb7fdd7d460774dc68e2db1ef3e8c75b04d307cb;
resultSHA0c9b75267a8d6f26a54c071ab26e263ea3fc0b292a438ff112acd2a15dd5299f
khớp manifest. Evaluator837.866527s,peak151107072B,total916.399822s.
Ảnh/cache/checkpoints private;held-out test không dùng.
