# Cải tiến perceptual scoring cho DDPM MRI: khảo sát và thiết kế thí nghiệm

Ngày đối chiếu nguồn: 06/11/2026. Thực hiện yêu cầu trong [test.md](test.md),
bổ sung theo bảng vai trò encoder người dùng xác định. Đây là khảo sát và
đề xuất ở thời điểm khảo sát. Sau đó người dùng cho phép triển khai D trước,
với SqueezeNet hiện có; trạng thái actualruns theo NORMAL_CALIBRATED_LPIPS.md.
Các encoder medical A/B và learnedattention C chưa được chạy.
Giữ nguyên tài liệu yêu cầu của người dùng.

## 1. Kết luận ưu tiên và mốc so sánh thực tế

Ưu tiên một hướng: **perceptual distance đa tầng, hiệu chuẩn bằng sai khác
reconstruction trên MRI normal**. Khởi đầu bằng các tầng LPIPS SqueezeNet hiện
có để tách tác dụng của hiệu chuẩn; sau đó kiểm tra encoder medical bằng đối
chứng cùng kiến trúc. Đây là giả thuyết nghiên cứu, không phải phương pháp đã
được xác nhận tăng AP. Không coi thay backbone là một đóng góp mới tự thân.

Lý do chọn: tận dụng checkpoint/cặp ảnh đã có, FLAIR một kênh, không cần tumor
mask để học encoder, không cần train lại DDPM, và kiểm tra được riêng từng yếu
tố. LPIPS vốn đã đa tầng; chỉ cộng thêm các tầng không đủ tạo đóng góp mới.

Baseline phù hợp cho nghiên cứu score trên **ảnh final DDIM20+30** hiện là
SqueezeNet LPIPS: Pixel AP **0.312330**, micro Dice **0.371215**, Image AP
**0.944386** trên 43 ảnh group holdout. Ancestral300 + LPIPS-only AP 0.266899
là mốc khác sampler/reconstruction. Không dùng hai mốc như một so sánh chỉ thay
encoder. Brightness hiện đạt Pixel AP 0.389506 trên cùng holdout.
Xem [báo cáo DDIM20+30](AUTODDPM_DDIM20_30.md).

## 2. Vai trò các encoder và kiểm chứng nguồn

| Encoder | Dữ liệu pretrained | Vai trò trong DACN |
|---|---|---|
| SqueezeNet LPIPS | ImageNet trunk và perceptual calibration | Baseline thực dụng hiện tại |
| RAD-DINO | X-quang ngực | Thử nghiệm chuyển miền, không đại diện MRI-pretrained |
| Brain MRI encoder | MRI não, ưu tiên có FLAIR | Giả thuyết phù hợp miền hơn, cần weights/provenance thực |
| DINOv2 cùng ViT-B/14 | Ảnh tự nhiên | Đối chứng kiến trúc cho RAD-DINO |
| ResNet50 ImageNet / RadImageNet | Ảnh tự nhiên / dữ liệu radiology | Cặp đối chứng CNN cùng kiến trúc |

SqueezeNet khác ViT cả kiến trúc lẫn calibration head. Nếu RAD-DINO tốt/xấu hơn
SqueezeNet LPIPS thì chưa thể quy nguyên nhân riêng cho miền pretrained.
Mọi so sánh DINO với DINO phải dùng cùng tap layers, pooling/upsampling và
công thức distance; mọi so sánh ResNet50 với ResNet50 cũng vậy.

### LPIPS hiện tại

[LPIPS chính thức](https://github.com/richzhang/PerceptualSimilarity) cung cấp
SqueezeNet/AlexNet/VGG với learned linear calibration; code BSD-2-Clause.
Repo DACN đang dùng squeeze, version 0.1, spatial=True, frozen/eval, crop240,
grayscale lặp RGB, input [-1,1], normalize=False. Không phải VGG/AlexNet trong
phép thử hiện tại. Learned head không phải tumor classifier.

Không thể thay ResNet/ViT vào rồi giữ nguyên head SqueezeNet: số tầng, kênh và
ý nghĩa feature khác nhau. Distance feature medical không có perceptual head
đã học tương ứng phải được gọi là **spatial feature distance/Med-LPIPS-style**,
không gán nhãn LPIPS calibrated nguyên bản. Cấu trúc head và retPerLayer được
đối chiếu từ [code LPIPS](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py).

### RadImageNet: ứng viên CNN medical thực tế, không bảo đảm tốt hơn

[Paper RadImageNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/) có CT/MRI/US,
pretraining classification ở 224×224, không chứng minh frozen perceptual score
cho anomaly segmentation FLAIR của DACN. Chưa xác nhận riêng mức bao phủ FLAIR.
[Repo tác giả](https://github.com/BMEII-AI/RadImageNet) công bố link weights
TensorFlow và PyTorch, gồm ResNet50/DenseNet121; code MIT.

[Model card RSNA](https://atlas.rsna.org/cards/61caf4ba-8941-4699-9824-9e341d9661b9)
ghi CC BY 4.0 cho model. Cần giữ riêng thông tin license code/model/data và đọc
điều khoản trong đúng archive tải về; không tự đồng nhất mọi artifact với MIT.
Link PyTorch chính thức đã xác định nhưng trang Drive không mở được bằng công
cụ web hiện tại: **chưa tải bytes, chưa strict-load/hash checkpoint, chưa chốt
normalization của đúng bản PyTorch**. Không ghi là smoke-ready.

Không suy từ modality name rằng model chắc tăng AP: nghiên cứu đối chiếu
[Juodelyte et al., TMLR](https://arxiv.org/abs/2302.08272) báo cáo ImageNet có thể
vượt RadImageNet trên các tác vụ classification họ khảo sát. Đó cũng không phải
kết quả anomaly segmentation của DACN.

### RAD-DINO và các ứng viên MRI DINO

[RAD-DINO Microsoft](https://huggingface.co/microsoft/rad-dino) có weights công
khai/model card MIT, pretrained chest X-ray bằng DINOv2; không pretrained MRI.
Checkpoint công khai không hoàn toàn giống checkpoint paper, theo model card.
[Config](https://huggingface.co/microsoft/rad-dino/blob/main/config.json) xác nhận
ViT-B, 12 blocks, 768 chiều, patch14 và input518; patch grid là 37×37.
[Processor](https://huggingface.co/microsoft/rad-dino/blob/main/preprocessor_config.json)
phải được lưu cùng revision và phép biến đổi tọa độ. Không center-crop mất ROI.
Weights chưa được tải/smoke trong lượt khảo sát này.

Các nguồn MRI thực sự liên quan:

- [BrainFound paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13639087/) mô tả
  self-supervised brain MRI T1/T2/FLAIR, có BraTS trong nguồn pretraining.
  [Repo](https://github.com/Moona-Mazher/BrainFound) có mã scaffold; module
  [brainfound.py](https://github.com/Moona-Mazher/BrainFound/blob/main/models/brainfound.py)
  hiện ghi `coming soon!`. Chưa xác minh được một checkpoint MRI-trained đầy đủ
  kèm input contract và license weights để đưa vào runner.
- [BrainDINO tác giả](https://github.com/mclwu22/BrainDINO) nêu weights/pipeline
  pretraining chưa phát hành ở thời điểm kiểm tra. Không biến thành một model
  có sẵn bằng cách tải DINOv2 ảnh tự nhiên rồi đổi tên.
- [MM-DINOv2](https://github.com/daniel-scholz/mm-dinov2) nghiên cứu glioma MRI
  đa chuỗi; README hiện hướng dẫn tải **DINOv2 gốc** rồi train. Chưa xác nhận
  release checkpoint MRI-adapted trong nguồn đã đọc. Repo ghi CC BY-NC 4.0.
- [3DINO paper/code](https://github.com/AICONSlab/3DINO) có pretrained 3D medical
  model và [HF weights](https://huggingface.co/AICONSlab/3DINO-ViT), nhưng gated,
  CC BY-NC-ND 4.0 theo cả repo/model card. Dữ liệu hiện tại là PNG2D, không đủ
  spacing/volume để gọi trực tiếp như 3D MRI. Cần xác minh quyền dùng/adaptation
  của đúng artifact; không tự nhận quyền từ việc code có thể tải được.

Nếu encoder đã pretrained trên BraTS, cần kiểm tra khả năng trùng cohort với
BMAD-derived BraTS2021 trước khi gọi target holdout/test là độc lập với toàn bộ
pretraining. Chưa có patient provenance thì ghi unknown; không công bố patient
IDs để giải quyết việc này. External medical pretraining có thể dùng pathology
labels: phải phân biệt với DDPM chỉ học normal và target không dùng tumor mask
để training. Không gộp thành tuyên bố toàn bộ pretraining là normal-only.

### MedSAM và RadFM: không phải thay thế LPIPS trực tiếp

[MedSAM repo](https://github.com/bowang-lab/MedSAM) có checkpoint, code Apache-2.0.
Encoder có thể dùng không prompt để lấy feature; decoder segmentation cần prompt,
không được lấy bbox từ tumor mask để tạo anomaly score. [Inference chính thức](https://github.com/bowang-lab/MedSAM/blob/main/MedSAM_Inference.py)
dùng RGB lặp từ grayscale, resize1024 và normalize[0,1], cho embedding256×64×64.
Đây không phải contract LPIPS [-1,1]. Resize không tạo thêm chi tiết tổn thương;
độc lập min-max hai ảnh có thể làm thay đổi quan hệ intensity và phải được kiểm
soát như một ablation preprocessing. Chưa smoke checkpoint tại DACN.

[RadFM chính thức](https://github.com/chaoyi-wu/RadFM) là vision-language model
2D/3D, không có spatial LPIPS head sẵn. README mô tả input3×H×W×D, token nén
qua Perceiver và khuyến nghị A10080GB cho pipeline đầy đủ. [Code license MIT](https://github.com/chaoyi-wu/RadFM/blob/main/LICENSE)
không thay cho việc kiểm tra điều khoản đúng model/base components tải về.
Không ưu tiên trên T4; nếu lấy visual features phải hook trước pooling/token
compression và kiểm chứng alignment, không dùng text embedding/global tokens
làm bản đồ pixel rồi upsample tùy ý.

Một bài tìm được rất gần chủ đề, [arXiv:2608.28773](https://arxiv.org/abs/2608.28773),
đã **withdrawn**. Không dùng các số liệu trong abstract đó làm bằng chứng đang
được xác nhận cho hướng này, cũng không suy ra nguyên nhân khoa học của rút bài.

## 3. Bảng so sánh theo yêu cầu

Khả năng cải thiện AP dưới đây đều là giả thuyết; không có AP dự báo hoặc đảm bảo.
Chi phí là đánh giá kỹ thuật tương đối, chưa benchmark các encoder mới trên T4.

| Phương pháp | Pretrained weights | Cần training mới? | Phức tạp | Giả thuyết lợi ích | Rủi ro chính |
|---|---|---|---|---|---|
| Original LPIPS | SqueezeNet ImageNet + perceptual head đã xác minh | Không | Thấp | Baseline đã đo | Domain gap có thể còn, không phải tumor probability |
| RadImageNet-style distance | Official link ResNet50/DenseNet121; load/proc còn kiểm tra | Không ở pilot frozen | Vừa | Feature radiology có thể nhạy cấu trúc phù hợp | Weight/proc/architecture confounds; không có LPIPS head tương ứng |
| Medical DINO distance | RAD-DINO có publicweights nhưng CXR; MRI checkpoint cần xác minh | Frozen:0; MRI adaptation là task khác | Vừa–cao | Encoder MRI thật có thể phù hợp miền hơn | Token resolution; weights unavailable/gated; pretrain overlap |
| Attention LPIPS | Trunk cũ; không có pretrained MRI attention head phù hợp sẵn | Có nếu neural gate | Vừa | Downweight sai khác normal không hữu ích | Collapse/background shortcut, bỏ sót lesion, synthetic-domain overfit |
| Multi-contrast distance | Cần encoder/inputs thích hợp T1/T2/FLAIR | Có thể cần DDPM/reconstruction mới | Cao, hiện không khả thi trên export | Dùng thông tin bổ sung thật | Dataset chỉ FLAIR; registration/contrast leakage |
| Normal-calibrated multi-scale score | Giữ trunk hiện có trước; medical là controlled extension | 0epoch; fit statistics normal | Thấp–vừa | Cân bằng sai khác normal giữa các tầng | Ít normalcalibration, IQR nhỏ, có thể giảm AP |

## 4. Kiến trúc và công thức ưu tiên

Giữ cặp `(x,r)` cố định: ảnh gốc và **final DDIM20+30 đã cache**. Không dùng
medical features tạo lại pseudo-mask trong thí nghiệm scoring đầu tiên; như vậy
không thay reconstruction, seed hay chuỗi diffusion khi so các distance.

```text
x và final r đã lưu
       ↓ cùng encoder frozen, eval và preprocessing của checkpoint
feature maps tại nhiều tầng
       ↓ normalized channel features / per-layer perceptual contribution
d_l(p): spatial distance từng tầng → cùng lưới240×240
       ↓ fit median/IQR từ cặp ảnh normal calibration
z_l(p): distance hiệu chuẩn cho sai khác normal
       ↓ trọng số tầng cố định bằng nhau
S(p): anomaly map liên tục
```

Với encoder medical không có learned LPIPS head, đặt:

    u_l(x,p) = φ_l(T(x),p) / sqrt(sum_c φ_l(T(x),p,c)^2 + ε)
    d_l(p)   = U_l[ sum_c (u_l(x,p,c)-u_l(r,p,c))^2 / C_l ]

`U_l` là bilinear align_corners=False, trả về lưới gốc240. T là preprocessing
của checkpoint, không thay preprocessing đầu vào DDPM. Không suy từ feature
distance thành xác suất tumor. Với SqueezeNet, dùng đúng per-layer contribution
trả bởi LPIPS `retPerLayer=True` để giữ learned head, không thay nó bằng metric
khác rồi gọi là cùng baseline.

Chỉ trên 20 ảnh normal calibration, với foreground heuristic `x>-0.95`:

    m_l = median(d_l(p));  s_l = max(Q75(d_l)-Q25(d_l), 1e-6)
    z_l(p) = (d_l(p)-m_l) / s_l
    S(p) = (1/L) sum_l z_l(p)

Trọng số cố định, không học từ tumor labels, không chọn layer/weight theo holdout.
Áp dụng score lên **toàn bộ240crop**, không tự đặt background về0. Foreground
chỉ dùng fit stats; không gọi đó là brain segmentation. Ghi mọi floor activation
và distribution shift. So ablation calibration-off/on cho **cùng encoder và
cùng layer set**. Không dùng voxel-specific atlas priors khi chưa đăng ký không
gian; bộ ảnh hiện chưa có correspondence giải phẫu chuẩn giữa bệnh nhân.

Khảo sát này đề xuất công thức, chưa đo hiệu quả. Nhiều tầng không bảo đảm bắt
tổn thương nhỏ: receptive field lớn, feature pooling và bilinear upsampling có
thể làm mờ tín hiệu. Cần báo cáo riêng lesion boundary/components và FP proxies.

## 5. Kế hoạch từng thí nghiệm A–D

| Arm | Pipeline/layers dự kiến | Training/loss | Chi phí tương đối và kiểm tra |
|---|---|---|---|
| Baseline | Cached final → LPIPS squeeze0.1 → distance hiện có | 0epoch; giữ head | Thấp; tái lập AP0.312330 trước mọi thay đổi |
| A: RadImageNet | Cached pair → frozen ResNet50 → layer1/2/3 → d_l; equal weights | 0epoch; không fine-tune | Vừa; phải load/hash/proc trước smoke; so ResNet50 ImageNet cùng layers/distance |
| B: DINO | Cùng cặp → frozen ViT-B/14 → patch tokens blocks3/6/9/12 → distance | Frozen:0epoch. Không tự train MRI DINO nếu chưa có checkpoint | Vừa–cao; RAD-DINO là cross-domain, so DINOv2 base cùng kiến trúc; MRI-trained arm chỉ mở khi artifact hợp lệ |
| C: Attention | Frozen encoder → channel gate hoặc spatial score gate → distance | Nếu làm: pilot3epochs trên tối đa512normal train, batch8,192updates | Thêm backprop; không dùng tumor GT; cần synthetic/SSL objective chống collapse, chưa được phê duyệt train |
| D: Normal-calibrated multi-scale | Per-layer maps → normal-only median/IQR → fixed equal-weight sum | 0epoch; chỉ fit stats20normalcal | Gần chi phí frozen extraction; ưu tiên kiểm tra bằng trunk hiện có rồi controlled medical extension |

**Layers và độ phân giải:** SqueezeNet dùng cả bảy contributions có sẵn, không
cần download encoder mới. ResNet50 layer1/2/3 dự kiến C=256/512/1024; với fully
convolutional input240 thường có60/30/15cells, với input224 là56/28/14. Đây là
shape cần assert bằng checkpoint thực; chưa phải shape đã smoke RadImageNet.
Không center-crop bỏ ROI; nếu adapter buộc resize224 phải map về240 và áp dụng
cùng cách lên hai pretrained ResNet controls. Không mặc định ImageNet mean/std
cho RadImageNet khi chưa đối chiếu normalization của đúng weights artifact.

RAD-DINO input518/patch14 cho37×37tokens; loại CLS, giữ patch order khi reshape.
Các blocks3/6/9/12 là tap set dự kiến đăng ký trước thử nghiệm, không chọn theo AP.
Upsample trở lại240 không khôi phục chi tiết thật đã mất. MRI encoder khác ViT-B
cần chốt mapping riêng và không được gọi là đối chứng cùng kiến trúc nếu khác.

**Calibration khác training:** learned perceptual calibration head cần dữ liệu
judgment/pairs thích hợp. Normal median/IQR ở D không phải học lại BAPPS head.
Không gắn BAPPS-calibrated hay MRI-calibrated claim nếu chưa train/validate head.
Không có DDPM training epochs mới trong bất kỳ frozen arm nào.

### Attention và anatomy: placement/loss/labels

[CBAM paper](https://arxiv.org/abs/1807.06521) và [code tác giả](https://github.com/Jongchan/attention-module)
chứng minh attention module trong các bài toán họ thử, không chứng minh tăng
anomaly AP trên MRI này. Spatial attention có thể đặt **sau distance aggregation**;
channel gate đặt trước distance và dùng đối xứng cho hai ảnh. Nếu nhân cùng một
spatial scalar vào cả vector rồi L2-normalize, scalar có thể triệt tiêu: phải
xác định đúng placement, không chỉ thêm một module tên attention.

Normal-only neural attention khả thi nhưng cần tín hiệu học. Chỉ minimize
normal-score có nghiệm suy biến attention=0 hoặc chỉ nhìn background. Một
pilot **đề xuất**, chưa chạy: frozen features, tạo synthetic local corruption
trên ảnh normal và giữ reconstruction normal đã cache làm reference; học gate
nhỏ bằng ranking loss:

    L_rank = max(0, margin + mean(S_normal) - mean(S_synthetic trên vùng đã tạo))

Kèm regularization/entropy hoặc ràng buộc trọng số có tổng1 để tránh collapse.
Vùng corruption là mask tự tạo, không phải tumor annotation. Synthetic patches
có thể khác lesion thật và gây shortcut; cần so gate-off, statistical weighting
và learned gate. Số3epochs/192updates chỉ là budget pilot, không phải khẳng định
hội tụ. Không có train mới nào được chạy bởi bản khảo sát này.

Không lấy attention map hay foreground threshold làm nhãn anatomy. Muốn weighting
theo não thất/CSF/cortex cần nhãn/segmenter/registration đã kiểm tra và phải tính
chi phí, domain errors, cũng như điều kiện pretrained. Bắt đầu bằng fixed normal
statistics ít tham số hơn neural gate, nhưng cũng không bảo đảm không bỏ sót tumor.

## 6. Multi-contrast: điều kiện và cách fusion

Export hiện chỉ có FLAIR đã render; lặp grayscale3channels không thành T1/T2/FLAIR.
Không có dữ liệu để chạy thí nghiệm multi-contrast thật trên export này. Cần cặp
chuỗi cùng bệnh nhân/slice và correspondence không gian đã kiểm tra; ảnh không
align làm feature difference lớn ở mô lành. Không giả định đăng ký đúng chỉ từ
việc tên dataset là BraTS.

- Early fusion: stack các contrasts nhưng phải xử lý semantics/intensity input,
  không tự xem ba contrasts như RGB pretrained. Có thể cần encoder adaptation.
- Late fusion: score riêng từng contrast, normal-calibrate rồi fusion; dễ tách
  đóng góp hơn cho pilot có đủ paired reconstructions, nhưng tăng nhánh tính toán.
- Cross-attention: nhiều tham số và alignment/training phức tạp; không ưu tiên với
  ít dữ liệu và T4 budget hiện tại. Lợi ích phân biệt lesion là giả thuyết cần đo.

FLAIR DDPM không được đưa thẳng T1/T2 rồi coi là reconstruction hợp lệ. Muốn score
trên tất cả contrasts cần reconstruction reference tương ứng: model riêng từng
contrast hoặc DDPM đa kênh có huấn luyện/fine-tune thích hợp. Chỉ thay LPIPS không
tạo ra những reference còn thiếu. Copy T1/T2 bệnh vào cả input/reference cũng
không làm chúng trở thành pseudo-healthy. Giữ hướng này ngoài thực nghiệm đầu tiên.

## 7. Protocol công bằng và thứ tự thực hiện

1. Dùng đúng cached final DDIM20+30: evaluation
   `autoddpm-ddim2030-full83-t4-20261106T074200Z`, private revision
   `ceba5af6b7cdc46b5404554575add3c5cb09d9c7`, private_ablation SHA
   `22656e1f63a0047d3dbe32f7833eea8e303988414e5942dec1ba5a1d9ed1ef19`.
   Hash/index/order, original/final/masks/labels đều phải khớp; không chạy lại DDPM.
2. Chạy lại baseline scoring để tái lập Pixel AP0.312330 trên samefinal. Đừng so
   medical-final với natural-initial rồi quy công cho backbone.
3. Kiểm tra D với SqueezeNet: original summed map vs calibrated per-layer sum,
   cùng weights/layers/preprocessing; freeze công thức trước full evaluation.
4. A chỉ bắt đầu khi weights/license/checksum/strict state load/proc được xác minh.
   Thử smoke1normal+1abnormal, assert shapes, finite scores, spatial alignment.
   Không tự thay bằng third-party mirror không ghi provenance.
5. A/D medical dùng cùng kiến trúc ResNet50 ImageNet control. Tách một-tầng vs
   ba-tầng và calibration-off/on như các arm đã đăng ký, không sweep chọn winner
   theo43holdout. B kiểm tra tương tự với DINOv2 matched control; MRI branch chờ
   checkpoint thực. C là nhánh sau, chỉ train khi được yêu cầu riêng.
6. Giữ dataset revision350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7, cùng40calibration/
   43patient-group holdout, seed17 và crop240. Fit normal stats chỉ20normalcal;
   pixel/image F1 thresholds có thể fit labeled40cal theo protocol hiện hành.
   Áp dụng thresholds cố định trênholdout; AP/ROC dùng continuous scores.
7. Báo cáo pixelAP/AUROC/P/R/F1(micro Dice), normalFPR, imageAUROC/AP/P/R/F1,
   top576aggregation và inference-time/peakVRAM thực đo. Có thể thêm bootstrap
   paired theo patient-group; CI không sửa được adaptive overfitting vì đã xem
   holdout nhiều lần. Không chỉ báo cáo một score được chọn sau xem kết quả.
8. Bổ sung kiểm tra naive-stitch + LPIPS trước khi quy toàn bộ lợi ích final-score
   cho learned re-sampling. Scoring controls này dùng cache, không cần train.
9. Chốt method/weights/stats/thresholds/code rồi mới xác nhận độc lập. Finaltest
   giữ kín trong development; không đổi split hay seed trên cùng ảnh rồi gọi là
   cohort độc lập. Pretrain overlap của encoder cần được ghi rõ.

Đóng góp DACN hợp lý là câu hỏi có đối chứng: **hiệu chuẩn sai khác normal ở từng
tầng có giúp perceptual anomaly maps ổn định hơn không, và medical pretraining
có thêm lợi ích khi đã kiểm soát kiến trúc, score và reconstruction?** Câu trả
lời có thể là âm tính. Chưa có căn cứ gọi phương pháp mới hoặc công bố AP cao hơn.

## 8. Trạng thái thực hiện của lượt khảo sát

Đã đọc yêu cầu, kiểm tra code/protocol/kết quả local và các nguồn sơ cấp liên
quan, đối chiếu model cards/license/input/weights availability như ghi trên.
Đã phát hiện RAD-DINO là CXR, một số MRI releases chưa đủ artifact thực thi,
3DINO gated/license riêng và một tài liệu gần chủ đề đã withdrawn.
Tại thời điểm hoàn tất khảo sát: chưa tải weights medical, chưa train
encoder/attention/DDPM, chưa chạy GPU A–D và chưa có AP mới.
Sau đó người dùng đã duyệt chạy [normal-calibrated LPIPS](NORMAL_CALIBRATED_LPIPS.md);
trạng thái thực thi nằm trong RESULT_TRACKING. Đây là đầu ra khảo sát/thiết kế theo test.md, không phải log
một thực nghiệm đã hoàn tất. Các kiểm tra artifact còn thiếu là preflight của
thí nghiệm tương lai, không được biến thành một run đã chạy.
