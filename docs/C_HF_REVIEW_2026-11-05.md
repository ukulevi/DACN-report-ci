# Review công việc C từ Hugging Face — 05/11/2026

Nguồn: private model repo `dacn-bmad-team/dacn-runs`, revision
`73552266d80393164a2a1dcf3fb3eacbe1d15a57`. Đây là review manifest/config/history/
metric do C tạo, chưa chạy tái lập feature checkpoint. Evidence aggregate ở
[JSON review](evidence/2026-11-05-c-hf-feature-review.json).

## C đã thực hiện được gì

- Gaussian batch8 epoch30 và Simplex epoch20 đã có cache validation được B tái
  lập trước đây. Pipeline checkpoint/continuation/normal-only training hoạt động
  qua các artifact đã ghi nhận, không chỉ smoke.
- Feature A: từ EMA Gaussian30, thêm ResNet18 khởi tạo random và train joint
  encoder/projection/U-Net10epochs,9380updates mới. Chọn best epoch6 bằng
  normal-validation noise-MSE0,00734351; inference full83/t300/seed17 hoàn tất.
- Feature B: encoder được masked-pretrain trên normal MRI5epochs/4690updates,
  rồi joint10epochs/9380updates từ cùng parent Gaussian30. Không phải pretrained
  ImageNet hay foundation model bên ngoài. Best joint epoch6, noise-MSE0,00734149.
- Có ablation conditioning `self`, `zero`(bypass cả bias), `normal_donor` từ train;
  cùng input/noise/reverse RNG, không đưa lesion mask vào model. Có preflight
  tái lập reference sai lệch0,40calibration/43holdout theo patient group.
- Manifest có dataset revision, checkpoint hash, config, packages, source snapshot,
  lịch sử epoch và thời gian. Source dirty được ghi rõ: chỉ dẫn commit7134b10
  không đủ tái lập, phải dùng snapshot riêng của từng run.

Kiểm chứng mới: tải các JSON ở revision pin;202đối chiếu artifact hash từ manifest
đều khớp. JSON kiểm từ bytes đã tải; checkpoint/source-zip lớn chỉ đối chiếu
metadata LFS, chưa tải/deserialise để chứng minh runtime hay resume thực tế.
Đây là bằng chứng provenance, chưa phải chứng minh phương pháp đúng hoặc tốt hơn.

## Chất lượng model đang ở đâu

Cùng43validation holdout, crop240, threshold riêng cho từng method được fit trên
40calibration. Full83 descriptive không được trộn vào bảng này. Một model seed17.

| Model / score | Pixel AP | Pixel AUROC | Precision | Recall | F1/Dice | Normal pixel FPR |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian30 / difference | .101066 | .876854 | .119934 | .314702 | .173678 | .058079 |
| Feature A / difference | .104645 | .878721 | .126465 | .316731 | .180757 | .056641 |
| Feature B / difference | .104464 | .878343 | .126436 | .310393 | .179681 | .055678 |
| Gaussian30 / local SSIM | .102402 | .888870 | .110476 | .257712 | .154654 | .055234 |
| Feature A / local SSIM | .104247 | .888060 | .115090 | .252128 | .158039 | .053224 |
| Feature B / local SSIM | .104084 | .887708 | .115097 | .251140 | .157851 | .053144 |

Exact Gaussian baseline values must be taken from the JSON review if additional
digits are required. Các metric ở đây là pixel-level global micro. Image AUROC
holdout của feature A/B localSSIM chỉ khoảng0,412/0,414; image precision24/43,
recall1 tương ứng dự đoán tất cả ảnh abnormal(TP24,FP19,TN0,FN0). F1 khoảng0,716
không chứng minh phân loại ảnh tốt; specificity bằng0.

Lợi ích pixel AP của A/B so với Gaussian30 chỉ vài phần nghìn tuyệt đối và còn
thêm chi phí fine-tune. A và B gần nhau; chưa có bằng chứng pretraining encoder
giúp tăng anomaly performance. Loss noise chỉ đo nhiệm vụ epsilon prediction.
Khoảng74phút joint training mỗi arm; B cộng khoảng5,2phút encoder pretraining.
Đây là thời gian thực từ manifest, không phải hóa đơn hoặc ước tính chi phí USD.

## Ablation nói gì về conditioning

| Feature B conditioning | Holdout AP difference | Holdout AP local SSIM |
|---|---:|---:|
| Self | .104464 | .104084 |
| Zero | .104579 | .104416 |
| Normal donor từ train | .105132 | .104699 |

Tắt conditioning hoặc dùng donor normal không làm AP giảm ở lần chạy này.
Điều này chưa chứng minh model hoàn toàn bỏ qua encoder: mean absolute change
trên miền[0,1] khoảng0,0017–0,0020 nhưng max pixel change khoảng0,51–0,54.
Kết luận có căn cứ là **lợi ích của conditioning riêng từ ảnh đang xét chưa được
chứng minh**, không phải khẳng định đã tìm thấy bug encoder.

Brightness control cùng holdout AP0,389506 và Dice0,268051 vẫn cao hơn các
diffusion variants. Nó là sanity control để đo phần giá trị model thêm vào,
không thay mục tiêu diffusion của đồ án. Cần báo cáo kết quả âm này trung thực.

## Phương án tiếp theo cho nhóm

1. Giữ feature A/B và checkpoints như kết quả ablation; chưa tăng depth, train
   thêm epoch hay mở nhiều noise/encoder variants dựa trên chênh lệch nhỏ hiện tại.
2. Đối chứng Gaussian20/Simplex20 của B đã hoàn tất; cập nhật06/11:holdout AP
   .099777/.087416,deltaCI(Simplex–Gaussian)[-.030666,.011189]. Chưa chứng minh
   noise-family advantage ổn định. Xem RESULT_TRACKING; không gộp full83 với43.
3. Với A/B, B cần cache UUID-aligned riêng tư hoặc đầy đủ mapping được kiểm chứng
   để paired bootstrap cùng patient groups và threshold đã cố định; chưa tải cache
   mới trong review này. Chỉ JSON không đủ tính paired uncertainty.
4. Nếu nhóm giữ feature branch, ưu tiên thí nghiệm kiểm sensitivity có giới hạn
   và control fine-tune U-Net không conditioning cùng parent/updates. Gaussian30
   chưa fine-tune không phải control đầy đủ cho10epoch joint thích nghi thêm.
5. Tập trung residual/anomaly score và cách tổng hợp image score; foreground mask
   chỉ giải quyết một phần FP, không tự sửa image AUROC. Chọn thay đổi qua protocol
   validation đã đăng ký, rồi freeze trước test. Không tune theo test.

Đánh giá tổng thể: phần triển khai và lưu artifact của C đã đi xa hơn smoke và có
đối chứng có ích. Phần chứng minh hiệu quả cải tiến vẫn chưa đủ mạnh: một seed,
validation đã được xem nhiều lần, compute không đồng nhất và feature benefit nhỏ.
Không đóng blocker chất lượng anomaly map chỉ vì run completed hoặc loss thấp.
