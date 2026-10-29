# Thiết kế nghiên cứu: DDPM partial reconstruction

## Quyết định và câu hỏi

Ngày 28/10/2026 nhóm chọn đề xuất DDPM reconstruction thay kế hoạch ANDi-FLAIR trước đây. ANDi là related work/extension, không là đường găng. Đây là kế hoạch, không phải kết quả hay tuyên bố phương pháp mới.

**RQ1:** Mức nhiễu khởi đầu `t*` ảnh hưởng thế nào tới anomaly detection và localization, thời gian và VRAM trên BMAD Brain FLAIR 2D?
**RQ2:** Trên cùng ảnh tái tạo, pixel absolute difference và local SSIM difference thay đổi kết quả như thế nào? Khi đủ nguồn lực, so với DAE coarse-noise.

Giả thuyết: nhiễu ít có thể giữ cả tổn thương; nhiễu nhiều có thể xóa tổn thương nhưng cũng làm sai mô bình thường. Chỉ xác nhận hoặc bác bỏ bằng artifact thực tế. Partial diffusion đã có tiền lệ trong [AnoDDPM](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf).

## Hợp đồng mô hình và thực nghiệm

- Input: một kênh PNG FLAIR đã render, pad 240→256, normalized [-1, 1] theo [DATA_PROTOCOL](DATA_PROTOCOL.md). Train chỉ trên normal slice; normal ở đây là slice không có tumor annotation, không đồng nghĩa bệnh nhân khỏe mạnh.
- **Contract triển khai DAC-8 — bản đề xuất, chờ A/B review:** train U-Net dự đoán Gaussian noise với MSE (`epsilon-prediction`), `T=1000`, linear β từ 0.0001 đến 0.02 theo [DDPM](REFERENCES.md); timestep huấn luyện lấy đều trên `[0,999]`. Kiến trúc một kênh: U-Net hiện có, base channels 32, multipliers `(1,1,2,2,4,4)`, hai residual blocks/level, attention ở resolution 16 và 8, hai heads, dropout 0. Tối ưu AdamW (`lr=1e-4`, weight decay 0), gradient norm clip 1, EMA 0.9999. Đây là cấu hình triển khai được đề xuất, chưa phải cấu hình đã được nhóm phê duyệt qua review hay đã có kết quả.
- **Quy ước partial reconstruction:** `t*` bên ngoài là one-based và chỉ mức forward-noise: `t*=100` ánh xạ tới timestep array index 99. Tạo một noisy input bằng forward process, rồi chạy toàn bộ ancestral DDPM reverse transitions từ index `t*-1` về 0 (`t*` reverse steps); fixed-large reverse variance theo `GaussianDiffusion`, dự đoán clean image được clip trong `[-1,1]`, đầu ra cuối cũng clip trong `[-1,1]`. Phân biệt `t*`, tổng schedule 1000 và số reverse steps. Ghi rõ khác biệt với nguồn tham khảo nếu contract review đổi lựa chọn nào.
- **Checkpoint proposal:** chọn epoch có epsilon-MSE thấp nhất trên normal validation slices với timestep/noise cố định, dùng EMA weights. Labels dùng để stratify validation cap và lấy normal subset; anomaly labels/masks không tham gia loss hay tiêu chí checkpoint. Held-out test không được trainer nhận. A/B cần xem xét checkpoint rule trước nghiên cứu chính.
- `t*` ứng viên nghiên cứu vẫn là 100, 200, 300; 50 và 500 chỉ mở rộng nếu budget được chốt. Trainer cung cấp `load_ema_checkpoint`, `build_diffusion`, và `partial_reconstruct`; B dùng reconstruction trả về cho pixel absolute difference và local SSIM trên cùng input, checkpoint, timestep, và sampling seed. Không train lại theo score/timestep. Đo latency reconstruction và score riêng, tổng end-to-end, VRAM và số ảnh thực đo.
- **Bounded smoke proposal:** seed 17, một epoch, batch 1, tối đa 128 normal train slices và 16 validation slices, preview `t*=100`, tối đa 4 ảnh validation. Đây chỉ là pipeline/sanity smoke, không phải model comparison hay nghiên cứu. Chưa được chạy trong phần DDPM này; DAC-8/A-B review và DAC-9 data gate phải hoàn tất trước khi bắt đầu GPU job.
- **Pixel difference:** map `abs(input - reconstruction)`, không lấy MAE toàn ảnh trước khi tính pixel metrics. **Local SSIM:** map `1 - local SSIM`; chốt window, data range, xử lý biên và range đầu vào trước so sánh. Score cao là bất thường.
- DAE là baseline khi khả thi, không chặn diffusion MVP. Feature difference chỉ làm khi còn thời gian và đã chốt encoder, nguồn/training, layer, map, license và chi phí. ANDi, VAE và Isolation Forest là related work/extension tùy chọn.

## Prototype AnoDDPM tách biệt

Repo có một trainer `dacn.training.anoddpm:train` được port có chọn lọc từ AnoDDPM
để kiểm tra sớm pipeline diffusion một kênh trên BMAD. Nó dùng U-Net dự đoán noise,
simplex hoặc Gaussian noise, full-shard epochs và squared reconstruction error. Đây
là **AnoDDPM-style BMAD adaptation** để kiểm tra data/model/run boundary, không thay
thế protocol DDPM partial reconstruction đã chốt: nó chưa có pixel absolute/local
SSIM comparison và chưa freeze `t*`/sampler theo DAC-8. Không suy diễn smoke test
hoặc validation subset thành kết quả nghiên cứu.

## Protocol đánh giá

- Split `patient-disjoint-v1` và revision Hub cố định theo DATA_PROTOCOL. Validation có nhãn dùng chọn checkpoint/config/threshold: mô tả là normal-only training với labeled validation selection. Không dùng held-out test để lựa chọn hoặc tuning.
- Pixel AP chính: global micro average precision trên vùng ảnh gốc 240×240, crop padding, gồm ảnh normal/abnormal. Report pixel/image ROC-AUC, Precision, Recall, F1; thêm pixel AP, Dice, latency, VRAM khi phù hợp. Image score là mean top 1% pixel score. Chốt threshold image và pixel riêng trên validation; pixel Dice micro-optimal trên validation là quy tắc dự kiến, phải freeze trước test.
- Đề xuất chọn timestep/score chính theo pixel AP validation; trước sweep cần chốt
  tie-break và budget. Chốt riêng quy tắc image threshold (ví dụ tối đa image F1
  trên validation) trước khi dùng validation để lựa chọn, rồi ghi giá trị đã khóa.
- Chốt foreground/post-processing cho từng method, công bố khác biệt; cùng vùng metric cho mọi phương pháp. Kiểm tra trên validation tỷ lệ lesion pixel bị loại bởi foreground erosion của DAE trước so sánh; không thay protocol dựa trên test.
- Ghi macro Dice trên slice lesion và false-positive pixel rate ở mask rỗng; báo prevalence. Dùng cùng tập validation và quy tắc ROI cho đối chứng. Pixel metrics dựa trên cùng output tái tạo, score và threshold đã đăng ký.
- Validation chỉ 83 slice/50 patient groups: hạn chế sweep, không xem một lần chọn trên validation như kết luận chắc chắn. Cố gắng 3 train seeds khi ngân sách cho phép; ghi số thực chạy. Bootstrap theo patient group, ghi số replicate/seed và xử lý replicate thiếu lớp; không coi slice/pixel là bệnh nhân độc lập.
- Latency: cùng GPU, batch, precision và resolution; đề xuất warm-up 20 batch,
  đo ít nhất 100 slice khi khả thi và GPU synchronize, báo median/p95 ms/ảnh và
  peak VRAM. Nêu rõ nếu không đạt số lượng này và tách scoring khỏi I/O.
- Sau review, freeze model, seed policy, `t*`, sampler, scores, preprocessing, thresholds và metric trước đánh giá held-out test. Mọi thay đổi sau test phải công khai; không tuning lại trên test.

## Kết quả và giới hạn

Theo dõi mọi run thực bắt đầu, cả lỗi/gián đoạn, trong RESULT_TRACKING với manifest khi có. Hình validation: ảnh đầu vào/noisy/reconstruction/map và trường hợp lỗi; test chỉ xem sau freeze. PNG không giữ intensity MRI gốc; train normal slice có thể thuộc bệnh nhân u; chưa chứng minh generalization hoặc tính lâm sàng. Dùng shared private artifacts và chỉ đưa số aggregate an toàn vào Linear.
