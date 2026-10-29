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
- `t*` ứng viên nghiên cứu vẫn là 100, 200, 300; 50 và 500 chỉ mở rộng sau quyết định budget. Trainer cung cấp `load_ema_checkpoint`, `build_diffusion`, `derive_sampling_seed` và `partial_reconstruct`; B dùng cùng reconstruction cho pixel absolute difference và local SSIM. Không train lại theo score/timestep. Đo latency reconstruction và score riêng, tổng end-to-end, VRAM và số ảnh thực đo.
- **Bounded smoke proposal:** seed 17, một epoch, batch 1, tối đa 128 normal train slices và 16 validation slices, preview `t*=100`, tối đa 4 ảnh validation. Đây chỉ là pipeline/sanity smoke, không phải model comparison hay nghiên cứu. Chưa được chạy trong phần DDPM này; DAC-8/A-B review và DAC-9 data gate phải hoàn tất trước khi bắt đầu GPU job.
- **Training trước sweep:** sau smoke/pilot và review, đặt `max_train_samples=null`, `max_validation_samples=null` để dùng cả 7.500 normal train slices và 39 normal validation slices cho checkpoint loss; các validation abnormal slices không vào loss. Cố định số epoch và mọi tham số training theo budget pilot **trước** run nghiên cứu, ghi config/seed/checkpoint; không mang checkpoint của smoke 128/16 vào bảng kết quả. Epsilon-MSE validation dùng seed `train_seed + 1000000`, batch size và thứ tự validation cố định trong config.
- **Score contract đề xuất cho DAC-8:** crop padding trước khi tính score; hai ảnh float trong `[-1,1]`, ROI là toàn vùng 240×240, không foreground mask hay post-processing. Pixel map là `abs(input - reconstruction)`. Local SSIM map là `1 - SSIM`, dùng Gaussian window 11×11, sigma 1.5, chuẩn hóa kernel về tổng 1, weighted population variance/covariance, padding `reflect` trên ảnh đã crop, `data_range=2`, `C1=(0.01*2)^2`, `C2=(0.03*2)^2`. Không clip hay chuẩn hóa score map; score cao hơn là bất thường. Image score là mean của 576 pixel có score cao nhất (1% của 240×240). B cần review contract và implementation ở DAC-30 trước sweep.
- **Seed policy đề xuất:** train seed và sampling base seed ghi riêng; sweep đầu dùng mỗi loại seed 17. Cho mỗi sample UUID của export HF, canonicalize UUID và tính 4 byte đầu SHA-256 của UTF-8 `dacn-ddpm-seed-v1:<base_seed>:<canonical_uuid>` theo big-endian làm seed 32-bit. Gọi reconstruction với batch 1 và dùng cùng seed của sample cho mọi `t*`, rồi dùng chính tensor reconstruction đó cho cả hai score. UUID chỉ dùng trong bộ nhớ/private data, không ghi vào Git, Linear hoặc manifest công khai. Quy tắc này không biến seed thành dữ liệu ẩn danh. Các lần lặp sau phải đăng ký sampling seed và train seed trước khi chạy; train seeds 42/2026 là mục tiêu khi đủ budget, không tự coi là đã chạy.
- **Sweep validation đề xuất:** một checkpoint EMA đã chọn theo normal-validation epsilon-MSE, một train seed 17 và một sampling base seed 17, toàn bộ 83 validation slices theo revision đã pin, `t*` = 100/200/300 và hai score trên cùng reconstruction: 249 reconstruction, sáu tổ hợp score/timestep. Không dùng 50/500, feature score, nhiều checkpoint hay train lại trong sweep này. Chọn tổ hợp bằng global micro pixel AP trên validation; nếu AP bằng nhau ở độ chính xác đầy đủ, ưu tiên `t*` nhỏ hơn, sau đó pixel difference. Chọn riêng image và pixel threshold trên validation bằng F1 cao nhất với rule dự đoán `score >= threshold`, xét các giá trị score phân biệt; nếu F1 bằng nhau, chọn threshold lớn hơn. Pixel micro F1 và Dice tương đương trong phép chọn này. Chốt checkpoint, thresholds và toàn protocol trước held-out test. Ước lượng runtime/quota từ pilot trước khi chạy sweep; nếu không đủ, ghi lại thay đổi và review trước khi chạy, không chọn bằng test.
- **Sai khác nguồn tham khảo:** phần forward dùng Gaussian noise và epsilon-MSE theo DDPM; backbone/process được tái sử dụng từ adaptation AnoDDPM, nhưng cấu hình này dùng Gaussian thay simplex, BMAD FLAIR 2D đã preprocess, fixed-large ancestral reverse, clipping và EMA checkpoint chọn theo normal-validation epsilon-MSE. Đây là adaptation của nhóm, không tuyên bố tái lập nguyên thí nghiệm DDPM/AnoDDPM. Checkpoint rule đo dự đoán noise, không bảo đảm checkpoint tốt nhất cho anomaly detection; A/B cần review hạn chế này.
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
