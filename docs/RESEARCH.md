# Thiết kế nghiên cứu: DDPM partial reconstruction

## Quyết định và câu hỏi

Ngày 28/10/2026 nhóm chọn đề xuất DDPM reconstruction thay kế hoạch ANDi-FLAIR trước đây. ANDi là related work/extension, không là đường găng. Đây là kế hoạch, không phải kết quả hay tuyên bố phương pháp mới.

**RQ1:** Mức nhiễu khởi đầu `t*` ảnh hưởng thế nào tới anomaly detection và localization, thời gian và VRAM trên BMAD Brain FLAIR 2D?
**RQ2:** Trên cùng ảnh tái tạo, pixel absolute difference và local SSIM difference thay đổi kết quả như thế nào? Khi đủ nguồn lực, so với DAE coarse-noise.

Giả thuyết: nhiễu ít có thể giữ cả tổn thương; nhiễu nhiều có thể xóa tổn thương nhưng cũng làm sai mô bình thường. Chỉ xác nhận hoặc bác bỏ bằng artifact thực tế. Partial diffusion đã có tiền lệ trong [AnoDDPM](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf).

## Hợp đồng mô hình và thực nghiệm

- Input: một kênh PNG FLAIR đã render, pad 240→256, normalized [-1, 1] theo [DATA_PROTOCOL](DATA_PROTOCOL.md). Train chỉ trên normal slice; normal ở đây là slice không có tumor annotation, không đồng nghĩa bệnh nhân khỏe mạnh.
- Train DDPM; dùng forward noising đến `t*` và reverse sampler tạo ảnh tái tạo pseudo-healthy. Chốt objective, kiến trúc, noise, tổng bước, schedule, quy ước chỉ số, sampler, số reverse steps và clipping trong DAC-8 trước các run chọn model. `t*` chỉ là mức nhiễu bắt đầu, không đồng nghĩa số bước reverse hay số timestep tổng hợp của ANDi. Ghi sai khác so với nguồn tham khảo.
- Cùng checkpoint, ảnh, tiền xử lý, `t*`, sampler và seed khi so hai score. Lưu reconstruction/score artifacts trong không gian private, không đưa ảnh y tế lên Git/Linear. Ghi seed train và noise/sampling riêng; lặp noise seed nếu pilot cho phép.
- **Pixel difference:** map `abs(input - reconstruction)`, không lấy MAE toàn ảnh trước khi tính pixel metrics. **Local SSIM:** map `1 - local SSIM`; chốt window, data range, xử lý biên và range đầu vào trước so sánh. Score cao là bất thường.
- Đề xuất ban đầu: `t*` = 100, 200, 300 nếu tổng bước và schedule đã chốt làm chúng hợp lệ; 50, 500 là mở rộng theo pilot. Chưa có giá trị tối ưu. Cố định danh sách thí nghiệm, lựa chọn model và budget trước khi chạy validation sweep.
- Tạo cùng reconstruction một lần cho mỗi input, timestep, checkpoint và seed rồi so pixel/SSIM trên cùng output. Không train lại mỗi score/timestep. Ghi latency reconstruction và score riêng, tổng end-to-end, VRAM và số ảnh thực đo.
- DAE là baseline khi khả thi, không chặn diffusion MVP. Feature difference chỉ làm khi còn thời gian và đã chốt encoder, nguồn/training, layer, map, license và chi phí. ANDi, VAE và Isolation Forest là related work/extension tùy chọn.

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
