# Thiết kế nghiên cứu và thực nghiệm

## Câu hỏi nghiên cứu

**RQ1:** ANDi thích nghi cho BMAD FLAIR 2D đạt chất lượng định vị và chi phí thế nào
so với DAE với coarse noise? **RQ2:** giữ nguyên checkpoint ANDi, dùng ít timestep
hơn để tính anomaly score có giữ được AP với thời gian thấp hơn không?

**Giả thuyết, chưa phải kết quả:** một tập timestep thưa có thể giữ phần lớn tín hiệu
định vị. ANDi đã nghiên cứu độ nhạy khoảng timestep; không gọi việc chọn timestep là
ý tưởng mới chỉ vì chưa thử trên BMAD. Đóng góp phù hợp đồ án là tái triển khai có
kiểm soát, benchmark hiệu quả và phân tích giới hạn. Chỉ dùng từ “cải thiện” sau số đo.

## Adaptation phải ghi rõ

| ANDi gốc | Thiết lập đồ án |
|---|---|
| BraTS, bốn chuỗi MRI | BMAD Brain, một kênh từ độ sáng PNG FLAIR |
| Pipeline intensity riêng của tác giả | PNG đã render theo slice; Pillow RGB→L; padding 240→256; x/127.5−1 |
| Config công bố 128×128 | Giữ chi tiết 240×240, pad 256; đo chi phí bằng pilot |
| Có lọc median 3D trong đánh giá | Không dựng volume giả từ các slice thưa; bản chính không hậu xử lý, nếu thử lọc 2D phải khai báo riêng |
| Training/scoring và khoảng timestep trong repo | Tái triển khai đúng công thức trước; ghi mọi sai khác về noise, schedule, backbone, step indexing |

Đây là **ANDi-FLAIR adaptation**, không phải tái lập chính xác con số paper.
Đọc [repo ANDi](https://github.com/AlexanderFrotscher/ANDi) và
[paper](https://arxiv.org/html/2312.01904v1) trước khi viết scorer.
Mốc khởi đầu trong config tác giả: diffusion 1.000 step, scoring `start=75, stop=200`,
training pyramid noise và evaluation Gaussian noise. Cần kiểm tra quy ước endpoint
và công thức variance/geometric aggregation theo code đã pin commit.

## Ma trận thí nghiệm

| ID | Phương pháp | Biến thay đổi | Vai trò |
|---|---|---|---|
| E0 | DAE + coarse noise | Baseline riêng, tune noise rất hạn chế trên val | Đối chứng rẻ |
| E1 | ANDi-FLAIR full interval | Cấu hình gốc thích nghi | Đối chứng chính |
| E2a–c | ANDi-FLAIR sparse K=8,16,32 | Chỉ tập timestep scoring; giữ checkpoint | Trả lời RQ2 |
| E3 (tùy chọn) | Normal calibration | Chỉ cách chuẩn hóa score | Chỉ làm khi E0–E2 đã ổn |
| E4 (tùy chọn) | Sparse + calibration | Kết hợp hai thành phần | Bắt buộc có E1/E2/E3 để tách tác động |

Chọn timestep đều trong cùng khoảng, ghi danh sách số nguyên thực dùng, không chỉ
ghi K. Cùng một noise seed trên các timestep chung để so sánh paired. Không đánh đồng
số timestep với latency: phải đo cả truyền dữ liệu và chi phí aggregation.
Calibration nếu thực hiện: fit trên phần normal của train/validation được chỉ định
trước; chưa chốt công thức trong đợt chuẩn bị này. Không dùng test để fit hay chọn nó.

## Protocol đánh giá cần triển khai ở tuần 2

- Train chỉ normal slice; mask không đi vào model/scorer. Validation có nhãn dùng
  chọn cấu hình và ngưỡng, nên gọi training normal-only với labeled model selection.
- Primary: **pixel average precision (AP)** trên toàn bộ pixel vùng ảnh gốc 240×240,
  loại padding, gồm cả normal và abnormal slice. Đây là global micro AP, không phải
  trung bình AP từng ảnh và không phải diện tích PR theo trapezoid.
- Secondary: pixel AUROC; image AUROC/AP với image score cố định là trung bình top
  1% pixel score trong ảnh gốc (ít nhất 1 pixel). Mọi thay đổi aggregation chỉ dựa val.
- Dice: chọn một ngưỡng pixel toàn cục tối ưu Dice micro trên val; khóa trước test.
  Báo Dice micro toàn tập, Dice macro trên các slice mask dương, và false-positive
  pixel rate trên mask rỗng. Không để Dice=1 của ảnh normal che chất lượng lesion.
- Báo prevalence ở mức slice và pixel; nhãn ảnh cần đối chiếu với mask theo audit.
- Lặp 3 seed train cho cấu hình cuối nếu pilot cho phép; báo mean/std, số seed thực
  chạy. E2 dùng checkpoint tương ứng E1, không huấn luyện riêng để lấy lợi thế.
- CI 95% bằng bootstrap theo patient, giữ các slice cùng patient trong một đơn vị
  resample. Ghi seed và số lần bootstrap; bỏ/ghi nhận replicate thiếu một lớp ở AUROC.
  Không coi các pixel hay slice là các bệnh nhân độc lập.
- Latency trên cùng GPU, batch, precision và resolution; warm-up 20 batch, đo ít
  nhất 100 slice, GPU synchronize; báo median/p95 ms/ảnh và peak VRAM. Tách model
  scoring khỏi I/O và báo thêm end-to-end nếu dùng demo.

Validation chỉ 83 slice: pre-register sweep nhỏ E0–E2, không chạy search vô hạn.
Test chỉ mở cho đánh giá sau khi khóa cấu hình. QC cấu trúc/nhãn trước đó không phải
đánh giá mô hình; mọi sửa nhãn phải dựa quy tắc nguồn và công bố, không dựa score.

## Hình/bảng cho báo cáo

1. Bảng nguồn và split từ audit; flow ảnh → preprocessing → model → score → metric.
2. Bảng E0/E1/E2 (AP, AUROC, Dice, ms/ảnh, VRAM, seeds).
3. Đồ thị pixel AP–latency; cùng checkpoint và thiết bị.
4. Heatmap thành công/thất bại được chọn theo quy tắc công khai, gồm lesion nhỏ và
   false positives. Chỉ trình bày test sau khi khóa mô hình.
5. Giới hạn: PNG mất intensity gốc, validation nhỏ, normal slice từ người có u,
   khác domain và chưa có kiểm chứng lâm sàng/ngoài bộ dữ liệu.
