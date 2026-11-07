# Nghiệm thu bounded CPU — 07/11/2026

## Kết luận

Đạt nghiệm thu kỹ thuật **hai mẫu validation** cho trained G30 DDIM20+30:
100 U-Net calls ở lượt ngắt/resume và 100 calls ở lượt liên tục đối chứng.
Tất cả arrays label/score và metrics giữa hai lượt giống hệt, max abs error 0.
Không đổi candidate thành frozen hoặc mở test từ kết quả này.

## Kiểm tra đã thực hiện

- Checkpoint epoch30 SHA256 d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946,
  model revision fae9b50a4b78114fb51d87e803c77d019668f523.
- Tám artifact được kiểm hash trước khi dataset factory chạy. Dependency versions
  khớp export; source thực được chụp vào ZIP trong output nghiệm thu.
- Index validation khớp hash đăng ký. Đối chiếu toàn83 image/mask/label với cache
  đã pin trước khi chọn đúng một normal và một lesion slice.
- Trainer architecture/schedule lấy từ checkpoint; sampler config khớp mask JSON.
- Mỗi mẫu đủ50 U-Net calls. Ngắt chủ động sau sample đầu, load model mới, resume
  sample còn lại. Shard đầu không đổi. Completed-cache resume không gọi inference.
- Lượt liên tục độc lập cùng CPU/seed/config/data cho arrays và metrics bằng nhau.
- 13 tests gate/runner/adapter đạt; Ruff check/format đạt (189 files).

Attempt1 dừng ở checksum config trước inference: adapter nhầm sampler config và
trainer config. Đã sửa; giữ output và log failed attempt. Attempt2 và comparator
hoàn tất, không giấu lượt lỗi. Đây là lời giải thích cho việc checksum gate có
giá trị, không phải bằng chứng checkpoint/model bị lỗi.

## Receipt và số đo

Receipt riêng tư: runs/ddim-acceptance-20261107-attempt2/aggregate_receipt.json
và clean_compare_receipt.json. Source ZIP nghiệm thu hash
242e840d6134f81e8bb27083ad15bdb04a6622bf4dc1be3a881beb3c698de08e.
Wall ngắt/resume58.769387s, gồm load/I-O; comparator44.1546885s. Hai giá trị
không là GPU benchmark hoặc production latency.

Metrics diagnostic, chỉ2samples, threshold giữ từ calibration lịch sử:

| Level | ROC-AUC | AP | Precision | Recall | F1/Dice |
|---|---:|---:|---:|---:|---:|
| Pixel, pooled toàn crop240 | .905501256 | .018280274 | .000465983 | .002293578 | .000774593 |
| Image, top576 mean | 1 | 1 | 1 | 1 | 1 |

Macro lesion Dice .000774593; normal pixel FPR0. Pixel threshold .26617634296417236,
image threshold .24719371760470998. Không fit lại. Một model/sampling seed17;
CPU RNG khác CUDA nên không đòi arrays reconstruction trùng run T4.
Các số này không thay thế full83 validation hoặc chứng minh model tốt/kém;
image AUROC1 trên2samples đặc biệt không có giá trị kết luận hiệu quả.

## Còn trước final freeze

Nghiệm thu này kiểm evaluation resume, không phải resume optimizer training hay
remote Modal interruption. Còn nghiệm thu controls theo contract riêng, full
inference validation/compute budget nếu cần, RAM/GPU latency/VRAM và xác minh
source/dependency/dataset binding tại runtime final. CI bị chặn cấp account;
Windows rename vẫn chưa có root cause. Script CLI hiện gom các bước đã chạy riêng
thành quy trình; phần combined CLI sau bổ sung comparator chưa chạy lại toàn bộ.
