# Review tiến độ sau tích hợp — 06/11/2026

## Kết luận

Dự án đã có đủ nền tảng và kết quả validation để chuyển sang xác nhận shortlist,
freeze và viết báo cáo. Chưa đủ bằng chứng để công bố kết quả cuối trên test hoặc
đóng toàn bộ blocker về resume. Điểm nghẽn là nghiệm thu và hoàn thiện bằng chứng,
không phải thiếu thêm một model để train. Deadline trong hồ sơ: 04/12/2026,
còn 28 ngày từ ngày review.

## Nguồn đã kiểm tra trong lượt này

- Git fetch thành công: B `9aa42c5`, C `c02addb`, main `da1eea4`. Nhánh B đã
  tích hợp C. Team cần dùng một source chung đã chốt thay vì tiếp tục chạy main cũ.
- GitHub Actions của `e83d97d`, `a1ae5a4`, `9aa42c5`: completed/success.
  [CI của commit mới nhất](https://github.com/tanh1c/DACN/actions/runs/42203313937).
  Workflow chỉ cài `[data,dev]`; success này không bao gồm thực thi các test
  phụ thuộc Torch đã được skip. Không gọi đây là GPU/Modal verification.
- HF model vẫn `fae9b50a4b78114fb51d87e803c77d019668f523`; dataset vẫn
  `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`. Không có revision mới so với replay.
- Connector Linear vẫn ở `traffic-agent-assistant`, không phải DACN-KHMT.
  Không cập nhật task qua connector này. Số 4 Done / 6 In Progress / 16 Backlog
  là snapshot UI đã ghi ở audit trước, không phải lượt đọc live mới này.
- Review source và bằng chứng đã lưu; không chạy lại thí nghiệm hoặc test suite.

## Việc thực sự đã hoàn thành

| Phần | Bằng chứng | Đánh giá |
|---|---|---|
| Data | Dataset pin revision; train normal-only; patient-disjoint; có run truy cập thành công | Đủ cho nghiên cứu; A chốt acceptance/handoff |
| DDPM và thử nghiệm của C | G20/G30, DDIM, LPIPS, naive-stitch, brightness và các kết quả âm | Đã vượt smoke; cần đóng sổ các run lịch sử |
| Evaluation của B | Hai cache hash khớp; image/mask/index khớp dataset; AP/Dice replay khớp C | Hoàn thành replay score maps, chưa tái chạy độc lập U-Net/LPIPS |
| Uncertainty | 1.000 bootstrap draws, 999 hợp lệ; 25 groups/43 slices | Hoàn thành kiểm tra thăm dò theo patient group |
| Tích hợp | C đã vào nhánh B; CI push success; test môi trường tối thiểu và Torch có bằng chứng local | Source đã thống nhất ở nhánh làm việc; chưa nghiệm thu toàn bộ resume |
| Kết quả cuối | Chưa có frozen test result | Chưa hoàn thành |

Không yêu cầu B lặp lại replay/bootstrap đã có. Các metric AUROC/Precision/Recall
trong bảng C đã được kiểm từ JSON, còn lần replay B vừa qua độc lập tái tính
AP/Dice. B cần hoàn tất đối chiếu cả bộ metric bắt buộc khi làm final runner.

## Đánh giá khoa học

Final DDIM20+30 + LPIPS: AP 0,312330 / Dice 0,371215. Paired delta AP final
minus initial có CI95 [0,039686; 0,204806], final minus naive-stitch
[0,026164; 0,177945]. Đây là bằng chứng thăm dò ủng hộ pha 2 trong cấu hình đã
chọn. Final minus brightness có CI95 [-0,287385; 0,232914], chưa cho kết luận
phương pháp nào tổng quát tốt hơn. Brightness vẫn có point AP cao hơn.

Các khoảng trên điều kiện theo một model/protocol đã chọn và validation đã xem
nhiều lần; không khắc phục selection bias và không thay thế repeat training seed.
Giữ brightness và naive-stitch trong báo cáo. Không mô tả LPIPS scoring là một
model mới đã train, hoặc adaptation của AutoDDPM là tái lập nguyên bản paper.
Xem [kết quả và provenance](RESULT_TRACKING.md).

## Những điểm review chưa thể nghiệm thu

1. **P2 — Resume thiếu danh tính nguồn donor và code.**
   `src/dacn/evaluation/patch_validation.py:133` ghi mode nhưng không pin donor
   index/revision hay source snapshot. Với `normal_donor`, thay thứ tự/tập donor
   vẫn có thể dùng prefix cũ rồi tạo phần sau từ donor khác. B/C bổ sung contract,
   test đổi donor/source phải bị từ chối. Đây là đường optional conditioning;
   không phải bằng chứng cache DDIM20+30 đã replay bị sai.
2. **P2 — Sidecar chưa gắn hash với NPZ.**
   `patch_validation.py:174` đọc cache vào bytes nhưng không so checksum cache;
   kiểm labels/images/masks không phát hiện reconstruction/noisy maps bị thay.
   Hai file cache/sidecar cũng được thay thế riêng. Cần binding của payload với
   contract và xử lý interruption; cache cũ phải từ chối hoặc migrate có receipt.
   Không nên mô tả NPZ ở đường này là “bytes đã kiểm hash”.
3. **P2 — Hash checkpoint ghi sau model load.**
   Checkpoint được load ở đầu evaluator, rồi hash tại `patch_validation.py:130`.
   Nếu path đổi giữa hai bước, provenance có thể không đại diện model đã load.
   Dùng cùng snapshot/bytes cho load và hash; kiểm cả đường condition intervention.
4. **P2 — Chưa có test trực tiếp cho Modal continuation mới.**
   19 test đã báo gồm patch-validation và trainer DAC-8; chúng không gọi nhánh
   orchestration `modal_train.py` vừa sửa. Cần test hash thiếu/sai, source thay đổi,
   bytes copy và startup interruption bằng filesystem/mocks trước remote smoke.
   CI tối thiểu không thay thế test này. Không nhất thiết dùng GPU để đóng phần code.

Đây là các khoảng trống xác minh, không kết luận thí nghiệm của C sai. Ưu tiên
đường final DDIM thực sự sẽ dùng; không để refactor các thử nghiệm optional chặn
freeze nếu nhóm loại chúng khỏi shortlist và ghi rõ đường không được hỗ trợ.

## Việc giao tiếp theo và điều kiện hoàn thành

| Ưu tiên / hạn | Owner | Đầu ra cụ thể |
|---|---|---|
| P0, 07–09/11 | A — Tuấn Anh | Chốt nhánh/source chung, phạm vi shortlist, ngân sách xác nhận; cập nhật Linear đúng workspace và bỏ task đã làm khỏi TODO |
| P0, 07–09/11 | B — Chí Vĩ + C review | Chốt receipt replay/bootstrap; đối chiếu AUROC/P/R/F1 và image metrics; xử lý guard của đường được chọn, test negative cases |
| P0, 07–09/11 | C — Hải Tuấn | Nghiệm thu continuation bằng mocks/local artifacts; xác nhận manifest parent; ghi trạng thái cuối run batch-1 cũ |
| P1, 09–13/11 | B/C | Bảng so sánh cố định baseline pixel/SSIM, LPIPS-only, initial/naive-stitch/final, brightness; benchmark latency/VRAM cùng điều kiện |
| P1, 14–20/11 | A/B/C | Repeat sampling seeds nếu budget cho phép, phân biệt train seeds; freeze checkpoint/source/LPIPS weights/ROI/mask/sampler/threshold/seed; runner test có shard/resume và gate |
| P1, 21–27/11 | B/C | Chạy test sau freeze, xuất metric image/pixel đầy đủ, không chọn lại theo test |
| Song song từ bây giờ | A tổng hợp, B/C viết phần mình | Phương pháp, pipeline, bảng kết quả âm, limitations, citations và demo |
| 28/11–04/12 | Cả nhóm | Hoàn thiện báo cáo, kiểm khả năng tái chạy và diễn tập bảo vệ |

Đề xuất Linear: DAC-12/30 có thể chuyển review cho phạm vi baseline/evaluator đã
replay; DAC-15 cần quyết định nghiệm thu có điều kiện rõ; DAC-11/14 chỉ Done khi
resume acceptance được xác minh. Giữ task xác nhận nghiên cứu/freeze/test riêng.
Không giữ tất cả task code mở chỉ vì AP thấp, cũng không đánh Done chỉ vì có PR/CI.
Nếu thiếu GPU, ưu tiên hoàn thành metric replay, code verification, freeze và
báo cáo; ghi limitation một seed thay vì hứa multi-seed chưa có nguồn lực.
