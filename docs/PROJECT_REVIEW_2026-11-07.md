# Review mức độ hoàn thiện — 07/11/2026

## Kết luận

Dự án ở giai đoạn **đã có hệ thống nghiên cứu và validation, đang chuẩn bị
nghiệm thu/final evaluation**. Chưa phải gói đồ án sẵn sàng nộp: thiếu runner
test thực có freeze, pin môi trường/weights, kết quả test cuối, báo cáo thống
nhất và demo phương pháp đã chọn. Còn 27 ngày tới deadline 04/12 trong hồ sơ.
Không dùng số file/commit/test làm phần trăm hoàn thành đề tài.

## Phạm vi và trạng thái xác minh

Inventory tracked: 76 files trong src, 38 trong tests, 23 configs, 9 scripts,
3 notebooks, 64 docs (trước tài liệu review này). Đã rà cấu trúc, requirements,
entry points, các đường inference/freeze chính, hồ sơ kết quả và kiểm thử toàn
bộ; không khẳng định review từng dòng mọi nhánh thử nghiệm hoặc đọc toàn bộ
Paper_Reference.xlsx trong lượt này. Các file private dưới runs/data không
được xuất nội dung theo sample, không mở test MRI.

- Git fetch: nhánh làm việc `ecad491`, C `c02addb`, main `da1eea4`.
  C đã tích hợp vào nhánh làm việc. Main vẫn cũ.
- HF model `fae9b50a4b78114fb51d87e803c77d019668f523`, dataset
  `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`: không đổi từ lần replay.
- Local full suite: **249 passed / 1 failed**, 45,04 giây. Failure:
  `test_prepare_corrected_policy_records_exclusions`, Windows WinError 5 khi
  rename thư mục staging. Chưa có root cause; không báo all green.
- Ruff check pass; format 179 files pass.
- GitHub Actions của `ecad491` và `b986080`: **startup_failure**.
  [Run mới nhất](https://github.com/tanh1c/DACN/actions/runs/42240463952).
  Jobs API trả rỗng, không có check-run output để xác định lý do. Không suy
  thành lỗi test, billing hoặc workflow syntax khi chưa có bằng chứng.
  Những run xanh trước đó là lịch sử, không chứng minh head hiện tại xanh.
- Linear connector xác minh vẫn ở workspace khác. Mặc định repo đã đặt
  DACN-KHMT; người dùng đang được hướng dẫn reconnect. Không cập nhật status
  bằng connector sai workspace hoặc dùng snapshot cũ như live state.

## Đánh giá từng phần

| Hạng mục | Mức hoàn thiện | Còn thiếu |
|---|---|---|
| Data + protocol | Đã sử dụng thực, version và patient-disjoint rõ | Acceptance cuối và lifecycle run lịch sử |
| Training DDPM | Có checkpoints, EMA, artifact, thí nghiệm G20/G30 | Kiểm tích hợp remote resume nếu cần chạy lại; không bắt buộc train mới để tiếp tục |
| Scoring/evaluation | Đã có pixel/SSIM/LPIPS và controls | Đóng shortlist; đóng gói đủ controls cho final runner |
| B replay | Hoàn thành AP/Dice/bootstrap và full image/pixel metrics từ cache | Chưa rerun độc lập U-Net/LPIPS với môi trường mới |
| Guard kỹ thuật | Đã cải thiện snapshot bytes, prefix binding, continuation helper | Gate freeze chưa được gọi bởi runner; full orchestration vẫn cần kiểm |
| Final evaluation | Có candidate JSON và helper gate | Còn null pins; chưa runner test thực/chunked aggregation và result cuối |
| Báo cáo | Nhiều records/phân tích có thể dùng | Chưa thấy báo cáo nộp/slide thống nhất trong tracked tree |
| Demo | CLI synthetic, notebook train/validation | Cần demo ứng viên DDIM+LPIPS từ artifact thật, quy trình tái chạy rõ |
| Quản lý | Đã thống nhất nhóm cùng làm | README/plan/snapshot cũ cần dẫn về trạng thái mới; kết nối Linear cần đúng workspace |

`capture_frozen_inputs` hiện chỉ có definition và tests, không có caller trong
runner. `split="test"` ở CLI là demo synthetic. Không được tính hai phần đó là
đã có final MRI test pipeline.

LPIPS chưa có trong optional dependencies của pyproject; Modal cài riêng
`lpips==0.1.4` và torchvision không pin. Chưa có dependency lock/weights hashes
đủ cho môi trường final. Không tự lấy weights ở môi trường khác rồi tuyên bố
khớp run lịch sử; cần kiểm trực tiếp và ghi khác biệt nếu có.

## Giá trị khoa học đã có

Final DDIM20+30 LPIPS: AP .312330, Dice .371215. B tái tính từ cached scores
khớp C cả image/pixel AUROC/AP/P/R/F1, thresholds, macro lesion Dice và normal
FPR; bootstrap patient-group dùng 999/1000 draws hợp lệ trên 25 groups/43 slices.
Delta AP final minus naive stitch CI95 [.026164,.177945]; final minus brightness
[-.287385,.232914]. Đây là kết quả validation thăm dò có selection bias, một
model/sampling seed. Không kết luận vượt brightness trên dữ liệu mới.

Đã đủ nguyên liệu viết câu chuyện nghiên cứu: diffusion normal-only, hạn chế
noise-MSE, đóng góp perceptual scoring/pha 2, controls và kết quả âm. Đề bài
không yêu cầu một AP tùy ý hoặc bắt buộc model mới; nên ưu tiên bằng chứng và
tái lập hơn việc mở thêm backbone hoặc train để đẩy validation score.

## Thứ tự công việc chung

| Hạn dự kiến | Việc | Điều kiện xong |
|---|---|---|
| 07–09/11 | Sửa kết nối Linear, kiểm CI startup và Windows rename | Đúng workspace; nguyên nhân/biện pháp có evidence, ghi log nếu còn blocker |
| 07–10/11 | Chốt shortlist và chuẩn bị môi trường inference | Source, config, checkpoint, LPIPS và dependency pins thực; controls có thresholds riêng |
| 09–13/11 | Nối gate → runner → artifacts → aggregate metrics | Test synthetic/validation; resume cùng contract; global AP/AUROC đúng, không mean theo shard |
| 14–20/11 | Xác nhận validation bounded, benchmark cùng môi trường và freeze | Receipt end-to-end; seed/budget/limitations đăng ký trước; trạng thái frozen có căn cứ |
| 21–27/11 | Final test theo protocol đã freeze | Kết quả image/pixel đầy đủ; ghi failed/interrupted runs; không tune theo test |
| Ngay từ 07/11 | Viết báo cáo, figures, citations, demo | Một bản báo cáo thống nhất, bảng provenance và controls; dữ liệu riêng tư giữ đúng quyền |
| 28/11–04/12 | Đóng gói và diễn tập | Reproduction instructions, slides/demo, rà format và kết luận |

Nhóm làm chung; không chờ theo role. Có thể làm báo cáo, runner synthetic,
artifact pinning và xử lý CI ngay bằng CPU. GPU chỉ dành cho xác nhận/inference
cần thiết sau khi runner sẵn sàng. Không tự train thêm trong review này.
