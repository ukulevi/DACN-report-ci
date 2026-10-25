# Theo dõi dự án bằng Linear

**Project:** [DACN — BMAD Brain & ANDi-FLAIR](https://linear.app/traffic-agent-assistant/project/dacn-bmad-brain-and-andi-flair-637bb0ca9b0e)  
**PR nền tảng:** [tanh1c/DACN #11](https://github.com/tanh1c/DACN/pull/11)  
Team: Traffic-agent-assistant, theo lựa chọn của người dùng ngày 25/10/2026.

Linear là nguồn trạng thái công việc hiện hành; GitHub là nơi review và lưu code.
Bảng dưới là backlog khởi tạo, không tự đồng bộ trạng thái Linear về repo.

## Mốc 6 tuần

| Milestone | Hạn dự kiến | Gate |
|---|---|---|
| W1 — Data, protocol & GPU pilot | 30/10/2026 | Data được review/bàn giao; chốt adaptation; đo GPU/VRAM/thời gian |
| W2 — DAE & ANDi chạy được | 06/11/2026 | Có checkpoint/scorer/evaluator, resume được, đánh giá validation |
| W3 — Full vs sparse timestep | 13/11/2026 | So sánh cùng checkpoint/seeds, AP–latency; chọn tối đa 1 biến thể |
| W4 — Repeatability & freeze | 20/11/2026 | Repeat final configs; khóa threshold/model/protocol trước test |
| W5 — Held-out test & analysis | 27/11/2026 | Test theo freeze, CI theo patient, bảng/plot và phân tích lỗi |
| W6 — Report, slides & defense | 04/12/2026 | Báo cáo/slide/demo tái chạy được, kiểm tra gói nộp |

Ngày là dự kiến theo kế hoạch 6 tuần; C01 xác nhận lịch trường và sửa các mốc nếu cần.

## Workflow và tiêu chí Done

- Backlog: chưa đưa vào công việc đang thực hiện.
- Todo: sẵn sàng xem xét cho tuần hiện tại; kiểm tra dependency trước khi nhận.
- In Progress: một người đang thực hiện; mỗi người tối đa một task chính.
- In Review: có PR/artifact để teammate kiểm tra.
- Done: acceptance criteria đạt, có bằng chứng và review. PR đang mở chưa là Done.
- Canceled: nhánh bị bỏ có lý do, đặc biệt extension tùy chọn.

Dependency dùng quan hệ blocked-by trong Linear; không tạo thêm trạng thái Blocked
vào workflow của team chung. Khi bị chặn, cập nhật nguyên nhân, người xử lý và hạn
giải quyết trong issue; không chuyển Done để làm đẹp progress.

Mỗi ngày cập nhật ngắn: kết quả có bằng chứng, blocker, việc kế tiếp. Cuối tuần
đối chiếu gate milestone, cập nhật project health (on track / at risk / off track)
theo bằng chứng; số issue Done không thay thế chất lượng thực nghiệm.

Để xem tiến độ, lọc issues trong project theo milestone, status, assignee hoặc nhãn
dacn:data / baseline / diffusion / evaluation / reporting / coordination. Extension
calibration mang nhãn dacn:optional và không chặn freeze MVP.

## Phân công và truy vết

Chí Vĩ phụ trách data và điều phối các mốc freeze/handoff. Vai trò B (DAE/evaluation)
và C (diffusion/tích hợp) còn tạm thời; issues tương ứng để unassigned vì hai thành
viên chưa có trong workspace. C01 xử lý mời thành viên/chốt assignee. Không gán công
việc cho bot hay tự bật quyền chạy agent.

Mỗi PR ghi issue Linear URL/ID; mỗi issue đính PR, commit hoặc run manifest tương
ứng. Hiện đã gắn PR #11 với F01. Tích hợp GitHub–Linear tự động chưa được cấu hình;
cập nhật trạng thái/đính liên kết bằng tay cho tới khi nhóm bật integration phù hợp.

## Backlog đã tạo

Tất cả 25 issue có due date, priority, milestone, vai trò, acceptance criteria và
dependency khi cần; 31 quan hệ blocked-by được khai báo lúc khởi tạo.

| Key / Linear | Công việc | Tuần | Hạn | Vai trò |
|---|---|---:|---|---|
| [F01 / TRA-70](https://linear.app/traffic-agent-assistant/issue/TRA-70/f01-review-va-tich-hop-pr-nen-tang-bmaddata) | Review và tích hợp PR nền tảng BMAD/data | 1 | 2026-10-27 | Data + teammate reviewer |
| [C01 / TRA-71](https://linear.app/traffic-agent-assistant/issue/TRA-71/c01-them-2-thanh-vien-va-chot-vai-tro-bc-tren-linear) | Thêm 2 thành viên và chốt vai trò B/C trên Linear | 1 | 2026-10-27 | Điều phối / Chí Vĩ |
| [D01 / TRA-72](https://linear.app/traffic-agent-assistant/issue/TRA-72/d01-chot-nguon-djieu-kien-su-dung-va-dataset-card-bmad) | Chốt nguồn, điều kiện sử dụng và dataset card BMAD | 1 | 2026-10-28 | Data / Chí Vĩ |
| [R01 / TRA-73](https://linear.app/traffic-agent-assistant/issue/TRA-73/r01-djoc-andi-va-pin-hop-djong-adaptation-flair-2d) | Đọc ANDi và pin hợp đồng adaptation FLAIR 2D | 1 | 2026-10-28 | C — Diffusion, B review |
| [D02 / TRA-74](https://linear.app/traffic-agent-assistant/issue/TRA-74/d02-ban-giao-data-contract-va-kiem-chung-tren-may-thanh-vien) | Bàn giao data contract và kiểm chứng trên máy thành viên | 1 | 2026-10-29 | Data / Chí Vĩ |
| [I01 / TRA-75](https://linear.app/traffic-agent-assistant/issue/TRA-75/i01-djo-gpu-pilot-va-quyet-djinh-ngan-sach-train) | Đo GPU pilot và quyết định ngân sách train | 1 | 2026-10-30 | B + C, Chí Vĩ hỗ trợ data |
| [I02 / TRA-76](https://linear.app/traffic-agent-assistant/issue/TRA-76/i02-runner-tai-lap-checkpoint-resume-va-run-manifest) | Runner tái lập, checkpoint resume và run manifest | 2 | 2026-11-02 | C — Diffusion / tích hợp |
| [E01 / TRA-77](https://linear.app/traffic-agent-assistant/issue/TRA-77/e01-trien-khai-evaluator-pixelimage-va-threshold-validation) | Triển khai evaluator pixel/image và threshold validation | 2 | 2026-11-03 | B — Evaluation |
| [B01 / TRA-78](https://linear.app/traffic-agent-assistant/issue/TRA-78/b01-dae-coarse-noise-baseline-train-overfit-va-checkpoint) | DAE coarse-noise baseline: train, overfit và checkpoint | 2 | 2026-11-04 | B — Baseline |
| [A01 / TRA-79](https://linear.app/traffic-agent-assistant/issue/TRA-79/a01-andi-flair-train-model-va-full-interval-scorer) | ANDi-FLAIR: train model và full-interval scorer | 2 | 2026-11-05 | C — Diffusion |
| [E02 / TRA-80](https://linear.app/traffic-agent-assistant/issue/TRA-80/e02-djanh-gia-daeandi-djoi-chung-va-review-gate-tuan-2) | Đánh giá DAE/ANDi đối chứng và review gate tuần 2 | 2 | 2026-11-06 | B + C |
| [A02 / TRA-81](https://linear.app/traffic-agent-assistant/issue/TRA-81/a02-sparse-timestep-scoring-k81632-cung-checkpoint) | Sparse timestep scoring K=8/16/32 cùng checkpoint | 3 | 2026-11-10 | C — Diffusion |
| [E03 / TRA-82](https://linear.app/traffic-agent-assistant/issue/TRA-82/e03-benchmark-latencyvram-va-bieu-djo-ap-thoi-gian) | Benchmark latency/VRAM và biểu đồ AP–thời gian | 3 | 2026-11-11 | B — Evaluation, C hỗ trợ |
| [R02 / TRA-83](https://linear.app/traffic-agent-assistant/issue/TRA-83/r02-chon-bien-the-chinh-va-khoa-search-budget-validation) | Chọn biến thể chính và khóa search budget validation | 3 | 2026-11-13 | B + C, cả nhóm review |
| [R03 / TRA-84](https://linear.app/traffic-agent-assistant/issue/TRA-84/r03-lap-cau-hinh-cuoi-va-xac-nhan-repeatability) | Lặp cấu hình cuối và xác nhận repeatability | 4 | 2026-11-18 | B + C |
| [O01 / TRA-85](https://linear.app/traffic-agent-assistant/issue/TRA-85/o01-tuy-chon-normal-only-score-calibration-ablation) | Tùy chọn: normal-only score calibration + ablation | 4 | 2026-11-18 | C — Research extension |
| [F02 / TRA-86](https://linear.app/traffic-agent-assistant/issue/TRA-86/f02-freeze-model-threshold-va-protocol-truoc-test) | Freeze model, threshold và protocol trước test | 4 | 2026-11-20 | Cả nhóm / Chí Vĩ điều phối |
| [E04 / TRA-87](https://linear.app/traffic-agent-assistant/issue/TRA-87/e04-chay-held-out-test-theo-manifest-dja-freeze) | Chạy held-out test theo manifest đã freeze | 5 | 2026-11-23 | B — Evaluation |
| [E05 / TRA-88](https://linear.app/traffic-agent-assistant/issue/TRA-88/e05-patient-bootstrap-ci-va-bang-thong-ke-ket-qua) | Patient-bootstrap CI và bảng thống kê kết quả | 5 | 2026-11-25 | B — Evaluation |
| [D03 / TRA-89](https://linear.app/traffic-agent-assistant/issue/TRA-89/d03-phan-tich-loi-lesion-size-va-gioi-han-du-lieu) | Phân tích lỗi, lesion size và giới hạn dữ liệu | 5 | 2026-11-26 | Data / Chí Vĩ |
| [R04 / TRA-90](https://linear.app/traffic-agent-assistant/issue/TRA-90/r04-chot-bang-e0e1e2-va-djo-thi-ablation) | Chốt bảng E0/E1/E2 và đồ thị ablation | 5 | 2026-11-27 | B + C, Data review |
| [W01 / TRA-91](https://linear.app/traffic-agent-assistant/issue/TRA-91/w01-hoan-thien-bao-cao-va-citation) | Hoàn thiện báo cáo và citation | 6 | 2026-12-01 | Cả nhóm / Data phụ trách nguồn |
| [W02 / TRA-92](https://linear.app/traffic-agent-assistant/issue/TRA-92/w02-slide-bao-ve-cau-chuyen-ket-qua-va-gioi-han) | Slide bảo vệ: câu chuyện, kết quả và giới hạn | 6 | 2026-12-02 | Cả nhóm |
| [W03 / TRA-93](https://linear.app/traffic-agent-assistant/issue/TRA-93/w03-demo-va-huong-dan-tai-chay-tren-may-moi) | Demo và hướng dẫn tái chạy trên máy mới | 6 | 2026-12-02 | C — Tích hợp, B review |
| [W04 / TRA-94](https://linear.app/traffic-agent-assistant/issue/TRA-94/w04-dry-run-bao-ve-va-djong-goi-nop-djo-an) | Dry-run bảo vệ và đóng gói nộp đồ án | 6 | 2026-12-04 | Cả nhóm / Chí Vĩ điều phối |

## Liên hệ với GitHub issues cũ

| GitHub | Linear / phạm vi hiện tại |
|---|---|
| #1 Data benchmark/split | F01, D01, D02; PR #11 đóng #1 sau merge |
| #2 IF/AE baselines | B01 triển khai DAE theo phạm vi mới; IF không bắt buộc |
| #3 VAE/GAN | Ngoài MVP 6 tuần; không dùng issue cũ để tự mở rộng scope |
| #4 DDPM training | R01, I02, A01: ANDi-FLAIR adaptation normal-only |
| #5 DDIM/DDPM | A02, E03: full-vs-sparse ANDi scoring; DDIM chưa bắt buộc |
| #6 Evaluation | E01, E02, E04, E05, D03, F02 |
| #7 Novel hypothesis | R01, R02, R04, W01; O01 tùy chọn, không hứa novelty |
| #8 Collaboration/tracking | C01 và project Linear này |
| #9 Runner/provenance | I01, I02, W03 |

Không tự đóng issue GitHub cũ chỉ vì đã có task Linear. Sau khi nhóm duyệt scope,
có thể cập nhật/đóng issue trùng; Linear giữ trạng thái thực thi chi tiết.

## Snapshot khởi tạo — 25/10/2026

F01 In Review; 5 task tuần 1 Todo; 19 task còn lại Backlog; 0 Done.
Đã có một project update đầu tiên. Chưa có model được train hoặc kết quả nghiên cứu.
Cần review/merge PR, thêm hai thành viên, xác nhận deadline và đo pilot GPU.

