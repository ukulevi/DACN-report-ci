# Theo dõi dự án bằng Linear

**Project:** [DACN — Diffusion Models for Anomaly Detection](https://linear.app/dacn-khmt/project/dacn-diffusion-models-for-anomaly-detection-39ad392101f4)
**Team:** DACN-KHMT (key: DAC)
**Target:** 04/12/2026
**Status tại 28/10/2026:** In Progress; health at risk vì chưa chốt owner B/C và chưa có GPU pilot.

Linear là nguồn trạng thái công việc hiện hành; GitHub là nơi review và lưu code. Bảng bên dưới là snapshot lúc khởi tạo workspace chính.

## Mục tiêu và phạm vi

- ANDi adaptation cho BMAD Brain FLAIR 2D, train chủ yếu trên normal samples; anomaly scores cao hơn nghĩa là bất thường hơn.
- DAE là baseline. So sánh full-interval scoring với sparse timestep K=8/16/32 trên cùng checkpoint.
- Dùng split patient-disjoint đã pin; chọn model/ngưỡng bằng validation. Chỉ mở held-out test sau khi freeze protocol.
- Báo cáo image-level và pixel-level ROC-AUC, Precision, Recall, F1; pixel AP, Dice, latency, VRAM khi phù hợp. Ghi aggregation, split, threshold rule và số seed.
- VAE và Isolation Forest là ý tưởng tùy chọn, không tự mở rộng MVP.
- Dataset/checkpoint repo là private. Không ghi token, patient ID, source path hay dữ liệu y tế riêng tư vào Linear.

## Mốc tuần

| Milestone | Hạn | Gate |
|---|---|---|
| W1 — Data, protocol & GPU pilot | 30/10/2026 | Dataset/data contract tái lập; ANDi adaptation được review; pilot đo GPU/VRAM/throughput/chi phí thực tế |
| W2 — DAE & ANDi running | 06/11/2026 | Có DAE, ANDi scorer, evaluator validation-only, checkpoint và manifest |
| W3 — Full vs sparse timestep | 13/11/2026 | So sánh K=8/16/32 cùng checkpoint; chốt tối đa một biến thể chính |
| W4 — Repeatability & pre-test freeze | 20/11/2026 | Lặp cấu hình cuối; freeze model, score, threshold và protocol trước test |
| W5 — Held-out evaluation & analysis | 27/11/2026 | Test theo freeze; thống kê theo patient; phân tích lỗi và giới hạn |
| W6 — Report, demo & defense | 04/12/2026 | Báo cáo, slide, demo tái chạy và gói bảo vệ nhất quán với artifact |

Các task đầu tuần được rebaseline theo ngày tạo workspace: C01 sang 30/10; D01/R01 sang 29/10. Hạn cuối 04/12 không đổi.

## Phân công và nhịp phối hợp

- Project lead/data: các task nguồn, handoff data, phân tích giới hạn và điều phối đã gán cho người dùng Linear hiện tại.
- B: DAE và evaluation; C: ANDi/diffusion và integration. Chưa gán issue B/C tới khi xác nhận đúng danh tính Linear.
- C01 chốt thành viên, vai trò, owner rồi mới gán ticket. Không giao việc cho bot.
- Trước weekly sync, mỗi người cập nhật bằng chứng đã làm, blocker/decision cần hỗ trợ và bước kế tiếp. Giữ tối đa một issue chính ở In Progress mỗi người.

## Workflow và tiêu chí Done

- Backlog: chưa sẵn sàng bắt đầu; Todo: có thể nhận sau khi kiểm tra blockers; In Progress: đang làm; In Review: có PR/artifact cho teammate review.
- Done chỉ khi acceptance criteria đạt và có bằng chứng. PR mở chưa được xem là Done.
- Blocker dùng quan hệ blocked-by; khi vướng, ghi lý do, người cần xử lý và ngày cần gỡ.
- Mọi run bắt đầu, kể cả thất bại/gián đoạn, phải có manifest và được ghi trong docs/RESULT_TRACKING.md. Kế hoạch chạy không phải kết quả.

## Backlog snapshot

Tổng 25 issue: 1 Done, 5 Todo, 19 Backlog; 31 quan hệ blocked-by. F01 Done dựa trên PR #11 đã merge vào main. B/C còn để unassigned tới khi xác nhận thành viên. Project update ban đầu: [health at risk](https://linear.app/dacn-khmt/project/dacn-diffusion-models-for-anomaly-detection-39ad392101f4/activity#project-update-2b5db0fb).

| Key / Linear | Công việc | Tuần | Hạn | Owner role | Status |
|---|---|---:|---|---|---|
| [F01 / DAC-5](https://linear.app/dacn-khmt/issue/DAC-5/f01-review-va-tich-hop-pr-nen-tang-bmaddata) | Review và tích hợp PR nền tảng BMAD/data | 1 | 27/10/2026 | Data + reviewer | Done |
| [C01 / DAC-6](https://linear.app/dacn-khmt/issue/DAC-6/c01-them-thanh-vien-va-chot-vai-tro-bc-tren-linear) | Thêm thành viên và chốt vai trò B/C trên Linear | 1 | 30/10/2026 | Điều phối | Todo |
| [D01 / DAC-7](https://linear.app/dacn-khmt/issue/DAC-7/d01-chot-nguon-djieu-kien-su-dung-va-dataset-card-bmad) | Chốt nguồn, điều kiện sử dụng và dataset card BMAD | 1 | 29/10/2026 | Data | Todo |
| [R01 / DAC-8](https://linear.app/dacn-khmt/issue/DAC-8/r01-djoc-andi-va-pin-hop-djong-adaptation-flair-2d) | Đọc ANDi và pin hợp đồng adaptation FLAIR 2D | 1 | 29/10/2026 | C — Diffusion; B review | Todo |
| [D02 / DAC-9](https://linear.app/dacn-khmt/issue/DAC-9/d02-ban-giao-data-contract-va-kiem-chung-tren-may-thanh-vien) | Bàn giao data contract và kiểm chứng trên máy thành viên | 1 | 30/10/2026 | Data; B/C verify | Todo |
| [I01 / DAC-10](https://linear.app/dacn-khmt/issue/DAC-10/i01-djo-gpu-pilot-va-quyet-djinh-ngan-sach-train) | Đo GPU pilot và quyết định ngân sách train | 1 | 30/10/2026 | B + C; data hỗ trợ | Todo |
| [I02 / DAC-11](https://linear.app/dacn-khmt/issue/DAC-11/i02-runner-tai-lap-checkpoint-resume-va-run-manifest) | Runner tái lập, checkpoint resume và run manifest | 2 | 02/11/2026 | C — Diffusion/integration | Backlog |
| [E01 / DAC-12](https://linear.app/dacn-khmt/issue/DAC-12/e01-trien-khai-evaluator-pixelimage-va-threshold-validation) | Triển khai evaluator pixel/image và threshold validation | 2 | 03/11/2026 | B — Evaluation | Backlog |
| [B01 / DAC-13](https://linear.app/dacn-khmt/issue/DAC-13/b01-dae-coarse-noise-baseline-train-overfit-va-checkpoint) | DAE coarse-noise baseline: train, overfit và checkpoint | 2 | 04/11/2026 | B — Baseline | Backlog |
| [A01 / DAC-14](https://linear.app/dacn-khmt/issue/DAC-14/a01-andi-flair-trien-khai-trainer-va-full-interval-scorer) | ANDi-FLAIR trainer và full-interval scorer | 2 | 05/11/2026 | C — Diffusion | Backlog |
| [E02 / DAC-15](https://linear.app/dacn-khmt/issue/DAC-15/e02-djanh-gia-daeandi-djoi-chung-va-review-gate-tuan-2) | Đánh giá DAE/ANDi và review gate tuần 2 | 2 | 06/11/2026 | B + C | Backlog |
| [A02 / DAC-16](https://linear.app/dacn-khmt/issue/DAC-16/a02-sparse-timestep-scoring-k81632-cung-checkpoint) | Sparse timestep K=8/16/32 cùng checkpoint | 3 | 10/11/2026 | C — Diffusion | Backlog |
| [E03 / DAC-17](https://linear.app/dacn-khmt/issue/DAC-17/e03-benchmark-latencyvram-va-bieu-djo-ap-thoi-gian) | Benchmark latency/VRAM và đồ thị AP–thời gian | 3 | 11/11/2026 | B — Evaluation; C hỗ trợ | Backlog |
| [R02 / DAC-18](https://linear.app/dacn-khmt/issue/DAC-18/r02-chon-bien-the-chinh-va-khoa-search-budget-validation) | Chọn biến thể chính và khóa search budget validation | 3 | 13/11/2026 | B + C | Backlog |
| [R03 / DAC-19](https://linear.app/dacn-khmt/issue/DAC-19/r03-lap-cau-hinh-cuoi-va-xac-nhan-repeatability) | Lặp cấu hình cuối và xác nhận repeatability | 4 | 18/11/2026 | B + C | Backlog |
| [O01 / DAC-20](https://linear.app/dacn-khmt/issue/DAC-20/o01-optional-normal-only-score-calibration-va-ablation) | Optional: normal-only calibration và ablation | 4 | 18/11/2026 | C — extension | Backlog |
| [F02 / DAC-21](https://linear.app/dacn-khmt/issue/DAC-21/f02-freeze-model-threshold-va-protocol-truoc-test) | Freeze model, threshold và protocol trước test | 4 | 20/11/2026 | Cả nhóm | Backlog |
| [E04 / DAC-22](https://linear.app/dacn-khmt/issue/DAC-22/e04-chay-held-out-test-theo-manifest-dja-freeze) | Chạy held-out test theo manifest đã freeze | 5 | 23/11/2026 | B — Evaluation | Backlog |
| [E05 / DAC-23](https://linear.app/dacn-khmt/issue/DAC-23/e05-patient-bootstrap-ci-va-bang-thong-ke-ket-qua) | Patient-bootstrap CI và bảng thống kê kết quả | 5 | 25/11/2026 | B — Evaluation | Backlog |
| [D03 / DAC-24](https://linear.app/dacn-khmt/issue/DAC-24/d03-phan-tich-loi-lesion-size-va-gioi-han-du-lieu) | Phân tích lỗi, lesion size và giới hạn dữ liệu | 5 | 26/11/2026 | Data | Backlog |
| [R04 / DAC-25](https://linear.app/dacn-khmt/issue/DAC-25/r04-chot-bang-e0e1e2-va-djo-thi-ablation) | Chốt bảng E0/E1/E2 và đồ thị ablation | 5 | 27/11/2026 | B + C; Data review | Backlog |
| [W01 / DAC-26](https://linear.app/dacn-khmt/issue/DAC-26/w01-hoan-thien-bao-cao-va-citations) | Hoàn thiện báo cáo và citations | 6 | 01/12/2026 | Cả nhóm; Data/source review | Backlog |
| [W02 / DAC-27](https://linear.app/dacn-khmt/issue/DAC-27/w02-slide-bao-ve-cau-chuyen-ket-qua-va-gioi-han) | Slide bảo vệ: câu chuyện, kết quả và giới hạn | 6 | 02/12/2026 | Cả nhóm | Backlog |
| [W03 / DAC-28](https://linear.app/dacn-khmt/issue/DAC-28/w03-demo-va-huong-dan-tai-chay-tren-may-moi) | Demo và hướng dẫn tái chạy trên máy mới | 6 | 02/12/2026 | C — integration; B review | Backlog |
| [W04 / DAC-29](https://linear.app/dacn-khmt/issue/DAC-29/w04-dry-run-bao-ve-va-dong-goi-nop-djo-an) | Dry-run bảo vệ và đóng gói nộp đồ án | 6 | 04/12/2026 | Cả nhóm / điều phối | Backlog |

## Liên kết dự án

- Repository: https://github.com/tanh1c/DACN
- Dataset card: https://huggingface.co/datasets/dacn-bmad-team/bmad-flair-patient-disjoint-v1
- Dataset revision: fbb884dfe144be5c71e205a73701507163557ea4
- Runbook: docs/HUGGINGFACE_MODAL.md
- Result tracking: docs/RESULT_TRACKING.md
