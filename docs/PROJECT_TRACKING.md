# Theo dõi dự án bằng Linear

**Project:** [DACN — Diffusion Models for Anomaly Detection](https://linear.app/dacn-khmt/project/dacn-diffusion-models-for-anomaly-detection-39ad392101f4)

**Team:** DACN-KHMT; **target:** 04/12/2026.
**Snapshot:** 29/10/2026; trạng thái hiện hành xem trực tiếp Linear.

## Quyết định và phạm vi

- Quyết định 28/10: DDPM partial reconstruction train normal slice, nghiên cứu starting timestep và pixel difference/local SSIM. ANDi là related work hoặc optional extension.
- Bắt đầu với ứng viên t = 100, 200, 300 nếu hợp lệ với schedule đã chốt; 50/500 tùy budget. Đây là phương án, không phải timestep tối ưu đã đo.
- DAE đối chứng khi khả thi; feature difference/ANDi/VAE/Isolation Forest là tùy chọn. Chưa có kết quả model được xác minh.
- Dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`; giữ patient-disjoint, train normal-only, chọn model/score/threshold trên validation, freeze trước held-out test.
- Metric image/pixel ROC-AUC, Precision, Recall, F1, pixel AP, Dice, latency và VRAM theo protocol; không ghi patient ID, ảnh riêng tư hoặc secrets vào Linear.

## Mốc tuần

| Mốc | Hạn | Gate |
|---|---|---|
| W1 — Data, DDPM & score protocol | 30/10 | Chốt data, owner, DDPM/score contract và chuẩn bị run recording. |
| W2 — DDPM smoke, scoring & pilot | 06/11 | DDPM smoke, evaluator, pixel/SSIM; GPU pilot giới hạn và DAE nếu khả thi. |
| W3 — Timestep & anomaly-score study | 13/11 | Review gate, so t/score trên validation cùng checkpoint; benchmark. |
| W4 — Repeatability & pre-test freeze | 20/11 | Lặp khi khả thi; review và freeze model, t, sampler, score, ngưỡng trước test. |
| W5 — Held-out evaluation & analysis | 27/11 | Test theo freeze, uncertainty nếu khả thi, phân tích lỗi. |
| W6 — Report, demo & defense | 04/12 | Bàn giao report, slide, demo; viết khung báo cáo sớm. |

## Phân công và workflow

A — Chu Nguyen Tuan Anh — phụ trách điều phối/data; B — Lê Hoàng Chí Vĩ — phụ trách evaluation, pixel/SSIM và DAE khi khả thi; C — Hai Tuan — phụ trách DDPM, reconstruction và integration. B/C đã được thêm vào project Linear ngày 28/10/2026. Mỗi người cập nhật evidence, blocker và bước tới; Done cần acceptance criteria và review. Các run bắt đầu, lỗi hoặc ngắt được ghi trong RESULT_TRACKING; model result chỉ đến từ artifact thực.

Runner DAC-11 có thể làm trước pilot DAC-10. DAE DAC-13 không chặn gate DDPM DAC-15. DAC-30 có issue riêng cho pixel/SSIM. Các mốc là mục tiêu và cần điều chỉnh nếu pilot cho thấy không đủ tài nguyên.

## Backlog snapshot

| Linear | Công việc | W | Hạn | Vai trò dự kiến | Status | Blocked by |
|---|---|---:|---|---|---|---|
| [DAC-5](https://linear.app/dacn-khmt/issue/DAC-5/f01-review-va-tich-hop-pr-nen-tang-bmaddata) | Review và tích hợp PR nền tảng BMAD/data | 1 | 2026-10-27 | Data lead; teammate reviewer | Done | — |
| [DAC-6](https://linear.app/dacn-khmt/issue/DAC-6/c01-them-thanh-vien-va-chot-vai-tro-bc-tren-linear) | Thêm thành viên và chốt vai trò B/C trên Linear | 1 | 2026-10-30 | A — Project lead | In Progress | — |
| [DAC-7](https://linear.app/dacn-khmt/issue/DAC-7/d01-chot-nguon-djieu-kien-su-dung-va-dataset-card-bmad) | Chốt nguồn, điều kiện sử dụng và dataset card BMAD | 1 | 2026-10-29 | Data lead | In Progress | — |
| [DAC-8](https://linear.app/dacn-khmt/issue/DAC-8/r01-chot-hop-djong-ddpm-partial-reconstruction-va-score) | Chốt hợp đồng DDPM partial reconstruction và score | 1 | 2026-10-29 | C — Hai Tuan; B (Chí Vĩ) reviews evaluation | Todo | — |
| [DAC-9](https://linear.app/dacn-khmt/issue/DAC-9/d02-ban-giao-data-contract-va-kiem-chung-tren-may-thanh-vien) | Bàn giao data contract và kiểm chứng trên máy thành viên | 1 | 2026-10-30 | A — Data lead; B (Chí Vĩ) and C (Hai Tuan) verify access locally | In Progress | DAC-7, DAC-5 |
| [DAC-10](https://linear.app/dacn-khmt/issue/DAC-10/i01-djo-gpu-pilot-va-quyet-djinh-ngan-sach-train) | Đo GPU pilot và quyết định ngân sách train | 2 | 2026-11-04 | C — Hai Tuan lead; B DAE pilot; A data access | Todo | DAC-11, DAC-9, DAC-8 |
| [DAC-11](https://linear.app/dacn-khmt/issue/DAC-11/i02-run-lifecycle-checkpoint-resume-va-manifest) | Run lifecycle, checkpoint resume và manifest | 2 | 2026-11-02 | C — Hai Tuan | Backlog | — |
| [DAC-12](https://linear.app/dacn-khmt/issue/DAC-12/e01-trien-khai-evaluator-pixelimage-va-threshold-validation) | Triển khai evaluator pixel/image và threshold validation | 2 | 2026-11-03 | B — Lê Hoàng Chí Vĩ; C reviews scorer interface | Backlog | DAC-9 |
| [DAC-13](https://linear.app/dacn-khmt/issue/DAC-13/b01-dae-coarse-noise-baseline-train-overfit-va-checkpoint) | DAE coarse-noise baseline: train, overfit và checkpoint | 2 | 2026-11-06 | B — Lê Hoàng Chí Vĩ; optional comparison | Backlog | DAC-9, DAC-10, DAC-11 |
| [DAC-14](https://linear.app/dacn-khmt/issue/DAC-14/a01-ddpm-trainer-va-partial-reconstruction) | DDPM trainer và partial reconstruction | 2 | 2026-11-05 | C — Hai Tuan | Backlog | DAC-8, DAC-11, DAC-9 |
| [DAC-15](https://linear.app/dacn-khmt/issue/DAC-15/e02-review-gate-ddpm-reconstruction-va-evaluator) | Review gate: DDPM reconstruction và evaluator | 3 | 2026-11-09 | B — Chí Vĩ accountable; C co-review | Backlog | DAC-30, DAC-14, DAC-12 |
| [DAC-16](https://linear.app/dacn-khmt/issue/DAC-16/a02-so-sanh-timestep-bat-djau-ddpm-reconstruction) | So sánh timestep bắt đầu DDPM reconstruction | 3 | 2026-11-10 | C — Hai Tuan; B reviews evaluation | Backlog | DAC-15 |
| [DAC-17](https://linear.app/dacn-khmt/issue/DAC-17/e03-benchmark-timestep-scores-latency-va-vram) | Benchmark timestep, scores, latency và VRAM | 3 | 2026-11-11 | B — Lê Hoàng Chí Vĩ; C supplies reconstruction details | Backlog | DAC-30, DAC-16 |
| [DAC-18](https://linear.app/dacn-khmt/issue/DAC-18/r02-chon-timestepscore-va-khoa-validation-budget) | Chọn timestep/score và khóa validation budget | 3 | 2026-11-13 | B — Chí Vĩ accountable; C co-review | Backlog | DAC-17 |
| [DAC-19](https://linear.app/dacn-khmt/issue/DAC-19/r03-lap-cau-hinh-cuoi-va-xac-nhan-repeatability) | Lặp cấu hình cuối và xác nhận repeatability | 4 | 2026-11-18 | B — Chí Vĩ accountable; C runs/reviews model repeats | Backlog | DAC-18 |
| [DAC-20](https://linear.app/dacn-khmt/issue/DAC-20/o01-optional-feature-difference-hoac-andi) | Optional: feature difference hoặc ANDi | 4 | 2026-11-18 | C — Hai Tuan; B supports optional extension | Backlog | DAC-18 |
| [DAC-21](https://linear.app/dacn-khmt/issue/DAC-21/f02-freeze-model-threshold-va-protocol-truoc-test) | Freeze model, threshold và protocol trước test | 4 | 2026-11-20 | Whole team; project lead records the freeze | Backlog | DAC-19 |
| [DAC-22](https://linear.app/dacn-khmt/issue/DAC-22/e04-chay-held-out-test-theo-manifest-dja-freeze) | Chạy held-out test theo manifest đã freeze | 5 | 2026-11-23 | B — Lê Hoàng Chí Vĩ | Backlog | DAC-21 |
| [DAC-23](https://linear.app/dacn-khmt/issue/DAC-23/e05-patient-bootstrap-ci-va-bang-thong-ke-ket-qua) | Patient-bootstrap CI và bảng thống kê kết quả | 5 | 2026-11-25 | B — Lê Hoàng Chí Vĩ | Backlog | DAC-22 |
| [DAC-24](https://linear.app/dacn-khmt/issue/DAC-24/d03-phan-tich-loi-lesion-size-va-gioi-han-du-lieu) | Phân tích lỗi, lesion size và giới hạn dữ liệu | 5 | 2026-11-26 | A — data lead; B supplies aggregate evaluation output | Backlog | DAC-22 |
| [DAC-25](https://linear.app/dacn-khmt/issue/DAC-25/r04-bang-ddpm-timestepscore-va-phan-tich) | Bảng DDPM timestep/score và phân tích | 5 | 2026-11-27 | B — Chí Vĩ accountable; C supplies recon; A reviews claims | Backlog | DAC-24, DAC-23 |
| [DAC-26](https://linear.app/dacn-khmt/issue/DAC-26/w01-hoan-thien-bao-cao-va-citations) | Hoàn thiện báo cáo và citations | 6 | 2026-12-01 | Whole team; A owns source/citation checks | Backlog | DAC-25 |
| [DAC-27](https://linear.app/dacn-khmt/issue/DAC-27/w02-slide-bao-ve-cau-chuyen-ket-qua-va-gioi-han) | Slide bảo vệ: câu chuyện, kết quả và giới hạn | 6 | 2026-12-02 | Whole team; A coordinates | Backlog | DAC-26 |
| [DAC-28](https://linear.app/dacn-khmt/issue/DAC-28/w03-demo-va-huong-dan-tai-chay-tren-may-moi) | Demo và hướng dẫn tái chạy trên máy mới | 6 | 2026-12-02 | C — Hai Tuan; B reviews the evaluation path | Backlog | DAC-25 |
| [DAC-29](https://linear.app/dacn-khmt/issue/DAC-29/w04-dry-run-bao-ve-va-djong-goi-nop-djo-an) | Dry-run bảo vệ và đóng gói nộp đồ án | 6 | 2026-12-04 | Whole team; A coordinates | Backlog | DAC-28, DAC-27 |
| [DAC-30](https://linear.app/dacn-khmt/issue/DAC-30/s01-pixel-difference-va-local-ssim-anomaly-maps) | Pixel difference và local SSIM anomaly maps | 2 | 2026-11-06 | B — Lê Hoàng Chí Vĩ; C reviews reconstruction interface | Backlog | DAC-14, DAC-12 |

`LINEAR_BACKLOG.csv` là snapshot tương ứng, không thay thế trạng thái live. Các tài liệu và PR cần teammate review theo CONTRIBUTING.
