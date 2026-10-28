# Kế hoạch đồ án — DDPM reconstruction cho BMAD Brain

Theo [PROJECT_REQUIREMENTS](PROJECT_REQUIREMENTS.md), [RESEARCH](RESEARCH.md) và [Linear](PROJECT_TRACKING.md). Quyết định ngày 28/10/2026: thực hiện đề xuất của thành viên về partial DDPM reconstruction, `t*` và anomaly score thay kế hoạch ANDi-FLAIR ngày 24/10. Deadline 04/12/2026 giữ nguyên. Chưa có kết quả model.

MVP: train DDPM trên normal slice, reconstruction với các `t*` được đăng ký trước, so pixel absolute difference và local SSIM trên cùng output, đánh giá validation rồi freeze và đánh giá test. DAE là đối chứng khi khả thi; feature difference, ANDi và các baseline khác chỉ là mở rộng khi nhóm có thời gian và nguồn lực.

## Roadmap và gate

| Tuần | Hạn | Gate và đầu ra |
|---|---|---|
| W1 | 30/10 | Chốt owner, nguồn/data contract, thiết kế DDPM và score, quy tắc ghi run; pilot nhỏ khi đủ điều kiện, không coi quota là đã xác minh. |
| W2 | 06/11 | DDPM có train/reconstruction smoke + checkpoint; evaluator, pixel/SSIM có đường chạy validation; DAE pilot khi khả thi. Ghi lỗi và chi phí thực. |
| W3 | 13/11 | So `t*` và score cùng checkpoint trên validation; đo latency/VRAM; chốt ứng viên và budget. |
| W4 | 20/11 | Lặp cấu hình nếu khả thi; freeze checkpoint, sampler, scores, thresholds, protocol trước test và được review. |
| W5 | 27/11 | Test theo freeze; thống kê theo patient nếu khả thi; phân tích lỗi và hạn chế. |
| W6 | 04/12 | Report, slide, demo tái chạy; số liệu khớp artifacts. Bắt đầu viết khung báo cáo từ W1/W2. |

Vai trò A/B/C được xác nhận ngày 28/10/2026; B và C đã được thêm vào project Linear. Các ngày là mục tiêu, không cam kết đã đủ GPU hay nhân lực. Nếu chậm, giảm các nhánh tùy chọn và sweep trước khi rút ngắn freeze/test. Ba seed và patient-bootstrap là mục tiêu tốt khi khả thi; báo số thực hoàn thành.

## Vai trò đã xác nhận

- A — Chu Nguyen Tuan Anh / project-data: nguồn, quyền, version, QC, handoff, điều phối, review protocol và freeze.
- B — Lê Hoàng Chí Vĩ / evaluation: evaluator, pixel/SSIM scores, threshold, DAE khi khả thi, metric, bootstrap và phân tích; review C. Cùng C chốt interface score.
- C — Hai Tuan / diffusion-integration: contract DDPM, trainer, partial reconstruction, `t*`, checkpoint, runner và demo; B review.
- Cả nhóm: pilot, chọn validation, freeze, phân tích, báo cáo và bảo vệ.

## Thứ tự trước khi dùng GPU

1. Xác nhận quyền dataset trên máy từng thành viên; hoàn tất nguồn và dataset card.
2. Review contract DDPM, score, ROI và protocol; chuẩn bị runner ghi run bắt đầu, lỗi/gián đoạn và artifact khi có. Thiết kế evaluator có thể làm song song.
3. Chạy smoke/pilot giới hạn sau khi contract và data access sẵn sàng; ghi GPU, batch, thời gian, VRAM và chi phí thực; ước lượng trước run dài.
4. Train/validation, đăng ký tập score và timestep hữu hạn, freeze rồi mới test.

Không khẳng định synthetic/planned output là kết quả. Mọi run bắt đầu ghi trong [RESULT_TRACKING](RESULT_TRACKING.md), kể cả run lỗi hoặc gián đoạn.
