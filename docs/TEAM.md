# Vai trò và cách phối hợp

Phân công cũ trong repo chỉ là tham khảo. Theo trao đổi ngày 24/10/2026, người dùng
phụ trách nguồn tham khảo và data; hai vai trò còn lại phân theo năng lực/thời gian.

| Vai trò | Trách nhiệm | Người review |
|---|---|---|
| Bạn — Data | Quyền sử dụng/nguồn, version, manifest, decode, QC, mô tả dataset | B |
| B — DAE & evaluation | Baseline, metric, threshold, bootstrap, log thí nghiệm | C |
| C — Diffusion | DDPM partial reconstruction, noise timestep, checkpoint, demo tích hợp | Bạn (protocol), B (code/score) |

Mỗi thay đổi split, label, preprocessing tạo version mới và có người thứ hai review.
Không đẩy ảnh, manifest chứa ID hoặc checkpoint vào Git. Không dùng chung secrets.
Mọi run có config, seed, commit, dataset hash, checkpoint hash và môi trường.

Quyết định nghiên cứu ghi trong `docs/RESEARCH.md`; tiến độ theo `docs/PROJECT_PLAN.md`.
Không âm thầm đổi thiết kế sau khi xem kết quả test. Việc bàn giao gồm lệnh chạy,
đầu vào, đầu ra, bằng chứng kiểm tra và giới hạn còn tồn tại.
