# Vai trò và cách phối hợp

## Cập nhật 07/11/2026

Nhóm đã thống nhất làm chung, không phân biệt ranh giới role. Ưu tiên công việc
theo prerequisite và bằng chứng nghiệm thu; bảng bên dưới giữ lịch sử phân công.
Workspace Linear mặc định của dự án là **DACN-KHMT**
(`https://linear.app/dacn-khmt`). Quyền OAuth của connector cần trỏ đúng workspace;
đổi tài liệu hoặc workspace đang mở trong browser không tự đổi kết nối.

Phân công lịch sử được nhóm xác nhận ngày 28/10/2026. A là project lead Tuấn Anh;
người dùng trong cuộc hội thoại này là B — Chí Vĩ.

| Vai trò | Trách nhiệm | Người review |
|---|---|---|
| A — Chu Nguyen Tuan Anh — Data/project lead | Quyền sử dụng/nguồn, version, manifest, decode, QC, mô tả dataset, điều phối và freeze | B |
| B — Lê Hoàng Chí Vĩ — Evaluation | Evaluator, pixel/SSIM score, threshold, DAE nếu khả thi, bootstrap và phân tích metric | C |
| C — Hai Tuan — Diffusion | DDPM partial reconstruction, noise timestep, checkpoint, run integration và demo | A (protocol), B (code/score) |

Mỗi thay đổi split, label, preprocessing tạo version mới và có người thứ hai review.
Không đẩy ảnh, manifest chứa ID hoặc checkpoint vào Git. Không dùng chung secrets.
Mọi run có config, seed, commit, dataset hash, checkpoint hash và môi trường.

Quyết định nghiên cứu ghi trong `docs/RESEARCH.md`; tiến độ theo `docs/PROJECT_PLAN.md`.
Không âm thầm đổi thiết kế sau khi xem kết quả test. Việc bàn giao gồm lệnh chạy,
đầu vào, đầu ra, bằng chứng kiểm tra và giới hạn còn tồn tại.
