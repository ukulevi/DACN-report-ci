# Blockers and team handoff

File này theo dõi các blocker chưa giải quyết cần thành viên khác hỗ trợ hoặc
cần quyết định chung. Linear vẫn là nguồn chính cho owner/status của task;
`RESULT_TRACKING.md` vẫn là nguồn chính cho provenance và metric của các run.

## Cách sử dụng

- Chỉ tạo blocker khi công việc không thể tiếp tục an toàn do thiếu quyền truy
  cập, artifact, prerequisite, quyết định của nhóm hoặc thay đổi từ hệ thống
  bên ngoài. TODO thông thường và experiment đang chạy đúng dự kiến không phải
  blocker.
- Dùng ID `BLK-YYYYMMDD-NN`. Cập nhật entry hiện có thay vì tạo trùng.
- Status hợp lệ: `Open`, `Investigating`, `Waiting`, `Resolved`.
- Khi handoff, ghi rõ người/role tiếp theo, hành động cụ thể và điều kiện đóng.
- Khi giải quyết xong, đổi status thành `Resolved`, thêm bằng chứng và ngày đóng;
  không xóa lịch sử.
- Không ghi token, credential, patient ID, đường dẫn nguồn bệnh nhân, ảnh y tế
  riêng tư hoặc seed suy ra cho từng sample.

## Bảng theo dõi

| ID | Blocker | Status | Owner / next handoff | Affects | Last updated |
|---|---|---|---|---|---|
| `BLK-20261101-01` | Primary DDPM run mới hoàn thành epoch 10/30 | Open | C — Hai Tuan | Checkpoint 20/30, kết luận convergence | 2026-11-01 |
| `BLK-20261101-02` | Pixel anomaly map có nhiều false positive | Investigating | B — Chí Vĩ; C — Hai Tuan | Score/model freeze trước held-out test | 2026-11-01 |

## Blocker details

### BLK-20261101-01 — Primary DDPM run chưa hoàn thành

- **Status:** Open
- **Reported / updated:** 2026-11-01 / 2026-11-01
- **Owner / next handoff:** C — Hai Tuan xác minh provenance và resume; người có
  quyền shared Modal Workspace hỗ trợ nếu access thay đổi.
- **Affected work:** Chưa có checkpoint epoch 20/30, chưa thể phân biệt rõ
  undertraining với hạn chế của reconstruction/score, và chưa đủ điều kiện chốt
  model cuối.
- **Evidence:** Run
  `ddpm-partial-a100-s17-e30-milestones-rngfix-20261031T041542Z` dừng trong
  epoch 11. Epoch 10 đã hoàn thành 75.000 optimizer steps; normal-validation
  epsilon-MSE tốt nhất là `0.0101424818`. Modal Volume giữ `ddpm_last.pt`,
  `ddpm_best.pt` và milestone epoch 5/10. Checkpoint epoch 10 cùng manifest,
  history và source snapshot đã được backup vào private Hugging Face model
  revision `07ec8b455a3318a267df5568f7af3fa4f15d7a3c`. Chi tiết provenance nằm
  trong [RESULT_TRACKING.md](RESULT_TRACKING.md).
- **Attempts / constraints:** Resume chỉ hợp lệ khi cùng run ID, config, trainer,
  dataset revision và source snapshot SHA256 đã lưu. Phần epoch 11 chạy dở không
  có trong checkpoint và sẽ chạy lại từ đầu.
- **Action needed:** Xác minh `ddpm_last.pt`, manifest và exact source snapshot
  còn khớp; resume cùng run từ epoch 11; giữ milestone epoch 20/30; cập nhật
  `RESULT_TRACKING.md` từ artifact thực tế. Không dùng held-out test.
- **Close when:** Epoch 30 hoàn thành hoặc nhóm ghi quyết định có bằng chứng rằng
  run sẽ không tiếp tục; mọi checkpoint/status cuối đã được lưu và track.
- **Resolution:** Chưa có.

### BLK-20261101-02 — Pixel anomaly map có nhiều false positive

- **Status:** Investigating
- **Reported / updated:** 2026-11-01 / 2026-11-01
- **Owner / next handoff:** B — Chí Vĩ dẫn evaluation; C — Hai Tuan cung cấp
  checkpoint/reconstruction và cùng review inference contract.
- **Affected work:** Chưa nên freeze checkpoint/score/threshold để chạy held-out
  test hoặc đưa ra kết luận nghiên cứu cuối.
- **Evidence:** Trên 83 validation slices, candidate epoch 10, `t*=300`, local
  one-minus-SSIM đạt pixel AUROC `0.892010` nhưng Precision `0.101571`, Recall
  `0.395303`, micro F1/Dice `0.161616` và pixel AP `0.095895`. Đây là một
  inference seed; test chưa được dùng. Heatmap định tính cho thấy score cao ở
  nhiều mô bình thường, phù hợp với giả thuyết false positive nhưng không tự nó
  chứng minh bug.
- **Attempts / constraints:** Đã so checkpoint epoch 5/10, timestep 100/200/300
  và pixel-difference/local-SSIM theo validation protocol. Noise-MSE thấp đo
  epsilon prediction trên normal validation, không đo segmentation quality.
- **Action needed:** Trước hết đánh giá checkpoint epoch 20/30 bằng cùng
  validation protocol và seed. Báo empty-mask pixel false-positive rate, TP/FP/FN,
  lỗi theo lesion size và epsilon-MSE tại timestep 100/200/300. Nếu vẫn thấp,
  đăng ký và review riêng mọi ablation về sampler, foreground mask,
  post-processing hoặc score; không tune bằng test.
- **Close when:** Nhóm có diagnosis dựa trên artifact, chọn/freeze protocol hoặc
  phê duyệt ablation mới, và ghi rõ giới hạn cùng số seed thực chạy.
- **Resolution:** Chưa có.

## Mẫu blocker mới

```markdown
### BLK-YYYYMMDD-NN — Tiêu đề ngắn

- **Status:** Open | Investigating | Waiting | Resolved
- **Reported / updated:** YYYY-MM-DD / YYYY-MM-DD
- **Owner / next handoff:**
- **Affected work:**
- **Evidence:**
- **Attempts / constraints:**
- **Action needed:**
- **Close when:**
- **Resolution:** Chưa có hoặc bằng chứng đóng blocker và ngày đóng.
```
