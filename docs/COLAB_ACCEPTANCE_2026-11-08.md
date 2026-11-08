# Nghiệm thu Colab T4 — 08/11/2026

**Đạt nghiệm thu tái lập validation**, chưa frozen và chưa chạy test. G30 EMA,
DDIM20(t300)+30(t50), eta0, một phase2, cùng seed17 và threshold đã chốt.
Không train lại, không chọn lại threshold/checkpoint. Nhóm đã khảo sát validation
nhiều lần; 43 ảnh dưới đây không phải một tập nghiên cứu chưa từng quan sát.

## Artifact và kiểm chứng

- Smoke2/2 rồi full83/83; báo cáo riêng registered40cal/43holdout, 25 nhóm
  holdout tách bệnh nhân khỏi calibration. Membership lấy từ file đã pin hash,
  không tái chia bằng phiên bản sklearn khác.
- ZIP riêng tư đã tải qua Chrome, sao chép vào ignored
  `runs/colab-controls-20261107/dacn_g30_controls_registered_private.zip`.
  53,129,178 byte; SHA256
  `699af4f3791b9f33688abcbdd9ffcd44e7d33169964ab672ecc03b607be4162a`.
  ZIP và cả10 file trong manifest đều được kiểm checksum trước khi giải nén.
- Tính lại toàn bộ metric43holdout bằng CPU từ map đã tải, fixed thresholds;
  mọi trường numeric khớp báo cáo Colab trong tolerance1e-10. Input ảnh/mask/label
  của83validation đối chiếu cache C đều khớp. Không xuất nội dung riêng tư.
- Maxabs map so với cache C: final4.768371582e-7, initial4.991889e-7,
  naive-stitch3.576278687e-7, brightness0; cả4arms đạt atol1e-6.
- [Evidence tổng hợp](evidence/2026-11-08-colab-registered-acceptance.json)
  lưu hashes, metric và phạm vi kiểm tra; không chứa UUID/ảnh/derived seeds.

## Kết quả43holdout

Pixel metric global micro trên crop240x240; Dice bằng pixel F1. Image score là
mean top576pixels; thresholds riêng từng arm trong final_control_candidates.json,
lấy từ40calibration cũ và giữ nguyên. Một training/sampling seed17.

| Arm | Pixel AUROC | Pixel AP | Precision | Recall | F1/Dice |
|---|---:|---:|---:|---:|---:|
| Final LPIPS | .949036 | .312330 | .297471 | .493572 | .371215 |
| Initial LPIPS | .926110 | .157876 | .194958 | .336464 | .246871 |
| Naive-stitch LPIPS | .936299 | .180593 | .218753 | .461666 | .296849 |
| Brightness | .950197 | .389506 | .161469 | .788577 | .268051 |

| Arm | Image AUROC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Final LPIPS | .914474 | .687500 | .916667 | .785714 |
| Initial LPIPS | .787281 | .611111 | .916667 | .733333 |
| Naive-stitch LPIPS | .785088 | .617647 | .875000 | .724138 |
| Brightness | .679825 | .571429 | 1.000000 | .727273 |

Final macro Dice trên lesion slices .313571; normal-empty-mask pixel FPR .005772.
Brightness vẫn dẫn pixel AP; không kết luận diffusion vượt mọi baseline. Final
LPIPS có F1 cao hơn controls trong cohort này, chưa chứng minh lợi thế nhiều seed
hoặc trên test. Không so LPIPS-only ancestral với DDIM như cùng compute budget.

## Runtime và đính chính provenance

Full83:4150U-Net/249LPIPS calls; tổng adapter time164.659411s, không phải wall time
toàn notebook. Peak CUDA allocated112928768bytes (~107.7MiB), không phải reserved
VRAM hoặc toàn bộ memory GPU. T4, Torch2.14.1+cu130, torchvision0.29.1+cu130,
LPIPS0.1.4, numpy2.4.6; dependency snapshot là inventory phiên bản, không là
wheel/container lock hoàn chỉnh.

Source thực là archive **d5889d5**, SHA256
`a3c5345eb494b88cc75da95e7a7a588e58c0e864d7c039d77239f1f2affefd8b`.
Raw run_manifest kế thừa execution_source_commit84382a8, environment_scope CPU
và pending labels từ candidate. Các nhãn này lỗi thời, **không mô tả lượt T4**.
Evidence sidecar đính chính bằng source ZIP và environment receipt thực đã kiểm;
giữ nguyên raw bundle để audit. Metadata generator đã sửa với24kiểm tra CPU đạt;
archive comment chỉ là nhãn provenance, ZIP digest vẫn là định danh kiểm toàn vẹn.
không sửa ngược raw artifact và không train/rerun chỉ để sửa nhãn.

Cell chuẩn bị có một lần lỗi trước inference do nhập mã qua UI làm lệch thụt
dòng; khôi phục smoke/full literals từ generator trong source đã pin bằng AST.
Smoke/full sau đó hoàn tất. ZIP cũ07Nov chưa tải được trước runtime expiry;
không tuyên bố đã kiểm ZIP cũ. Runtime hiện vẫn kết nối, không còn cell chạy;
artifact đã bền vững ở local, có thể đóng phiên khi người dùng review xong.

## Việc tiếp theo

1. Đã sửa nhãn provenance kế thừa và kiểm generated manifest preflight bằng CPU;
   áp dụng generator mới cho notebook final, không sửa raw artifact cũ.
2. Repin candidate cho runtime cuối; nhóm review shortlist, seed, thresholds,
   controls và hạn chế một seed trước freeze. Không dùng test để quyết định.
3. Xử lý CI account/billing ở BLK-20261107-01; local verification không thay CI.
4. Sau freeze mới chạy final evaluation theo protocol; hoàn thiện báo cáo/demo
   bằng evidence đã có. BLK-20261101-02 và gate chung chưa tự động đóng.
