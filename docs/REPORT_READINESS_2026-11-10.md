# Mức sẵn sàng báo cáo và bàn giao review freeze — 10/11/2026

**Đủ để bắt đầu viết, chưa hoàn tất toàn bộ mục tiêu đánh giá cuối.** Người dùng
đang nhờ Hải Tuấn review/freeze. Agent giữ manifest ở candidate, không tự freeze,
đổi cấu hình hoặc mở test. Deadline hiện ghi trong PROJECT_PLAN là04/12/2026.

## Những phần đã có bằng chứng để viết

| Mục tiêu | Trạng thái | Nguồn cho báo cáo |
|---|---|---|
| Nguồn/QC/preprocessing/split | Hoàn thành bản patient-disjoint-v1 | DATA_PROTOCOL, DATA_AUDIT, dataset revision cố định |
| DDPM normal-only và checkpoint | Có G30 EMA và artifact thực | RESULT_TRACKING, checkpoint/hash trong candidate |
| Reconstruction và anomaly scoring | Có DDPM, pixel/SSIM, LPIPS và DDIM hai pha | RESEARCH, phương pháp/acceptance tương ứng |
| Metric/threshold/split protocol | Có evaluator,40calibration/43holdout | DAC-30 Done; evidence metric replay |
| Validation chính | Đã nghiệm thu T4 smoke2/full83 | COLAB_ACCEPTANCE_2026-11-08 |
| Nghiên cứu timestep/chia bước | Có thêm11full validation runs của C ngày09Nov | HF producer audit10Nov, bảng ablation bên dưới |
| Cache/resume của runner |4arms×43cached validation slices đạt |172score shards/60numeric metrics, evidence08Nov |
| Benchmark | Có số đo đúng scope của lượt T4 | Adapter164.659s/full83, allocated CUDA112928768bytes |
| Freeze/held-out test | **Chưa hoàn thành** | Đang chờ Hải Tuấn review/freeze, chưa mở test |
| Launcher four-arm final | **Chưa hoàn thành** | API model chính có; verified dataset factory/durable4arm wrapper còn thiếu |
| CI/reliability | **Còn blocker/giới hạn** | CI account failure; local274pass ngoài sandbox, rename Windows chưa rõ nguyên nhân |
| Báo cáo/slides/demo cuối | **Đang chuẩn bị** | REPORT_DRAFT có nội dung và số validation; chưa có final-test table |

Không cần triển khai thêm GAN/VAE/Isolation Forest để đủ phạm vi đã thống nhất.
DAE và các encoder mở rộng là tùy chọn; không biến mọi related work thành hạng mục
bắt buộc. Một seed là giới hạn phải báo thật, không điền ba seed dự kiến vào kết quả.

## Cập nhật HF09Nov đã kiểm

- Model revision mới: `03ec5b35cc134277b0ce4e76a0a315490b46bd21`,
  modified09Nov10:15:14UTC. Dataset vẫn revision
  `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.
- So với revision đã nghiệm thu, thêm22run manifests:11smoke/11full83;
  audit88JSON artifact checksums khớp producer manifests. Root/comparison metric
  JSON thống nhất; config thống nhất giữa manifest/results, threshold calibration
  được giữ khi chấm group holdout. Producer ghi completed và held_out_test_used=False.
- Đây là **inference ablation trên cùng checkpoint G30**, không train11model mới.
 11full runs dùng cùng checkpoint/index/dataset hashes; mỗi output báo50U-Net calls.
- Các manifest ghi code_revision c02addb, working_tree_dirty=True và4source ZIP
  digests. Không dùng tên Git branch để thay cho source ZIP thực. Đợt này chỉ
  kiểm JSON/hashes, chưa tải/replay private arrays hoặc review4source archives mới.
- [Evidence tổng hợp](evidence/2026-11-10-hf-producer-update.json) và
  [bảng CSV](evidence/2026-11-10-ddim-ablation-table.csv) dùng để review/soạn bảng.

### Final LPIPS — validation exploratory, một seed17

Tất cả hàng mới:83validation,40calibration/43group-holdout, crop240, pixel AP
global micro. AP calibration được ghi riêng để tránh chọn theo AP holdout.
Nhóm đã xem holdout nhiều lần; đây không phải untouched test. Audit JSON xác nhận
count/index pins, chưa độc lập xác minh membership bằng arrays của mọi run mới.

| Bước pha1+pha2 | t* | AP calibration | AP holdout | Dice holdout |
|---|---:|---:|---:|---:|
|10+40|300|.316494|.290202|.361368|
|20+30|100|.264552|.227742|.277763|
|20+30|200|.306747|.291391|.347222|
|20+30|500|.320219|.321148|.373395|
|20+30|800|.129664|.186101|.266069|
|25+25|300|.314489|.312615|.370732|
|30+20|300|.312250|.316214|.376996|
|30+20|100|.262530|.230805|.276962|
|30+20|200|.303725|.291021|.347028|
|30+20|500|.318400|.328809|.379883|
|30+20|800|.136095|.192461|.275502|

Ứng viên20+30/t300 đã nghiệm thu08Nov có holdout AP.312330/Dice.371215;
không tự thay candidate bằng hàng có holdout cao nhất. Trong11hàng mới,
calibration AP cao nhất thuộc20+30/t500, còn holdout AP cao nhất thuộc30+20/t500.
Sự khác nhau này là lý do phải ghi trước quy tắc chọn và review, không chọn cột
thuận lợi sau khi đã xem kết quả. Brightness control holdout AP vẫn.389506 ở mọi
run mới. Không có bằng chứng các biến thể mới vượt brightness về pixel AP.
Các chênh lệch nhỏ chưa có CI/multiple-seed evidence riêng; không suy tính mới
hoặc cải thiện chắc chắn từ một seed và một cohort đã khảo sát.

## Handoff cụ thể cho Hải Tuấn

1. Review source ZIP thực của cấu hình muốn freeze, dataset/index/partition,
   checkpoint/sampler/LPIPS/mask, và giải thích quy tắc lựa chọn ứng viên. Giữ
   distinction source mới09Nov với source d5889d5 đã nghiệm thu08Nov.
2. Ghi quyết định **giữ20+30/t300** hoặc đề nghị cấu hình khác kèm lý do từ protocol/
   calibration. Không chọn lại bằng test. Thay cấu hình/source cần cập nhật pins,
   score/mask thresholds và validation acceptance, không chỉ sửa status JSON.
3. Review cả4controls, seed17/một seed, ROI, metric aggregation, threshold rule,
   và giới hạn validation đã bị quan sát lặp. Ghi reviewer/ngày/commit rõ ràng.
4. Quyết định xử lý CI account hoặc ngoại lệ local checks; ghi đầy đủ failure
   Windows và local274pass. User chưa chấp thuận ngoại lệ này.
5. Sau review mới freeze hồ sơ; launcher/runtime validation acceptance vẫn cần
   hoàn thiện trước final-test execution. Mọi thay đổi sau test phải khai báo.

## Viết gì ngay trong lúc chờ review

- Chương1: bài toán/mục tiêu/phạm vi, đóng góp tái triển khai và kiểm chứng.
- Chương2: DDPM/DDIM/AnoDDPM/AutoDDPM, anomaly scoring và metric;
  phân biệt công trình gốc với adaptation của nhóm.
- Chương3: audit dữ liệu, preprocessing, normal-only training và pipeline hai pha.
- Chương4: validation protocols, bảng kết quả đã nghiệm thu và ablation09Nov với
  nhãn producer-audited/exploratory, chưa viết thành final-test result.
- Chương5: thảo luận/giới hạn và kết luận tạm. Sau freeze/test mới chốt bảng cuối.

Viết nội dung song song với review giúp giữ tiến độ; không cần đợi test để viết
phương pháp. Mẫu trình bày của trường, hình minh họa an toàn, slides và demo vẫn
cần hoàn thiện; không gọi REPORT_DRAFT hiện tại là bản nộp hoàn chỉnh.
