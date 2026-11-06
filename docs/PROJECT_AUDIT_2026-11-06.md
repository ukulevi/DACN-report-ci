# Audit dự án DACN — 06/11/2026

> Snapshot dưới là audit trước tích hợp. Xem [review sau tích hợp](PROGRESS_REVIEW_2026-11-06.md)
> để biết CI đã success, replay/bootstrap đã hoàn thành và các guard còn cần nghiệm thu.

## Kết luận và phạm vi

Dự án đã vượt giai đoạn smoke: có data contract, DDPM được train, nhiều đối
chứng và evaluation thực. Hướng đáng xác nhận tiếp là **Gaussian EMA30 +
AutoDDPM-style DDIM20+30 + spatial LPIPS**, chưa cần train lại. Đây là adaptation,
không phải tái lập nguyên paper AutoDDPM hoặc phương pháp mới đã chứng minh.
Điểm nghẽn hiện tại là tích hợp source, kiểm chứng độc lập score/cache, đóng
review gate và freeze; mở thêm nhiều kiến trúc sẽ tăng nguy cơ chọn theo validation.

Audit này đọc source, chạy test tổng hợp và kiểm manifest/JSON; **không tái chạy
metric trên MRI mới, không training/GPU và không mở held-out test**. Không xem
ảnh reconstruction riêng tư; ảnh chụp người dùng gửi là bảng/bản tóm tắt kết quả.

## Snapshot các công cụ

| Nguồn | Đã kiểm tra | Khoảng trống |
|---|---|---|
| GitHub main | `da1eea4`, vẫn ở hồ sơ 28/10 | Chưa chứa các thí nghiệm B/C mới |
| Branch B | `codex/bc-evaluation-colab`, `a556a85` trước audit | Tích hợp C đến `7134b10`, chưa có cập nhật sau đó |
| Branch C | `feat/anoddpm-notebook-training`, mới nhất đã fetch `c02addb` | Gồm `d282f02`, `e75469b`, `c02addb`; chưa tích hợp sửa handoff/CI mới của B |
| HF dataset | Private; revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7` không đổi | Không cần chuẩn bị lại dữ liệu |
| HF model | Private; mới nhất đã đọc `fae9b50a4b78114fb51d87e803c77d019668f523` | Một số run lịch sử thiếu terminal status |
| Linear DACN-KHMT | UI: 26 issues, 4 Done / 6 In Progress / 16 Backlog; target 04/12 | Latest project update 28/10 còn sai vai trò; description còn pin revision cũ |
| Modal cá nhân | Profile/workspace `ukulevi`; app list trả rỗng | Không xác minh live app/billing của workspace C `phutungcogioihaitan` |
| Colab B | Notebook hiện Connect T4, full83 output completed còn lưu | Runtime đã ngắt; không reconnect |

Connector Linear vẫn trỏ `traffic-agent-assistant`; audit dùng UI đúng DACN-KHMT.
Không đổi owner/status trên Linear trong lượt này. GitHub còn PR20 và 8 issues
legacy mở; backlog cũ gồm GAN/VAE/Isolation Forest không phải phạm vi bắt buộc.
Workflow CI chỉ trigger push main hoặc PR, nên push branch riêng không tự chạy CI.

## Bằng chứng mới từ C

Các bảng dưới dùng cùng **43 slice internal group holdout / 25 groups**, threshold
fit trên 40 calibration bằng max F1 (score >= threshold, chọn threshold lớn nhất
khi hòa), full crop 240×240, pixel AP global micro. Validation đã
được xem nhiều lần; không gọi đây là test độc lập. Các phương pháp không cùng
sampler/compute được ghi riêng, không suy ra nhân quả từ bảng tổng hợp.

| Phương pháp | Pixel AP | Dice/F1 | Diễn giải |
|---|---:|---:|---|
| Gaussian30 ancestral300 + L1 | .101066 | .173678 | Baseline |
| Gaussian30 ancestral300 + LPIPS-only | .266899 | .339668 | Đổi score trên reconstruction có sẵn, không train lại |
| Ancestral AutoDDPM 50×5, weighted residual | .154785 | .226256 | Khác sampler/score với DDIM bên dưới |
| DDIM20 initial + LPIPS | .157876 | .246871 | Đối chứng đúng đầu vào cho pha 2 DDIM |
| DDIM20 + naive stitch + LPIPS | .180593 | .296849 | Đối chứng không denoise pha 2, đã được C bổ sung |
| DDIM20+30 final + LPIPS | **.312330** | **.371215** | Ứng viên diffusion ưu tiên xác nhận |
| Cùng final, LPIPS × residual | .196812 | .272453 | Nhân residual làm giảm AP trong trial này |
| Cùng final, normal-calibrated LPIPS | .307160 | .373649 | Dice tăng nhẹ nhưng metric chính AP giảm |
| Cùng final, frozen RAD-DINO | .289211 | .288010 | Chưa vượt LPIPS về localization |
| Brightness control | .389506 | .268051 | AP vẫn cao hơn; Dice ở threshold riêng thấp hơn |

DDIM final LPIPS: pixel AUROC .949036, Precision .297471, Recall .493572;
normal pixel FPR .005772. Image AUROC .914474, AP .944386, P .687500,
R .916667, F1 .785714. Không đánh đồng image detection với pixel segmentation.
Không gọi FPR thấp hơn là tốt hơn tại cùng sensitivity: brightness Recall .788577.

Manifest/results xác nhận 83 output × 50 U-Net calls = 4.150 calls, 166 LPIPS
calls, một lượt pha 2. `repeats=1` của cấu hình 20+30 khác default 5 của helper.
50 calls không bao gồm compute encoder LPIPS. Component time khoảng 1,477 s/ảnh
cho final LPIPS; total trước upload 186,57 s. Không dùng con số speedup trên hai
ảnh T4 khác container như benchmark end-to-end toàn bộ cohort.

Đối chứng naive-stitch LPIPS mới đã lấp khoảng trống trong báo cáo DDIM ban đầu:
AP .180593→.312330 hỗ trợ đóng góp pha 2 trong cấu hình cố định. Chưa chứng minh
thay tumor bằng mô khỏe về giải phẫu. Pseudo-mask còn bỏ sót khoảng 24,07% lesion
pixels holdout; ngoài mask bằng ảnh gốc là do thiết kế, không có nghĩa mọi mô lành
được nhận diện đúng.

Các hướng trước đó vẫn có giá trị là kết quả âm: matched unconditional fine-tune
SSIM AP .104004 gần A .104247/B .104084; spatial ImageNet .065459; K4
variance-normalized .045729; patch Mahalanobis .077657; memory-only .064212.
Không có lý do từ các số này để tự tăng epoch, K hoặc kích thước model.

**Mức xác minh:** tại HF `af0bf400`, 147 JSON được tải, 255 artifact references
được đối chiếu: 101 JSON bytes, 154 LFS metadata, không mismatch. Ba file
best/last/epoch5 được manifest run batch1 cũ tham chiếu nhưng không có ở snapshot
đó. Đợt cập nhật `fae9b50`, tải thêm 58 JSON và kiểm 94 references, không mismatch.
Checkpoint/NPZ lớn chỉ đối chiếu LFS metadata, không tuyên bố đã tải/deserialize.
Số liệu mới và hashes nằm trong [evidence](evidence/2026-11-06-project-audit-latest.json).
Xem [run receipt và liên kết HF bất biến](RESULT_TRACKING.md#2026-11-06--audit-receipt-for-cs-completed-runs).
Branch source được đọc tại [commit c02addb](https://github.com/tanh1c/DACN/tree/c02addbb3261076972cd7e8ae80b2d61764408d3).

## Đọc phản biện hội thoại Gemini

Đã đọc [hội thoại được gửi](https://share.gemini.google/1GwMNc0dIaSJ).
Đây là nguồn ý tưởng, không phải bằng chứng thực nghiệm hoặc tài liệu trích dẫn
thay paper. Các điểm cần sửa khi trình bày với team:

- LPIPS × residual trong repo là **anomaly score**, không mặc nhiên là một mạng
  residual một bước hay loss đã dùng train. Hội thoại ban đầu lẫn hai khái niệm.
- DDIM dùng được với objective DDPM; giảm số lần gọi model là hợp lý, nhưng tốc
  độ/chất lượng phải đo. Xem [paper DDIM](https://arxiv.org/abs/2010.02502).
- AutoDDPM mask/stitch/resample là ý tưởng phù hợp; không bảo đảm giữ đúng mọi
  mô lành hoặc xóa sạch tumor. Xem [paper gốc](https://arxiv.org/abs/2305.19643).
- Không bắt buộc thay LPIPS bằng encoder y tế. LPIPS có backbone và calibration
  học sẵn; khoảng cách feature của backbone thay thế không tự trở thành LPIPS.
  Xem [implementation tác giả](https://github.com/richzhang/PerceptualSimilarity).
- RAD-DINO được pretrain **chest X-ray**, không phải MRI não; nhãn “medical”
  không xóa domain gap. [Model card Microsoft](https://huggingface.co/microsoft/rad-dino).
- Simplex chưa chứng minh ưu thế trong đối chứng G20/S20 của nhóm; không đổi
  noise của checkpoint Gaussian tại inference rồi coi đó là train Simplex.
- Multi-contrast cần dataset mới có T1/T2/FLAIR tương ứng; export hiện tại chỉ
  FLAIR PNG. Symmetry cần xác minh orientation/alignment và bilateral lesions.
  Hai hướng này không nên mở ngay trong 4 tuần còn lại.
- Không tune trên test; không dùng nhãn tumor để fine-tune perceptual head rồi
  vẫn mô tả toàn pipeline là normal-only training.

## Chất lượng source và khả năng tái lập

Review sâu C `d282f02` phát hiện các việc phải xử lý khi tích hợp:

1. **CI:** năm test module mới import Torch trực tiếp trong khi CI chỉ cài
   `[data,dev]`; collection thực cho C có 8 lỗi thiếu Torch (gồm 3 module cũ).
   B đã sửa 3 module cũ; cần giữ và mở rộng quy tắc optional dependency.
2. **Test float32:** epsilon-loss test yêu cầu bitwise equality cho hai thứ tự
   reduction; đo lệch một ULP `1.1920928955078125e-07`, cùng objective. Sửa test
   bằng reduction tương ứng/tolerance chặt, không sửa objective để làm đẹp test.
3. **Cache prefix:** `patch_validation.py` kiểm images/masks/labels nhưng chưa
   ràng buộc đầy đủ checkpoint/timestep/condition/seed. Cần contract trước reuse.
4. **Hash rồi đọc lại:** `feature_condition.py` và `condition_ablation.py` có
   đường hash file rồi reopen sau; dùng cùng verified bytes như adapter B.
5. **Continuation:** `modal_train.py` lưu parent hashes nhưng chưa thực thi kiểm
   bytes trước copy/load best/last. Không có bằng chứng checkpoint thực bị sai.
6. **Adapter:** cache C có thêm `noisy_images`, adapter B nhận đúng 4 legacy keys;
   cần adapter có validation, không bỏ kiểm shape/identity để đọc cho được.

Đây là rủi ro kỹ thuật có vị trí code, không chứng minh metric C bị sai. Snapshot
`c02addb` bổ sung khoảng 6.915 dòng sau `d282f02`; audit mới tập trung DDIM gap,
call count, calibration và JSON, chưa là review toàn bộ mọi dòng/remote recovery.

QA B trong lượt này: Ruff check/format 107 files pass; full suite lần 1 có 165 pass
+ 1 lỗi Windows rename; lần 2 ngoài sandbox 165 pass + 1 setup error rename ở test
khác. Nhóm 35 data tests riêng pass. Vì lỗi vẫn xảy ra ngoài sandbox, chưa quy
nguyên nhân chỉ cho sandbox; cần điều tra handle/file lock/Windows filesystem.
Không báo full suite xanh từ lần chạy cũ. C `d282f02`: 145 pass + 1 float32 failure;
Ruff 4 lỗi legacy, 120 files format pass. C `c02addb`: **181 pass + 1 failure**,
vẫn là test float32 reduction nói trên; không phải một suite hoàn toàn xanh.

## Tiến độ và thứ tự tiếp theo

| Nhóm việc | Evidence hiện có | Việc còn lại / owner |
|---|---|---|
| Data DAC-7/9 | Revision ổn định; B Colab và C Modal có run thật | A gắn access receipt và đóng acceptance DAC-9 |
| DDPM DAC-11/14 | Full checkpoint, resume design, G20/G30 và fine-tune | C xử lý guards; quyết định terminal run batch1 cũ |
| Score/evaluator DAC-12/30 | B reproduction, bootstrap, C LPIPS/DDIM | B review adapter mới và tái tính từ cache; C co-review |
| Gate DAC-15 | Đủ nhiều evidence để review ngay | Ghi quyết định kỹ thuật; không dùng AP thấp làm lý do giữ mọi task code mở |
| DAC-16/17/18 | Nhiều timestep/sampler/score đã khảo sát | B/C chuẩn hóa bảng, benchmark và đăng ký shortlist cuối |
| DAC-19/21 | Chưa có independent multi-train-seed/final freeze | Lặp có giới hạn khi được duyệt budget; freeze đầy đủ |
| DAC-22–25 | Chưa có final test | Chuẩn bị test runner có freeze gate trước, chạy sau freeze |
| DAC-26–29 | Có nhiều tài liệu thí nghiệm | Viết report/demo ngay; đưa kết quả âm và limitations vào |

Khuyến nghị chọn **củng cố LPIPS/DDIM có sẵn** thay vì (a) train thêm model hoặc
(b) mở multi-contrast/symmetry. Giữ baseline, LPIPS-only, initial/naive-stitch/final
LPIPS và brightness làm các đối chứng; không chọn lại theo internal holdout.

- **06–09/11 — B/C:** tích hợp `c02addb` với B trên branch làm việc, sửa test/guards,
  CI trên push branch, tiếp nhận cache private và tái lập metric. Đóng DAC-15 bằng
  evidence/decision. Không cần PR mới chặn tiến độ, chưa merge main tự động.
- **09–13/11 — B:** paired patient bootstrap 1.000 draws/seed17 cho final-vs-initial,
  final-vs-naive-stitch và final-vs-brightness trên cùng 25 holdout groups; thresholds
  cố định. Đây là CI thăm dò, không sửa được adaptive overfitting. Chốt shortlist.
- **14–20/11 — B/C/A:** repeat sampling seeds có budget, ghi khác train seeds;
  nếu đủ nguồn lực mới lặp training. B làm frozen-test runner, A ghi freeze
  checkpoint/source/LPIPS weights/ROI/mask/sampler/threshold/seed/budget.
- **21–27/11:** chạy test sau freeze; không tune lại. Test 3.665 ảnh cần checkpoint
  và output theo shard/resume; không mặc định stack mọi map float64 vào RAM.
- **28/11–04/12:** bảng kết quả, citations, demo tái chạy và dry-run bảo vệ.

Chưa khởi chạy các việc GPU trên. Linear status giữ nguyên cho đến review
acceptance; đề xuất DAC-12/30 chuyển review sau reproduction, DAC-15 thực hiện
ngay; DAE là optional, không chặn gate. DAE cũ có all83 result nhưng khác protocol
và thiếu lifecycle metadata, không ghép số đó vào bảng 43 holdout.
