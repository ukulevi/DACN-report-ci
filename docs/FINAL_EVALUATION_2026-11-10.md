# Final evaluation — 10/11/2026

Người dùng yêu cầu thực hiện review/freeze, launcher bốn phương án, validation,
test và hình/báo cáo trong cuộc trao đổi10Nov. Đây là authorization mới, thay
quyết định giữ candidate08Nov; không thay đổi phương pháp theo test.

## Quyết định phương pháp trước test

Giữ nguyên Gaussian EMA epoch30, checkpoint SHA256
`d3e06d2082e18234899b201718abd59078c8a800d66da3cc5d8b9e0d9ca9b946`;
dataset revision `350c74f2313e4e73ca65e7ffc6e9dcf74040b3b7`.
DDIM20 từt300 + DDIM30 từt50, eta0, một lượt pha hai. Seed17 và UUID-derived
sampling seed; batch1. LPIPS0.1 SqueezeNet spatial, grayscale lặpRGB, crop240,
input[-1,1], normalizeFalse. Pseudo-mask LPIPS×residual chuẩn hóa, threshold
0.13344676792621613, strict>, dilation3. Không refit mask hoặc score thresholds.

| Arm | Pixel threshold | Image threshold |
|---|---:|---:|
| Final LPIPS | .26617634296417236 | .24719371760470998 |
| Initial LPIPS | .2501829266548157 | .2262084392261588 |
| Naive-stitch LPIPS | .21649277210235596 | .20856854640361336 |
| Brightness | .48235294222831726 | .4997753345200585 |

Metric sử dụng score>=threshold, pixel global micro trên240crop; Dice=F1;
thêm macro lesion-slice Dice và normal-empty-mask pixel FPR. Image score mean
top576pixels. Báo đủ4arms, image/pixel AUROC/AP/Precision/Recall/F1; không chọn
winner rồi bỏ các đối chứng. Normal-only training có labeled-validation
selection; validation đã khảo sát lặp, một training/sampling seed. Brightness
AP cao hơn diffusion trên validation; không khẳng định diffusion vượt baseline.

## Ngoại lệ CI cho đợt đánh giá này

API GitHub10Nov xác nhận HEAD1c13e5f:
[run42455732782](https://github.com/tanh1c/DACN/actions/runs/42455732782)
startup_failure, jobs=0. Không sửa tài khoản/billing hoặc tăng hạn mức.
Theo yêu cầu xử lý hoặc ghi quyết định của người dùng, agent chọn dùng bằng
chứng kiểm tra local cho đợt inference này:282passed/1skipped41.29s; LPIPS
chưa cài local nên test roundtrip LPIPS bị skip. Ruff check/format đạt sau
format code mới. LPIPS thực cần được xác nhận bằng smoke và validation remote.
Đây không là CI success; BLK-20261107-01 vẫn Waiting cho repo owner.
Không merge main hoặc bỏ yêu cầu teammate review cho PR.

## Thay môi trường phải có bằng chứng mới

Bundle Colab08Nov nằm trên máy khác; không giả định inventory đó là môi trường
Modal. Protocol mới `final-fourarm-20261110-a` giữ nguyên scientific artifacts,
xuất native LPIPS weights, actual dependency inventory và source snapshot đầy
đủ src/scripts. Kiểm native LPIPS state hash với historical receipt.
Khôi phục partition từ index/group seed17 và yêu cầu tái lập metrics bốn arm;
không tuyên bố bytes của partition JSON mới bằng JSON private cũ.

Phải qua smoke2, validation43 cùng input/source/runtime, final map maxabs<=1e-6
và image/pixel metrics bốn arm lệch<=1e-6 so với evidence08Nov. Chỉ khi đạt mới
seal manifest thành frozen, gắn checksum receipt validation và cho dataset
factory mở test. Nếu không đạt, dừng trước test, điều tra nguyên nhân mà không
retune ngưỡng. Hồ sơ candidate08Nov được giữ nguyên cho lịch sử.

## Launcher và artifact

`scripts/run_final_evaluation.py` chạy các stage prepare → smoke → validation →
test trên Modal Workspace phutungcogioihaitan; local chỉ nhận aggregate receipts.
`--detach` trả app/call IDs; `--call-id` thu kết quả của cùng call, không submit
lượt mới. Mỗi attempt dùng ID riêng. Resume cùng attempt chỉ khi source/config/
artifact/scope/thresholds/order/content khớp; không có writer đồng thời.

Ví dụ từ root checkout, với Python đã cài Modal và PYTHONPATH=src:

```bash
python scripts/run_final_evaluation.py --stage prepare --protocol-id final-fourarm-20261110-a --output runs/final-20261110/prepare --detach
python scripts/run_final_evaluation.py --stage smoke --protocol-id final-fourarm-20261110-a --attempt-id final4-smoke-20261110-a --output runs/final-20261110/smoke --detach
python scripts/run_final_evaluation.py --stage validation --protocol-id final-fourarm-20261110-a --attempt-id final4-val-20261110-a --output runs/final-20261110/validation --detach
python scripts/run_final_evaluation.py --stage test --protocol-id final-fourarm-20261110-a --attempt-id final4-test-20261110-a --output runs/final-20261110/test --detach
```

Không chạy stage sau trước khi thu và kiểm receipt stage trước. Manifest/start
status được commit trên Volume; mỗi shard chứa4maps/mask được commit atomically.
Per-sample receipt, maps/masks, memmaps nằm trên private Volume. Aggregate metrics
tính trên toàn bộ pixels từng arm, không trung bình AUROC/AP theo shard.
Curves được lấy thưa để vẽ; scalar metrics vẫn tính exact score ties trên mọi
pixels. Test đầy đủ3665ảnh; không thay bằng một subset rồi gọi là full test.

## Trạng thái thực hiện

Chuẩn bị CPU đã submit; chưa có kết quả inference/test tại thời điểm tạo hồ sơ.
Actual run/status và metrics chỉ cập nhật từ produced artifacts trong
RESULT_TRACKING.md. Báo cáo/figures dùng scripts/build_final_report.py từ receipts
thực. Mẫu trường chưa được cung cấp; không tự tuyên bố đáp ứng format của trường.

### CPU preparation retry

Prepare-a rejected config bytes before model inference: macOS LF checkout hash
f1eaf46f…; Windows CRLF historical hash1d731fe0…. Only newline conversion gives
exact old hash. Strict recovery rejects all other content changes. Protocol-b
is the fresh source attempt; a is preserved, no test/model inference from a.
Automatic source mounting disabled; explicit /opt/dacn-final/src import required.

### Validation acceptance and final freeze

Protocol-b smoke2 and fresh-model validation43 completed/uploaded. Validation
map maxabs2.980232238769531e-7 and all4arm image/pixel metric maxabs
2.2600697424657312e-8 reproduce historical evidence below1e-6. No selection or
threshold refit. Actual acceptance receipt seals the new runtime/source manifest
frozen, with its SHA linked in configs/final_evaluation_frozen_20261110.json;
controls in configs/final_controls_frozen_20261110.json. Evidence file:
docs/evidence/2026-11-10-fourarm-validation.json. These are the final inputs;
scientific parameters remain exactly those registered before test.
