# Hồ sơ quyết định freeze — 08/11/2026

Trạng thái: **đề nghị review, chưa frozen**. Không cấp quyền mở test từ tài liệu
này. Review dựa vào artifact T4 đã kiểm và preflight, không chọn lại từ holdout.

**Quyết định08Nov của người dùng:** giữ candidate, chờ CI hoặc nhóm review.
Chưa chấp thuận ngoại lệ CI/local checks, chưa chấp thuận freeze hoặc mở test.

Update10Nov: người dùng đang nhờ Hải Tuấn review/freeze. HF có thêm11full
inference ablations09Nov; xem REPORT_READINESS_2026-11-10.md trước khi chốt,
không tự thay candidate bằng hàng có holdout cao nhất. Cấu hình trong tài liệu
này vẫn là đề xuất20+30/t300 đã nghiệm thu, chưa là quyết định freeze mới.

## Cấu hình đề nghị giữ nguyên

- Gaussian EMA epoch30, checkpoint SHA trong final_evaluation_candidate.json.
- DDIM20 từt300 + DDIM30 từt50, eta0, phase2repeats1;50U-Net calls/ảnh.
- LPIPS0.1 SqueezeNet spatial, grayscale lặpRGB, input[-1,1], crop240,
  normalizeFalse. Source archive d5889d5, T4 inventory đã nghiệm thu; source/
  checkpoint/weights/config/dataset/partition/mask đều có immutable pins.
- Pseudo-mask LPIPS×residual: normal-calibration95percentile,
  threshold.13344676792621613, strict>, dilation3, epsilon1e-8.
- Bốn phương án: finalLPIPS, initialLPIPS, naive-stitchLPIPS, brightness;
  dùng cùng input và trajectory đã đăng ký. Không thêm pretrained encoder,
  score fusion, postprocessing, checkpoint/timestep/threshold sweep.
- Sampling base17 + UUID-derived seed; một training/sampling seed đã thực hiện.
  Không công bố UUID hoặc derived seeds. Không giả định thêm seed đã chạy.

| Arm | Pixel threshold | Image threshold |
|---|---:|---:|
| Final LPIPS | .26617634296417236 | .24719371760470998 |
| Initial LPIPS | .2501829266548157 | .2262084392261588 |
| Naive-stitch LPIPS | .21649277210235596 | .20856854640361336 |
| Brightness | .48235294222831726 | .4997753345200585 |

Thresholds từ40calibration, không đổi theo43holdout. Score lớn hơn nghĩa là
bất thường hơn; metric sử dụng >=threshold. Pixel ROC-AUC/AP/P/R/F1 global
micro trên240×240, Dice=F1; thêm macro lesion Dice và normal-empty-mask FPR.
Image score mean top576pixels, image ROC-AUC/P/R/F1. Báo actual calls, thời gian
đúng scope và CUDA allocated memory; không gọi adapter time là end-to-end.

## Bằng chứng để review

1. [T4 smoke2/full83](COLAB_ACCEPTANCE_2026-11-08.md): private ZIP đã tải,
   10file hashes đạt, maps của4arms khớp C tại1e-6; CPU metric replay khớp1e-10.
2. [Static preflight](evidence/2026-11-08-final-candidate-preflight.json):
   9artifacts/95Python files, candidate-final rejection và CPU-runtime mismatch.
   Không thay kiểm live Colab cuối.
3. [Cache/resume validation](evidence/2026-11-08-controls-cache-resume.json):
   4arms×43slices, mỗi arm ngắt sau1durable shard rồi resume42;
   completed replay không gọi backend,43score shards khớp input và15numeric
   metric fields/arm khớp T4. Không có U-Net/LPIPS calls trong phép replay này.
4. Paired absolute/SSIM và DDPM validation có27focused tests08Nov; DAC-30 Done.
   Training resume CPU/smoke riêng không là bằng chứng đã kiểm remote Modal job
   interruption/end-to-end orchestration.
5. Full local suite ngoài sandbox274pass38.57s. Lượt trong sandbox trước đó
   273pass/1WinError5 ở rename staging BMAD CLI; chạy riêng test đó đạt. Chưa có
   root cause hoặc code fix, không bỏ lịch sử failure và không suy CI xanh.

## Quyết định còn cần ghi trước final

- Chấp nhận protocol/4controls và giới hạn một seed, validation đã khảo sát lặp.
  Brightness pixel AP vẫn cao hơn diffusion; không tuyên bố model vượt baseline.
- CI account/billing chưa giải quyết: chủ repo xử lý để có CI thực, hoặc nhóm
  ghi rõ quyết định dùng local-check evidence thay cho CI trong đợt nghiệm thu.
  Đây là ngoại lệ cần quyết định, không tự gán local checks thành CI success.
- Ghi người/ngày phê duyệt và commit hồ sơ frozen trước bất kỳ final-test access.
  Live runtime phải khớp source/import/dependency pins; khác thì dừng kiểm lại.

## Phạm vi đường chạy hiện có và còn thiếu

`run_frozen_ddim` đã nối input gate, runtime binding và adapter cho **model chính**;
caller còn phải cung cấp dataset factory đã kiểm revision/content. Chưa có CLI
turnkey đóng gói tất cả4arms cho final dataset. Replay cache/resume ở trên là
nghiệm thu orchestration từ validation caches, không thay bằng chứng fresh-model
four-arm final entry. Không công bố một lệnh final hoàn chỉnh khi wrapper và
dataset factory chưa được nghiệm thu. Có thể freeze phương pháp trước, sau đó
hoàn thiện launcher giữ nguyên protocol và kiểm bằng validation trước final.
