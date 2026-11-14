# Advisor feedback tracker

Living record of advisor feedback and how each request is being addressed.
Update the tracker rows when a task changes status; add a new section for each
new email or meeting. Link evidence (RESULT_TRACKING entries, summary docs,
evidence JSON) instead of copying metrics here. Status values: `Done`,
`Partial`, `In progress`, `Not started`.

## 2026-11-13 — Email from the advisor (Trương Vĩnh Lân)

Received by email and shared with the agent on 2026-11-13 (screenshot). The
group's questions are marked **Nhóm**; the advisor's answers are quoted as
written (Vietnamese).

**Overall.** "Nhìn chung, nhóm đã triển khai được khá nhiều thí nghiệm và kết
quả ban đầu tương đối tốt. Đặc biệt, việc kết hợp LPIPS với residual mask để
hướng dẫn quá trình reconstruction lần hai đã giúp cải thiện Pixel AP từ 0.2246
lên 0.3182. Đây là một hướng nhóm có thể tiếp tục nghiên cứu sâu hơn."

**Scoring focus.**

- "Trước mắt, nhóm nên tập trung nghiên cứu và cải thiện phương pháp LPIPS thay
  vì mở rộng quá nhiều sang các phương pháp anomaly scoring khác nhau."
- "Cụ thể, nhóm cần phân tích rõ vai trò của từng thành phần trong phương pháp
  hiện tại, đặc biệt là LPIPS và residual mask. Cần làm thêm các thí nghiệm
  ablation để xác định thành phần nào thực sự đóng góp vào sự cải thiện kết quả."
- "Bên cạnh đó, nhóm có thể thử nghiệm một số cách kết hợp anomaly score khác
  nhau, chẳng hạn weighted combination giữa LPIPS và reconstruction residual,
  nhưng nên có giả thuyết rõ ràng cho mỗi thí nghiệm."

**Nhóm — Hướng cải tiến:** nên thay đổi kiến trúc/cách huấn luyện DDPM, hay ưu
tiên cải thiện reconstruction và anomaly scoring trên mô hình hiện tại?

- "Thầy đề xuất trước mắt giữ nguyên kiến trúc và cách huấn luyện DDPM hiện tại.
  Nhóm nên ưu tiên cải thiện quá trình reconstruction và anomaly scoring."
- "Việc thay đổi kiến trúc hoặc huấn luyện lại DDPM sẽ làm tăng đáng kể phạm vi
  thực nghiệm. Chúng ta chỉ nên xem xét hướng này sau khi đã hiểu rõ những hạn
  chế của phương pháp hiện tại và xác định được liệu chúng có thực sự xuất phát
  từ mô hình DDPM hay không."

**Nhóm — Mức nhiễu và quá trình tái tạo:** kết quả có thể phụ thuộc vào mức
nhiễu, số bước DDIM và cách chọn vùng tái tạo; chạy bao nhiêu mức nhiễu là đủ?

- "Đây là một hướng thực nghiệm quan trọng. Nhóm nên khảo sát ảnh hưởng của mức
  nhiễu được thêm vào ảnh đến khả năng phát hiện bất thường."
- "Trước mắt, có thể chọn một số mức nhiễu đại diện, chẳng hạn 50, 100, 200, 300
  và 400 (nếu phù hợp với noise schedule của mô hình). Không nhất thiết phải thử
  quá nhiều mức ngay từ đầu."
- "Nhóm nên cố định các thành phần khác và chỉ thay đổi noise level để đánh giá
  ảnh hưởng của tham số này. Sau đó, có thể khảo sát thêm số bước DDIM và cách
  lựa chọn vùng cần reconstruction."
- "Mục tiêu là tìm hiểu mối quan hệ giữa noise level, reconstruction quality và
  anomaly detection performance, chứ không chỉ đơn thuần tìm một bộ tham số cho
  kết quả tốt nhất."

**Nhóm — Ưu tiên nghiên cứu:** xin góp ý hướng ưu tiên để thực hiện có hệ thống.

- "Thầy đề nghị nhóm chú ý hơn đến tính hệ thống và độ tin cậy của việc đánh giá."
- Cụ thể: báo cáo đầy đủ Pixel AP, Pixel AUROC và Dice; thực hiện ablation study
  để đánh giá đóng góp của từng thành phần; lựa chọn tham số trên validation set
  và chỉ sử dụng test set để đánh giá cuối cùng; so sánh với các baseline hiện có
  để xác định mức độ cải thiện thực sự của phương pháp.
- Ba công việc ưu tiên: (1) hoàn thiện và phân tích phương pháp Final LPIPS, đặc
  biệt là vai trò của residual mask; (2) thực hiện các thí nghiệm khảo sát noise
  level và số bước DDIM; (3) thực hiện ablation study và so sánh với các baseline
  để đánh giá tính hiệu quả và ổn định của phương pháp.
- "Sau khi có kết quả của các thí nghiệm này, chúng ta sẽ cùng xem xét hướng phát
  triển tiếp theo."

**Nhóm — Kênh liên lạc:** "Nhóm có thể trao đổi với thầy qua email. Thầy sẽ đọc
và trả lời dần nhé, vì thầy cũng rất bận."

### Tracker

| ID | Request | Status | Evidence / where | Next action |
|---|---|---|---|---|
| AF-01 | Focus on LPIPS; stop expanding to other scoring families | Done | SNC, Sobel, PraNet-style and CGFF recorded as negative/neutral in [ABLATION_SUMMARY §3](ABLATION_SUMMARY_2026-11-13.md) | Do not start new scoring families without a written hypothesis. To explain to the advisor (14 Nov): the best configuration, Final LPIPS × brightness, keeps LPIPS as the score and uses input brightness only as a gate, not as a new scoring family. The hypothesis and rule were written before running (`postprocess_hybrid` docstring), and the gate ablation shows LPIPS is required (L1 × brightness ≈ 0.18) |
| AF-02 | Ablation of each component, especially LPIPS and residual mask | Done | [ABLATION_SUMMARY §1–§2](ABLATION_SUMMARY_2026-11-13.md): initial → naive-stitch → LPIPS-only mask → Final, residual-in-score, mask rules; LPIPS-only mask lowers validation Dice beyond uncertainty (`variants-full-20261114T041258Z`) and test AP (12 Nov) | Test CIs done for the four frozen arms; LPIPS-only mask test row has no CI |
| AF-03 | Weighted LPIPS + residual combination with a clear hypothesis | Done | Fixed-weight rank fusion on the Final reconstruction, hypothesis written before running ([ABLATION_SUMMARY §3](ABLATION_SUMMARY_2026-11-13.md), `final-fusion-full-20261114T034226Z`): no pixel-AP gain, w ≥ 0.5 worse; LPIPS alone kept | Report as a negative result. Contrast: gating LPIPS by input brightness, rather than by the residual, gives a large validation gain (ABLATION_SUMMARY §3, hybrid subsection) |
| AF-04 | Keep DDPM architecture/training unchanged for now | Done | No retraining since EMA30 checkpoint `d3e06d20…b946` | Revisit only if analysis shows limits come from the DDPM |
| AF-05 | Noise-level sweep with other components fixed (suggested 50/100/200/300/400) | Done | t50/100/200/300/400/500/800 with patient-group bootstrap CIs ([ABLATION_SUMMARY §2](ABLATION_SUMMARY_2026-11-13.md), `noise-ext-full-20261114T034843Z`): inverted U, t50/t100/t800 worse, t200–t500 plateau not separable; t300 kept | Optional: extra sampling seeds for t300/t400 |
| AF-06 | Relationship noise level → reconstruction quality → detection | Done | Descriptive, no intervals: [ABLATION_SUMMARY §2, quality table](ABLATION_SUMMARY_2026-11-13.md); run `sweep-quality-full-20261113T162803Z`: under-healing at t100, plateau t200–500, normal-tissue breakdown at t800 | t50/t400 added 14 Nov in the combined curve |
| AF-07 | Then DDIM steps and reconstruction-region choice | Done | Mask rules p90–99 × d1/3, DDIM budget 10+15/20+30/40+50, splits at 50 calls, phase-2 start t30/t50/t100 ([ABLATION_SUMMARY §1–§2](ABLATION_SUMMARY_2026-11-13.md)); none changes AP/Dice beyond uncertainty; control kept | – |
| AF-08 | Report Pixel AP, Pixel AUROC and Dice | Done | FINAL_TEST_RESULTS, LPIPS_COMPARISON, ABLATION_SUMMARY tables | Keep all three in every new table |
| AF-09 | Select parameters on validation; test only for final evaluation | Done | All selection on validation. Test used after freeze: four-arm, DDIM50 and LPIPS-only mask (exploratory), plus the 14 Nov re-run of the frozen four-arm protocol for CIs only (unchanged method and thresholds, published metrics reproduced within 1e-6, no selection); and a fifth access, 14 Nov, for one evaluation of Final LPIPS × brightness, frozen on validation before the run | Keep test closed; any new configuration must be frozen on validation first |
| AF-10 | Compare with existing baselines | Done | [ABLATION_SUMMARY §1 test CIs](ABLATION_SUMMARY_2026-11-13.md): Final beats Initial and Naive-stitch on all metrics beyond uncertainty. Brightness has higher pixel AP (Δ [0.161, 0.228]) and Dice. Final has higher image AUROC (Δ vs brightness [−0.213, −0.107]). ConvAE/ConvVAE validation only | Report brightness honestly: wins pixel-level, loses image-level. Follow-up 14 Nov (validation, `hybrid-full-20261114T103245Z`): LPIPS × brightness beats both Final and brightness on holdout pixel AP beyond uncertainty (0.656 vs 0.312/0.390) without lowering image AUROC. Gate ablation (`gate-full-20261114T105141Z`): Initial LPIPS × brightness 0.530, L1 gates ≈0.18, so the re-sampling and LPIPS both contribute. Frozen test run (`hybrid-test-full-20261114T105717Z`, fifth test access): pixel AP 0.651 [0.626, 0.673] vs brightness 0.511 and Final 0.318; image AUROC 0.840; all beyond uncertainty |
| AF-11 | Stability / reliability of the evaluation | Partial | Patient-group bootstrap on validation for every ablation; sampling seeds 17/18/19 (seed SD ≈0.013 holdout AP, partition-dependent t400 ranking); test CIs for the four frozen arms (767 groups, 1000 replicates, `test-ci-full-20261114T042202Z`) | Training-seed variability not measured |
