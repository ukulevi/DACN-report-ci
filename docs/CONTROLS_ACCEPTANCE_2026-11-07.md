# Controls và CPU benchmark — 07/11/2026

Đạt bounded acceptance2samples trên cùng trajectory DDIM20+30: final LPIPS,
initial LPIPS, naive-stitch LPIPS và brightness. Contract riêng tại
configs/final_control_candidates.json; thresholds lấy từ40calibration cũ,
không refit. Hai mẫu chỉ phục vụ kiểm kỹ thuật, không xếp hạng chất lượng model.

100 U-Net calls +6LPIPS calls; final score map bằng đúng lượt nghiệm thu trước,
maxabs0. Default adapter vẫn không tính thêm naive control khi không yêu cầu.
Naive-stitch lấy context từ input ngoài mask và initial reconstruction trong mask,
không dùng final reconstruction hoặc nhân mask vào score.

## Số đo CPU thực

- threads2, batch1; cold adapter sample gồm load model24.398705s.
- warm sample21.706827s; hai mẫu khác nhau, không suy startup overhead bằng hiệu số.
- Peak process working set908849152bytes (~867MiB), gồm startup/data/model/capture
  đến trước bước metrics; không là peak tensor memory hay GPU VRAM.
- Timing gồm adapter inference/scoring/transfers; không gồm QC/checksum/metrics.
- Receipt: runs/ddim-controls-acceptance-20261107-attempt2/aggregate_receipt.json.

Attempt1 dừng trước inference do CRLF/LF khi so dependency lock; đã sửa so theo
dòng package/version. Giữ cả hai attempts trong RESULT_TRACKING.

## Metrics diagnostic hai mẫu

Pixel global micro trên crop240, một model/sampling seed17. Thresholds riêng của
từng arm trong config; image score là mean top576. Không dùng các số này tune.

| Arm | Pixel AUROC | AP | Precision | Recall | F1/Dice |
|---|---:|---:|---:|---:|---:|
| Final LPIPS | .905501256 | .018280274 | .000465983 | .002293578 | .000774593 |
| Initial LPIPS | .914517225 | .021144831 | .006319115 | .009174312 | .007483630 |
| Naive stitch | .919759795 | .021590412 | .002144772 | .009174312 | .003476749 |
| Brightness | .990425695 | .723173895 | .093538605 | .919724771 | .169807326 |

Image AUROC/AP tất cả1; Final P/R/F1=1/1/1; ba controls P/R/F1=.5/1/.666667.
Image AUROC1 trên2samples không chứng minh hiệu quả thực tế. CPU RNG khác CUDA.
Không có test result từ lượt này.

## Weights và runtime

Native LPIPS state digest18fed4d1fc5c04f2918f64df3a2463344108576d1888f0d75538ca6d5c4dffe6
khớp artifact C lpips-normalcal-full83 tại model revisionfae9b50a4b78114fb51d87e803c77d019668f523;
results.json đã kiểm hash theo run manifest. Điều này xác nhận state weights,
không đồng nghĩa dependency/GPU/RNG runtime giống nhau.

Final API có runtime_binding kiểm source tree thực (src/scripts), import root và
dependency freeze trước dataset access.17tests liên quan đạt; Ruff đạt193files.
Candidate chưa frozen. Người dùng chọn Colab GPU cho validation/final tiếp theo;
cần export lại chính môi trường Colab, GPU smoke trước full83.
