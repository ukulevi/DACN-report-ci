# Colab GPU controls validation — 07/11/2026

Notebook riêng: https://colab.research.google.com/drive/1-EsZiZ1uHxWpPXVrzC8nVmLAxjQ3cbeR

User đã chọn Colab GPU cá nhân. Notebook mới giữ nguyên notebook Gaussian20 cũ.
Source snapshot bf82522 đã cài thành công, Colab cấp Tesla T4. HF auth/checksums
đã qua. Đã sửa dependency snapshot cho direct-source Colab packages và import
path của kernel; source thực sau patch được chụp hash76ea57a97e00a7a6d23687a2a470d801f043c5d108cbae0cdff0acaf52e5ccf6.
Smoke2 đã hoàn tất; full83 đang chạy với báo cáo43group-holdout riêng.

## Cách tạo lại notebook

Chạy git archive commit cần dùng vào ZIP riêng tư rồi:

```powershell
.venv/Scripts/python.exe scripts/make_controls_colab.py --source <source.zip> --partition <private-registered-partition.json> --output <private-notebook.ipynb>
```

Notebook có source embedded, tự kiểm SHA; không publish lên Git hoặc chia sẻ
public. Generator source có thể commit, notebook tạo ra giữ riêng tư.
Sourcebf82522 chứa adapter/runtime guard và control contracts đã nghiệm thu CPU.
Cell cuối notebook đã đồng bộ full43holdout reporting trước khi full83.
Không dùng all83 metrics làm holdout.

## Trình tự

1. Bootstrap cài source/package, xác minh CUDA.
2. Người dùng nhập HF read token. Chỉ tải checkpoint G30 và val tại immutable
   revisions, kiểm metadata/index/shard/checkpoint/mask hashes. Test không tải.
3. Export LPIPS weights và dependency freeze của Colab. Gate kiểm artifact;
   runtime binding kiểm source thực/import tree và packages trước dataset access.
4. Smoke2samples, mỗi mẫu50U-Net/3LPIPS; lưu start/terminal manifests, VRAM/timing.
5. Sau smoke đạt, full83 với cùng tham số, không refit threshold. Dùng membership
   đã pin hash/index, không re-derive bằng sklearn khác phiên bản. Tách metrics43
   group-holdout khỏi engineering all83, giữ đúng40/43 cũ. Gate hiện yêu cầu
   artifact thứ9 validation_partition; scripts acceptance/benchmark cần --partition.
6. Lưu aggregate reports/provenance và private maps trước khi runtime reset.

Giới hạn45phút cooperative giữa calls, không hard GPU shutdown. Không tự mua
compute/đổi gói. Source và data riêng tư đã được người dùng cho phép dùng trên
Colab; token chỉ nhập ở runtime. Kết quả GPU actual đọc từ receipt trong notebook,
không suy từ kế hoạch hoặc kết quả CPU. Xem RESULT_TRACKING.
