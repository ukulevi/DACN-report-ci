# Colab GPU controls validation — 07/11/2026

Notebook riêng: https://colab.research.google.com/drive/1-EsZiZ1uHxWpPXVrzC8nVmLAxjQ3cbeR

User đã chọn Colab GPU cá nhân. Notebook mới giữ nguyên notebook Gaussian20 cũ.
Source snapshot bf82522 đã cài thành công, Colab cấp Tesla T4. Chưa chạy inference:
cell acquisition đang chờ HF read token qua getpass. Không ghi token vào code/log.

## Cách tạo lại notebook

Chạy git archive commit cần dùng vào ZIP riêng tư rồi:

```powershell
.venv/Scripts/python.exe scripts/make_controls_colab.py --source <source.zip> --output <private-notebook.ipynb>
```

Notebook có source embedded, tự kiểm SHA; không publish lên Git hoặc chia sẻ
public. Generator source có thể commit, notebook tạo ra giữ riêng tư.
Sourcebf82522 chứa adapter/runtime guard và control contracts đã nghiệm thu CPU.
Notebook generator đang cập nhật full43holdout reporting; cần đồng bộ cell cuối
trên notebook đã upload trước khi full83, không dùng all83 metrics làm holdout.

## Trình tự

1. Bootstrap cài source/package, xác minh CUDA.
2. Người dùng nhập HF read token. Chỉ tải checkpoint G30 và val tại immutable
   revisions, kiểm metadata/index/shard/checkpoint/mask hashes. Test không tải.
3. Export LPIPS weights và dependency freeze của Colab. Gate kiểm artifact;
   runtime binding kiểm source thực/import tree và packages trước dataset access.
4. Smoke2samples, mỗi mẫu50U-Net/3LPIPS; lưu start/terminal manifests, VRAM/timing.
5. Sau smoke đạt, full83 với cùng tham số, không refit threshold. Generator mới
   tách metrics43group-holdout khỏi engineering all83 và giữ partition40/43.
6. Lưu aggregate reports/provenance và private maps trước khi runtime reset.

Giới hạn45phút cooperative giữa calls, không hard GPU shutdown. Không tự mua
compute/đổi gói. Source và data riêng tư đã được người dùng cho phép dùng trên
Colab; token chỉ nhập ở runtime. Chưa có kết quả GPU mới để báo cáo.
