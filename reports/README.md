# Build báo cáo PDF tự động

Nguồn nội dung là [`docs/REPORT_DRAFT.md`](../docs/REPORT_DRAFT.md). Nhóm sửa
Markdown như bình thường; Pandoc tạo LaTeX và Tectonic biên dịch PDF tiếng Việt.
Không cần sửa hai bản nội dung Markdown/LaTeX song song.

Đây là **bản nháp validation**, chưa là báo cáo nộp hoàn chỉnh hoặc kết quả test.
Build báo cáo không freeze phương pháp, không chọn model/threshold, không truy
cập HF/Modal/Colab và không dùng dữ liệu riêng tư trong runs/checkpoints.

## Nơi chỉnh sửa

- Nội dung: `docs/REPORT_DRAFT.md`; các heading `##` thành mục cấp1 trong PDF.
- Tên đề tài/nhóm/ngày/margins/font: `reports/metadata.json`. Bổ sung thông tin
  trường, giảng viên, MSSV theo mẫu thật trước bản nộp; chưa tự điền thông tin đó.
- Header/footers: `reports/header.tex`.
- Phụ lục sinh tự động: accepted aggregate evidence08Nov và ablation CSV10Nov,
  đường dẫn trong INPUTS của `scripts/build_report.py`. Không sửa tay số trong
  PDF. Khi có evidence cuối đã review, cập nhật nguồn theo commit rõ ràng.

PDF có bảng chính và phụ lục từ evidence, phân biệt validation đã nghiệm thu
với producer-audited ablations09Nov. Các nguồn hiện hành vẫn giữ candidate.

## Chạy local — Windows/Linux x86_64

Chỉ cần Python3.11+ và Git; không cài thêm package vào môi trường training.
Hai compiler portable từ release chính thức được pin checksum:
Pandoc3.13 và Tectonic0.17.0. Chúng được lưu trong ignored `runs/report-tools`.

```powershell
python scripts/report_tools.py --output runs/report-tools
python scripts/build_report.py
```

Hoặc dùng `.venv/Scripts/python.exe` thay `python` trên Windows. Lần đầu cần
Internet để tải compiler và TeX bundle; các lần sau dùng cache tài nguyên TeX.
Tectonic vẫn có thể cần kiểm bundle online. Không cần quyền admin/cài TeX toàn máy.

Nếu chỉ kiểm dữ liệu/nội dung, không cần compiler/network:

```powershell
python scripts/build_report.py --prepare-only
```

Kết quả trong ignored `reports/build/`:

- `report.pdf`: PDF sau build thành công.
- `report.md`, `report.tex`: nội dung đã ghép và source LaTeX sinh tự động.
- `build-manifest.json`: status, research status, source Git/dirty flag, hash đầu
  vào/layout/builder, tool versions và PDF SHA khi completed.
- `build.log`, `report.log`: chẩn đoán compiler. Khi lỗi, kiểm manifest/status;
  một PDF còn từ build khác không là bằng chứng thành công của build mới.

Script dọn các file report.aux/toc/out/lof/lot/xdv/pdf cũ trước compile để không
reuse TOC hoặc PDF stale khi thay document class. Không xóa source Markdown.
Compiler chạy untrusted mode; không bật shell escape.

Phiên bản compiler được pin, TeX bundle và metadata PDF vẫn có thể tạo khác biệt
byte giữa môi trường/lần build; không hứa byte-identical PDFs. Manifest giúp
truy vết source/input/tool của từng build.

## GitHub Actions

Workflow: [`.github/workflows/report.yml`](../.github/workflows/report.yml).

- Tự chạy khi push thay đổi report/evidence lên `main` hoặc
  `codex/bc-evaluation-colab`, và PR có thay đổi liên quan.
- Có `workflow_dispatch`. Nút **Run workflow** có sau khi workflow đã nằm trên
  default branch; khi chỉ ở nhánh hiện tại, push có thay đổi paths đã khai báo
  để kích hoạt build. Không cần tạo PR chỉ để build trên nhánh này.
- Khi thành công, mở Actions → **Build draft report PDF** → run → Artifacts →
  tải `dacn-report-<commit>`. Artifact có PDF và build-manifest, giữ14ngày.
- Diagnostic artifact `dacn-report-logs-<commit>` giữ7ngày, kể cả compile lỗi.
- `contents: read`, checkout không lưu credentials; không tạo Release/Pages.
  Không cung cấp HF/Modal token hay quyền write repo cho bước biên dịch.

**Blocker hiện có:** GitHub Actions tài khoản DACN đang startup_failure trước
khi tạo job. Workflow mới không giải quyết billing/account. Chủ repo cần xử lý
BLK-20261107-01 rồi mới xác minh run remote thực; local PDF build không là CI xanh.
Không thay payment settings, tăng budget hoặc retry hàng loạt từ công cụ này.

## Nguồn tham khảo thiết kế

- [EE5217 build workflow](https://github.com/HThanh-how/EE5217/blob/d7041f5da9a03db13c954463a836e757848b7b8e/.github/workflows/build.yml):
  tham khảo compile PDF/upload artifacts/logs; không sao chép nội dung môn học,
  mẫu có thông tin sinh viên, release hay Pages deployment.
- [hw2_introAI](https://github.com/HatakekkSheeshh/hw2_introAI): kết nối hiện trả404,
  chưa đọc/áp dụng workflow từ repo này; có thể đối chiếu thêm khi có quyền.
- [Pandoc manual](https://pandoc.org/MANUAL.html),
  [Tectonic](https://github.com/tectonic-typesetting/tectonic),
  [upload-artifact v4](https://github.com/actions/upload-artifact/tree/v4).
