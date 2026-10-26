# Protocol data — BMAD Brain, bản patient-disjoint-v1

## Nguồn và giới hạn

Nguồn chính: [BMAD repository](https://github.com/DorisBao/BMAD),
[paper BMAD](https://arxiv.org/html/2306.11876v3),
[Drive của tác giả](https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw).
File `Brain_AD.zip`, ID `1xuapqiL1s18eMeKMd-Vi1Cqp2wp7Pghp`, 401.031.573 byte.
SHA256 bản đã tải ngày 24/10/2026:

```text
eb7a5aae9ea1752b16d3a76b47595ffd956c0b210444da76853455d4d7be61c8
```

BMAD công bố điều kiện CC BY-NC-SA; giữ trích dẫn và kiểm tra thêm điều kiện nguồn
BraTS trước khi chia sẻ lại dữ liệu. Đồ án dùng dữ liệu cho nghiên cứu; không đưa
ảnh, patient manifest hay checkpoint lên Git. Xem điều kiện đầy đủ trong nguồn,
không suy ra quyền tái phân phối từ việc link tải công khai.

Ảnh là FLAIR 2D từ BraTS2021, lưu PNG đã render, **không phải intensity NIfTI gốc**.
“Normal” là lát cắt không có tumor annotation, không có nghĩa bệnh nhân khỏe mạnh.
Không có metadata scanner/site/demographics đủ để kết luận tính tổng quát lâm sàng.

## Cấu trúc bản tải thực tế

```text
BraTS2021_slice/
  train/good/<patient>_<slice>.png
  valid/{good,Ungood}/img/<patient>_<slice>.png
  valid/{good,Ungood}/label/<patient>_<slice>.png
  test/{good,Ungood}/img/<patient>_<slice>.png
  test/{good,Ungood}/label/<patient>_<slice>.png
```

ID giữ dạng chuỗi để không mất số 0 đầu. Mã preprocessing được xem để hiểu cách đặt ID và lưu ảnh: [`bras.py` ở commit `5b6d583898b0a53c72aaebfa9a3f075c126e1bd5`](https://github.com/DorisBao/BMAD/blob/5b6d583898b0a53c72aaebfa9a3f075c126e1bd5/data_processing/brain/bras.py). Đây là snapshot công khai dùng để đối chiếu, **không xác nhận** archive Drive được tạo từ revision này.

Script lấy cohort abnormal validation từ `list(set(...))` rồi tạo test bằng phần bù. Hai cohort rời nhau trong một lần chạy sạch, nhưng ID validation có thể đổi giữa lần chạy. Script ghi vào thư mục output hiện có mà không dọn file cũ; file tồn dư có thể tạo overlap, nhưng chưa xác minh đây là nguyên nhân của archive đang dùng.

Phiên bản script, lệnh chạy và lịch sử output tạo ra `Brain_AD.zip` chưa xác minh được. Do đó, phát hiện overlap chỉ áp dụng cho archive có ID và SHA256 ở trên; không kết luận split chính thức của BMAD bị leakage. Archive cùng SHA256 là đầu vào nguồn duy nhất tái lập được; mọi số liệu từ `patient-disjoint-v1` đều thuộc split dẫn xuất này. Xem [DATA_AUDIT](DATA_AUDIT.md) để biết giới hạn provenance.

## Quy tắc split đã sửa

QC bản gốc tìm thấy 10 patient ID giao validation/test, 10 ảnh trùng pixel ở hai
split. Mặc định `--split-policy original` **báo lỗi**, không tự che leakage.

Chính sách `patient-disjoint-v1` được quyết định trước huấn luyện:

1. Giữ toàn bộ train và validation gốc.
2. Lấy tập patient ID validation; loại **mọi** slice test có ID thuộc tập này.
3. Không chuyển sample qua split, không chọn theo tổn thương hoặc model score.
4. Kiểm tra lại patient/sample/pixel overlap, đủ lớp và mọi ảnh/mask.
5. Ghi từng sample bị loại, đường dẫn, hash và lý do trong `qc_report.json/excluded`.

| Split | Normal | Abnormal | Tổng sau sửa | Patient IDs |
|---|---:|---:|---:|---:|
| Train | 7.500 | 0 | 7.500 | 424 |
| Validation | 39 | 44 | 83 | 50 |
| Test | 640 | 3.025 | 3.665 | 767 |

50 slice của 10 patient bị loại khỏi test; tổng sử dụng 11.248 ảnh. Đây là
**BMAD-derived patient-disjoint split**, không phải split BMAD chính thức. Không
so trực tiếp số cuối với paper BMAD. So sánh DAE/ANDi và các ablation trên cùng bản này.

## Decode và preprocessing v1

- PNG 240×240 thực tế là RGBA. Bật `--encoding bmad-bone` một cách tường minh.
  Nhánh này xử lý RGB/RGBA theo protocol nguồn; không tự nhận diện colormap bất kỳ.
- Alpha phải bằng 255. Ảnh: Pillow `convert('L')` tạo độ sáng hiển thị
  (xấp xỉ 0,299R + 0,587G + 0,114B). Không tuyên bố khôi phục intensity MRI.
- Chỉ hỗ trợ PNG mode L/RGB/RGBA theo từng encoding; từ chối palette (P), mode khác
  và transparency metadata tRNS để không nhầm chỉ số bảng màu với giá trị pixel.
- Mask màu: foreground nếu **bất kỳ kênh RGB > 0**, bỏ alpha. Gộp các vùng tumor
  thành whole-lesion binary mask. Không threshold cả 4 kênh vì alpha255 sẽ biến
  toàn ảnh thành lesion. Grayscale mask chỉ nhận 0/1 hoặc 0/255.
- Kiểm tra nhãn nhóm khớp mask; mask dương không được rỗng; normal mask phải rỗng.
  Mask validation/test bắt buộc tồn tại; train không có mask nguồn nên tạo zero mask.
- Pad đối xứng 240→256 (8 pixel mỗi phía), không resize/interpolate. Chuẩn hóa
  cố định `uint8 / 127.5 - 1`, vì vậy padding đen thành -1.
- NPZ có `image`: float32 `(1,256,256)` và `mask`: uint8 `(256,256)`, giá trị 0/1.
  Khi tính metric crop bỏ padding; loader train chỉ trả image, không đưa mask vào model.
- Không học normalization trên test. Không histogram matching/equalization trong v1.

QC kiểm tra tên/path, đủ nhóm, ảnh hỏng, mask thiếu/sai shape, nhãn không khớp,
orphan mask, PNG ngoài layout, patient/sample/pixel trùng giữa split. Hash pixel
tính trên ảnh sau decode về một kênh, bổ sung hash file nguồn. Đây không phải kiểm tra
near-duplicate hay xác nhận giải phẫu bởi bác sĩ.

## Chạy lại trên Windows PowerShell

Từ thư mục `C:\Users\Admin\Downloads\DACN-source`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[data,dev]"

# Bản gốc phải exit 1 do overlap; report phải dùng tên chưa tồn tại.
.\.venv\Scripts\python.exe -m dacn.cli data inspect `
  --root data/raw/bmad/BraTS2021_slice --encoding bmad-bone `
  --output runs/data-qc/original-check.json

# Bản được chọn: chỉ tạo ở một thư mục output MỚI.
.\.venv\Scripts\python.exe -m dacn.cli data prepare `
  --root data/raw/bmad/BraTS2021_slice --encoding bmad-bone `
  --split-policy patient-disjoint-v1 `
  --output data/processed/bmad-flair-patient-disjoint-v1
```

Nếu output đã tồn tại, lệnh từ chối ghi đè; dùng tên mới cho lần kiểm chứng. Không
xóa raw để chạy lại. Việc xuất qua thư mục tạm rồi đổi tên giúp lỗi giữa chừng không
để lại một dataset trông như đã hoàn tất.

Trên máy thành viên khác, tạo venv trước. Tải file từ Drive của tác giả hoặc:

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
.\.venv\Scripts\python.exe -m gdown 1xuapqiL1s18eMeKMd-Vi1Cqp2wp7Pghp `
  --no-cookies -O data/raw/Brain_AD.zip
Get-FileHash data/raw/Brain_AD.zip -Algorithm SHA256
# Chỉ giải nén khi hash khớp dòng SHA256 ở trên; dùng thư mục đích mới.
Expand-Archive -LiteralPath data/raw/Brain_AD.zip -DestinationPath data/raw/bmad
```

## Hợp đồng bàn giao data

`manifest.csv`: source, modality, sample_id, patient_id, slice_index, split, label,
image_path, mask_path, width, height, file_sha256, pixel_sha256, mask_sha256,
lesion_pixels, prepared_path. Đường dẫn nguồn tương đối với raw root; prepared_path
tương đối với processed root. Đọc patient/sample ID bằng dtype string.

`qc_report.json`: policy, encoding, số lượng, cảnh báo, exclusion ledger và
preprocessing. Giữ cùng source receipt, dataset version và hash của manifest trong
metadata từng run. Mọi thay đổi decode/split/normalization phải tăng version.

Bạn chịu trách nhiệm nguồn/QC; B/C không tự chia lại từng ảnh. Review ảnh bằng train
và val; dùng test duy nhất cho QC dữ liệu và lần đánh giá mô hình đã khóa.

## Export lên Hugging Face

`dacn data publish-hf` tạo NPZ shard từ prepared samples trước khi upload. Export
không chứa `manifest.csv`, `qc_report.json`, patient ID nguồn, tên file nguồn
hoặc exclusion ledger. `index.csv` dùng UUID ngẫu nhiên cho sample và patient group;
group UUID phục vụ patient-bootstrap, giữ ổn định trong revision Hub đã pin, còn
bản đồ về ID nguồn không được upload. Exporter chỉ chấp nhận thư mục có đúng các
shard/index sinh ra. Repo Hub mặc định private. Dùng revision SHA do lệnh trả về
để pin dataset cho từng run.

Dataset là ảnh y tế đã render; private storage không thay thế điều kiện sử dụng.
BMAD nêu CC BY-NC-SA nhưng BraTS có thể áp thêm điều kiện. Không bật public trước
khi nhóm xác nhận có quyền chia sẻ bản dẫn xuất. Chi tiết Modal, secret và lưu
checkpoint/kết quả nằm ở [HUGGINGFACE_MODAL](HUGGINGFACE_MODAL.md).
