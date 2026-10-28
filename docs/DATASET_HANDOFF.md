# Dataset handoff — verified 26/10/2026

Dataset private của nhóm đã được xác minh đầy đủ:
[dacn-bmad-team/bmad-flair-patient-disjoint-v1](https://huggingface.co/datasets/dacn-bmad-team/bmad-flair-patient-disjoint-v1).

**Revision cố định:** `fbb884dfe144be5c71e205a73701507163557ea4`.
Luôn pin revision này trong run manifest và lệnh Modal; không dùng moving `main`
cho số liệu nghiên cứu đã chốt. Task: [DAC-9](https://linear.app/dacn-khmt/issue/DAC-9/d02-ban-giao-data-contract-va-kiem-chung-tren-may-thanh-vien).

Repo này đã có nội dung khi bắt đầu lần bàn giao. Đã kiểm tra lại thay vì ghi đè:
snapshot tải về đầy đủ, metadata nguồn khớp, 95 file remote đạt kiểm tra checksum
(bao gồm `.gitattributes`), và toàn bộ 11.248 bộ image/mask/label khớp bitwise với
prepared dataset local. Không có upload mới hay đổi revision trong lần xác minh này.

| Split | Mẫu | Normal | Anomaly | Nhóm bệnh nhân ẩn danh |
|---|---:|---:|---:|---:|
| train | 7.500 | 7.500 | 0 | 424 |
| val | 83 | 39 | 44 | 50 |
| test | 3.665 | 640 | 3.025 | 767 |

## Thành viên tải và kiểm tra

Từ repo có pipeline HF/Modal trên main (tối thiểu commit `12c3088`):

```powershell
python -m pip install -e ".[data,hub]"
hf auth login
```

Đăng nhập bằng trình duyệt; account phải có quyền đọc repo private trong tổ chức.
Không đưa token vào chat, source code hoặc run manifest. Chạy đoạn Python sau:

```python
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from dacn.data.hub import validate_hf_export

revision = "fbb884dfe144be5c71e205a73701507163557ea4"
folder = Path(
    snapshot_download(
        repo_id="dacn-bmad-team/bmad-flair-patient-disjoint-v1",
        repo_type="dataset",
        revision=revision,
        token=True,
        local_dir="data/shared/bmad-flair-patient-disjoint-v1",
    )
)
metadata = validate_hf_export(folder, allow_hf_cache=True)
print({split: values["samples"] for split, values in metadata["splits"].items()})

with np.load(folder / "data/train/shard-00000.npz", allow_pickle=False) as batch:
    images = batch["images"]  # float32 (N,1,256,256), [-1,1]
    masks = batch["masks"]  # uint8 (N,256,256), 0/1
    labels = batch["labels"]  # uint8 (N,), 0=normal,1=anomaly
```

`local_dir` tránh dùng trực tiếp các symlink của cache snapshot trên Linux; validator
cố ý từ chối symlink trong payload. `allow_hf_cache=True` chỉ cho phép metadata cache
của Hub và `.gitattributes`, không bỏ kiểm tra shard/index/checksum.

## Hợp đồng dữ liệu

- Có 89 NPZ shard, 3 index CSV, dataset card và metadata; khoảng 260 MB.
- Đọc trực tiếp array đã chuẩn hóa, không chuẩn hóa lại từ 0–255.
- Mỗi index ghi UUID sample, UUID patient group, split, label, shard và offset.
  Giữ UUID group để bootstrap theo bệnh nhân. ID/path nguồn và bản đồ UUID không upload.
- Không dùng `load_dataset()` kiểu Parquet cho gói này. NPZ là schema đã thống nhất
  với `dacn.training.dae` và Modal runner.
- Mask/label là dữ liệu đánh giá; model/scorer chỉ nhận ảnh trong normal-only training.
- Khi tính metric crop padding về vùng240×240; chỉ dùng validation để chọn model/ngưỡng.
- Test giữ kín đến freeze protocol. Handoff data không phải một run huấn luyện.

## Provenance đã xác minh

| Trường | Giá trị |
|---|---|
| Source archive SHA256 | `eb7a5aae9ea1752b16d3a76b47595ffd956c0b210444da76853455d4d7be61c8` |
| Canonical source manifest SHA256 (LF) | `36ceb0571637a634fad94f36b997e142302e4fac0f928cb99012e63c76aa978c` |
| Local manifest file SHA256 (CRLF) | `6266a6200475643da9f5c4130e04079ce070994e3b2fcdbd8178a0bc84d5a90b` |
| Remote dataset_metadata.json SHA256 | `9c60fac0ff2a2d6274e82db466e5e338303d0dc24986b3bab4b32b6e710af218` |

Hai hash manifest khác nhau do quy tắc newline: exporter tính hash trên nội dung
đã chuẩn hóa LF; file local dùng CRLF. Đây không phải thay đổi hàng dữ liệu/split.

Nguồn và điều kiện: [BMAD paper](https://arxiv.org/html/2306.11876v3),
[BMAD repository](https://github.com/DorisBao/BMAD), và [BraTS2021 — Data Usage
Agreement / Citations](https://www.med.upenn.edu/cbica/brats2021/). BMAD tuyên bố
CC BY-NC-SA nhưng không chỉ rõ phiên bản. Trang BraTS2021 cho phép dùng dữ liệu
cho nghiên cứu nếu trích dẫn Baid et al. (2021), Menze et al. (2015), và Bakas et
al. (2017); trang này không nêu license hoặc quyền re-host bản dẫn xuất. Không tự
gán CC BY-NC-SA 4.0 cho ảnh. Giữ repo private và không chia sẻ lại trước khi xác
nhận đầy đủ điều kiện áp dụng. Metadata trích dẫn có DOI trong [REFERENCES](REFERENCES.md).
Phát hiện split chỉ áp dụng archive đã pin, không khẳng định mọi bản BMAD bị leakage.

## Việc tiếp theo

B/C xác nhận đọc được revision này trên máy/Colab của mình, rồi đính kết quả vào
[DAC-9](https://linear.app/dacn-khmt/issue/DAC-9/d02-ban-giao-data-contract-va-kiem-chung-tren-may-thanh-vien). Khi đã nhận data,
đo GPU pilot theo [DAC-10](https://linear.app/dacn-khmt/issue/DAC-10/i01-djo-gpu-pilot-va-quyet-djinh-ngan-sach-train).
Lệnh Modal đã có trong [HUGGINGFACE_MODAL](HUGGINGFACE_MODAL.md); thay placeholder
dataset revision bằng SHA phía trên. Chưa khởi chạy job GPU trong task này.
