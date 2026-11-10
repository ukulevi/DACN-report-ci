# Tài liệu tham khảo và thứ tự đọc

Đối chiếu `Paper_Reference.xlsx` ngày 24/10/2026: 9 paper nền tảng/phương pháp;
sheet Dataset chưa có dữ liệu. Workbook của nhóm được giữ nguyên. Bảng dưới bổ sung
nguồn sơ cấp và mục đích đọc; không hàm ý tất cả phương pháp đều phải triển khai.

| Tài liệu | Nguồn chính | Dùng cho đồ án |
|---|---|---|
| GAN — Goodfellow et al., 2014 | [Paper](https://arxiv.org/abs/1406.2661) | Related work; không đưa vào baseline bắt buộc |
| Deep autoencoder — Hinton & Salakhutdinov, 2006 | [Trang tác giả](https://www.cs.toronto.edu/~rsalakhu/bibtex/HinSal06.html), DOI 10.1126/science.1127647 | Nền tảng reconstruction |
| VAE — Kingma & Welling, 2013/2014 | [Paper](https://arxiv.org/abs/1312.6114) | Điền nguồn còn thiếu; nền tảng latent generative models |
| Isolation Forest — Liu et al., 2008 | [PDF tác giả](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) | Điền nguồn còn thiếu; anomaly detection cổ điển |
| Score matching — Song & Ermon, 2019 | [Paper](https://arxiv.org/abs/1907.05600) | Hiểu score-based modeling |
| DDPM — Ho et al., 2020 | [Paper](https://arxiv.org/abs/2006.11239) | Noise schedule, objective, reverse process |
| DDIM — Song et al., 2020/2021 | [Paper](https://arxiv.org/abs/2010.02502) | Phân biệt sampler/reverse steps và timestep bắt đầu tái tạo |
| LPIPS — Zhang, Isola, Efros, Shechtman & Wang, CVPR 2018 | [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric](https://arxiv.org/abs/1801.03924v2) | Spatial perceptual distance; input adaptation và SqueezeNet weights của đồ án phải mô tả riêng |
| Score-based SDE — Song et al., 2020/2021 | [Paper](https://arxiv.org/abs/2011.13456) | Related work; đọc sau DDPM |
| Diffusion Models for Medical Anomaly Detection — Wolleb et al., 2022 | [Paper](https://arxiv.org/abs/2203.04306) | Có classifier guidance; không mô tả như normal-only thuần túy |
| AnoDDPM — Wyatt et al., 2022 | [Paper](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf), [code](https://github.com/Julian-Wyatt/AnoDDPM) | Partial diffusion và ảnh hưởng của noise; không tuyên bố tái lập nguyên bản |
| On the Importance of Noise in Denoising Autoencoders for Unsupervised Anomaly Detection — Kascenas et al., 2023 | [Paper](https://arxiv.org/abs/2301.08330) | DAE/coarse noise baseline; đọc trước khi chọn noise |
| ANDi — Frotscher et al. | [Paper](https://arxiv.org/html/2312.01904v1), [code](https://github.com/AlexanderFrotscher/ANDi) | Related work; tổng hợp denoising khác với reconstruction |
| Structure-based scoring — Behrendt et al., 2024 | [Paper](https://arxiv.org/abs/2403.14262) | Tham khảo local SSIM và giới hạn chọn window |
| Conditioned diffusion — Behrendt et al., 2025 | [Paper](https://arxiv.org/abs/2312.04215) | Related work về tái tạo có điều kiện; không mặc định triển khai |
| THOR, 2024 | [Paper](https://arxiv.org/abs/2403.08464) | Related work về restoration/localization; chưa đưa vào phạm vi triển khai |
| BMAD — Bao et al. | [Paper](https://arxiv.org/html/2306.11876v3), [repo](https://github.com/DorisBao/BMAD), [preprocessing](https://github.com/DorisBao/BMAD/blob/main/data_processing/brain/bras.py) | Dataset chính; cần kiểm tra archive thực tế, không chỉ chép bảng paper |

### Trích dẫn dữ liệu BraTS bắt buộc

Trang chính thức [BraTS 2021 — Data Usage Agreement / Citations](https://www.med.upenn.edu/cbica/brats2021/)
nói có thể dùng dữ liệu trong nghiên cứu nếu luôn trích dẫn ba công trình dưới đây.
Trang đó không nêu license cho việc re-host dữ liệu phái sinh; không suy ra quyền
công bố lại từ quyền dùng cho nghiên cứu.

1. U. Baid et al., [“The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification”](https://arxiv.org/abs/2107.02314), 2021.
2. B. H. Menze et al., [“The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)”](https://doi.org/10.1109/TMI.2014.2377694), *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024, 2015.
3. S. Bakas et al., [“Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features”](https://doi.org/10.1038/sdata.2017.117), *Scientific Data*, 4, 170117, 2017.

## Lộ trình đọc theo vai trò

**Bạn / data:** BMAD phần Brain → mã `bras.py` → audit của dự án → AnoDDPM và DAE
phần dữ liệu/preprocessing. Mỗi nguồn ghi: tên/version, link gốc, điều kiện
sử dụng, modality, đơn vị sample, nhãn, patient ID, split, kích thước, cách tải,
checksum, hạn chế. Phân biệt số paper công bố với số đo trên bản tải.

**B / baseline & metric:** DAE noise → protocol đánh giá dự án → BMAD experiments.
**C / diffusion:** DDPM → AnoDDPM partial diffusion → SSIM scoring; ANDi là related work.
THOR và Wolleb dùng xác định vị trí nghiên cứu, tránh gộp các phương pháp khác mức
giám sát vào một bảng mà không giải thích.

Mẫu ghi chú paper: bài toán; training có dùng nhãn không; dataset/split; công thức
anomaly score; inference cost; code/license; kết quả và hạn chế; điểm có thể tái lập;
sai khác khi chuyển sang thiết lập của nhóm.

## Nguồn dự phòng đã cân nhắc

- [IXI](https://brain-development.org/ixi-dataset/): không chọn làm nguồn normal chính
  vì danh sách chuỗi công bố không có FLAIR; ghép với BMAD gây khác domain/modality.
- [BraTS/Synapse thông báo phiên bản](https://www.synapse.org/Synapse:syn25829067/discussion/threadId=12691):
  quyền truy cập/phiên bản có thay đổi; không thay BraTS2021 bằng bản mới mà giữ nhãn
  version cũ. Dữ liệu bốn chuỗi để giai đoạn sau nếu cần tái lập ANDi sát hơn.

Giảm timestep đã nằm trong vùng ý tưởng được nghiên cứu. Nếu muốn tuyên bố tính mới,
phải cập nhật literature search và đối chiếu công thức trước; đợt này chọn đóng góp
tái triển khai, audit dữ liệu và đánh giá trade-off, không hứa một thuật toán mới.
