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
| DDIM — Song et al., 2020/2021 | [Paper](https://arxiv.org/abs/2010.02502) | Phân biệt giảm sampling steps và giảm timestep scoring |
| Score-based SDE — Song et al., 2020/2021 | [Paper](https://arxiv.org/abs/2011.13456) | Related work; đọc sau DDPM |
| Diffusion Models for Medical Anomaly Detection — Wolleb et al., 2022 | [Paper](https://arxiv.org/abs/2203.04306) | Có classifier guidance; không mô tả như normal-only thuần túy |
| AnoDDPM — Wyatt et al., 2022 | [Mã nguồn tác giả](https://github.com/Julian-Wyatt/AnoDDPM) | Noise cho medical AD; đối chiếu lịch sử với ANDi |
| On the Importance of Noise in Denoising Autoencoders for Unsupervised Anomaly Detection — Kascenas et al., 2023 | [Paper](https://arxiv.org/abs/2301.08330) | DAE/coarse noise baseline; đọc trước khi chọn noise |
| ANDi — Frotscher et al. | [Paper](https://arxiv.org/html/2312.01904v1), [code](https://github.com/AlexanderFrotscher/ANDi) | Phương pháp chính; data, noise, deviation, geometric aggregation, appendix timestep |
| THOR, 2024 | [Paper](https://arxiv.org/abs/2403.08464) | Related work về restoration/localization; chưa đưa vào phạm vi triển khai |
| BMAD — Bao et al. | [Paper](https://arxiv.org/html/2306.11876v3), [repo](https://github.com/DorisBao/BMAD), [preprocessing](https://github.com/DorisBao/BMAD/blob/main/data_processing/brain/bras.py) | Dataset chính; cần kiểm tra archive thực tế, không chỉ chép bảng paper |

## Lộ trình đọc theo vai trò

**Bạn / data:** BMAD phần Brain → mã `bras.py` → audit của dự án → ANDi phần dữ
liệu và preprocessing → DAE noise. Mỗi nguồn ghi: tên/version, link gốc, điều kiện
sử dụng, modality, đơn vị sample, nhãn, patient ID, split, kích thước, cách tải,
checksum, hạn chế. Phân biệt số paper công bố với số đo trên bản tải.

**B / baseline & metric:** DAE noise → protocol đánh giá dự án → BMAD experiments.
**C / diffusion:** DDPM → AnoDDPM → ANDi paper + config/code → appendix timestep.
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
