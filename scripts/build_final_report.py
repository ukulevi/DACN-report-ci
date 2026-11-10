"""Build report tables/figures from aggregate artifacts, without sample data."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "final_lpips": "Final LPIPS",
    "initial_lpips": "Initial LPIPS",
    "naive_stitch_lpips": "Naive-stitch LPIPS",
    "brightness": "Brightness",
}


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path)
    parser.add_argument("--training-history", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    boxes = [
        (0, "FLAIR input\n240 crop / 256 pad"),
        (3, "DDIM phase 1\nt300, 20 U-Net calls"),
        (6, "LPIPS x residual\nfrozen pseudo-mask"),
        (9, "Stitch + DDIM phase 2\nt50, 30 U-Net calls"),
    ]
    for x, label in boxes:
        ax.text(
            x,
            1,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=.6", "facecolor": "#e6eff9", "edgecolor": "#31567b"},
        )
    for x in (1, 4, 7):
        ax.annotate("", xy=(x + 1, 1), xytext=(x, 1), arrowprops={"arrowstyle": "->"})
    ax.text(
        4.5,
        -0.1,
        "Controls: initial LPIPS / naive-stitch LPIPS / brightness\n"
        "Final score: LPIPS(input, final reconstruction); fixed pixel/image thresholds",
        ha="center",
        va="center",
    )
    ax.set(xlim=(-1.5, 10.5), ylim=(-0.7, 1.7))
    save(fig, figures, "pipeline")
    text = [
        "# Diffusion Models for Anomaly Detection — báo cáo thực nghiệm\n",
        (
            "Dữ liệu BMAD-derived Brain FLAIR 2D, patient-disjoint-v1: "
            "7.500 train normal, 83 validation, 3.665 test. Normal chỉ có nghĩa "
            "slice không có tumor annotation. Đầu vào là PNG đã render, không phải intensity MRI gốc.\n"
        ),
        "## Phương pháp\n",
        (
            "Gaussian DDPM EMA epoch30, normal-only training; DDIM20(t300) + "
            "DDIM30(t50), eta=0, một lượt pha hai. LPIPS SqueezeNet spatial "
            "tạo score; pseudo-mask dùng LPIPS×normalized residual, threshold "
            "0.13344676792621613, dilation3. Không dùng ground truth để tạo mask.\n"
        ),
        "![Pipeline](figures/pipeline.png)\n",
        "## Protocol\n",
        (
            "Ngưỡng image/pixel riêng từng arm lấy từ 40 calibration validation, "
            "giữ cố định trước đánh giá. Validation holdout43 đã được khảo sát lặp; "
            "không gọi đây là test độc lập. Một training/sampling seed17, "
            "sampling seed suy ra từ UUID riêng tư. Pixel global micro trên crop240; "
            "image score là mean top576pixels; dự đoán score≥threshold.\n"
        ),
    ]
    provenance = {}
    for stage, directory in (("validation", args.validation), ("test", args.test)):
        if directory is None:
            if stage == "test":
                text += ["## Test\n", "Chưa có artifact test được cung cấp cho bản báo cáo này.\n"]
            continue
        path = directory / "aggregate.json"
        report = json.loads(path.read_bytes())
        if report["status"] != "completed" or report["split"] != stage:
            raise ValueError("Report input must be a completed matching-split artifact")
        provenance[stage] = {
            "aggregate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "samples": report["sample_count"],
        }
        text += [
            f"## Kết quả {stage} — {report['sample_count']} ảnh\n",
            (
                "| Arm | Pixel ROC-AUC | AP | Precision | Recall | F1/Dice | Image ROC-AUC | Image F1 |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|\n"
            ),
        ]
        for arm, label in LABELS.items():
            pixel, image = report["arms"][arm]["pixel"], report["arms"][arm]["image"]
            values = [pixel[k] for k in ("auroc", "auprc", "precision", "recall", "f1")]
            values += [image["auroc"], image["f1"]]
            text.append(f"| {label} | " + " | ".join(f"{v:.6f}" for v in values) + " |\n")
        text += [
            (
                "\n| Arm | Image Precision | Image Recall | Image AP | Macro lesion Dice | Normal pixel FPR |\n"
                "|---|---:|---:|---:|---:|---:|\n"
            )
        ]
        for arm, label in LABELS.items():
            p, im = report["arms"][arm]["pixel"], report["arms"][arm]["image"]
            values = [
                im["precision"],
                im["recall"],
                im["auprc"],
                p["macro_lesion_dice"],
                p["normal_pixel_fpr"],
            ]
            text.append(f"| {label} | " + " | ".join(f"{v:.6f}" for v in values) + " |\n")
        text.append(
            f"\nRuntime: {report['unet_calls']} U-Net / {report['lpips_calls']} LPIPS calls; "
            f"adapter time {report['adapter_seconds_sum']:.3f}s, "
            f"median {report['adapter_seconds_median']:.3f}s/ảnh, "
            f"p95 {report['adapter_seconds_p95']:.3f}s/ảnh. "
            "Đây không phải wall time toàn pipeline; xem receipt về scope/memory.\n"
        )
        curves_path = directory / "curves.json"
        if curves_path.exists():
            curves = json.loads(curves_path.read_bytes())
            for level in ("pixel", "image"):
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for arm, label in LABELS.items():
                    c = curves[arm][level]
                    axes[0].plot(c["fpr"], c["tpr"], label=label)
                    axes[1].plot(c["recall"], c["precision"], label=label)
                axes[0].set(
                    xlabel="False positive rate",
                    ylabel="True positive rate",
                    title=f"{stage}: {level} ROC",
                )
                axes[1].set(xlabel="Recall", ylabel="Precision", title=f"{stage}: {level} PR")
                for ax in axes:
                    ax.set(xlim=(0, 1), ylim=(0, 1))
                    ax.legend(fontsize=8)
                    ax.grid(alpha=0.2)
                save(fig, figures, f"{stage}_{level}_curves")
                text.append(f"\n![{stage} {level} ROC/PR](figures/{stage}_{level}_curves.png)\n")
    if args.training_history:
        payload = args.training_history.read_bytes()
        epochs = json.loads(payload)["epochs"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for field, label in (
            ("train_epsilon_mse", "Train"),
            ("normal_validation_epsilon_mse", "Normal validation"),
        ):
            ax.plot([e["epoch"] for e in epochs], [e[field] for e in epochs], label=label)
        ax.set(xlabel="Epoch", ylabel="Epsilon MSE", title="DDPM training (not anomaly quality)")
        ax.legend()
        ax.grid(alpha=0.2)
        save(fig, figures, "training_curve")
        provenance["training_history_sha256"] = hashlib.sha256(payload).hexdigest()
        text += [
            "## Huấn luyện\n",
            "![Training curve](figures/training_curve.png)\n",
            "Epsilon-MSE đo dự đoán noise; không thay thế chất lượng anomaly detection.\n",
        ]
    text += [
        "## Thảo luận và giới hạn\n",
        (
            "Giữ cả bốn phương án khi báo cáo, kể cả kết quả âm. Trên validation lịch sử, "
            "brightness có AP cao hơn diffusion; không tuyên bố diffusion vượt mọi baseline. "
            "Một seed, một contrast MRI 2D, validation đã khảo sát lặp và thiếu metadata "
            "scanner/site hạn chế khả năng suy rộng. Không suy ra hiệu quả chẩn đoán lâm sàng.\n"
        ),
        "## Tài liệu tham khảo\n",
        (
            "DDPM: Ho et al. (2020), arXiv:2006.11239. DDIM: Song et al. (2021), "
            "arXiv:2010.02502. AnoDDPM: Wyatt et al. (CVPRW 2022). "
            "LPIPS: Zhang et al., The Unreasonable Effectiveness of Deep Features as a "
            "Perceptual Metric (CVPR 2018). BMAD: Bao et al., arXiv:2306.11876. "
            "BraTS: Baid et al. (2021), Menze et al. (2015), Bakas et al. (2017). "
            "Xem REFERENCES.md về nguồn và metadata đầy đủ.\n"
        ),
        "## Đóng gói\n",
        (
            "Bản Markdown có bảng/hình từ artifact thực. Chưa áp dụng mẫu trường khi chưa "
            "có mẫu được cung cấp. Ảnh MRI minh họa và bản đồ theo mẫu chỉ được giữ trong "
            "artifact riêng tư, không đưa vào Git hoặc bản phát hành công khai.\n"
        ),
    ]
    (args.output / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
