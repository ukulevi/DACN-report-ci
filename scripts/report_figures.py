"""Regenerate aggregate-only report vector figures (optional reportlab dependency).

Run with a Python environment containing reportlab. CI consumes these committed
PDFs; it does not install plotting or training packages. No patient data loaded.
"""

import csv
import hashlib
import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/figures"
BLUE, GRAY, ORANGE = map(HexColor, ("#135D8C", "#607080", "#C56C29"))


def canvas(name, height):
    OUT.mkdir(parents=True, exist_ok=True)
    return Canvas(str(OUT / f"{name}.pdf"), pagesize=(500, height), invariant=1)


def text(c, x, y, value, size=10, color=GRAY):
    c.setFillColor(color)
    c.setFont("Helvetica", size)
    c.drawString(x, y, value)


def pipeline():
    c = canvas("pipeline", 235)
    steps = [
        ("Audit and decode", "Patient-disjoint split"),
        ("Normal-only train", "Gaussian DDPM + EMA"),
        ("DDIM20, t=300", "Initial reconstruction"),
        ("Pseudo-mask", "Normal-calibrated"),
        ("Stitch + DDIM30", "One pass, t=50"),
        ("LPIPS + evaluation", "40 calibration / 43 holdout"),
    ]
    for i, (title, detail) in enumerate(steps):
        row, col = divmod(i, 3)
        x, y = 5 + col * 168, 150 - row * 100
        c.setStrokeColor(BLUE)
        c.setFillColor(HexColor("#F1F6F9"))
        c.roundRect(x, y, 155, 60, 5, fill=1)
        text(c, x + 9, y + 37, title, 11, BLUE)
        text(c, x + 9, y + 18, detail, 9)
        if col < 2:
            c.line(x + 156, y + 30, x + 167, y + 30)
    c.line(418, 150, 418, 128)
    c.line(418, 128, 82, 128)
    c.line(82, 128, 82, 110)
    text(c, 5, 17, "Conceptual workflow; no private images or individual predictions.", 10)
    c.save()


def validation():
    data = json.loads(
        (ROOT / "docs/evidence/2026-11-08-colab-registered-acceptance.json").read_text()
    )
    c = canvas("validation", 245)
    labels = ["Initial", "Naive stitch", "Final", "Brightness"]
    keys = ["initial_lpips", "naive_stitch_lpips", "final_lpips", "brightness"]
    for panel, (metric, title) in enumerate(
        [("average_precision", "Pixel average precision"), ("f1", "Pixel micro Dice")]
    ):
        base = panel * 255
        text(c, base + 10, 225, title, 12, BLUE)
        c.setStrokeColor(GRAY)
        c.line(base + 20, 45, base + 240, 45)
        for i, key in enumerate(keys):
            value = data["arms"][key]["metrics"]["pixel"][metric]
            x = base + 28 + i * 54
            c.setFillColor(ORANGE if key == "brightness" else BLUE)
            c.rect(x, 45, 34, value / 0.5 * 150, fill=1, stroke=0)
            text(c, x - 1, 52 + value / 0.5 * 150, f"{value:.3f}", 9)
            text(c, x - 9, 29, labels[i], 8)
        text(c, base + 10, 8, "Scale: 0 to 0.5; 43 holdout slices, seed 17", 8)
    c.save()


def ablations():
    rows = list(csv.DictReader((ROOT / "docs/evidence/2026-11-10-ddim-ablation-table.csv").open()))
    c = canvas("ablations", 260)
    for panel, phase1 in enumerate((20, 30)):
        base = panel * 255
        text(c, base + 15, 242, f"{phase1}+{50 - phase1} model calls", 12, BLUE)
        points = sorted(
            (r for r in rows if int(r["phase1_steps"]) == phase1),
            key=lambda r: int(r["t_start"]),
        )
        for yv in (0.1, 0.2, 0.3, 0.4):
            y = 40 + (yv - 0.1) / 0.3 * 165
            c.setStrokeColor(HexColor("#D8E0E6"))
            c.line(base + 35, y, base + 238, y)
            text(c, base + 5, y - 3, f"{yv:.1f}", 9)
        for field, color in (("calibration_pixel_ap", BLUE), ("holdout_pixel_ap", ORANGE)):
            c.setStrokeColor(color)
            previous = None
            for r in points:
                x = base + 35 + (int(r["t_start"]) - 100) / 700 * 203
                y = 40 + (float(r[field]) - 0.1) / 0.3 * 165
                if previous:
                    c.line(*previous, x, y)
                c.setFillColor(color)
                c.circle(x, y, 2.5, fill=1, stroke=0)
                previous = x, y
        for t in (100, 300, 500, 800):
            text(c, base + 30 + (t - 100) / 700 * 203, 25, str(t), 9)
    text(c, 12, 5, "Pixel AP vs starting noise level; blue: calibration, orange: holdout", 10)
    c.save()


if __name__ == "__main__":
    pipeline()
    validation()
    ablations()
    sources = (
        "docs/evidence/2026-11-08-colab-registered-acceptance.json",
        "docs/evidence/2026-11-10-ddim-ablation-table.csv",
    )
    (OUT / "provenance.json").write_text(
        json.dumps(
            {
                "hash_normalization": "CRLF-to-LF",
                "inputs": {
                    k: hashlib.sha256((ROOT / k).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                    for k in sources
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
