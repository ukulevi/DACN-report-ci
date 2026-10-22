# DACN — Medical Anomaly Detection with Diffusion Models

**Team:** Tuấn Anh (research/diffusion), Hải Tuấn (baselines), Chí Vĩ (medical data/evaluation).

Research-first reproducible framework for unsupervised/semi-supervised medical-image anomaly detection. **Not for clinical use.**

## Roadmap
1. Data governance and patient-level protocol.
2. AE, VAE, GAN and Isolation Forest baselines.
3. DDPM and DDIM reconstruction benchmarks.
4. Novel diffusion anomaly-scoring / conditioning method, ablations and independent validation.

## Quick start
```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest -q
python -m dacn.cli --help
python -m dacn.cli demo --output runs/demo
```

GPU PyTorch installation is hardware-specific: refer to https://pytorch.org/get-started/locally/ before training.

## Repository
- `src/dacn/`: shared typed contracts, reproducible scoring and metrics
- `configs/`: versioned experiment configurations
- `tests/`: fast unit tests without medical datasets/GPU
- `docs/`: protocols, ownership, research and reproducibility
- `data/`, `runs/`, `checkpoints/`: ignored local artefacts

**Important:** the demo uses synthetic numeric scores, not trained medical models. No baseline or diffusion results are claimed. Select and verify a real medical dataset and licensing conditions before training. Never commit patient-level data, scans, credentials or identifiable metadata.

## Collaboration
Never push to main directly. Create a `feat/<area>-<description>` branch from main; open a PR with tests, config, evaluation outputs, reproducibility metadata, and request another member's review. See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/TEAM.md](docs/TEAM.md).
