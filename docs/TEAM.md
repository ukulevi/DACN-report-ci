# Parallel work ownership

| Owner | Workstream | Primary paths | Deliverable |
|---|---|---|---|
| Tuấn Anh | Diffusion / integration / new algorithm | `src/dacn/models/diffusion/`, `configs/diffusion/`, `docs/RESEARCH.md` | DDPM/DDIM reproducible benchmark, novel method, ablations |
| Hải Tuấn | Traditional/generative baselines | `src/dacn/models/baselines/`, `configs/baselines/` | Isolation Forest, AE, VAE, GAN anomaly score benchmark |
| Chí Vĩ | Medical data & evaluation | `src/dacn/data/`, `src/dacn/evaluation/`, `docs/DATA_PROTOCOL.md` | Legal dataset ingestion, leakage-free splits, metrics and visualization |

All three co-own tests, documentation and review. Shared interfaces in `src/dacn/contracts.py` require one review from each affected workstream before modification. Assign actual GitHub handles when team joins.

## First four sprints (proposed, one week each)
- Sprint 1: agree a single medical modality/task, dataset access, patient splits, contracts, and compute budget; smoke-test CI.
- Sprint 2: baseline training + comparable metric pipeline; reproduce reference DDPM denoising on a tiny subset.
- Sprint 3: DDPM vs DDIM scores, inference-speed trade-offs, reproducible initial report.
- Sprint 4: run one well-specified novel-method hypothesis and ablations, hold out test set until selection is frozen.

Each weekly sync: 10 min blockers, 15 min results with configs and run links, 10 min decisions. Document decisions as Issues/PRs.
