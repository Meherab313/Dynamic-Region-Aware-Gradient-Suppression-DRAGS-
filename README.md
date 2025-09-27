# DRAGS — Dynamic Region-Aware Gradient Suppression

> A light-weight, training-time module that suppresses gradients from noisy spatial regions to make vision models a bit less brittle.

This repo contains clean, paper-ready PyTorch code for DRAGS (ResNet‑18 on CIFAR‑10) plus scripts to reproduce the results, ablations, and figures.

## TL;DR

- **Drop‑in wrapper** (`DRAGSLayerWrapper`) for early ResNet stages (e.g., `layer1`, `layer2`).
- **Idea:** compute a simple spatial *noise score*, keep the **top‑k** (most informative) locations, **detach** gradients elsewhere.
- **Outcome:** on CIFAR‑10, **+1.39** clean‑accuracy points over a strong baseline while preserving robustness to Gaussian noise and occlusion (3 seeds). See Table I of the paper for exact numbers.  📈

According to the paper, mean±std over three seeds (ResNet‑18, short schedule):

- Baseline: **62.76 ± 1.05** (clean), **57.30 ± 2.51** (gaussian), **27.16 ± 1.64** (motion_blur), **55.06 ± 2.71** (occlusion), **113.25 ± 2.69s** (train time)
- DRAGS: **64.15 ± 2.30**, **57.54 ± 2.08**, **25.41 ± 1.12**, **55.26 ± 1.38**, **225.76 ± 3.21s**

## Install

```bash
git clone <your-repo-url>.git
cd DRAGS-robustness
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart (single run)

```bash
python -m scripts.run_experiment --epochs 5 --batch-size 128
```

This trains a **baseline** and a **DRAGS** model, evaluates on clean + 3 corruptions, and writes a CSV under `drags_outputs/csv/`.

## Multi‑seed & Aggregation (for the paper table)

```bash
python -m scripts.run_multi_seed --epochs 5 --batch-size 128 --seeds 42 43 44
```

This produces raw per‑seed results and a `multi_seed_summary_*.csv` with `mean ± std` strings ready to paste into LaTeX.

## Ablation study

```bash
python -m scripts.run_ablation --topk 0.05 0.10 0.15 --layers layer1 layer1+layer2 --epochs 2
```

## Heatmap figure (IEEE two‑column)

```bash
python -m scripts.make_heatmap_pdf --num-images 5
```
Saves **`drags_heatmap_ieee.pdf`** under `drags_outputs/figures/`.

## Replot the paper bar chart

```bash
python -m scripts.plot_paper_bars
```

## Project layout

```
drags/
  ├─ layers.py           # DRAGS wrapper
  ├─ models.py           # ResNet-18 builder (with optional DRAGS)
  ├─ data.py             # CIFAR-10 loader (train/test + normalization)
  ├─ robust_eval.py      # clean/gaussian/motion_blur/occlusion eval
  ├─ train.py            # vanilla SGD training
  ├─ experiments.py      # single, multi-seed, aggregate, ablation
  ├─ viz.py              # heatmap PDF (IEEE two-column)
  └─ utils.py            # seeds, device, OUT_DIR helpers
scripts/
  ├─ run_experiment.py
  ├─ run_multi_seed.py
  ├─ run_ablation.py
  ├─ make_heatmap_pdf.py
  └─ plot_paper_bars.py
paper/
  └─ DRAGS_preprint.pdf
```

## Notes & tips
- We keep DRAGS in **early layers** to steer features without affecting late semantics.
- The **keep ratio** (`topk_ratio`) of **0.10–0.15** is a reliable default from the ablation.
- Training here intentionally uses a **short schedule** for quick, reproducible comparisons.
- Inference is unaffected: DRAGS only gates **gradients** during training.

## Citation
If you use this code, please cite the paper (update fields as needed):

```bibtex
@inproceedings{meherab2025drags,
  title={Dynamic Region-Aware Gradient Suppression (DRAGS): Enhancing Vision Model Robustness by Suppressing Noisy Feature Regions During Training},
  author={Meherab, Md Muntaqim and Author, Second B. and Author, Third C.},
  booktitle={IEEE Indexed Conference on Machine Learning},
  year={2025}
}
```

## License
MIT — see `LICENSE`.
