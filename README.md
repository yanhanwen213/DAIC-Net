# DAIC-Net: Differentiable Alignment and Illumination-Invariant Contrastive Network for Parameter-Efficient Finger Vein Recognition

Official implementation of **DAIC-Net** — a parameter-efficient framework for
finger vein recognition when the verification session is unavailable. DAIC-Net
separates cross-session variation into two components and resolves both without
any target-session data:

- **Differentiable alignment front-end** — `AlignmentHead` predicts a rigid
  affine pose `p = (θ, t_x, t_y)` and the differentiable `Warper` rectifies the
  input (`x̂ = W(x; Θ)`), jointly optimized with recognition through the pose
  regularization `L_pose = ||p||²`.
- **Illumination-invariant contrastive module** — within-identity (intra-class)
  augmentation synthesizes geometric and photometric variation from Session-1
  alone, and an `IlluminationInvariantHead` projects the backbone embedding to
  `z ∈ R^128` trained with supervised contrastive learning (`L_supcon`, SupCon,
  τ = 0.07), while a `CosFace` margin (`L_cos`, s = 64, m = 0.35) preserves
  discrimination.

The backbone is the lightweight `LRFVNet`, with `MixStyle` (p = 0.5) after
conv1/conv2. With **1.47M parameters at inference**, DAIC-Net reaches 90.66%
Rank-1 on FV-USM and 94.63% Rank-1 averaged over three UTFVP splits
(3.27% and 2.21% EER).

> **Strict protocol.** Session-2 is invisible during training *and*
> hyper-parameter tuning. FV-USM: train/tune on Session-1 only (Gallery = S1,
> Probe = S2); checkpoints are picked on a held-out 10% split of Session-1.
> UTFVP: three 3-train/1-test splits with disjoint training sets (seeds
> 42/43/44, 1080/360; pairwise probe overlap 84–96 of 360).

## Pretrained weights and splits

<!-- TODO before submission: upload the weights (GitHub Release or Zenodo) and
     the three UTFVP split files, then replace the placeholders below. -->

| Item | Location |
|---|---|
| FV-USM seed-42 model (M5, FT30) | `<TODO: link>` |
| FV-USM three-seed models (42/43/44) | `<TODO: link>` |
| UTFVP splits (seeds 42/43/44) | `splits/UTFVP_3T1T{,_43,_44}/` |

Each checkpoint folder contains `daic_best.pth` and `config.json` (see
`checkpoints/` below).

## Results (from the paper)

Strict protocol (S1-only training, S2-only testing; `112×112×1`; %).

| Method | Group | Infer. params | FV-USM Rank-1 ↑ | FV-USM EER ↓ | UTFVP Rank-1 ↑ | UTFVP EER ↓ |
|---|---|---|---|---|---|---|
| LRFVNet | A | 0.68M | 45.66±2.08 | 12.89±1.24 | 81.85±1.60 | 4.97±0.75 |
| ResNet-18 | A | 11.42M | 54.37±1.47 | 11.74±0.74 | 85.37±1.89 | 3.81±0.09 |
| RepViT-M1.1 | A | 7.77M | 50.73±1.39 | 8.19±0.12 | 82.50±2.16 | 3.71±0.13 |
| FastViT-T8 | A | 3.25M | 38.08±2.58 | 14.20±1.89 | 74.44±4.60 | 5.47±0.69 |
| ConvNeXtV2-Femto | A | 4.82M | 34.10±1.17 | 15.19±0.86 | 85.00±0.79 | 3.60±0.22 |
| **DAIC-Net (ours)** | **A** | **1.47M** | **90.66±0.81** | **3.27±0.41** | **94.63±0.32** | **2.21±0.24** |
| Miura (reproduced) | B | n.a. | 58.37 | 12.45 | 77.96±0.35 | 8.09±0.40 |
| Asaari (reproduced) | B | n.a. | 74.90 | 9.20 | 69.17±2.39 | 9.82±0.36 |

Group A: re-trained from scratch under the strict protocol (no pre-training, no
mixup, S1-validation best checkpoint). Group B: classical methods reproduced
under the strict recipe. Fang and Shaheed are not included as strict-protocol
controls (no released training code); their joint-protocol figures are quoted in
the paper for lineage only.

### Progressive component buildup (FV-USM, strict)

| ID | Method | Rank-1 | EER |
|---|---|---|---|
| M1 | Bare backbone | 51.86 | 9.14 |
| M2 | + augmented samples (within-identity aug.) | 86.96 | 4.04 |
| M3 | + alignment + MixStyle | 88.55 | 3.36 |
| M4 | + CosFace | 89.97 | 3.33 |
| M5 | + FT30 (full DAIC-Net) | **90.66±0.81** | **3.27±0.41** |
| M6 | Joint oracle (S2 visible) | 99.49 | 0.10 |

M6 is a non-strict oracle and not comparable. The rigid front-end sustains
rotation ≤ 30° and shift ≤ 30 px; residual failures are 83.0±0.4%
pose-recoverable, 0.5±0.2% illumination and 16.5±0.6% intrinsically ambiguous
(3 seeds, 276±20 failures).

## Layout

```
DAIC-Net/
├── README.md
├── LICENSE                     # MIT
├── requirements.txt
├── daicnet/                    # training + evaluation entry points
│   ├── train.py                # training (strict s1_only / joint ablation)
│   ├── evaluate.py             # strict cross-session evaluation (S1 gallery vs S2 probe)
│   ├── resnet18_baseline.py    # strict-protocol ResNet-18 control
│   ├── vis_roi_pose_native.py
│   ├── logger.py
│   ├── sam.py
│   └── daicnet/                # inner model package (imported as `daicnet`)
│       ├── daic_model.py       # DAICNet (AlignmentHead + Warper + LRFVNet + IlluminationInvariantHead)
│       ├── alignment.py
│       ├── warping.py
│       ├── illumination_head.py
│       ├── angular_margin.py
│       ├── style_layers.py     # MixStyle / DSU / pAdaIN
│       ├── domain_adv.py
│       ├── daic_dataset.py
│       ├── augment.py
│       ├── roi.py
│       ├── daic_losses.py
│       └── daic_metrics.py
├── evaluation/                 # analysis and figures
│   ├── pose_sweep.py           # fixed pose-robustness scan (rot / shift / combo)
│   ├── kfold.py                # intra-session control
│   ├── error_attribution.py    # 3-seed residual-error attribution
│   ├── plot_attribution.py     # publication figure for the attribution
│   └── load_model.py
├── baselines/                  # reproduced classical methods (deterministic, no training)
│   ├── miura/                  # miura_baseline.py (FV-USM), miura_utfvp.py (UTFVP)
│   ├── asaari/                 # asaari_baseline.py, asaari_utfvp.py
│   └── shaheed/                # shaheed_baseline.py, shaheed_utfvp.py
├── splits/                     # UTFVP 3-train/1-test split lists (seeds 42/43/44)
├── checkpoints/                # put trained run folders here (see below)
└── figures/                    # generated by evaluation/plot_attribution.py
```

The `LRFVNet` backbone is an external dependency (imported as `lrfvnet`).
Place it next to this repo (`../LRFVNet`) or set `LRFVNET_PATH`.
FV-USM / UTFVP datasets are not included; configure paths as below.

## Environment

- Python 3.10+, PyTorch 2.x + torchvision, scikit-image, scipy, tqdm,
  matplotlib, opencv-python, timm (Shaheed Xception option only)
- CUDA GPU recommended

```bash
pip install -r requirements.txt
```

Reference environment used for the reported numbers: `<TODO: PyTorch and CUDA
versions, GPU model, and CPU model plus thread count used for the latency
measurements>` (latency measured at batch size 1, median and p90 over 100 runs).

## Data paths

| Variable | Meaning | Used by |
|---|---|---|
| `FVUSM_ROOT` | FV-USM database root | `baselines/miura`, `baselines/asaari` |
| `UTFVP_ROI` | UTFVP per-class ROI images (`extractedvein`) | `baselines/*/*_utfvp.py` |
| `UTFVP_SPLIT` | parent folder of the `UTFVP_3T1T*` split folders | `baselines/*/*_utfvp.py` |
| `LRFVNET_PATH` | external `LRFVNet` repo location | all `daicnet` / `evaluation` scripts |
| `DAIC_CKPT_DIR` | folder holding the three FT30 run folders | `evaluation/error_attribution.py` |

`train.py` / `evaluate.py` take `--session1_root` / `--session2_root` directly.

## Reproduce

The `train.py` defaults already match the paper configuration (MixStyle
`p=0.5`, CosFace `s=64, m=0.35`, SupCon `τ=0.07`, `L_pose` weight `0.01`,
160 epochs, batch 64, weight decay 5e-4).

```bash
# 1) Train DAIC-Net (Session-1 only; Session-2 never seen in training)
python daicnet/train.py --session1_root <S1>/extractedvein \
  --session2_root <S2>/extractedvein --train_protocol s1_only \
  --epochs 160 --batch_size 64

# 30-epoch fine-tune (FT30)
python daicnet/train.py --session1_root <S1>/extractedvein \
  --session2_root <S2>/extractedvein --train_protocol s1_only \
  --resume <ckpt> --epochs 30 --lr 2e-5 --run_name s1_only_ft30

# 2) Classical baselines (hyper-parameters calibrated on train only)
export FVUSM_ROOT=<path-to-FV-USM> UTFVP_ROI=<path-to-extractedvein> \
  UTFVP_SPLIT=<parent-of-UTFVP_3T1T*>
python baselines/miura/miura_baseline.py
python baselines/miura/miura_utfvp.py
python baselines/asaari/asaari_baseline.py
python baselines/asaari/asaari_utfvp.py

# 3) Error attribution (3 seeds) + figure
python evaluation/error_attribution.py --ckpt_dir checkpoints/daic
python evaluation/plot_attribution.py

# 4) Pose-robustness scan / intra-session control
python evaluation/pose_sweep.py --ckpt <ckpt> --data_root <root>
python evaluation/kfold.py
```

Expected layout of `checkpoints/daic` for step 3:

```
checkpoints/daic/
├── daic_s1_only_mixstyle_cosface_ft30/       # seed 42
├── daic_s1_only_mixstyle_cosface_ft30_s43/   # seed 43
└── daic_s1_only_mixstyle_cosface_ft30_s44/   # seed 44
```

each holding `daic_best.pth` + `config.json`.

### Group-A controls

ResNet-18 is implemented here (`daicnet/resnet18_baseline.py`). RepViT-M1.1,
FastViT-T8 and ConvNeXtV2-Femto were re-trained from their official
repositories with the same data pipeline, optimizer, schedule and
model-selection rule (no pre-training, no mixup, S1-validation best); the shared
pipeline lives in `daicnet/daicnet/daic_dataset.py`.

## Notes

- Classical baselines are deterministic (no training); UTFVP values are
  mean±std over the three splits.
- All trained-model numbers in the paper are three-seed mean±std (42/43/44).
- `AngularMarginHead` (CosFace) is training-only: it adds no inference
  parameters, and verification uses cosine similarity in `z`-space.
- Inference removes the classification head and the augmentation branch; the
  1.47M count refers to this deployment path (0.30 GFLOPs per 112×112 ROI).

## Troubleshooting

- `ModuleNotFoundError: lrfvnet` — set `LRFVNET_PATH` or clone `LRFVNet` next to
  this repository.
- CUDA out of memory — reduce `--batch_size`; the reported results use 64.
- Latency differs from the paper — fix the CPU/GPU, thread count and batch size
  (batch 1) before comparing.

## Citation

<!-- TODO before submission: replace the placeholder author names and add the
     volume/pages/DOI after acceptance. -->

```bibtex
@article{daicnet2026,
  title   = {DAIC-Net: Differentiable Alignment and Illumination-Invariant
             Contrastive Network for Parameter-Efficient Finger Vein Recognition},
  author  = {First Author and Second Author and Third Author},
  journal = {Journal of Visual Communication and Image Representation},
  year    = {2026}
}
```

## License

MIT (see `LICENSE`). The FV-USM and UTFVP datasets keep their original licenses
and are **not** redistributed here; the sample ROIs shown in the paper are taken
from the public FV-USM database and are used for illustration only.
