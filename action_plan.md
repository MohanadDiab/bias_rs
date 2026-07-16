# Action Plan: Bias-Aware Robust Detection in Remote Sensing

## Status

**Now:** solidify experimental design (this document).  
**Next:** download datasets via [Dataset Ninja](https://datasetninja.com/) (`dataset-tools`).  
**Later:** deferred items (§7).

---

## 1. Research Question

**Primary question.**  
Under label noise in remote sensing (RS) object detection, how do **objectness**, **localization**, and **confirmation/classification** biases arise, and which mitigation strategies—across fusion, calibration, and model asymmetry—best control each bias?

**Sub-questions.**
1. For each bias type, which interventions improve robustness under controlled corruption of that bias?
2. Do solutions transfer across datasets (satellite, UAV, domain-specific), or are they dataset-dependent?
3. Does architectural asymmetry (YOLO26 vs RF-DETR vs custom) outperform symmetric ensembles under the same fusion/calibration recipe?
4. Among fusion families (NMS / Soft-NMS / WBF / MoE variants), which best trades AP, recall, and cost under increasing noise?

**Working hypothesis.**  
Biases are partially separable: localization responds most to fusion (NMS family vs WBF), objectness to calibration and confidence-aware gating, confirmation to architectural diversity. A systematic per-bias factorial design will identify which components matter—and whether a full MoCaE-style stack is necessary or overkill.

---

## 2. Approach

Keep the **three-bias framing**, but stop assuming a single fixed pipeline. Instead:

1. **Induce** each bias in isolation (and in controlled combinations) via noise protocols.
2. **Probe** multiple candidate solutions per bias.
3. **Compare** asymmetric vs symmetric expert sets on the same data and noise.
4. **Validate** across a diverse RS dataset suite (not DOTA-only).

| Bias | What we control / measure | Solution candidates (non-exhaustive) |
|------|---------------------------|--------------------------------------|
| **Localization** | Box jitter, spurious/missing boxes; IoU/recall | NMS, Soft-NMS, WBF |
| **Objectness** | Confidence distortion under noise; ECE / reliability | Uncalibrated scores, calibrated MoE, gated MoE |
| **Confirmation / class** | Class flips; correlated errors across experts | Simple MoE, learned MoE, asymmetric models, symmetric models |

**Design principles.**
- One primary factor per controlled run; report interactions only after main effects.
- Same train/cal/val splits and metrics across all methods for fair comparison.
- Asymmetry = architectural diversity (YOLO26, RF-DETR, custom network), not co-teaching unless added later.
- Plug-in / post-hoc methods preferred for fusion and calibration; learned MoE and custom net are the trainable extensions.

---

## 3. Methodology

### 3.1 Detector experts

| Expert | Role |
|--------|------|
| **YOLO26** | Modern CNN/YOLO-family baseline |
| **RF-DETR** | Transformer / DETR-family baseline |
| **Custom network** | Controllable third expert (architecture TBD in implementation phase) |

**Symmetry conditions.**
- **Asymmetric set:** YOLO26 + RF-DETR + custom (heterogeneous).
- **Symmetric sets:** replicas / same-family variants of one architecture (e.g., three YOLO-scale variants), used as control for confirmation-bias claims.

### 3.2 Solution matrix (methods to implement and compare)

**Fusion / localization**
- NMS
- Soft-NMS
- WBF

**Mixture / objectness & confirmation**
- Simple MoE (score-based merge without learned gate)
- Calibrated MoE (per-expert \(\phi_k\) on held-out cal set, then fuse)
- Gated MoE (explicit gate over experts; details TBD)
- Learned MoE (trainable combination; details TBD)

**Architecture regime**
- Asymmetric multi-expert
- Symmetric multi-expert

**Calibration methods** (for calibrated MoE and diagnostics)  
IR, TS, Platt, Beta, Dirichlet (and LR if useful)—select per experiment after a fixed bake-off on one dataset.

### 3.3 Controlled bias runs

Each run family holds dataset and experts fixed and varies **one primary corruption** (plus a clean control):

| Run family | Primary corruption | Target bias | Primary methods under test |
|------------|-------------------|-------------|----------------------------|
| L-runs | Box jitter; optional spurious / removed boxes | Localization | NMS, Soft-NMS, WBF (± MoE wrappers) |
| O-runs | Score/label noise that stresses confidence; report ECE | Objectness | Simple vs calibrated vs gated MoE |
| C-runs | Class flips; measure cross-expert error correlation | Confirmation / class | Asymmetric vs symmetric; Simple vs learned MoE |
| Mix-runs (later) | Combined corruptions at fixed ratios | Interactions | Best-of-each from above |

Noise ratios (default grid): `0, 1, 2, 5, 10, 20, 30, 50%` (subset early; full grid on primary datasets).

### 3.4 Metrics
- Detection: AP, AR (and class-wise AP where meaningful).
- Calibration: ECE, reliability diagrams; confidence KDEs split by **correct vs incorrect** matches.
- Bias diagnostics: cross-expert agreement / disagreement, IoU of fused vs expert boxes, expert coverage complementarity.
- Cost: train time, inference latency, #experts (report always for MoE variants).

### 3.5 Protocol per dataset
1. Standardize annotations to a common detection format (boxes + class).
2. Fixed splits: train / cal / val (cal disjoint; used only for calibration / gate fitting where applicable).
3. Train each expert on clean and noisy trains as required by the run family.
4. Apply each method in the solution matrix; log predictions and metrics.
5. Ablate: fusion only → +calibration → +gate/learned MoE → asymmetric vs symmetric.

---

## 4. Datasets

Acquire via **Dataset Ninja** (`pip install dataset-tools`):

```python
import dataset_tools as dtools
dtools.download(dataset="<Exact Dataset Ninja name>", dst_dir="data/dataset-ninja/")
```

### 4.1 Target suite (initial)

| Dataset | Notes / role |
|---------|----------------|
| **xView** | Large-scale satellite; dense / small objects |
| **HIT** | Confirm exact Dataset Ninja listing name at download time |
| **UAV** | Aerial / drone regime (e.g. UAVDT, VisDrone, or named UAV set on Ninja) |
| **HRSC2016-MS** | Ships / multi-scale maritime |
| **Palm trees** | Domain-specific vegetation / counting-style detection |
| **Additional RS sets** | Expand as needed (“and so on”) once primary suite is stable |

Exact Dataset Ninja page names will be pinned in a download log during the next step.

### 4.2 Dataset readiness checklist (per set)
- [ ] Downloaded via `dataset-tools`
- [ ] License / citation recorded
- [ ] Converted to unified format
- [ ] Split into train / cal / val
- [ ] Class map documented
- [ ] Smoke-train one expert (YOLO26) on clean data

---

## 5. Testing & Experiment Matrix

### 5.1 Minimal viable redo (before full factorial)
1. One primary dataset ready end-to-end (format + splits + smoke train).
2. Clean baseline: YOLO26, RF-DETR, custom (if ready) individually.
3. L-runs: NMS vs Soft-NMS vs WBF on asymmetric trio.
4. O-runs: Simple MoE vs Calibrated MoE.
5. C-runs: asymmetric vs symmetric under Simple MoE.
6. Correct/incorrect confidence + ECE plots (addresses prior overconfidence critique).

### 5.2 Full redo (after MVP)
- Sweep noise grid × datasets × methods in §3.2.
- Add gated MoE and learned MoE once Simple/Calibrated are stable.
- Mix-runs and cost–performance Pareto.

### 5.3 Success criteria
- Per bias: at least one method clearly beats the naive baseline for that bias family.
- Asymmetric > symmetric under confirmation-stress (C-runs), or document when it fails.
- Calibrated MoE ≥ Simple MoE on ECE and fusion AP under O-runs.
- WBF (or Soft-NMS) improves recall vs hard NMS under L-runs where expected.
- Results hold on ≥2 datasets beyond a single benchmark.

---

## 6. Reviewer Concerns (still apply; how this redo addresses them)

| Concern | How this plan addresses it |
|---------|----------------------------|
| Single dataset / DOTA-only | Multi-dataset suite via Dataset Ninja |
| Synthetic noise only | Controlled per-bias noise first; structured/real noise deferred to §7 |
| No SOTA / method comparisons | Explicit solution matrix (NMS family, MoE variants, asym/sym) |
| Overconfidence vs high confidence | Correct vs incorrect + ECE required in O-runs |
| “Just a good ensemble” | Factorial isolation of fusion vs calibration vs asymmetry |
| Novelty / clarity | Bias-disentangled experimental story + multiple solutions per bias |
| Reproducibility / cost | Unified protocol, download log, latency tables |
| Ablation ambiguity | Named expert sets; fixed method list; no hand-wavy “best trend” |

Deferred reviewer items that wait for §7: real annotation noise corpora, heavy SOTA co-teaching bake-off, full deployment study.

---

## 7. Deferred (“some other stuff later”)

- Real-world / weakly labeled noise corpora
- Extra baselines (co-teaching, sample selection, prior MoCaE stack on old experts)
- Homogeneous deep ensembles at larger K
- Full gated/learned MoE architecture search
- Paper writing pass (Algorithm box, MoCaE naming, qualitative interpretation polish)
- Compute/Pareto optimization for deployment

---

## 8. Phased Execution

### Phase 0 — Design freeze *(this document)*
- [x] Research question, approach, method matrix, testing plan locked

### Phase 1 — Data *(next)*
- [ ] Install `dataset-tools`
- [ ] Download xView, HIT, UAV, HRSC2016-MS, palm trees (+ candidates)
- [ ] Pin exact Ninja names, sizes, licenses in `data/DOWNLOAD_LOG.md`
- [ ] Unify format + splits; smoke-check labels

### Phase 2 — Models
- [ ] YOLO26 training recipe
- [ ] RF-DETR training recipe
- [ ] Spec + train custom network
- [ ] Symmetric control variants

### Phase 3 — Methods
- [ ] NMS / Soft-NMS / WBF
- [ ] Simple MoE, Calibrated MoE
- [ ] Gated MoE, Learned MoE
- [ ] Calibration bake-off utilities + ECE

### Phase 4 — Controlled bias experiments
- [ ] L-runs, O-runs, C-runs on primary dataset
- [ ] Expand to remaining datasets
- [ ] Mix-runs (optional)

### Phase 5 — Analysis & write-up
- [ ] Tables/figures per bias
- [ ] Cross-dataset summary
- [ ] Map results back to reviewer checklist

---

## 9. Immediate Next Action

**Start Phase 1:** download the Dataset Ninja suite (xView, HIT, UAV, HRSC2016-MS, palm trees, and related RS sets), record exact names/paths, then standardize formats before any training.
