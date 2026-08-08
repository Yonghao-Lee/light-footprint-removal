# Results — Measuring and Removing an Object's Light Footprint

All numbers are PSNR (dB) against **exact clean plates** (the same scene re-rendered
without the object, same cameras), evaluated on the 11 held-out novel views
(every 8th of 81 frames). "Footprint" restricts the metric to ground-truth
footprint pixels (shadow + mirror reflection + floor reflection + colour bleed,
obtained by differencing the with/without renders outside the object silhouette).
All removal rows are measured **after re-fitting a fresh 3DGS on the edited
frames** — i.e. they score what survives the lift back to 3D, not the 2D edit.

## Main table

| Method                            | PSNR full | PSNR footprint |
|-----------------------------------|-----------|----------------|
| Reconstruction ceiling            | 45.7      | —              |
| M1: Gaussian deletion             | 23.1      | 14.2           |
| M3: VACE, object mask             | 21.6      | 14.2           |
| M3: VACE, oracle mask             | 18.2      | 13.1           |
| M3: ROSE, object mask             | 23.2      | 15.0           |
| M3: ROSE, oracle mask             | 31.1      | 28.4           |
| **M3: ROSE, SAM2 masks (4 clicks)** | **31.3**  | **28.4**       |

## One line per experiment

- **Phase 2 — reconstruction ceiling** (`checkpoints/phase2_with/eval_with.json`):
  splatfacto, 15k iters, random init, 45.69 dB / 0.992 SSIM / 0.021 LPIPS on
  held-out views — reconstruction is essentially perfect, so every later
  degradation is attributable to removal, not to 3DGS fitting.
- **Phase 3 — M1 deletion** (`checkpoints/phase3_m1/m1_baseline.json`): mask-lifted
  selection (2,888 Gaussians; validated against ground-truth geometry at 100%
  precision / 90% recall, plus 318 interior blobs found only geometrically)
  deletes the sphere but leaves its mirror image, shadow and floor reflection
  intact — 23.1 / 14.2 dB quantifies the problem the project is about.
- **Phase 4 — VACE, object mask**: the generalist prior treated the masked region
  as "put an object here" and **rebuilt the sphere from its own footprint**
  (gray sphere, `figures/fig4_vace_substitution.png` left) — direct visual
  evidence the footprint encodes the object; 21.6 / 14.2 dB, no better than
  deletion in the footprint.
- **Phase 4 — VACE, oracle mask**: with the full footprint masked, VACE
  substituted a coherent cyan sphere *with a matching cyan mirror reflection*
  (fig4 right) — the prior understands object–reflection coupling, but
  "generate emptiness" is not an operation it learned; 18.2 / 13.1 dB, worse
  than deletion full-frame. A stronger emptiness prompt (50 steps, gs 7.5)
  still substituted (orange sphere) and was not refit.
- **Phase 4b — ROSE, object mask**: the removal-specialised prior deletes cleanly
  inside the mask but by design touches nothing outside it, so the footprint
  survives (fig5 left); 23.2 / 15.0 dB — masking, not the prior, is the
  bottleneck at this operating point.
- **Phase 4b — ROSE, oracle mask**: object and footprint removed together
  (fig5 right); 31.1 / 28.4 dB — **+14.2 dB in the footprint over deletion**,
  the headline positive result (upper bound: mask uses GT differencing).
- **Phase 5 — ROSE, SAM2 masks**: 4 clicks on one frame (sphere, mirror image,
  shadow, floor reflection; fig6), propagated over all 81 frames by SAM 2.1
  (fig7), union + 15px dilation: 31.3 / 28.4 dB — **the practical protocol
  ties the oracle**. Wall colour-bleed was too faint to click yet did not
  measurably hurt.
- **Multi-view consistency (extension, resolved by measurement)**: ROSE edits all
  81 frames independently, yet the refit reaches 31.3 dB on held-out views —
  the 3D representation absorbs the residual disagreement at this operating
  point, so iterative dataset update / warped noise were not needed.

## Provenance

- Numbers for phase 2 and M1 are persisted in the JSONs above; VACE/ROSE rows
  were computed in the phase-4 notebook's scoring cell (per-view PSNR printed
  in-session). The ROSE+SAM2 refit checkpoint is **not yet persisted** — re-run
  the refit+score cell on `checkpoints/phase4_rose/edited_rose_sam2mask/` to
  regenerate it (~30–40 min Colab).
- Figures: `figures/` (fig2, render-vs-GT, still needs a 10-min render from the
  phase-2 checkpoint in Colab). Videos: `videos/` (before/after orbits,
  side-by-side).
