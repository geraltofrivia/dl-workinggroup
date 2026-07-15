# Lesson 4.2: Classification

- **Notebook file(s):** `notebooks/4.2 Classification.ipynb` (single notebook — no `[COMPLETE]` twin exists)
- **Scaffolding model:** single notebook (sparse scaffold; nearly all modelling done live)
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## Concepts Covered
- **Binary classification on 2D data** — DEPTH: moderate. Presented via a synthetic dataset and a stated three-experiment agenda (linear data / linear model; quadratic data / linear model; quadratic data / quadratic model). Only the linear dataset is set up in the file; the classification models themselves are coded live.
- **Decision boundary as a diagnostic** — DEPTH: deep-from-first-principles (via viz). The pre-built `viz_all` helper sweeps a dense grid over input space, runs the model, and paints a probability contour (`P(y=1)`) with the data overlaid. This is the central pedagogical device: "does the model's boundary match the shape of the data?"
- **Model capacity vs. data complexity** — DEPTH: moderate. The three-experiment structure is the whole point: a linear model fits linear data but fails on quadratic data, motivating a higher-capacity (quadratic / feature-expanded) model. This is demonstrated empirically rather than derived.
- **Generating labelled data from a known "true function"** — DEPTH: surface/moderate, code. `true_fn_lin` defines a ground-truth linear separator (`2.3*x[0] + 0.1 > x[1]`) so labels are known by construction.
- **Probabilistic output / `P(y=1)`** — DEPTH: surface here. Implied by the colorbar and contour semantics in the viz helper; the sigmoid/logistic mechanics are expected to be introduced live.

## Narrative & Depth
The lesson is framed by an explicit goals list of three escalating experiments. The narrative arc is capacity matching: start where a linear model succeeds (linear data), then break it (curved data with a straight boundary), then fix it (a model that can bend its boundary). The fundamental principle being built is intuitive: **a model can only draw boundaries its functional form allows** — visualised directly as the decision boundary morphing (or failing to morph) around the data.

Because there is no complete twin and only the first dataset plus visual helpers are present, the file is a launchpad: the instructor derives the models, the loss, and the training loop at the whiteboard/live. It deliberately does NOT cover: multi-class classification, real-world/noisy data, train/test splitting or generalization metrics, the mathematical derivation of the logistic/sigmoid link, or `nn`-module abstractions in the file itself (those exist in the surrounding structure sessions). It connects backward to `4.1 Modeling Capacity` and forward to the `5.x Classification` sessions.

## Scaffolding: provided vs. live-coded
- **PRE-FILLED / already present:**
  - Imports: `matplotlib.pyplot`, `numpy`, `IPython.display`, `seaborn`, `torch`, `torch.nn`.
  - `torch.manual_seed(42)`.
  - Three plotting helpers — `viz_1d`, `viz_2d`, and the elaborate `viz_all` (grid sweep + `contourf` probability surface + scatter overlay + colorbar) — plus a `report(model)` helper that prints named parameters and their gradients.
  - Linear dataset: `true_fn_lin`, 200 samples `X_lin` in [-1, 1]^2, labels `Y_lin`, and a `viz_2d` call to display it.
- **What participants FILL IN live (no `...`/TODO markers — the file simply ends):** everything about modelling. The final cell (index 6) is empty. Live work covers: defining the linear model, a quadratic dataset + quadratic/feature-expanded model, the loss, the optimization/training loop, and repeated calls to `viz_all` to watch each experiment's decision boundary. The `report` helper signals that live gradient inspection is expected.
- **Notable differences between scaffold and COMPLETE:** none — there is no complete version. The three experiments in the goals list are never written down in the file; they are realised entirely during the session.

## Libraries / APIs used
- `torch` (tensors, `torch.randint`, `manual_seed`, `no_grad`), `torch.nn` (imported; used live).
- `numpy` (`np.mgrid`, `np.c_`, `ravel`) for the decision-boundary grid.
- `matplotlib.pyplot` and `seaborn` (`contourf`, `scatter`, colorbar) for visualization.
- `IPython.display` (imported; supports live/animated redraw).

## Notable pedagogical choices
- Heavy investment in one reusable visualization (`viz_all`) so every experiment is judged by the same lens — the decision boundary — rather than by a loss number.
- Data is synthesised from an explicit `true_fn`, so "ground truth" is literally visible and the model's job is unambiguous.
- The near-empty file is intentional: this is a live-coding scaffold where the instructor builds the three models from scratch, keeping the depth-first, from-principles style. The provided `report` helper reinforces the recurring theme of watching parameters and gradients move.
- Escalating three-experiment design makes model-capacity failure a felt experience (watch the straight line fail on curved data) before the fix is introduced.
