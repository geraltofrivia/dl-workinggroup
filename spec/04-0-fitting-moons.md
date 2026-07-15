# Lesson 4.?: Fitting Moons

- **Notebook file(s):** `notebooks/4.? Fitting Moons.ipynb` (single live-coding scaffold; NO `[COMPLETE]` twin exists)
- **Scaffolding model:** single notebook
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

> **Ordering note (the "4.?" in the filename):** The instructor left the sequence number deliberately unresolved. Content-wise this notebook does NOT belong at the head of Chapter 4. Its concepts — 2D binary classification, `sigmoid` on logits, a multi-layer `nn.Sequential` net with `ReLU`, and decision-boundary visualization — are strictly *more advanced* than the sibling `4.1 Modeling Capacity`, which is 1D regression with a hand-rolled `y = mx + c`. This looks like a **motivational teaser / capacity demo** that pairs conceptually with `4.2 Classification` rather than preceding `4.1`. The single most defensible reading: it is the "hook" that motivates the whole capacity theme (why a straight line can't fit a curvy dataset, therefore we need depth/non-linearity), whose *mechanics* are then unpacked slowly in `4.1`. It is also unfinished/rough: several cells are one-line `# comment` stubs, the scaffold `Model` class in cell 9 is broken (references `self.b`, `self.c` that are never defined, and `forward` returns nothing), and there is no training loop actually written — training is only gestured at with comments. The instructor almost certainly intended to live-code the missing pieces (or this notebook was superseded by the more polished `4.1`/`4.2` split), which is why it never got a clean number or a `[COMPLETE]` version.

## Concepts Covered

- **Non-linearly-separable data (the two-moons dataset)** — DEPTH: moderate. Presented via `sklearn.datasets.make_moons` and a scatter plot. The interleaving-crescents shape is the whole pedagogical point: it is the canonical "a line cannot cut this cleanly" dataset.
- **Linear model as a decision boundary** — DEPTH: moderate. A hand-written `LinearRegression(nn.Module)` (weights + optional bias, `x @ w + b`) is fed through `show_separation`, which renders the probability field. Intuition: a single linear layer can only draw a straight boundary.
- **The core question: can a straight line ever fit this?** — DEPTH: surface but pointed. Left as an explicit `# Q:` comment (cell 8) — a Socratic prompt to be discussed live, not answered in code.
- **Depth / non-linearity via a multi-layer MLP** — DEPTH: moderate. A `nn.Sequential` of `Linear(2,30) → ReLU → Linear(30,20) → ReLU → Linear(20,1)` is contrasted against the linear model on the same visualization. Intuition: stacking linear layers with `ReLU` between them lets the boundary bend.
- **Sigmoid → probability** — DEPTH: surface. Appears only inside the `show_separation` helper (`torch.sigmoid(model(batch))`), interpreting the raw logit output as `P(y=1)`. Not dwelt on.
- **Train/val split & DataLoader plumbing** — DEPTH: surface. `train_test_split`, `TensorDataset`, `DataLoader(batch_size=128)` are set up but never consumed by a training loop in this notebook.

## Narrative & Depth

The intended flow: (1) generate a visibly non-linear dataset (moons); (2) look at it; (3) build the simplest possible model — a linear one — and *see* (via the decision-boundary plot) that its boundary is a straight line; (4) ask the pointed question "can a linear line ever fit it?"; (5) build a "deep" model and see the boundary curve to follow the crescents; (6) train both and compare.

The fundamental principle being built: **model capacity must match data complexity** — a linear model is structurally incapable of separating non-linearly-separable classes, and non-linearity (hidden layers + `ReLU`) is what buys the extra expressiveness. This is the same thesis as `4.1`, expressed in 2D classification instead of 1D regression.

What it deliberately does NOT cover: the actual training loop (only stubbed with comments `# Lets train this model`, `# Train both models. See what works :)`), any explanation of `ReLU`'s mechanics, loss functions for classification (`BCE`), evaluation/accuracy metrics, or *why* the split/DataLoader are needed. Backprop and optimization are assumed from earlier chapters (Chapters 2–3 "Naked Tensors" / "Introducing some structure").

## Scaffolding: provided vs. live-coded

**PRE-FILLED / already present:**
- All imports (numpy, seaborn, matplotlib, sklearn datasets + split, torch/nn/functional, TensorDataset/DataLoader). One commented-out `catalyst` import.
- `make_moons(n_samples=10000, noise=0.07)` data generation and the initial scatter plot.
- `show_separation(model, ...)` — a fully-written decision-boundary plotting helper (builds a mesh grid, runs the model, sigmoids to probabilities, contour-plots with a colorbar, overlays data points).
- Train/val split and DataLoader setup.
- The hand-rolled `LinearRegression(nn.Module)` class (complete).
- The "deep" `nn.Sequential` model (complete) and its `show_separation` call.

**Live-coded / to be FILLED IN:**
- The training loop(s) — only present as comment stubs (`# Lets train this model`, `# Train both models`).
- The `Model` class in cell 9 — a **broken scaffold**: `self.a = nn.Linear(2, 30)` is given, but `self.b = ...` (blank), and `forward` calls `self.b(x)` and `self.c(x)` which are never defined, and returns nothing. Intended to be filled in live as a manual re-derivation of the Sequential model (declaring the intermediate layers + activations).
- The discussion answer to `# Q: Can a linear line ever fit it?`.

**Notable differences between scaffold and COMPLETE:** none — there is no `[COMPLETE]` version of this notebook.

## Libraries / APIs used

- `sklearn.datasets.make_moons` (also imports `make_circles`, `make_blobs`, unused), `sklearn.model_selection.train_test_split`
- `torch`, `torch.nn` (`nn.Module`, `nn.Parameter`, `nn.Linear`, `nn.ReLU`, `nn.Sequential`), `torch.nn.functional`, `torch.sigmoid`, `torch.no_grad`
- `torch.utils.data.TensorDataset`, `DataLoader`
- `numpy` (mesh grid via `np.mgrid`, `np.c_`), `matplotlib.pyplot`, `seaborn`

## Notable pedagogical choices

- **Show-don't-tell via decision boundaries.** The `show_separation` helper is the star: capacity is taught by *looking* at the shape of the boundary (straight line vs. curved), not by discussing VC dimension or theory.
- **Socratic stub comments.** Key beats are left as bare `#` comments (`# Q: Can a linear line ever fit it?`, `# Lets train this model`) — prompts for live discussion rather than pre-written answers.
- **Hand-rolled linear layer before `nn.Sequential`.** Reinforces the "from scratch" ethos: students see `x @ weights + bias` as a `Module` before being shown the batteries-included `nn.Sequential` equivalent.
- **Deliberately rough / unfinished.** The broken `Model` scaffold and the absent training loop are consistent with a depth-first, build-it-live style — but combined with the unresolved `4.?` number, they strongly suggest this notebook was a draft/teaser that the cleaner `4.1` + `4.2` notebooks later replaced.
