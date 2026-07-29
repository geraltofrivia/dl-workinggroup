# Lesson 5.2.1: BCE vs MSE (Why Cross-Entropy for Classification?)

- **Notebook file(s):** `notebooks/5.2.1 BCE vs MSE.ipynb` (single notebook, no complete twin; largely pre-filled and run-through)
- **Scaffolding model:** single notebook
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## Concepts Covered

- **Classification vs. regression, framed geometrically** — DEPTH: moderate. "Regression = *what is the curve*; classification = *which side of the curve are we on*." Made vivid with a single dataset: a sine curve is both the regression target (fit the curve) and the decision boundary (a "sine threshold" splitting a 2-D plane into classes). Presented via viz + markdown.
- **Mean Squared Error as (squared) Euclidean distance** — DEPTH: deep-from-first-principles. Full LaTeX derivation showing the Euclidean distance between `(xᵢ, ŷᵢ)` and `(xᵢ, yᵢ)` collapses to `√((ŷ−y)²)`, so squared distance *is* MSE (×2). Punchline: "MSE is really just how far away is my output from the ideal output." Reinforced with a plot drawing the dashed residual lines from prediction to target.
- **Information / self-information `I(E) = −log(p)`** — DEPTH: deep-from-first-principles. Built from a "how *surprised* are you?" analogy (missing-semicolon build fail = 0 bits; junior rewriting the repo in Python overnight = a lot of bits). Worked examples: coin `−log₂(½)=1 bit`, die `−log₂(⅙)≈2.58 bits`.
- **Entropy `H(X) = −Σ pⱼ log₂ pⱼ`** — DEPTH: moderate. "High entropy = high uncertainty." Bounds given concretely for binary: max 1 bit at 50/50, min 0 bits when all one class. Supported by embedded images from Eli Bendersky's cross-entropy/KL post.
- **Cross-entropy `H(P,Q) = −Σ P(x) log Q(x)`** — DEPTH: deep-from-first-principles. Framed as measuring *mismatch* between true distribution `P` and predicted `Q`: "bits needed to encode P's events using Q's optimal code" / "how surprised we'd be." Reduced to the binary form `−[y log ŷ + (1−y) log(1−ŷ)]` and, for `y=1`, to `−log(ŷ)`.
- **The core contrast — distance vs. probability** — DEPTH: deep-from-first-principles. "Regression: loss should represent *how far* → MSE measures distance. Classification: we want to *compare probabilities* → CE measures gain in information." This is the thesis of the whole lesson.
- **Loss-curve shape comparison** — DEPTH: moderate. For a target of 1, MSE and BCE are plotted against `ŷ ∈ [0,1]` with annotated points (0.01, 0.1, 0.5, 0.9). Observation: BCE → ∞ as `ŷ → 0`; it penalizes confidently-wrong predictions far more steeply (0.1 vs 0.01 is ~2× worse under BCE, only ~20% worse under MSE).
- **Gradient behavior / vanishing gradients** — DEPTH: moderate (stated, not derived). MSE-with-sigmoid gradient carries a `ŷ(1−ŷ)` factor that vanishes near 0/1 → "learning happens slowest when you need it most." BCE-with-sigmoid gradient simplifies to `ŷ − y`, avoiding that. Ties back to the earlier observed symptom ("the model would go haywire / take foreverrr").
- **"Vibe math" summary** — DEPTH: surface. Plain-language wrap-up: MSE is a "bring everything together" loss with no notion that outputs are bounded in [0,1]; BCE emerges from information theory + likelihood and optimizes directly for the probability of the correct class.

## Narrative & Depth

This is an explicitly motivated digression: the opening markdown says Sebastian asked in the previous session "why cross entropy for classification?", and this notebook is the answer. The arc: (1) set up one dataset that serves as both a regression target (the sine curve) and a classification boundary (points above/below the sine), making the reg-vs-clf distinction physical. (2) Fit the curve with an MLP under MSE and *prove* that MSE is Euclidean distance — legitimizing MSE as the right tool for "how far off is the curve." (3) Switch to the classification framing and ask what we actually want there: not distance, but a comparison of probability distributions. (4) Build the machinery for that from scratch — information → entropy → cross-entropy — using surprise analogies before the formulas. (5) Reduce CE to the binary `−log ŷ` case and *empirically* compare the MSE and BCE loss curves for a fixed target, showing BCE's asymmetric, blows-up-when-confidently-wrong shape. (6) Explain the *training* consequence via gradients (the `ŷ(1−ŷ)` vanishing factor under MSE vs. the clean `ŷ−y` under BCE), connecting it back to bad training behavior the group had already witnessed. (7) Close with an intuitive "vibe math" summary.

Fundamental principle: **the loss encodes what you're optimizing for — MSE optimizes geometric distance (right for curve-fitting), BCE optimizes agreement between probability distributions (right for classification), and this choice shows up concretely in both the loss shape and the gradient.** It deliberately does NOT cover: multi-class/softmax cross-entropy (explicitly restricted to binary), KL divergence beyond the cited link, the full likelihood/MLE derivation of BCE (asserted, not proven), numerical stability / `BCEWithLogitsLoss` internals, or a rigorous gradient derivation (the gradient expressions are stated as facts).

## Scaffolding: provided vs. live-coded

This is effectively a **pre-filled demonstration/explainer notebook**, not a fill-in-the-blanks scaffold. There are no `...`/`# TODO` blanks; nearly everything is wrapped in `if True:` blocks and meant to be *run and discussed* rather than typed live.

**PRE-FILLED / already present:**
- A large cell-0 setup block: imports, `torch.manual_seed(42)`, and a full-featured `train()` helper with live `clear_output` visualization for both 1-feature (curve overlay) and 2-feature (3-D surface from three view angles) models, early stopping, and a loss curve. Also `visualize_dataset`, `viz_2d`, `scale_labels` helpers (and a commented-out piecewise-function dataset).
- Dataset construction: `sine_fn`, `sine_threshold`, `linspace_2d`, the combined regression+classification plot.
- The `MLP` class (now extended with an optional `final_act`, e.g. `torch.sigmoid`).
- The regression training call (`MLP(1,2)` under default MSE) and the classification training call (`MLP(2,3, final_act=sigmoid)` under `BCELoss`).
- The residual-line MSE illustration plot.
- The full MSE-vs-BCE loss-curve comparison plot with annotated points.
- All the theory markdown (information, entropy, cross-entropy, binary reduction, gradients, vibe-math), including two embedded attachment images and a source link.

**Live element:** the pedagogy is in *running* the two training calls live (watching the animated fit / decision surface) and in walking the math markdown — not in code completion. The only thing a presenter typically edits live is nudging hyperparameters (learning rate, `viz_every`, epochs).

## Libraries / APIs used

- **PyTorch:** `torch.nn.Module`, `torch.nn.Linear`, `torch.nn.Sequential`, `torch.nn.ReLU`, `torch.sigmoid`, `torch.nn.MSELoss`, `torch.nn.BCELoss`, `torch.optim.SGD`, `torch.no_grad`, `torch.manual_seed`, plus tensor ops (`linspace`, `rand`/`randn`, `meshgrid`, `column_stack`, `where`).
- **matplotlib.pyplot** (2-D scatter/line plots and 3-D `plot_surface` with multiple `view_init` angles), **numpy**.
- **IPython.display** (`clear_output`, `display`) for the live animated training visualization.

## Notable pedagogical choices

- Directly answers a participant's real question ("why cross-entropy?"), making the digression feel earned and audience-driven.
- One clever dataset does double duty — the *same* sine curve is the regression target and the classification boundary — so the reg-vs-clf distinction is shown, not just stated.
- Grounds MSE in something already intuitive (Euclidean distance) before contrasting it, rather than presenting it as an arbitrary formula.
- Teaches information theory through emotional "surprise" analogies (build-fail vs. overnight-rewrite) before any equations — classic zero-familiarity on-ramp.
- Argues from *two* independent angles — the loss-curve shape (behavior at the boundary) and the gradient (behavior during training) — and explicitly ties the gradient point back to misbehavior the group had personally seen ("go haywire", "take foreverrr").
- Keeps it honest about scope: restricts to binary cross-entropy and states the gradient results as facts rather than pretending to derive everything.
- Ends with a deliberately informal "vibe math" summary so the takeaway survives even if the derivations don't.
