# Lesson 5.0: Composition (Why Depth Needs Non-Linearity)

- **Notebook file(s):** `notebooks/5.0 Composition.ipynb` (live-coding scaffold, model classes blanked with `...`) and `notebooks/5.0 Composition [COMPLETE].ipynb` (pre-filled, runnable end-to-end)
- **Scaffolding model:** scaffold + complete pair
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## Concepts Covered

- **Function composition (`f ∘ g = f(g(x))`)** — DEPTH: deep-from-first-principles. Presented with a trivial numeric analogy (`f(x)=x+1`, `g(x)=2x` → `2x+1`) in markdown, then full LaTeX derivation.
- **Composing linear functions stays linear** — DEPTH: deep-from-first-principles. Full algebraic proof in markdown: two affine layers `ω₁(ω₃x+ω₂)+ω₀` collapse to a single `ω₄x+ω₅`. This is the conceptual heart of the lesson.
- **Modeling capacity / why stacking linear layers is pointless** — DEPTH: moderate. Shown empirically first (a 2-layer and a 10-layer "big" linear model fail on a non-linear target) then explained by the composition proof.
- **Activation functions (sigmoid, ReLU, tanh)** — DEPTH: moderate. Plotted visually; sigmoid given a formula `1/(1+e⁻ˣ)`; derivative/nice-properties explicitly deferred ("we don't worry about it for now").
- **Non-linearity unlocks capacity** — DEPTH: deep-from-first-principles. LaTeX derivation of `f ∘ sigmoid ∘ g` showing it no longer collapses to a line ("that bend"), then demonstrated by fitting a piecewise function.
- **MLP (Multi-Layer Perceptron)** — DEPTH: surface. Given fully pre-built via `nn.Sequential` as the generalization / "further reading" payoff.
- **Model save/load (`torch.save`/`torch.load`)** — DEPTH: surface. Only in the COMPLETE notebook, as a practical appendix.
- **Recap of full DL stack** — DEPTH: surface. Conclusion markdown lists loss, gradient descent, forward/back-prop, non-linear+linear composition as "all the core concepts."

## Narrative & Depth

The lesson is a build-fail-explain-fix arc. (1) Fit a genuinely linear dataset (`y=10x+2`) with a one-parameter-per-weight `LinReg` — it works. (2) Introduce a harder piecewise-continuous target (`0` for `x<0`, `1-e⁻ˣ` otherwise). The linear model fails. (3) Attempt to fix it by brute force — more parameters (`BigLinReg`, 100 hidden units) and more layers (`LinRegCustom`, 5–10 layers). Still fails on the non-linear target. (4) Pose "Why?" and deliver the first-principles answer: composing linear functions always yields a linear function, so no amount of stacking adds capacity. (5) Introduce activation functions, prove (via the `f ∘ sigmoid ∘ g` expansion) that a non-linearity in the middle breaks the collapse, then fit the piecewise target successfully with `ActLinReg`. (6) Crank difficulty with a wavy sin/cos target to motivate needing more/tuning. (7) Generalize to a reusable `MLP` class as the takeaway architecture.

The fundamental principle: **depth is meaningless without non-linearity; the activation function is what actually buys modeling capacity.** The lesson deliberately does NOT cover: why sigmoid/ReLU specifically (derivatives, vanishing gradients, dead ReLUs), universal approximation theory, initialization, batching/dataloaders, train/test splits or generalization/overfitting, or the internals of autograd/optimizers (treated as already-known "fluff"). It assumes gradient descent, loss, and back-prop are prior knowledge (confirmed by the conclusion recap).

## Scaffolding: provided vs. live-coded

**PRE-FILLED / already present in BOTH notebooks:**
- Imports (`torch`, `matplotlib`, `time`, typing), `torch.manual_seed(42)`.
- Helper cell: `visualize_dataset`, `scale_labels`, and the full `train()` loop (optimizer, zero_grad/backward/step, live `clear_output` plotting, early stopping, loss curve). Explicitly framed as "stupid and simple" so class focus stays on principles.
- Data generation: constants (`n_samples`, `noise_scale`), linear dataset, `piecewise_fn` + dataset, `wavy_fn` + dataset, and activation-function plots (cell 16).
- All markdown/math cells: the composition proof, sigmoid derivation, conclusion, "stuff to try" prompts. Identical between the two files.
- The `MLP` class (cell 26) is fully written in BOTH notebooks (it is the "further reading" reveal, not something coded live).

**FILLED IN LIVE (blanked with `...` in the scaffold):** the model class bodies — this is where the pedagogy lives:
- `LinReg` (cell 7): `__init__`/`forward` + the `train(...)` call.
- `train()` calls for the linear model on the piecewise target (cell 10) and squared/piecewise on `BigLinReg` (cell 12) are commented-out stubs in the scaffold.
- `BigLinReg` (cell 11): two-layer body.
- `LinRegCustom` (cell 13): the variable-depth `nn.Sequential` builder.
- `ActLinReg` (cell 18): the input/hidden/output layers plus `act_fn` application in `forward` — the key "add a non-linearity" moment.
- The `train(ActLinReg(), ...)` call on the piecewise target (cell 19) is a stub in the scaffold.

**Notable differences between scaffold and COMPLETE:**
- Scaffold ends at cell 28 (last is a `train(MLP(...))` call using `Adagrad`); COMPLETE has 32 cells and adds a whole "Loading and saving a model" section (cells 28–31: `torch.save`/`torch.load` to `potato.pt`, then inference on new inputs). Note: that save/load section in COMPLETE references a `trained_model` variable that isn't defined in a prior cell (uses `model=` in train calls), so it would need adjustment to actually run.
- Tiny markdown wording drift: scaffold says composing linears "will **always** lead to another linear function"; COMPLETE drops "always."
- Final MLP `train` call differs: scaffold uses `MLP(act_fn=torch.nn.Sigmoid)` with `Adagrad`, lr 0.05; COMPLETE uses default `MLP()` (ReLU), plain SGD, lr 0.1, 20000 epochs.

## Libraries / APIs used

- `torch`: `nn.Module`, `nn.Linear`, `nn.Sequential`, `nn.MSELoss`, `torch.optim.SGD`/`Adagrad`, `torch.sigmoid`/`relu`/`tanh`, `nn.ReLU`/`nn.Sigmoid`, `linspace`, `where`, `exp`, `sin`/`cos`, `manual_seed`, `no_grad`, `save`/`load`.
- `matplotlib.pyplot` for scatter/line plots and live training visualization.
- `IPython.display.clear_output` for in-place animated training plots.
- `time.sleep` (optional pacing of the live viz).

## Notable pedagogical choices

- **Fail-first motivation:** learners watch brute-force scaling (more params, more layers) visibly fail before the theory is revealed, so the algebra answers a question they already feel.
- **Theory sandwiched between two empirical fits:** the composition proof sits exactly between "linear stack fails" and "add a sigmoid and it works," making the math load-bearing rather than decorative.
- **Helpers black-boxed on purpose:** `train()` is provided and dismissed as "fluff" so class time isn't re-spent on the training loop from earlier lessons.
- **Progressive difficulty of targets:** linear → piecewise-continuous → wavy sin/cos, each chosen to defeat the previous model class.
- **Instructor "note to self" cells left in** (e.g., "Try with sigmoid, then ReLU", "see with and without the noise") — these are live-demo cues, signaling activation-function swapping is meant to be done interactively.
- **Homework reframed as optional exploration:** "Further Reading or **home work**" plus a "stuff to try" list (vary activations, add layers, try Adam, re-run unchanged to see seed effects) — consistent with the no-homework format.
- **`MLP` handed over complete** as the synthesized takeaway, letting learners generalize the hand-built `ActLinReg` without live-coding boilerplate.
