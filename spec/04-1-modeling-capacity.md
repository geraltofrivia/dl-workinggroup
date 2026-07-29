# Lesson 4.1: Modeling Capacity

- **Notebook file(s):** `notebooks/4.1 Modeling Capacity.ipynb` (live-coding scaffold) + `notebooks/4.1 Modeling Capacity [COMPLETE].ipynb` (pre-filled)
- **Scaffolding model:** scaffold + complete pair
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## Concepts Covered

- **Model capacity vs. data complexity** — DEPTH: deep-from-first-principles. The explicit lesson goal (stated in the title markdown): run three experiments — (1) linear data + linear model, (2) quadratic data + linear model, (3) quadratic data + quadratic model — and *watch* the linear model fail on the curve while the quadratic model fits it. Taught by hands-on experiment + live visualization, not theory.
- **A model as a parameterized function `y = f(x; θ)`** — DEPTH: deep. The `LinClf` module literally implements `y = m*x + c` with `m` and `c` as `nn.Parameter`s. The quadratic `TwoOrderClf` implements `a1*x² + a2*x + a3`. Presented as code, tightly coupled to the algebra (`fx = lambda x: (m*x)+c`).
- **`nn.Parameter` and `nn.Module`** — DEPTH: moderate-to-deep. Params are declared by hand in `__init__`; `forward` defines "what to do with the params". No `nn.Linear` used — the point is to demystify what a layer *is*.
- **The training loop from scratch** — DEPTH: deep. Explicit five-step "recipe": `opt.zero_grad()` → predict → compute loss → `loss.backward()` → `opt.step()`. Shown as code with a `report()` call printing parameters and gradients *before and after* each update.
- **Gradients as the update signal** — DEPTH: moderate. The `report()` helper prints each parameter's `.data` and `.grad` so students see gradients appear after `backward()` and the values move after `step()`.
- **Mini-batch sampling / SGD** — DEPTH: moderate. Each step samples a random batch of 20 from the 200-point dataset via `torch.randint`; optimizer is `torch.optim.SGD`.
- **Loss (MSE) and convergence** — DEPTH: moderate. `nn.MSELoss(reduction='sum')`; a convergence early-stop at `loss < 0.001`; final cell plots the loss curve both raw and in log space.
- **Interactive, step-by-step optimization** — DEPTH: presentation device. The loop pauses on `input('$: ')` each iteration and re-renders the model's current fit, so the class can literally step through gradient descent and watch the line/curve move toward the data.

## Narrative & Depth

Flow: define two ground-truth functions (`true_fn_lin = 3x+10`, `true_fn_quad = x²−x+3`), sample 200 noisy-ish points from each, and visualize. Build a linear model by hand. Build a quadratic model by hand. Set hyperparameters (2000 epochs, lr 0.01, MSE-sum loss). Then run the interactive training loop, swapping which dataset (`X`/`Y`) and which model (`m`) are plugged in to walk through the three goal experiments. Finish by plotting the loss curve.

Fundamental principle: **a model can only represent functions within its hypothesis class.** A `y=mx+c` model has no term that can bend, so on quadratic data it converges to the best-possible *line* and still has irreducible error; adding an `x²` term (raising capacity) lets it fit. Capacity is made viscerally obvious by watching the fit animate and the loss plateau (linear-on-quad) vs. drive to zero (quad-on-quad).

What it deliberately does NOT cover: `nn.Linear`/built-in layers (intentionally hand-rolled), non-linear activations, over-fitting / regularization / train-val generalization (there is no validation split here — it is purely about representational capacity, not generalization), classification (that is `4.2`), higher-dimensional inputs (everything is 1D `x`), and any principled hyperparameter tuning.

## Scaffolding: provided vs. live-coded

**PRE-FILLED / already present in BOTH notebooks:**
- Imports (matplotlib, numpy, `IPython.display`, seaborn, torch) and `torch.manual_seed(42)`.
- All viz/util helpers: `viz_1d`, `viz_pred`, `viz_all` (mesh-grid model curve + gold/pred scatter, supports incremental redraw on a shared axis), and `report()` (prints named params, values, grads).
- Data setup: `true_fn_lin`, `true_fn_quad`, 200 samples each, and the two initial `viz_1d` plots.
- The complete `LinClf` linear module (`m`, `c` params; `forward` returns `x*m + c`).
- Hyperparameters (epochs=2000, lr=0.01) and `lfn = nn.MSELoss(reduction='sum')`.
- The full interactive training loop cell (batch sampling, zero_grad/backward/step, `report` before/after, incremental `viz_all` redraw, `input()` pause, convergence check) — identical in both.
- The final loss-curve plotting cell (raw + log).

**What participants FILL IN live (scaffold only):**
- **Cell 6 — the quadratic model.** Scaffold has only the comment `# How about a quadratic model (a1*x*x + a2*x + a3)` and an empty body. This is the central live-coding task: define `TwoOrderClf` with three `nn.Parameter`s and the quadratic `forward`.
- **Cell 7 — model instantiation.** Scaffold: `m = LinClf()`. Live task is to also build the quadratic model.
- **Cell 10 — wiring the experiment.** Scaffold sets `X = X_lin; Y = Y_lin; opt = SGD(m.parameters())` — students change which dataset and which model are bound to run the three capacity experiments.
- The `...` placeholder inside the `with torch.no_grad():` block of the training loop (present in both — a spot to add per-step logging/inspection live).

**Notable differences between scaffold and COMPLETE:**
- COMPLETE cell 6 contains the finished `TwoOrderClf` (note: comment reads `a*x*x + b*x + c` but params are named `a1/a2/a3`, initialized with `torch.randn(1)`).
- COMPLETE cell 7 instantiates BOTH models: `m_lin = LinClf()` and `m_quad = TwoOrderClf()`.
- COMPLETE cell 10 additionally sets `m = m_quad` (i.e. it is left configured to run the quadratic-model experiment).
- COMPLETE has 14 cells vs. the scaffold's 13 — the extra is a trailing empty code cell (cell 13).
- Everything else (helpers, loop, loss plot) is byte-for-byte identical.

## Libraries / APIs used

- `torch`, `torch.nn` (`nn.Module`, `nn.Parameter`, `nn.MSELoss`), `torch.optim.SGD`, `torch.randint`, `torch.no_grad`, `torch.manual_seed`
- `matplotlib.pyplot`, `numpy` (`np.mgrid`), `seaborn`
- `IPython.display` (`display`, `clear_output`) for the animated in-place redraw

## Notable pedagogical choices

- **Hand-rolled models, no `nn.Linear`.** Parameters are declared explicitly so a zero-familiarity audience sees that a "model" is just a function with learnable numbers — the from-scratch ethos.
- **The `report()` before/after print.** Making gradients and parameter deltas visible on every step turns backprop from a black box into something observable.
- **Human-in-the-loop stepping (`input('$: ')`).** The training loop deliberately blocks after each iteration and re-plots, so the instructor can narrate gradient descent frame-by-frame during a live 1–1.5h session (type `q`/`quit` to stop). This is a teaching device, not something you would ever ship.
- **Three-experiment matrix as the spine.** The goals markdown (linear/linear, quad/linear, quad/quad) frames the whole session; the only "code you change" is the two binding lines in cell 10, keeping cognitive load low.
- **Capacity, not generalization.** No validation set and `reduction='sum'` loss — the focus is purely on whether the model *can* represent the target function, sidestepping over/under-fitting nuance appropriate for a later lesson.
- **Tiny, synthetic, 1D data.** Ground-truth functions are known exactly, so "did it fit?" has an unambiguous visual answer.
