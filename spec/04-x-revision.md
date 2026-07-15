# Lesson 4.x: Revision (Warm Restart)

- **Notebook file(s):** `notebooks/4.x Revision.ipynb` (scaffold, blanks/`...` to fill live) and `notebooks/4.x Revision [COMPLETE].ipynb` (pre-filled, run-only). Both are 45 cells and structurally identical.
- **Scaffolding model:** scaffold + complete pair
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## What it revises
A "Warm Restart" consolidation lesson that recaps the first block of the course — the material from `1. Play with tensors`, `2. Naked Tensors - Linear Clf`, and `3. Introducing some structure`. The explicit topics list is: tensors & gradients, loss, backpropagation, the training loop, fitting a curve, multi-dimensional inputs, and limitations of polynomial models. It re-walks the whole "manual → structured" journey in one sitting before the course moves to real-world problems and data handling ("Next time...").

## Concepts Covered
- **Tensors & dtypes** — DEPTH: moderate, code. Create scalar tensors, observe `int` vs `float` dtype, mixed-dtype arithmetic (`float32` vs `float64`).
- **Tensor arithmetic (many spellings)** — DEPTH: surface, code. `a+b`, `torch.add`, `a.__add__(b)`, `/`, `//`, `*`, `-` — showing operators, functions, and dunder methods are the same thing.
- **Automatic differentiation** — DEPTH: deep-from-first-principles. `requires_grad` flag, `grad_fn` appearing on results of ops over leaf tensors that track gradients.
- **The computation graph (visualized)** — DEPTH: moderate, viz. `torchviz.make_dot` renders the graph for `d = a/b + b`; an instructor note toggles `requires_grad` on `b` to show the graph grow/shrink. Explicitly "see it on my screen" — install optional.
- **Multidimensionality** — DEPTH: moderate, code. 1D vs 2D tensor construction, `.shape`, elementwise `*` vs matrix product `@` / `torch.mm` / `torch.matmul` — with the deliberate "was this the expected result?" beat contrasting `a*b` and `a@b`.
- **Broadcasting** — DEPTH: surface (explicitly deferred). Named as "a rabbit hole we are not going to go into for now."
- **Linear model from naked tensors** — DEPTH: deep-from-first-principles, math+code. Single data point `(x=10, y=2)`; parameters `m, c` as `requires_grad` tensors; `f(x) = m*x + c`; MSE `L = (1/2n)Σ(ŷ−y)²`; gradient-descent update rule shown as LaTeX.
- **Manual backprop & training loop** — DEPTH: deep-from-first-principles. `loss.backward()`, inspecting `.grad`, manual `grad.zero_()`, `torch.no_grad()` parameter update via `copy_`, step-by-step `input()`-gated loop over 2000 iters with live printouts of params/grads before and after each update.
- **Adding structure** — DEPTH: moderate, code. Progression: wrap params in a plain class → subclass `torch.nn.Module` with `nn.Parameter` → replace with `nn.Linear(1,1)`; swap hand-written MSE for `torch.nn.MSELoss`; introduce `torch.optim.SGD`, `param_groups`, and `opt.step()` emulating a single batch.
- **Limitations of polynomial/linear models** — DEPTH: surface (listed as a topic, sets up later capacity lessons).

## Narrative & Depth
The flow deliberately retraces the course's spine: primitives (tensors, dtypes, ops) → autodiff (the "magic" that makes learning possible, made concrete via the graph viz) → dimensionality (so inputs can be more than scalars) → a full learning system built by hand (params, model, loss, gradient, update loop) → then the same system rebuilt with PyTorch's abstractions (`nn.Module`, `nn.MSELoss`, `optim.SGD`). The fundamental principle reinforced throughout: **learning = repeatedly nudging parameters against the gradient of a loss**, and the framework's classes are just ergonomic wrappers over the manual loop the class just wrote by hand. It deliberately does NOT cover broadcasting, real datasets / data loading, or non-linear capacity (flagged as "next time").

## Scaffolding: provided vs. live-coded
- **PRE-FILLED in both:** imports (`torch`, `numpy`); the `torchviz.make_dot` graph-rendering cells (10, 29, 33); all markdown headings and LaTeX (data space, model `f`, MSE, GD update); the fully written 2000-iteration manual training loop body's *scaffolding* including bookkeeping lists, print statements, `input()` gate, and convergence check.
- **What participants FILL IN live in the scaffold (`...` / blanks):**
  - Cell 3: create the `3.0` tensor and check dtype.
  - Cell 4: tensor arithmetic examples.
  - Cell 6–8: `requires_grad` setup, revisit ops, capture result and inspect `grad_fn`.
  - Cell 12–15: build 1D/2D tensors, elementwise vs matrix multiply, alternative matmul names.
  - Cell 20: the single data instance `x, y`.
  - Cell 22–23: define parameters `m, c` and the `linear` model body.
  - Cell 25: MSE body.
  - Cell 27: learning rate value.
  - Cell 28, 30–32: forward pass, loss, `backward()`, gradient inspection.
  - Cell 35: inside the loop — `y_pred`, `loss`, `grad.zero_()`, and the `updated_m/updated_c` update expressions.
  - Cell 37–41: the `LinearRegressor` class (plain → `nn.Module` with `nn.Parameter` → `nn.Linear`), `nn.MSELoss`, `optim.SGD`, and the single-batch step demo.
  - Cell 43: the "nice" training loop — left as an empty sketch in BOTH files (a live exercise, not completed even in COMPLETE).
- **Notable scaffold vs COMPLETE differences:** COMPLETE fills every `...` with concrete answers (e.g. `m = torch.randn(1, requires_grad=True)`, `lr = 0.01`, `mse = torch.nn.MSELoss()`, `opt = torch.optim.SGD(model.parameters())`). Two content notes: cell 3's answer uses `torch.tensor(3)` (int) to make the dtype point via a deliberate "gotcha"; the `nn.Module` `forward` in cell 37 references a stray global `c` (a small bug left in). Cell 43 (structured training loop) is empty in both — the intended in-session capstone.

## Libraries / APIs used
- `torch`: `tensor`, dtypes (`float64`), `randn`, `requires_grad`, `grad`/`grad_fn`, `backward`, `no_grad`, `copy_`, `grad.zero_`, `@`/`mm`/`matmul`.
- `torch.nn`: `Module`, `Parameter`, `Linear`, `MSELoss`.
- `torch.optim`: `SGD`, `param_groups`, `step`.
- `numpy` (array construction); `torchviz.make_dot` + graphviz (optional, instructor-screen only).

## Notable pedagogical choices
- Framed as a "Warm Restart" — a single-session sweep re-deriving everything from tensors to a structured trainer, ideal for a weekly cadence with zero homework where reinforcement matters.
- The `input()`-gated manual loop turns training into a hand-cranked, step-by-step demo so participants watch each gradient and parameter update land — the from-first-principles core.
- Self-aware asides ("You will never do this again in your life" about manual `grad.zero_()`) explicitly contrast the painful manual version with the framework version taught minutes later, motivating the abstractions.
- Broadcasting is named and explicitly deferred rather than silently skipped — honest scope control for a zero-familiarity audience.
- Graph visualization (`make_dot`) is optional-install and instructor-screen only, keeping setup friction off participants while still showing the computation graph.
- The empty structured training loop (cell 43) in both files is a deliberate open-ended live exercise / cliffhanger into the next block.
