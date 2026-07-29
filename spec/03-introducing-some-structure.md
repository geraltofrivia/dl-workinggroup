# Lesson 3: Introducing Some Structure

- **Notebook file(s):**
  - `notebooks/3. Introducing some structure.ipynb` — live-coding scaffold (blanks marked `...`, empty class bodies, empty TODO regions in the training loop).
  - `notebooks/3. Introducing some structure [COMPLETE].ipynb` — pre-filled version with every blank implemented. Cell structure is identical to the scaffold; only the blanks differ.
- **Scaffolding model:** scaffold + complete pair
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## Concepts Covered

- **Generalizing from one point to a dataset** — surface/moderate. Session 2 fit a line to essentially a single relationship; here the same idea is applied to a full generated dataset of 200 `(x, y)` points. Presented via code (data generation) and a scatter viz.
- **`torch.nn.Module` as the standard way to define models** — deep-from-first-principles (this is the core of the lesson). Introduced by explicit contrast: first "how we did it in the past" (bare tensors `m`, `c` with `requires_grad=True` and a lambda), then "how we should do it in the future" (a `LinearRegressor(torch.nn.Module)` subclass). Presented as code plus a bulleted markdown rationale.
- **`torch.nn.Linear` layer** — moderate. Used as the internal building block of the model (`nn.Linear(1, 1)`). The lesson also shows you can bypass layers entirely and register raw `nn.Parameter`s, to demystify what a "layer" actually is.
- **Named parameters / parameter introspection** — moderate. `model.named_parameters()` is used to reveal that the module automatically tracks its weight and bias tensors — motivating why the class approach beats loose variables.
- **The `__call__` → `forward` convention and input broadcasting/shape rules** — moderate, via an exploratory quiz. A cell throws many differently-shaped inputs at the model (`torch.tensor(4)`, `randn(4,)`, `randn(10,1)`, a Python int, a NumPy array, high-rank tensors) and asks which shapes work — building intuition for what shapes a `Linear` layer accepts.
- **Optimizers (`torch.optim.SGD`)** — deep-from-first-principles. Replaces the manual `param = param - lr * grad` update from earlier sessions with an optimizer object. The manual update rule is kept as an inline comment (`m = m - lr*(dL/dm)`) to anchor the new abstraction to the old math. `opt.param_groups` is inspected to show what the optimizer holds.
- **The canonical PyTorch training loop** — deep-from-first-principles. The five-step cycle: `opt.zero_grad()` → forward (`m(_X)`) → loss (`lfn`) → `loss.backward()` → `opt.step()`. Presented as code with heavy per-iteration print statements exposing weights and gradients before/after each update.
- **Mini-batch sampling (SGD in practice)** — moderate. Each epoch samples a random batch of 20 indices from the 200-point dataset rather than using the full set — a first, low-key exposure to stochastic/mini-batch gradient descent.
- **Loss curves** — surface. At the end, losses are plotted raw and on a log scale to show convergence.

## Narrative & Depth

The lesson's stated goals are: (1) generalize the previous codebase to many data points, (2) introduce `torch.nn.Module`, (3) introduce optimizers, (4) train a linear classifier [regressor], (5) train a FFNN.

The flow is a deliberate "refactor the thing you already understand" arc. Sessions 1–2 built gradient descent by hand on bare tensors. Session 3 keeps the *exact same problem* (fit `y = 3x + 10`) and reframes it in idiomatic PyTorch, so every new abstraction (`Module`, `Linear`, `optim`) is introduced as a replacement for something the audience already wrote manually. This is the fundamental principle driven home: **the framework isn't magic — `nn.Module` is a container for parameters, an optimizer is just the update rule you wrote by hand, and the training loop is the same four steps you already did.**

The "past vs. future" pairing is explicit in the cells: `m = tensor(..., requires_grad=True)` / lambda, then the `Module` subclass. The follow-up cell "Can we implement it without using layers?" re-implements the same model using raw `nn.Parameter`s instead of `nn.Linear`, closing the loop and proving a layer is nothing more than managed parameters. A markdown cell ("Why model class?") lists the practical payoffs: saving/loading, train/eval mode, easy parameter access, structure for complex models, and "that is the PyTorch way."

Training is framed as needing exactly three things: the model, the loss function, and a way to update parameters. The loop itself is interactive — it pauses on `input()` each iteration, re-drawing the fitted line over the sampled batch via `viz_all` and clearing output, so the class can watch the line rotate/shift into place step by step (and type `q` to stop).

Deliberately NOT covered: despite goal #5 ("Train a FFNN"), the notebook contains **no** feed-forward neural network, no hidden layers, and no non-linear activation — it stops at linear regression. The FFNN is an aspirational/next-session goal, not delivered here. Also omitted: proper Dataset/DataLoader abstractions (batching is done with manual `randint` indexing), train/test split, evaluation metrics, and any classification (the "classifier" in the goals is really a regressor).

## Scaffolding: provided vs. live-coded

**PRE-FILLED / already present in both scaffold and complete:**
- All imports (`matplotlib`, `numpy`, `IPython.display`, `seaborn`, `torch`) and `torch.manual_seed(42)`.
- The entire visualization helper block (`viz_1d`, `viz_pred`, `viz_all`), wrapped in `if True:` and flagged "Ignore these functions."
- Data-generation scaffolding: the `true_fn` signature/docstring, `n_samples = 200`, the `X` sampling expression, the `Y` list comprehension and tensor conversion, and the `viz_1d(X, Y)` call.
- Markdown section headers and the "Why model class?" and "Let's train it now" explanatory cells.
- `lfn = torch.nn.MSELoss(reduction='sum')` (loss fn given, as in prior sessions).
- `epochs = 2000`, `opt.param_groups` inspection cell, the final loss-plotting cell.
- The training loop's skeleton: figure setup, `opt.zero_grad()`, the batch-sampling code (`Xrange`, `_X`, `_Y`), the elaborate before/after `print` statements, the `viz_all` redraw, the `input()`/quit handling, and the `loss < 0.001` early-stop.

**Live-coded (the `...` / empty blanks participants fill in):**
- The body of `true_fn` → `return float(3*x + 10)`.
- The "past" solution: `m`, `c` as `requires_grad=True` tensors and the `fx` lambda.
- The `LinearRegressor(torch.nn.Module)` class: `__init__` with `super().__init__()` and `self.linear_layer = torch.nn.Linear(1, 1)`, plus `forward` returning `self.linear_layer(inputs)`.
- The `named_parameters()` inspection call.
- The layer-free re-implementation using `nn.Parameter(torch.tensor(0.5))` / `nn.Parameter(torch.tensor(0.1))`.
- The optimizer: `opt = torch.optim.SGD(m.parameters(), lr=0.01)`.
- The four missing lines inside the training loop: `Y_pred = m(_X)`, `loss = lfn(Y_pred, _Y)`, `loss.backward()`, `opt.step()`.

**Notable differences between scaffold and COMPLETE:** none structural — same 24 cells in the same order. The COMPLETE notebook simply has all of the above blanks filled and carries saved outputs. One tiny wording drift left in both: the model-parameters comment reads "See model parameters (named params)" in the scaffold vs. "See model parameters" in COMPLETE. The two debug `print` labels are both essentially "before update" (a copy-paste artifact where the second should read "after update"), present in both files.

## Libraries / APIs used

- **PyTorch:** `torch.nn.Module` (subclassing, `super().__init__()`, `forward`), `torch.nn.Linear`, `torch.nn.Parameter`, `model.named_parameters()`, `model.parameters()`, `torch.nn.MSELoss(reduction='sum')`, `torch.optim.SGD(params, lr=...)`, `opt.zero_grad()` / `opt.step()` / `opt.param_groups`, `loss.backward()`, `torch.no_grad()`, `torch.manual_seed`, `torch.randint`, `torch.tensor`, `requires_grad=True`.
- **matplotlib.pyplot** + **IPython.display** (`display.display`, `display.clear_output`) for the animated in-loop training visualization.
- **numpy** (`np.mgrid` inside the viz helper), **seaborn** (imported but not actively used).

## Notable pedagogical choices

- **Refactor-what-you-know framing.** The whole session re-solves the identical `3x + 10` fitting problem from the prior session so that every abstraction lands as a replacement for hand-written code, not a new idea. Cell comments literally say "How we did it in the past" vs. "How should we do it in the future."
- **Derive-then-verify / demystify the abstraction.** Immediately after introducing `nn.Linear`, the "Can we implement it without using layers?" cell rebuilds the model from raw `nn.Parameter`s — proving a layer is just tracked parameters and that `named_parameters()` finds them either way.
- **Shape-intuition quiz.** The cell that throws nine differently-shaped inputs (scalars, 1-D, 2-D, high-rank, NumPy, Python int) at the model as "Which of these should work?" is an interactive prediction exercise about broadcasting and what `Linear` accepts, rather than a lecture.
- **Anchoring new APIs to old math.** The optimizer step keeps the comment `# Update Parameters (m = m - lr*(dL/dm))` so `opt.step()` is understood as exactly the manual update from earlier sessions.
- **Human-in-the-loop animation.** The training loop blocks on `input()` each iteration and redraws the fitted line, letting the instructor narrate convergence step-by-step and stop on demand (`q`/`quit`/`exit`/`stop`). Verbose before/after prints of weight, bias, and their gradients make the mechanics visible.
- **Batch sampling introduced quietly.** Random 20-sample mini-batches per epoch expose stochastic gradient descent without naming or dwelling on it.
- **Honest scope gap.** Goal #5 ("Train a FFNN") is listed but intentionally not reached — the lesson caps at a fully-understood linear model, consistent with the depth-first, "few concepts deeply" philosophy, leaving the neural net for later.
