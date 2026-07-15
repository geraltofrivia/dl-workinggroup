# Lesson 2: Naked Tensors — Linear Classifier (Gradient Descent from Scratch)

- **Notebook file(s):**
  - `notebooks/2. Naked Tensors - Linear Clf.ipynb` — **SCAFFOLD** (contains the `...` / `# TODO` blanks).
  - `notebooks/2. Naked Tensors - Linear Clf [COMPLETE].ipynb` — **COMPLETE / filled** version (all code present).
  - Naming caveat: despite the standard convention, the intro markdown *inside* the `[COMPLETE]` file is stale — it says "To fill by yourself... The `prefilled` version is available already as `2. Naked Tensors - Linear Clf.ipynb`" and asks to "**disable copilot**". That text is contradicted by the file's own contents (the `[COMPLETE]` file is fully filled; the non-suffixed file holds the blanks). Treat actual cell contents, not that header, as ground truth.
- **Scaffolding model:** scaffold + complete pair
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## Concepts Covered
- **Supervised regression setup / notation** — moderate, math notation. Markdown states $x_i \in \mathcal{X} \subset \mathcal{R}$, $y_i \in \mathcal{Y} \subset \mathcal{R}$, then a model as a map $f: \mathcal{X} \to \mathcal{Y}$ with $\hat{y}_i = f(x;(m,c)) = m\times x + c$. Math notation precedes every code implementation.
- **Linear model** — deep-from-first-principles, math-then-code. The "simplest possible model": two scalar parameters `m`, `c` created as `requires_grad=True` tensors, and a hand-written `fx(x) = m*x + c` (no `nn.Module`, no layers — hence "naked tensors").
- **MSE loss** — deep, math-then-code. $L_{mse} = \frac{1}{2n}\sum (\hat{y}_i - y_i)^2$. First implemented for a single sample as `0.5*(y_pred-y_true)**2`, later re-implemented as `mse_avg` with `torch.mean` for the multi-example case. The $\frac{1}{2}$ factor is deliberate so the derivative is clean.
- **Gradient descent update rule** — deep, math-then-code. $\theta_i^{T+1} = \theta_i^{T} - \gamma \frac{\partial L}{\partial \theta_i}$, then realized manually with a learning rate `lr` and in-place parameter updates.
- **Autodiff / `.backward()` / `.grad`** — deep. Single-sample forward → `loss.backward()` → inspect `m.grad`, `c.grad`. Participants are prompted to spot "what patterns do we see?" in the gradient values.
- **Manual gradient derivation via chain rule** — deep-from-first-principles, math then verify-in-code. A full markdown derivation introduces the error term $e = mx + c - y$, derives $\partial e/\partial m = x$, $\partial e/\partial c = 1$, $\partial L/\partial e = e$, and composes $\partial L/\partial m = ex$, $\partial L/\partial c = e$. Then a code cell (`(m*x+c-y)*x, (m*x+c-y)`) checks the hand-derived formulas against PyTorch's autograd numbers. Classic derive-then-verify.
- **Gradient accumulation** — moderate, code demonstration. Calling `.backward()` twice without zeroing shows `.grad` doubles; verified via `m.grad/2, c.grad/2`. Motivates the need to zero grads each step.
- **Zeroing gradients** — moderate, code. `m.grad.zero_()` / `c.grad.zero_()` (in-place), with a comment joking "You will never do this again in your life" (i.e. an optimizer will do it later).
- **`torch.no_grad()` for updates** — moderate, code. Parameter updates wrapped in `with torch.no_grad():` and applied with `m.copy_(...)` so the update itself isn't tracked by autograd.
- **Training loop** — deep, code. A full manual loop: forward → loss → zero grads → backward → no_grad update → bookkeeping, with an `input()` step-through (press Enter to continue, `q` to stop) so the class watches parameters evolve step by step.
- **Batch / multiple examples** — moderate, code. Second half swaps the single `(x,y)` for vectors and shows why a per-element loss must be *reduced* to a scalar (`mse_avg`) before `.backward()` works.
- **Reproducibility** — surface. `torch.manual_seed(42)`.
- **Visualization of training dynamics** — moderate, code. A `plot_trace` matplotlib helper plots traces of loss, `y_pred`, `m`, and `c` over iterations.

## Narrative & Depth
This is the conceptual heart of the early course: build and train a model with nothing but raw tensors — no `nn.Linear`, no optimizer — so the mechanics of gradient descent are fully exposed ("naked tensors"). The lesson is rigorously math-then-code: each markdown cell states the formula (data spaces, model, MSE, the update rule) and the following code cell implements exactly that. It starts with the most degenerate possible problem — a *single* data point `(x=10, y=2)` — so gradients are one number each and can be reasoned about by hand.

The intellectual centerpiece is the manual chain-rule derivation of $\partial L/\partial m$ and $\partial L/\partial c$, immediately followed by a cell that evaluates those closed-form expressions and compares them to `m.grad`/`c.grad`. This drives home the fundamental principle that `.backward()` is not magic — it computes the same derivatives you can do with pencil and paper. Two "gotcha" concepts are then taught experimentally rather than asserted: gradients *accumulate* in `.grad` (shown by backpropagating twice and seeing the value double), which motivates `zero_()`; and updates must happen under `torch.no_grad()`. The manual training loop uses an `input()` pause each iteration so the room can watch `m`, `c`, and the loss move — turning gradient descent into something observed frame by frame. Finally the problem is generalized from one sample to a vector of samples, exposing why a loss must reduce many errors to a single scalar (`torch.mean`) before backprop is defined.

It deliberately does **not** cover: `nn.Module`, `torch.optim` optimizers, autograd internals/graph mechanics, mini-batching/dataloaders, or any nonlinear model — all of that is intentionally postponed so this lesson can stay on the bare mechanics. The title says "classifier" but the task is actually linear *regression* (fitting a line); that's a loose naming, not a classification treatment.

## Scaffolding: provided vs. live-coded
**Pre-filled / already present in the SCAFFOLD (`... Linear Clf.ipynb`):**
- All markdown: the data-space notation, the model equation, the MSE equation, the update-rule equation, and the *entire* manual gradient-derivation markdown block (chain-rule walkthrough is given, not derived live).
- The `import torch` and `torch.manual_seed(42)` cells.
- Full function *signatures* with type hints for `linear(x)` and `mse(y_pred, y_true)` (bodies blanked).
- The full `plot_trace` matplotlib helper.
- For the multi-example section: the dataset is pre-filled (`x = [1..5]`, `y = [2,4,6,8,10]`), and the second training loop (cell 34) is fully written out (not blanked).

**Filled in LIVE (the `...` / TODO blanks in the scaffold):**
- Creating the single data instance `x=10`, `y=2` (comment hint provided).
- Defining parameters `m`, `c` as `requires_grad=True` tensors.
- The body of the `linear` model (`m*x + c`).
- The body of the `mse` loss (`0.5*(y_pred-y_true)**2`).
- The learning rate value `lr`.
- The single forward pass `ypred`, the loss computation, and the `.backward()` call.
- Implementing the two hand-derived gradient formulas (cell "Implement both these formulas").
- The gradient-accumulation demonstration.
- The full first training loop body: prediction, loss, grad-zeroing, `backward()`, and the `no_grad` update expressions (`updated_m`, `updated_c`).
- The multi-example `y_pred`, the `mse_avg` reduction body (`torch.mean`), and its `lr` / `backward` cells.

**Notable differences between scaffold and COMPLETE:**
- **Model function name mismatch:** the scaffold defines the model as `linear(x)`, but both training loops call `fx(x)`; the COMPLETE version defines it as `fx(x)`. (Anyone filling the scaffold must define `fx` or rename, or the loops break.)
- **Different multi-example data:** scaffold uses `y = [2,4,6,8,10]` (a pure `y=2x`, i.e. m=2,c=0); COMPLETE uses `y = [2,3,4,5,6]` (i.e. `y=x+1`, m=1,c=1). So the two files converge to different parameters.
- **Minor scaffold typo:** scaffold's multi-example cell calls `torch.randn(1., requires_grad=True)` (float `1.` as a size, which would error); COMPLETE correctly uses `torch.randn(1, ...)`.
- **Shared latent bug (present in BOTH files):** in the update step the code reads `c.copy_(m - (lr*m.grad))` — `c` is updated using `m`'s value and `m`'s gradient instead of its own. The single-sample loop still converges because the line fits with `c` small, but it is a genuine copy-paste bug worth flagging for anyone using these as reference.
- The COMPLETE file additionally has a couple of `# TODO` placeholder cells (e.g. "can we show the loss curve and this point on that curve?") that are aspirational and left unimplemented.

## Libraries / APIs used
- `torch`: `torch.tensor`, `torch.randn`, `torch.manual_seed`, `requires_grad=True`, `Tensor.backward()`, `Tensor.grad`, `Tensor.grad.zero_()`, `torch.no_grad()`, `Tensor.copy_()`, `torch.mean`, `.item()`.
- `matplotlib.pyplot` for the `plot_trace` loss/parameter curves.
- Deliberately **no** `torch.nn`, `torch.optim`, or dataloaders — the whole point is doing without them.

## Notable pedagogical choices
- **"Naked tensors":** the framing conceit — implement a trainable model with raw tensors only, so nothing is hidden behind `nn`/`optim` abstractions.
- **Strict math-then-code cadence:** every implementation is preceded by its LaTeX formula (data spaces → model → loss → update rule).
- **Derive-then-verify:** hand-derive the gradients with the chain rule, then run a cell comparing the closed form to autograd's `.grad`, proving `.backward()` reproduces the pencil-and-paper answer.
- **Learn the gotchas by experiment:** gradient accumulation is *shown* (backprop twice → value doubles → divide by 2) rather than stated, which then motivates `zero_()`.
- **Jokes / asides in comments:** "PPS: You will never do this again in your life." about manually zeroing grads; "Set a learning rate (0.01 ;)" nudging the value; "Play around with lr to see different things ;)".
- **Interactive step-through:** the training loop blocks on `input()` each iteration (Enter to continue, `q`/`exit`/`break` to stop) so parameter updates can be narrated live in the room.
- **Single-sample → many-samples progression:** starts with one data point so gradients are scalars you can reason about, then generalizes to vectors specifically to expose *why* a reduction to a scalar loss (`torch.mean`) is necessary before backprop.
- **"Disable copilot" instruction** in the fill-yourself header — the instructor explicitly wants participants to type the answers themselves, not autocomplete them.
