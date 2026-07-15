# Lesson 5.1: Sigmoid Classification

- **Notebook file(s):** `notebooks/5.1 Classification.ipynb` (live-coding scaffold, blanks with `...`) and `notebooks/5.1 Classification [COMPLETE].ipynb` (pre-filled, run-only)
- **Scaffolding model:** scaffold + complete pair
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## Concepts Covered

- **Framing regression vs. classification** — DEPTH: moderate. Presented via code + narrative. The lesson starts by re-using the previous session's MLP (a regressor trained with MSE) on a binary-labeled dataset, deliberately showing that a raw-output network "sort of works" but produces unbounded values, motivating the need for a probabilistic output.
- **Output as a probability / confidence** — DEPTH: moderate-to-deep-from-first-principles. Presented as analogy (treat model output as a confidence between 0 and 1) plus markdown. The core question posed to participants: "we want an output between 0 and 1 — how do we restrict output to this?"
- **The sigmoid function** — DEPTH: deep-from-first-principles. Presented via code + visualization: participants plot `torch.sigmoid` over a linspace (`-5` to `10`) to *see* the S-curve squashing any real number into (0, 1). It is then wired into the model's `forward()` as the output non-linearity.
- **Decision boundary** — DEPTH: moderate. Presented via visualization: the training loop renders filled probability contours and the 0.5 boundary line each epoch, so the boundary is watched forming live. For the circle dataset this is a non-linear (circular) boundary, reinforcing that the MLP learns curved separators.
- **Binary cross-entropy loss (BCELoss)** — DEPTH: moderate. Presented via code and by contrast. The notebook first trains the sigmoid model still using the default MSELoss (slow / poor), then swaps in `torch.nn.BCELoss()` as "the one more thing it's missing," showing dramatically better/faster convergence. The *why* BCE suits probabilities is conveyed empirically rather than derived.

## Narrative & Depth

Flow:
1. Reuse the prior MLP unchanged on a 2D binary "inside/outside a circle" dataset, trained with MSE — establishes a baseline and the problem.
2. Inspect raw model outputs (arbitrary real numbers) — motivates bounding the output.
3. Markdown reframes the output as a probability/confidence in [0, 1].
4. Visualize the sigmoid curve in 1D to build intuition for the squashing function.
5. Build `MLPClf`, identical to the MLP but with `torch.sigmoid` wrapping the final layer's output.
6. Train the sigmoid model with the *wrong* loss (MSE) first to show it under-performs, then switch to `BCELoss` to show the proper pairing.
7. Closing markdown restates the mental model: classification = treating output as a probability between 0.0 (negative class) and 1.0 (positive class).

Fundamental principle built: a classifier is just a regressor whose output is squashed into a probability and trained with a loss designed for probabilities. Intuition is layered from concrete (unbounded outputs look wrong) to visual (sigmoid curve) to empirical (BCE beats MSE).

What it deliberately does NOT cover: the mathematical derivation of BCE / maximum likelihood / log-loss; softmax and multi-class classification; logits vs. probabilities as a numerical-stability concern (`BCEWithLogitsLoss` is not used); train/test splits, accuracy metrics, precision/recall, or any evaluation beyond the loss curve and visual boundary; overfitting/regularization. The focus is single-logit binary classification from first principles.

## Scaffolding: provided vs. live-coded

PRE-FILLED / already present in BOTH notebooks:
- All imports and `torch.manual_seed(42)` (cell 1).
- Helper/viz functions `viz_1d`, `viz_2d`, `viz_all`, `report` (cell 3).
- The full `train()` function including the live per-epoch decision-boundary + probability-contour plotting and the final loss-curve plot (cell 4). Note this `train()` internally applies `torch.sigmoid` to the model output when drawing the boundary.
- Data-generation constants `n_samples`, `noise_scale` (cell 6), the `gen_circle` function and `X_cir` generation + visualization (cells 7–8), and label generation (cell 8).
- The output-inspection cell (11) and all markdown cells.

What participants FILL IN live (blanks in the scaffold `5.1 Classification.ipynb`):
- Cell 9: the entire `MLP` class body (reconstructing the previous session's MLP with a variable `input_dim`). Scaffold shows only the comment; COMPLETE has the full `nn.Module` with input layer, looped hidden layers + activations, and a single-unit output.
- Cell 10: the first `train(...)` call — filled to `train(MLP(2), X_cir, Y_cir, loss_function=torch.nn.MSELoss(), viz_every=10)`. Concept: baseline regressor on classification data.
- Cell 13: the sigmoid visualization — `viz_1d(X, ...)` filled to `viz_1d(X, torch.sigmoid(X))`. Concept: the sigmoid curve.
- Cell 14: the `MLPClf` class — scaffold leaves the body blank; COMPLETE is identical to `MLP` except `forward` returns `torch.sigmoid(self.layers(x))`. Concept: wiring sigmoid into the model.
- Cell 15: `train(...)` call — filled to `train(MLPClf(2), X_cir, Y_cir, viz_every=200, epochs=5000, learning_rate=0.5)` (still default MSELoss, deliberately weak result).
- Cell 16: `train(...)` call — filled to use `loss_function=torch.nn.BCELoss()`, `epochs=1000`, `lr=0.5`. Concept: correct loss for probabilities.

Notable differences between scaffold and COMPLETE: purely the fill-ins above — the two model classes, the four `train(...)`/`viz_1d(...)` invocations, and their hyperparameters. Everything structural (helpers, train loop, data) is identical. The scaffold's blanks are the two model definitions and the argument choices (loss function, epochs, learning rate) that drive the MSE-vs-BCE contrast.

## Libraries / APIs used

- `torch`, `torch.nn` — `nn.Module`, `nn.Linear`, `nn.Sequential`, `nn.ReLU`, `nn.MSELoss`, `nn.BCELoss`, `torch.sigmoid`, `torch.rand`, `torch.linspace`, `torch.manual_seed`, `torch.no_grad`.
- `torch.optim.SGD` (default optimizer in `train()`).
- `numpy` — meshgrid / grid construction for boundary plotting.
- `matplotlib.pyplot` — scatter, `contourf`/`contour`, colorbar, loss curve.
- `seaborn` — styling in `viz_all`.
- `IPython.display` — `clear_output` for the live-updating training animation.

## Notable pedagogical choices

- Teaches classification as a minimal delta from the previous regression MLP: reuse the exact same network, change only the output squashing and the loss. Lowers cognitive load for a zero-familiarity audience.
- Uses a "circle" dataset so the decision boundary is visibly non-linear, showing off what an MLP can learn while keeping the task 2D and plottable.
- Deliberately trains the "wrong" configurations first (MSE on the regressor, then MSE on the sigmoid model) before revealing BCE — learning by contrast/failure rather than presenting the correct answer up front.
- The live per-epoch probability-contour animation makes the abstract "learning a boundary" concrete and watchable in real time.
- Sigmoid is introduced visually (plot the curve) before it is used, so the mechanism is understood before it is applied.
- Uses `BCELoss` on already-sigmoided outputs (conceptually clean: model outputs a probability, loss consumes a probability) rather than the numerically-preferred `BCEWithLogitsLoss` — a deliberate depth-first simplification favoring clarity over production correctness.
