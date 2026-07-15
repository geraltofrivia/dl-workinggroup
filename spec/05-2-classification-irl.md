# Lesson 5.2: Classification IRL (In Real Life)

- **Notebook file(s):** `notebooks/5.2 Classification IRL.ipynb` (live-coding scaffold — the whole training routine is left as bare comments to be written live) and `notebooks/5.2 Classification IRL [COMPLETE].ipynb` (pre-filled, runnable end-to-end)
- **Scaffolding model:** scaffold + complete pair
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## Concepts Covered

- **Moving from toy data to a "real" dataset** — DEPTH: moderate. The scikit-learn Breast Cancer Wisconsin dataset (569 samples, 30 numeric features, binary target) replaces the hand-generated 1-D/2-D datasets of earlier lessons. Presented via code + a `pandas.DataFrame` inspection.
- **Multiple / many features** — DEPTH: moderate. The jump from 1–2 features to 30 features. Handled by parameterizing model input dimension (`df.shape[1]`, `X.shape[1]`); the point is that nothing conceptually changes, only the input width.
- **Train/test split & generalization** — DEPTH: moderate. `train_test_split` (80/20) is used up front, and the lesson explicitly separates "did it fit / memorize the training data" from "does it work on unseen data?" (dedicated markdown section + `evaluate` on the held-out test set).
- **Feature scaling / standardization** — DEPTH: moderate. `StandardScaler` fit on train, applied (transform-only) to test. Framed as "preprocessing in general" and as the thing that makes training actually work on raw-magnitude features.
- **Data-representation zoo** — DEPTH: surface. Topic list flags "the very many solutions (dataframe, HF datasets, torch datasets, and lists)"; in practice the notebook shows numpy ↔ pandas ↔ torch.tensor conversions and the reshaping/dtype gotchas (`unsqueeze(1)`, `dtype=torch.float`).
- **Logits vs. probabilities / `BCEWithLogitsLoss`** — DEPTH: moderate. The complete notebook drops the explicit sigmoid and uses `BCEWithLogitsLoss`, with the accuracy helper thresholding at logit `0` rather than probability `0.5` (a subtle but deliberate difference from the scaffold — see below).
- **Linear classifier vs. MLP** — DEPTH: moderate. A single-`Linear` `LinClf` is trained first; then the previous session's `MLP` is reused (only the input dim changes) to contrast capacity.
- **Regularization via weight decay** — DEPTH: surface. `weight_decay=1e-4` appears on the MLP's SGD optimizer, plus a learning-rate bump — introduced pragmatically, not derived.
- **Gradient Descent vs. Stochastic Gradient Descent** — DEPTH: surface (listed in topics). The code does full-batch gradient descent; SGD/minibatching is a talking point rather than implemented.
- **Harder datasets as teaser** — DEPTH: surface. `make_moons` and a hand-written `generate_spiral` close the notebook as "give me moar!" — non-linearly-separable problems that motivate what comes next.

## Narrative & Depth

This is the "in real life" counterpart to the earlier idealized classification lesson (5.1). Where 5.1 built the classifier machinery on clean, low-dimensional, synthetic data, 5.2 takes the *same* machinery and drops it onto a genuine, messy, 30-feature medical dataset to surface everything the idealized version glossed over: the data no longer arrives as tidy tensors (numpy/pandas conversions, dtype/shape fixes), features live on wildly different scales (so standardization becomes necessary, not optional), there are many features rather than one or two, and — crucially — fitting the training data is no longer the goal, generalizing to a held-out test set is.

The intended flow: (1) load a real dataset and *look* at it as a DataFrame; (2) "your turn — train a predictor" (the live-coding core); (3) discover that without scaling it misbehaves, add `StandardScaler`; (4) ask the honest question "does it work on unseen data?" and evaluate on the test set; (5) swap the linear classifier for an MLP and see the difference (and the failure modes — the complete notebook inspects `model.parameters()` under "what happened?"); (6) end by throwing genuinely non-linear datasets (moons, spiral) at the audience as a cliffhanger.

Fundamental principle: **the model and training loop you already understand are exactly what you use on real data — the hard part of "real life" is the data plumbing and the generalization question, not new math.** It deliberately does NOT cover: proper validation methodology (cross-validation, no validation set distinct from test), precision/recall/ROC (accuracy only), class imbalance, categorical/missing-value handling (the dataset is all-numeric and clean), minibatch DataLoaders (full-batch only despite naming SGD), or actually solving moons/spiral (left open).

## Scaffolding: provided vs. live-coded

**PRE-FILLED / already present in BOTH notebooks:**
- Imports (`torch`, `numpy`, `matplotlib`, `pandas`, `typing`, and the sklearn trio: `StandardScaler`, `load_breast_cancer`, `train_test_split`), plus the pandas display option.
- Data loading + `train_test_split` (cell 1) and the DataFrame inspection cell.
- The `accuracy` helper and the `evaluate` helper (handles numpy→tensor coercion, `unsqueeze`, `torch.no_grad`).
- The `StandardScaler` fit/transform cell.
- All the "harder datasets" tail: `make_moons` scatter, the "moar" gif markdown, and the `generate_spiral` function + plot.
- All section markdown ("Some helpers", "Does it work on unseen data?", "Hah! I did this one too! Give me moar!").

**FILLED IN LIVE (left as bare `# Your turn, train a predictor` comment in the scaffold):** essentially the entire modeling + training routine. In the scaffold, cell 7 is just the comment — participants must produce, live, everything the complete notebook spells out in cells 8–22:
- The `LinClf` module (single `nn.Linear(n_features, 1)`).
- The reused `MLP` class.
- Model / optimizer / loss instantiation (`SGD`, `BCEWithLogitsLoss`).
- The numpy→tensor conversion (`Xt`, `Yt` with dtype/`unsqueeze`) and the full training loop (zero_grad / forward / loss / backward / step, logging loss + accuracy).
- Loss/accuracy plotting.
- The MLP variant with `weight_decay` and tuned `lr`, plus the `model.parameters()` "what happened?" inspection.

**Notable differences between scaffold and COMPLETE:**
- **Accuracy threshold.** The scaffold's `accuracy` thresholds predictions at `> 0.5` (expecting probabilities); the COMPLETE version thresholds at `> 0` (expecting raw logits, consistent with its use of `BCEWithLogitsLoss`). The completed solution never applies an explicit sigmoid — it lets the logits-aware loss and logits-aware accuracy handle it.
- The scaffold has an extra scratch cell showing the `df[['worst compactness', 'mean radius']]` two-column selection idea (commented) and a `data.target_names` cell; the COMPLETE notebook trims these.
- Scaffold cell 11 already references `model` in the `evaluate(...)` call even though the model is meant to be built live — i.e. the "does it work on unseen data?" check is pre-wired to whatever the participant trains.

## Libraries / APIs used

- **PyTorch:** `torch.nn.Module`, `torch.nn.Linear`, `torch.nn.Sequential`, `torch.nn.ReLU`, `torch.nn.BCEWithLogitsLoss`, `torch.optim.SGD` (with `weight_decay`, `lr`), `torch.tensor`, `torch.no_grad`, `unsqueeze`.
- **scikit-learn:** `datasets.load_breast_cancer`, `datasets.make_moons`, `model_selection.train_test_split`, `preprocessing.StandardScaler`.
- **pandas** (`DataFrame`, display options), **numpy**, **matplotlib.pyplot**.

## Notable pedagogical choices

- Uses a real, recognizable medical dataset (breast-cancer diagnosis) to make "real life" concrete and stakes-y, while keeping it all-numeric/clean so plumbing — not data cleaning — is the lesson.
- The entire model + training loop is deliberately blanked to a single comment, forcing the class to reconstruct the full workflow from memory — a consolidation/"can you do it yourself now?" exercise rather than new theory.
- Explicitly stages "it memorized the training data" → "but does it generalize?" to plant the overfitting/generalization intuition through experience rather than definition.
- Standardization is introduced by pain: train first, observe misbehavior, then add the scaler — motivating preprocessing empirically.
- Switches to `BCEWithLogitsLoss` (numerically-stable logits path) in the solution, quietly modeling best practice, while adjusting the accuracy threshold to match — a small, deliberate "logits vs probabilities" teaching moment.
- Ends on unsolved harder datasets (moons, spiral) with a meme gif — a motivational cliffhanger rather than closure.
