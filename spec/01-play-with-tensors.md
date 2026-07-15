# Lesson 1: Play with Tensors

- **Notebook file(s):** `notebooks/1. Play with tensors.ipynb` (single notebook, no scaffold/complete pair)
- **Scaffolding model:** single notebook
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## Concepts Covered
- **What a tensor is** — surface. Presented as a short markdown paragraph: an n-dimensional array (like numpy) that *also* carries metadata and can track every operation done to it to build a computational graph, which enables automatic differentiation. Framed up-front as the reason tensors matter for deep learning.
- **Tensor creation** — moderate, code-first. `torch.tensor(3)` (int), `torch.tensor(2.0)` (float), `torch.randn(...)` (random), `torch.arange(...)` (ranged). Shown for scalars and vectors.
- **Dtype implicitly** — surface. Integer vs. float tensors created side by side (`torch.tensor(3)` vs `torch.tensor(2.0)`); `.to(torch.float)` used once to cast an `arange`.
- **Arithmetic ops** — surface, code. `+ - * /` demonstrated on scalar tensors and (element-wise) on vectors, all in one line each so participants see the results together.
- **Rank / dimensionality vocabulary** — moderate, code + comments. Scalar (0d), Vector (1d), Matrix (2d), 3-Tensor (3d) named explicitly in comments.
- **Shapes** — moderate, code. `.shape` inspected on several tensors including a contrived `torch.randn(1,3,4,5,1)` to show high-rank shapes and singleton dims.
- **Views / reshaping** — moderate, code. `a.view(6,2)`, `a.view(2,6)`, `a.view(2,2,3)` on a 12-element tensor to show shape is reinterpretable without copying data ("interchangeable").
- **Matrix multiplication vs. element-wise** — moderate, code. `torch.matmul` on `(1,4)@(4,1)` vs `(4,1)@(1,4)` to contrast inner- vs outer-product shapes; separately `a*b` shown as element-wise.
- **Automatic differentiation / `requires_grad`** — surface-to-moderate, code. `requires_grad=True` on a scalar and on matrices; operations run to show the graph is being built. This is an intro/tease — no `.backward()` or `.grad` inspection happens yet (that is the job of Lesson 2).

## Narrative & Depth
This is a hands-on "play" notebook — the very first exposure for a zero-familiarity audience. It is almost entirely runnable code with terse comments; the participant runs cells and reads outputs rather than filling blanks. The flow climbs the dimensionality ladder deliberately: scalars → vectors → (naming matrices/3-tensors) → shapes and views → matmul. The single fundamental idea it drives home is that a tensor is "numpy-plus": the same familiar array arithmetic, but with the extra machinery (metadata, operation tracking) that will later enable autodiff. The final section flips on `requires_grad=True` purely to plant the seed that these objects remember what was done to them, setting up the next lesson.

It deliberately does **not** cover: `.backward()`, `.grad`, broadcasting rules in any formal way, device/GPU, indexing/slicing, or any model. There is no loss, no training, no math notation — it is a tactile familiarization session.

## Scaffolding: provided vs. live-coded
There is no scaffold/complete split. Every code cell is written out and meant to be executed as-is (or lightly tweaked live). A few cells are effectively comment-only headers used as living section dividers (e.g. `## Matrices`, and a comment block listing Scalar/Vector/Matrix/3-Tensor). No `...` or `# TODO` blanks are present, so participants are not implementing anything themselves here — they observe and experiment.

## Libraries / APIs used
- `torch`: `torch.tensor`, `torch.randn`, `torch.arange`, `.shape`, `.view`, `.to(torch.float)`, `torch.matmul`, arithmetic operators, `requires_grad=True`.
- `numpy` imported at the top but not actually used (kept as the familiar reference point mentioned in the markdown).

## Notable pedagogical choices
- **Analogy anchor:** tensors introduced explicitly as "like numpy but with metadata / a computational graph" — leaning on whatever numpy familiarity exists rather than defining arrays from scratch.
- **Motivation-first framing:** the opening markdown states *why* (automatic differentiation → "crucial for deep learning") before any API is shown.
- **Results-together style:** most cells return a tuple like `a+b, a-b, a*b, a/b` so multiple outputs render side by side in one execution — cheap, immediate visual comparison.
- **Contrast by construction:** int vs float tensors, `matmul(a,b)` vs `matmul(b,a)`, element-wise vs matmul are shown adjacently so the difference is seen, not told.
- **Deliberate cliffhanger:** ends on `requires_grad` with operations but no gradient computation, handing off to Lesson 2.
