# Lesson 5.3: Images

- **Notebook file(s):** `notebooks/5.3 Images.ipynb` (live-coding scaffold with blanks) and `notebooks/5.3 Images [COMPLETED].ipynb` (pre-filled, runs end-to-end)
- **Scaffolding model:** scaffold + complete pair
- **Approx. session:** 1h–1h30 (weekly, no homework, zero-familiarity audience)

## Concepts Covered

- **Images as tensors** — DEPTH: moderate. Presented via code + viz. MNIST digits loaded as `1 x 28 x 28` tensors scaled to `[0,1]`; a helper plots the raw matrix with `plt.imshow`. Establishes that an image is just a grid of numbers.
- **Flattening an image into a vector for an MLP** — DEPTH: moderate. Code. The 28x28 image is `reshape(-1, 784)` and fed to a plain two-layer non-linear classifier (`NonLinCLf`). Reuses the MLP idea from earlier lessons applied to pixels.
- **Multi-class classification / reading model outputs** — DEPTH: moderate. Code + viz. `viz_pred` bar-charts the 10 logits; `F.softmax` shown to turn logits into a probability-like distribution. Accuracy computed via `argmax == labels`.
- **RGB / channels (CIFAR-10)** — DEPTH: surface-to-moderate. Code. CIFAR-10 loaded from HuggingFace as `3 x 32 x 32`; contrast with single-channel MNIST. The same flatten-then-MLP recipe is applied to `3*32*32 = 3072` inputs and shown to perform poorly.
- **The core motivating insight: flattening destroys spatial locality** — DEPTH: deep-from-first-principles. Analogy/prose (markdown cell): "Each pixel contained some information but it also had information based on its neighbours. We took away this information." This is the pedagogical pivot of the lesson.
- **Convolution as a sliding kernel (from scratch)** — DEPTH: deep-from-first-principles. Code + animated viz + math. A grayscale grumpy-cat image is convolved by hand with an explicit NumPy double loop: extract ROI, element-wise multiply with kernel, sum → one activation. An edge-detection kernel (Sobel-like / averaged gradient) is used and the output feature map animates cell-by-cell alongside a red sliding-window box. Introduces output-size arithmetic `out = in - k + 1`.
- **Kernels as feature detectors** — DEPTH: moderate. Code + viz. The commented-out 3x3 Sobel vs. the active 5x5 horizontal-edge kernel show that different weights detect different features; the coolwarm feature map reveals detected edges.
- **`nn.Conv2d`, pooling, and a real CNN** — DEPTH: moderate. Code. A `ImgClf` CNN: two `Conv2d` layers (3→16→32 channels, `kernel_size=3, padding=1`), `MaxPool2d(2,2)` halving spatial dims each time, then flatten to an MLP head. Channel/spatial dimensions are annotated in comments at every stage.
- **Training/eval loop reuse** — DEPTH: surface. Code. Same optimizer/loss/epoch loop as prior lessons, showing the CNN slots into the identical training machinery.

## Narrative & Depth

Flow: (1) load MNIST, see an image is a number grid; (2) flatten and classify with a familiar MLP — it works well on easy digits; (3) move to harder, color CIFAR-10, apply the exact same MLP recipe — it works badly; (4) stop and ask "what did we do here?" — the markdown answers that flattening threw away neighbour/spatial information; (5) "what else can we do?" — build convolution by hand on a real photo with an animated sliding kernel so participants literally watch a feature map form; (6) lift that hand-rolled operation into `nn.Conv2d` + pooling to make a CNN and retrain on CIFAR-10.

The fundamental principle built: spatial structure is information, and convolution is the mechanism that preserves and exploits local neighbourhoods rather than discarding them. The from-scratch NumPy convolution loop (not `nn.Conv2d`) is where the depth-first teaching happens — the framework version comes only after the manual mechanics are seen.

Deliberately NOT covered: stride/padding math beyond the single `out = in - k + 1` case (padding appears only as a `padding=1` argument, not derived); backprop through convolutions; how conv kernels are *learned* (the hand kernel is fixed/hand-designed while `nn.Conv2d` kernels are learned — this distinction is left implicit); batch norm, dropout, modern architectures; GPU; data augmentation; achieving good accuracy (results are left modest, no tuning). No formal convolution-vs-cross-correlation discussion.

## Scaffolding: provided vs. live-coded

PRE-FILLED / already present in BOTH notebooks:
- `%matplotlib inline`, all imports (torch, torchvision, datasets, tqdm, matplotlib, numpy).
- MNIST download + `DataLoader` setup; CIFAR-10 HuggingFace load with `apply_transforms`, `collate_fn`, and `DataLoader`s.
- Viz helpers `viz_img` and `viz_pred`; data-exploration cells (features, shapes, single-batch inspection).
- The entire from-scratch convolution section: `prepimage`, kernel definitions, output-map init, and the full animated double-loop visualization (cells 29–33 identical in both).
- Standard training loops and test-collection loops (the loop bodies are given; only the model/head differs).
- Both narrative markdown cells ("What did we do here?", "What else can we do?").

FILLED IN live (blanks in the scaffold, completed in the pair):
- `NonLinCLf` — `__init__` bodies and `forward` empty in scaffold (concept: build the MLP classifier: `Linear(input,100)` → ReLU → `Linear(100,n_classes)`).
- MNIST accuracy line (cell 11): scaffold ends at comment "How do we use this to calc acc?"; completed adds `(argmax == labels).float().mean()`.
- CIFAR-10 MLP block (cells 21–22): scaffold has bare `# Init everything` / `# Training on it`; completed instantiates `NonLinCLf(3*32*32, 10)` and writes the training loop.
- CIFAR-10 MLP accuracy (cell 26): comment only in scaffold; computed in completed.
- `ImgClf` CNN (cell 35): scaffold leaves `self.conv1 =`, `self.conv2 =`, `self.pool =`, `self.mlp1/2 =` and the entire `forward` blank — this is the main live-coded concept (defining conv layers, pooling, flatten, MLP head).
- Final CNN accuracy (cell 39): comment `# Calc acc` in scaffold; computed in completed.

Notable scaffold vs. COMPLETE difference (beyond blanks): in the CNN test loop (cell 38) the scaffold still calls `model(inputs.reshape(-1, 3*32*32))` — a leftover from the MLP path that would break a conv model — whereas the COMPLETED version correctly passes `model(inputs)` (the 4D `bs x 3 x 32 x 32` tensor). Worth flagging as a live fix.

## Libraries / APIs used

- `torch`, `torch.nn` (`Module`, `Linear`, `Conv2d`, `MaxPool2d`), `torch.nn.functional` (`relu`, `softmax`), `torch.optim.SGD`, `CrossEntropyLoss`, `DataLoader`, `torch.reshape`/`argmax`.
- `torchvision.datasets.MNIST`, `torchvision.transforms` (`Compose`, `ToTensor`).
- HuggingFace `datasets.load_dataset("uoft-cs/cifar10")` with `set_transform` + custom `collate_fn`.
- `numpy` for the hand-rolled convolution.
- `PIL.Image` / `ImageOps.fit` for load/grayscale/resize; `matplotlib` (`pyplot`, `patches.Rectangle`) and `IPython.display.clear_output` for the step-by-step animation.
- `tqdm.auto` for progress bars.

## Notable pedagogical choices

- Deliberately shows the MLP *failing* on CIFAR-10 before introducing convolutions — motivation by contrast, not by assertion.
- Convolution is taught by literally animating a kernel sliding across a real photo (grumpy cat), building the feature map one activation at a time, before any framework abstraction is shown.
- Hand-designed edge kernels (with a commented alternative) let participants see that "weights = feature detector" concretely, seeding the later idea that a CNN *learns* such kernels.
- Heavy reuse: the same `NonLinCLf`, training loop, and accuracy pattern recur across MNIST and CIFAR-10, so the only new idea each step is isolated.
- Dimension bookkeeping is written as inline comments at every conv/pool stage, teaching participants to track tensor shapes rather than trust the framework.
- Uses `test.test_data`/`test.test_labels` (deprecated torchvision attributes) for a quick full-set MNIST accuracy — a pragmatic shortcut typical of a live session, not production style.
