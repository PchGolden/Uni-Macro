# Code quality refactor audit

Baseline: `702e70801b460ac0a9b5970263d5fe23339eb558` (`main`).
Branch: `refactor/code-quality-20260921`.
All eight Python source files were reviewed. The source directory layout and all
13 dataset CSV files are unchanged.

## Changes

- **Models:** remove the duplicate hierarchical attention-mask construction;
  share categorical-index validation and ordered numeric-feature projection;
  reuse the atom-feature reduction; simplify ablation branches and constructor
  dimensions; remove unused intermediates. Document token ordering, return
  shapes, broadcast masks and the pair FFN's behavior under ablation.
- **Training:** separate finetuning and pretraining runners inside `src/main.py`;
  share the three pretraining loss calculations, frozen-encoder mode handling
  and CUDA batch transfer; consolidate optimizer and loader setup; remove the
  unreachable `no_random = False` branch and commented-out implementations.
- **Data:** share padding allocation/copying while retaining feature-dimension
  checks and the special base-mask padding rule; share legacy/current LMDB key
  lookup; close pickle inputs explicitly; share preprocessing output metadata;
  replace the diagonal mask loop with `np.fill_diagonal`.
- **Maintenance:** repair invalid comment encoding, correct misleading shape and
  offset documentation, and consistently format Python sources. Add reproducible
  differential tests and formatting configuration. Existing CRLF line endings
  are retained, with a Git whitespace attribute for clean diff checks.

Parameter names, parameter order, tied embeddings, initialization order, tensor
operation order, RNG consumption, scientific constants, loss reductions, task
defaults, splits, schedules and output formats were preserved. In particular,
`shorest_path_encoder` retains its legacy spelling for checkpoint compatibility.
Finetuning retains its two model initializations because removing one changes
seeded behavior. Coordinate loss still sums three coordinates per atom, and
distance loss still includes diagonal and inter-segment pairs.

## Baseline and verification

The raw baseline has non-UTF-8 bytes in comments in `multi_mol_model.py`,
`utils.py` and `molecular_features.py`. A fresh direct model import reproduced
`UnicodeDecodeError` when PyTorch's startup traceback inspected source lines.
For numerical comparison, the baseline was exported from Git and only those
comment bytes were repaired. Executable AST equality was checked before using
that reference. The raw Git commit remains available unchanged.

Environment: Python 3.9.20, PyTorch 2.5.0, NumPy 1.26.4, pandas 2.3.3,
RDKit 2022.9.5, scikit-learn 1.6.1, LMDB 1.5.1 and msgpack 1.1.0.
Numerical tests used CPU FP32, one thread and deterministic PyTorch algorithms.

| Check | Result |
| --- | --- |
| Differential regression suite | 582 / 582 records identical; zero differing records |
| Twelve model configurations, both tasks, train/eval | Exact outputs, gradients, RNG and AdamW state/update agreement |
| Initialization/checkpoint schema | Exact parameter values, names/order and state keys; tied weights retained |
| Data and preprocessing | Exact graph/conformer tensors, augmentation, folds, collation and exported CSV/PKL contents |
| Short complete training runs | Exact final weights, metrics, saved optimizer/scaler state, scheduler effects and RNG |
| Real checkpoint at saved dimensions | 91,245,361 parameters; exact forward/backward agreement for both tasks |
| Translation sanity on Egc molecules | Pass for predictions, coordinate equivariance and pair distances; `rtol=atol=2e-5` |
| Public CLI help | Identical for training and preprocessing |
| Static checks | Python compilation, Black formatting and `git diff --check` pass |
| Dataset integrity | All 13 CSV files byte-identical to baseline |

The differential comparison uses exact bytes, dtype, shape and scalar values,
without a numerical tolerance. Models are also tested with nonzero perturbed
projections so zero initialization cannot conceal changes. Tests capture
attention maps, exception diagnostics, regression/classification metrics,
freezing, gradient accumulation and early stopping.

The production checkpoint check used the locally available
`checkpoint_adaptive.pt` and the first two molecules in `datasets/Egc.csv`.
Strict loading fails on both baseline and candidate because this checkpoint
lacks 24 attention-gating parameters (weight/bias for each of 12 layers).
The numerical check follows the existing finetune loader's missing-key behavior:
retain initialized parameters and load available weights. Missing keys, strict
error text, initialized values and loaded values agree exactly. No architecture
change was made to accommodate this older checkpoint.

## Behavior and remaining limits

No scientific/numerical behavior difference was observed in the tested paths.
The comment-encoding repair removes the reproduced source-decoding failure;
that is the intentional operational behavior change.

CUDA, AMP and multi-process DDP were not run because the available node has no
CUDA device. CPU tests replace the trainer's `.cuda()` transfers with identity
operations; the production source remains CUDA-based. Full paper training and
all benchmark folds were not repeated.

Existing issues identified during review were preserved separately from the
refactor:

- Pretraining CSV sharding constructs four-element tasks, while `_process_row`
  expects five elements, including `mol_from_file`.
- Pretraining's validation/early-stop path calls `dist.broadcast` without a
  distributed-mode guard. The short pretraining entry-point test covers optimizer
  steps before this existing failure path; evaluation losses are tested separately.
- Pickle pretraining feature dropping mutates stored feature tensors through a
  shallow sample copy. The regression suite explicitly preserves this behavior.
- The existing `activation_fn` option does not select a different activation;
  changing it would alter the model definition.

Run the checks with the commands in [tests/README.md](tests/README.md). Large
checkpoints, generated datasets and numerical snapshots are not committed.
