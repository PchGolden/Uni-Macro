# Behavior regression checks

Run with the project's numerical dependencies installed. These tests use the
standard library test utilities; pytest is not required.

```bash
python tests/check_behavior.py \
  --baseline-ref 702e70801b460ac0a9b5970263d5fe23339eb558 \
  --output-dir /tmp/unimacro-audit
```

The command exports the local Git revision, runs reference and candidate in
separate Python processes, and exits with a nonzero status on any difference.
It writes `baseline.json`, `candidate.json`, execution logs and `comparison.json`.
No network access or baseline source vendoring is needed.

The reference commit contains invalid UTF-8 bytes in three files' comments.
The runner repairs those bytes in its temporary reference only after verifying
identical executable ASTs. The original Git objects remain untouched.

The snapshots compare exact tensor/array bytes, shapes, dtypes and scalar values:

- RDKit features, conformers, explicit/random folds and the public CSV-to-PKL CLI.
- Pickle and both LMDB key/metadata formats, augmentation and collation.
- Model initialization, parameter ordering, tied embeddings and RNG states.
- Twelve configurations covering each ablation, combined ablations and DropPath,
  with both finetuning and pretraining heads in evaluation and training modes.
- Outputs, attention maps, losses, every parameter gradient, and AdamW updates.
- Nonzero perturbed projections, so zero-initialized heads cannot hide regressions.
- Regression/classification metrics, frozen encoder training, gradient accumulation,
  complete short finetuning/pretraining runs, early stopping and exported artifacts.
- Existing index error diagnostics and empty regression/pretraining evaluation.

CPU execution sets one numerical-library thread and enables deterministic PyTorch
algorithms. The original trainer requires CUDA; tests replace `.cuda()` transfers
with identity operations. They do not exercise CUDA kernels, AMP or multi-process
DDP, and they do not retrain the full paper experiments.

An optional full-size checkpoint check uses real molecules from `datasets/Egc.csv`:

```bash
python tests/capture_checkpoint.py \
  --root /path/to/reference-checkout \
  --checkpoint /path/to/checkpoint_adaptive.pt \
  --output /tmp/checkpoint-reference.json
python tests/capture_checkpoint.py \
  --root . \
  --checkpoint /path/to/checkpoint_adaptive.pt \
  --output /tmp/checkpoint-candidate.json
cmp /tmp/checkpoint-reference.json /tmp/checkpoint-candidate.json
```

The reference checkout should have the same comment-only UTF-8 repair described
above. This check records strict-loading errors and missing keys, then loads
available parameters while retaining missing parameters' initialization, as the
original finetuning loader does. Shape mismatches and unexpected keys fail the
check. It exercises both heads, dropout, backpropagation and translation sanity
checks. Baseline comparison is exact; translation checks use `rtol=atol=2e-5` to
allow floating-point rounding.
