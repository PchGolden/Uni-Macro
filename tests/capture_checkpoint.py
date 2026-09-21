"""Check a real pretrained checkpoint at its saved model dimensions on CPU."""

import argparse
import copy
import gc
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd
import torch

from capture_behavior import fingerprint, finite, rng_state, seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    root = options.root.resolve()
    sys.path[:0] = [str(root / "src"), str(root / "src/preprocess")]
    from dataloader.dataloader_polymer import collate_fn
    from main import parse_args
    from models.multi_mol_model import MultiMolModel
    from preprocessing_polymer import _process_row

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(options.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    with patch.object(sys, "argv", ["main.py"]):
        args = parse_args()
    vars(args).update(checkpoint["args"])
    seed()
    model = MultiMolModel(args)
    strict_error = None
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except RuntimeError as error:
        strict_error = str(error)
    # Match the existing finetune loader: missing parameters retain initialization.
    # Shape mismatches still fail; no model changes are made to fit the checkpoint.
    compatibility = model.load_state_dict(checkpoint["model_state"], strict=False)
    assert not compatibility.unexpected_keys
    assert model.lm_head.weight is model.embed_tokens.weight
    result = {
        "checkpoint": options.checkpoint.name,
        "strict_load_error": strict_error,
        "missing_keys": compatibility.missing_keys,
        "unexpected_keys": compatibility.unexpected_keys,
        "state": fingerprint(model.state_dict()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "rng_after_initialization": fingerprint(rng_state()),
    }
    del checkpoint
    gc.collect()
    # Take actual repository benchmark molecules rather than training on a fixture.
    frame = pd.read_csv(root / "datasets/Egc.csv").head(2)
    samples = []
    smiles_columns = [name for name in ["SMILES0", "SMILES1"] if name in frame]
    for idx, row in frame.iterrows():
        sample = _process_row((idx, row, smiles_columns, [], False))
        assert not sample.get("_error"), sample
        sample["src_pos"] = sample["src_pos"][0]
        samples.append(sample)
    batch = collate_fn(samples)
    result["benchmark_batch"] = fingerprint(batch)
    for task in ["finetune", "pretrain"]:
        args.main_task = task
        model.eval()
        with torch.no_grad():
            output = model(copy.deepcopy(batch))
            finite(output)
            result[f"{task}/eval"] = fingerprint(output)
            # Check rigid translation invariance/equivariance on real weights.
            shifted = copy.deepcopy(batch)
            translation = torch.tensor([1.0, -2.0, 0.5])
            shifted["src_pos"] += translation
            shifted_output = model(shifted)
            if task == "finetune":
                torch.testing.assert_close(shifted_output, output, rtol=2e-5, atol=2e-5)
            else:
                torch.testing.assert_close(shifted_output[0], output[0], rtol=2e-5, atol=2e-5)
                torch.testing.assert_close(
                    shifted_output[1], output[1] + translation, rtol=2e-5, atol=2e-5
                )
                torch.testing.assert_close(shifted_output[2], output[2], rtol=2e-5, atol=2e-5)
            result[f"{task}/translated"] = fingerprint(shifted_output)
        model.train()
        model.zero_grad(set_to_none=True)
        seed(37)
        output = model(copy.deepcopy(batch))
        finite(output)
        loss = (
            sum(t.square().mean() for t in output)
            if isinstance(output, tuple)
            else output.square().mean()
        )
        loss.backward()
        gradients = {name: param.grad for name, param in model.named_parameters()}
        finite([grad for grad in gradients.values() if grad is not None])
        result[f"{task}/train"] = fingerprint([output, loss, gradients, rng_state()])
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"Both tasks passed: {result['parameter_count']:,} parameters; "
        f"{len(compatibility.missing_keys)} initialized keys absent from checkpoint"
    )


if __name__ == "__main__":
    main()
