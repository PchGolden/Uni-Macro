"""Capture deterministic scientific behavior; run in a fresh process per revision.

Snapshots hash the exact bytes, dtype and shape of every tensor/array. They include
initialization, RNG states, outputs, gradients and optimizer states, not just losses.
The trainer currently requires CUDA: its CPU tests replace only .cuda() transfers.
"""

import argparse
import copy
import hashlib
import importlib
import io
import json
import pickle
import random
import subprocess
import sys
import tempfile
import warnings
import zlib
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lmdb
import msgpack
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem


def fingerprint(value):
    if torch.is_tensor(value):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        return {
            "tensor": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        return {
            "array": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if isinstance(value, np.generic):
        return fingerprint(value.item())
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(type(value))


def seed(value=1729):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def model_args(**overrides):
    values = dict(
        encoder_embed_dim=24,
        pair_embed_dim=12,
        pair_hidden_dim=8,
        encoder_layers=2,
        encoder_ffn_embed_dim=48,
        encoder_attention_heads=4,
        num_kernel=8,
        num_tasks=2,
        wo_node=False,
        wo_atom_feat=None,
        wo_spd=False,
        wo_edge=False,
        wo_geom_3d=False,
        wo_triopm=False,
        wo_pair=False,
        dropout=0.15,
        attention_dropout=0.15,
        activation_dropout=0.1,
        pair_dropout=0.2,
        droppath_prob=0.0,
        main_task="finetune",
        task_type="reg",
        amp=False,
        rank=0,
        distributed=False,
        freeze_encoder=False,
        grad_accum_steps=2,
        grad_clip=1.0,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        fold=0,
        seed=42,
        pretrain_val_path="",
        world_size=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def perturb_parameters(model):
    # Zero-initialized output projections would hide attention/pair regressions.
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * 0.025)


def finite(value):
    if torch.is_tensor(value):
        assert torch.isfinite(value).all(), "nonfinite model output/gradient"
    elif isinstance(value, (tuple, list)):
        for item in value:
            finite(item)


def capture(root, output):
    sys.path[:0] = [str(root / "src"), str(root / "src/preprocess")]
    data = importlib.import_module("dataloader.dataloader_polymer")
    features = importlib.import_module("molecular_features")
    preprocess = importlib.import_module("preprocessing_polymer")
    trainer = importlib.import_module("main")
    model_module = importlib.import_module("models.multi_mol_model")
    utils = importlib.import_module("models.utils")
    attention = importlib.import_module("models.attention")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results = {}

    def record(name, value):
        assert name not in results, name
        results[name] = fingerprint(value)

    with patch.object(sys, "argv", ["main.py"]):
        record("cli/defaults", vars(trainer.parse_args()))

    # Real RDKit molecular graphs, including disconnected, stereo and dummy atoms.
    for smiles in ["CCO", "c1ccccc1", "*CC*", "[Na+].[Cl-]", "[He]", "F/C=C/F", "C[C@H](O)F"]:
        graph = features.build_initial_graph(Chem.MolFromSmiles(smiles))
        for dropped in [False, True]:
            record(
                f"graph/{smiles}/{dropped}",
                features.build_graph_features(*graph[:4], drop_feat=dropped),
            )
        record(f"graph/{smiles}/atomic_numbers", graph[4])

    rows = pd.DataFrame(
        [
            dict(
                SMILES0="CCO",
                SMILES1="CC",
                glob_feat0=2.0,
                glob_feat1=-0.5,
                seg0_feat0=0.4,
                seg1_feat0=0.6,
                y=1.25,
                z=-0.75,
                fold=0,
            ),
            dict(SMILES0="*CC*", SMILES1="", y=-2.0, z=3.0, fold=1),
            dict(
                SMILES0="c1ccccc1",
                SMILES1="O",
                glob_feat0=0.0,
                seg0_feat1=4.0,
                seg1_feat1=1.0,
                y=0.1,
                z=0.2,
                fold=0,
            ),
        ]
    )
    samples = []
    for idx, row in rows.iterrows():
        sample = preprocess._process_row((idx, row, ["SMILES0", "SMILES1"], ["y", "z"], False))
        assert not sample.get("_error"), sample
        samples.append(sample)
    record("preprocess/samples", samples)
    for index, smiles in enumerate(["not_a_smiles", ""]):
        record(
            f"preprocess/invalid/{index}",
            preprocess._process_row(
                (index, pd.Series({"SMILES0": smiles, "y": 1}), ["SMILES0"], ["y"], False)
            ),
        )
    for explicit in [True, False]:
        split_samples = copy.deepcopy(samples[::-1])
        frame = rows if explicit else rows.drop(columns="fold")
        folds = preprocess._assign_folds_to_samples(split_samples, frame, 3, 42)
        record(f"splits/{explicit}", [folds, split_samples])
        for k in range(3):
            record(
                f"splits/{explicit}/{k}",
                preprocess._build_train_val_test_indices_for_fold(folds, k, 42),
            )

    with tempfile.TemporaryDirectory(prefix="unimacro-capture-") as tmp:
        tmp = Path(tmp)
        preprocess._assign_folds_to_samples(samples, rows, 2, 42)
        samples[-1]["is_chain_aug"] = True
        pkl_path = tmp / "samples.pkl"
        with pkl_path.open("wb") as handle:
            pickle.dump({"samples": samples}, handle)
        for mode in ["train", "val", "full", "other"]:
            for parent_only in [False, True]:
                dataset = data.PolymerPickleDataset(pkl_path, 0, mode, parent_only)
                seed()
                record(f"pickle/{mode}/{parent_only}", [dataset.ids, list(dataset), rng_state()])

        # Exercise both LMDB key conventions and metadata conventions.
        for legacy in [False, True]:
            path = tmp / f"samples-{legacy}.lmdb"
            with lmdb.open(str(path), map_size=8 * 1024 * 1024) as env:
                with env.begin(write=True) as txn:
                    if legacy:
                        txn.put(b"__len__", str(len(samples)).encode())
                    else:
                        txn.put(
                            b"__meta__",
                            pickle.dumps({"num_samples": len(samples), "label_names": ["y", "z"]}),
                        )
                    for i, sample in enumerate(samples):
                        key = str(i) if legacy else f"sample_{i}"
                        txn.put(key.encode(), pickle.dumps(sample))
            for mode in ["train", "val", "full", "other"]:
                for parent_only in [False, True]:
                    dataset = data.PolymerLmDBDataset(str(path), 0, mode, parent_only)
                    seed()
                    record(
                        f"lmdb/{legacy}/{mode}/{parent_only}",
                        [dataset.ids, list(dataset), rng_state()],
                    )
                    if dataset.env is not None:
                        dataset.env.close()
        encoded = {
            "x": {"__tensor__": True, "dtype": "torch.float32", "shape": [2], "data": [1.0, 2.0]},
            "items": [1, "abc", None],
        }
        record("lmdb/msgpack", data._lmdb_load_obj(zlib.compress(msgpack.packb(encoded))))
        try:
            data._lmdb_load_obj(b"invalid")
        except RuntimeError as error:
            record("lmdb/error", [type(error).__name__, str(error)])

        # Preserve the existing in-place feature-drop behavior of pickle samples.
        for seed_value in [1, 2, 42]:
            seed(seed_value)
            dataset = data.PolymerPretrainDataset(pkl_path, 0, "full")
            record(f"augmentation/{seed_value}/first", dataset[0])
            record(f"augmentation/{seed_value}/second", dataset[0])
            record(f"augmentation/{seed_value}/stored", dataset.samples[0])
            record(f"augmentation/{seed_value}/rng", rng_state())

        eval_samples = list(data.PolymerPickleDataset(pkl_path, 0, "full"))
        batch = data.collate_fn(copy.deepcopy(eval_samples))
        record("collate/standard", batch)
        alternate = copy.deepcopy(eval_samples)
        for sample in alternate:
            sample["pair_type"] = sample["pair_type"].unsqueeze(-1).repeat(1, 1, 2)
        record("collate/3d_pair_type", data.collate_fn(alternate))
        for task, mode, distributed, validation in [
            ("finetune", "train", False, False),
            ("finetune", "val", False, False),
            ("pretrain", "full", False, False),
            ("pretrain", "full", False, True),
            ("finetune", "train", True, False),
            ("finetune", "val", True, False),
        ]:
            seed()
            args = model_args(
                main_task=task,
                distributed=distributed,
                pretrain_val_path=str(pkl_path) if validation else "",
            )
            loader, sampler = data.build_dataloader(str(pkl_path), args, mode)
            record(
                f"loader/{task}/{mode}/{distributed}/{validation}",
                [
                    loader.drop_last,
                    list(loader),
                    rng_state(),
                    None if sampler is None else list(sampler),
                ],
            )

        # Run the public finetune CSV -> PKL/export CLI, independently of helper tests.
        csv_path = tmp / "fixture.csv"
        rows.to_csv(csv_path, index=False)
        outroot = tmp / "processed"
        subprocess.run(
            [
                sys.executable,
                str(root / "src/preprocess/preprocessing_polymer.py"),
                "--task",
                "finetune",
                "--csv",
                str(csv_path),
                "--labels",
                "y,z",
                "--workers",
                "1",
                "--kfold",
                "2",
                "--seed",
                "42",
                "--outroot",
                str(outroot),
                "--dataset-name",
                "fixture",
                "--export-splits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for path in sorted(outroot.rglob("*")):
            if path.suffix == ".pkl":
                with path.open("rb") as handle:
                    record(f"cli/preprocess/{path.relative_to(outroot)}", pickle.load(handle))
            elif path.suffix == ".csv":
                record(f"cli/preprocess/{path.relative_to(outroot)}", path.read_text())

        # Hierarchical and padding-only masks include missing numeric features/padding.
        record(
            "mask/hierarchical",
            utils._build_attn_mask(
                batch["base_mask"],
                batch["segment_id"],
                batch["atom_mask"],
                batch["seg_valid_mask"],
                batch["glob_valid_mask"],
                num_heads=4,
            ),
        )
        record(
            "mask/padding",
            utils.build_padding_only_attn_mask(
                batch["atom_mask"], batch["seg_valid_mask"], batch["glob_valid_mask"], num_heads=4
            ),
        )

        variants = [
            ({}, "default"),
            ({"wo_node": True}, "wo_node"),
            ({"wo_atom_feat": [0, 1, 7]}, "wo_atom_feat_degree"),
            ({"wo_atom_feat": [0, 7]}, "wo_atom_feat"),
            ({"wo_spd": True}, "wo_spd"),
            ({"wo_edge": True}, "wo_edge"),
            ({"wo_spd": True, "wo_edge": True}, "wo_spd_edge"),
            ({"wo_geom_3d": True}, "wo_geom_3d"),
            ({"wo_triopm": True}, "wo_triopm"),
            ({"wo_pair": True}, "wo_pair"),
            ({"droppath_prob": 0.25}, "droppath"),
            (
                {
                    "wo_node": True,
                    "wo_pair": True,
                    "wo_geom_3d": True,
                    "wo_edge": True,
                    "wo_spd": True,
                    "wo_triopm": True,
                },
                "all_ablated",
            ),
        ]
        for overrides, name in variants:
            for task in ["finetune", "pretrain"]:
                seed()
                args = model_args(main_task=task, **overrides)
                model = model_module.MultiMolModel(args)
                prefix = f"model/{name}/{task}"
                record(prefix + "/initial", model.state_dict())
                record(prefix + "/parameter_order", list(dict(model.named_parameters())))
                record(prefix + "/state_key_order", list(model.state_dict()))
                record(prefix + "/initial_rng", rng_state())
                assert model.lm_head.weight is model.embed_tokens.weight
                # Check ordinary initialization AND nonzero trained projections.
                model.eval()
                with torch.no_grad():
                    original_output = model(copy.deepcopy(batch))
                finite(original_output)
                record(prefix + "/untrained_eval", original_output)
                perturb_parameters(model)
                checkpoint = {k: v.clone() for k, v in model.state_dict().items()}
                clone = model_module.MultiMolModel(args)
                clone.load_state_dict(checkpoint, strict=True)
                assert clone.lm_head.weight is clone.embed_tokens.weight
                for training in [False, True]:
                    model.load_state_dict(checkpoint, strict=True)
                    model.train(training)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
                    model.zero_grad(set_to_none=True)
                    seed(37)
                    out = model(copy.deepcopy(batch))
                    finite(out)
                    loss = (
                        sum(t.square().mean() for t in out)
                        if isinstance(out, tuple)
                        else out.square().mean()
                    )
                    loss.backward()
                    gradients = {k: p.grad for k, p in model.named_parameters()}
                    finite([g for g in gradients.values() if g is not None])
                    record(prefix + f"/{training}/output", out)
                    record(prefix + f"/{training}/loss", loss)
                    record(prefix + f"/{training}/gradients", gradients)
                    record(
                        prefix + f"/{training}/attention",
                        [layer.self_attn.last_attn for layer in model.encoder.layers],
                    )
                    record(prefix + f"/{training}/rng", rng_state())
                    optimizer.step()
                    record(prefix + f"/{training}/updated", model.state_dict())
                    record(prefix + f"/{training}/optimizer", optimizer.state_dict())

        # Direct attention: gating off and all-masked rows, beyond model defaults.
        for gating in [False, True]:
            seed()
            attn = attention.Attention(12, 12, 12, 6, 4, 3, gating=gating, dropout=0.2)
            perturb_parameters(attn)
            q = torch.randn(2, 4, 12)
            pair = torch.randn(2, 4, 4, 6)
            mask = torch.zeros(2, 1, 4, 4)
            mask[0, :, 0, :] = float("-inf")
            result = attn(q, q, q, pair, mask)
            finite(result)
            record(f"attention/gating/{gating}", [result, attn.last_attn, rng_state()])

        # Legacy CUDA-only training entry points; only transfers are shims here.
        with patch.object(torch.Tensor, "cuda", lambda tensor, *a, **kw: tensor):
            for task_type in ["reg", "cls"]:
                for frozen in [False, True]:
                    seed()
                    args = model_args(task_type=task_type, num_tasks=2, freeze_encoder=frozen)
                    model = model_module.MultiMolModel(args)
                    perturb_parameters(model)
                    if frozen:
                        trainer.freeze_all_but_head(model)
                    batches = [copy.deepcopy(batch) for _ in range(3)]
                    if task_type == "cls":
                        for item in batches:
                            item["label"] = torch.tensor([[0.0], [1.0], [0.0]])
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    scaler = torch.cuda.amp.GradScaler(enabled=False)
                    loss = trainer.train_one_epoch(model, batches, optimizer, scaler, 1, args, None)
                    record(
                        f"trainer/{task_type}/{frozen}",
                        [
                            loss,
                            model.state_dict(),
                            optimizer.state_dict(),
                            trainer.evaluate(model, copy.deepcopy(batches), args),
                            rng_state(),
                            {k: (p.requires_grad, p.grad) for k, p in model.named_parameters()},
                        ],
                    )
            seed()
            pre_model = model_module.MultiMolModel(model_args(main_task="pretrain"))
            perturb_parameters(pre_model)
            for masked in [False, True]:
                pre_batch = copy.deepcopy(batch)
                pre_batch["target_pos"] = batch["src_pos"] + 0.05
                pre_batch["target_token"] = (
                    batch["src_token"].clone() if masked else torch.zeros_like(batch["src_token"])
                )
                record(
                    f"pretrain/evaluate/{masked}",
                    trainer.evaluate_pretrain(
                        pre_model, [pre_batch], model_args(main_task="pretrain")
                    ),
                )
            record("evaluate/empty_reg", trainer.evaluate(pre_model, [], model_args()))
            record(
                "evaluate/empty_pretrain", trainer.evaluate_pretrain(pre_model, [], model_args())
            )

        # Execute complete finetuning entry points (checkpoint loading, schedules,
        # exports and early stopping), plus pretraining optimizer/accumulation.
        for task, frozen in [("finetune", False), ("finetune", True), ("pretrain", False)]:
            run_root = tmp / f"run-{task}-{frozen}"
            argv = [
                "main.py",
                "--main_task",
                task,
                "--dataset_name",
                "fixture",
                "--fold",
                "0",
                "--pkl_path",
                str(pkl_path),
                "--pretrain_train_path",
                str(pkl_path),
                "--pretrain_val_path",
                str(tmp / "validation.pkl"),
                "--results_root",
                str(run_root),
                "--weight_path",
                str(tmp / "missing.pt"),
                "--num_tasks",
                "2",
                "--epochs",
                "2",
                "--batch_size",
                "1",
                "--num_workers",
                "0",
                "--encoder_embed_dim",
                "24",
                "--pair_embed_dim",
                "12",
                "--pair_hidden_dim",
                "8",
                "--encoder_layers",
                "2",
                "--encoder_ffn_embed_dim",
                "48",
                "--encoder_attention_heads",
                "4",
                "--num_kernel",
                "8",
                "--grad_accum_steps",
                "2",
            ]
            if frozen:
                argv += ["--freeze_encoder", "--stop_on_target_rmse", "--target_rmse", "100"]
            with (tmp / "validation.pkl").open("wb") as handle:
                pickle.dump({"samples": samples}, handle)
            constructed = []

            def make_model(args):
                model = model_module.MultiMolModel(args)
                constructed.append(model)
                return model

            with patch.object(sys, "argv", argv), patch.object(
                trainer, "MultiMolModel", make_model
            ), patch.object(torch.Tensor, "cuda", lambda value, *a, **kw: value), patch.object(
                torch.nn.Module, "cuda", lambda value, *a, **kw: value
            ), redirect_stdout(
                io.StringIO()
            ):
                trainer.main()
            record(f"entry/{task}/{frozen}/model", constructed[-1].state_dict())
            record(f"entry/{task}/{frozen}/rng", rng_state())
            record(f"entry/{task}/{frozen}/constructions", len(constructed))
            for path in sorted(run_root.rglob("*")):
                key = f"entry/{task}/{frozen}/{path.relative_to(run_root)}"
                if path.suffix == ".json":
                    record(key, json.loads(path.read_text()))
                elif path.suffix == ".npy":
                    record(key, np.load(path))
                elif path.suffix == ".pt":
                    saved = torch.load(path, weights_only=False)
                    # Temporary absolute paths differ across processes.
                    for name in [
                        "pkl_path",
                        "pretrain_train_path",
                        "pretrain_val_path",
                        "results_root",
                        "weight_path",
                    ]:
                        saved["args"][name] = str(Path(saved["args"][name]).relative_to(tmp))
                    record(key, saved)

        # Error diagnostics remain part of the public behavior, including text.
        for key, bad, message in [
            ("atom_feat", -1, "atom_feat < 0"),
            ("atom_feat", 512, "atom_feat >= num_atom"),
            ("degree", -1, "degree < 0"),
            ("degree", 128, "degree >= num_degree"),
            ("edge_feat", -1, "edge_feat < 0"),
            ("edge_feat", 64, "edge_feat >= num_edge"),
            ("shortest_path", -1, "shortest_path < 0"),
            ("pair_type", -1, "atom_pair < 0"),
            ("pair_type", 16384, "atom_pair >= num_pair"),
        ]:
            model = model_module.MultiMolModel(model_args())
            bad_batch = copy.deepcopy(batch)
            bad_batch[key].flatten()[0] = bad
            stream = io.StringIO()
            with redirect_stdout(stream):
                try:
                    model(bad_batch)
                except RuntimeError as error:
                    assert str(error) == message
                    record(f"error/{key}/{bad}", [str(error), stream.getvalue()])
                else:
                    raise AssertionError(f"expected {message}")

        # Public checkpoint and JSON output formats.
        trainer.save_json(
            {"float": np.float32(1.2), "int": np.int64(2), "array": np.arange(3)},
            tmp / "metrics.json",
        )
        record("io/metrics", (tmp / "metrics.json").read_text())
        state = {"model_state": pre_model.state_dict(), "epoch": 3}
        trainer.save_checkpoint_finetune(state, tmp / "ckpt")
        trainer.save_checkpoint_pretrain(state, tmp / "ckpt", 5)
        for name in ["checkpoint.pt", "checkpoint_5.pt"]:
            record(f"io/{name}", torch.load(tmp / "ckpt" / name, weights_only=False))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
    print(f"Captured {len(results)} behavior records in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        capture(options.root.resolve(), options.output.resolve())
