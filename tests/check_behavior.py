"""Compare exact behavior with a local Git revision, without network access.

Usage: python tests/check_behavior.py --baseline-ref 702e708 --output-dir /tmp/audit
Use the same Python/PyTorch/RDKit environment for both revisions.
"""

import argparse
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="702e70801b460ac0a9b5970263d5fe23339eb558")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.baseline_ref + "^{commit}"], cwd=repo, text=True
    ).strip()
    archive = subprocess.check_output(["git", "archive", commit], cwd=repo)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    normalized = []
    with tempfile.TemporaryDirectory(prefix="unimacro-reference-") as directory:
        reference = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            # The input is an archive of this repository's local Git commit.
            for member in tar.getmembers():
                target = (reference / member.name).resolve()
                if reference not in target.parents or member.issym() or member.islnk():
                    raise ValueError(f"Unsafe archive member: {member.name}")
            tar.extractall(reference)
        for source in reference.rglob("*.py"):
            original = source.read_bytes()
            try:
                original.decode("utf8")
            except UnicodeDecodeError:
                repaired = original.decode("utf8", errors="replace").encode("utf8")
                # Legacy comments contain invalid bytes. Only a byte-for-byte
                # equivalent executable AST is eligible as a reference.
                assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(repaired))
                source.write_bytes(repaired)
                normalized.append(str(source.relative_to(reference)))
        for name, root in [("baseline", reference), ("candidate", repo)]:
            with (out / f"{name}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(repo / "tests/capture_behavior.py"),
                        "--root",
                        str(root),
                        "--output",
                        str(out / f"{name}.json"),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(f"Captured {name}", flush=True)
    baseline = json.loads((out / "baseline.json").read_text())
    candidate = json.loads((out / "candidate.json").read_text())
    differences = [
        key
        for key in sorted(baseline.keys() | candidate.keys())
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]
    ]
    report = {
        "baseline_commit": commit,
        "records": len(baseline),
        "candidate_records": len(candidate),
        "different_records": differences,
        "equal": not differences,
        "comparison": "exact bytes, dtype, shape and scalar values",
        "baseline_comment_encoding_repairs": sorted(normalized),
        "baseline_executable_ast_preserved": True,
        "device": "CPU; trainer CUDA transfers replaced by identity in tests",
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if differences:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
