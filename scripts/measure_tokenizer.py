"""Measure a learner tokenizer on a fixed, handwritten CPU-only fixture.

This is measurement infrastructure, not an implementation of any tokenizer.
No artifact is written when the requested algorithms are unimplemented.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter

from lm_lab import __version__
from lm_lab.tokenization import BPETokenizer, ByteTokenizer, train_bpe

ROOT = Path(__file__).resolve().parents[1]
FIXED_TEXTS = (
    "Hello, language model! abab abab.\n",
    "亲手实现分词器，保留中文和标点。\n",
    "🙂🚀 emoji and  consecutive  spaces\n\n",
    "\tTabs, CRLF\r\n and e\u0301 stay unchanged.",
    "",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def code_version() -> dict[str, object]:
    """An uncommitted repository has no invented commit hash."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False,
        )
    except FileNotFoundError:
        return {"commit": None, "dirty": None, "note": "git is unavailable"}
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "note": "record the working-tree diff as well when dirty",
    }


def measure(args: argparse.Namespace) -> dict[str, object]:
    train_seconds = None
    if args.task == "byte":
        tokenizer = ByteTokenizer()
        config = {"vocab_size": 256, "pretokenization": "none", "special_tokens": []}
    else:
        start = perf_counter()
        model = train_bpe(FIXED_TEXTS, args.vocab_size, pretokenization=args.pretokenization)
        train_seconds = perf_counter() - start
        tokenizer = BPETokenizer(model)
        config = {
            "requested_vocab_size": args.vocab_size,
            "actual_vocab_size": len(model.vocab),
            "pretokenization": model.pretokenization,
            "special_tokens": model.special_tokens,
        }

    # Warm up and check correctness before recording any measurement.
    expected = [tokenizer.encode(text) for text in FIXED_TEXTS]
    for text, ids in zip(FIXED_TEXTS, expected, strict=True):
        if tokenizer.decode(ids) != text:
            raise ValueError("round-trip check failed; refusing to record performance")
    if [tokenizer.encode(text) for text in FIXED_TEXTS] != expected:
        raise ValueError("repeat encoding differs; refusing to record performance")

    raw_seconds = []
    for _ in range(args.repeats):
        start = perf_counter()
        for _ in range(args.iterations):
            batch = [tokenizer.encode(text) for text in FIXED_TEXTS]
        raw_seconds.append(perf_counter() - start)
        if batch != expected:
            raise ValueError("timed encoding differs from the correctness check")
    for text, ids in zip(FIXED_TEXTS, batch, strict=True):
        if tokenizer.decode(ids) != text:
            raise ValueError("round-trip check failed after timing")

    fixture_bytes = json.dumps(FIXED_TEXTS, ensure_ascii=False).encode("utf-8")
    total_bytes = sum(len(text.encode("utf-8")) for text in FIXED_TEXTS)
    total_tokens = sum(len(ids) for ids in expected)
    seconds = median(raw_seconds)
    return {
        "task": args.task,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "code_version": code_version(),
        "command": [sys.executable, *sys.argv],
        "config": config,
        "seed": None,
        "randomness": "none; fixed ordered fixture, deterministic tokenizer required",
        "data": {
            "source": "handwritten teaching fixture in scripts/measure_tokenizer.py; not a real dataset",
            "sha256": hashlib.sha256(fixture_bytes).hexdigest(),
            "samples": len(FIXED_TEXTS),
            "utf8_bytes": total_bytes,
            "training_and_measurement_overlap": args.task == "bpe",
        },
        "hardware": {
            "os": platform.platform(), "cpu": platform.processor() or "not reported",
            "architecture": platform.machine(), "gpu_used": False,
        },
        "python": sys.version,
        "correctness": {"round_trip": True, "repeat_encoding": True},
        "measurement": {
            "scope": "encoding only; includes Python batch-loop overhead",
            "warmup_batches": 2,
            "iterations_per_repeat": args.iterations,
            "repeats": args.repeats,
            "raw_seconds": raw_seconds,
            "median_seconds": seconds,
            "utf8_bytes_per_second": total_bytes * args.iterations / seconds,
            "tokens_per_second": total_tokens * args.iterations / seconds,
            "tokens_per_fixture": total_tokens,
            "tokens_per_utf8_byte": total_tokens / total_bytes,
            "training_seconds": train_seconds,
        },
        "limitations": [
            "tiny synthetic teaching fixture, not representative of real corpora",
            "BPE training and measurement share the fixture; no generalization claim",
            "CPU timing includes interpreter and loop overhead; no universal speedup claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["byte", "bpe"], required=True)
    parser.add_argument("--vocab-size", type=positive_int, default=280)
    parser.add_argument("--pretokenization", choices=["none", "simple"], default="simple")
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument("--output", type=Path, help="new JSON file; existing files are never overwritten")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new experiment path")
    try:
        result = measure(args)
    except NotImplementedError as exc:
        print(f"PENDING: {exc}. No measurement artifact was written.", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output if args.output is not None else ROOT / "runs" / "tokenizer" / f"{args.task}-{stamp}.json"
    result["raw_output_path"] = str(output.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Wrote {output}; inspect raw timings and limitations before drawing conclusions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
