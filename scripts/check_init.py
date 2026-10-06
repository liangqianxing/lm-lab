"""Check scaffolding only; never execute unfinished tokenizer algorithms."""

import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", "AGENTS.md", "pyproject.toml", ".gitignore",
    "src/lm_lab/__init__.py", "src/lm_lab/tokenization/__init__.py",
    "src/lm_lab/tokenization/byte_tokenizer.py", "src/lm_lab/tokenization/bpe.py",
    "tests/tokenization/conftest.py", "tests/tokenization/test_byte_tokenizer.py",
    "tests/tokenization/test_bpe_training.py", "tests/tokenization/test_bpe_encoding.py",
    "tests/tokenization/test_text_boundaries.py", "tests/tokenization/test_persistence.py",
    "docs/roadmap.md", "docs/tokenizer.md", "docs/experiment_template.md",
    "scripts/check_init.py", "scripts/measure_tokenizer.py", "data/README.md", "runs/README.md",
)


def main() -> int:
    failures: list[str] = []

    def check(condition: bool, description: str) -> None:
        print(f"{'PASS' if condition else 'FAIL'}: {description}")
        if not condition:
            failures.append(description)

    check(sys.version_info >= (3, 11), "Python 3.11 or newer")
    check(sys.prefix != sys.base_prefix, "running inside a virtual environment")
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    check(not missing, f"required files present; missing={missing}")

    try:
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        check(config["project"]["dependencies"] == [], "no external runtime dependencies")
        check(
            config["project"]["optional-dependencies"]["dev"] == ["pytest>=8,<10"],
            "pytest is the sole development dependency",
        )
        package = importlib.import_module("lm_lab")
        tokenizer = importlib.import_module("lm_lab.tokenization")
        check(version("lm-lab") == package.__version__, "installed distribution version matches package")
        check(Path(package.__file__).resolve().is_relative_to(ROOT / "src"), "imports use this source tree")
        names = ("ByteTokenizer", "BPEModel", "BPETokenizer", "Merge", "count_pairs", "merge_pair", "pretokenize", "train_bpe")
        check(all(hasattr(tokenizer, name) for name in names), "all exercise interfaces are importable")
        pytest_version = version("pytest")
        check(int(pytest_version.split(".")[0]) in (8, 9), f"compatible pytest installed ({pytest_version})")
    except (ImportError, PackageNotFoundError, KeyError, OSError, ValueError) as exc:
        check(False, f"configuration/imports: {exc}")

    text_files = {ROOT / name for name in REQUIRED if (ROOT / name).is_file()}
    for folder, suffix in [("src", "*.py"), ("tests", "*.py"), ("scripts", "*.py"), ("docs", "*.md")]:
        text_files.update((ROOT / folder).rglob(suffix))
    encoding_errors: list[str] = []
    syntax_errors: list[str] = []
    for path in sorted(text_files):
        try:
            text = path.read_text(encoding="utf-8")
            if chr(0xFFFD) in text:
                encoding_errors.append(f"{path.relative_to(ROOT)}: replacement character")
            if path.suffix == ".py":
                compile(text, str(path), "exec")
        except (UnicodeError, OSError) as exc:
            encoding_errors.append(f"{path.relative_to(ROOT)}: {exc}")
        except SyntaxError as exc:
            syntax_errors.append(f"{path.relative_to(ROOT)}: {exc}")
    check(not encoding_errors, f"UTF-8 integrity of {len(text_files)} files; errors={encoding_errors}")
    check(not syntax_errors, f"Python syntax; errors={syntax_errors}")
    print("Initialization only: tokenizer functionality has NOT been evaluated.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
