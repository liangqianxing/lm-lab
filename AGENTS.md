# Repository Guidelines

## Project Structure & Module Organization

This repository is a hands-on language model implementation lab. `src/lm_lab/`
contains the Python package; `tokenization/` holds the first exercises.
`tests/tokenization/` contains small behavioral acceptance tests grouped by
T1–T5. `docs/` contains the roadmap, exercise contracts, and experiment
template. `scripts/` provides initialization checks and tokenizer measurement.
`data/` and `runs/` hold local inputs and outputs; only their READMEs are tracked.
Add later modules when their learning stage begins.

## Build, Test, and Development Commands

Use Python 3.11 or newer. In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/check_init.py
.\.venv\Scripts\python.exe -m pytest -m t1 -q
.\.venv\Scripts\python.exe -m pytest -q
```

The editable installation uses setuptools. `check_init.py` checks project
structure, imports, syntax, and UTF-8; it does not validate tokenizer algorithms.
Marker selections run individual exercises; the full suite checks T1–T5.
Run `scripts/measure_tokenizer.py --help` for local CPU measurement options.

## Coding Style & Naming Conventions

Use UTF-8, four-space indentation, type annotations on public interfaces,
`snake_case` for functions and files, and `PascalCase` for classes.
Document byte IDs, merge ranks, and boundary behavior. Prefer the standard
library; pytest is the only development dependency. No formatter or linter
is currently required. Keep changes focused and preserve existing work.

## Testing Guidelines

Name tests `test_*.py` and use the registered `t1`–`t5` markers. Test actual
outputs, round trips, deterministic merges, and serialization. There is no
numeric coverage target. Unimplemented exercises must raise
`NotImplementedError` and remain visible as failures. Do not hide them with
unconditional passes, permanent skips, or broad expected-failure markers.

## Commit & Pull Request Guidelines

No commit history exists yet. Use concise imperative subjects, optionally
prefixed with `docs:`, `test:`, or `feat:`. PRs should state the learning task,
changed behavior, commands run, passing checks, and remaining failures.
Link relevant issues and experiment records; never report unmeasured gains.

## Learning Workflow & Local Artifacts

Assist with explanations, staged hints, acceptance design, and debugging.
Leave BPE, attention, optimizers, and other core algorithms to the learner
unless explicitly asked to implement them. Do not download assignment
solutions or replace BPE with a tokenizer library. Official compatibility
requires the corresponding handout. Mark milestones verified only after
implementation and acceptance. Keep datasets, weights, caches, outputs,
and secrets out of Git; avoid large downloads and paid compute.
