"""Execute notebooks in notebooks/ and report pass/fail/skip.

Usage:
    python scripts/run_notebooks.py [glob]     # default glob: ch*.ipynb

Env switches:
    BOOK_SMOKE=1     smoke mode — notebooks read this and swap in tiny models/data
    BOOK_NB_SAVE=1   write executed outputs back into the .ipynb (used before
                     `quarto render` so chapter {{< embed >}} shortcodes pick up
                     stored outputs instead of re-executing)
    BOOK_NB_TIMEOUT  per-notebook timeout in seconds (default 1200)

Notebooks that need optional dependency groups declare them in metadata:
    nb.metadata["book"]["requires"] = ["llm"]        # or ["mechinterp"]
When the group's marker module is missing locally, the notebook is SKIPPED
(CI's default env installs neither group). Weekly CI runs this as the
API-churn canary.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"
TIMEOUT = int(os.environ.get("BOOK_NB_TIMEOUT", "1200"))

# dependency group -> marker module proving the group is installed
GROUP_MARKERS = {"llm": "torch", "mechinterp": "transformer_lens"}


def missing_groups(nb) -> list[str]:
    required = nb.metadata.get("book", {}).get("requires", [])
    return [
        group
        for group in required
        if GROUP_MARKERS.get(group) and importlib.util.find_spec(GROUP_MARKERS[group]) is None
    ]


def run(path: Path, save: bool) -> tuple[str, float, str]:
    nb = nbformat.read(path, as_version=4)
    absent = missing_groups(nb)
    if absent:
        return "SKIP", 0.0, f"missing dependency group(s): {', '.join(absent)}"
    client = NotebookClient(
        nb,
        timeout=TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    start = time.monotonic()
    try:
        client.execute()
        if save:
            nbformat.write(nb, path)
        return "PASS", time.monotonic() - start, ""
    except Exception as exc:  # noqa: BLE001 — report and continue
        return "FAIL", time.monotonic() - start, f"{type(exc).__name__}: {exc}"


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "ch*.ipynb"
    smoke = os.environ.get("BOOK_SMOKE") == "1"
    save = os.environ.get("BOOK_NB_SAVE") == "1"
    notebooks = sorted(NOTEBOOK_DIR.glob(pattern))
    if not notebooks:
        print(f"no notebooks match {pattern!r}", file=sys.stderr)
        return 1

    print(f"mode: {'smoke' if smoke else 'full'}{' +save' if save else ''} | {len(notebooks)} notebook(s)\n")
    failures = []
    for path in notebooks:
        status, seconds, detail = run(path, save)
        note = f"  [{detail}]" if status == "SKIP" else ""
        print(f"  {status:4s}  {path.name}  ({seconds:.1f}s){note}")
        if status == "FAIL":
            failures.append((path.name, detail))

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for name, err in failures:
            print(f"  {name}: {err[:400]}")
        return 1
    print("\ndone — no failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
