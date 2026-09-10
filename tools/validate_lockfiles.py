#!/usr/bin/env python3
"""Validate hash-complete CI/release lock files without historical reports."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = ROOT / "constraints/locks"
TARGETS = {
    "base-py310",
    "base-py312",
    "build-dev-py310",
    "build-dev-py312",
    "host-py310",
    "host-py312",
    "qualification-py310",
    "qualification-py312",
}
REQUIRED_OPTIONS = (
    "--index-url https://pypi.org/simple",
    "--extra-index-url https://download.pytorch.org/whl/cpu",
    "--only-binary :all:",
    "--require-hashes",
)
REQUIREMENT = re.compile(
    r"^([a-z0-9][a-z0-9-]*)==([^\s\\]+) \\\n    --hash=sha256:([0-9a-f]{64})$",
    re.MULTILINE,
)


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path: Path) -> dict[str, tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    for option in REQUIRED_OPTIONS:
        if text.count(option) != 1:
            raise ValueError(f"{path}: expected exactly one {option!r}")
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    if re.search(r"(^|\s)(-e|--editable|git\+|hg\+|svn\+|bzr\+|file:|\.\.?/)", body):
        raise ValueError(f"{path}: editable, VCS, or local-path requirement found")
    rows: dict[str, tuple[str, str]] = {}
    for name, version, digest in REQUIREMENT.findall(text):
        normalized = canonical(name)
        if normalized in rows:
            raise ValueError(f"{path}: duplicate requirement {normalized}")
        rows[normalized] = (version, digest)
    requirement_lines = [line for line in text.splitlines() if line and not line.startswith(("#", "--", " "))]
    if len(rows) != len(requirement_lines):
        raise ValueError(f"{path}: an unparsed or unhashed requirement remains")
    if "tt-transformers" in rows:
        raise ValueError(f"{path}: lock must not contain the project itself")
    return rows


def require_names(
    rows: dict[str, tuple[str, str]],
    required: set[str],
    *,
    target: str,
) -> None:
    missing = required - set(rows)
    if missing:
        raise ValueError(f"{target}: missing direct requirements {sorted(missing)}")


def main() -> int:
    paths = sorted(LOCK_DIR.glob("*.txt"))
    names = {path.stem for path in paths}
    if names != TARGETS:
        raise ValueError(f"lock target set drifted: {sorted(names)}")
    locks = {path.stem: parse_lock(path) for path in paths}

    for python in ("py310", "py312"):
        base = locks[f"base-{python}"]
        host = locks[f"host-{python}"]
        require_names(base, {"ttnn", "torch", "loguru", "transformers"}, target=f"base-{python}")
        require_names(
            host,
            {"ttnn", "torch", "loguru", "pytest", "jsonschema", "transformers", "tqdm"},
            target=f"host-{python}",
        )
        if base["ttnn"][0] != "0.77.0" or base["torch"][0] != "2.11.0+cpu":
            raise ValueError(f"base-{python}: TTNN/Torch target drifted")
        if base["transformers"][0] != "5.12.1":
            raise ValueError(f"base-{python}: Transformers target drifted")
        for name, identity in base.items():
            if host.get(name) != identity:
                raise ValueError(f"host-{python}: base artifact {name} differs")

        qualification = locks[f"qualification-{python}"]
        require_names(qualification, {"openai", "requests"}, target=f"qualification-{python}")
        if "torch" in qualification or "tt-transformers" in qualification:
            raise ValueError(f"qualification-{python}: runtime project dependency leaked in")

        build = locks[f"build-dev-{python}"]
        require_names(
            build,
            {"build", "mypy", "pre-commit", "readme-renderer", "ruff", "setuptools", "twine", "wheel"},
            target=f"build-dev-{python}",
        )
        if "black" in build:
            raise ValueError(f"build-dev-{python}: Black remains after selecting Ruff format")

    counts = ", ".join(f"{name}={len(locks[name])}" for name in sorted(locks))
    print(f"Validated eight hash-complete lock files: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
