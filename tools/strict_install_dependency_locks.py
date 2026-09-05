#!/usr/bin/env python3
"""Install every dependency lock into a fresh venv with hash enforcement."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "base-py310": ("310", "constraints/locks/base-py310.txt"),
    "host-py310": ("310", "constraints/locks/host-py310.txt"),
    "qualification-py310": ("310", "constraints/locks/qualification-py310.txt"),
    "build-dev-py310": ("310", "constraints/locks/build-dev-py310.txt"),
    "base-py312": ("312", "constraints/locks/base-py312.txt"),
    "host-py312": ("312", "constraints/locks/host-py312.txt"),
    "qualification-py312": ("312", "constraints/locks/qualification-py312.txt"),
    "build-dev-py312": ("312", "constraints/locks/build-dev-py312.txt"),
}


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python310", type=Path, required=True)
    parser.add_argument("--python312", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=tuple(TARGETS), default=tuple(TARGETS))
    args = parser.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=False)
    cache = args.work_root / "pip-cache"
    validations = []
    interpreters = {"310": args.python310, "312": args.python312}
    for target, (python_tag, relative_lock) in TARGETS.items():
        if target not in args.targets:
            continue
        interpreter = interpreters[python_tag]
        if not interpreter.is_file():
            raise FileNotFoundError(f"{target}: interpreter does not exist: {interpreter}")
        venv = args.work_root / target
        created = run([str(interpreter), "-m", "venv", str(venv)])
        if created.returncode:
            raise RuntimeError(f"{target}: venv creation failed\n{created.stdout}")
        python = venv / "bin/python"
        env = dict(os.environ)
        env["PIP_CACHE_DIR"] = str(cache)
        env.pop("PYTHONPATH", None)
        command = [str(python), "-m", "pip", "install", "--require-hashes", "-r", str(ROOT / relative_lock)]
        installed = run(command, env=env)
        checked = run([str(python), "-m", "pip", "check"], env=env) if installed.returncode == 0 else None
        frozen = run([str(python), "-m", "pip", "freeze", "--all"], env=env) if installed.returncode == 0 else None
        requires_torch = target.startswith(("base-", "host-"))
        torch_probe = (
            run([str(python), "-c", "import importlib.metadata as m; print(m.version('torch'))"], env=env)
            if installed.returncode == 0 and requires_torch
            else None
        )
        torch_absence_probe = (
            run(
                [
                    str(python),
                    "-c",
                    "import importlib.util; raise SystemExit(importlib.util.find_spec('torch') is not None)",
                ],
                env=env,
            )
            if installed.returncode == 0 and not requires_torch
            else None
        )
        tt_transformers_probe = (
            run(
                [
                    str(python),
                    "-c",
                    "import importlib.util; raise SystemExit(importlib.util.find_spec('tt_transformers') is not None)",
                ],
                env=env,
            )
            if installed.returncode == 0
            else None
        )
        passed = bool(
            installed.returncode == 0
            and checked is not None
            and checked.returncode == 0
            and (
                (
                    requires_torch
                    and torch_probe is not None
                    and torch_probe.returncode == 0
                    and torch_probe.stdout.strip() == "2.11.0+cpu"
                )
                or (not requires_torch and torch_absence_probe is not None and torch_absence_probe.returncode == 0)
            )
            and tt_transformers_probe is not None
            and tt_transformers_probe.returncode == 0
        )
        record = {
            "target": target,
            "strict": True,
            "require_hashes": True,
            "only_binary_from_lock": True,
            "clean_venv": True,
            "python": run([str(python), "-c", "import platform; print(platform.python_version())"]).stdout.strip(),
            "command": "python -m pip install --require-hashes -r " + relative_lock,
            "install_exit_code": installed.returncode,
            "pip_check_exit_code": None if checked is None else checked.returncode,
            "pip_check_output": None if checked is None else checked.stdout.strip(),
            "torch_version": None if torch_probe is None else torch_probe.stdout.strip(),
            "torch_absent": bool(torch_absence_probe and torch_absence_probe.returncode == 0),
            "tt_transformers_absent": bool(tt_transformers_probe and tt_transformers_probe.returncode == 0),
            "freeze": [] if frozen is None else [line for line in frozen.stdout.splitlines() if line],
            "passed": passed,
        }
        validations.append(record)
        (args.work_root / f"{target}-install.log").write_text(installed.stdout)
        if not passed:
            args.output.write_text(json.dumps({"schema_version": 1, "validations": validations}, indent=2) + "\n")
            print(f"{target}: strict install failed; see {args.work_root}/{target}-install.log")
            return 1
        print(f"{target}: strict install and pip check passed")
    args.output.write_text(json.dumps({"schema_version": 1, "validations": validations}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
