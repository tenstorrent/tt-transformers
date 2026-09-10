#!/usr/bin/env python3
"""Develop tt_transformers with editable TTNN or a commit-addressed wheel bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ttnn_dev_support.build import (  # noqa: E402
    attach,
    checkout_lock,
    clone_commit,
    export_wheel,
    rebuild,
    resolve_ref,
)
from tools.ttnn_dev_support.common import (  # noqa: E402
    ROOT,
    DevError,
    command,
    interpreter_info,
    load_environment,
    load_runtime,
    read_json,
    runtime_lock,
    runtime_python,
    source_state,
    tool_fingerprint,
    write_json,
)
from tools.ttnn_dev_support.environment import create_environment, doctor, requirements, run_environment  # noqa: E402

WORKFLOW = "ttnn-development.yml"
GITHUB_REPOSITORY = "tenstorrent/tt_transformers"


def config_path(args) -> Path:
    if args.environment:
        return args.environment.resolve()
    if not args.runtime:
        raise DevError("Supply --runtime or --environment.")
    manifest = args.runtime.resolve()
    data = load_runtime(manifest)
    return manifest.parent / ("environment.json" if data["mode"] == "source" else "environment/environment.json")


def configuration_lock(config: Path) -> Path:
    settings = load_environment(config)
    data = load_runtime(Path(settings["runtime"]))
    return (
        checkout_lock(Path(data["source"]["checkout"])) if data["mode"] == "source" else config.parent / ".runtime.lock"
    )


def container_execution(args, argv: list[str]) -> int:
    engine = shutil.which("docker") or shutil.which("podman")
    if not engine:
        raise DevError(
            "Install Docker/Podman, or run --executor native inside a prepared tt-metal development container."
        )
    image = args.image
    mounts = {ROOT.resolve()}
    if args.action in {"attach", "build", "env"}:
        mounts.add(args.project.resolve())
        output = args.output or args.runtime.parent / "environment"
        mounts.add(output.resolve())
        if args.action == "env":
            data = load_runtime(args.runtime.resolve())
            mounts.add(args.runtime.resolve().parent)
            image = image or data["recipe"].get("image")
    else:
        if args.action in {"doctor", "run"}:
            config = config_path(args)
            settings = load_environment(config)
            manifest = Path(settings["runtime"])
            mounts.update([config.parent, Path(settings["project"])])
        else:
            manifest = args.runtime.resolve()
        data = load_runtime(manifest)
        mounts.add(manifest.parent)
        image = image or data["recipe"].get("image")
        if data["mode"] == "source":
            mounts.update([Path(data["source"]["checkout"]), Path(data["source"]["build_dir"])])
            settings = read_json(manifest.parent / "workspace.json")
            mounts.add(Path(settings["project"]))
        if args.action == "export":
            mounts.add(args.output.resolve())
    if not image or not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise DevError("Container execution requires --image with an immutable sha256 digest.")
    if getattr(args, "tt_metal_checkout", None):
        mounts.add(args.tt_metal_checkout.resolve())
    if getattr(args, "build_dir", None):
        mounts.add(args.build_dir.resolve())
    for path in mounts:
        path.mkdir(parents=True, exist_ok=True)
    inner = list(argv)
    for index, argument in enumerate(inner):
        if argument == "--executor":
            inner[index + 1] = "native"
            break
        if argument.startswith("--executor="):
            inner[index] = "--executor=native"
            break
    # Reuse exactly the same absolute paths across container sessions for editable installs.
    launch = [engine, "run", "--rm", "--workdir", str(ROOT), "--env", f"TTNN_DEV_IMAGE={image}"]
    if getattr(args, "device", False):
        launch.extend(["--device", "/dev/tenstorrent", "--env", "TT_GH_CI_INFRA"])
    for path in sorted(mounts):
        launch.extend(["--mount", f"type=bind,src={path},dst={path}"])
    launch.extend([image, "python3", str(Path(__file__).resolve()), *inner])
    return subprocess.run(launch).returncode


def dispatch_build(args) -> int:
    if (
        args.tt_metal_checkout
        or args.build_dir
        or args.distributed
        or args.build_type != "Release"
        or args.extras != "examples,test"
    ):
        raise DevError(
            "The initial CI profile uses a remote commit, Release, single-host support, and examples,test extras. "
            "Use native/local-container execution for other profiles."
        )
    if interpreter_info(args.python)["minor"] != "3.10":
        raise DevError("The initial CI build profile targets Python 3.10.")
    sha = resolve_ref(args.tt_metal_ref)
    request_id = uuid.uuid4().hex
    payload = {
        "ref": args.workflow_ref,
        "inputs": {
            "metal_ref": sha,
            "request_id": request_id,
            "run_hardware": str(args.run_hardware).lower(),
            "image": args.image or "",
        },
    }
    endpoint = f"repos/{GITHUB_REPOSITORY}/actions/workflows/{WORKFLOW}/dispatches"
    response = subprocess.run(
        ["gh", "api", "--method", "POST", endpoint, "--input", "-"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
    )
    if response.returncode:
        raise DevError(response.stderr.strip() + f"\nThe workflow must be available on {args.workflow_ref}.")
    run_id = json.loads(response.stdout or "{}").get("workflow_run_id")
    for _ in range(18):
        if run_id:
            break
        listing = json.loads(
            command(
                ["gh", "api", f"repos/{GITHUB_REPOSITORY}/actions/runs?event=workflow_dispatch&per_page=30"],
                capture=True,
            )
        )
        run_id = next((run["id"] for run in listing["workflow_runs"] if request_id in run["display_title"]), None)
        if not run_id:
            time.sleep(5)
    if not run_id:
        raise DevError(f"Dispatch accepted but run ID is not visible yet. Search Actions for {request_id}.")
    print(f"CI build: https://github.com/{GITHUB_REPOSITORY}/actions/runs/{run_id}", flush=True)
    command(["gh", "run", "watch", str(run_id), "--repo", GITHUB_REPOSITORY, "--exit-status", "--interval", "30"])
    command(
        [
            "gh",
            "run",
            "download",
            str(run_id),
            "--repo",
            GITHUB_REPOSITORY,
            "--name",
            "ttnn-runtime",
            "--dir",
            args.output,
        ]
    )
    load_runtime(args.output / "runtime.json")
    print(args.output / "runtime.json")
    return 0


def build_command(args) -> Path:
    work = args.output.resolve()
    if args.tt_metal_checkout:
        selected = source_state(args.tt_metal_checkout.resolve())
        if selected["dirty"]:
            raise DevError("Commit source changes before a fixed-wheel build; use attach for local edits.")
        sha = selected["sha"]
    else:
        sha = resolve_ref(args.tt_metal_ref)
    if (work / "bundle/runtime.json").exists():
        data = load_runtime(work / "bundle/runtime.json")
        matches = (
            data["metal"]["sha"] == sha
            and data["recipe"]["image"] == args.image
            and data["recipe"]["build_type"] == args.build_type
            and data["recipe"]["distributed"] == args.distributed
            and data["recipe"].get("tool_fingerprint") == tool_fingerprint()
            and data["python"]["abi"] == interpreter_info(args.python)["abi"]
            and data["dependencies"]["extras"] == args.extras.split(",")
            and data["dependencies"]["requirements"] == requirements(args.project, args.extras.split(","))
        )
        if matches:
            print("Reusing verified runtime bundle.")
            return work / "bundle/runtime.json"
        raise DevError("Existing output contains another runtime; choose a new --output.")
    work.mkdir(parents=True, exist_ok=True)
    if args.tt_metal_checkout:
        checkout = args.tt_metal_checkout.resolve()
    else:
        checkout = work / "tt-metal"
        clone_commit(sha, checkout)
    print(f"Building tt-metal commit {sha}", flush=True)
    attach(
        checkout,
        args.project,
        work / "source",
        python=args.python,
        build_dir=args.build_dir,
        build_type=args.build_type,
        jobs=args.jobs,
        distributed=args.distributed,
        extras=args.extras.split(","),
        image=args.image,
    )
    doctor(work / "source/environment.json")
    if load_runtime(work / "source/runtime.json")["metal"]["sha"] != sha:
        raise DevError("Built source differs from the resolved commit.")
    return export_wheel(work / "source/runtime.json", work / "bundle")


def launch_command(args) -> int:
    config = config_path(args)
    argv = args.command
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise DevError("Provide a command after --.")
    with runtime_lock(configuration_lock(config), exclusive=False):
        report = doctor(config)
        settings = load_environment(config)
        data = load_runtime(Path(settings["runtime"]))
        env = run_environment(config, data, identity=report["fingerprint"])
        if argv[0] in {"python", "python3"}:
            argv[0] = str(runtime_python(Path(settings["venv"])))
        evidence = config.parent / "runs" / (time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8] + ".json")
        record = {"before": report, "command": argv, "started_at": time.time()}
        write_json(evidence, record)
        result = subprocess.run(argv, cwd=settings["project"], env=env)
        record.update(
            exit_code=result.returncode, finished_at=time.time(), project_after=source_state(Path(settings["project"]))
        )
        if data["mode"] == "source":
            record["metal_after"] = source_state(Path(data["source"]["checkout"]))
        write_json(evidence, record)
        print(f"Execution evidence: {evidence}", flush=True)
        return result.returncode


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="action", required=True)
    for name in ("attach", "build"):
        sub = commands.add_parser(name)
        sub.add_argument("--tt-metal-checkout", type=Path, required=name == "attach")
        sub.add_argument("--project", type=Path, default=ROOT)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument(
            "--python",
            default="3.10" if name == "build" else sys.executable,
            help="Target Python interpreter path or version (3.10 or 3.12)",
        )
        sub.add_argument("--build-dir", type=Path)
        sub.add_argument("--build-type", choices=("Release", "RelWithDebInfo", "Debug"), default="Release")
        sub.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 16))
        sub.add_argument("--distributed", action="store_true", help="Enable OpenMPI/multihost support")
        sub.add_argument("--extras", default="examples,test")
        sub.add_argument("--image", help="Pinned development image, also recorded in the build manifest")
        sub.add_argument("--executor", choices=("native", "local-container", "ci"), default="native")
        if name == "build":
            sub.add_argument("--tt-metal-ref", default="main")
            sub.add_argument("--workflow-ref", default="main")
            sub.add_argument("--run-hardware", action="store_true")
    sub = commands.add_parser("rebuild")
    sub.add_argument("--runtime", type=Path, required=True)
    sub.add_argument("--executor", choices=("native", "local-container"), default="native")
    sub.add_argument("--image")
    sub = commands.add_parser("export")
    sub.add_argument("--runtime", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--executor", choices=("native", "local-container"), default="native")
    sub.add_argument("--image")
    sub = commands.add_parser("env")
    sub.add_argument("--runtime", type=Path, required=True)
    sub.add_argument("--project", type=Path, default=ROOT)
    sub.add_argument("--output", type=Path)
    sub.add_argument("--python", default=sys.executable)
    sub.add_argument("--executor", choices=("native", "local-container"), default="native")
    sub.add_argument("--image")
    for name in ("doctor", "run"):
        sub = commands.add_parser(name)
        group = sub.add_mutually_exclusive_group(required=True)
        group.add_argument("--runtime", type=Path)
        group.add_argument("--environment", type=Path)
        sub.add_argument("--executor", choices=("native", "local-container"), default="native")
        sub.add_argument("--image")
        sub.add_argument("--device", action="store_true", help="Expose /dev/tenstorrent to the local container")
        if name == "doctor":
            sub.add_argument("--output", type=Path)
        else:
            sub.add_argument("command", nargs=argparse.REMAINDER)
    return root


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments)
    try:
        if hasattr(args, "jobs") and args.jobs < 1:
            raise DevError("--jobs must be positive.")
        if getattr(args, "executor", None) == "local-container":
            return container_execution(args, arguments)
        if getattr(args, "executor", None) == "ci":
            if args.action != "build":
                raise DevError("Only fixed-wheel builds can run in CI; attach uses your local checkout.")
            return dispatch_build(args)
        if args.action == "attach":
            result = attach(
                args.tt_metal_checkout,
                args.project,
                args.output,
                python=args.python,
                build_dir=args.build_dir,
                build_type=args.build_type,
                jobs=args.jobs,
                distributed=args.distributed,
                extras=args.extras.split(","),
                image=args.image,
            )
            print(json.dumps(doctor(result), indent=2))
            print(result)
        elif args.action == "build":
            print(build_command(args))
        elif args.action == "rebuild":
            result = rebuild(args.runtime.resolve())
            print(json.dumps(doctor(result), indent=2))
        elif args.action == "export":
            print(export_wheel(args.runtime.resolve(), args.output.resolve()))
        elif args.action == "env":
            result = create_environment(
                args.runtime.resolve(),
                args.project,
                (args.output or args.runtime.parent / "environment").resolve(),
                args.python,
            )
            print(json.dumps(doctor(result), indent=2))
            print(result)
        elif args.action == "doctor":
            config = config_path(args)
            with runtime_lock(configuration_lock(config), exclusive=False):
                result = doctor(config)
                if args.output:
                    write_json(args.output, result)
                print(json.dumps(result, indent=2))
        else:
            return launch_command(args)
        return 0
    except (DevError, OSError, ValueError) as error:
        print(f"ttnn-dev: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
