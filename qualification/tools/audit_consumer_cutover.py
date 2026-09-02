#!/usr/bin/env python3
"""Inventory pinned tt-metal and local vLLM consumers of in-tree TTTv2."""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "qualification/reports/consumer_import_sites.csv"
TT_METAL = Path("/localdev/gwang/tt-metal")
TT_METAL_REV = "00748e6ac7b65f50e5c2af07f6e7c1c535c7f4c0"
VLLM_CANDIDATES = (Path("/localdev/gwang/vllm"), Path("/localdev/gwang/vllm_duo/vllm"))
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(models\.common\.(?:sampling|modules|llm_runtime|models)(?:\.[A-Za-z0-9_\.]+)?)")
REGISTRY_RE = re.compile(r'"(models\.common\.models\.[A-Za-z0-9_\.]+:[A-Za-z0-9_]+)"')
VERSION = "tt-transformers==<approved-release> (exact pin plus wheel hash/lockfile)"
FIELDS = (
    "repository",
    "revision",
    "current_site",
    "line",
    "reference_kind",
    "current_reference",
    "consumer_owner",
    "replacement_tt_transformers_contract",
    "compatibility_shim_need",
    "shim_duration",
    "test_gate",
    "cutover_order",
    "external_status",
)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def git_bytes(repo: Path, revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"])


def owner(path: str) -> str:
    if path.startswith("tests/ttnn/"):
        return "TTNN test owners"
    if path.startswith("ttnn/") or path.startswith("tt_metal/"):
        return "TTNN/tt-metal core owners"
    if path.startswith("tt-train/"):
        return "tt-train GRPO rollout owners"
    if path.startswith("tests/emule/"):
        return "Emule model-integration owners"
    if path.startswith("models/tt_transformers/"):
        return "legacy TTTv1 model owners"
    if path.startswith("models/experimental/"):
        return "experimental model owners"
    if path.startswith("models/demos/"):
        parts = path.split("/")
        return f"tt-metal model owner: {parts[2] if len(parts) > 2 else 'demos'}"
    if path.startswith(("tests/pipeline_reorg/", "tests/scripts/", ".github/")):
        return "tt-metal models CI owners"
    if path.startswith("models/tttv2_") or path.startswith("models/test_tttv2_"):
        return "tt-metal TTTv2 qualification owners"
    return "tt-metal repository owners"


def replacement(module: str, path: str) -> tuple[str, str, str, str, str]:
    if path.startswith("tests/ttnn/"):
        return (
            "Relocate this model-level test to tt-transformers tests, or replace it with a TTNN-owned primitive test; no TTNN import of tt_transformers.",
            "no",
            "none",
            "TTNN core import audit plus relocated standalone sampling test",
            "1-before-any-consumer-switch",
        )
    if path == "models/experimental/llama32_1b_quasar/models/generator.py":
        return (
            f"{VERSION}; refactor to the public Llama3Generator/Llama3Executor contract. EagerLlamaExecutor and TracedLlamaExecutor have no symbol-compatible standalone export.",
            "no symbol shim: explicit experimental consumer refactor required",
            "none",
            "experimental Quasar generator contract test against the released standalone API",
            "1-before-any-consumer-switch",
        )
    new_module = module.replace("models.common.", "tt_transformers.", 1)
    return (
        f"{VERSION}; import {new_module}",
        "yes: temporary tt-metal old-namespace forwarding shim",
        "one tt-metal release after every inventoried consumer is migrated",
        "existing owner test plus clean-environment import with no tt-metal source on sys.path",
        "3-switch-consumer-after-release",
    )


def row(**kwargs) -> dict[str, str]:
    result = {field: "" for field in FIELDS}
    result.update(kwargs)
    result["external_status"] = "not_cut_over"
    return result


def tt_metal_import_rows() -> list[dict[str, str]]:
    output = git(
        TT_METAL,
        "grep",
        "-n",
        "-E",
        r"^[[:space:]]*(from|import)[[:space:]]+models\.common\.(sampling|modules|llm_runtime|models)",
        TT_METAL_REV,
        "--",
        ".",
    )
    rows = []
    for record in output.splitlines():
        _revision, path, line, text = record.split(":", 3)
        if path.startswith("models/common/"):
            continue
        match = IMPORT_RE.match(text)
        if not match:
            continue
        module = match.group(1)
        contract, shim, duration, gate, order = replacement(module, path)
        rows.append(
            row(
                repository="tt-metal",
                revision=TT_METAL_REV,
                current_site=path,
                line=line,
                reference_kind="python_import",
                current_reference=module,
                consumer_owner=owner(path),
                replacement_tt_transformers_contract=contract,
                compatibility_shim_need=shim,
                shim_duration=duration,
                test_gate=gate,
                cutover_order=order,
            )
        )
    return rows


def source_ci_rows() -> list[dict[str, str]]:
    files = (
        ".github/workflows/silencer.lock.yml",
        ".github/workflows/t3000-unit-tests.yaml",
        ".github/workflows/test-command.lock.yml",
        "tests/pipeline_reorg/models_e2e_tests.yaml",
        "tests/pipeline_reorg/models_sweep_tests.yaml",
        "tests/pipeline_reorg/models_unit_tests.yaml",
        "tests/scripts/t3000/run_t3000_unit_tests.sh",
    )
    rows = []
    for path in files:
        text = git_bytes(TT_METAL, TT_METAL_REV, path).decode(errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            if not re.search(r"tttv2|models/common/(?:tests|modules|models|llm_runtime)", line, re.IGNORECASE):
                continue
            rows.append(
                row(
                    repository="tt-metal",
                    revision=TT_METAL_REV,
                    current_site=path,
                    line=str(number),
                    reference_kind="ci_or_test_path",
                    current_reference=line.strip(),
                    consumer_owner=owner(path),
                    replacement_tt_transformers_contract=f"Install {VERSION}; invoke standalone tests/examples or qualification tools from the pinned release checkout, never the removed in-tree path.",
                    compatibility_shim_need="no path shim; CI workflow changes atomically",
                    shim_duration="none",
                    test_gate="CI manifest syntax plus host dry-run and scheduled same-version hardware lane",
                    cutover_order="4-switch-ci-after-consumer-imports",
                )
            )
    return rows


def qualification_rows() -> list[dict[str, str]]:
    paths = git(TT_METAL, "ls-tree", "-r", "--name-only", TT_METAL_REV, "models").splitlines()
    rows = []
    for path in paths:
        if not (Path(path).name.startswith("tttv2_") or Path(path).name.startswith("test_tttv2_")):
            continue
        rows.append(
            row(
                repository="tt-metal",
                revision=TT_METAL_REV,
                current_site=path,
                line="file",
                reference_kind="qualification_tool_or_contract",
                current_reference=path,
                consumer_owner=owner(path),
                replacement_tt_transformers_contract=f"Use {VERSION} with the corresponding qualification/ path from the exact standalone release; retain tt-metal copy only as a forwarding notice during overlap.",
                compatibility_shim_need="documentation/CLI forwarding notice only",
                shim_duration="one tt-metal release",
                test_gate="standalone qualification host tests and schema validation",
                cutover_order="4-switch-ci-after-consumer-imports",
            )
        )
    return rows


def vllm_rows() -> tuple[list[dict[str, str]], Path, str]:
    available = [path for path in VLLM_CANDIDATES if (path / ".git").is_dir()]
    if not available:
        return [], Path(""), ""
    repo = available[0]
    revision = git(repo, "rev-parse", "HEAD")
    platform = "plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py"
    text = git_bytes(repo, revision, platform).decode()
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        registry = REGISTRY_RE.search(line)
        if registry:
            current = registry.group(1)
            module, symbol = current.split(":", 1)
            new = module.replace("models.common.", "tt_transformers.", 1) + ":" + symbol
            rows.append(
                row(
                    repository="vllm TT plugin",
                    revision=revision,
                    current_site=platform,
                    line=str(number),
                    reference_kind="generator_registration",
                    current_reference=current,
                    consumer_owner="vLLM TT plugin platform/registration owners",
                    replacement_tt_transformers_contract=f"Plugin dependency {VERSION}; registry value {new}",
                    compatibility_shim_need="no import shim after plugin pin; retain tt_transformers_v2 selector alias temporarily",
                    shim_duration="selector alias for one plugin release after default switch",
                    test_gate="new registry unit test for all 12 IDs plus isolated import and representative server smoke",
                    cutover_order="2-update-plugin-registration-and-dependency",
                )
            )
        if "tt_transformers_v2" in line:
            rows.append(
                row(
                    repository="vllm TT plugin",
                    revision=revision,
                    current_site=platform,
                    line=str(number),
                    reference_kind="selector_or_error_contract",
                    current_reference=line.strip(),
                    consumer_owner="vLLM TT plugin platform/registration owners",
                    replacement_tt_transformers_contract=f"Resolve this selector only to generator paths from plugin-pinned {VERSION}; later make standalone selection the documented default.",
                    compatibility_shim_need="retain environment selector spelling during overlap",
                    shim_duration="one plugin release after standalone default",
                    test_gate="selector tests for each family, unknown-model fail-closed tests, then representative server smoke",
                    cutover_order="2-update-plugin-registration-and-dependency",
                )
            )
    rows.append(
        row(
            repository="vllm TT plugin",
            revision=revision,
            current_site="plugins/vllm-tt-plugin/tests/",
            line="gap",
            reference_kind="missing_test_gate",
            current_reference="No test references tt_transformers_v2 or the 12 standalone generator registrations at this revision.",
            consumer_owner="vLLM TT plugin test owners",
            replacement_tt_transformers_contract=f"Add tests against plugin-pinned {VERSION} without a tt-metal source checkout.",
            compatibility_shim_need="none",
            shim_duration="none",
            test_gate="required before registry switch",
            cutover_order="1-add-gates-before-switch",
        )
    )
    rows.append(
        row(
            repository="vllm TT plugin",
            revision=revision,
            current_site="plugins/vllm-tt-plugin/pyproject.toml",
            line="dependencies",
            reference_kind="missing_dependency_contract",
            current_reference='dependencies = ["tblib>=3.1.0"] (no tt-transformers dependency)',
            consumer_owner="vLLM TT plugin packaging owners",
            replacement_tt_transformers_contract=f"Add {VERSION} to plugin dependency/lock metadata without replacing the locally built TT vLLM package.",
            compatibility_shim_need="none",
            shim_duration="none",
            test_gate="clean plugin install and twelve-generator isolated import test",
            cutover_order="1-add-gates-before-switch",
        )
    )
    return rows, repo, revision


def verify_no_reverse_dependency(rows: list[dict[str, str]]) -> None:
    for prefix in ("ttnn/ttnn", "ttnn/cpp"):
        result = subprocess.run(
            ["git", "-C", str(TT_METAL), "grep", "-n", "-E", "(tt_transformers|models\\.common\\.(sampling|modules|models|llm_runtime))", TT_METAL_REV, "--", prefix],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr)
        matches = [line for line in result.stdout.splitlines() if "/tests/" not in line]
        if matches:
            raise RuntimeError(f"production TTNN reverse-dependency candidates found: {matches}")
    for item in rows:
        if item["current_site"].startswith("tests/ttnn/") and "import tt_transformers" in item[
            "replacement_tt_transformers_contract"
        ]:
            raise RuntimeError("proposed TTNN test reverse dependency")


def main() -> None:
    if git(TT_METAL, "rev-parse", "HEAD") != TT_METAL_REV:
        raise SystemExit("tt-metal checkout is not at the pinned source revision")
    rows = tt_metal_import_rows() + source_ci_rows() + qualification_rows()
    plugin_rows, plugin_repo, plugin_revision = vllm_rows()
    rows += plugin_rows
    verify_no_reverse_dependency(rows)
    rows.sort(key=lambda item: (item["repository"], item["current_site"], str(item["line"]), item["reference_kind"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} consumer sites from pinned tt-metal and {plugin_repo or 'no vLLM checkout'} {plugin_revision}")
    print("verified no proposed production TTNN -> tt-transformers reverse dependency")


if __name__ == "__main__":
    main()
