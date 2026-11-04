# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the changed-file -> hardware-matrix node selection."""

from __future__ import annotations

import textwrap

import pytest

from qualification.tools import affected_hardware_nodes as affected
from qualification.tools import run_hardware_matrix as runner

pytestmark = pytest.mark.host

DEVICE_TEST = "import pytest\n\n\n@pytest.mark.device\ndef test_on_device():\n    pass\n"


def _write(root, files):
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))


def _node(node_id, target, enabled=True):
    return {
        "id": node_id,
        "mesh_device": "P150",
        "stage": "e2e",
        "selector": {"target": f"{target}::test_on_device"},
        "enabled": enabled,
    }


@pytest.fixture
def repo(tmp_path):
    _write(
        tmp_path,
        {
            "src/tt_transformers/__init__.py": "",
            "src/tt_transformers/ops/__init__.py": "",
            "src/tt_transformers/ops/gdn/__init__.py": "",
            "src/tt_transformers/ops/gdn/ops.py": "",
            "src/tt_transformers/ops/gdn/kernels/head.cpp": "",
            "src/tt_transformers/ops/attn/__init__.py": "",
            "src/tt_transformers/ops/attn/ops.py": "from ..gdn.ops import helper\n",
            "src/tt_transformers/modules/__init__.py": "",
            "src/tt_transformers/modules/norm.py": "",
            "src/tt_transformers/models/__init__.py": "",
            "src/tt_transformers/models/alpha/__init__.py": (
                '_EXPORTS = {"Alpha": ("tt_transformers.models.alpha.model", "Alpha")}\n'
            ),
            "src/tt_transformers/models/alpha/model.py": (
                "def build():\n    from tt_transformers.ops.attn import ops\n    return ops\n"
            ),
            "src/tt_transformers/models/beta/__init__.py": "",
            "src/tt_transformers/models/beta/model.py": "from tt_transformers.modules import norm\n",
            "examples/__init__.py": "",
            "examples/alpha/__init__.py": "",
            "examples/alpha/demo.py": "from tt_transformers.models.alpha import Alpha\n",
            "tests/__init__.py": "",
            "tests/conftest.py": "",
            "tests/hardware/alpha/test_demo.py": 'import subprocess\nCOMMAND = ["-m", "examples.alpha.demo"]\n'
            + DEVICE_TEST,
            "tests/hardware/beta/test_demo.py": "from tt_transformers.models.beta import model\n" + DEVICE_TEST,
            "tests/ops/test_gdn.py": "from tt_transformers.ops.gdn.ops import run\n" + DEVICE_TEST,
            "tests/ops/test_gdn_host.py": 'from tt_transformers.ops.gdn.ops import run\nTEXT = "pytest.mark.device"\n',
        },
    )
    return tmp_path


@pytest.fixture
def matrix():
    return {
        "nodes": [
            _node("bh-alpha", "tests/hardware/alpha/test_demo.py"),
            _node("bh-beta", "tests/hardware/beta/test_demo.py"),
            _node("bh-beta-disabled", "tests/hardware/beta/test_demo.py", enabled=False),
        ]
    }


def _select(repo, matrix, *files):
    return affected.select(files, matrix, affected.DependencyGraph(repo))


def _ids(selection):
    return [node["id"] for node in selection.nodes]


@pytest.mark.host
def test_op_change_reaches_the_model_through_relative_lazy_and_string_imports(repo, matrix):
    selection = _select(repo, matrix, "src/tt_transformers/ops/gdn/ops.py")
    assert _ids(selection) == ["bh-alpha"]
    assert selection.nodes[0]["via"] == [
        "tests.hardware.alpha.test_demo",
        "examples.alpha.demo",
        "tt_transformers.models.alpha",
        "tt_transformers.models.alpha.model",
        "tt_transformers.ops.attn.ops",
        "tt_transformers.ops.gdn.ops (src/tt_transformers/ops/gdn/ops.py)",
    ]
    assert selection.uncovered_device_tests == ["tests/ops/test_gdn.py"]


@pytest.mark.host
def test_kernel_file_counts_as_a_change_to_its_package(repo, matrix):
    selection = _select(repo, matrix, "src/tt_transformers/ops/gdn/kernels/head.cpp")
    assert _ids(selection) == ["bh-alpha"]
    assert selection.uncovered_device_tests == ["tests/ops/test_gdn.py"]


@pytest.mark.host
def test_unrelated_module_change_selects_only_its_consumers(repo, matrix):
    assert _ids(_select(repo, matrix, "src/tt_transformers/modules/norm.py")) == ["bh-beta"]


@pytest.mark.host
def test_package_root_and_conftest_reach_every_enabled_node(repo, matrix):
    for path in ("src/tt_transformers/__init__.py", "tests/conftest.py"):
        assert _ids(_select(repo, matrix, path)) == ["bh-alpha", "bh-beta"], path


@pytest.mark.host
def test_deleted_module_still_selects_its_importers(repo, matrix):
    (repo / "src/tt_transformers/modules/norm.py").unlink()
    assert _ids(_select(repo, matrix, "src/tt_transformers/modules/norm.py")) == ["bh-beta"]


@pytest.mark.host
def test_global_files_select_everything_and_documentation_selects_nothing(repo, matrix):
    selection = _select(repo, matrix, "pyproject.toml", "constraints/locks/host-py310.txt")
    assert _ids(selection) == ["bh-alpha", "bh-beta"]
    assert selection.global_files == ["constraints/locks/host-py310.txt", "pyproject.toml"]
    selection = _select(repo, matrix, "src/tt_transformers/ops/gdn/README.md", "docs/architecture.md")
    assert _ids(selection) == []
    assert selection.unmapped_files == ["docs/architecture.md", "src/tt_transformers/ops/gdn/README.md"]


@pytest.mark.host
def test_checked_in_matrix_targets_are_all_in_the_graph():
    graph = affected.DependencyGraph()
    matrix = runner.load_json(runner.DEFAULT_MATRIX)
    for node in matrix["nodes"]:
        name = affected.module_name(affected.selector_file(node))
        assert graph.files.get(name) == affected.selector_file(node), node["id"]


@pytest.mark.host
def test_checked_in_model_change_selects_only_that_models_nodes():
    matrix = runner.load_json(runner.DEFAULT_MATRIX)
    selection = affected.select(
        ["src/tt_transformers/models/qwen3_32b/weight_utils.py"], matrix, affected.DependencyGraph()
    )
    assert _ids(selection) and all("qwen3" in node_id for node_id in _ids(selection))
