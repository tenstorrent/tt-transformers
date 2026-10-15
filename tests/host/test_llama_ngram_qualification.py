# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Keep real-model qualification launch and provenance checks device-free."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/qualify_llama32_ngram.py"
_SPEC = importlib.util.spec_from_file_location("llama_ngram_qualification", _SCRIPT)
qualification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qualification)


@pytest.mark.host
@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_launch_selects_standalone_eager_sync_host_model(monkeypatch, batch, k):
    monkeypatch.setenv("TT_MODEL_CLASS_OVERRIDES", "TTLlamaForCausalLM=wrong.module:WrongModel")
    monkeypatch.setenv("MESH_DEVICE", "wrong")
    monkeypatch.delenv("DISABLE_BATCHED_PREFILL", raising=False)
    args = SimpleNamespace(plugin_root=Path("/plugin"), checkpoint_revision="a" * 40, port=8123)
    command, environment = qualification.launch_settings(args, batch, k)
    assert command[1] == "/plugin/examples/server_example_tt.py"
    assert command[command.index("--model") + 1] == "meta-llama/Llama-3.2-1B"
    assert command[command.index("--block_size") + 1] == "32"
    assert command[command.index("--max_num_seqs") + 1] == str(batch)
    assert "--no-async-scheduling" in command
    assert "--no-enable-prefix-caching" in command
    tt = json.loads(command[command.index("--additional-config") + 1])["tt"]
    assert tt == {"trace_mode": "none", "optimizations": "performance"}
    assert environment["MESH_DEVICE"] == "(1, 1)"
    assert environment["DISABLE_BATCHED_PREFILL"] == "1"
    assert qualification.os.environ.get("DISABLE_BATCHED_PREFILL") is None
    assert environment["TT_MODEL_CLASS_OVERRIDES"] == f"TTLlamaForCausalLM={qualification.ADAPTER}"
    if k:
        speculation = json.loads(command[command.index("--speculative-config") + 1])
        assert speculation == {
            "method": "ngram",
            "num_speculative_tokens": k,
            "prompt_lookup_min": 2,
            "prompt_lookup_max": 4,
        }
    else:
        assert "--speculative-config" not in command


@pytest.mark.host
@pytest.mark.parametrize("revision", ["main", "a" * 39, "g" * 40, "a" * 41])
def test_checkpoint_preflight_rejects_unpinned_revision_before_cache_lookup(revision):
    lookup = Mock()
    with pytest.raises(ValueError, match="full commit SHA"):
        qualification.require_cached_checkpoint(revision, lookup)
    lookup.assert_not_called()


@pytest.mark.host
@pytest.mark.parametrize("path", [None, "/cache/snapshots/" + "b" * 40 + "/config.json"])
def test_checkpoint_preflight_rejects_missing_or_different_base_snapshot(path):
    lookup = Mock(return_value=path)
    with pytest.raises(ValueError, match="base-checkpoint refs/main"):
        qualification.require_cached_checkpoint("a" * 40, lookup)
    lookup.assert_called_once_with("meta-llama/Llama-3.2-1B", "config.json", revision="main")


@pytest.mark.host
def test_checkpoint_preflight_accepts_exact_snapshot():
    path = "/cache/snapshots/" + "a" * 40 + "/config.json"
    assert qualification.require_cached_checkpoint("a" * 40, Mock(return_value=path)) == path


@pytest.mark.host
@pytest.mark.parametrize("metadata", [None, {"url": "https://example.invalid/vllm.tar.gz"}])
def test_package_provenance_labels_absent_vcs_revision_unverified(monkeypatch, metadata):
    distribution = SimpleNamespace(
        version="0.30.0", read_text=Mock(return_value=None if metadata is None else json.dumps(metadata))
    )
    monkeypatch.setattr(qualification.importlib.metadata, "distribution", Mock(return_value=distribution))
    source = qualification.package_source("vllm")
    assert source["source_revision"] is None
    assert source["revision_verified"] is False
    assert source["revision_status"].startswith("unverified:")


@pytest.mark.host
def test_package_provenance_preserves_installed_commit(monkeypatch):
    metadata = {"vcs_info": {"vcs": "git", "commit_id": "c" * 40}}
    distribution = SimpleNamespace(version="0.30.0", read_text=Mock(return_value=json.dumps(metadata)))
    monkeypatch.setattr(qualification.importlib.metadata, "distribution", Mock(return_value=distribution))
    source = qualification.package_source("vllm")
    assert source["source_revision"] == "c" * 40
    assert source["revision_verified"] is True


@pytest.mark.host
@pytest.mark.parametrize("batch,k", [(3, 1), (8, 2), (1, 4), (2, -1)])
def test_launch_refuses_unqualified_capacity(batch, k):
    args = SimpleNamespace(plugin_root=Path("/plugin"), checkpoint_revision="a" * 40, port=8123)
    with pytest.raises(ValueError, match="B=1,2,4"):
        qualification.launch_settings(args, batch, k)


@pytest.mark.host
@pytest.mark.parametrize("count,child", [(0, ""), (17, ""), (18, "<skipped/>"), (18, "<failure/>"), (18, "<error/>")])
def test_device_matrix_requires_every_real_test_to_pass(tmp_path, count, child):
    report = tmp_path / "matrix.xml"
    report.write_text("<testsuite>" + f"<testcase>{child}</testcase>" * count + "</testsuite>")
    with pytest.raises(AssertionError, match="18 B/K device tests"):
        qualification.require_device_matrix(report)


@pytest.mark.host
def test_device_matrix_accepts_all_18_executed_cases(tmp_path):
    report = tmp_path / "matrix.xml"
    report.write_text("<testsuite>" + "<testcase/>" * 18 + "</testsuite>")
    qualification.require_device_matrix(report)


@pytest.mark.host
@pytest.mark.parametrize("filename", ["device-matrix.xml", "device-matrix_20261015_012345.xml"])
def test_junit_report_resolves_exact_or_ci_timestamp_and_preserves_artifact(tmp_path, filename):
    original = tmp_path / filename
    content = "<testsuite>" + "<testcase/>" * 18 + "</testsuite>"
    original.write_text(content)
    expected = tmp_path / "device-matrix.xml"
    resolved = qualification.canonicalize_junit_report(expected)
    assert resolved == expected
    assert resolved.read_text() == original.read_text() == content
    qualification.require_device_matrix(resolved)


@pytest.mark.host
def test_junit_report_rejects_missing_report_and_ignores_unrelated_or_invalid_timestamps(tmp_path):
    for filename in (
        "host-contracts_20261015_012345.xml",
        "device-matrix_backup.xml",
        "device-matrix_20260230_012345.xml",
        "device-matrix_20261015_252345.xml",
    ):
        (tmp_path / filename).write_text("<testsuite/>")
    with pytest.raises(FileNotFoundError, match="No JUnit report"):
        qualification.canonicalize_junit_report(tmp_path / "device-matrix.xml")


@pytest.mark.host
@pytest.mark.parametrize("first", ["device-matrix.xml", "device-matrix_20261015_012345.xml"])
def test_junit_report_rejects_ambiguous_exact_or_timestamped_candidates(tmp_path, first):
    (tmp_path / first).write_text("<testsuite/>")
    (tmp_path / "device-matrix_20261015_012346.xml").write_text("<testsuite/>")
    with pytest.raises(ValueError, match="Ambiguous JUnit reports"):
        qualification.canonicalize_junit_report(tmp_path / "device-matrix.xml")


@pytest.mark.host
def test_junit_report_rejects_malformed_xml_without_creating_canonical_artifact(tmp_path):
    (tmp_path / "device-matrix_20261015_012345.xml").write_text("<testsuite>")
    expected = tmp_path / "device-matrix.xml"
    with pytest.raises(qualification.ET.ParseError):
        qualification.canonicalize_junit_report(expected)
    assert not expected.exists()


@pytest.mark.host
def test_rerun_removes_only_owned_reports_and_cannot_reuse_old_device_success(tmp_path):
    for filename in ("device-matrix.xml", "device-matrix_20261015_012345.xml"):
        (tmp_path / filename).write_text("<testsuite>" + "<testcase/>" * 18 + "</testsuite>")
    preserved = [
        tmp_path / "host-contracts_20261015_012345.xml",
        tmp_path / "device-matrix_backup.xml",
        tmp_path / "device-matrix_20260230_012345.xml",
    ]
    for path in preserved:
        path.write_text("retained diagnostics")
    qualification.reset_completion_markers(tmp_path)
    assert all(path.read_text() == "retained diagnostics" for path in preserved)
    with pytest.raises(FileNotFoundError, match="No JUnit report"):
        qualification.canonicalize_junit_report(tmp_path / "device-matrix.xml")


@pytest.mark.host
@pytest.mark.parametrize("child", ["skipped", "failure", "error"])
def test_fresh_timestamped_device_failure_cannot_inherit_old_18_case_success(tmp_path, child):
    (tmp_path / "device-matrix_20261015_012345.xml").write_text("<testsuite>" + "<testcase/>" * 18 + "</testsuite>")
    qualification.reset_completion_markers(tmp_path)
    fresh = tmp_path / "device-matrix_20261015_012346.xml"
    content = "<testsuite>" + f"<testcase><{child}/></testcase>" * 18 + "</testsuite>"
    fresh.write_text(content)
    expected = qualification.canonicalize_junit_report(tmp_path / "device-matrix.xml")
    assert expected.read_text() == content
    with pytest.raises(AssertionError, match="18 B/K device tests"):
        qualification.require_device_matrix(expected)


@pytest.mark.host
def test_partial_rerun_cannot_retain_full_qualification_success(tmp_path):
    (tmp_path / "success.json").write_text('{"device_matrix_executed":true}')
    (tmp_path / "device-matrix.xml").write_text("<testsuite>" + "<testcase/>" * 18 + "</testsuite>")
    (tmp_path / "partial.json").write_text("{}")
    (tmp_path / "server.log").write_text("retained diagnostics")

    qualification.reset_completion_markers(tmp_path)
    qualification.write_json(tmp_path / "partial.json", {"device_matrix_executed": False})

    assert not (tmp_path / "success.json").exists()
    assert not (tmp_path / "device-matrix.xml").exists()
    assert json.loads((tmp_path / "partial.json").read_text()) == {"device_matrix_executed": False}
    assert (tmp_path / "server.log").read_text() == "retained diagnostics"


def _completion(token_ids=None, logprobs=None):
    token_ids = [19, 19, 19] if token_ids is None else token_ids
    logprobs = [-4.0, -3.0, -2.0] if logprobs is None else logprobs
    return SimpleNamespace(
        status=200,
        token_ids=token_ids,
        body={"choices": [{"logprobs": {"token_logprobs": logprobs}}]},
    )


@pytest.mark.host
def test_completion_record_retains_each_raw_logprob_and_response():
    result = _completion()
    record = qualification.completion_record(result)
    assert record["token_ids"] == [19, 19, 19]
    assert record["logprobs"] == [-4.0, -3.0, -2.0]
    assert record["response"] is result.body


@pytest.mark.host
@pytest.mark.parametrize(
    "logprobs",
    [[-4.0], [-4.0, -3.0, float("nan")], [-4.0, float("inf"), -2.0], [-4.0, None, -2.0], [-4.0, "-3.0", -2.0]],
)
def test_completion_record_rejects_missing_or_nonfinite_raw_logprobs(logprobs):
    with pytest.raises(AssertionError, match="one finite raw logprob"):
        qualification.completion_record(_completion(logprobs=logprobs))


def _assert_full_length_completion(result, count):
    assert result.status == 200
    assert len(result.token_ids) == result.completion_tokens == count
    assert result.finish_reason == "length"


@pytest.fixture
def verification_probe(tmp_path):
    server_log = tmp_path / "server.log"
    server_log.write_text("")
    prompt = [700] * 96
    result = _completion([700] * 512, [-1.5] * 512)
    result.completion_tokens = 512
    result.finish_reason = "length"

    def complete(*args, **kwargs):
        server_log.write_text("128 verify in logits mode\n")
        return result

    client = SimpleNamespace(complete=Mock(side_effect=complete))
    return client, prompt, server_log, result


@pytest.mark.host
@pytest.mark.parametrize("initial_log", ["", "0 verify in logits mode\n", "128 ordinary decode, 0 verify\n"])
def test_missing_positive_logits_count_runs_bounded_probe_and_requires_executed_verification(
    verification_probe, initial_log
):
    client, prompt, server_log, result = verification_probe
    server_log.write_text(initial_log)
    record = qualification.ensure_logged_logits_verification(
        client, prompt, 700, server_log, _assert_full_length_completion
    )
    client.complete.assert_called_once_with(
        prompt,
        max_tokens=512,
        temperature=0.7,
        seed=1171,
        top_k=8,
        top_p=0.8,
        allowed_token_ids=[700],
        ignore_eos=True,
        logprobs=1,
        return_tokens_as_token_ids=True,
    )
    assert len(record["prompt"]) + record["controls"]["max_tokens"] < 1024
    assert record["token_ids"] == [700] * 512
    assert record["logprobs"] == [-1.5] * 512
    assert record["response"] is result.body
    assert record["executed_logits_verifications"] == 128


@pytest.mark.host
def test_existing_positive_logits_count_avoids_probe(verification_probe):
    client, prompt, server_log, _ = verification_probe
    server_log.write_text("17 verify in logits mode\n0 verify in logits mode\n")
    assert (
        qualification.ensure_logged_logits_verification(client, prompt, 700, server_log, _assert_full_length_completion)
        is None
    )
    client.complete.assert_not_called()


@pytest.mark.host
def test_logits_probe_rejects_http_failure(verification_probe):
    client, prompt, server_log, result = verification_probe
    result.status = 400
    result.body = {"error": "probe rejected"}
    with pytest.raises(AssertionError, match="Completion failed"):
        qualification.ensure_logged_logits_verification(client, prompt, 700, server_log, _assert_full_length_completion)


@pytest.mark.host
def test_logits_probe_completion_without_executed_count_fails(verification_probe):
    client, prompt, server_log, result = verification_probe
    client.complete.side_effect = None
    client.complete.return_value = result
    with pytest.raises(AssertionError, match="no executed logits verification after the 512-token probe"):
        qualification.ensure_logged_logits_verification(client, prompt, 700, server_log, _assert_full_length_completion)


@pytest.mark.host
@pytest.mark.parametrize("failure", ["short", "usage", "stopped", "token", "nonfinite"])
def test_logits_probe_requires_full_length_correct_tokens_and_finite_raw_logprobs(verification_probe, failure):
    client, prompt, server_log, result = verification_probe
    if failure == "short":
        result.token_ids.pop()
        result.body["choices"][0]["logprobs"]["token_logprobs"].pop()
        result.completion_tokens = 511
    elif failure == "usage":
        result.completion_tokens = 511
    elif failure == "stopped":
        result.finish_reason = "stop"
    elif failure == "token":
        result.token_ids[-1] = 701
    else:
        result.body["choices"][0]["logprobs"]["token_logprobs"][-1] = float("nan")
    with pytest.raises(AssertionError):
        qualification.ensure_logged_logits_verification(client, prompt, 700, server_log, _assert_full_length_completion)


@pytest.mark.host
def test_reference_comparison_accepts_numerical_tolerance():
    expected = qualification.completion_record(_completion())
    actual = qualification.completion_record(_completion(logprobs=[-4.0005, -3.0005, -2.0005]))
    qualification.assert_reference_record(actual, expected, "concurrent request 2")


@pytest.mark.host
@pytest.mark.parametrize("label", ["concurrent request 2", "preemption/resume request 1", "abort/reuse replacement"])
def test_reference_comparison_detects_wrong_state_even_when_allowed_token_ids_match(label):
    expected = qualification.completion_record(_completion())
    actual = qualification.completion_record(_completion(logprobs=[-4.0, -3.25, -2.0]))
    assert actual["token_ids"] == expected["token_ids"]
    with pytest.raises(AssertionError, match="raw logprobs differ"):
        qualification.assert_reference_record(actual, expected, label)


@pytest.mark.host
@pytest.mark.parametrize(
    ("actual", "expected", "message"),
    [
        (
            {"token_ids": [19, 20, 19], "logprobs": [-4.0, -3.0, -2.0]},
            {"token_ids": [19, 19, 19], "logprobs": [-4.0, -3.0, -2.0]},
            "token IDs differ",
        ),
        (
            {"token_ids": [19, 19, 19], "logprobs": [-4.0]},
            {"token_ids": [19, 19, 19], "logprobs": [-4.0, -3.0, -2.0]},
            "count",
        ),
        ({"token_ids": [19], "logprobs": [-4.0]}, {"token_ids": [19], "logprobs": [float("nan")]}, "finite"),
        ({"token_ids": [19], "logprobs": [float("nan")]}, {"token_ids": [19], "logprobs": [-4.0]}, "finite"),
    ],
)
def test_reference_comparison_rejects_token_length_and_nan_errors(actual, expected, message):
    with pytest.raises(AssertionError, match=message):
        qualification.assert_reference_record(actual, expected, "request")


@pytest.mark.host
def test_comparison_artifacts_sanitize_labels_and_reject_duplicate_ownership(tmp_path):
    record = qualification.completion_record(_completion())
    qualification.record_reference_comparison(record, record, "../abort/reuse replacement", tmp_path)
    artifact = tmp_path / "abort-reuse-replacement.json"
    original = artifact.read_text()
    assert json.loads(original)["max_abs_difference"] == 0
    with pytest.raises(FileExistsError):
        qualification.record_reference_comparison(record, record, "abort reuse replacement", tmp_path)
    assert artifact.read_text() == original
    assert list(tmp_path.iterdir()) == [artifact]


@pytest.mark.host
def test_server_clears_owned_comparisons_before_a_failed_startup(tmp_path, monkeypatch):
    directory = tmp_path / "B2-K1"
    comparisons = directory / "comparisons"
    comparisons.mkdir(parents=True)
    (comparisons / "concurrent-request-0.json").write_text('{"max_abs_difference":0}')
    retained = directory / "retained-diagnostic.txt"
    retained.write_text("diagnostics")

    @contextmanager
    def unavailable_socket():
        yield SimpleNamespace(bind=Mock(side_effect=OSError("port is occupied")))

    monkeypatch.setattr(qualification, "engine_processes", lambda: [])
    monkeypatch.setattr(qualification.socket, "socket", unavailable_socket)
    with (
        pytest.raises(OSError, match="port is occupied"),
        qualification.server(SimpleNamespace(output=tmp_path, port=8123), 2, 1),
    ):
        pytest.fail("An occupied port must prevent server startup")
    assert list(comparisons.iterdir()) == []
    assert retained.read_text() == "diagnostics"


@pytest.mark.host
def test_control_comparison_preserves_actual_expected_and_differences_before_failure(tmp_path):
    expected = qualification.completion_record(_completion())
    result = _completion(logprobs=[-4.0, -3.25, -2.0])
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [19] if text == " red" else [20])
    client = SimpleNamespace(complete=Mock(return_value=result))
    with pytest.raises(AssertionError, match="raw logprobs differ"):
        qualification.exercise_controls(
            client,
            tokenizer,
            [19] * 96,
            {"penalties": expected},
            comparison_directory=tmp_path,
        )
    artifact = json.loads((tmp_path / "penalties.json").read_text())
    assert artifact["actual"] == qualification.completion_record(result)
    assert artifact["expected"] == expected
    assert artifact["differences"] == [0.0, 0.25, 0.0]
    assert artifact["max_abs_difference"] == 0.25


@pytest.mark.host
def test_pressure_workload_has_distinct_histories_matching_lengths_and_logprobs():
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [ord(character) for character in text])
    requests = qualification.pressure_requests(tokenizer, 700)
    assert len(requests) == 4
    assert len({tuple(prompt[:168]) for prompt, _ in requests}) == 4
    assert all(len(prompt) == 200 and prompt[-32:] == [700] * 32 for prompt, _ in requests)
    assert [controls["seed"] for _, controls in requests] == [901, 902, 903, 904]
    assert all(controls["max_tokens"] == 160 for _, controls in requests)
    assert all(controls["allowed_token_ids"] == [700] for _, controls in requests)
    assert all(controls["logprobs"] == 1 and controls["return_tokens_as_token_ids"] for _, controls in requests)


def _source_checkout(tmp_path, import_root):
    checkout = tmp_path / "checkout"
    package = checkout / "src" / import_root
    package.mkdir(parents=True)
    origin = package / "__init__.py"
    origin.write_text("VERSION = 1\n")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Qualification Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Initial test source",
        ],
        check=True,
    )
    return checkout, SimpleNamespace(__file__=str(origin), __version__="1.0")


@pytest.mark.host
@pytest.mark.parametrize("import_root", ["tt_transformers", "vllm_tt_plugin"])
def test_source_provenance_rejects_import_from_a_different_checkout(tmp_path, import_root):
    checkout = tmp_path / "expected"
    wrong = SimpleNamespace(__file__=str(tmp_path / "other" / "src" / import_root / "__init__.py"))
    with pytest.raises(ValueError, match="does not belong"):
        qualification.source_provenance(wrong, checkout, import_root)


@pytest.mark.host
def test_source_provenance_records_source_exposure_without_distribution_metadata(tmp_path, monkeypatch):
    checkout, module = _source_checkout(tmp_path, "tt_transformers")
    monkeypatch.setattr(
        qualification.importlib.metadata, "version", Mock(side_effect=importlib.metadata.PackageNotFoundError)
    )
    source = qualification.source_provenance(module, checkout, "tt_transformers")
    assert source["module_origin"] == str(Path(module.__file__).resolve())
    assert source["exposure"] == "source checkout"
    assert source["source_version"] == "1.0"
    assert source["installed_distribution_version"] is None
    assert source["dirty"] is False
    assert len(source["sha"]) == 40
    assert len(source["fingerprint"]) == 64


@pytest.mark.host
def test_source_fingerprint_distinguishes_tracked_and_untracked_edits_at_same_head(tmp_path):
    checkout, module = _source_checkout(tmp_path, "tt_transformers")
    clean = qualification.source_provenance(module, checkout, "tt_transformers")
    Path(module.__file__).write_text("VERSION = 2\n")
    tracked = qualification.source_provenance(module, checkout, "tt_transformers")
    (checkout / "src/tt_transformers/new_adapter.py").write_text("VALUE = 3\n")
    untracked = qualification.source_provenance(module, checkout, "tt_transformers")
    assert clean["sha"] == tracked["sha"] == untracked["sha"]
    assert clean["dirty"] is False
    assert tracked["dirty"] is untracked["dirty"] is True
    assert len({source["fingerprint"] for source in (clean, tracked, untracked)}) == 3
    assert clean["patch"] != tracked["patch"]
    assert "src/tt_transformers/new_adapter.py" in untracked["untracked"]


@pytest.mark.host
def test_runtime_provenance_does_not_claim_source_checkout_is_verified_binary_revision(tmp_path):
    checkout, _ = _source_checkout(tmp_path, "tt_transformers")
    runtime_root = tmp_path / "runtime-root"
    source = qualification.runtime_provenance(runtime_root, checkout)
    assert source["source_checkout_revision"] == qualification.git_revision(checkout)
    assert source["runtime_root"] == str(runtime_root.resolve())
    assert source["source_checkout"] == str(checkout.resolve())
    assert source["official_commit"] is None
    assert source["official_commit_verified"] is False
    assert source["revision_status"].startswith("unverified:")


@pytest.mark.host
def test_runtime_provenance_records_selected_manifest_and_separate_runtime_root(tmp_path, monkeypatch):
    runtime_root = tmp_path / "runtime-root"
    manifest = tmp_path / "bundle/runtime.json"
    (tmp_path / "environment.json").write_text(
        json.dumps({"runtime_root": str(runtime_root), "runtime": str(manifest)})
    )
    from tools.ttnn_dev_support import common

    verified = Mock(return_value={"metal": {"sha": "c" * 40}, "mode": "wheel", "fingerprint": "d" * 64})
    monkeypatch.setattr(common, "load_runtime", verified)
    source = qualification.runtime_provenance(runtime_root, None)
    verified.assert_called_once_with(manifest)
    assert source["official_commit"] == "c" * 40
    assert source["official_commit_verified"] is True
    assert source["runtime_manifest"] == str(manifest.resolve())
    assert source["runtime_fingerprint"] == "d" * 64
    assert source["runtime_mode"] == "wheel"


@pytest.mark.host
def test_runtime_provenance_rejects_mismatched_manifest_and_source_checkout(tmp_path, monkeypatch):
    checkout, _ = _source_checkout(tmp_path, "tt_transformers")
    from tools.ttnn_dev_support import common

    monkeypatch.setattr(common, "load_runtime", Mock(return_value={"metal": {"sha": "e" * 40}}))
    with pytest.raises(ValueError, match="different tt-metal revisions"):
        qualification.runtime_provenance(tmp_path / "runtime-root", checkout, tmp_path / "runtime.json")


@pytest.fixture
def http_qualification(tmp_path, monkeypatch):
    state = {"k": 0, "corrupt": None, "initial_verifications": 128, "probe_calls": 0}
    args = SimpleNamespace(output=tmp_path, port=8123)
    tokenizer = SimpleNamespace(
        encode=lambda text, **kwargs: [700] if text == " red" else [ord(character) for character in text],
        eos_token_id=999,
    )

    @contextmanager
    def running_server(args, batch, k):
        state["k"] = k
        state["batch"] = batch
        state["first_request_calls"] = 0
        state["preemptions"] = 0
        directory = args.output / f"B{batch}-K{k}"
        directory.mkdir(parents=True, exist_ok=True)
        qualification.reset_reference_comparisons(directory)
        (directory / "backend-preempt.jsonl").unlink(missing_ok=True)
        log = directory / "server.log"
        log.write_text(f"{state['initial_verifications']} verify in logits mode\n")
        yield directory, log

    class HTTPClient:
        def __init__(self, *args):
            pass

        def complete(self, prompt, **controls):
            count = controls["max_tokens"]
            if controls.get("seed") == 1171:
                state["probe_calls"] += 1
                (args.output / f"B{state['batch']}-K{state['k']}" / "server.log").write_text(
                    "128 verify in logits mode\n"
                )
            token = controls.get("allowed_token_ids", [123])[0]
            values = [-1.0 - (sum(prompt) + index * token) * 1e-6 for index in range(count)]
            if controls.get("seed") == 101:
                state["first_request_calls"] += 1
                phase = "replay" if state["first_request_calls"] > 1 else None
            elif controls.get("seed") == 102:
                phase = "concurrent"
            elif controls.get("seed") == 902:
                phase = "pressure"
            elif prompt == list(range(300, 331)):
                phase = "abort_reuse"
            else:
                phase = None
            if phase is not None and state["corrupt"] == phase and (phase != "pressure" or state["k"] == 3):
                values[-1] -= 0.5
            result = _completion([token] * count, values)
            result.body["id"] = f"cmpl-fixture-{controls.get('seed', 0)}"
            if controls.get("seed") == 901 and count == 160:
                state["preemptions"] += 1
                history = prompt + [token] * 8
                event = {
                    "request_id": result.body["id"] + "-0",
                    "all_token_ids": history,
                    "post_all_token_ids": history,
                    "num_prompt_tokens": len(prompt),
                    "num_output_tokens": 8,
                    "num_computed_tokens": len(history) - 1,
                    "num_preemptions": 0,
                    "num_in_flight_tokens": 0,
                    "num_output_placeholders": 0,
                    "num_stale_output_tokens": 0,
                    "drop_stale_output": False,
                    "post_status": "PREEMPTED",
                    "post_num_computed_tokens": 0,
                    "post_num_preemptions": 1,
                }
                with (args.output / f"B{state['batch']}-K{state['k']}" / "backend-preempt.jsonl").open("a") as output:
                    output.write(json.dumps(event) + "\n")
            result.completion_tokens = count
            result.finish_reason = "stop" if token == 999 else "length"
            return result

        def metrics(self):
            return SimpleNamespace(raw="metrics\n", probe_calls=state["probe_calls"], preemptions=state["preemptions"])

        def stream_and_abandon(self, *args, **kwargs):
            return 2

    def acceptance_delta(before, after, *args):
        return SimpleNamespace(
            drafts=1 if state["k"] else 0,
            draft_tokens=3 if state["k"] else 0,
            accepted=3 if state["k"] else 0,
            preemptions=after.preemptions - before.preemptions,
            as_dict=lambda: {
                "preemptions": after.preemptions - before.preemptions,
                "probe_requests": after.probe_calls - before.probe_calls,
            },
        )

    monkeypatch.setattr(qualification, "server", running_server)
    monkeypatch.setattr(
        qualification, "exercise_controls", lambda *args, **kwargs: {"serial_controls": "checked separately"}
    )
    monkeypatch.setattr(qualification, "exercise_abort", lambda *args: {"backend": "checked by observer contracts"})
    helpers = HTTPClient, acceptance_delta, _assert_full_length_completion, lambda log: 4
    return args, tokenizer, helpers, state


@pytest.mark.host
def test_http_qualification_preserves_same_batch_references_for_all_lifecycle_workloads(http_qualification):
    args, tokenizer, helpers, _ = http_qualification
    ordinary = qualification.exercise_http(args, 4, 0, tokenizer, helpers)
    speculative = qualification.exercise_http(args, 4, 3, tokenizer, helpers, ordinary)
    assert len(ordinary["concurrent"]) == len(speculative["concurrent"]) == 20
    assert len(ordinary["pressure"]["records"]) == len(speculative["pressure"]["records"]) == 4
    assert ordinary["abort_reuse"]["token_ids"] == speculative["abort_reuse"]["token_ids"]
    assert ordinary["abort_reuse"]["logprobs"] == speculative["abort_reuse"]["logprobs"]


@pytest.mark.host
def test_http_qualification_records_probe_and_includes_probe_metrics_without_changing_references(http_qualification):
    args, tokenizer, helpers, state = http_qualification
    ordinary = qualification.exercise_http(args, 2, 0, tokenizer, helpers)
    state["initial_verifications"] = 0
    speculative = qualification.exercise_http(args, 2, 2, tokenizer, helpers, ordinary)
    saved = json.loads((args.output / "B2-K2" / "results.json").read_text())
    assert state["probe_calls"] == 1
    assert saved["verification_probe"]["prompt"] == [700] * 96
    assert saved["verification_probe"]["token_ids"] == [700] * 512
    assert saved["verification_probe"]["executed_logits_verifications"] == 128
    assert saved["acceptance"]["probe_requests"] == 1
    assert saved["completions"] == ordinary["concurrent"] == speculative["concurrent"]
    assert len(speculative["concurrent"]) == 20
    assert json.loads((args.output / "B2-K0" / "results.json").read_text())["verification_probe"] is None


@pytest.mark.host
@pytest.mark.parametrize("phase", ["concurrent", "pressure", "abort_reuse", "replay"])
def test_http_qualification_rejects_corrupt_state_despite_identical_constrained_tokens(http_qualification, phase):
    args, tokenizer, helpers, state = http_qualification
    ordinary = qualification.exercise_http(args, 4, 0, tokenizer, helpers)
    state["corrupt"] = phase
    with pytest.raises(AssertionError, match="raw logprobs differ"):
        qualification.qualify_http(args, 4, 3, tokenizer, helpers, ordinary)
    filenames = {
        "concurrent": "concurrent-request-1.json",
        "pressure": "pressure-request-1-outputs-0-to-160.json",
        "abort_reuse": "abort-reuse-replacement.json",
        "replay": "replay-request-0.json",
    }
    directory = args.output / "B4-K3-pressure-reference" / "B4-K0" if phase == "pressure" else args.output / "B4-K3"
    artifact = json.loads((directory / "comparisons" / filenames[phase]).read_text())
    assert artifact["actual"]["token_ids"] == artifact["expected"]["token_ids"]
    assert artifact["actual"]["logprobs"] != artifact["expected"]["logprobs"]
    assert artifact["differences"][:-1] == [0.0] * (len(artifact["differences"]) - 1)
    assert artifact["differences"][-1] == pytest.approx(0.5)
    assert artifact["max_abs_difference"] == pytest.approx(0.5)


@pytest.mark.host
def test_ordinary_http_replay_checks_raw_logprobs_and_preserves_failure(http_qualification):
    args, tokenizer, helpers, state = http_qualification
    state["corrupt"] = "replay"
    with pytest.raises(AssertionError, match="raw logprobs differ"):
        qualification.exercise_http(args, 2, 0, tokenizer, helpers)
    artifact = json.loads((args.output / "B2-K0" / "comparisons" / "replay-request-0.json").read_text())
    assert artifact["actual"]["token_ids"] == artifact["expected"]["token_ids"]
    assert artifact["max_abs_difference"] == pytest.approx(0.5)
    assert not (args.output / "B2-K0" / "results.json").exists()


@pytest.mark.host
def test_http_rerun_removes_old_comparisons_before_recording_a_new_failure(http_qualification):
    args, tokenizer, helpers, state = http_qualification
    ordinary = qualification.exercise_http(args, 4, 0, tokenizer, helpers)
    qualification.exercise_http(args, 4, 3, tokenizer, helpers, ordinary)
    comparisons = args.output / "B4-K3" / "comparisons"
    assert (comparisons / "replay-request-0.json").exists()
    state["corrupt"] = "concurrent"
    with pytest.raises(AssertionError, match="raw logprobs differ"):
        qualification.exercise_http(args, 4, 3, tokenizer, helpers, ordinary)
    assert {artifact.name for artifact in comparisons.iterdir()} == {
        "concurrent-request-0.json",
        "concurrent-request-1.json",
    }
    artifact = json.loads((comparisons / "concurrent-request-1.json").read_text())
    assert artifact["max_abs_difference"] == pytest.approx(0.5)


def _observed_request():
    return SimpleNamespace(
        request_id="cmpl-llama-ngram-abort-abc-0-1234abcd",
        status=SimpleNamespace(name="FINISHED_ABORTED"),
        num_output_tokens=4,
        max_tokens=512,
    )


def _released_scheduler():
    return SimpleNamespace(requests={}, running=[], waiting=[], skipped_waiting=[])


@pytest.mark.host
def test_finish_observer_preserves_original_arguments_return_value_and_backend_status(tmp_path):
    event_path = tmp_path / "backend-finish.jsonl"
    request = _observed_request()

    class Scheduler:
        def __init__(self):
            self.requests = {request.request_id: request}
            self.running = [request]
            self.waiting = []
            self.skipped_waiting = []
            self.returned = [request]

        def finish_requests(self, request_ids, status, *, marker):
            self.received = request_ids, status, marker
            request.status = status
            self.requests.pop(request.request_id)
            self.running.remove(request)
            return self.returned

    qualification.install_abort_observer(event_path, Scheduler)
    scheduler = Scheduler()
    request_ids = [request.request_id]
    status = request.status
    marker = object()

    result = scheduler.finish_requests(request_ids, status, marker=marker)

    assert result is scheduler.returned
    assert scheduler.received[0] is request_ids
    assert scheduler.received[1] is status
    assert scheduler.received[2] is marker
    event = json.loads(event_path.read_text())
    qualification.assert_backend_abort(event, "cmpl-llama-ngram-abort-abc", 512)
    assert event["generated_tokens"] == 4
    assert event["owned_by_requests"] is event["owned_by_running"] is event["owned_by_waiting"] is False


@pytest.mark.host
def test_finish_observer_does_not_hide_original_failure(tmp_path):
    failure = RuntimeError("original scheduler failure")

    class Scheduler:
        def finish_requests(self, *args, **kwargs):
            raise failure

    event_path = tmp_path / "backend-finish.jsonl"
    qualification.install_abort_observer(event_path, Scheduler)
    with pytest.raises(RuntimeError) as raised:
        Scheduler().finish_requests(["request"], marker=4)
    assert raised.value is failure
    assert not event_path.exists()


@pytest.mark.host
def test_finish_observer_io_failure_does_not_change_original_return(tmp_path, monkeypatch, capsys):
    request = _observed_request()
    returned = [request]

    class Scheduler:
        requests = {}
        running = []
        waiting = []
        skipped_waiting = []

        def finish_requests(self):
            return returned

    qualification.install_abort_observer(tmp_path / "backend-finish.jsonl", Scheduler)
    monkeypatch.setattr(qualification.os, "open", Mock(side_effect=OSError("cannot record evidence")))
    assert Scheduler().finish_requests() is returned
    assert "LLAMA_ABORT_OBSERVER_ERROR" in capsys.readouterr().err


def _preemptible_request():
    return SimpleNamespace(
        request_id="cmpl-preempt-0",
        all_token_ids=[11, 12, 19, 20],
        num_prompt_tokens=2,
        num_output_tokens=2,
        num_computed_tokens=3,
        num_preemptions=1,
        num_in_flight_tokens=2,
        num_output_placeholders=1,
        num_stale_output_tokens=1,
        status=SimpleNamespace(name="RUNNING"),
        spec_token_ids=[999],
    )


@pytest.mark.host
@pytest.mark.parametrize("call_style", ["positional", "keyword", "default", "positional_drop"])
def test_preempt_observer_preserves_call_and_snapshots_committed_history_before_mutation(tmp_path, call_style):
    request = _preemptible_request()
    event_path = tmp_path / "backend-preempt.jsonl"
    returned = object()
    calls = []

    class Scheduler:
        def _preempt_request(self, *args, **kwargs):
            assert not event_path.exists()
            calls.append((args, kwargs))
            request.all_token_ids.append(21)
            request.num_output_tokens += 1
            request.spec_token_ids.clear()
            request.num_computed_tokens = 0
            request.num_preemptions += 1
            request.num_in_flight_tokens = 0
            request.num_output_placeholders = 0
            request.num_stale_output_tokens = 0
            request.status.name = "PREEMPTED"
            return returned

    qualification.install_preempt_observer(event_path, Scheduler)
    installed = Scheduler._preempt_request
    qualification.install_preempt_observer(event_path, Scheduler)
    assert Scheduler._preempt_request is installed
    scheduler = Scheduler()
    args = () if call_style == "keyword" else (request, 12.5)
    kwargs = {"drop_stale_output": False}
    if call_style == "keyword":
        kwargs.update(request=request, timestamp=12.5)
    elif call_style == "default":
        kwargs = {}
    elif call_style == "positional_drop":
        args = (request, 12.5, True)
        kwargs = {}
    assert scheduler._preempt_request(*args, **kwargs) is returned
    assert len(calls) == 1
    assert len(calls[0][0]) == len(args)
    assert all(actual is expected for actual, expected in zip(calls[0][0], args))
    assert calls[0][1].keys() == kwargs.keys()
    assert all(calls[0][1][name] is value for name, value in kwargs.items())
    events = event_path.read_text().splitlines()
    assert len(events) == 1
    event = json.loads(events[0])
    assert event == {
        "request_id": request.request_id,
        "all_token_ids": [11, 12, 19, 20],
        "num_prompt_tokens": 2,
        "num_output_tokens": 2,
        "num_computed_tokens": 3,
        "num_preemptions": 1,
        "num_in_flight_tokens": 2,
        "num_output_placeholders": 1,
        "num_stale_output_tokens": 1,
        "drop_stale_output": call_style == "positional_drop",
        "post_all_token_ids": [11, 12, 19, 20, 21],
        "post_status": "PREEMPTED",
        "post_num_computed_tokens": 0,
        "post_num_preemptions": 2,
    }


@pytest.mark.host
def test_preempt_observer_preserves_original_exception_and_emits_no_completed_event(tmp_path):
    failure = RuntimeError("original preemption failure")
    original = Mock(side_effect=failure)

    class Scheduler:
        def _preempt_request(self, *args, **kwargs):
            return original(self, *args, **kwargs)

    event_path = tmp_path / "backend-preempt.jsonl"
    qualification.install_preempt_observer(event_path, Scheduler)
    scheduler = Scheduler()
    request = _preemptible_request()
    with pytest.raises(RuntimeError) as raised:
        scheduler._preempt_request(request, 12.5, drop_stale_output=True)
    assert raised.value is failure
    original.assert_called_once_with(scheduler, request, 12.5, drop_stale_output=True)
    assert not event_path.exists()


@pytest.mark.host
@pytest.mark.parametrize("broken_error_log", [False, True])
def test_preempt_observer_snapshot_failure_always_calls_original(tmp_path, monkeypatch, capsys, broken_error_log):
    returned = object()
    original = Mock(return_value=returned)

    class Scheduler:
        def _preempt_request(self, *args, **kwargs):
            return original(self, *args, **kwargs)

    event_path = tmp_path / "backend-preempt.jsonl"
    qualification.install_preempt_observer(event_path, Scheduler)
    monkeypatch.setattr(qualification, "backend_preempt_event", Mock(side_effect=RuntimeError("cannot snapshot")))
    if broken_error_log:
        monkeypatch.setattr(qualification, "print", Mock(side_effect=OSError("cannot report")), raising=False)
    scheduler = Scheduler()
    request = _preemptible_request()
    assert scheduler._preempt_request(request, timestamp=12.5) is returned
    original.assert_called_once_with(scheduler, request, timestamp=12.5)
    assert not event_path.exists()
    if not broken_error_log:
        assert "LLAMA_PREEMPT_OBSERVER_ERROR" in capsys.readouterr().err


@pytest.mark.host
@pytest.mark.parametrize("operation", ["open", "write"])
def test_preempt_observer_io_failure_preserves_original_return_and_call(tmp_path, monkeypatch, capsys, operation):
    returned = object()
    original = Mock(return_value=returned)

    class Scheduler:
        def _preempt_request(self, request, *args, **kwargs):
            request.status.name = "PREEMPTED"
            request.num_computed_tokens = 0
            request.num_preemptions += 1
            return original(self, request, *args, **kwargs)

    event_path = tmp_path / "backend-preempt.jsonl"
    qualification.install_preempt_observer(event_path, Scheduler)
    monkeypatch.setattr(qualification.os, operation, Mock(side_effect=OSError("cannot record evidence")))
    scheduler = Scheduler()
    request = _preemptible_request()
    assert scheduler._preempt_request(request, 12.5) is returned
    original.assert_called_once_with(scheduler, request, 12.5)
    assert not event_path.exists() or event_path.read_text() == ""
    assert "LLAMA_PREEMPT_OBSERVER_ERROR" in capsys.readouterr().err


@pytest.mark.host
def test_preempt_observer_snapshot_defaults_absent_async_state_to_zero():
    request = _preemptible_request()
    for name in ("num_in_flight_tokens", "num_output_placeholders", "num_stale_output_tokens"):
        delattr(request, name)
    event = qualification.backend_preempt_event(request)
    assert event["num_in_flight_tokens"] == event["num_output_placeholders"] == event["num_stale_output_tokens"] == 0


@pytest.mark.host
@pytest.mark.parametrize("status", ["FINISHED_LENGTH_CAPPED", "FINISHED_STOPPED"])
def test_backend_cancellation_oracle_rejects_normal_completion_after_disconnect(tmp_path, status):
    request = _observed_request()
    request.status.name = status
    event = qualification.backend_finish_event(_released_scheduler(), request)
    event_path = tmp_path / "backend-finish.jsonl"
    event_path.write_text(json.dumps(event) + "\n")
    with pytest.raises(AssertionError, match="finished normally"):
        qualification.wait_for_backend_abort(event_path, "cmpl-llama-ngram-abort-abc", 512, timeout=0)


@pytest.mark.host
@pytest.mark.parametrize("owner", ["requests", "running", "waiting", "skipped_waiting"])
def test_backend_cancellation_oracle_rejects_retained_request_ownership(owner):
    request = _observed_request()
    scheduler = _released_scheduler()
    setattr(scheduler, owner, {request.request_id: request} if owner == "requests" else [request])
    event = qualification.backend_finish_event(scheduler, request)
    with pytest.raises(AssertionError, match="retains scheduler ownership"):
        qualification.assert_backend_abort(event, "cmpl-llama-ngram-abort-abc", 512)


@pytest.mark.host
def test_backend_cancellation_oracle_rejects_request_that_reached_output_limit():
    request = _observed_request()
    request.num_output_tokens = 512
    event = qualification.backend_finish_event(_released_scheduler(), request)
    with pytest.raises(AssertionError, match="before its requested output length"):
        qualification.assert_backend_abort(event, "cmpl-llama-ngram-abort-abc", 512)


@pytest.mark.host
def test_backend_cancellation_oracle_rejects_unrelated_request_id():
    event = qualification.backend_finish_event(_released_scheduler(), _observed_request())
    with pytest.raises(AssertionError, match="different stream request"):
        qualification.assert_backend_abort(event, "cmpl-another-request", 512)


@pytest.mark.host
def test_backend_cancellation_requires_an_actual_matching_event(tmp_path):
    with pytest.raises(AssertionError, match="No matching backend cancellation"):
        qualification.wait_for_backend_abort(tmp_path / "missing.jsonl", "cmpl-llama-ngram-abort-abc", 512, timeout=0)


@pytest.mark.host
def test_backend_cancellation_uses_complete_event_and_ignores_partial_append(tmp_path):
    event = qualification.backend_finish_event(_released_scheduler(), _observed_request())
    event_path = tmp_path / "backend-finish.jsonl"
    event_path.write_text(json.dumps(event) + '\n{"request_id":')
    assert qualification.wait_for_backend_abort(event_path, "cmpl-llama-ngram-abort-abc", 512, timeout=0) == event


@pytest.mark.host
def test_child_observer_environment_is_private_and_resets_backend_evidence(tmp_path):
    original = {"PYTHONPATH": "/existing/source", "OTHER_SETTING": "preserved"}
    (tmp_path / "backend-finish.jsonl").write_text("stale backend event\n")
    (tmp_path / "backend-preempt.jsonl").write_text("stale preemption event\n")
    environment = qualification.observer_environment(tmp_path, original)
    assert original["PYTHONPATH"] == "/existing/source"
    assert environment["PYTHONPATH"] == str(tmp_path / "observer") + qualification.os.pathsep + "/existing/source"
    assert environment["OTHER_SETTING"] == "preserved"
    assert not (tmp_path / "backend-finish.jsonl").exists()
    assert not (tmp_path / "backend-preempt.jsonl").exists()
    observer = tmp_path / "observer/sitecustomize.py"
    compile(observer.read_text(), str(observer), "exec")
    assert str(tmp_path / "backend-finish.jsonl") in observer.read_text()
    assert str(tmp_path / "backend-preempt.jsonl") in observer.read_text()
    assert "_observer.install_preempt_observer" in observer.read_text()
    assert str(_SCRIPT) in observer.read_text()


@pytest.fixture
def named_stream(monkeypatch):
    monkeypatch.setattr(qualification.uuid, "uuid4", lambda: SimpleNamespace(hex="abc"))
    response_id = "cmpl-llama-ngram-abort-abc"
    events = [
        {"id": response_id, "choices": [{"token_ids": [19], "finish_reason": None}]},
        {"id": response_id, "choices": [{"token_ids": [20, 21], "finish_reason": None}]},
    ]
    sent = {}

    @contextmanager
    def stream(method, url, *, json, timeout):
        sent.update(json)
        response = SimpleNamespace(
            raise_for_status=lambda: None,
            iter_lines=lambda: iter(
                "data: " + (event if isinstance(event, str) else qualification.json.dumps(event)) for event in events
            ),
        )
        yield response

    monkeypatch.setattr(qualification.httpx, "stream", stream)
    client = SimpleNamespace(
        base_url="http://127.0.0.1:8123",
        request_body=lambda prompt, **controls: {"prompt": prompt, "return_token_ids": True, **controls},
    )
    return client, events, sent


@pytest.mark.host
def test_named_stream_disconnects_after_two_unfinished_token_chunks(named_stream):
    client, _, sent = named_stream
    result = qualification.disconnect_named_stream(client, [1, 2, 3], 512)
    assert result == {
        "response_id": "cmpl-llama-ngram-abort-abc",
        "token_chunks": 2,
        "tokens_seen": 3,
        "max_tokens": 512,
    }
    assert sent["request_id"] == "llama-ngram-abort-abc"
    assert sent["ignore_eos"] is True
    assert sent["stop"] == sent["stop_token_ids"] == []


@pytest.mark.host
@pytest.mark.parametrize("terminal", ["done", "length", "stop"])
def test_named_stream_rejects_completion_before_client_disconnect(named_stream, terminal):
    client, events, _ = named_stream
    if terminal == "done":
        events[1] = "[DONE]"
    else:
        events[1]["choices"][0]["finish_reason"] = terminal
    with pytest.raises(AssertionError, match="finished before client cancellation"):
        qualification.disconnect_named_stream(client, [1, 2, 3], 512)
