# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qualify ordinary serving, real target verification, and CPU n-gram serving.

Each server inherits a private Scheduler.finish_requests observer. The observer
calls the original Scheduler.finish_requests unchanged and records the returned request's status
and scheduler ownership. Backend cancellation requires FINISHED_ABORTED before
the requested output length and released scheduler ownership. Frontend abort
logs and counters alone do not establish backend cancellation in vLLM 0.30.

If the concurrent workload has not reached the plugin's 128-submission log
interval, one constrained 512-token request supplies at least 128 target
verifications for K <= 3. Qualification still requires a positive executed
logits-verification count from the plugin log.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import httpx

MODEL = "meta-llama/Llama-3.2-1B"
ADAPTER = "tt_transformers.models.llama32_1b.vllm_generator:Llama32_1BGenerator"
ROOT = Path(__file__).resolve().parents[2]


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _timestamped_junit_reports(expected):
    pattern = re.compile(rf"{re.escape(expected.stem)}_(\d{{8}}_\d{{6}}){re.escape(expected.suffix)}")
    reports = []
    for candidate in expected.parent.glob(f"{expected.stem}_*{expected.suffix}"):
        match = pattern.fullmatch(candidate.name)
        if not candidate.is_file() or match is None:
            continue
        try:
            datetime.strptime(match[1], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        reports.append(candidate)
    return sorted(reports)


def canonicalize_junit_report(expected):
    expected = Path(expected)
    candidates = ([expected] if expected.is_file() else []) + _timestamped_junit_reports(expected)
    if not candidates:
        raise FileNotFoundError(f"No JUnit report for {expected}")
    if len(candidates) != 1:
        raise ValueError(f"Ambiguous JUnit reports for {expected}: {candidates}")
    ET.parse(candidates[0])
    if candidates[0] != expected:
        shutil.copyfile(candidates[0], expected)
    return expected


def reset_completion_markers(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("success.json", "partial.json", "device-matrix.xml"):
        (directory / filename).unlink(missing_ok=True)
    for report in _timestamped_junit_reports(directory / "device-matrix.xml"):
        report.unlink()


def require_commit_sha(name, value):
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} must be a full commit SHA")


def require_cached_checkpoint(revision, lookup):
    require_commit_sha("checkpoint_revision", revision)
    config_path = lookup(MODEL, "config.json", revision="main")
    if not isinstance(config_path, str) or Path(config_path).parent.name != revision:
        raise ValueError("Populate an isolated HF cache whose base-checkpoint refs/main matches --checkpoint-revision")
    return config_path


def launch_settings(args, batch, k):
    if batch not in (1, 2, 4) or k not in (0, 1, 2, 3):
        raise ValueError("The qualification launch requires B=1,2,4 and K=0,1,2,3")
    require_commit_sha("checkpoint_revision", args.checkpoint_revision)
    command = [
        sys.executable,
        str(args.plugin_root / "examples/server_example_tt.py"),
        "--model",
        MODEL,
        "--revision",
        args.checkpoint_revision,
        "--max_num_seqs",
        str(batch),
        "--max_model_len",
        "1024",
        "--block_size",
        "32",
        "--port",
        str(args.port),
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--additional-config",
        json.dumps({"tt": {"trace_mode": "none", "optimizations": "performance"}}),
    ]
    if k:
        command += [
            "--speculative-config",
            json.dumps(
                {"method": "ngram", "num_speculative_tokens": k, "prompt_lookup_min": 2, "prompt_lookup_max": 4}
            ),
        ]
    environment = os.environ.copy()
    environment.update(
        TT_MODEL_CLASS_OVERRIDES=f"TTLlamaForCausalLM={ADAPTER}",
        MESH_DEVICE="(1, 1)",
        PYTHONUNBUFFERED="1",
    )
    return command, environment


def package_source(name):
    distribution = importlib.metadata.distribution(name)
    raw = distribution.read_text("direct_url.json")
    direct_url = json.loads(raw) if raw else {}
    revision = direct_url.get("vcs_info", {}).get("commit_id")
    return {
        "version": distribution.version,
        "source_revision": revision,
        "revision_verified": bool(revision),
        "source": direct_url,
        "revision_status": "recorded by installed VCS metadata"
        if revision
        else "unverified: installed package has no VCS commit metadata",
    }


def source_provenance(module, checkout, import_root):
    checkout = Path(checkout).resolve()
    origin = getattr(module, "__file__", None)
    expected = checkout / "src" / import_root
    if not origin or not Path(origin).resolve().is_relative_to(expected.resolve()):
        raise ValueError(f"Loaded {import_root} origin {origin!r} does not belong to {expected}")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.ttnn_dev_support.common import source_state

    try:
        distribution_version = importlib.metadata.version(import_root.replace("_", "-"))
    except importlib.metadata.PackageNotFoundError:
        distribution_version = None
    return {
        "checkout": str(checkout),
        "module_origin": str(Path(origin).resolve()),
        "exposure": "source checkout",
        "source_version": getattr(module, "__version__", None),
        "installed_distribution_version": distribution_version,
        **source_state(checkout),
    }


def runtime_provenance(runtime_root, source_checkout, manifest=None):
    result = {
        "runtime_root": str(Path(runtime_root).resolve()) if runtime_root else None,
        "source_checkout": str(Path(source_checkout).resolve()) if source_checkout else None,
        "source_checkout_revision": git_revision(source_checkout) if source_checkout else None,
        "official_commit": None,
        "official_commit_verified": False,
        "revision_status": "unverified: no selected runtime manifest was found",
    }
    if manifest is None and runtime_root:
        environment_path = Path(runtime_root).resolve().parent / "environment.json"
        if environment_path.is_file():
            environment = json.loads(environment_path.read_text())
            if Path(environment["runtime_root"]).resolve() != Path(runtime_root).resolve():
                raise ValueError("The TTNN environment does not match TT_METAL_RUNTIME_ROOT")
            manifest = Path(environment["runtime"])
    if manifest is not None:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from tools.ttnn_dev_support.common import load_runtime

        runtime = load_runtime(Path(manifest))
        official_commit = runtime["metal"]["sha"]
        require_commit_sha("runtime metal revision", official_commit)
        if result["source_checkout_revision"] and result["source_checkout_revision"] != official_commit:
            raise ValueError("The TTNN runtime manifest and TT_METAL_HOME identify different tt-metal revisions")
        result.update(
            runtime_manifest=str(Path(manifest).resolve()),
            runtime_mode=runtime["mode"],
            runtime_fingerprint=runtime["fingerprint"],
            official_commit=official_commit,
            official_commit_verified=True,
            revision_status="recorded by the selected verified runtime manifest",
        )
    return result


def require_device_matrix(path):
    cases = list(ET.parse(path).iter("testcase"))
    if len(cases) != 18 or any(case.find(tag) is not None for case in cases for tag in ("skipped", "failure", "error")):
        raise AssertionError("Real Llama qualification requires all 18 B/K device tests to pass without skips")


def engine_processes():
    result = subprocess.run(["pgrep", "-f", "VLLM::EngineCore"], capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"Cannot inspect existing TT engines: {result.stderr}")
    return result.stdout.split()


def backend_finish_event(scheduler, request):
    request_id = request.request_id
    waiting = (*scheduler.waiting, *scheduler.skipped_waiting)
    return {
        "request_id": request_id,
        "finished_status": request.status.name,
        "generated_tokens": request.num_output_tokens,
        "max_tokens": request.max_tokens,
        "owned_by_requests": request_id in scheduler.requests,
        "owned_by_running": any(item.request_id == request_id for item in scheduler.running),
        "owned_by_waiting": any(item.request_id == request_id for item in waiting),
        "num_running": len(scheduler.running),
        "num_waiting": len(waiting),
    }


def install_abort_observer(event_path, scheduler_class=None):
    if scheduler_class is None:
        from vllm.v1.core.sched.scheduler import Scheduler

        scheduler_class = Scheduler
    original = scheduler_class.finish_requests
    if getattr(original, "_llama_ngram_observer", False):
        return

    def observed_finish_requests(scheduler, *args, **kwargs):
        finished = original(scheduler, *args, **kwargs)
        try:
            for request in finished:
                event = backend_finish_event(scheduler, request)
                descriptor = os.open(event_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    os.write(descriptor, (json.dumps(event, sort_keys=True) + "\n").encode())
                finally:
                    os.close(descriptor)
        except Exception as error:
            print(f"LLAMA_ABORT_OBSERVER_ERROR: {error!r}", file=sys.stderr, flush=True)
        return finished

    observed_finish_requests._llama_ngram_observer = True
    scheduler_class.finish_requests = observed_finish_requests


def observer_environment(directory, environment):
    observer_directory = directory / "observer"
    observer_directory.mkdir(exist_ok=True)
    event_path = directory / "backend-finish.jsonl"
    event_path.unlink(missing_ok=True)
    (observer_directory / "sitecustomize.py").write_text(
        "import importlib.util\n"
        "from pathlib import Path\n"
        f"_spec = importlib.util.spec_from_file_location('llama_ngram_qualification_observer', {str(Path(__file__).resolve())!r})\n"
        "_observer = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_observer)\n"
        f"_observer.install_abort_observer(Path({str(event_path.resolve())!r}))\n"
    )
    return {
        **environment,
        "PYTHONPATH": str(observer_directory.resolve()) + os.pathsep + environment.get("PYTHONPATH", ""),
    }


@contextmanager
def server(args, batch, k):
    label = f"B{batch}-K{k}"
    directory = args.output / label
    directory.mkdir(parents=True, exist_ok=True)
    if active := engine_processes():
        raise RuntimeError(f"Refusing {label}: another engine is running: {active}")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    command, environment = launch_settings(args, batch, k)
    environment = observer_environment(directory, environment)
    write_json(directory / "launch.json", {"argv": command, "adapter": ADAPTER, "batch": batch, "k": k})
    log_path = directory / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, env=environment, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{label} server exited {process.returncode}; inspect {log_path}")
                try:
                    response = httpx.get(f"http://127.0.0.1:{args.port}/health", timeout=2)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(f"{label} did not become healthy; inspect {log_path}")
            if f"TTLlamaForCausalLM -> {ADAPTER}" not in log_path.read_text(errors="replace"):
                raise AssertionError(f"{label} did not report standalone adapter registration")
            yield directory, log_path
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            remaining = engine_processes()
            write_json(
                directory / "teardown.json",
                {"launcher_returncode": process.returncode, "remaining_engine_pids": remaining},
            )
            if remaining:
                raise RuntimeError(f"{label} left engine processes running after teardown: {remaining}")


def complete_batch(client, requests, batch):
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:
        futures = [pool.submit(client.complete, prompt, **controls) for prompt, controls in requests]
        return [future.result() for future in futures]


def completion_record(result):
    if result.status != 200:
        raise AssertionError(f"Completion failed: {result.body}")
    token_ids = result.token_ids
    logprobs = result.body["choices"][0]["logprobs"]["token_logprobs"]
    if (
        not isinstance(logprobs, list)
        or len(logprobs) != len(token_ids)
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in logprobs)
    ):
        raise AssertionError("Every returned token must have one finite raw logprob")
    return {"token_ids": token_ids, "logprobs": logprobs, "response": result.body}


def ensure_logged_logits_verification(client, prompt, token, server_log, assert_full_length_completion):
    def executed_count():
        submits = re.findall(r"(\d+) verify in logits mode", server_log.read_text(errors="replace"))
        return max(map(int, submits), default=0)

    if executed_count() > 0:
        return None
    # After prefill, 511 tokens require at least 128 verifications at K <= 3,
    # reaching the pinned plugin's _SUBMISSION_LOG_INTERVAL = 128.
    controls = {
        "max_tokens": 512,
        "temperature": 0.7,
        "seed": 1171,
        "top_k": 8,
        "top_p": 0.8,
        "allowed_token_ids": [token],
        "ignore_eos": True,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
    }
    result = client.complete(prompt, **controls)
    record = completion_record(result)
    assert_full_length_completion(result, controls["max_tokens"])
    assert record["token_ids"] == [token] * controls["max_tokens"], "Logits-verification probe returned wrong tokens"
    count = executed_count()
    assert count > 0, "server reported no executed logits verification after the 512-token probe"
    return {"prompt": prompt, "controls": controls, **record, "executed_logits_verifications": count}


def assert_reference_record(actual, expected, label):
    if actual["token_ids"] != expected["token_ids"]:
        raise AssertionError(f"{label}: token IDs differ from ordinary target execution")
    current, reference = actual["logprobs"], expected["logprobs"]
    if len(current) != len(actual["token_ids"]) or len(reference) != len(expected["token_ids"]):
        raise AssertionError(f"{label}: raw logprob count does not match token count")
    if len(current) != len(reference) or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in [*current, *reference]
    ):
        raise AssertionError(f"{label}: raw logprobs must have matching lengths and finite values")
    if any(abs(value - target) > 1e-3 for value, target in zip(current, reference)):
        raise AssertionError(f"{label}: raw logprobs differ from ordinary target execution")


def pressure_requests(tokenizer, token):
    subjects = (
        "A river crosses a forest. Fish move beneath the trees. ",
        "A spacecraft studies a planet. Instruments measure its atmosphere. ",
        "A musician rehearses a concerto. The orchestra follows the score. ",
        "A baker prepares bread. Flour and water form the dough. ",
    )
    requests = []
    for row, subject in enumerate(subjects):
        prefix = tokenizer.encode(subject, add_special_tokens=False)
        if not prefix:
            raise AssertionError("The pressure prompt prefix must contain tokens")
        prompt = (prefix * (168 // len(prefix) + 1))[:168] + [token] * 32
        requests.append(
            (
                prompt,
                {
                    "max_tokens": 160,
                    "temperature": 0.7,
                    "allowed_token_ids": [token],
                    "seed": 901 + row,
                    "logprobs": 1,
                    "return_tokens_as_token_ids": True,
                },
            )
        )
    if len({tuple(prompt) for prompt, _ in requests}) != 4:
        raise AssertionError("The pressure workload requires four distinct request histories")
    return requests


def disconnect_named_stream(client, prompt, max_tokens):
    request_id = f"llama-ngram-abort-{uuid.uuid4().hex}"
    body = client.request_body(
        prompt,
        stream=True,
        request_id=request_id,
        max_tokens=max_tokens,
        temperature=0.7,
        seed=811,
        ignore_eos=True,
        stop=[],
        stop_token_ids=[],
    )
    response_id = f"cmpl-{request_id}"
    chunks = 0
    tokens_seen = 0
    with httpx.stream("POST", f"{client.base_url}/v1/completions", json=body, timeout=600) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                raise AssertionError("The named stream finished before client cancellation")
            event = json.loads(payload)
            if event["id"] != response_id:
                raise AssertionError("The named stream returned an unexpected request ID")
            choice = event["choices"][0]
            if choice.get("finish_reason") is not None:
                raise AssertionError("The named stream finished before client cancellation")
            token_ids = choice.get("token_ids") or []
            if not token_ids:
                continue
            chunks += 1
            tokens_seen += len(token_ids)
            if chunks == 2:
                break
    if chunks != 2 or not 0 < tokens_seen < max_tokens:
        raise AssertionError("Client cancellation requires two unfinished token chunks")
    return {"response_id": response_id, "token_chunks": chunks, "tokens_seen": tokens_seen, "max_tokens": max_tokens}


def assert_backend_abort(event, response_id, max_tokens):
    expected_id = re.escape(response_id) + r"-0(?:-[0-9a-f]{8})?"
    if re.fullmatch(expected_id, event.get("request_id", "")) is None:
        raise AssertionError("Backend cancellation evidence names a different stream request")
    if event.get("finished_status") != "FINISHED_ABORTED":
        raise AssertionError("The backend request finished normally instead of being aborted")
    generated = event.get("generated_tokens")
    if event.get("max_tokens") != max_tokens or type(generated) is not int or not 0 < generated < max_tokens:
        raise AssertionError("The backend request did not stop before its requested output length")
    for field in ("owned_by_requests", "owned_by_running", "owned_by_waiting"):
        if event.get(field) is not False:
            raise AssertionError(f"The aborted backend request retains scheduler ownership: {field}")
    if event.get("num_running") != 0 or event.get("num_waiting") != 0:
        raise AssertionError("The scheduler has live requests before replacement admission")


def wait_for_backend_abort(event_path, response_id, max_tokens, timeout=60):
    deadline = time.monotonic() + timeout
    while True:
        if event_path.is_file():
            text = event_path.read_text()
            lines = text.splitlines() if text.endswith("\n") else text.splitlines()[:-1]
            for line in lines:
                event = json.loads(line)
                if event.get("request_id", "").startswith(response_id + "-0"):
                    assert_backend_abort(event, response_id, max_tokens)
                    return event
        if time.monotonic() >= deadline:
            raise AssertionError("No matching backend cancellation and scheduler release event was recorded")
        time.sleep(0.1)


def exercise_abort(client, prompt, event_path):
    stream = disconnect_named_stream(client, prompt, max_tokens=512)
    backend = wait_for_backend_abort(event_path, stream["response_id"], stream["max_tokens"])
    return {"stream": stream, "backend": backend}


def exercise_controls(client, tokenizer, repeated, ordinary_reference=None):
    red = tokenizer.encode(" red", add_special_tokens=False)
    blue = tokenizer.encode(" blue", add_special_tokens=False)
    if len(red) != 1 or len(blue) != 1:
        raise AssertionError("The pinned tokenizer must encode each control token as one token")
    controls = {
        "penalties": {"presence_penalty": 0.6, "frequency_penalty": 0.3, "repetition_penalty": 1.4},
        "bad_words": {"allowed_token_ids": [*red, *blue], "bad_words": [" red"]},
        "minimum": {"allowed_token_ids": [*red, *blue], "stop_token_ids": red, "min_tokens": 5},
    }
    records = {}
    for name, settings in controls.items():
        result = client.complete(
            repeated,
            temperature=0,
            max_tokens=12,
            seed=99,
            logprobs=3,
            return_tokens_as_token_ids=True,
            **settings,
        )
        assert result.status == 200 and result.token_ids, result.body
        if name == "bad_words":
            assert set(result.token_ids) == set(blue), result.token_ids
        if name == "minimum":
            assert result.completion_tokens >= 5 and red[0] not in result.token_ids[:5], result.body
        records[name] = completion_record(result)
        if ordinary_reference is not None:
            assert_reference_record(records[name], ordinary_reference[name], name)
    sampled_settings = {
        "temperature": 0.7,
        "top_k": 2,
        "top_p": 1.0,
        "allowed_token_ids": [*red, *blue],
        "seed": 783,
        "max_tokens": 24,
    }
    first = client.complete(repeated, **sampled_settings)
    second = client.complete(repeated, **sampled_settings)
    assert first.status == second.status == 200
    assert first.token_ids == second.token_ids, "An identical sampled request schedule did not reproduce"
    records["seeded_sampled"] = [first.body, second.body]
    grammar = "red red red red"
    structured = client.complete(
        repeated,
        max_tokens=32,
        temperature=0.7,
        seed=921,
        ignore_eos=False,
        structured_outputs={"regex": grammar},
    )
    assert structured.status == 200 and re.fullmatch(grammar, structured.text), structured.body
    records["structured"] = structured.body
    return records


def exercise_http(args, batch, k, tokenizer, helpers, ordinary_reference=None):
    SpecServer, acceptance_delta, assert_full_length_completion, widest_decode_batch_size = helpers
    with server(args, batch, k) as (directory, server_log):
        client = SpecServer(f"http://127.0.0.1:{args.port}", MODEL)
        token = tokenizer.encode(" red", add_special_tokens=False)[0]
        repeated = [token] * 96
        requests = []
        for index in range(max(20, batch * 5)):
            controls = {
                "max_tokens": 32 + index % 3,
                "temperature": 0.0 if index % 2 else 0.7,
                "seed": 101 + index,
                "top_k": 1 if index % 2 else 8,
                "top_p": 1.0 if index % 2 else 0.8,
                "allowed_token_ids": [token],
                "logprobs": 1,
                "return_tokens_as_token_ids": True,
            }
            prefix = tokenizer.encode(f"Request {index}: a story about {index * 17} birds. ", add_special_tokens=False)
            requests.append((prefix + repeated[: 64 + index % 31], controls))
        before = client.metrics()
        results = complete_batch(client, requests, batch)
        records = []
        for index, (result, (prompt, request_controls)) in enumerate(zip(results, requests)):
            assert_full_length_completion(result, request_controls["max_tokens"])
            assert result.token_ids == [token] * request_controls["max_tokens"]
            record = {"prompt": prompt, "controls": request_controls, **completion_record(result)}
            if ordinary_reference is not None:
                expected = ordinary_reference["concurrent"][index]
                assert record["prompt"] == expected["prompt"] and record["controls"] == expected["controls"]
                assert_reference_record(record, expected, f"concurrent request {index}")
            records.append(record)
        delta = acceptance_delta(before, client.metrics(), max(k, 1))
        verification_probe = None
        if k:
            assert delta.drafts > 0 and delta.draft_tokens > 0, "CPU n-gram produced no measured proposals"
            assert delta.accepted > 0, "real Llama accepted no n-gram draft"
            verification_probe = ensure_logged_logits_verification(
                client, repeated, token, server_log, assert_full_length_completion
            )
            if verification_probe is not None:
                delta = acceptance_delta(before, client.metrics(), max(k, 1))
        else:
            assert delta.drafts == 0 and delta.accepted == 0
        if batch > 1:
            assert widest_decode_batch_size(server_log) >= batch, (
                "scheduler did not execute the requested concurrent width"
            )
        replay = client.complete(requests[0][0], **requests[0][1])
        assert replay.status == 200 and replay.token_ids == results[0].token_ids
        controls = exercise_controls(
            client, tokenizer, repeated, None if ordinary_reference is None else ordinary_reference["controls"]
        )
        abort_observation = exercise_abort(client, repeated, directory / "backend-finish.jsonl")
        reused = client.complete(
            list(range(300, 331)),
            max_tokens=3,
            temperature=0,
            ignore_eos=True,
            logprobs=3,
            return_tokens_as_token_ids=True,
        )
        assert_full_length_completion(reused, 3)
        reused_record = completion_record(reused)
        if ordinary_reference is not None:
            assert_reference_record(reused_record, ordinary_reference["abort_reuse"], "abort/reuse replacement")
        eos = tokenizer.eos_token_id
        ended = client.complete(repeated, max_tokens=8, temperature=0.7, allowed_token_ids=[eos], ignore_eos=False)
        assert ended.status == 200 and ended.finish_reason == "stop", ended.body
        preemption = None
        pressure_records = None
        if batch == 4 and k in (0, 3):
            before_preemption = client.metrics()
            pressure = pressure_requests(tokenizer, token)
            pressured = complete_batch(client, pressure, 4)
            pressure_records = []
            for index, (result, (prompt, request_controls)) in enumerate(zip(pressured, pressure)):
                assert_full_length_completion(result, 160)
                assert result.token_ids == [token] * 160
                record = {"prompt": prompt, "controls": request_controls, **completion_record(result)}
                if ordinary_reference is not None:
                    expected = ordinary_reference["pressure"][index]
                    assert record["prompt"] == expected["prompt"] and record["controls"] == expected["controls"]
                    assert_reference_record(record, expected, f"preemption/resume request {index}")
                pressure_records.append(record)
            preemption = acceptance_delta(before_preemption, client.metrics(), max(k, 1))
            assert preemption.preemptions > 0, "The KV-pressure workload did not exercise preemption/resume"
        write_json(
            directory / "results.json",
            {
                "acceptance": delta.as_dict(),
                "verification_probe": verification_probe,
                "completions": records,
                "controls": controls,
                "abort_reuse": reused_record,
                "abort_observation": abort_observation,
                "eos": ended.body,
                "preemption": None if preemption is None else preemption.as_dict(),
                "pressure": pressure_records,
            },
        )
        (directory / "metrics.txt").write_text(client.metrics().raw)
        print(f"LLAMA_HTTP_PASS: B={batch} K={k} drafts={delta.draft_tokens} accepted={delta.accepted}", flush=True)
        return {"controls": controls, "concurrent": records, "abort_reuse": reused_record, "pressure": pressure_records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--plugin-revision", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--skip-device-matrix", action="store_true")
    parser.add_argument("--runtime-manifest", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.plugin_root = args.plugin_root.resolve()
    reset_completion_markers(args.output)
    if git_revision(args.plugin_root) != args.plugin_revision:
        raise ValueError("The plugin checkout does not match --plugin-revision")
    require_commit_sha("plugin_revision", args.plugin_revision)
    os.environ.update(
        HF_MODEL=MODEL,
        LLAMA_CHECKPOINT_REVISION=args.checkpoint_revision,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TT_TRANSFORMERS_OFFLINE="1",
    )
    from huggingface_hub import try_to_load_from_cache

    require_cached_checkpoint(args.checkpoint_revision, try_to_load_from_cache)
    runtime_modules = {
        name: importlib.import_module(name)
        for name in ("ttnn", "ttnn.device", "vllm", "tt_transformers", "vllm_tt_plugin")
    }
    from vllm.platforms import current_platform

    assert current_platform.device_type == "tt", current_platform
    provenance = {
        "tt_transformers_revision": git_revision(ROOT),
        "plugin_revision": git_revision(args.plugin_root),
        "checkpoint": MODEL,
        "checkpoint_revision": args.checkpoint_revision,
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("ttnn", "torch", "transformers", "vllm", "vllm-tt-plugin")
        },
        "tt_transformers_source_version": importlib.import_module("tt_transformers").__version__,
        "tt_transformers_source": source_provenance(runtime_modules["tt_transformers"], ROOT, "tt_transformers"),
        "plugin_source": source_provenance(runtime_modules["vllm_tt_plugin"], args.plugin_root, "vllm_tt_plugin"),
        "vllm_source": package_source("vllm"),
        "ttnn_source": package_source("ttnn"),
        "module_origins": {name: module.__file__ for name, module in runtime_modules.items()},
        "trace_mode": "none",
        "mesh": os.environ.get("MESH_DEVICE"),
        "tt_metal_runtime_root": os.environ.get("TT_METAL_RUNTIME_ROOT"),
        "ttnn_runtime": runtime_provenance(
            os.environ.get("TT_METAL_RUNTIME_ROOT"), os.environ.get("TT_METAL_HOME"), args.runtime_manifest
        ),
        "argv": sys.argv,
    }
    if os.environ.get("TT_METAL_HOME"):
        provenance["tt_metal_revision"] = git_revision(os.environ["TT_METAL_HOME"])
    write_json(args.output / "provenance.json", provenance)
    sys.path.insert(0, str(args.plugin_root / "tests/tt/spec"))
    from spec_client import SpecServer, acceptance_delta, assert_full_length_completion, widest_decode_batch_size

    helpers = SpecServer, acceptance_delta, assert_full_length_completion, widest_decode_batch_size
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=args.checkpoint_revision, local_files_only=True)
    ordinary = {}
    for batch in (1, 2, 4):
        ordinary[batch] = exercise_http(args, batch, 0, tokenizer, helpers)
    if not args.skip_device_matrix:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-s",
                "tests/hardware/models/llama32_1b/test_speculation.py",
                f"--junitxml={args.output / 'device-matrix.xml'}",
            ],
            cwd=ROOT,
            check=True,
        )
        require_device_matrix(canonicalize_junit_report(args.output / "device-matrix.xml"))
    for batch in (1, 2, 4):
        for k in (1, 2, 3):
            exercise_http(args, batch, k, tokenizer, helpers, ordinary[batch])
    write_json(
        args.output / ("partial.json" if args.skip_device_matrix else "success.json"),
        {
            "ordinary_batches": [1, 2, 4],
            "speculative_batches": [1, 2, 4],
            "draft_widths": [1, 2, 3],
            "device_matrix_executed": not args.skip_device_matrix,
        },
    )


if __name__ == "__main__":
    main()
