"""Typed environment, cache-location, identity, and preflight policy.

The compatibility cache layout is intentionally separate from CacheIdentity:
existing explicit paths remain usable without forced tensor rematerialization,
while the complete identity is available to reports and future cache schemas.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, Mapping


CONVERSION_SCHEMA = "tttv2-lazy-weight-v1"
CACHE_NAMESPACE_PREFIX = "identity"
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})
SECRET_NAME = re.compile(r"(?:^|_)(?:TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIALS?|API_KEY|ACCESS_KEY)(?:_|$)", re.I)
CacheCreateMode = Literal["always", "explicit_only", "never"]
EnvironmentKind = Literal["bool", "int", "path", "str"]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvironmentVariableSpec:
    name: str
    kind: EnvironmentKind
    default: Any = None
    production: bool = True
    description: str = ""


# Phase 0's 21 resolved keys plus the explicit standalone controls introduced
# by this policy. Demo-only keys are typed here but are not consumed by model
# construction.
ENVIRONMENT_SPECS: dict[str, EnvironmentVariableSpec] = {
    "CI": EnvironmentVariableSpec("CI", "bool", False, description="report-only CI context"),
    "DISABLE_BATCHED_EXTRACT": EnvironmentVariableSpec("DISABLE_BATCHED_EXTRACT", "bool", False),
    "DISABLE_BATCHED_PREFILL": EnvironmentVariableSpec("DISABLE_BATCHED_PREFILL", "bool", False),
    "DISABLE_MINIMAL_MATMUL": EnvironmentVariableSpec("DISABLE_MINIMAL_MATMUL", "bool", False),
    "DISABLE_PREFILL_AG_BF8": EnvironmentVariableSpec("DISABLE_PREFILL_AG_BF8", "bool", False),
    "DISABLE_PREFILL_REDUCE_BF8": EnvironmentVariableSpec("DISABLE_PREFILL_REDUCE_BF8", "bool", False),
    "HF_HUB_OFFLINE": EnvironmentVariableSpec("HF_HUB_OFFLINE", "bool", False),
    "HF_MODEL": EnvironmentVariableSpec("HF_MODEL", "str"),
    "MAX_PREFILL_CHUNK_SIZE": EnvironmentVariableSpec("MAX_PREFILL_CHUNK_SIZE", "int"),
    "MESH_DEVICE": EnvironmentVariableSpec("MESH_DEVICE", "str"),
    "QWEN25_CODER_32B_DEMO_NUM_LAYERS": EnvironmentVariableSpec(
        "QWEN25_CODER_32B_DEMO_NUM_LAYERS", "int", production=False
    ),
    "QWEN25_CODER_32B_NUMDIV_LAYERS": EnvironmentVariableSpec(
        "QWEN25_CODER_32B_NUMDIV_LAYERS", "int", production=False
    ),
    "QWEN25_CODER_32B_NUMDIV_OUT": EnvironmentVariableSpec(
        "QWEN25_CODER_32B_NUMDIV_OUT", "path", production=False
    ),
    "QWEN25_CODER_32B_NUMDIV_SEQ": EnvironmentVariableSpec(
        "QWEN25_CODER_32B_NUMDIV_SEQ", "int", production=False
    ),
    "QWEN3_32B_DEMO_NUM_LAYERS": EnvironmentVariableSpec("QWEN3_32B_DEMO_NUM_LAYERS", "int", production=False),
    "QWEN3_32B_NUMDIV_LAYERS": EnvironmentVariableSpec("QWEN3_32B_NUMDIV_LAYERS", "int", production=False),
    "QWEN3_32B_NUMDIV_OUT": EnvironmentVariableSpec("QWEN3_32B_NUMDIV_OUT", "path", production=False),
    "QWEN3_32B_NUMDIV_SEQ": EnvironmentVariableSpec("QWEN3_32B_NUMDIV_SEQ", "int", production=False),
    "TRANSFORMERS_OFFLINE": EnvironmentVariableSpec("TRANSFORMERS_OFFLINE", "bool", False),
    "TT_CACHE_FALLBACK_PATH": EnvironmentVariableSpec("TT_CACHE_FALLBACK_PATH", "path"),
    "TT_CACHE_PATH": EnvironmentVariableSpec("TT_CACHE_PATH", "path"),
    "TT_TRANSFORMERS_CACHE": EnvironmentVariableSpec("TT_TRANSFORMERS_CACHE", "path"),
    "TT_TRANSFORMERS_OFFLINE": EnvironmentVariableSpec("TT_TRANSFORMERS_OFFLINE", "bool", False),
    "XDG_CACHE_HOME": EnvironmentVariableSpec("XDG_CACHE_HOME", "path"),
}


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def parse_bool(value: str | bool | None, *, name: str, default: bool = False) -> bool:
    """Parse one normalized environment boolean or raise on ambiguity."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
    raise ValueError(f"{name} must be one of {{{accepted}}}, got {value!r}")


def environment_value(name: str, environ: Mapping[str, str] | None = None) -> Any:
    """Read a declared environment value using its typed parser."""

    try:
        spec = ENVIRONMENT_SPECS[name]
    except KeyError as error:
        raise KeyError(f"undeclared tt-transformers environment key: {name}") from error
    raw = _environment(environ).get(name)
    if raw is None:
        return spec.default
    if spec.kind == "bool":
        return parse_bool(raw, name=name, default=bool(spec.default))
    if spec.kind == "int":
        try:
            return int(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if spec.kind == "path":
        return Path(raw).expanduser() if raw else spec.default
    return raw


def environment_flag(name: str, environ: Mapping[str, str] | None = None) -> bool:
    value = environment_value(name, environ)
    if not isinstance(value, bool):
        raise TypeError(f"{name} is not declared as a boolean")
    return value


def environment_int(name: str, environ: Mapping[str, str] | None = None) -> int | None:
    value = environment_value(name, environ)
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise TypeError(f"{name} is not declared as an integer")
    return value


def offline_mode(environ: Mapping[str, str] | None = None) -> bool:
    """Return explicit offline policy; CI alone never selects offline mode."""

    return any(
        environment_flag(name, environ)
        for name in ("TT_TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _installed_version(distribution: str, fallback: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return fallback


def _stable_value(value: Any) -> str:
    if value is None:
        return "none"
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _pairs(values: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), _stable_value(value)) for key, value in (values or {}).items()))


@dataclass(frozen=True)
class CacheIdentity:
    schema_version: int
    tt_transformers_version: str
    ttnn_version: str
    hf_model_id: str
    hf_revision: str
    conversion_schema: str
    architecture: str
    topology: str
    dtype_inputs: tuple[tuple[str, str], ...]
    layout_inputs: tuple[tuple[str, str], ...]
    sharding_inputs: tuple[tuple[str, str], ...]

    @classmethod
    def resolve(
        cls,
        *,
        hf_model_id: str,
        hf_revision: str | None,
        architecture: Any,
        topology: str,
        dtype_inputs: Mapping[str, Any] | None,
        layout_inputs: Mapping[str, Any] | None,
        sharding_inputs: Mapping[str, Any] | None,
        conversion_schema: str = CONVERSION_SCHEMA,
    ) -> "CacheIdentity":
        return cls(
            schema_version=1,
            tt_transformers_version=_installed_version("tt-transformers", "0.1.0.dev0"),
            ttnn_version=_installed_version("ttnn", "unknown"),
            hf_model_id=hf_model_id,
            hf_revision=hf_revision or "unversioned",
            conversion_schema=conversion_schema,
            architecture=_stable_value(architecture),
            topology=topology,
            dtype_inputs=_pairs(dtype_inputs),
            layout_inputs=_pairs(layout_inputs),
            sharding_inputs=_pairs(sharding_inputs),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tt_transformers_version": self.tt_transformers_version,
            "ttnn_version": self.ttnn_version,
            "hf_model_id": self.hf_model_id,
            "hf_revision": self.hf_revision,
            "conversion_schema": self.conversion_schema,
            "architecture": self.architecture,
            "topology": self.topology,
            "dtype_inputs": dict(self.dtype_inputs),
            "layout_inputs": dict(self.layout_inputs),
            "sharding_inputs": dict(self.sharding_inputs),
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class CacheResolution:
    path: Path
    source: str
    topology: str
    legacy_cwd_path: Path
    created: bool
    fallback_from: Path | None = None
    cache_identity: CacheIdentity | None = None
    identity_applied_to_path: bool = False
    release_warning: str | None = None


def _model_parts(hf_model_id: str) -> tuple[str, ...]:
    parts = tuple(part for part in hf_model_id.strip("/").split("/") if part not in {"", "."})
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"invalid HF model ID for cache layout: {hf_model_id!r}")
    return parts


def _default_cache_root(environ: Mapping[str, str]) -> tuple[Path, str]:
    configured = environment_value("TT_TRANSFORMERS_CACHE", environ)
    if configured is not None:
        return configured, "TT_TRANSFORMERS_CACHE"
    xdg = environment_value("XDG_CACHE_HOME", environ)
    if xdg is not None:
        return xdg / "tt-transformers", "XDG_CACHE_HOME"
    home = Path(environ.get("HOME", str(Path.home()))).expanduser()
    return home / ".cache" / "tt-transformers", "user_cache"


def resolve_model_cache(
    *,
    hf_model_id: str,
    topology: str,
    cache_dir: Path | str | None = None,
    append_topology_to_tt_cache_path: bool = False,
    create_mode: CacheCreateMode = "always",
    permission_fallback: bool = False,
    hf_revision: str | None = None,
    mesh_device: Any = None,
    dtype: Any = None,
    layout_inputs: Mapping[str, Any] | None = None,
    sharding_inputs: Mapping[str, Any] | None = None,
    conversion_schema: str = CONVERSION_SCHEMA,
    environ: Mapping[str, str] | None = None,
) -> CacheResolution:
    """Resolve a versioned implicit cache or an exact compatibility override."""

    env = _environment(environ)
    model_parts = _model_parts(hf_model_id)
    if not topology or "/" in topology:
        raise ValueError(f"cache topology must be one path segment, got {topology!r}")
    explicit = cache_dir is not None
    identity = resolve_cache_identity(
        hf_model_id=hf_model_id,
        hf_revision=hf_revision,
        topology=topology,
        mesh_device=mesh_device,
        dtype=dtype,
        layout_inputs=layout_inputs,
        sharding_inputs=sharding_inputs,
        conversion_schema=conversion_schema,
    )
    identity_applied_to_path = False
    release_warning = None
    if explicit:
        path = Path(cache_dir).expanduser()
        source = "cache_dir"
    else:
        legacy_env = environment_value("TT_CACHE_PATH", env)
        if legacy_env is not None:
            path = legacy_env / topology if append_topology_to_tt_cache_path else legacy_env
            source = "TT_CACHE_PATH"
            explicit = True
        else:
            root, source = _default_cache_root(env)
            path = root.joinpath(
                *model_parts,
                f"{CACHE_NAMESPACE_PREFIX}-{identity.schema_version}-{identity.digest}",
                topology,
            )
            identity_applied_to_path = True

    if explicit:
        release_warning = (
            "release qualification warning: explicit cache override is not identity-namespaced and "
            "preserves its established path; CacheIdentity is reported but identity_applied_to_path=false"
        )

    should_create = create_mode == "always" or (create_mode == "explicit_only" and explicit)
    created = False
    fallback_from: Path | None = None
    if should_create:
        try:
            path.mkdir(parents=True, exist_ok=True)
            created = True
        except OSError as error:
            if not permission_fallback or error.errno not in (errno.EROFS, errno.EACCES, errno.EPERM):
                raise
            fallback_from = path
            fallback_root = environment_value("TT_CACHE_FALLBACK_PATH", env) or Path("/tmp/tttv2_model_cache")
            path = fallback_root / model_parts[-1] / topology
            path.mkdir(parents=True, exist_ok=True)
            source = "permission_fallback"
            created = True
            identity_applied_to_path = False
            release_warning = (
                "release qualification warning: permission fallback from an explicit cache override "
                "is not identity-namespaced and preserves compatibility layout; CacheIdentity is reported "
                "but identity_applied_to_path=false"
            )

    legacy_cwd_path = Path.cwd().joinpath("model_cache", *model_parts, topology)
    return CacheResolution(
        path=path,
        source=source,
        topology=topology,
        legacy_cwd_path=legacy_cwd_path,
        created=created,
        fallback_from=fallback_from,
        cache_identity=identity,
        identity_applied_to_path=identity_applied_to_path,
        release_warning=release_warning,
    )


def resolve_model_cache_path(**kwargs: Any) -> Path:
    return resolve_model_cache(**kwargs).path


def _mesh_identity(mesh_device: Any) -> tuple[Any, dict[str, Any]]:
    arch = mesh_device.arch() if callable(getattr(mesh_device, "arch", None)) else getattr(mesh_device, "arch", None)
    shape = getattr(mesh_device, "shape", None)
    try:
        mesh_shape = tuple(int(value) for value in shape)
    except (TypeError, ValueError):
        mesh_shape = _stable_value(shape)
    num_devices = (
        mesh_device.get_num_devices()
        if callable(getattr(mesh_device, "get_num_devices", None))
        else None
    )
    return arch, {"mesh_shape": mesh_shape, "num_devices": num_devices}


def resolve_cache_identity(
    *,
    hf_model_id: str,
    hf_revision: str | None,
    topology: str,
    mesh_device: Any,
    dtype: Any,
    layout_inputs: Mapping[str, Any] | None = None,
    sharding_inputs: Mapping[str, Any] | None = None,
    conversion_schema: str = CONVERSION_SCHEMA,
) -> CacheIdentity:
    """Build the single cache identity used by path resolution and preflight."""

    arch, mesh_sharding = _mesh_identity(mesh_device)
    return CacheIdentity.resolve(
        hf_model_id=hf_model_id,
        hf_revision=hf_revision,
        architecture=arch,
        topology=topology,
        dtype_inputs={"requested_weight_dtype": dtype},
        layout_inputs=layout_inputs or {"default_weight_layout": "TILE_LAYOUT"},
        sharding_inputs=sharding_inputs or mesh_sharding,
        conversion_schema=conversion_schema,
    )


def _cache_source_for_report(
    cache_path: Path,
    *,
    cache_dir: Path | str | None,
    environ: Mapping[str, str],
) -> str:
    if cache_dir is not None:
        return "cache_dir"
    fallback_root = environment_value("TT_CACHE_FALLBACK_PATH", environ) or Path("/tmp/tttv2_model_cache")
    try:
        cache_path.relative_to(fallback_root)
    except ValueError:
        pass
    else:
        if environment_value("TT_CACHE_PATH", environ) is not None:
            return "permission_fallback"
    if environment_value("TT_CACHE_PATH", environ) is not None:
        return "TT_CACHE_PATH"
    return _default_cache_root(environ)[1]


def redact_environment(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return only policy-relevant values, redacting any present secret keys."""

    env = _environment(environ)
    report: dict[str, Any] = {}
    for name in sorted(ENVIRONMENT_SPECS):
        if name not in env:
            continue
        report[name] = "<redacted>" if SECRET_NAME.search(name) else _stable_value(environment_value(name, env))
    for name in sorted(env):
        if name not in report and SECRET_NAME.search(name):
            report[name] = "<redacted>"
    return report


def model_preflight_report(
    *,
    hf_model_id: str,
    hf_revision: str | None,
    cache_path: Path,
    topology: str,
    mesh_device: Any,
    dtype: Any,
    cache_dir: Path | str | None = None,
    cache_resolution: CacheResolution | None = None,
    conversion_schema: str = CONVERSION_SCHEMA,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env = _environment(environ)
    if cache_resolution is not None:
        if cache_resolution.path != cache_path or cache_resolution.topology != topology:
            raise ValueError("cache preflight resolution does not match the reported path/topology")
        identity = cache_resolution.cache_identity
        if identity is None:
            raise ValueError("cache preflight resolution has no CacheIdentity")
        cache_source = cache_resolution.source
        identity_applied_to_path = cache_resolution.identity_applied_to_path
        release_warning = cache_resolution.release_warning
    else:
        identity = resolve_cache_identity(
            hf_model_id=hf_model_id,
            hf_revision=hf_revision,
            topology=topology,
            mesh_device=mesh_device,
            dtype=dtype,
            conversion_schema=conversion_schema,
        )
        cache_source = _cache_source_for_report(cache_path, cache_dir=cache_dir, environ=env)
        identity_component = f"{CACHE_NAMESPACE_PREFIX}-{identity.schema_version}-{identity.digest}"
        identity_applied_to_path = cache_source in {
            "TT_TRANSFORMERS_CACHE",
            "XDG_CACHE_HOME",
            "user_cache",
        } and identity_component in cache_path.parts
        release_warning = None
        if cache_source in {"cache_dir", "TT_CACHE_PATH", "permission_fallback"}:
            release_warning = (
                "explicit cache override is not identity-namespaced; release qualification must bind reuse "
                "to the reported CacheIdentity"
            )
        elif not identity_applied_to_path:
            release_warning = "implicit cache path is missing its required CacheIdentity namespace"
    legacy = Path.cwd().joinpath("model_cache", *_model_parts(hf_model_id), topology)
    return {
        "offline": offline_mode(env),
        "ci": environment_flag("CI", env),
        "checkpoint": {"hf_model_id": hf_model_id, "hf_revision": hf_revision or "unversioned"},
        "cache": {
            "path": str(cache_path),
            "source": cache_source,
            "legacy_cwd_path": str(legacy),
            "identity_applied_to_path": identity_applied_to_path,
            "release_warning": release_warning,
        },
        "cache_identity": {**identity.as_dict(), "sha256": identity.digest},
        "environment": redact_environment(env),
    }


def report_model_preflight(*, logger: Any = None, **kwargs: Any) -> dict[str, Any]:
    """Build and emit one secret-redacted model preflight report."""

    report = model_preflight_report(**kwargs)
    (logger or LOGGER).info(f"tt-transformers preflight: {json.dumps(report, sort_keys=True)}")
    return report
