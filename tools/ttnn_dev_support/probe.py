"""Executed in an isolated child interpreter, never imported by production code."""

from __future__ import annotations

import importlib.metadata
import json
import site
import sys
from pathlib import Path


def installed_project_version() -> str | None:
    return next(
        (
            dist.version
            for dist in importlib.metadata.distributions(path=site.getsitepackages())
            if dist.metadata["Name"].replace("_", "-").lower() == "tt-transformers"
        ),
        None,
    )


def main() -> None:
    project = Path(sys.argv[1]).resolve()
    sys.path[:0] = [str(project / "src"), str(project)]
    import ttnn
    import ttnn._ttnn

    import tt_transformers

    distribution = importlib.metadata.distribution("ttnn")
    project_version = installed_project_version()
    direct = distribution.read_text("direct_url.json")
    libraries = set()
    maps = Path("/proc/self/maps")
    if maps.exists():
        for line in maps.read_text().splitlines():
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and parts[5].startswith("/"):
                path = parts[5]
                if any(name in Path(path).name for name in ("_ttnn", "libtt_metal", "libtt-umd", "libtt_stl")):
                    libraries.add(path)
    result = {
        "ttnn_origin": ttnn.__file__,
        "extension_origin": ttnn._ttnn.__file__,
        "project_origin": tt_transformers.__file__,
        "ttnn_version": distribution.version,
        "ttnn_requires": distribution.requires or [],
        "tt_transformers_distribution": project_version,
        "direct_url": json.loads(direct) if direct else None,
        "native_libraries": sorted(libraries),
    }
    print("TTNN_DEV_PROBE=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
