# Contributing

Keep changes within the standalone dependency boundary: production code may
import only `tt_transformers`, declared third-party dependencies, and the
Python standard library. Do not add imports through the legacy `models.*`
namespace or from `tests`.

Run the host checks, import-boundary test, wheel build, and isolated wheel
smoke before requesting review. Hardware results must name the exact Git SHA,
software versions, physical host/SKU, mesh, environment, command, and log.
Skipped device tests are not evidence of support.
