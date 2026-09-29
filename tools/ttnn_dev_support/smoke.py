# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Explicit on-device JIT smoke check for a selected TTNN development runtime."""

from __future__ import annotations

import json


def main() -> None:
    import torch
    import ttnn
    from examples.common.runtime import open_mesh_device

    with open_mesh_device({"mesh_shape": (1, 1), "num_command_queues": 1}) as mesh:
        host = torch.arange(1024, dtype=torch.float32).reshape(1, 1, 32, 32).to(torch.bfloat16)
        tensor = ttnn.from_torch(host, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        result = ttnn.add(tensor, tensor)
        torch.testing.assert_close(ttnn.to_torch(result), host + host, rtol=0, atol=0)
        ttnn.deallocate(result)
        ttnn.deallocate(tensor)
    print(json.dumps({"status": "pass", "operation": "ttnn.add", "mesh": [1, 1], "elements": 1024}))


if __name__ == "__main__":
    main()
