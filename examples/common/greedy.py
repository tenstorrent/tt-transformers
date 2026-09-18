"""Greedy conversion helpers owned by focused example smoke paths."""

import torch
import ttnn

from examples.common.auto_compose import to_torch_auto_compose


def greedy_argmax_from_logits(logits: ttnn.Tensor, *, mesh_device: ttnn.MeshDevice) -> int:
    """Return the global argmax from possibly sharded logits."""

    host_logits = to_torch_auto_compose(logits, device=mesh_device).float()
    if host_logits.dim() == 4:
        host_logits = host_logits[0, 0, 0]
    elif host_logits.dim() == 3:
        host_logits = host_logits[0, 0]
    return int(torch.argmax(host_logits).item())


def greedy_decode_one_step(model, token_id: int, *, current_pos: int) -> int:
    """Decode one token and return the next greedy token ID."""

    token = torch.tensor([[[[token_id]]]], dtype=torch.int32)
    device_token = ttnn.from_torch(
        token,
        device=model.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(model.mesh_device),
    )
    hidden = model.decode_from_token_ids(device_token, current_pos=current_pos)
    logits = model.lm_logits(hidden)
    return greedy_argmax_from_logits(logits, mesh_device=model.mesh_device)
