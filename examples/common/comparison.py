"""Narrow PCC helper used by diagnostic examples."""

import torch


def comp_pcc(golden, calculated, pcc=0.99, rtol=1e-5, atol=1e-4):
    golden = torch.as_tensor(golden)
    calculated = torch.as_tensor(calculated).to(golden.dtype)
    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        return True, 1.0
    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        return False, 0.0
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        result = torch.allclose(golden, calculated, rtol=rtol, atol=atol)
        return result, float(result)
    golden = torch.nan_to_num(golden.squeeze().flatten().float())
    calculated = torch.nan_to_num(calculated.squeeze().flatten().float())
    if golden.numel() == 0 or calculated.numel() == 0:
        return False, 0.0
    if torch.allclose(golden, golden[0]) or torch.allclose(calculated, calculated[0]):
        result = torch.allclose(golden, calculated, rtol=rtol, atol=atol)
        return result, float(result)
    value = float(torch.corrcoef(torch.stack((golden, calculated)))[0, 1])
    return value >= pcc, value
