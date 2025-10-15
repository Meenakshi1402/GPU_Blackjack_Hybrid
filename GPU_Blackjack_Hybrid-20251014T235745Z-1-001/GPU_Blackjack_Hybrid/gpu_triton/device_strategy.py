"""
device_strategy.py
------------------
Converts the CPU blackjack rule matrix into a GPU-ready tensor.
This function bridges your CPU-based BasicStrategy_ logic
with the Triton GPU kernel.
"""
import torch
from blackjack_pkg.strategy import BasicStrategy_

def rules_43x10_tensor(device="cuda"):
    bs = BasicStrategy_()
    rules = torch.tensor(bs.rules, dtype=torch.int32).reshape(43, 10)
    return rules.to(device, non_blocking=True)
