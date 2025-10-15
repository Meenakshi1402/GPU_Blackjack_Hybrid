"""
Package initializer for GPU Blackjack Triton project.
This makes the folder importable as a Python module.

Exports key API functions and classes so users can do:
    from gpu_triton import run_gpu_blackjack, GPUStats
"""

from .engine import run_gpu_blackjack, GPUStats
