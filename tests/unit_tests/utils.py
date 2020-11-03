"""
    Here defines utility methods and constructs that are used throughout the unit_test package
"""
import pytest
import torch

# Pytest skipping decorators
# Skip a test if cuda is not enabled on the host system
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda enabled")

