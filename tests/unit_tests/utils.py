"""
    Here defines utility methods and constructs that are used throughout the unit_test package
"""
import pytest
import torch

# Pytest skipping decorators
# Skip a test if cuda is not enabled on the host system
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda enabled")
# Numerical precision
EPS = 1e-6


# Test the equality of two tensors within a given numerical precision
def equal_within_error(tensor_1, tensor_2):
    return torch.max(torch.abs(tensor_1 - tensor_2)) < EPS

