"""
    Here defines utility methods and constructs that are used throughout the unit_test package
"""
import pytest
import torch
from torch import Size

# Pytest skipping decorators
# Skip a test if cuda is not enabled on the host system
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda enabled")
# Numerical precision
EPS = 1e-6


# Test the equality of two tensors within a given numerical precision
def equal_within_error(tensor_1: torch.Tensor, tensor_2: torch.Tensor, precision=EPS):
    return torch.max(torch.abs(tensor_1 - tensor_2)) < precision


# Generate a positive definite batched matrix
# Mostly used as the covariance matrix for MultivariateNormal distribution
def generate_positive_definite(b_shape: Size, e_size: int):
    # Generate random positive-valued matrix A
    A = torch.randn(b_shape + Size([e_size, e_size]))
    # Get the transpose
    A_T = torch.transpose(A, dim0=-1, dim1=-2)
    # Matrix multiplication to obtain a semi-definite matrix
    S_D = torch.matmul(A, A_T)
    # Ensure that the matrix is positive-definite - strictly grater than 0
    I = torch.eye(e_size)
    for i in range(len(b_shape)):
        I = I.unsqueeze(dim=0)
    I = I.expand_as(S_D)
    D = S_D + I

    return D

