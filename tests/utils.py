"""
    Here defines utility methods and constructs that are used throughout the unit_test package
"""
import pytest
import torch
from torch import Size
from pysigma.defs import Message, MessageType

# Pytest skipping decorators
# Skip a test if cuda is not enabled on the host system
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda enabled")
# Numerical precision
EPS = 1e-6


# Test the equality of two tensors within a given numerical precision
def equal_within_error(tensor_1: torch.Tensor, tensor_2: torch.Tensor, precision=EPS):
    return torch.max(torch.abs(tensor_1 - tensor_2)) < precision


# Self-assertion version of the above method. When assertion fails, reports the maximum absolute difference value.
def assert_equal_within_error(tensor_1: torch.Tensor, tensor_2: torch.Tensor, precision=EPS):
    diff = torch.max(torch.abs(tensor_1 - tensor_2))
    assert diff < precision, "Maximum absolute difference: {}".format(diff.item())


# Test that two tensors are off by a constant along the given dims within the given precision
def constant_difference_within_error(tensor_1: torch.Tensor, tensor_2: torch.Tensor, dims: list, precision=EPS):
    diff = tensor_1 - tensor_2
    diff_max, diff_min = diff, diff
    for dim in dims:
        diff_max = torch.max(diff_max, dim=dim)[0]
        diff_min = torch.min(diff_min, dim=dim)[0]
    diff_diff = torch.max(torch.abs(diff_max - diff_min))
    return diff_diff < precision


# Self-assertion version of the above method.
def assert_constant_difference_within_error(tensor_1: torch.Tensor, tensor_2: torch.Tensor, dims: list, precision=EPS):
    diff = tensor_1 - tensor_2
    diff_max, diff_min = diff, diff
    for dim in dims:
        diff_max = torch.max(diff_max, dim=dim)[0]
        diff_min = torch.min(diff_min, dim=dim)[0]
    diff_diff = torch.max(torch.abs(diff_max - diff_min))
    assert diff_diff < precision, "Maximum variance between the differences: {}".format(diff_diff)


# Test that two tensors are proportional to each other along the given dims with the ratio variance smaller than
#   the given precision
def proportional_within_error(tensor_1: torch.Tensor, tensor_2: torch.Tensor, dims: list, precision=EPS):
    ratio = tensor_1 / tensor_2
    ratio_max, ratio_min = ratio, ratio
    for dim in dims:
        ratio_max = torch.max(ratio_max, dim=dim)[0]
        ratio_min = torch.min(ratio_min, dim=dim)[0]
    ratio_diff = torch.max(torch.abs(ratio_max - ratio_min))
    return ratio_diff < precision


# Self-assertion version of the above method.
def assert_proportional_within_error(tensor_1: torch.Tensor, tensor_2: torch.Tensor, dims: list, precision=EPS):
    ratio = tensor_1 / tensor_2
    ratio_max, ratio_min = ratio, ratio
    for dim in dims:
        ratio_max = torch.max(ratio_max, dim=dim, keepdim=True)[0]
        ratio_min = torch.min(ratio_min, dim=dim, keepdim=True)[0]
    ratio_diff = torch.max(torch.abs(ratio_max - ratio_min))
    assert ratio_diff < precision, "Maximum ratio difference: {}".format(ratio_max - ratio_min)


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


# Generate a random message of given type
def random_message(msg_type, b_shape, p_shape, s_shape, e_shape):
    param = torch.randn(b_shape + p_shape) if MessageType.Parameter in msg_type else 0
    weight = torch.rand(b_shape + s_shape) if MessageType.Particles in msg_type else 1
    ptcl = [torch.randn([s, e]) for s, e in zip(list(s_shape), list(e_shape))] \
        if MessageType.Particles in msg_type else None
    dens = [torch.randn(s) for s in zip(list(s_shape))] \
        if MessageType.Particles in msg_type else None
    return Message(msg_type, b_shape, p_shape, s_shape, e_shape,
                   param, ptcl, weight, dens)

