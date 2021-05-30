"""
    Global pytest configurations. Contains session-scoped fixtures for pretest setup.
"""
import pytest
import torch


# Declare autouse=True to execute at the beginning of tests and need not be referenced in test methods
@pytest.fixture(scope="session", autouse=True)
def set_torch_random_seed():
    # Set pytorch random seed
    torch.manual_seed(10)

