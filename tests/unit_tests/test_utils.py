"""
    Unit tests for pysigma.utils module
"""
import pytest
from unittest.mock import MagicMock, patch
import torch
from torch import Size
import torch.distributions as D

from pysigma.utils import intern_name, extern_name, compatible_shape, DistributionServer, KnowledgeServer


def test_intern_name_invalid():
    name, struc_type = "Test", "something else"
    with pytest.raises(AssertionError):
        intern_name(name, struc_type)


def test_intern_name_correct():
    name, struc_type = "Test", "type"
    expected_name = "TYPE_[Test]"
    assert intern_name(name, struc_type) == expected_name

    name, struc_type = "Test", "predicate"
    expected_name = "PRED_[Test]"
    assert intern_name(name, struc_type) == expected_name

    name, struc_type = "Test", "conditional"
    expected_name = "COND_[Test]"
    assert intern_name(name, struc_type) == expected_name


def test_extern_name_invalid():
    name, struc_type = "Test", "something else"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

    name, struc_type = "Test", "type"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

    name, struc_type = "TYPE_Test", "type"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

    name, struc_type = "_TYPE_[Test]", "type"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

    name, struc_type = "type_[Test]", "type"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

    name, struc_type = "TYPE_[Test", "type"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

def test_extern_name_correct():
    name, struc_type = "TYPE_[Test_name]", "type"
    expected_name = "Test_name"
    assert extern_name(name, struc_type) == expected_name

    name, struc_type = "PRED_[Test_name]", "predicate"
    expected_name = "Test_name"
    assert extern_name(name, struc_type) == expected_name

    name, struc_type = "COND_[Test_name]", "conditional"
    expected_name = "Test_name"
    assert extern_name(name, struc_type) == expected_name
