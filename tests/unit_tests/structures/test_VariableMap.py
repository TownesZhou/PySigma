"""
    Unit tests for variable_map module
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import Union, Optional, Tuple, Callable
import torch
from torch.distributions import Distribution as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import MessageType, Message
from pysigma.pattern_structures.variable_map import VariableMap, VariableMapPrototype, VariableMapCallable, \
    VariableMapInitializationError, VariableMapRuntimeError, VariableMapValueError
from pysigma.pattern_structures.variable_map import DYNAMIC_DEFAULT_VAL, STRICT_DEFAULT_VAL

from ...utils import random_message, assert_equal_within_error, equal_within_error, assert_proportional_within_error


class TestVariableMap:

    def test_no_param(self):
        # Test return value correctness using the decorator without parameters
        # Test func
        @VariableMap
        def test_func(x):
            return x + 2

        assert isinstance(test_func, VariableMapPrototype)
        assert test_func.dynamic == DYNAMIC_DEFAULT_VAL
        assert test_func.strict == STRICT_DEFAULT_VAL
        assert callable(test_func.func) and test_func.func(2) == 4

    def test_default_param(self):
        # Test return value correctness without specifying parameters but with parentheses
        # Test func
        @VariableMap()
        def test_func(x):
            return x + 2

        assert isinstance(test_func, VariableMapPrototype)
        assert test_func.dynamic == DYNAMIC_DEFAULT_VAL
        assert test_func.strict == STRICT_DEFAULT_VAL
        assert callable(test_func.func) and test_func.func(2) == 4

    def test_with_param(self):
        # Test return value correctness with param
        @VariableMap(False, False)
        def test_func(x):
            return x + 2

        assert isinstance(test_func, VariableMapPrototype)
        assert not test_func.dynamic
        assert not test_func.strict
        assert callable(test_func.func) and test_func.func(2) == 4

    def test_invalid_param_1(self):
        # Test assertion error is raised if param type is invalid
        with pytest.raises(ValueError) as excinfo:
            @VariableMap("something random", False)
            def test_func(x):
                return x + 2
        assert str(excinfo.value) == "Unrecognizable argument type for the VariableMap decorator: {}".format(
            [str, bool])

    def test_invalid_param_2(self):
        # Test assertion error is raised if param type is invalid
        with pytest.raises(ValueError) as excinfo:
            @VariableMap(False, 1)
            def test_func(x):
                return x + 2
        assert str(excinfo.value) == "Unrecognizable argument type for the VariableMap decorator: {}".format(
            [bool, int])
