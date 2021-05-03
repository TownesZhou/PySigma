"""
    Unit tests for variable_map module
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Union, Optional, Tuple, Callable
import torch
from torch.distributions import Distribution as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import MessageType, Message, Variable, VariableMetatype
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


class TestVariableMapPrototype:

    def test_property_pred_arg_pat_var_1(self):
        # Test function
        def test_func(x):
            return x + 3

        # Variables
        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 10)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)

        test_vmp = VariableMapPrototype(test_func, True, True)
        test_vmp(test_pred_arg, test_pat_var)

        assert test_vmp.pred_arg is test_pred_arg
        assert test_vmp.pat_var is test_pat_var

    def test_property_pred_arg_pat_var_2(self):
        # Test function
        def test_func(x):
            return x + 3

        # Variables
        test_pred_arg = 'test_pred_arg'
        test_pat_var = 'test_pat_var'

        test_vmp = VariableMapPrototype(test_func, True, True)
        test_vmp(test_pred_arg, test_pat_var)

        assert test_vmp.pred_arg is test_pred_arg
        assert test_vmp.pat_var is test_pat_var

    def test_property_variable_map_1(self):
        # Test function
        def test_func(x):
            return x + 3
        # Mock variable_map callable
        mock_variable_map = MagicMock(spec_set=VariableMapCallable)

        test_vmp = VariableMapPrototype(test_func, True, True)
        test_vmp._variable_map = mock_variable_map

        assert test_vmp.variable_map is mock_variable_map

    def test_property_variable_map_2(self):
        # Test function
        def test_func(x):
            return x + 3
        # Mock variable_map callable

        test_vmp = VariableMapPrototype(test_func, True, True)
        test_vmp._variable_map = None

        with pytest.raises(AssertionError) as excinfo:
            _ = test_vmp.variable_map

        assert str(excinfo.value) == "Predicate argument and/or pattern variable are not finalized."

    def test_call_1(self):
        # Test function
        def test_func(x):
            return x + 3

        # Variables
        test_pred_arg = 'test_pred_arg'
        test_pat_var = 'test_pat_var'

        test_vmp = VariableMapPrototype(test_func, True, True)
        return_obj = test_vmp(test_pred_arg, test_pat_var)

        assert test_vmp._variable_map is None
        assert isinstance(return_obj, VariableMapPrototype)

    def test_call_2(self):
        # Test function
        def test_func(x):
            return x + 3

        # Variables
        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 10)
        test_pat_var = 'test_pat_var'

        test_vmp = VariableMapPrototype(test_func, True, True)
        return_obj = test_vmp(test_pred_arg, test_pat_var)

        assert test_vmp._variable_map is None
        assert isinstance(return_obj, VariableMapPrototype)

    def test_call_3(self):
        # Test function
        def test_func(x):
            return x + 3

        # Variables
        test_pred_arg = 'test_pred_arg'
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)

        test_vmp = VariableMapPrototype(test_func, True, True)
        return_obj = test_vmp(test_pred_arg, test_pat_var)

        assert test_vmp._variable_map is None
        assert isinstance(return_obj, VariableMapPrototype)

    def test_call_4(self):
        # Test function
        def test_func(x):
            return x + 3

        # Variables
        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 10)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)

        test_vmp = VariableMapPrototype(test_func, True, True)
        return_obj = test_vmp(test_pred_arg, test_pat_var)

        assert isinstance(test_vmp._variable_map, VariableMapCallable)
        assert isinstance(return_obj, VariableMapPrototype)


class TestVariableMapCallable:

    def test_init_dynamic(self):
        # Test that set_map is not called during init
        # Test function
        def test_func(x):
            return x + 3
        # Variables
        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 10)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)
        # Mock set_map callable
        mock_callable = MagicMock()

        # Path the _set_map() method
        with patch.object(VariableMapCallable, '_set_map', mock_callable):
            test_vmc = VariableMapCallable(test_func, test_pred_arg, test_pat_var, True, True)

        # Check that the _set_map method was not called
        mock_callable.assert_not_called()

    def test_init_not_dynamic(self):
        # Test that set_map is not called during init
        # Test function
        def test_func(x):
            return x + 3
        # Variables
        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 10)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)
        # Mock set_map callable
        mock_callable = MagicMock()

        # Path the _set_map() method
        with patch.object(VariableMapCallable, '_set_map', mock_callable):
            test_vmc = VariableMapCallable(test_func, test_pred_arg, test_pat_var, False, True)

        # check that the _set_map method was called exactly once
        mock_callable.assert_called_once()

    def test_call_new_call(self):
        # Test return value correctness and _set_map() is actually called if new_call is true
        mock_func = MagicMock()
        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 10)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)
        # Mock that new_call property returns True
        with patch.object(VariableMapCallable, 'new_call', new_callable=PropertyMock) as mock_new_call:
            mock_new_call.return_value = True
            # Mock the _set_map method
            with patch.object(VariableMapCallable, '_set_map', new_callable=MagicMock) as mock_set_map:
                # Make a dynamic VMC so that _set_map was not called during init
                test_vmc = VariableMapCallable(mock_func, test_pred_arg, test_pat_var, True, True)
                # Mock result_cache
                mock_cache = MagicMock()
                test_vmc.result_cache = mock_cache

                return_val = test_vmc()

                # Check that _set_map was called exactly once
                mock_set_map.assert_called_once()
                assert return_val is mock_cache

    def test_call_not_new_call(self):
        # Test return value correctness and _set_map() is NOT called if new_call is False
        mock_func = MagicMock()
        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 10)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)
        # Mock that new_call property returns False
        with patch.object(VariableMapCallable, 'new_call', new_callable=PropertyMock) as mock_new_call:
            mock_new_call.return_value = False
            # Mock the _set_map method
            with patch.object(VariableMapCallable, '_set_map', new_callable=MagicMock) as mock_set_map:
                # Make a dynamic VMC so that _set_map was not called during init
                test_vmc = VariableMapCallable(mock_func, test_pred_arg, test_pat_var, True, True)
                # Mock result_cache
                mock_cache = MagicMock()
                test_vmc.result_cache = mock_cache

                return_val = test_vmc()

                # Check that _set_map was never called
                mock_set_map.assert_not_called()
                assert return_val is mock_cache

    def test_property_new_call_dynamic(self):
        # Test the correct value of the new_call property when the VMC is dynamic
        mock_func = MagicMock()
        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 10)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)
        # Mock the _set_map method to do nothing
        with patch.object(VariableMapCallable, '_set_map', new_callable=MagicMock):
            # Make a dynamic VMC
            test_vmc = VariableMapCallable(mock_func, test_pred_arg, test_pat_var, True, True)

            # Before first call, new_call should be True
            assert test_vmc.new_call

            # After one call, new_call should be False
            test_vmc()
            assert not test_vmc.new_call

            # Once set new_call to True, it should be True
            test_vmc.new_call = True
            assert test_vmc.new_call

    def test_property_new_call_not_dynamic(self):
        # Test the correct value of the new_call property when the VMC is NOT dynamic
        mock_func = MagicMock()
        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 10)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)
        # Mock the _set_map method to do nothing
        with patch.object(VariableMapCallable, '_set_map', new_callable=MagicMock):
            # Make a dynamic VMC
            test_vmc = VariableMapCallable(mock_func, test_pred_arg, test_pat_var, False, True)

            # Before first call, new_call is also False
            assert not test_vmc.new_call

            # After one call, new_call is also False
            test_vmc()
            assert not test_vmc.new_call

            # Even after set new_call to True, it is still False
            test_vmc.new_call = True
            assert not test_vmc.new_call

    def test_set_map_callback_exception(self):
        # Test that exception from the callback is handled.
        # Test func that raises some exception
        def test_exc_func(x):
            raise ValueError(x)

        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 12)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)

        # Make a dynamic vmc so _set_map is not called during init
        test_vmc = VariableMapCallable(test_exc_func, test_pred_arg, test_pat_var, True, True)

        with pytest.raises(VariableMapRuntimeError) as excinfo:
            test_vmc._set_map()

        assert str(excinfo.value) == "In VariableMap mapping pattern variable 'test_pat_var' to predicate argument " \
                                     "'test_pred_arg' with callback 'test_exc_func': unable to proceed. An exception " \
                                     "internal to the callback occurred."

    def test_set_map_return_value_wrong_type(self):
        # Test that exception is raised if return values has wrong type
        # Test func that returns some random value
        def test_exc_func(x):
            return [1, 2, 3]

        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 12)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)
        # Make a dynamic vmc so _set_map is not called during init
        test_vmc = VariableMapCallable(test_exc_func, test_pred_arg, test_pat_var, True, True)

        with pytest.raises(VariableMapValueError) as excinfo:
            test_vmc._set_map()

        assert str(excinfo.value) ==  "In VariableMap mapping pattern variable 'test_pat_var' to predicate argument " \
                                      "'test_pred_arg' with callback 'test_exc_func': the return value from the " \
                                      "custom callback must be a LongTensor. Found return value type: '{}'".format(list)

    def test_set_map_return_value_wrong_dimension(self):
        # Test that exception is raised if return value has wrong shape
        # Test func that returns some wrong shape LongTensor
        def test_exc_func(x):
            return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 12)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 12)
        # Make a dynamic vmc so _set_map is not called during init
        test_vmc = VariableMapCallable(test_exc_func, test_pred_arg, test_pat_var, True, True)

        with pytest.raises(VariableMapValueError) as excinfo:
            test_vmc._set_map()

        assert str(excinfo.value) ==  "In VariableMap mapping pattern variable 'test_pat_var' to predicate argument " \
                                      "'test_pred_arg' with callback 'test_exc_func': the returned LongTensor must " \
                                      "have the same shape as the input LongTensor. Expect shape torch.Size([12]), " \
                                      "but found shape torch.Size([2, 3])."

    def test_set_map_strict_value_check_1(self):
        # Test that when strict, out-of-range output values are reported via raising exceptions.
        def test_exc_func(x):
            return x + 1

        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 3)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 3)
        # Make a dynamic vmc so _set_map is not called during init
        test_vmc = VariableMapCallable(test_exc_func, test_pred_arg, test_pat_var, True, True)

        with pytest.raises(VariableMapValueError) as excinfo:
            test_vmc._set_map()

        assert str(excinfo.value) == "In VariableMap mapping pattern variable 'test_pat_var' to predicate argument " \
                                     "'test_pred_arg' with callback 'test_exc_func': returned LongTensor value out " \
                                     "of range. Expect values in range [0, 2], but found maximum value 3 and minimum " \
                                     "value 1."

    def test_set_map_strict_value_check_2(self):
        # Test that when strict, out-of-range output values are reported via raising exceptions.
        def test_exc_func(x):
            return x // 2 - 1

        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 6)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 6)
        # Make a dynamic vmc so _set_map is not called during init
        test_vmc = VariableMapCallable(test_exc_func, test_pred_arg, test_pat_var, True, True)

        with pytest.raises(VariableMapValueError) as excinfo:
            test_vmc._set_map()

        assert str(excinfo.value) == "In VariableMap mapping pattern variable 'test_pat_var' to predicate argument " \
                                     "'test_pred_arg' with callback 'test_exc_func': returned LongTensor value out " \
                                     "of range. Expect values in range [0, 5], but found maximum value 1 and minimum " \
                                     "value -1."

    def test_set_map_non_strict_exclude_invalid_mapping(self):
        # Test that when non strict, out-of-range mapping are excluded in the return value
        def test_exc_func(x):
            return x * 2 + 1

        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 5)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 3)
        # Make a dynamic vmc so _set_map is not called during init, and non strict value check
        test_vmc = VariableMapCallable(test_exc_func, test_pred_arg, test_pat_var, True, False)

        expected_in = torch.tensor([0, 1], dtype=torch.long)
        expected_out = torch.tensor([1, 3], dtype=torch.long)
        test_vmc._set_map()

        assert torch.equal(test_vmc.result_cache[0], expected_in)
        assert torch.equal(test_vmc.result_cache[1], expected_out)

    def test_set_map_non_strict_permutation(self):
        # Test with a permutation mapping
        def test_perm(x):
            y = x + 1   # add 1 to each value
            y[-1] = 0
            return y

        test_pred_arg = Variable('test_pred_arg', VariableMetatype.Relational, 5)
        test_pat_var = Variable('test_pat_var', VariableMetatype.Relational, 5)
        # Make a dynamic vmc so _set_map is not called during init, and non strict value check
        test_vmc = VariableMapCallable(test_perm, test_pred_arg, test_pat_var, True, False)

        expected_in = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        expected_out = torch.tensor([1, 2, 3, 4, 0], dtype=torch.long)
        test_vmc._set_map()

        assert torch.equal(test_vmc.result_cache[0], expected_in)
        assert torch.equal(test_vmc.result_cache[1], expected_out)


