"""
    Unit tests for Factor Function related pattern structures
"""
import pytest
from typing import Callable
import torch
from torch.nn import Module
from torch.distributions.transforms import ExpTransform, AffineTransform
import torch.distributions.constraints as C

from pysigma.defs import Variable, VariableMetatype
from pysigma.pattern_structures.factor_function import FactorFunction, FactorFunctionRuntimeError, \
    FactorFunctionValueError
from tests.utils import assert_equal_within_error, cuda_only


def gen_module_callable(func: Callable):
    # Define a custom Module subclass
    class TempModule(Module):
        def forward(self, *args):
            return func(*args)
    # Return an instance of the temp class
    return TempModule()


class TestFactorFunction_Init:

    def test_incorrect_vector_func_specified_with_one_ran_var(self):
        # Test AssertionError is raised if vector_func is specified but only one random variable is specified.
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
        ]
        # Vector func
        vector_func = [gen_module_callable(lambda x: x + 4)]
        with pytest.raises(AssertionError) as excinfo:
            ff = FactorFunction(ran_vars, vector_func=vector_func)
        assert str(excinfo.value) == "If `vector_func` is specified, `ran_vars` must contain at least two Variables."

    def test_incorrect_vector_func_all_none(self):
        # Test AssertionError is raised if vector_func is a list of all None
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Vector func
        vector_func = [None, None, None]

        with pytest.raises(AssertionError) as excinfo:
            ff = FactorFunction(ran_vars, vector_func=vector_func)
        assert str(excinfo.value) == "If specified, at least one element in `vector_func` must be a torch nn module."

    def test_incorrect_vector_func_wrong_length(self):
        # Test AssertionError is raised if vector_func has the wrong length
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        # Vector func
        vector_func = [None, gen_module_callable(lambda x: x + 4)]  # Should have 3 items

        with pytest.raises(AssertionError) as excinfo:
            ff = FactorFunction(ran_vars, vector_func=vector_func)
        assert str(excinfo.value) == "`vector_func` must have the same length as `ran_vars`. Expect length 3, " \
                                     "instead found 2."

    def test_incorrect_no_function_specified(self):
        # Test AssertionError is raised if neither prob or vector function is specified
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]

        with pytest.raises(AssertionError) as excinfo:
            ff = FactorFunction(ran_vars)
        assert str(excinfo.value) == "At least one of `prob_func` or `vector_func` must be specified."

    def test_incorrect_vector_func_fully_specified_when_given_prob_func(self):
        # Test AssertionError is raised if vector_func is fully specified when prob_func is given
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        # Vector func
        prob_func = gen_module_callable(lambda x: x)
        vector_func = [gen_module_callable(lambda x: x + 3),
                       gen_module_callable(lambda x: x + 4),
                       gen_module_callable(lambda x: x + 5), ]

        with pytest.raises(AssertionError) as excinfo:
            ff = FactorFunction(ran_vars, prob_func=prob_func, vector_func=vector_func)
        assert str(excinfo.value) == "When `prob_func` is specified, `vector_func` shall not be fully specified."

    def test_correct_prob_func_only(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        # Prob func
        prob_func = gen_module_callable(lambda x: x)

        ff = FactorFunction(ran_vars, prob_func=prob_func)

        assert ff.ran_vars == tuple(ran_vars)
        assert ff._prob_func is prob_func

    def test_correct_vector_func_only(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        # Vector func
        vector_func = [gen_module_callable(lambda x: x + 3),
                       gen_module_callable(lambda x: x + 4),
                       gen_module_callable(lambda x: x + 5), ]

        ff = FactorFunction(ran_vars, vector_func=vector_func)

        assert ff.ran_vars == tuple(ran_vars)
        assert ff._vector_func == tuple(vector_func)

    def test_correct_prob_func_and_vector_func(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        # Prob func
        prob_func = gen_module_callable(lambda x: x)
        # Vector func
        vector_func = [None,
                       gen_module_callable(lambda x: x + 4),
                       gen_module_callable(lambda x: x + 5), ]

        ff = FactorFunction(ran_vars, prob_func=prob_func, vector_func=vector_func)

        assert ff.ran_vars == tuple(ran_vars)
        assert ff._prob_func is prob_func
        assert ff._vector_func == tuple(vector_func)


class TestFactorFunction_Property:

    def test_finalized_1(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        # Vector func
        vector_func = [gen_module_callable(lambda x: x + 3),
                       gen_module_callable(lambda x: x + 4),
                       gen_module_callable(lambda x: x + 5)]

        ff = FactorFunction(ran_vars, vector_func=vector_func)

        assert ff.finalized

    def test_finalized_2(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            'test_ran_var_2',
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        # Vector func
        vector_func = [gen_module_callable(lambda x: x + 3),
                       gen_module_callable(lambda x: x + 4),
                       gen_module_callable(lambda x: x + 5)]

        ff = FactorFunction(ran_vars, vector_func=vector_func)

        assert not ff.finalized

    def test_generative_1(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        # Vector func
        prob_func = gen_module_callable(lambda x: x)
        vector_func = [None,
                       gen_module_callable(lambda x: x + 4),
                       gen_module_callable(lambda x: x + 5)]

        ff = FactorFunction(ran_vars, prob_func=prob_func, vector_func=vector_func)

        assert ff.generative

    def test_generative_2(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        # Vector func
        vector_func = [gen_module_callable(lambda x: x + 3),
                       gen_module_callable(lambda x: x + 4),
                       gen_module_callable(lambda x: x + 5)]

        ff = FactorFunction(ran_vars, vector_func=vector_func)

        assert not ff.generative

    def test_discriminative_1(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        prob_func = gen_module_callable(lambda x: x)

        ff = FactorFunction(ran_vars, prob_func=prob_func)

        assert not ff.discriminative

    def test_discriminative_2(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        prob_func = gen_module_callable(lambda x: x)
        vector_func = [None,
                       gen_module_callable(lambda x: x + 4),
                       gen_module_callable(lambda x: x + 5)]

        ff = FactorFunction(ran_vars, prob_func=prob_func, vector_func=vector_func)

        assert ff.discriminative

    def test_directional_vars(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        prob_func = gen_module_callable(lambda x: x)
        vector_func = [None,
                       gen_module_callable(lambda x: x + 4),
                       gen_module_callable(lambda x: x + 5)]

        ff = FactorFunction(ran_vars, prob_func=prob_func, vector_func=vector_func)

        assert ff.directional_vars == tuple(ran_vars[1:])

    def test_directional_vars_no_vector_func(self):
        # Test directional_vars values when no vector_func is specified
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        prob_func = gen_module_callable(lambda x: x)
        ff = FactorFunction(ran_vars, prob_func=prob_func)

        assert ff.directional_vars == ()


class TestFactorFunction_Call:

    def test_generative_callback_error(self):
        # Test RuntimeError is raised if the callback throws an internal error.
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Let the callback raise an error
        @gen_module_callable
        def prob_func(x1, x2, x3):
            raise Exception("something went wrong")

        ff = FactorFunction(ran_vars, prob_func=prob_func)

        with pytest.raises(FactorFunctionRuntimeError) as excinfo:
            ff(inputs, func_type='generative')
        assert str(excinfo.value) == "An exception occurred when calling the generative joint probability density " \
                                     "factor function {}.".format(prob_func)

    def test_generative_wrong_result_type(self):
        # Test ValueError is raised if result is not a torch tensor
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Returns something not a tensor
        @gen_module_callable
        def prob_func(x1, x2, x3):
            return "something random"

        ff = FactorFunction(ran_vars, prob_func=prob_func)

        with pytest.raises(FactorFunctionValueError) as excinfo:
            ff(inputs, func_type='generative')
        assert str(excinfo.value) == "The returning value from a factor function callback must be a torch tensor. " \
                                     "Found value with type {}.".format(str)

    def test_generative_wrong_result_shape(self):
        # Test ValueError is raised if result has wrong shape
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Returns something with a wrong shape
        @gen_module_callable
        def prob_func(x1, x2, x3):
            return torch.randn(11)

        ff = FactorFunction(ran_vars, prob_func=prob_func)

        with pytest.raises(FactorFunctionValueError) as excinfo:
            ff(inputs, func_type='generative')
        assert str(excinfo.value) == "The returning tensor from a factor function must represent batched probability " \
                                     "densities. Expect shape {}, instead found shape {}."\
                                     .format(torch.Size([b_size]), torch.Size([11]))

    def test_generative_wrong_result_value(self):
        # Test ValueError is raised if result has negative values
        # Test ValueError is raised if result has wrong shape
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Returns something with a wrong shape
        return_val = torch.randn(b_size) - 1
        @gen_module_callable
        def prob_func(x1, x2, x3):
            return return_val

        ff = FactorFunction(ran_vars, prob_func=prob_func)

        with pytest.raises(FactorFunctionValueError) as excinfo:
            ff(inputs, func_type='generative')
        assert str(excinfo.value) == "The returning tensor from a factor function must represent batched probability " \
                                     "densities. Expect all values in the tensor be non-negative, instead found " \
                                     "minimum value {}.".format(return_val.min())

    def test_generative_correct_result(self):
        # Test that correct result is relayed
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        @gen_module_callable
        def prob_func(x1, x2, x3):
            return (x1.sum(dim=1) + x2.sum(dim=1) + x3.sum(dim=1)).abs()

        ff = FactorFunction(ran_vars, prob_func=prob_func)

        expected_return_val = (inputs[0].sum(dim=1) + inputs[1].sum(dim=1) + inputs[2].sum(dim=1)).abs()
        return_val = ff(inputs, func_type='generative')

        assert_equal_within_error(return_val, expected_return_val)

    @cuda_only
    def test_generative_correct_callback_device(self):
        # Test that tensors are converted to the callback's device before feeding them to the callback
        # callback device: gpu, node device: cpu
        cb_device, node_device = torch.device('cuda:{}'.format(torch.cuda.current_device())), torch.device('cpu')
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        @gen_module_callable
        def prob_func(x1, x2, x3):
            # Check that inputs are on the correct device
            assert x1.device == cb_device
            assert x2.device == cb_device
            assert x3.device == cb_device
            return (x1.sum(dim=1) + x2.sum(dim=1) + x3.sum(dim=1)).abs()

        ff = FactorFunction(ran_vars, prob_func=prob_func, device=cb_device)

        return_val = ff(inputs, func_type='generative')
        # Check that return value is on the correct device
        assert return_val.device == node_device

    @cuda_only
    def test_generative_correct_node_device(self):
        # Test that tensors are converted to the node's device before returning the value
        # callback device: cpu, node device: gpu
        cb_device, node_device = torch.device('cpu'), torch.device('cuda:{}'.format(torch.cuda.current_device()))
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        @gen_module_callable
        def prob_func(x1, x2, x3):
            # Check that inputs are on the correct device
            assert x1.device == cb_device
            assert x2.device == cb_device
            assert x3.device == cb_device
            return (x1.sum(dim=1) + x2.sum(dim=1) + x3.sum(dim=1)).abs()

        ff = FactorFunction(ran_vars, prob_func=prob_func, device=cb_device)

        return_val = ff(inputs, func_type='generative', node_device=node_device)
        # Check that return value is on the correct device
        assert return_val.device == node_device

    def test_discriminative_invalid_target_var(self):
        # Test AssertionError is raised if target_var is not one of the directional variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Random return value
        @gen_module_callable
        def x1_func(x2, x3):
            return torch.randn([b_size, 3])

        @gen_module_callable
        def x2_func(x1, x3):
            return torch.randn([b_size, 4])

        ff = FactorFunction(ran_vars, vector_func=[x1_func, x2_func, None])   # x3 is not a directional variable

        with pytest.raises(AssertionError) as excinfo:
            ff(inputs, target_var=ran_vars[2], func_type='discriminative')
        assert str(excinfo.value) == "The target_var 'test_ran_var_3' is not one of the directional variables {} for " \
                                     "which a discriminative function is specified." \
                                     .format(tuple(ran_vars[:2]))

    def test_discriminative_callback_error(self):
        # Test AssertionError is raised if the callback raises internal error
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Raise an error
        @gen_module_callable
        def x1_func(x2, x3):
            raise Exception("something went wrong")

        @gen_module_callable
        def x2_func(x1, x3):
            return torch.randn([b_size, 4])

        @gen_module_callable
        def x3_func(x1, x2):
            return torch.randn([b_size, 5])

        ff = FactorFunction(ran_vars, vector_func=[x1_func, x2_func, x3_func])

        with pytest.raises(FactorFunctionRuntimeError) as excinfo:
            ff(inputs, target_var=ran_vars[0], func_type='discriminative')
        assert str(excinfo.value) == "An exception occurred when calling the discriminative factor function {} " \
                                     "corresponding to the random variable {}.".format(x1_func, ran_vars[0])

    def test_discriminative_wrong_result_type(self):
        # Test AssertionError is raised if result from the callback is not a torch tensor
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Return something not a tensor
        @gen_module_callable
        def x1_func(x2, x3):
            return "something random"

        @gen_module_callable
        def x2_func(x1, x3):
            return torch.randn([b_size, 4])

        @gen_module_callable
        def x3_func(x1, x2):
            return torch.randn([b_size, 5])

        ff = FactorFunction(ran_vars, vector_func=[x1_func, x2_func, x3_func])

        with pytest.raises(FactorFunctionValueError) as excinfo:
            ff(inputs, target_var=ran_vars[0], func_type='discriminative')
        assert str(excinfo.value) == "The returning value from a factor function callback must be a torch tensor. " \
                                     "Found value with type {} in the discriminative factor function {} corresponding "\
                                     "to the random variable {}.".format(str, x1_func, ran_vars[0])

    def test_discriminative_wrong_result_dimension(self):
        # Test AssertionError is raised if result has wrong dimension
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Return a tensor with wrong dimension
        @gen_module_callable
        def x1_func(x2, x3):
            return torch.randn([b_size])

        @gen_module_callable
        def x2_func(x1, x3):
            return torch.randn([b_size, 4])

        @gen_module_callable
        def x3_func(x1, x2):
            return torch.randn([b_size, 5])

        ff = FactorFunction(ran_vars, vector_func=[x1_func, x2_func, x3_func])

        with pytest.raises(FactorFunctionValueError) as excinfo:
            ff(inputs, target_var=ran_vars[0], func_type='discriminative')
        assert str(excinfo.value) == "The returning tensor from a discriminative callback must represent batched " \
                                     "particles. Expect a 2-dimensional tensor. Instead, found a tensor with {} " \
                                     "dimensions in the discriminative factor function {} corresponding to the random "\
                                     "variable {}.".format(1, x1_func, ran_vars[0])

    def test_discriminative_wrong_result_shape(self):
        # Test AssertionError is raised if result has wrong shape
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Return a tensor with wrong shape
        @gen_module_callable
        def x1_func(x2, x3):
            return torch.randn([b_size, 4])

        @gen_module_callable
        def x2_func(x1, x3):
            return torch.randn([b_size, 4])

        @gen_module_callable
        def x3_func(x1, x2):
            return torch.randn([b_size, 5])

        ff = FactorFunction(ran_vars, vector_func=[x1_func, x2_func, x3_func])

        with pytest.raises(FactorFunctionValueError) as excinfo:
            ff(inputs, target_var=ran_vars[0], func_type='discriminative')
        assert str(excinfo.value) == "The returning tensor from a discriminative callback must represent batched " \
                                     "particles. Expect a tensor with shape {}. Instead, found a tensor with shape {} "\
                                     "in the discriminative factor function {} corresponding to the random variable " \
                                     "{}."\
                                     .format(torch.Size([b_size, 3]), torch.Size([b_size, 4]), x1_func, ran_vars[0])

    def test_discriminative_correct_result(self):
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]
        outputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        @gen_module_callable
        def x1_func(x2, x3):
            return outputs[0]

        @gen_module_callable
        def x2_func(x1, x3):
            return outputs[1]

        @gen_module_callable
        def x3_func(x1, x2):
            return outputs[2]

        ff = FactorFunction(ran_vars, vector_func=[x1_func, x2_func, x3_func])
        x1_result = ff(inputs, target_var=ran_vars[0], func_type='discriminative')
        x2_result = ff(inputs, target_var=ran_vars[1], func_type='discriminative')
        x3_result = ff(inputs, target_var=ran_vars[2], func_type='discriminative')

        assert [x1_result, x2_result, x3_result] == outputs

    def test_discriminative_correct_callback_device(self):
        # Test that input tensors are converted to the correct callback device before feeding to the callback
        # Devices
        cb_device, node_device = torch.device('cuda:{}'.format(torch.cuda.current_device())), torch.device('cpu')
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]
        outputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Check that input are on the correct devices
        @gen_module_callable
        def x1_func(x2, x3):
            assert x2.device == x3.device == cb_device
            return outputs[0]

        @gen_module_callable
        def x2_func(x1, x3):
            assert x1.device == x3.device == cb_device
            return outputs[1]

        @gen_module_callable
        def x3_func(x1, x2):
            assert x1.device == x2.device == cb_device
            return outputs[2]

        ff = FactorFunction(ran_vars, vector_func=[x1_func, x2_func, x3_func], device=cb_device)
        x1_result = ff(inputs, target_var=ran_vars[0], func_type='discriminative')
        x2_result = ff(inputs, target_var=ran_vars[1], func_type='discriminative')
        x3_result = ff(inputs, target_var=ran_vars[2], func_type='discriminative')

    def test_discriminative_correct_node_device(self):
        # Test that result tensors are converted to the correct node device before returning
        # Devices
        cb_device, node_device = torch.device('cpu'), torch.device('cuda:{}'.format(torch.cuda.current_device()))
        # Variables
        ran_vars = [
            Variable('test_ran_var_1', VariableMetatype.Random, 3, (C.real,)),
            Variable('test_ran_var_2', VariableMetatype.Random, 4, (C.real,)),
            Variable('test_ran_var_3', VariableMetatype.Random, 5, (C.real,)),
        ]
        # Test input tensors
        b_size = 10
        inputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]
        outputs = [torch.randn(b_size, 3), torch.randn(b_size, 4), torch.randn(b_size, 5)]

        # Check that input are on the correct devices
        @gen_module_callable
        def x1_func(x2, x3):
            assert x2.device == x3.device == cb_device
            return outputs[0]

        @gen_module_callable
        def x2_func(x1, x3):
            assert x1.device == x3.device == cb_device
            return outputs[1]

        @gen_module_callable
        def x3_func(x1, x2):
            assert x1.device == x2.device == cb_device
            return outputs[2]

        ff = FactorFunction(ran_vars, vector_func=[x1_func, x2_func, x3_func], device=cb_device)
        x1_result = ff(inputs, target_var=ran_vars[0], func_type='discriminative', node_device=node_device)
        x2_result = ff(inputs, target_var=ran_vars[1], func_type='discriminative', node_device=node_device)
        x3_result = ff(inputs, target_var=ran_vars[2], func_type='discriminative', node_device=node_device)
        assert all(r.device == node_device for r in [x1_result, x2_result, x3_result])


