"""
    Unit tests for Summarization module
"""
import pytest
from unittest.mock import MagicMock, patch
from typing import Union, Optional, Tuple, Callable
import torch
from torch.distributions import Distribution as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import MessageType, Message
from pysigma.pattern_structures.summarization import Summarization, SummarizationClass
from pysigma.pattern_structures.summarization import ContentFlagTyping as CFT
from pysigma.pattern_structures.summarization import DistributionTyping as DT
from pysigma.pattern_structures.summarization import ParticleTyping as PT
from pysigma.pattern_structures.summarization import WeightTyping as WT
from pysigma.pattern_structures.summarization import LogDensityTyping as LDT
from pysigma.pattern_structures.summarization import ReturnValueTyping as RVT
from pysigma.pattern_structures.summarization import SummarizationCallbackAnnotationError, \
    SummarizationContentTypeError, SummarizationValueError

from ...utils import random_message, assert_equal_within_error, equal_within_error, assert_proportional_within_error


class TestSummarization:

    # Utils
    # Custom ad-hoc Exception for avoiding the execution of the rest of the code in the target method
    class CustomJailBreakingException(Exception):
        pass

    def test_init_wrong_number_of_arguments(self):
        # Test that exception is raised if the callback has wrong number of arguments
        # Ad-hoc method
        with pytest.raises(SummarizationCallbackAnnotationError) as excinfo:
            @Summarization
            def test_callback(a, b, c):
                return a, b, c

        assert str(excinfo.value) == "Wrong number of arguments: A Summarization callback routine should accept " \
                                     "exactly 5 arguments. Found 3 arguments in the callable test_callback."

    def test_init_missing_argument_annotations(self):
        # Test that exception is raised if missing annotations for some arguments
        with pytest.raises(SummarizationCallbackAnnotationError) as excinfo:
            @Summarization
            def test_callback(a: CFT, b: DT, c: PT, d, e: LDT) -> RVT:
                return a, b, c, d, e

        assert str(excinfo.value) == "Incomplete typing annotations: All arguments and the return value of the " \
                                     "summarization callback routine should be annotated. Found unannotated arguments "\
                                     "in the callback test_callback."

    def test_init_missing_return_value_annotation(self):
        # Test that exception is raised if missing annotations for some arguments
        with pytest.raises(SummarizationCallbackAnnotationError) as excinfo:
            @Summarization
            def test_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT):
                return a, b, c, d, e

        assert str(excinfo.value) == "Incomplete typing annotations: All arguments and the return value of the " \
                                     "summarization callback routine should be annotated. The return value is " \
                                     "unannotated in the callback test_callback."

    def test_init_incorrect_argument_annotations(self):
        # Test that exception is raised if certain annotations are incorrect
        with pytest.raises(SummarizationCallbackAnnotationError) as excinfo:
            @Summarization
            def test_callback(a: CFT, b: str, c: PT, d: str, e: LDT) -> RVT:
                return a, b, c, d, e

        assert str(excinfo.value) == "Incorrect argument typing annotation: The arguments ['b', 'd'] in the callback " \
                                     "test_callback have incorrect annotations."

    def test_init_incorrect_return_value_annotation(self):
        # Test that exception is raised if certain annotations are incorrect
        with pytest.raises(SummarizationCallbackAnnotationError) as excinfo:
            @Summarization
            def test_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> str:
                return a, b, c, d, e

        assert str(excinfo.value) == "Incorrect return value typing annotation: The return value of the callback " \
                                     "test_callback has incorrect annotations."

    def test_init_correct_no_decorator_arguments(self):
        @Summarization
        def test_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, None

        assert isinstance(test_callback, SummarizationClass)
        assert callable(test_callback._sum_func)
        assert test_callback._repr_type == 'dual' and test_callback._dist_repr_type == 'distribution'

    def test_init_incorrect_decorator_argument_1(self):
        # Test that exception is raised if repr_type value is incorrect
        with pytest.raises(AssertionError) as excinfo:
            @Summarization(repr_type='something random')
            def test_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
                return None, None

        assert str(excinfo.value) == "When decorating test_callback: decorator parameter `repr_type` can only " \
                                     "have value 'distribution', 'particle', or 'dual'. Encountered 'something random'."

    def test_init_incorrect_decorator_argument_2(self):
        # Test that exception is raised if repr_type value is incorrect
        with pytest.raises(AssertionError) as excinfo:
            @Summarization(dist_repr_type='something random')
            def test_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
                return None, None

        assert str(excinfo.value) == "When decorating test_callback: decorator parameter `dist_repr_type` can " \
                                     "only have value 'distribution' or 'parameter'. Encountered 'something random'."

    def test_init_correct_with_decorator_arguments(self):
        @Summarization(repr_type='particle', dist_repr_type='parameter')
        def test_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, None

        assert isinstance(test_callback, SummarizationClass)
        assert callable(test_callback._sum_func)
        assert test_callback._repr_type == 'particle' and test_callback._dist_repr_type == 'parameter'

    def test_call_identity_msg(self):
        # Test return message correctness for identity message
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        # Modify to identity message
        test_msg.parameter = 0
        test_msg.weight = 1

        @Summarization
        def test_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, None

        return_msg = test_callback(test_msg)
        assert isinstance(return_msg, Message)
        assert return_msg.type is MessageType.Dual and return_msg.isid
        assert return_msg.b_shape == b_shape[1:]

    def test_call_conflicting_input_content_type_1(self):
        # Test that AssertionError is raised if input message has a different type than what is requested
        #   by the callable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Parameter, b_shape, p_shape, s_shape, e_shape)

        @Summarization(repr_type='particle')
        def test_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, None

        with pytest.raises(SummarizationContentTypeError) as excinfo:
            test_callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'test_callback': the decorator is specified with " \
                                     "argument `repr_type=particle`. However, the incoming message contains only " \
                                     "distribution parameters. Please double check model configuration."

    def test_call_conflicting_input_content_type_2(self):
        # Test that exception is raised if input message has a different type than what is requested
        #   by the callable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Particles, b_shape, p_shape, s_shape, e_shape)

        @Summarization(repr_type='distribution')
        def test_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, None

        with pytest.raises(SummarizationContentTypeError) as excinfo:
            test_callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'test_callback': the decorator is specified with " \
                                     "argument `repr_type=distribution`. However, the incoming message contains only " \
                                     "particle contents. Please double check model configuration."

    def test_call_malformed_message_missing_attributes(self):
        # Test that exception message if, when needed, incoming message does not specify necessary attributes
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        @Summarization
        def test_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, None

        with pytest.raises(AssertionError) as excinfo:
            test_callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'test_callback': incoming message containing " \
                                     "distribution parameters does not specify both 'dist_info' and 'dist_class' " \
                                     "key-ed special attributes. Please double check if the initial message was " \
                                     "generated with these special attributes, or if the attribute dictionary was " \
                                     "corrupted during message propagation."

    def test_call_correct_callback_arguments_distribution_type_parameter_subtype(self):
        # Test that correct arguments are passed to the callable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        # Use a mock to test correctness of input argument
        # Callable mock.
        test_mock = MagicMock()

        # Use a wrapper method to wrap the mock and throw the ad-hoc exception. This is to guarantee the callback has
        #   correct annotation
        @Summarization(repr_type='distribution', dist_repr_type='parameter')
        def wrapper_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            # Pass arguments as-is to the mock
            test_mock(a, b, c, d, e)
            # Throw exception to break jail
            raise self.CustomJailBreakingException

        # Use pytest to catch the jail breaking exception
        with pytest.raises(self.CustomJailBreakingException):
            wrapper_callback(test_msg)

        # Using the mock to check argument correctness
        test_mock.assert_called_once()
        call_args = test_mock.call_args[0]
        assert call_args[0] == 'distribution'
        assert_equal_within_error(call_args[1], test_msg.parameter)
        assert call_args[2] is None
        assert call_args[3] is None
        assert call_args[4] is None

    def test_call_correct_callback_arguments_distribution_type_distribution_subtype(self):
        # Test that correct arguments are passed to the callable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': {'test_key', 'test_val'}}

        # Patch param2dist() method to return a mocked distribution
        mock_dist = MagicMock(spec_set=D)
        with patch("pysigma.utils.DistributionServer.param2dist") as d:
            d.side_effect = lambda arg_dist_class, arg_param, b_shape=None, e_shape=None, dist_info=None:  mock_dist \
                if arg_dist_class is D and \
                   equal_within_error(arg_param, test_msg.parameter) and \
                   dist_info == {'test_key', 'test_val'} \
                else None

            # Use a mock to test correctness of input argument
            # Callable mock.
            test_mock = MagicMock()

            # Use a wrapper method to wrap the mock and throw the ad-hoc exception. This is to guarantee the callback has
            #   correct annotation
            @Summarization(repr_type='distribution', dist_repr_type='distribution')
            def wrapper_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
                # Pass arguments as-is to the mock
                test_mock(a, b, c, d, e)
                # Throw exception to break jail
                raise self.CustomJailBreakingException

            # Use pytest to catch the jail breaking exception
            with pytest.raises(self.CustomJailBreakingException):
                wrapper_callback(test_msg)

            # Using the mock to check argument correctness
            test_mock.assert_called_once()
            call_args = test_mock.call_args[0]
            assert call_args[0] == 'distribution'
            assert call_args[1] is mock_dist
            assert call_args[2] is None
            assert call_args[3] is None
            assert call_args[4] is None

    def test_call_correct_callback_arguments_particle_type(self):
        # Test that correct arguments are passed to the callable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        # Use a mock to test correctness of input argument
        # Callable mock.
        test_mock = MagicMock()

        # Use a wrapper method to wrap the mock and throw the ad-hoc exception. This is to guarantee the callback has
        #   correct annotation
        @Summarization(repr_type='particle')
        def wrapper_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            # Pass arguments as-is to the mock
            test_mock(a, b, c, d, e)
            # Throw exception to break jail
            raise self.CustomJailBreakingException

        # Use pytest to catch the jail breaking exception
        with pytest.raises(self.CustomJailBreakingException):
            wrapper_callback(test_msg)

        # Using the mock to check argument correctness
        test_mock.assert_called_once()
        call_args = test_mock.call_args[0]
        assert call_args[0] == 'particle'
        assert call_args[1] is None
        assert_equal_within_error(call_args[2], test_msg.particles[0])
        assert_equal_within_error(call_args[3], test_msg.weight)
        assert_equal_within_error(call_args[4], test_msg.log_densities[0])

    def test_call_correct_callback_arguments_dual_type(self):
        # Test that correct arguments are passed to the callable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': {'test_key', 'test_val'}}

        # Patch param2dist() method to return a mocked distribution
        mock_dist = MagicMock(spec_set=D)
        with patch("pysigma.utils.DistributionServer.param2dist") as d:
            d.side_effect = lambda arg_dist_class, arg_param, b_shape=None, e_shape=None, dist_info=None:  mock_dist \
                if arg_dist_class is D and \
                   equal_within_error(arg_param, test_msg.parameter) and \
                   dist_info == {'test_key', 'test_val'} \
                else None

            # Use a mock to test correctness of input argument
            # Callable mock.
            test_mock = MagicMock()

            # Use a wrapper method to wrap the mock and throw the ad-hoc exception. This is to guarantee the callback has
            #   correct annotation
            @Summarization(repr_type='dual')
            def wrapper_callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
                # Pass arguments as-is to the mock
                test_mock(a, b, c, d, e)
                # Throw exception to break jail
                raise self.CustomJailBreakingException

            # Use pytest to catch the jail breaking exception
            with pytest.raises(self.CustomJailBreakingException):
                wrapper_callback(test_msg)

            # Using the mock to check argument correctness
            test_mock.assert_called_once()
            call_args = test_mock.call_args[0]
            assert call_args[0] == 'dual'
            assert call_args[1] is mock_dist
            assert_equal_within_error(call_args[2], test_msg.particles[0])
            assert_equal_within_error(call_args[3], test_msg.weight)
            assert_equal_within_error(call_args[4], test_msg.log_densities[0])

    def test_call_incorrect_callback_return_value_format_1(self):
        # Test that exception is raised if the return value from the callback has wrong type
        # The callback returns something random
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        # Custom callback that deliberately returns a bad value
        # Use 'parameter' distribution content subtype to avoid patching the param2dist() method
        @Summarization(repr_type='dual', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return "something random"

        with pytest.raises(SummarizationValueError) as excinfo:
            callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'callback': the value returned by the summarization " \
                                     "callback routine should be a 2-tuple. Found return value type {}."\
                                     .format(str)

    def test_call_incorrect_callback_return_value_format_2(self):
        # Test that exception is raised if the return value from the callback has wrong type
        # The callback returns a tuple but with wrong length
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        # Custom callback that deliberately returns a bad value
        # Use 'parameter' distribution content subtype to avoid patching the param2dist() method
        @Summarization(repr_type='dual', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, None, None

        with pytest.raises(SummarizationValueError) as excinfo:
            callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'callback': the value returned by the summarization " \
                                     "callback routine should be a 2-tuple. Found return value type {}."\
                                     .format(tuple)

    def test_call_incorrect_callback_first_return_value_type_1(self):
        # Test that exception is raised if the first return value has wrong type
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        # Custom callback that deliberately returns a bad value
        # Use 'parameter' distribution content subtype to avoid patching the param2dist() method
        @Summarization(repr_type='dual', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return "something random", None

        with pytest.raises(SummarizationValueError) as excinfo:
            callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'callback': the first return value should be either " \
                                     "None, a torch distribution, or a torch tensor. Found type {}."\
                                     .format(str)

    def test_call_incorrect_callback_first_return_value_type_2(self):
        # Test that exception is raised if the first return value has wrong type that is inconsistent with the
        #   specified distribution content subtype
        # case 1: when distribution is specified, returned a parameter
        # provided
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        return_val_1_err = torch.randn(b_shape[1:] + s_shape)       # Wrong: should be a distribution
        # Custom callback that deliberately returns a bad value
        with patch("pysigma.utils.DistributionServer.param2dist") as d:
            @Summarization(repr_type='distribution', dist_repr_type='distribution')
            def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
                return return_val_1_err, None

            with pytest.raises(SummarizationValueError) as excinfo:
                callback(test_msg)

            assert str(excinfo.value) == "In summarization routine 'callback': when 'dist_repr_type' is " \
                                         "'distribution', the first return value from the callable should be a " \
                                         "torch.distribution.Distribution. However, found a torch.Tensor."\
                                         .format(str)

    def test_call_incorrect_callback_first_return_value_type_3(self):
        # Test that exception is raised if the first return value has wrong type that is inconsistent with the
        #   specified distribution content subtype
        # case 2: when parameter is specified, returned a distribution
        # provided
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        return_val_1_err = MagicMock(spec=D)        # Wrong return val: should be a parameter
        # Custom callback that deliberately returns a bad value
        @Summarization(repr_type='distribution', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return return_val_1_err, None

        with pytest.raises(SummarizationValueError) as excinfo:
            callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'callback': when 'dist_repr_type' is 'parameter', the " \
                                     "first return value from the callable should be a torch.Tensor. However, found " \
                                     "a torch.distribution.Distribution."\
                                     .format(str)

    def test_call_incorrect_callback_second_return_value_type(self):
        # Test that exception is raised if the first return value has wrong type
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        # Custom callback that deliberately returns a bad value
        # Use 'parameter' distribution content subtype to avoid patching the param2dist() method
        @Summarization(repr_type='dual', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, "something random"

        with pytest.raises(SummarizationValueError) as excinfo:
            callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'callback': the second return value should be either " \
                                     "None or a torch Tensor. Found type {}."\
                                     .format(str)

    def test_call_incorrect_callback_first_return_value_None(self):
        # Test that exception is raised if the first return value is None when it shouldn't be
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        return_val_2 = torch.randn(b_shape[1:] + s_shape)
        # Custom callback that deliberately returns a bad value
        # Use 'parameter' distribution content subtype to avoid patching the param2dist() method
        @Summarization(repr_type='distribution', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, return_val_2

        with pytest.raises(SummarizationValueError) as excinfo:
            callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'callback': found None as the first return value while "\
                                     "input contains distribution contents."

    def test_call_incorrect_callback_second_return_value_None(self):
        # Test that exception is raised if the second return value is None when it shouldn't be
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        return_val_1 = torch.randn(b_shape[1:] + p_shape)
        # Custom callback that deliberately returns a bad value
        # Use 'parameter' distribution content subtype to avoid patching the param2dist() method
        @Summarization(repr_type='particle', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return return_val_1, None

        with pytest.raises(SummarizationValueError) as excinfo:
            callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'callback': found None as the second return value " \
                                     "while input contains particle contents."

    def test_call_incorrect_callback_first_return_value_shape_distribution(self):
    # Test that exception is raised if the first return value, as a distribution instance, has wrong shape
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': {'test_key', 'test_val'}}

        return_val_1_err = MagicMock(spec=D, batch_shape=Size([1, 4]))  # Wrong shape
        # Patch param2dist() method to mock its behavior
        with patch("pysigma.utils.DistributionServer.param2dist") as d:
            @Summarization(repr_type='distribution', dist_repr_type='distribution')
            def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
                return return_val_1_err, None

            with pytest.raises(SummarizationValueError) as excinfo:
                callback(test_msg)

            assert str(excinfo.value) == "In summarization routine 'callback': expect the returned distribution " \
                                         "instance (first return value) to have batch shape {}. However, found batch " \
                                         "shape {}." \
                                         .format(Size([4]), Size([1, 4]))

    def test_call_incorrect_callback_first_return_value_shape_parameter(self):
        # Test that exception is raised if the first return value, as parameter, has wrong shape
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        return_val_1_err = torch.randn(Size([3, 4, 10]))       # Wrong shape
        # Custom callback that deliberately returns a bad value
        # Use 'parameter' distribution content subtype to avoid patching the param2dist() method
        @Summarization(repr_type='distribution', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return return_val_1_err, None

        with pytest.raises(SummarizationValueError) as excinfo:
            callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'callback': expect the returned distribution parameter "\
                                     "(first return value) to have shape {}. However, found shape {}." \
                                     .format(Size([4, 10]), Size([3, 4, 10]))

    def test_call_incorrect_callback_second_return_value_shape(self):
        # Test that exception is raised if the second return value has wrong shape
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': None}

        return_val_2_err = torch.randn(Size([1, 4, 20]))  # Wrong shape
        # Custom callback that deliberately returns a bad value
        # Use 'parameter' distribution content subtype to avoid patching the param2dist() method
        @Summarization(repr_type='particle')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, return_val_2_err

        with pytest.raises(SummarizationValueError) as excinfo:
            callback(test_msg)

        assert str(excinfo.value) == "In summarization routine 'callback': expect the returned particle weight " \
                                     "(second return value) to have shape {}. However, found shape {}." \
                                     .format(Size([4, 20]), Size([1, 4, 20]))

    def test_call_correct_return_message_distribution_type_distribution_subtype(self):
        # Test that with a correct execution, the returned message is also correct
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': {'test_key', 'test_val'}}

        mock_dist = MagicMock(D, batch_shape=Size([4]))
        mock_dist_param = torch.rand([4, 10])
        # Patch param2dist() method
        with patch("pysigma.utils.DistributionServer.param2dist") as p2d:
            # Patch dist2param() method
            with patch("pysigma.utils.DistributionServer.dist2param") as d2p:
                d2p.side_effect = lambda dist, dist_info: mock_dist_param \
                    if dist is mock_dist and dist_info == {'test_key', 'test_val'} else None

                @Summarization(repr_type='distribution', dist_repr_type='distribution')
                def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
                    return mock_dist, None

                return_msg = callback(test_msg)

        assert isinstance(return_msg, Message)
        assert return_msg.type is MessageType.Parameter
        assert return_msg.b_shape == Size([4]) and return_msg.p_shape == Size([10])
        assert_equal_within_error(return_msg.parameter, mock_dist_param)

    def test_call_correct_return_message_distribution_type_parameter_subtype(self):
        # Test that with a correct execution, the returned message is also correct
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': {'test_key', 'test_val'}}

        mock_dist_param = torch.rand([4, 10])

        @Summarization(repr_type='distribution', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return mock_dist_param, None

        return_msg = callback(test_msg)

        assert isinstance(return_msg, Message)
        assert return_msg.type is MessageType.Parameter
        assert return_msg.b_shape == Size([4]) and return_msg.p_shape == Size([10])
        assert_equal_within_error(return_msg.parameter, mock_dist_param)

    def test_call_correct_return_message_particle_type(self):
        # Test that with a correct execution, the returned message is also correct
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': {'test_key', 'test_val'}}

        mock_weight = torch.rand([4, 20])

        @Summarization(repr_type='particle')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return None, mock_weight

        return_msg = callback(test_msg)

        assert isinstance(return_msg, Message)
        assert return_msg.type is MessageType.Particles
        assert return_msg.b_shape == Size([4]) and return_msg.s_shape == Size([20]) and return_msg.e_shape == Size([5])
        assert_equal_within_error(return_msg.weight, mock_weight)
        assert all(equal_within_error(p1, p2) for p1, p2 in zip(return_msg.particles, test_msg.particles))
        assert all(equal_within_error(d1, d2) for d1, d2 in zip(return_msg.log_densities, test_msg.log_densities))

    def test_call_correct_return_message_dual_type(self):
        # Test that with a correct execution, the returned message is also correct
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20]), Size([5])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': {'test_key', 'test_val'}}

        mock_dist_param = torch.rand([4, 10])
        mock_weight = torch.rand([4, 20])

        # Use parameter subtype here for convenience, since distribution subtype has already been tested above
        @Summarization(repr_type='dual', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return mock_dist_param, mock_weight

        return_msg = callback(test_msg)

        assert isinstance(return_msg, Message)
        assert return_msg.type is MessageType.Dual
        assert return_msg.b_shape == Size([4]) and return_msg.p_shape == Size([10]) and \
               return_msg.s_shape == Size([20]) and return_msg.e_shape == Size([5])
        assert_equal_within_error(return_msg.parameter, mock_dist_param)
        assert_equal_within_error(return_msg.weight, mock_weight)
        assert all(equal_within_error(p1, p2) for p1, p2 in zip(return_msg.particles, test_msg.particles))
        assert all(equal_within_error(d1, d2) for d1, d2 in zip(return_msg.log_densities, test_msg.log_densities))

    def test_call_correct_return_message_multiple_rvs_1(self):
        # Test correctness of returning message when the original message has multiple random variables
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20, 30, 40]), Size([5, 6, 7])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': {'test_key', 'test_val'}}

        mock_dist_param = torch.rand([4, 10])
        mock_weight = torch.rand([4, 24000])

        @Summarization(repr_type='dual', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return mock_dist_param, mock_weight

        return_msg = callback(test_msg)

        assert isinstance(return_msg, Message)
        assert return_msg.s_shape == s_shape and return_msg.e_shape == e_shape
        assert return_msg.weight.shape == b_shape[1:] + s_shape

    def test_call_correct_return_message_multiple_rvs_2(self):
        # Test correctness of returning message when the original message has multiple random variables
        # Test returned weight tensor's content
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20, 30, 40]), Size([5, 6, 7])
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': {'test_key', 'test_val'}}

        expected_param = torch.randn(b_shape[1:] + p_shape)
        expected_weight = torch.rand(b_shape[1:] + s_shape)
        test_input_param = expected_param.unsqueeze(0).expand(b_shape + p_shape)
        test_input_weight = expected_weight.unsqueeze(0).expand(b_shape + s_shape)
        # Replace message parameter and weight
        test_msg.parameter = test_input_param
        test_msg.weight = test_input_weight
        # Clone the message to make sure everything fits
        test_msg = test_msg.clone()

        @Summarization(repr_type='dual', dist_repr_type='parameter')
        def callback(a: CFT, b: DT, c: PT, d: WT, e: LDT) -> RVT:
            return b[0], d[0]       # Only return the first slice

        return_msg = callback(test_msg)

        assert isinstance(return_msg, Message)
        assert return_msg.s_shape == s_shape and return_msg.e_shape == e_shape
        # Check content
        assert_equal_within_error(return_msg.parameter, expected_param)
        assert_proportional_within_error(return_msg.weight, expected_weight, dims=[-1, -2, -3])



