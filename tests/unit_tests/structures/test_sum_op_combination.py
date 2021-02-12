"""
    Unit tests for the builtin combination summarization operation.
"""
import pytest
import torch
from torch.distributions import Distribution as D
from torch import Size

from pysigma.defs import Message, MessageType
from pysigma.pattern_structures.builtin_sum_op import sum_op_combination
from tests.utils import assert_equal_within_error, assert_proportional_within_error


class TestSumOpCombination:

    def test_ad_hoc_parameter_message(self):
        # Test with an ad-hoc parameter type message.
        b_shape, p_shape, s_shape, e_shape = Size([2, 3]), Size([1]), Size([]), Size([])
        test_param = torch.tensor(
            [[[0.3], [0.4], [0.5]],
             [[0.6], [0.7], [0.8]]]
        )
        attr = {'dist_class': D, 'dist_info': None}
        test_msg = Message(MessageType.Parameter,
                           b_shape, p_shape, s_shape, e_shape,
                           parameter=test_param,
                           **attr)

        return_msg = sum_op_combination(test_msg)

        expected_param = torch.tensor(
            [[0.45], [0.55], [0.65]]
        )
        assert isinstance(return_msg, Message) and return_msg.type is MessageType.Parameter
        assert_equal_within_error(return_msg.parameter, expected_param)

    def test_ad_hoc_particles_message(self):
        # Test with ad-hoc particles type message
        b_shape, p_shape, s_shape, e_shape = Size([2, 3]), Size([]), Size([2]), Size([2])
        test_weight = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
             [[1.0, 1.1], [1.2, 1.3], [1.4, 1.5]]]
        )
        test_particles = [torch.randn(2, 2)]
        test_log_densities = [torch.rand(2)]
        test_msg = Message(MessageType.Particles,
                           b_shape, p_shape, s_shape, e_shape,
                           particles=test_particles, weight=test_weight, log_densities=test_log_densities)

        return_msg = sum_op_combination(test_msg)

        expected_weight = torch.tensor(
            [[[0.1, 0.22], [0.36, 0.52], [0.7, 0.9]]]
        )
        assert isinstance(return_msg, Message) and return_msg.type is MessageType.Particles
        assert_proportional_within_error(return_msg.weight, expected_weight, dims=[-1])


