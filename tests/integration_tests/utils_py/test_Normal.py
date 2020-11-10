"""
    Integration tests for univariate normal distribution implementations in utils.DistributionServer and
    utils.KnowledgeServer
"""
import pytest
import torch
from torch import Size
import torch.distributions as D
import torch.distributions.constraints as C

from pysigma.utils import DistributionServer as DS
from pysigma.utils import KnowledgeServer as KS

from ...utils import equal_within_error


class TestDistributionServer():

    def test_param2dist(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        param = torch.stack([loc, scale], dim=-1)

        dist = DS.param2dist(dist_class, param, b_shape, e_shape)

        assert isinstance(dist, dist_class)
        assert equal_within_error(dist.loc, loc) and equal_within_error(dist.scale, scale)

    def test_dist2param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        dist = D.Normal(loc, scale)

        expected_param = torch.stack([loc, scale], dim=-1)

        return_value = DS.dist2param(dist)

        assert equal_within_error(return_value, expected_param)

    def test_param_dist_conversion_invertible_1(self):
        # Test that the conversion between distribution parameters and distribution instance is invertible
        # Scenario 1: param -> dist -> param
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)

        param = torch.stack([loc, scale], dim=-1)

        val_1 = DS.param2dist(dist_class, param, b_shape, e_shape)
        val_2=  DS.dist2param(val_1)

        assert equal_within_error(param, val_2)

    def test_param_dist_conversion_invertible_2(self):
        # Test that the conversion between distribution parameters and distribution instance is invertible
        # Scenario 1: dist -> param -> dist
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        dist = D.Normal(loc, scale)

        val_1 = DS.dist2param(dist)
        val_2 = DS.param2dist(dist_class, val_1, b_shape, e_shape)

        assert isinstance(val_2, D.Normal)
        assert equal_within_error(val_2.loc, loc) and equal_within_error(val_2.scale, scale)


class TestKnowledgeServer():
    pass
