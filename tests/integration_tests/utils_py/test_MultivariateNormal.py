"""
    Integration tests for Multivariate Normal distribution implementations in utils.DistributionServer and
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
        b_shape, p_shape, e_shape = Size([4, 5, 6]), Size([2]), Size([7])
        dist_class = D.MultivariateNormal
        loc = torch.randn(b_shape + e_shape)
        cov = torch.eye(e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        cov = cov.expand_as(torch.ones(b_shape + e_shape + e_shape))
        flattened_cov = cov.view(b_shape + Size([e_shape[0] ** 2]))

        param = torch.cat([loc, flattened_cov], dim=-1)
        dist = DS.param2dist(dist_class, param, b_shape, e_shape)

        assert isinstance(dist, dist_class)
        assert equal_within_error(dist.loc, loc) and equal_within_error(dist.covariance_matrix, cov)

    def test_dist2param(self):
        b_shape, p_shape, e_shape = Size([4, 5, 6]), Size([2]), Size([7])
        loc = torch.randn(b_shape + e_shape)
        cov = torch.eye(e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        cov = cov.expand_as(torch.ones(b_shape + e_shape + e_shape))
        flattened_cov = cov.view(b_shape + Size([e_shape[0] ** 2]))

        expected_param = torch.cat([loc, flattened_cov], dim=-1)
        dist = D.MultivariateNormal(loc, cov)

        param = DS.dist2param(dist)

        assert equal_within_error(param, expected_param)


class TestKnowledgeServer():
    pass
