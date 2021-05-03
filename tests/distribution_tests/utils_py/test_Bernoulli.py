"""
    Integration tests for Bernoulli distribution implementations in utils.DistributionServer and utils.KnowledgeServer
"""
import pytest
import torch
from torch import Size
import torch.distributions as D
import torch.distributions.constraints as C

from pysigma.utils import DistributionServer as DS
from pysigma.utils import KnowledgeServer as KS

from ...utils import equal_within_error, assert_equal_within_error, assert_proportional_within_error, \
    assert_constant_difference_within_error


class TestDistributionServer():

    def test_param2dist_regular_param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([])
        dist_class = D.Bernoulli
        param = torch.rand(b_shape + p_shape)
        expected_param = param.squeeze(dim=-1)
        dist_info = {'param_type': 'regular'}

        dist = DS.param2dist(dist_class, param, b_shape, e_shape, dist_info)

        assert isinstance(dist, dist_class)
        assert_equal_within_error(dist.probs, expected_param)

    def test_dist2param_regular_param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([])
        dist_param = torch.rand(b_shape)
        dist = D.Bernoulli(probs=dist_param)
        expected_param = dist_param.unsqueeze(dim=-1)
        dist_info = {'param_type': 'regular'}

        param = DS.dist2param(dist, dist_info)

        assert_equal_within_error(param, expected_param)

    def test_param2dist2param_regular_param(self):
        # Test that param2dist and dist2param are invertible
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([])
        dist_class = D.Bernoulli
        param = torch.rand(b_shape + p_shape)
        dist_info = {'param_type': 'regular'}

        dist = DS.param2dist(dist_class, param, b_shape, e_shape, dist_info)
        return_param = DS.dist2param(dist, dist_info)

        assert_equal_within_error(param, return_param)

    def test_dist2param2dist_regular_param(self):
        # Test that param2dist and dist2param are invertible
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([])
        dist_class = D.Bernoulli
        dist_param = torch.rand(b_shape)
        dist = D.Bernoulli(probs=dist_param)
        expected_param = dist_param.unsqueeze(dim=-1)
        dist_info = {'param_type': 'regular'}

        param = DS.dist2param(dist, dist_info)
        return_dist = DS.param2dist(dist_class, param, b_shape, e_shape, dist_info)

        assert isinstance(return_dist, D.Bernoulli)
        assert_equal_within_error(dist.probs, return_dist.probs)

    def test_param2dist_natural_param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([])
        dist_class = D.Bernoulli
        natural_param = torch.randn(b_shape + p_shape)

        squeezed_natural_param = natural_param.squeeze(dim=-1)
        expected_regular_param = squeezed_natural_param.exp() / (1 + squeezed_natural_param.exp())

        dist = DS.param2dist(dist_class, natural_param, b_shape, e_shape)

        assert isinstance(dist, D.Bernoulli)
        assert_equal_within_error(dist.probs, expected_regular_param)

    def test_dist2param_natural_param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([])
        regular_param = torch.rand(b_shape)
        dist = D.Bernoulli(probs=regular_param)

        unsqueezed_regular_param = regular_param.unsqueeze(dim=-1)
        expected_natural_param = torch.log(unsqueezed_regular_param / (1 - unsqueezed_regular_param))

        param = DS.dist2param(dist)

        # Relaxed precision due to use of numerical stabilizer
        assert_equal_within_error(param, expected_natural_param, precision=1e-5)

    def test_param2dist2param_natural_param(self):
        # Test that param2dist and dist2param are invertible with natural parameters
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([])
        dist_class = D.Bernoulli
        natural_param = torch.randn(b_shape + p_shape)

        return_dist = DS.param2dist(dist_class, natural_param)
        return_param = DS.dist2param(return_dist)

        assert_equal_within_error(return_param, natural_param, 1e-5)

    def test_dist2param2dist_natural_param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([])
        dist_class = D.Bernoulli
        regular_param = torch.rand(b_shape)
        dist = D.Bernoulli(probs=regular_param)

        return_param = DS.dist2param(dist)
        return_dist = DS.param2dist(dist_class, return_param)

        assert isinstance(return_dist, D.Bernoulli)
        assert_equal_within_error(return_dist.probs, dist.probs, 1e-5)


class TestKnowledgeServer:

    def test_bernoulli_draw_particles(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([2]), Size([1])
        dist_class = D.Bernoulli
        rv_sizes = [1]
        rv_constraints = [C.boolean]
        dist_info = {'param_type': 'regular'}
        regular_param = torch.rand(b_shape + p_shape)

        ks = KS(dist_class, rv_sizes, rv_constraints, dist_info=dist_info)
        ptcl, dens = ks.draw_particles(regular_param, b_shape)

        assert len(ptcl) == len(dens) == 1
        assert ptcl[0].shape == s_shape + e_shape
        assert dens[0].shape == s_shape

        expected_ptcl = torch.tensor([[0.], [1.]])
        assert_equal_within_error(ptcl[0], expected_ptcl)

