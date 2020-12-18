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

from ...utils import equal_within_error, assert_equal_within_error


class TestDistributionServer():

    def test_param2dist_regular_param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        param = torch.stack([loc, scale], dim=-1)

        dist_info = {'param_type': 'regular'}
        dist = DS.param2dist(dist_class, param, b_shape, e_shape, dist_info)

        assert isinstance(dist, dist_class)
        assert equal_within_error(dist.loc, loc) and equal_within_error(dist.scale, scale)

    def test_dist2param_regular_param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        dist = D.Normal(loc, scale)

        expected_param = torch.stack([loc, scale], dim=-1)

        dist_info = {'param_type': 'regular'}
        return_value = DS.dist2param(dist, dist_info)

        assert equal_within_error(return_value, expected_param)

    def test_param2dist_natural_param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        p1 = loc / scale ** 2
        p2 = -1 / (2 * scale ** 2)
        param = torch.stack([p1, p2], dim=-1)

        dist = DS.param2dist(dist_class, param, b_shape, e_shape)

        assert isinstance(dist, dist_class)
        assert equal_within_error(dist.loc, loc) and equal_within_error(dist.scale, scale)

    def test_dist2param_natural_param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        p1 = loc / scale ** 2
        p2 = -1 / (2 * scale ** 2)
        dist = D.Normal(loc, scale)

        expected_param = torch.stack([p1, p2], dim=-1)

        return_value = DS.dist2param(dist)

        assert equal_within_error(return_value, expected_param)

    def test_param_dist_conversion_invertible_regular_param_1(self):
        # Test that the conversion between distribution parameters and distribution instance is invertible
        # Scenario 1: param -> dist -> param
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)

        param = torch.stack([loc, scale], dim=-1)

        dist_info = {'param_type': 'regular'}
        val_1 = DS.param2dist(dist_class, param, b_shape, e_shape, dist_info)
        val_2=  DS.dist2param(val_1, dist_info)

        assert equal_within_error(param, val_2)

    def test_param_dist_conversion_invertible_regular_param_2(self):
        # Test that the conversion between distribution parameters and distribution instance is invertible
        # Scenario 1: dist -> param -> dist
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        dist = D.Normal(loc, scale)

        dist_info = {'param_type': 'regular'}
        val_1 = DS.dist2param(dist, dist_info)
        val_2 = DS.param2dist(dist_class, val_1, b_shape, e_shape, dist_info)

        assert isinstance(val_2, D.Normal)
        assert equal_within_error(val_2.loc, loc) and equal_within_error(val_2.scale, scale)

    def test_param_dist_conversion_invertible_natural_param_1(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([2]), Size([])
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        p1 = loc / scale ** 2
        p2 = -1 / (2 * scale ** 2)

        param = torch.stack([p1, p2], dim=-1)

        val_1 = DS.param2dist(dist_class, param, b_shape, e_shape)
        val_2 = DS.dist2param(val_1)

        assert equal_within_error(param, val_2)


class TestKnowledgeServer():

    def test_draw_particles(self):
        b_shape, s_shape, e_shape = Size([4, 5, 6]), Size([20]), Size([1])

        dist_class = D.Normal
        rv_cstr = (C.real,)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        p1 = loc / scale ** 2
        p2 = -1 / (2 * scale ** 2)
        param = torch.stack([p1, p2], dim=-1)
        dist = D.Normal(loc, scale)

        ptcl, dens = ks.draw_particles(param, b_shape, update_cache=False)
        ptcl, dens = ptcl[0], dens[0]

        # Check shape
        assert ptcl.shape == s_shape + e_shape
        assert dens.shape == s_shape

        # Check content
        # Insert batch dimensions into particles
        expand_ptcl = ptcl
        for i in range(len(b_shape)):
            expand_ptcl = expand_ptcl.unsqueeze(dim=1)
        expand_ptcl = expand_ptcl.expand_as(torch.ones(s_shape + b_shape + e_shape)).squeeze(dim=-1)

        log_prob = dist.log_prob(expand_ptcl)
        prob = torch.exp(log_prob)
        avg_prob = prob.mean(dim=[1, 2, 3])
        expected_log_dens = torch.log(avg_prob)

        assert equal_within_error(dens, expected_log_dens)

    def test_surrogate_log_prob(self):
        # Note: Due to numerical precision issues, there is significant numerical error (up to 1e-5) between the
        #   log prob obtained from the normal distribution generated with regular parameters verses the log prob
        #   obtained from the distribution generated with natural parameters.
        b_shape, s_shape, e_shape = Size([4, 5, 6]), Size([20]), Size([1])

        dist_class = D.Normal
        rv_cstr = (C.real,)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        loc, scale = torch.randn(b_shape), torch.rand(b_shape) * 10     # larger std to cover extreme outliers
        p1 = loc / scale ** 2
        p2 = -1 / (2 * scale ** 2)
        param = torch.stack([p1, p2], dim=-1)
        dist = D.Normal(loc, scale)
        # dist = DS.param2dist(dist_class, param)

        alt_ptcl = torch.randn(s_shape + e_shape)
        alt_ptcl = torch.clamp(alt_ptcl, -1, 1)     # Clamp the particles to avoid extreme outlier values

        log_prob = ks.surrogate_log_prob(param, alt_particles=[alt_ptcl])

        expand_ptcl = alt_ptcl
        # Insert batch dimensions
        for i in range(len(b_shape)):
            expand_ptcl = expand_ptcl.unsqueeze(dim=1)
        expand_ptcl = expand_ptcl.expand_as(torch.ones(s_shape + b_shape + e_shape))
        expected_log_prob = dist.log_prob(expand_ptcl.squeeze(dim=-1))
        # Switch sample and batch dimensions
        perm_order = [1, 2, 3, 0]
        expected_log_prob = expected_log_prob.permute(perm_order)

        # Here we relax the numerical precision to pass the test
        precision = 1e-4
        assert_equal_within_error(log_prob, expected_log_prob, precision=precision)
