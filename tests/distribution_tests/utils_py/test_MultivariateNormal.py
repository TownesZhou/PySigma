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

from ...utils import equal_within_error, generate_positive_definite, assert_equal_within_error


class TestDistributionServer():

    def test_param2dist_regular_param(self):
        b_shape, e_shape = Size([4, 5, 6]), Size([7])
        dist_class = D.MultivariateNormal
        loc = torch.randn(b_shape + e_shape)
        cov = torch.eye(e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        cov = cov.expand_as(torch.ones(b_shape + e_shape + e_shape))
        flattened_cov = cov.view(b_shape + Size([e_shape[0] ** 2]))

        param = torch.cat([loc, flattened_cov], dim=-1)
        dist_info = {'param_type': 'regular'}
        dist = DS.param2dist(dist_class, param, b_shape, e_shape, dist_info)

        assert isinstance(dist, dist_class)
        assert equal_within_error(dist.loc, loc) and equal_within_error(dist.covariance_matrix, cov)

    def test_dist2param_regular_param(self):
        b_shape, e_shape = Size([4, 5, 6]), Size([7])
        loc = torch.randn(b_shape + e_shape)
        cov = torch.eye(e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        cov = cov.expand_as(torch.ones(b_shape + e_shape + e_shape))
        flattened_cov = cov.view(b_shape + Size([e_shape[0] ** 2]))

        expected_param = torch.cat([loc, flattened_cov], dim=-1)
        dist = D.MultivariateNormal(loc, cov)

        dist_info = {'param_type': 'regular'}
        param = DS.dist2param(dist, dist_info)

        assert equal_within_error(param, expected_param)

    def test_param2dist_natural_param(self):
        b_shape, e_shape = Size([4, 5, 6]), Size([7])
        dist_class = D.MultivariateNormal
        loc = torch.randn(b_shape + e_shape)
        cov = torch.eye(e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        cov = cov.expand_as(torch.ones(b_shape + e_shape + e_shape))

        p1 = cov.inverse().matmul(loc.unsqueeze(-1)).squeeze(-1)
        p2 = (-0.5 * cov.inverse()).contiguous().view(b_shape + Size([e_shape[0] ** 2]))
        param = torch.cat([p1, p2], dim=-1)

        dist = DS.param2dist(dist_class, param, b_shape, e_shape)

        assert isinstance(dist, dist_class)
        assert equal_within_error(dist.loc, loc) and equal_within_error(dist.covariance_matrix, cov)

    def test_dist2param_natural_param(self):
        b_shape, e_shape = Size([4, 5, 6]), Size([7])
        loc = torch.randn(b_shape + e_shape)
        cov = torch.eye(e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        cov = cov.expand_as(torch.ones(b_shape + e_shape + e_shape))

        p1 = cov.inverse().matmul(loc.unsqueeze(-1)).squeeze(-1)
        p2 = (-0.5 * cov.inverse()).contiguous().view(b_shape + Size([e_shape[0] ** 2]))
        expected_param = torch.cat([p1, p2], dim=-1)

        dist = D.MultivariateNormal(loc, cov)

        param = DS.dist2param(dist)

        assert equal_within_error(param, expected_param)

    def test_param_dist_conversion_invertible_natural_param_1(self):
        # Test 1: param -> dist -> param
        b_shape, e_shape = Size([4, 5, 6]), Size([7])
        dist_class = D.MultivariateNormal
        loc = torch.randn(b_shape + e_shape)
        cov = generate_positive_definite(b_shape, e_shape[0])

        p1 = cov.inverse().matmul(loc.unsqueeze(-1)).squeeze(-1)
        p2 = (-0.5 * cov.inverse()).contiguous().view(b_shape + Size([e_shape[0] ** 2]))
        param = torch.cat([p1, p2], dim=-1)

        val_1 = DS.param2dist(dist_class, param)
        val_2 = DS.dist2param(val_1)

        assert_equal_within_error(val_2, param)

    def test_param_dist_conversion_invertible_natural_param_2(self):
        # Test 2: dist -> param -> dist
        b_shape, e_shape = Size([4, 5, 6]), Size([7])
        dist_class = D.MultivariateNormal
        loc = torch.randn(b_shape + e_shape)
        cov = generate_positive_definite(b_shape, e_shape[0])

        dist = D.MultivariateNormal(loc, cov)

        val_1 = DS.dist2param(dist)
        val_2 = DS.param2dist(dist_class, val_1)

        # Note: due to numerical precision issues, here we relax precision a bit
        precision = 1e-4
        assert_equal_within_error(val_2.loc, dist.loc, precision=precision)
        assert_equal_within_error(val_2.covariance_matrix, dist.covariance_matrix, precision=precision)


class TestKnowledgeServer():

    def test_draw_particles(self):
        b_shape, s_shape, e_shape = Size([4, 5, 6]), Size([10, 15, 20]), Size([1, 2, 3])
        cat_e_shape = Size([sum(e_shape)])

        dist_class = D.MultivariateNormal
        rv_cstr = (C.real,) * len(e_shape)
        dist_info = {'param_type': 'regular'}
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape),
                dist_info=dist_info)

        loc = torch.randn(b_shape + cat_e_shape)
        cov = torch.eye(cat_e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        cov = cov.expand_as(torch.ones(b_shape + cat_e_shape + cat_e_shape))
        flattened_cov = cov.view(b_shape + Size([cat_e_shape[0] ** 2]))

        param = torch.cat([loc, flattened_cov], dim=-1)

        ptcl_list, dens_list = ks.draw_particles(param, b_shape, update_cache=False)

        # Check shape
        assert all(p.shape == Size([s, e]) for p, s, e in zip(ptcl_list, s_shape, e_shape))
        assert all(d.shape == Size([s]) for d, s in zip(dens_list, s_shape))

    def test_surrogate_log_prob(self):
        b_shape, s_shape, e_shape = Size([4, 5, 6]), Size([10, 15, 20]), Size([1, 2, 3])
        cat_e_shape = Size([sum(e_shape)])

        dist_class = D.MultivariateNormal
        rv_cstr = (C.real,) * len(e_shape)
        dist_info = {'param_type': 'regular'}
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape),
                dist_info=dist_info)

        alt_particles = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)]

        loc = torch.randn(b_shape + cat_e_shape)
        cov = torch.eye(cat_e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        cov = cov.expand_as(torch.ones(b_shape + cat_e_shape + cat_e_shape))
        flattened_cov = cov.view(b_shape + Size([cat_e_shape[0] ** 2]))
        param = torch.cat([loc, flattened_cov], dim=-1)

        log_prob = ks.surrogate_log_prob(param, alt_particles)

        # Check shape
        assert log_prob.shape == b_shape + s_shape
