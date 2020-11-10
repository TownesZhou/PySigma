"""
    Integration tests for Categorical distribution implementations in utils.DistributionServer and utils.KnowledgeServer
"""
import pytest
import torch
from torch import Size
import torch.distributions as D
import torch.distributions.constraints as C

from pysigma.utils import DistributionServer as DS
from pysigma.utils import KnowledgeServer as KS

from ...utils import equal_within_error


class TestDistributionServerCategorical():

    def test_param2dist(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([10]), Size([])
        dist_class = D.Categorical
        param = torch.rand(b_shape + p_shape)
        expected_param = param / param.sum(dim=-1, keepdim=True)

        dist = DS.param2dist(dist_class, param, b_shape, e_shape)

        assert isinstance(dist, dist_class)
        assert equal_within_error(dist.probs, expected_param)

    def test_dist2param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([10]), Size([])
        param = torch.rand(b_shape + p_shape)
        dist = D.Categorical(probs=param)
        dist_info = None
        expected_param = param / param.sum(dim=-1, keepdim=True)

        returned_value = DS.dist2param(dist, dist_info)

        assert equal_within_error(returned_value, expected_param)


class TestKnowledgeServerCategorical():

    def test_categorical_var_span_univariate(self):
        # Test var span under a single RV
        s_shape, e_shape = Size([15]), Size([1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 5),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        span = ks._categorical_var_span()
        expected_span = (6,)
        assert span == expected_span

    def test_categorical_var_span_multivariate(self):
        # Test var span under 3 RVs
        s_shape, e_shape = Size([15, 20, 25]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 5), C.integer_interval(0, 8), C.integer_interval(0, 3),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        span = ks._categorical_var_span()
        expected_span = (6, 9, 4)
        assert span == expected_span

    def test_categorical_enforced_sample_shape(self):
        s_shape, e_shape = Size([10, 15, 20]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 1), C.integer_interval(0, 4), C.integer_interval(0, 9),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        s_shape = ks._categorical_enforced_sample_shape()

        assert s_shape == Size([2, 5, 10])

    def test_categorical_2torch_event_univariate(self):
        # Test with a single RV
        s_shape, e_shape = Size([15]), Size([1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 5),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        particles = torch.randint(0, 6, s_shape + e_shape)
        return_val = ks._categorical_2torch_event(particles)

        assert equal_within_error(particles, return_val)

    def test_categorical_2torch_event_multivariate_ad_hoc(self):
        # Test with 3 RVs
        s_shape, e_shape = Size([10, 15, 20]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 1), C.integer_interval(0, 4), C.integer_interval(0, 9),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        particles = torch.tensor([
            [0., 0., 0.],
            [0., 0., 6.],
            [0., 2., 4.],
            [0., 3., 8.],
            [1., 0., 5.],
            [1., 3., 7.]
        ])
        expected_return = torch.tensor([
            [0.],
            [6.],
            [24.],
            [38.],
            [55.],
            [87.]
        ])

        return_val = ks._categorical_2torch_event(particles)
        assert equal_within_error(return_val, expected_return)

    def test_categorical_2cognitive_event_univariate(self):
        # Test with a single RV
        s_shape, e_shape = Size([15]), Size([1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 5),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        particles = torch.randint(0, 6, s_shape + e_shape)
        return_val = ks._categorical_2cognitive_event(particles)

        assert equal_within_error(particles, return_val)

    def test_categorical_2cognitive_event_multivariate(self):
        # Test with 3 RVs
        s_shape, e_shape = Size([10, 15, 20]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 1), C.integer_interval(0, 4), C.integer_interval(0, 9),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        particles = torch.tensor([
            [0.],
            [6.],
            [24.],
            [38.],
            [55.],
            [87.]
        ])
        expected_return = torch.tensor([
            [0., 0., 0.],
            [0., 0., 6.],
            [0., 2., 4.],
            [0., 3., 8.],
            [1., 0., 5.],
            [1., 3., 7.]
        ])

        return_val = ks._categorical_2cognitive_event(particles)
        assert equal_within_error(return_val, expected_return)

    def test_categorical_translation_invertible(self):
        # Test that _categorical_2torch_event() and _categorical_2cognitive_event() are mutually invertible
        s_shape, e_shape = Size([10, 15, 20]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 1), C.integer_interval(0, 4), C.integer_interval(0, 9),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        num_ptcl = 100
        # Composition 1: cognitive -> torch -> cognitive
        split_ptcl = [
            torch.randint(0, 2, [num_ptcl, 1]),
            torch.randint(0, 5, [num_ptcl, 1]),
            torch.randint(0, 10, [num_ptcl, 1]),
        ]
        ptcl = torch.cat(split_ptcl, dim=-1)
        return_val = ks._categorical_2cognitive_event(ks._categorical_2torch_event(ptcl))
        assert equal_within_error(return_val, ptcl)

        # Composition 2: torch -> cognitive -> torch
        ptcl = torch.randint(0, 100, [num_ptcl, 1])
        return_val = ks._categorical_2torch_event(ks._categorical_2cognitive_event(ptcl))
        assert equal_within_error(return_val, ptcl)

    def test_categorical_draw_univariate(self):
        # Test with a single RV
        s_shape, e_shape = Size([15]), Size([1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 5),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        ptcl, dens = ks._categorical_draw()

        assert len(ptcl) == len(dens) == 1
        assert equal_within_error(ptcl[0], torch.tensor([[0.], [1.], [2.], [3.], [4.], [5.]]))
        assert equal_within_error(torch.exp(dens[0]), torch.ones([6]) / 6)

    def test_categorical_draw_multivariate(self):
        # Test with a single RV
        s_shape, e_shape = Size([10, 15, 20]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 1), C.integer_interval(0, 4), C.integer_interval(0, 9),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        ptcl, dens = ks._categorical_draw()

        assert len(ptcl) == len(dens) == 3
        assert equal_within_error(ptcl[0], torch.tensor([[0.], [1.]]))
        assert equal_within_error(ptcl[1], torch.tensor([[0.], [1.], [2.], [3.], [4.]]))
        assert equal_within_error(ptcl[2], torch.tensor([[0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]]))
        assert equal_within_error(torch.exp(dens[0]), torch.ones([2]) / 2)
        assert equal_within_error(torch.exp(dens[1]), torch.ones([5]) / 5)
        assert equal_within_error(torch.exp(dens[2]), torch.ones([10]) / 10)

    def test_integration_categorical_init_enforced_sample_shape(self):
        # Integration test: sample shape is enforced during init
        s_shape, e_shape = Size([10, 15, 20]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 1), C.integer_interval(0, 4), C.integer_interval(0, 9),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        assert ks.s_shape == Size([2, 5, 10])

    def test_integration_categorical_event2torch_event(self):
        # Integration test: event2torch_event() with Categorical distribution
        s_shape, e_shape = Size([10, 15, 20]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 1), C.integer_interval(0, 4), C.integer_interval(0, 9),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        particles = torch.tensor([
            [0., 0., 0.],
            [0., 0., 6.],
            [0., 2., 4.],
            [0., 3., 8.],
            [1., 0., 5.],
            [1., 3., 7.]
        ])
        expected_return = torch.tensor([
            [0.],
            [6.],
            [24.],
            [38.],
            [55.],
            [87.]
        ])

        return_val = ks.event2torch_event(particles)
        assert equal_within_error(return_val, expected_return)

    def test_integration_categorical_event2cognitive_event(self):
        # Integration test: event2cognitive_event() with Categorical distribution
        s_shape, e_shape = Size([10, 15, 20]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 1), C.integer_interval(0, 4), C.integer_interval(0, 9),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        particles = torch.tensor([
            [0.],
            [6.],
            [24.],
            [38.],
            [55.],
            [87.]
        ])
        expected_return = torch.tensor([
            [0., 0., 0.],
            [0., 0., 6.],
            [0., 2., 4.],
            [0., 3., 8.],
            [1., 0., 5.],
            [1., 3., 7.]
        ])

        return_val = ks.event2cognitive_event(particles)
        assert equal_within_error(return_val, expected_return)

    def test_integration_draw_particles(self):
        # Integration test: draw_particles() with Categorical distribution
        b_shape, s_shape, e_shape = Size([3, 4, 5]), Size([10, 15, 20]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 1), C.integer_interval(0, 4), C.integer_interval(0, 9),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        param = torch.randn(b_shape + Size([100]))
        ptcl, dens = ks.draw_particles(param, b_shape, update_cache=False)

        assert len(ptcl) == len(dens) == 3
        assert equal_within_error(ptcl[0], torch.tensor([[0.], [1.]]))
        assert equal_within_error(ptcl[1], torch.tensor([[0.], [1.], [2.], [3.], [4.]]))
        assert equal_within_error(ptcl[2], torch.tensor([[0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]]))
        assert equal_within_error(torch.exp(dens[0]), torch.ones([2]) / 2)
        assert equal_within_error(torch.exp(dens[1]), torch.ones([5]) / 5)
        assert equal_within_error(torch.exp(dens[2]), torch.ones([10]) / 10)

    def test_integration_surrogate_log_prob_1(self):
        # Integration test: surrogate_log_prob() with Categorical distribution
        # Scenario 1: using cached particles
        b_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2, 5, 10]), Size([1, 1, 1])

        dist_class = D.Categorical
        rv_cstr = (C.integer_interval(0, 1), C.integer_interval(0, 4), C.integer_interval(0, 9),)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))
        ks.particles, _ = ks._categorical_draw()

        param = torch.randn(b_shape + Size([100]))

        log_prob = ks.surrogate_log_prob(param, alt_particles=None, index_map=None)

        assert log_prob.shape == b_shape + s_shape
