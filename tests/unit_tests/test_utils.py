"""
    Unit tests for pysigma.utils module
"""
import pytest
from unittest.mock import MagicMock, PropertyMock, patch
import torch
from torch import Size
import torch.distributions as D
import torch.distributions.constraints as C

from pysigma.utils import intern_name, extern_name, compatible_shape
from pysigma.utils import DistributionServer as DS
from pysigma.utils import KnowledgeServer as KS

from .utils import equal_within_error


# region: Testing module-level functions in utils.py
def test_intern_name_invalid():
    name, struc_type = "Test", "something else"
    with pytest.raises(AssertionError):
        intern_name(name, struc_type)


def test_intern_name_correct():
    name, struc_type = "Test", "type"
    expected_name = "TYPE_[Test]"
    assert intern_name(name, struc_type) == expected_name

    name, struc_type = "Test", "predicate"
    expected_name = "PRED_[Test]"
    assert intern_name(name, struc_type) == expected_name

    name, struc_type = "Test", "conditional"
    expected_name = "COND_[Test]"
    assert intern_name(name, struc_type) == expected_name


def test_extern_name_invalid():
    name, struc_type = "Test", "something else"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

    name, struc_type = "Test", "type"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

    name, struc_type = "TYPE_Test", "type"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

    name, struc_type = "_TYPE_[Test]", "type"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

    name, struc_type = "type_[Test]", "type"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)

    name, struc_type = "TYPE_[Test", "type"
    with pytest.raises(AssertionError):
        extern_name(name, struc_type)


def test_extern_name_correct():
    name, struc_type = "TYPE_[Test_name]", "type"
    expected_name = "Test_name"
    assert extern_name(name, struc_type) == expected_name

    name, struc_type = "PRED_[Test_name]", "predicate"
    expected_name = "Test_name"
    assert extern_name(name, struc_type) == expected_name

    name, struc_type = "COND_[Test_name]", "conditional"
    expected_name = "Test_name"
    assert extern_name(name, struc_type) == expected_name


def test_intern_extern_invertible_1():
    name, struc_type = "Test_Name", "type"
    v1 = intern_name(name, struc_type)
    v2 = extern_name(v1, struc_type)
    assert name == v2

    name, struc_type = "Test_Name", "predicate"
    v1 = intern_name(name, struc_type)
    v2 = extern_name(v1, struc_type)
    assert name == v2

    name, struc_type = "Test_Name", "conditional"
    v1 = intern_name(name, struc_type)
    v2 = extern_name(v1, struc_type)
    assert name == v2


def test_intern_extern_invertible_2():
    name, struc_type = "TYPE_[Test_name]", "type"
    v1 = extern_name(name, struc_type)
    v2 = intern_name(v1, struc_type)
    assert name == v2

    name, struc_type = "PRED_[Test_name]", "predicate"
    v1 = extern_name(name, struc_type)
    v2 = intern_name(v1, struc_type)
    assert name == v2

    name, struc_type = "COND_[Test_name]", "conditional"
    v1 = extern_name(name, struc_type)
    v2 = intern_name(v1, struc_type)
    assert name == v2


def test_compatible_shape_invalid_arguments():
    shape1 = None
    shape2 = None
    with pytest.raises(AssertionError):
        compatible_shape(shape1, shape2)

    shape1 = (None, None, None, None)
    shape2 = (None, None, None, None)
    with pytest.raises(AssertionError):
        compatible_shape(shape1, shape2)

    shape1 = (Size([1]), Size([2]), Size([3]),)
    shape2 = (Size([1]), Size([2]), Size([3]), Size([4]))
    with pytest.raises(AssertionError):
        compatible_shape(shape1, shape2)


def test_compatible_shape_compatible():
    shape1 = (Size([1]), Size([2]), Size([3]), Size([4]))
    shape2 = (Size([1]), Size([2]), Size([3]), Size([4]))
    assert compatible_shape(shape1, shape2)

    shape1 = (Size([1, 2]), Size([2, 3]), Size([3, 4]), Size([4, 5]))
    shape2 = (Size([1, 2]), Size([2, 3]), Size([3, 4]), Size([4, 5]))
    assert compatible_shape(shape1, shape2)

    shape1 = (Size([]), Size([2]), Size([3]), Size([4]))
    shape2 = (Size([1, 2]), Size([2]), Size([3]), Size([4]))
    assert compatible_shape(shape1, shape2)


def test_compatible_shape_incompatible():
    shape1 = (Size([1]), Size([2]), Size([3]), Size([4]))
    shape2 = (Size([1]), Size([2]), Size([3]), Size([5]))
    assert not compatible_shape(shape1, shape2)

    shape1 = (Size([1]), Size([2]), Size([3]), Size([4, 5]))
    shape2 = (Size([1]), Size([2]), Size([3]), Size([5, 4]))
    assert not compatible_shape(shape1, shape2)

# endregion


class TestDistributionServer:

    # region: Test top-level methods using mocks
    def test_param2dist_dist_class_not_found(self):
        # Test that the specified dist class cannot be found in the class's registry
        dist_class = D.Distribution
        param = torch.randn([3, 5])

        # Patch the class attribute to an empty dictionary
        with patch("pysigma.utils.DistributionServer.dict_param2dist", new_callable=PropertyMock) as d:
            d.return_value = {}
            with pytest.raises(NotImplementedError):
                DS.param2dist(dist_class, param)

    def test_param2dist_dist_wrong_shape(self):
        # Test that the returned dist instance has a different shape than expected
        dist_class = D.Distribution
        random_param = torch.randn([3, 5])
        b_shape, e_shape = Size([3]), Size([5])
        dist = MagicMock(spec_set=D.Distribution)       # Mock a dist instance
        # dist.batch_shape = PropertyMock(return_value=Size([1]))
        # dist.event_shape = PropertyMock(return_value=Size([5]))
        dist.batch_shape = Size([3, 5])
        dist.event_shape = Size([])

        with patch("pysigma.utils.DistributionServer.dict_param2dist", new_callable=PropertyMock) as d:
            # Mock the sub-level class method that are returned by looking up the class registry
            mock_callable = MagicMock(side_effect=lambda param, dist_info : dist)
            mock_descriptor = MagicMock()
            mock_descriptor.__func__ = mock_callable
            # Set content of mocked class registry dictionary
            d.return_value = {dist_class: mock_descriptor}

            with pytest.raises(ValueError):
                DS.param2dist(dist_class, random_param, b_shape=b_shape, e_shape=e_shape)

    def test_param2dist_correct(self):
        # Test correct return value
        dist_class = D.Distribution
        random_param = torch.randn([3, 5])
        b_shape, e_shape = Size([3]), Size([5])
        dist = MagicMock(spec_set=D.Distribution)  # Mock a dist instance
        dist.batch_shape = Size([3])
        dist.event_shape = Size([5])
        dist_info = None

        with patch("pysigma.utils.DistributionServer.dict_param2dist", new_callable=PropertyMock) as d:
            # Mock the sub-level class method that are returned by looking up the class registry
            mock_callable = MagicMock(side_effect=lambda param, dist_info: dist)
            mock_descriptor = MagicMock()
            mock_descriptor.__func__ = mock_callable
            # Set content of mocked class registry dictionary
            d.return_value = {dist_class: mock_descriptor}

            assert DS.param2dist(dist_class, random_param,
                                 b_shape=b_shape, e_shape=e_shape, dist_info=dist_info) is dist
            # mock_callable.assert_called_once_with(random_param, dist_info)
            mock_callable.assert_called_once()
            call_args = mock_callable.call_args[0]
            assert call_args[0] is random_param and call_args[1] is dist_info

    def test_dist2param_dist_class_not_found(self):
        # Test that specified dist class cannot be found in the class's registry
        mock_dist_class = D.Distribution
        mock_dist = MagicMock(spec_set=D.Distribution)

        # Mock the builtin type()
        with patch("pysigma.utils.type") as t:
            t.side_effect = lambda dist: mock_dist_class if dist is mock_dist else type(dist)

            with patch("pysigma.utils.DistributionServer.dict_dist2param", new_callable=PropertyMock) as d:
                d.return_value = {}

                with pytest.raises(NotImplementedError):
                    DS.dist2param(mock_dist)

    def test_dist2param_correct(self):
        # Test correct return value
        mock_dist_class = D.Distribution
        mock_dist = MagicMock(spec_set=D.Distribution)
        dist_info = None
        expected_param = torch.randn([3, 5])

        # Mock the builtin type()
        with patch("pysigma.utils.type") as t:
            t.side_effect = lambda dist: mock_dist_class if dist is mock_dist else type(dist)

            with patch("pysigma.utils.DistributionServer.dict_dist2param", new_callable=PropertyMock) as d:
                mock_callable = MagicMock(side_effect=
                                          lambda dist, dist_info: expected_param if dist is mock_dist else None)
                mock_descriptor = MagicMock()
                mock_descriptor.__func__ = mock_callable
                d.return_value = {mock_dist_class: mock_descriptor}

                assert torch.equal(DS.dist2param(mock_dist, dist_info=dist_info), expected_param)
                mock_callable.assert_called_once_with(mock_dist, dist_info)

    def test_get_moments_dist_class_not_found(self):
        # Test that specified dist class cannot be found in the class's registry
        mock_dist_class = D.Distribution
        mock_dist = MagicMock(spec_set=D.Distribution)

        # Mock the builtin type()
        with patch("pysigma.utils.type") as t:
            t.side_effect = lambda dist: mock_dist_class if dist is mock_dist else type(dist)

            with patch("pysigma.utils.DistributionServer.dict_get_moments", new_callable=PropertyMock) as d:
                d.return_value = {}

                with pytest.raises(NotImplementedError):
                    DS.get_moments(mock_dist)

    def test_get_moments_correct(self):
        # Test correct returned value
        mock_dist_class = D.Distribution
        mock_dist = MagicMock(spec_set=D.Distribution)
        n_moments = 2
        expected_moments = torch.randn([3, 5])

        # Mock the builtin type()
        with patch("pysigma.utils.type") as t:
            t.side_effect = lambda dist: mock_dist_class if dist is mock_dist else type(dist)

            with patch("pysigma.utils.DistributionServer.dict_get_moments", new_callable=PropertyMock) as d:
                mock_callable = MagicMock(side_effect=
                                          lambda dist, moments: expected_moments
                                          if dist is mock_dist and moments == n_moments else None)
                mock_descriptor = MagicMock()
                mock_descriptor.__func__ = mock_callable
                d.return_value = {mock_dist_class: mock_descriptor}

                assert torch.equal(DS.get_moments(mock_dist, n_moments), expected_moments)
                mock_callable.assert_called_once_with(mock_dist, n_moments)

    def test_draw_particles_special_method(self):
        # Test that special drawing method is called if the dist class is registered in the class's registry
        # Mock dist instance
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist_class = D.Distribution
        num_ptcl = 10
        dist_info = None
        expected_particles = torch.randn([num_ptcl, 1])

        # Patch type so that type(mock_dist) returns mock_dist_class
        with patch("pysigma.utils.type") as t:
            t.side_effect = lambda dist: mock_dist_class if dist is mock_dist else type(dist)

            # Patch class registry
            with patch("pysigma.utils.DistributionServer.dict_draw_particles", new_callable=PropertyMock) as d:
                # Mock special draw particles method
                mock_callable = MagicMock(side_effect=
                                          lambda dist, num_particles, dist_info: expected_particles
                                          if dist is mock_dist and num_particles is num_ptcl else None)
                mock_descriptor = MagicMock()
                mock_descriptor.__func__ = mock_callable
                d.return_value = {mock_dist_class: mock_descriptor}

                assert torch.equal(DS.draw_particles(mock_dist, num_ptcl, dist_info=dist_info), expected_particles)

    def test_draw_particles_default_non_singleton_mock(self):
        # Test the default drawing method with particles with non-singleton event dimension using mocks
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist_class = D.Distribution
        num_ptcl = 100
        dist_info = None

        mock_dist.batch_shape, mock_dist.event_shape = Size([3, 4]), Size([5])
        mock_dist.sample.side_effect = lambda size: torch.randn([size[0], 3, 4, 5])

        # Patch type so that type(mock_dist) returns mock_dist_class
        with patch("pysigma.utils.type") as t:
            t.side_effect = lambda dist: mock_dist_class if dist is mock_dist else type(dist)

            # Patch class registry
            with patch("pysigma.utils.DistributionServer.dict_draw_particles", new_callable=PropertyMock) as d:
                # Mock special draw particles method
                d.return_value = {}

                returned_ptcl = DS.draw_particles(mock_dist, num_ptcl, dist_info)
                assert returned_ptcl.shape == Size([num_ptcl, 5])

    def test_draw_particles_default_non_singleton_real(self):
        # Test the default drawing method with particles with non-singleton event dimension using real distribution
        # Testing with MultivariateNormal as example
        b_shape, e_shape = Size([3, 4]), Size([5])

        loc = torch.randn(b_shape + e_shape)
        cov = torch.eye(e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        repeat_times = list(b_shape) + [1, 1]
        cov = cov.repeat(repeat_times)

        dist = D.MultivariateNormal(loc, cov)
        num_ptcl = 100
        dist_info = None

        # Patch class registry
        with patch("pysigma.utils.DistributionServer.dict_draw_particles", new_callable=PropertyMock) as d:
            # Mock special draw particles method
            d.return_value = {}

            returned_ptcl = DS.draw_particles(dist, num_ptcl, dist_info)
            assert returned_ptcl.shape == Size([num_ptcl]) + e_shape

    def test_draw_particles_default_singleton_mock(self):
        # Test the default drawing method with particles with singleton event dimension using mocks
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist_class = D.Distribution
        num_ptcl = 100
        dist_info = None

        mock_dist.batch_shape, mock_dist.event_shape = Size([3, 4]), Size([])       # Empty event dimension
        mock_dist.sample.side_effect = lambda size: torch.randn([size[0], 3, 4])

        # Patch type so that type(mock_dist) returns mock_dist_class
        with patch("pysigma.utils.type") as t:
            t.side_effect = lambda dist: mock_dist_class if dist is mock_dist else type(dist)

            # Patch class registry
            with patch("pysigma.utils.DistributionServer.dict_draw_particles", new_callable=PropertyMock) as d:
                # Mock special draw particles method
                d.return_value = {}

                returned_ptcl = DS.draw_particles(mock_dist, num_ptcl, dist_info)
                assert returned_ptcl.shape == Size([num_ptcl, 1])       # Singleton event dimension

    def test_draw_particles_default_singleton_real(self):
        # Test the default drawing method with particles with singleton event dimension using real distribution
        # Testing with Normal as example
        b_shape, e_shape = Size([3, 4]), Size([1])

        loc = torch.randn(b_shape)
        scale = torch.randn(b_shape)

        dist = D.Normal(loc, scale)
        num_ptcl = 100
        dist_info = None

        # Patch class registry
        with patch("pysigma.utils.DistributionServer.dict_draw_particles", new_callable=PropertyMock) as d:
            # Mock special draw particles method
            d.return_value = {}

            returned_ptcl = DS.draw_particles(dist, num_ptcl, dist_info)
            assert returned_ptcl.shape == Size([num_ptcl]) + e_shape

    def test_log_prob_incompatible_event_shape_1(self):
        # Test that the dist instance has an incompatible event shape than the last dimension of the provided particles
        s_shape, b_shape, e_shape = Size([10]), Size([2, 3]), Size([4])
        ptcl = torch.randn(s_shape + e_shape)
        # Mock a distribution instance that have expected event shape part of its batch shape
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape = b_shape + e_shape
        mock_dist.event_shape = Size([])     # Empty event shape

        with pytest.raises(AssertionError):
            DS.log_prob(mock_dist, ptcl)

    def test_log_prob_incompatible_event_shape_2(self):
        # Test that the dist instance has an incompatible event shape than the last dimension of the provided particles
        s_shape, b_shape, e_shape = Size([10]), Size([2, 3]), Size([1])
        ptcl = torch.randn(s_shape + e_shape)
        # Mock a distribution instance that have expected event shape part of its batch shape
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape = b_shape + e_shape
        mock_dist.event_shape = Size([2])     # Empty event shape

        with pytest.raises(AssertionError):
            DS.log_prob(mock_dist, ptcl)

    def test_log_prob_correct_mock_non_singleton(self):
        # Test successful operation using mocks
        # Scenario 1: non-singleton event dimension
        s_shape, b_shape, e_shape = Size([10]), Size([2, 3]), Size([4])
        # Mock distribution instance
        ptcl = torch.randn(s_shape + e_shape)
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape = b_shape
        mock_dist.event_shape = e_shape
        # Mocking log_prob() method
        mock_dist.log_prob.side_effect = lambda values: torch.randn(values.shape[:-1])

        returned_value = DS.log_prob(mock_dist, ptcl)
        assert returned_value.shape == b_shape + s_shape

    def test_log_prob_correct_mock_non_singleton_large_shape(self):
        # Test successful operation using mocks
        # Scenario 2: non-singleton event dimension, large sample and batch shapes
        s_shape, b_shape, e_shape = Size([10, 11, 12, 13]), Size([2, 3, 4]), Size([4])
        # Mock distribution instance
        ptcl = torch.randn(s_shape + e_shape)
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape = b_shape
        mock_dist.event_shape = e_shape
        # Mocking log_prob() method
        mock_dist.log_prob.side_effect = lambda values: torch.randn(values.shape[:-1])

        returned_value = DS.log_prob(mock_dist, ptcl)
        assert returned_value.shape == b_shape + s_shape

    def test_log_prob_correct_mock_singleton(self):
        # Test successful operation using mocks
        # Scenario 3: singleton event dimension
        s_shape, b_shape, e_shape = Size([10]), Size([2, 3]), Size([1])
        # Mock distribution instance
        ptcl = torch.randn(s_shape + e_shape)
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape = b_shape
        mock_dist.event_shape = Size([])
        # Mocking log_prob() method
        mock_dist.log_prob.side_effect = lambda values: torch.randn_like(values)

        returned_value = DS.log_prob(mock_dist, ptcl)
        assert returned_value.shape == b_shape + s_shape

    def test_log_prob_correct_mock_singleton_large_shape(self):
        # Test successful operation using mocks
        # Scenario 4: singleton event dimension, large sample and batch shapes
        s_shape, b_shape, e_shape = Size([10, 11, 12, 13]), Size([2, 3, 4]), Size([1])
        # Mock distribution instance
        ptcl = torch.randn(s_shape + e_shape)
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape = b_shape
        mock_dist.event_shape = Size([])
        # Mocking log_prob() method
        mock_dist.log_prob.side_effect = lambda values: torch.randn_like(values)

        returned_value = DS.log_prob(mock_dist, ptcl)
        assert returned_value.shape == b_shape + s_shape

    def test_log_prob_correct_normal_distribution_non_singleton(self):
        # Use Multivariate Normal distribution class to test particles with non-singleton event dimension
        # Test successful operation using mocks
        s_shape, b_shape, e_shape = Size([10]), Size([2, 3]), Size([4])

        # Create MultivariateNormal dist instance with correct batch and event shape
        loc = torch.randn(b_shape + e_shape)
        cov = torch.eye(e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        repeat_times = list(b_shape) + [1, 1]
        cov = cov.repeat(repeat_times)

        dist = D.MultivariateNormal(loc, cov)

        ptcl = torch.randn(s_shape + e_shape)

        returned_value = DS.log_prob(dist, ptcl)
        assert returned_value.shape == b_shape + s_shape

    def test_log_prob_correct_normal_distribution_singleton(self):
        # Use univariate Normal distribution class to test particles with singleton event dimension
        # Test successful operation using mocks
        s_shape, b_shape, e_shape = Size([10]), Size([2, 3]), Size([1])

        # Create Normal dist instance with correct batch and event shape
        loc = torch.randn(b_shape)
        scale = torch.randn(b_shape)

        dist = D.Normal(loc, scale)

        ptcl = torch.randn(s_shape + e_shape)

        returned_value = DS.log_prob(dist, ptcl)
        assert returned_value.shape == b_shape + s_shape

    # endregion

    # region: special methods with Categorical distribution class
    def test_categorical_param2dist(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([10]), Size([])
        dist_class = D.Categorical
        param = torch.rand(b_shape + p_shape)
        expected_param = param / param.sum(dim=-1, keepdim=True)

        dist = DS.param2dist(dist_class, param, b_shape, e_shape)

        assert isinstance(dist, dist_class)
        assert equal_within_error(dist.probs, expected_param)

    def test_categorical_dist2param(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([10]), Size([])
        param = torch.rand(b_shape + p_shape)
        dist = D.Categorical(probs=param)
        dist_info = None
        expected_param = param / param.sum(dim=-1, keepdim=True)

        returned_value = DS.dist2param(dist, dist_info)

        assert equal_within_error(returned_value, expected_param)

    # endregion


class TestKnowledgeServer:

    def test_init(self):
        dist_class = D.MultivariateNormal
        rv_sizes = [2, 3]
        rv_cstr = [C.real, C.real]
        rv_num_particles = [10, 15]
        dist_info = None

        ks = KS(dist_class, rv_sizes, rv_cstr, rv_num_particles, dist_info)

        assert ks.num_rvs == 2 and ks.e_shape == Size(rv_sizes) and ks.s_shape == Size(rv_num_particles)

    # region: Test top-level methods using mocks
    def test_draw_particles(self):
        mock_dist_class = D.Distribution
        rv_sizes, rv_cstr, rv_num_particles = [2, 3], [C.real, C.real], [10, 15]
        dist_info = None
        ks = KS(mock_dist_class, rv_sizes, rv_cstr, rv_num_particles, dist_info)

        # Dummy parameter
        param = torch.randn([4, 5, 6])
        batch_shape = Size([4, 5])
        # Mock batched dist instance
        mock_dist = MagicMock(spec_set=D.Distribution)

        # Set mocked return values
        mock_particles = tuple([torch.randn([10, 2]), torch.randn([15, 3])])
        mock_log_densities = tuple([torch.rand([10]), torch.rand([15])])

        # Patch DistributionServer.param2dist
        with patch("pysigma.utils.DistributionServer.param2dist") as d:
            d.side_effect = lambda arg_dist_class, arg_param, b_shape=None, e_shape=None, dist_info=None: \
                mock_dist if arg_dist_class is mock_dist_class and equal_within_error(arg_param, param) else None

            # Patch class-level method registry to ensure that default_draw is called
            # type(ks).dict_2special_draw = PropertyMock(return_value={})
            with patch.object(ks, "dict_2special_draw", {}):
                # Patch ks._default_draw()
                with patch.object(ks, "_default_draw") as p:
                    p.return_value = (mock_particles, mock_log_densities)
                    return_val_1, return_val_2 = ks.draw_particles(param, batch_shape, update_cache=True)
                    assert all(equal_within_error(a, b) for a, b in zip(return_val_1, mock_particles))
                    assert all(equal_within_error(a, b) for a, b in zip(return_val_2, mock_log_densities))
                    assert all(equal_within_error(a, b) for a, b in zip(return_val_1, ks.particles))
                    assert all(equal_within_error(a, b) for a, b in zip(return_val_2, ks.log_densities))

    def test_surrogate_log_prob_default_particles_no_index_map_single_rv(self):
        # Test using mocks, alt_particles=None, index_map=None, 1 random variable
        # Define shapes
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10]), Size([2])

        # Test data and mock objects
        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = [torch.randn(s_shape + e_shape)]

        param = torch.randn(b_shape + p_shape)
        alt_particles = None
        index_map = None
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, e_shape

        # Mock method calls
        with patch.object(ks, "event2torch_event") as p1:
            p1.side_effect = lambda t: t

            with patch("pysigma.utils.DistributionServer.param2dist") as p2:
                p2.return_value = mock_dist

                with patch("pysigma.utils.DistributionServer.log_prob") as p3:
                    p3.side_effect = lambda dist, ptcl: torch.randn(dist.batch_shape + ptcl.shape[:-1])

                    return_val = ks.surrogate_log_prob(param, alt_particles, index_map)

                    assert return_val.shape == b_shape + s_shape

    def test_surrogate_log_prob_default_particles_no_index_map_multiple_rv(self):
        # Test using mocks, alt_particles=None, index_map=None, 3 random variable
        # Define shapes
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        # Test data and mock objects
        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real,C.real,C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)]

        param = torch.randn(b_shape + p_shape)
        alt_particles = None
        index_map = None
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, e_shape

        # Mock method calls
        with patch.object(ks, "event2torch_event") as p1:
            p1.side_effect = lambda t: t

            with patch("pysigma.utils.DistributionServer.param2dist") as p2:
                p2.return_value = mock_dist

                with patch("pysigma.utils.DistributionServer.log_prob") as p3:
                    p3.side_effect = lambda dist, ptcl: torch.randn(dist.batch_shape + ptcl.shape[:-1])

                    return_val = ks.surrogate_log_prob(param, alt_particles, index_map)

                    assert return_val.shape == b_shape + s_shape

    def test_surrogate_log_prob_full_alt_particles_no_index_map_multiple_rv(self):
        # Test using mocks, full alt_particles, index_map=None, 3 random variable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        # Test data and mock objects
        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)

        param = torch.randn(b_shape + p_shape)
        alt_particles = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)]
        index_map = None
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, e_shape

        # Mock method calls
        with patch.object(ks, "event2torch_event") as p1:
            p1.side_effect = lambda t: t

            with patch("pysigma.utils.DistributionServer.param2dist") as p2:
                p2.return_value = mock_dist

                with patch("pysigma.utils.DistributionServer.log_prob") as p3:
                    p3.side_effect = lambda dist, ptcl: torch.randn(dist.batch_shape + ptcl.shape[:-1])

                    return_val = ks.surrogate_log_prob(param, alt_particles, index_map)

                    assert return_val.shape == b_shape + s_shape

    def test_surrogate_log_prob_partial_alt_particles_no_index_map_multiple_rv(self):
        # Test using mocks, partial alt_particles with None elements, index_map=None, 3 random variable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        # Test data and mock objects
        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)]

        param = torch.randn(b_shape + p_shape)
        alt_particles = [torch.randn(s_shape[0], e_shape[0]), None, torch.randn(s_shape[2], e_shape[2])]
        index_map = None
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, e_shape

        # Mock method calls
        with patch.object(ks, "event2torch_event") as p1:
            p1.side_effect = lambda t: t

            with patch("pysigma.utils.DistributionServer.param2dist") as p2:
                p2.return_value = mock_dist

                with patch("pysigma.utils.DistributionServer.log_prob") as p3:
                    p3.side_effect = lambda dist, ptcl: torch.randn(dist.batch_shape + ptcl.shape[:-1])

                    return_val = ks.surrogate_log_prob(param, alt_particles, index_map)

                    assert return_val.shape == b_shape + s_shape

    def test_surrogate_log_prob_full_alt_particles_one_to_one_surjective_index_map_multiple_rv(self):
        # Test using mocks, full alt_particles, one-to-one index mapping that have all rv mapped, 3 random variable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        # Test data and mock objects
        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)

        param = torch.randn(b_shape + p_shape)
        alt_particles = [torch.randn([20, 4]), torch.randn([10, 2]), torch.randn([15, 3])]
        index_map = {0: 2, 1: 0, 2: 1}
        expected_s_shape = Size([20, 10, 15])
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, e_shape

        # Mock method calls
        with patch.object(ks, "event2torch_event") as p1:
            p1.side_effect = lambda t: t

            with patch("pysigma.utils.DistributionServer.param2dist") as p2:
                p2.return_value = mock_dist

                with patch("pysigma.utils.DistributionServer.log_prob") as p3:
                    p3.side_effect = lambda dist, ptcl: torch.randn(dist.batch_shape + ptcl.shape[:-1])

                    return_val = ks.surrogate_log_prob(param, alt_particles, index_map)

                    assert return_val.shape == b_shape + expected_s_shape

    def test_surrogate_log_prob_full_alt_particles_one_to_one_partial_index_map_multiple_rv_1(self):
        # Test using mocks, full alt_particles, one-to-one index mapping that misses 1 rv, 3 random variable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        # Test data and mock objects
        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)]

        param = torch.randn(b_shape + p_shape)
        # index_map only maps to a partial set of rv
        alt_particles = [torch.randn([20, 4]), torch.randn([15, 3])]
        index_map = {0: 2, 1: 1}
        expected_s_shape = Size([20, 15])
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, e_shape

        # Mock method calls
        with patch.object(ks, "event2torch_event") as p1:
            p1.side_effect = lambda t: t

            with patch("pysigma.utils.DistributionServer.param2dist") as p2:
                p2.return_value = mock_dist

                with patch("pysigma.utils.DistributionServer.log_prob") as p3:
                    p3.side_effect = lambda dist, ptcl: torch.randn(dist.batch_shape + ptcl.shape[:-1])

                    return_val = ks.surrogate_log_prob(param, alt_particles, index_map)

                    assert return_val.shape == b_shape + expected_s_shape

    def test_surrogate_log_prob_full_alt_particles_one_to_one_partial_index_map_multiple_rv_2(self):
        # Test using mocks, full alt_particles, one-to-one index mapping that misses 2 rv, 3 random variable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        # Test data and mock objects
        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)]

        param = torch.randn(b_shape + p_shape)
        # index_map only maps to a partial set of rv
        alt_particles = [torch.randn([20, 4])]
        index_map = {0: 2}
        expected_s_shape = Size([20])
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, e_shape

        # Mock method calls
        with patch.object(ks, "event2torch_event") as p1:
            p1.side_effect = lambda t: t

            with patch("pysigma.utils.DistributionServer.param2dist") as p2:
                p2.return_value = mock_dist

                with patch("pysigma.utils.DistributionServer.log_prob") as p3:
                    p3.side_effect = lambda dist, ptcl: torch.randn(dist.batch_shape + ptcl.shape[:-1])

                    return_val = ks.surrogate_log_prob(param, alt_particles, index_map)

                    assert return_val.shape == b_shape + expected_s_shape

    def test_surrogate_log_prob_full_alt_particles_one_to_multiple_full_index_map_multiple_rv_1(self):
        # Test using mocks, full alt_particles, one-to-multiple index mapping, 3 random variable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        # Test data and mock objects
        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)]

        param = torch.randn(b_shape + p_shape)
        # index_map only maps the first alt particle to 2 rv, and the second alt particle to 1 rv
        #   The first alt particle needs to be the result of combinatorial concat
        index_map = {0: [0, 2], 1: 1}
        to_cat = [torch.randn([10, 2]), torch.randn([20, 4])]
        cat = ks.combinatorial_cat(to_cat).view(-1, 6)
        alt_particles = [cat, torch.randn([15, 3])]
        expected_s_shape = Size([200, 15])
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, e_shape

        # Mock method calls
        with patch.object(ks, "event2torch_event") as p1:
            p1.side_effect = lambda t: t

            with patch("pysigma.utils.DistributionServer.param2dist") as p2:
                p2.return_value = mock_dist

                with patch("pysigma.utils.DistributionServer.log_prob") as p3:
                    p3.side_effect = lambda dist, ptcl: torch.randn(dist.batch_shape + ptcl.shape[:-1])

                    return_val = ks.surrogate_log_prob(param, alt_particles, index_map)

                    assert return_val.shape == b_shape + expected_s_shape

    def test_surrogate_log_prob_full_alt_particles_one_to_multiple_full_index_map_multiple_rv_2(self):
        # Test using mocks, full alt_particles, one-to-multiple index mapping, 3 random variable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        # Test data and mock objects
        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)]

        param = torch.randn(b_shape + p_shape)
        # index_map only maps the first alt particle to all rv
        #   The first alt particle needs to be the result of combinatorial concat
        index_map = {0: [0, 2, 1]}
        to_cat = [torch.randn([10, 2]), torch.randn([20, 4]), torch.randn([15, 3])]
        cat = ks.combinatorial_cat(to_cat).view(-1, 9)
        alt_particles = [cat]
        expected_s_shape = Size([3000])
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, e_shape

        # Mock method calls
        with patch.object(ks, "event2torch_event") as p1:
            p1.side_effect = lambda t: t

            with patch("pysigma.utils.DistributionServer.param2dist") as p2:
                p2.return_value = mock_dist

                with patch("pysigma.utils.DistributionServer.log_prob") as p3:
                    p3.side_effect = lambda dist, ptcl: torch.randn(dist.batch_shape + ptcl.shape[:-1])

                    return_val = ks.surrogate_log_prob(param, alt_particles, index_map)

                    assert return_val.shape == b_shape + expected_s_shape

    def test_surrogate_log_prob_exception_none_alt_particles_when_no_cache(self):
        # Test that AssertionError is thrown if alt_particles is None, or contains None when no cached particles
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = None

        param = torch.randn(b_shape + p_shape)

        # Test 1: alt_particles is None
        alt_particles = None
        with pytest.raises(AssertionError):
            ks.surrogate_log_prob(param, alt_particles)

        # Test 2: alt_particles is a list that contains None
        alt_particles = [torch.randn([10, 2]), None, None]
        with pytest.raises(AssertionError):
            ks.surrogate_log_prob(param, alt_particles)

    def test_surrogate_log_prob_exception_none_alt_particles_when_index_map_specified(self):
        # Test that AssertionError is thrown if alt_particles is None when index_map is specified
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = None

        param = torch.randn(b_shape + p_shape)
        alt_particles = None
        index_map = {0:0, 1:1, 2:2}
        with pytest.raises(AssertionError):
            ks.surrogate_log_prob(param, alt_particles, index_map)

    def test_surrogate_log_prob_exception_wrong_index_map_key(self):
        # Test that AssertionError is thrown if the keys in index_map are not indexable in alt_particles
        # Scenario 1: missing some indices
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = None

        param = torch.randn(b_shape + p_shape)
        alt_particles = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)]
        index_map = {0: 0, 1: 1}
        with pytest.raises(AssertionError):
            ks.surrogate_log_prob(param, alt_particles, index_map)

    def test_surrogate_log_prob_exception_wrong_index_map_value_1(self):
        # Test that AssertionError is thrown if the values in index_map are not indexable in alt_particles
        # Scenario 1: single value not indexable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = None

        param = torch.randn(b_shape + p_shape)
        alt_particles = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)]
        index_map = {0: 0, 1: 1, 2: 3}
        with pytest.raises(AssertionError):
            ks.surrogate_log_prob(param, alt_particles, index_map)

    def test_surrogate_log_prob_exception_wrong_index_map_value_2(self):
        # Test that AssertionError is thrown if the values in index_map are not indexable in alt_particles
        # Scenario 1: some elements in a list of values not indexable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = None

        param = torch.randn(b_shape + p_shape)
        alt_particles = [torch.randn([10, 2]),
                         ks.combinatorial_cat([torch.randn(15, 3), torch.randn([20, 4])]).view(-1, 7)]
        index_map = {0: 0, 1: [1, 3]}
        with pytest.raises(AssertionError):
            ks.surrogate_log_prob(param, alt_particles, index_map)

    def test_surrogate_log_prob_exception_none_entry_mapped_index_map_list(self):
        # Test that AssertionError is thrown if i in index_map is mapped to a list of indices, while the i-th item
        #   in alt_particles is None
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = None

        param = torch.randn(b_shape + p_shape)
        alt_particles = [torch.randn([10, 2]), None]
        index_map = {0: 0, 1: [1, 2]}
        with pytest.raises(AssertionError):
            ks.surrogate_log_prob(param, alt_particles, index_map)

    def test_surrogate_log_prob_exception_wrong_alt_particles_event_shape_1(self):
        # Test that Assertion error is thrown if the event shape of some item in alt_particles is wrong
        # Scenario 1: map to single rv
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = None

        param = torch.randn(b_shape + p_shape)
        alt_particles = [torch.randn([10, 2]), torch.randn([15, 3]), torch.randn([20, 5])]
        index_map = None
        with pytest.raises(AssertionError):
            ks.surrogate_log_prob(param, alt_particles, index_map)

    def test_surrogate_log_prob_exception_wrong_alt_particles_event_shape_2(self):
        # Test that Assertion error is thrown if the event shape of some item in alt_particles is wrong
        # Scenario 2: map to multiple rv
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)
        ks.particles = None

        param = torch.randn(b_shape + p_shape)
        alt_particles = [torch.randn([10, 2]), torch.randn([15, 8])]
        index_map = {0: 0, 1: [1, 2]}
        with pytest.raises(AssertionError):
            ks.surrogate_log_prob(param, alt_particles, index_map)

    def test_event2torch_event_invalid_event_shape(self):
        # Test that AssertionError is thrown if cat_particles has wrong concatenated event shape
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)

        cat_particles = torch.randn([11, 12, 13, 10])

        with pytest.raises(AssertionError):
            ks.event2torch_event(cat_particles)

    def test_event2torch_event_default(self):
        # Test default behavior using mocks
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)

        cat_particles = torch.randn([11, 12, 13, 9])

        # Patch registry dictionary to ensure dist_class is not registered
        with patch.object(ks, "dict_2torch_event", {}):
            return_val = ks.event2torch_event(cat_particles)

            assert equal_within_error(return_val, cat_particles)

    def test_event2torch_event_registry(self):
        # Test translation behavior using mocks
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)

        cat_particles = torch.randn([11, 12, 13, 9])

        # Patch registry dictionary to ensure dist_class is registered
        mock_callable = lambda t: torch.randn(t.shape[:-1] + Size([1]))
        with patch.object(ks, "dict_2torch_event", {dist_class: mock_callable}):
            return_val = ks.event2torch_event(cat_particles)

            assert return_val.shape == Size([11, 12, 13, 1])

    def test_event2cognitive_event_default(self):
        # Test default behavior using mocks
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)

        cat_particles = torch.randn([11, 12, 13, 9])

        # Patch registry dictionary to ensure dist_class is not registered
        with patch.object(ks, "dict_2cognitive_event", {}):
            return_val = ks.event2cognitive_event(cat_particles)

            assert equal_within_error(return_val, cat_particles)

    def test_event2cognitive_event_registry(self):
        # Test translation behavior using mocks
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([2]), Size([10, 15, 20]), Size([2, 3, 4])

        dist_class = D.Distribution
        rv_sizes = e_shape
        rv_constraints = (C.real, C.real, C.real,)
        rv_num_particles = s_shape
        dist_info = None
        ks = KS(dist_class, rv_sizes, rv_constraints, rv_num_particles, dist_info)

        cat_particles = torch.randn([11, 12, 13, 18])
        # Patch registry dictionary to ensure dist_class is registered
        mock_callable = lambda t: torch.randn(t.shape[:-1] + Size([9]))
        with patch.object(ks, "dict_2cognitive_event", {dist_class: mock_callable}):
            return_val = ks.event2cognitive_event(cat_particles)

            assert return_val.shape == Size([11, 12, 13, 9])

    def test_combinatorial_cat_single_particle(self):
        # Test that the original particle tensor is returned if there's only one particle in the input list
        particles = [torch.randn([20, 5])]
        return_val = KS.combinatorial_cat(particles)
        assert equal_within_error(return_val, particles[0])

    def test_combinatorial_cat_shape(self):
        # Test correctness of return value shape
        s_shape, e_shape = Size([2, 3, 4, 5]), Size([6, 7, 8, 9])
        particles = [torch.randn(s, e) for s, e in zip(s_shape, e_shape)]
        return_val = KS.combinatorial_cat(particles)
        assert return_val.shape == s_shape + Size([sum(e_shape)])

    def test_combinatorial_cat_content_ad_hoc_1(self):
        # Test correctness of contents using ad hoc values
        s_shape, e_shape = Size([2, 3]), Size([2, 1])
        particles = [
            torch.tensor(
                [[0., 1.],
                [2., 3.]]
            ),
            torch.tensor(
                [[4.], [5.], [6.]]
            )
        ]
        expected_val = torch.tensor(
            [[[0., 1., 4.],
             [0., 1., 5.],
             [0., 1., 6.]],
             [[2., 3., 4.],
             [2., 3., 5.],
             [2., 3., 6.]]]
        )
        return_val = KS.combinatorial_cat(particles)
        assert equal_within_error(return_val, expected_val)

    def test_combinatorial_cat_content_ad_hoc_1(self):
        # Test correctness of contents using ad hoc values
        s_shape, e_shape = Size([2, 3]), Size([2, 1])
        particles = [
            torch.tensor(
                [[4.], [5.], [6.]]
            ),
            torch.tensor(
                [[0., 1.],
                [2., 3.]]
            ),
        ]
        expected_val = torch.tensor(
            [[[4., 0., 1.],
              [4., 2., 3.]],
             [[5., 0., 1.],
              [5., 2., 3.]],
             [[6., 0., 1.],
              [6., 2., 3.]]]
        )
        return_val = KS.combinatorial_cat(particles)
        assert equal_within_error(return_val, expected_val)

    def test_combinatorial_decat_ad_hoc_1(self):
        # Test correctness using ad hoc values
        split_size = [2, 1]
        cat_particles = torch.tensor(
            [[[0., 1., 4.],
             [0., 1., 5.],
             [0., 1., 6.]],
             [[2., 3., 4.],
             [2., 3., 5.],
             [2., 3., 6.]]]
        )
        expected_val = [
            torch.tensor(
                [[0., 1.],
                [2., 3.]]
            ),
            torch.tensor(
                [[4.], [5.], [6.]]
            )
        ]
        return_val = KS.combinatorial_decat(cat_particles, split_size)
        assert all(equal_within_error(t, e) for t, e in zip(return_val, expected_val))

    def test_combinatorial_decat_ad_hoc_2(self):
        # Test correctness using ad hoc values
        split_size = [1, 2]
        cat_particles = torch.tensor(
            [[[4., 0., 1.],
              [4., 2., 3.]],
             [[5., 0., 1.],
              [5., 2., 3.]],
             [[6., 0., 1.],
              [6., 2., 3.]]]
        )
        expected_val = [
            torch.tensor(
                [[4.], [5.], [6.]]
            ),
            torch.tensor(
                [[0., 1.],
                [2., 3.]]
            ),
        ]
        return_val = KS.combinatorial_decat(cat_particles, split_size)
        assert all(equal_within_error(t, e) for t, e in zip(return_val, expected_val))

    def test_combinatorial_decat_invalid_val_ad_hoc(self):
        # Test that ValueError is thrown if provided particle cannot be decat, using ad hoc values
        split_size = [1, 2]     # Wrong split size. Therefore cannot be decat
        cat_particles = torch.tensor(
            [[[0., 1., 4.],
              [0., 1., 5.],
              [0., 1., 6.]],
             [[2., 3., 4.],
              [2., 3., 5.],
              [2., 3., 6.]]]
        )
        with pytest.raises(ValueError):
            KS.combinatorial_decat(cat_particles, split_size)

    def test_combinatorial_decat_invalid_val_random(self):
        # Test that ValueError is thrown if provided particle cannot be decat, using random values
        split_size = [1, 2]
        cat_particles = torch.randn([6, 3])
        with pytest.raises(ValueError):
            KS.combinatorial_decat(cat_particles, split_size)

    # endregion

    # region: distribution-class specific method
    def test_default_draw_univariate_single_batch_mock_shape(self):
        # Test correct shape with 1 RV, single batch, with mocks
        b_shape, s_shape, e_shape = Size([1]), Size([150]), Size([1])

        dist_class = D.Distribution
        rv_cstr = (C.real,)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, Size([])

        # Mock dist sample method
        mock_dist.sample.side_effect = lambda size: \
            torch.randn(Size(size) + mock_dist.batch_shape + mock_dist.event_shape)

        # Mock dist log_prob method
        def mock_log_prob(val):
            b_e_len = len(mock_dist.batch_shape) + len(mock_dist.event_shape)
            if val.shape[-b_e_len:] != mock_dist.batch_shape + mock_dist.event_shape:
                raise ValueError("Input value tensor shape not compatible: expect trailing shape {}, but found: {}"
                                 .format(mock_dist.batch_shape + mock_dist.event_shape, val.shape[-b_e_len:]))
            random_log_prob = torch.randn(val.shape[:-b_e_len] + mock_dist.batch_shape)
            return random_log_prob
        mock_dist.log_prob.side_effect = mock_log_prob

        return_ptcl, return_dens = ks._default_draw(mock_dist)

        assert len(return_ptcl) == len(return_dens) == 1
        assert all(p.shape == Size([s_shape[i], e_shape[i]]) for i, p in enumerate(return_ptcl))
        assert all(d.shape == Size([s_shape[i]]) for i, d in enumerate(return_dens))

    def test_default_draw_univariate_single_batch_real_shape(self):
        # Test correct shape with 1 RV, single batch, with real distribution instance
        # Test using univariate Normal distribution as instance
        b_shape, s_shape, e_shape = Size([1]), Size([150]), Size([1])

        # Use real Normal distribution
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        dist = D.Normal(loc, scale)

        rv_cstr = (C.real,)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        return_ptcl, return_dens = ks._default_draw(dist)

        assert len(return_ptcl) == len(return_dens) == 1
        assert all(p.shape == Size([s_shape[i], e_shape[i]]) for i, p in enumerate(return_ptcl))
        assert all(d.shape == Size([s_shape[i]]) for i, d in enumerate(return_dens))

    def test_default_draw_univariate_multiple_batch_mock_shape(self):
        # Test correct shape with 1 RV, multiple batch dimensions, with mocks
        b_shape, s_shape, e_shape = Size([3, 4, 5]), Size([150]), Size([1])

        dist_class = D.Distribution
        rv_cstr = (C.real,)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, Size([])

        # Mock dist sample method
        mock_dist.sample.side_effect = lambda size: \
            torch.randn(Size(size) + mock_dist.batch_shape + mock_dist.event_shape)

        # Mock dist log_prob method
        def mock_log_prob(val):
            b_e_len = len(mock_dist.batch_shape) + len(mock_dist.event_shape)
            if val.shape[-b_e_len:] != mock_dist.batch_shape + mock_dist.event_shape:
                raise ValueError("Input value tensor shape not compatible: expect trailing shape {}, but found: {}"
                                 .format(mock_dist.batch_shape + mock_dist.event_shape, val.shape[-b_e_len:]))
            random_log_prob = torch.randn(val.shape[:-b_e_len] + mock_dist.batch_shape)
            return random_log_prob
        mock_dist.log_prob.side_effect = mock_log_prob

        return_ptcl, return_dens = ks._default_draw(mock_dist)

        assert len(return_ptcl) == len(return_dens) == 1
        assert all(p.shape == Size([s_shape[i], e_shape[i]]) for i, p in enumerate(return_ptcl))
        assert all(d.shape == Size([s_shape[i]]) for i, d in enumerate(return_dens))

    def test_default_draw_univariate_multiple_batch_real_shape(self):
        # Test correct shape with 1 RV, multiple batch dimensions, with real distribution instance
        # Test using univariate Normal distribution as instance
        b_shape, s_shape, e_shape = Size([3, 4, 5]), Size([150]), Size([1])

        # Use real Normal distribution
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        dist = D.Normal(loc, scale)

        rv_cstr = (C.real,)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        return_ptcl, return_dens = ks._default_draw(dist)

        assert len(return_ptcl) == len(return_dens) == 1
        assert all(p.shape == Size([s_shape[i], e_shape[i]]) for i, p in enumerate(return_ptcl))
        assert all(d.shape == Size([s_shape[i]]) for i, d in enumerate(return_dens))

    def test_default_draw_univariate_multiple_batch_real_content(self):
        # Test correct content with 1 RV, multiple batch dimensions, with real distribution instance
        # Test using univariate Normal distribution as instance
        b_shape, s_shape, e_shape = Size([3, 4, 5]), Size([150]), Size([1])

        # Use real Normal distribution
        dist_class = D.Normal
        loc, scale = torch.randn(b_shape), torch.rand(b_shape)
        dist = D.Normal(loc, scale)

        rv_cstr = (C.real,)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        return_ptcl, return_dens = ks._default_draw(dist)

        # Test that the density is indeed the batch-average density
        ptcl, dens = return_ptcl[0], return_dens[0]

        # Get rid of singleton event dimension
        ptcl = ptcl.squeeze(dim=-1)
        # Unsqueeze and expand to insert batch dimensions
        for i in range(len(b_shape)):
            ptcl = ptcl.unsqueeze(dim=-1)
        ptcl = ptcl.expand_as(torch.ones(s_shape + b_shape))

        # Take average across batch
        log_prob = dist.log_prob(ptcl)
        prob = torch.exp(log_prob)
        batch_avg_prob = prob.mean(dim=list(range(len(s_shape), len(s_shape) + len(b_shape))))
        batch_avg_log_prob = torch.log(batch_avg_prob)

        assert equal_within_error(dens, batch_avg_log_prob)

    def test_default_draw_multivariate_single_batch_mock_shape(self):
        # Test correct shape with 3 RVs, single batch, with mocks
        b_shape, s_shape, e_shape = Size([1]), Size([10, 15, 20]), Size([1, 2, 3])

        dist_class = D.Distribution
        rv_cstr = (C.real, C.real, C.real)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, Size([sum(e_shape)])

        # Mock dist sample method
        mock_dist.sample.side_effect = lambda size: \
            torch.randn(Size(size) + mock_dist.batch_shape + mock_dist.event_shape)

        # Mock dist log_prob method
        def mock_log_prob(val):
            b_e_len = len(mock_dist.batch_shape) + len(mock_dist.event_shape)
            if val.shape[-b_e_len:] != mock_dist.batch_shape + mock_dist.event_shape:
                raise ValueError("Input value tensor shape not compatible: expect trailing shape {}, but found: {}"
                                 .format(mock_dist.batch_shape + mock_dist.event_shape, val.shape[-b_e_len:]))
            random_log_prob = torch.randn(val.shape[:-b_e_len] + mock_dist.batch_shape)
            return random_log_prob
        mock_dist.log_prob.side_effect = mock_log_prob

        return_ptcl, return_dens = ks._default_draw(mock_dist)

        assert len(return_ptcl) == len(return_dens) == 3
        assert all(p.shape == Size([s_shape[i], e_shape[i]]) for i, p in enumerate(return_ptcl))
        assert all(d.shape == Size([s_shape[i]]) for i, d in enumerate(return_dens))

    def test_default_draw_multivariate_single_batch_real_shape(self):
        # Test correct shape with 3 RVs, single batch, with real distribution instance
        # Test using MultivariateNormal distribution as instance
        b_shape, s_shape, e_shape = Size([1]), Size([10, 15, 20]), Size([1, 2, 3])

        # Use real Normal distribution
        dist_class = D.MultivariateNormal
        dist_e_shape = Size([sum(e_shape)])
        loc = torch.randn(b_shape + dist_e_shape)
        cov = torch.eye(dist_e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        repeat_times = list(b_shape) + [1, 1]
        cov = cov.repeat(repeat_times)

        dist = D.MultivariateNormal(loc, cov)

        rv_cstr = [C.real,] * len(e_shape)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        return_ptcl, return_dens = ks._default_draw(dist)

        assert len(return_ptcl) == len(return_dens) == 3
        assert all(p.shape == Size([s_shape[i], e_shape[i]]) for i, p in enumerate(return_ptcl))
        assert all(d.shape == Size([s_shape[i]]) for i, d in enumerate(return_dens))

    def test_default_draw_multivariate_single_batch_real_content(self):
        # Test correct content with 3 RVs, single batch, with real distribution instance
        # Test using MultivariateNormal distribution as instance
        # We test that the returned log densities are approximations of the actual marginal log densities up to a
        #   constant factor
        # Since we are using MultivariateNormal distribution, its marginal distributions w.r.t. RV of any size has a
        #   nice and closed form expression, and more conveniently, happens to also be a MultivariateNormal distribution
        #   with restricted mean vector and covariance matrix. See here for more details:
        #       https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Marginal_distributions
        b_shape, s_shape, e_shape = Size([1]), Size([10, 15, 20]), Size([1, 2, 3])

        # Use real Normal distribution
        dist_class = D.MultivariateNormal
        dist_e_shape = Size([sum(e_shape)])
        loc = torch.randn(b_shape + dist_e_shape)
        cov = torch.eye(dist_e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        repeat_times = list(b_shape) + [1, 1]
        cov = cov.repeat(repeat_times)

        dist = D.MultivariateNormal(loc, cov)

        rv_cstr = [C.real, ] * len(e_shape)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        return_ptcl, return_log_dens = ks._default_draw(dist)
        return_dens = [torch.exp(d) for d in return_log_dens]

        # The actual marginal distribution for each RV
        marg_dist_list = [
            D.MultivariateNormal(loc[0, :1], cov[0, :1, :1]),
            D.MultivariateNormal(loc[0, 1:3], cov[0, 1:3, 1:3]),
            D.MultivariateNormal(loc[0, 3:6], cov[0, 3:6, 3:6])
        ]
        actual_dens = [marg_dist.log_prob(p).exp() for marg_dist, p in zip(marg_dist_list, return_ptcl)]

        # Check that the returned densities are proportional to the actual densities
        return_dens_normal = [d / d.sum() for d in return_dens]
        actual_dens_normal = [d / d.sum() for d in actual_dens]
        assert all(equal_within_error(r, a) for r, a in zip(return_dens_normal, actual_dens_normal))

    def test_default_draw_multivariate_multiple_batch_mock_shape(self):
        # Test correct shape with 3 RVs, multiple batch, with mocks
        b_shape, s_shape, e_shape = Size([5, 6, 7, 8]), Size([10, 15, 20]), Size([1, 2, 3])

        dist_class = D.Distribution
        rv_cstr = (C.real, C.real, C.real)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape, mock_dist.event_shape = b_shape, Size([sum(e_shape)])

        # Mock dist sample method
        mock_dist.sample.side_effect = lambda size: \
            torch.randn(Size(size) + mock_dist.batch_shape + mock_dist.event_shape)

        # Mock dist log_prob method
        def mock_log_prob(val):
            b_e_len = len(mock_dist.batch_shape) + len(mock_dist.event_shape)
            if val.shape[-b_e_len:] != mock_dist.batch_shape + mock_dist.event_shape:
                raise ValueError("Input value tensor shape not compatible: expect trailing shape {}, but found: {}"
                                 .format(mock_dist.batch_shape + mock_dist.event_shape, val.shape[-b_e_len:]))
            random_log_prob = torch.randn(val.shape[:-b_e_len] + mock_dist.batch_shape)
            return random_log_prob

        mock_dist.log_prob.side_effect = mock_log_prob

        return_ptcl, return_dens = ks._default_draw(mock_dist)

        assert len(return_ptcl) == len(return_dens) == 3
        assert all(p.shape == Size([s_shape[i], e_shape[i]]) for i, p in enumerate(return_ptcl))
        assert all(d.shape == Size([s_shape[i]]) for i, d in enumerate(return_dens))

    def test_default_draw_multivariate_multiple_batch_real_shape(self):
        # Test correct shape with 3 RVs, multiple batch, with real distribution instance
        # Test using MultivariateNormal distribution as instance
        b_shape, s_shape, e_shape = Size([5, 6, 7, 8]), Size([10, 15, 20]), Size([1, 2, 3])

        # Use real Normal distribution
        dist_class = D.MultivariateNormal
        dist_e_shape = Size([sum(e_shape)])
        loc = torch.randn(b_shape + dist_e_shape)
        cov = torch.eye(dist_e_shape[0])
        for i in range(len(b_shape)):
            cov = cov.unsqueeze(dim=0)
        repeat_times = list(b_shape) + [1, 1]
        cov = cov.repeat(repeat_times)

        dist = D.MultivariateNormal(loc, cov)

        rv_cstr = [C.real,] * len(e_shape)
        ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))

        return_ptcl, return_dens = ks._default_draw(dist)

        assert len(return_ptcl) == len(return_dens) == 3
        assert all(p.shape == Size([s_shape[i], e_shape[i]]) for i, p in enumerate(return_ptcl))
        assert all(d.shape == Size([s_shape[i]]) for i, d in enumerate(return_dens))

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

    def test_categorical_2cognitive_event_univariate(self):
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
