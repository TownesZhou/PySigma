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

    def test_draw_particles_default(self):
        # Test the default drawing method.
        # Mock dist instance
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

    def test_log_prob_differnt_event_shape(self):
        # Test that the dist instance has a different event shape than the last dimension of the provided particles

        s_shape, b_shape, e_shape = Size([10]), Size([2, 3]), Size([4])
        ptcl = torch.randn(s_shape + e_shape)
        # Mock a distribution instance that have expected event shape part of its batch shape
        mock_dist = MagicMock(spec_set=D.Distribution)
        mock_dist.batch_shape = b_shape + e_shape
        mock_dist.event_shape = Size([])     # Empty event shape

        with pytest.raises(AssertionError):
            DS.log_prob(mock_dist, ptcl)

    def test_log_prob_correct_mock(self):
        # Test successful operation using mocks
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

    def test_log_prob_correct_normal_distribution(self):
        # Use Normal distribution class to test
        # Test successful operation using mocks
        s_shape, b_shape, e_shape = Size([10]), Size([2, 3]), Size([4])

        # Create Normal dist instance with correct batch and event shape
        loc, scale = torch.randn(b_shape + e_shape), torch.rand(b_shape + e_shape)
        org_dist = D.Normal(loc, scale)
        dist = D.Independent(org_dist, len(org_dist.batch_shape) - len(b_shape))

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

    # TODO: wait for after refactor
    @pytest.mark.skip("Wait for refactor")
    def test_categorical_draw_particles(self):
        b_shape, p_shape, e_shape = Size([1, 2, 3]), Size([10]), Size([])
        n_particles = 20
        param = torch.rand(b_shape + p_shape)
        dist = D.Categorical(probs=param)
        dist_info = None

        returned_value = DS.draw_particles(dist, n_particles, dist_info)
        assert returned_value.shape == Size([n_particles]) + e_shape

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
