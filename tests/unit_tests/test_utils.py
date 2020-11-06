"""
    Unit tests for pysigma.utils module
"""
import pytest
from unittest.mock import MagicMock, PropertyMock, patch
import torch
from torch import Size
import torch.distributions as D

from pysigma.utils import intern_name, extern_name, compatible_shape
from pysigma.utils import DistributionServer as DS
from pysigma.utils import KnowledgeServer as KS


# region
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

    # region
    # Test top-level methods using mocks
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
            # Set content of mocked class registry dictionary
            d.return_value = {dist_class: mock_callable}

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
            # Set content of mocked class registry dictionary
            d.return_value = {dist_class: mock_callable}

            assert DS.param2dist(dist_class, random_param,
                                 b_shape=b_shape, e_shape=e_shape, dist_info=dist_info) is dist
            mock_callable.assert_called_once_with(random_param, dist_info)

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
                d.return_value = {mock_dist_class: mock_callable}

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
                d.return_value = {mock_dist_class: mock_callable}

                assert torch.equal(DS.get_moments(mock_dist, n_moments), expected_moments)
                mock_callable.assert_called_once_with(mock_dist, n_moments)
    # endregion
