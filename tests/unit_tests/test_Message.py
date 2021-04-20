"""
    Unit tests for Message class
"""

import pytest
import torch
from torch import Size
import torch.distributions as D
import torch.distributions.transforms as T
from pysigma.defs import Message, MessageType
from tests.utils import cuda_only, assert_equal_within_error, assert_proportional_within_error, random_message

# Numerical accuracy
EPS = 1e-5


class TestMessageInitialization:
    # Test message initialization

    def test_not_id_parameter_type_single_relvar(self):
        # Test correct initialization
        # None-identity message
        # Parameter type
        # One single relational variables
        t1 = torch.randn([5, 3])
        msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        assert_equal_within_error(msg.parameter, t1)

    def test_not_id_parameter_type_multiple_relvar(self):
        # Test correct initialization
        # None-identity message
        # Parameter type
        # Two single relational variables
        t1 = torch.randn([5, 6, 3])
        msg = Message(MessageType.Parameter, batch_shape=Size([5, 6]), param_shape=Size([3]), parameter=t1)
        assert_equal_within_error(msg.parameter, t1)

    def test_not_id_particles_type_single_relvar_single_ranvar(self):
        # None-identity message
        # Particles type
        # One relational and random variable
        b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
        sample_dims = [-1]
        p1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                      particles=p1, weight=w1, log_densities=l1)
        # In this case, both the weight and log density tensor would be normalized.
        # Check that they are proportional to the original tensor and that their statistics are correct
        # Check particles
        for p1, p2 in zip(msg.particles, p1):
            assert_equal_within_error(p1, p2)
        # Check weight
        assert_proportional_within_error(msg.weight, w1, dims=sample_dims)
        assert_equal_within_error(msg.weight.sum(dim=sample_dims), torch.tensor([1.]))
        # Check density
        for d1, d2 in zip(msg.log_densities, l1):
            assert_proportional_within_error(d1.exp(), d2.exp(), dims=[0])
            assert_equal_within_error(d1.sum(), torch.tensor([0.]))

    def test_not_id_particles_type_multiple_relvar_multiple_ranvar(self):
        # None-identity message
        # Particles type
        # Multiple relational and multiple variables
        b_s, s_s, e_s = Size([5, 6, 7]), Size([10, 12, 14]), Size([3, 2, 1])
        sample_dims = [-1, -2, -3]
        p1 = [torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])]
        w1 = torch.rand(5, 6, 7, 10, 12, 14)
        l1 = [-torch.rand(10), -torch.rand(12), -torch.rand(14)]
        msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                      particles=p1, weight=w1, log_densities=l1)
        # In this case, both the weight and log density tensor would be normalized.
        # Check that they are proportional to the original tensor and that their statistics are correct
        # Check particles
        for p1, p2 in zip(msg.particles, p1):
            assert_equal_within_error(p1, p2)
        # Check weight
        assert_proportional_within_error(msg.weight, w1, dims=sample_dims)
        assert_equal_within_error(msg.weight.sum(dim=sample_dims), torch.tensor([1.]))
        # Check density
        for d1, d2 in zip(msg.log_densities, l1):
            assert_proportional_within_error(d1.exp(), d2.exp(), dims=[0])
            assert_equal_within_error(d1.sum(), torch.tensor([0.]))

    def test_not_id_dual_type_single_relvar_single_ranvar(self):
        # None-identity message
        # Dual type
        # One relational and random variable
        b_s, p_s, s_s, e_s = Size([5]), Size([4]), Size([10]), Size([3])
        sample_dims = [-1]
        param1 = torch.randn([5, 4])
        part1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg = Message(MessageType.Dual, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=param1, particles=part1, weight=w1, log_densities=l1)
        # In this case, both the weight and log density tensor would be normalized.
        # Check that they are proportional to the original tensor and that their statistics are correct
        # Check parameter
        assert_equal_within_error(msg.parameter, param1)
        # Check particle value
        for p1, p2 in zip(msg.particles, part1):
            assert_equal_within_error(p1, p2)
        # Check weight
        assert_proportional_within_error(msg.weight, w1, dims=sample_dims)
        assert_equal_within_error(msg.weight.sum(dim=sample_dims), torch.tensor([1.]))
        # Check density
        for d1, d2 in zip(msg.log_densities, l1):
            assert_proportional_within_error(d1.exp(), d2.exp(), dims=[0])
            assert_equal_within_error(d1.sum(), torch.tensor([0.]))

    def test_not_id_dual_type_multiple_relvar_multiple_ranvar(self):
        # None-identity message
        # Dual type
        # Multiple relational and random variables
        b_s, p_s, s_s, e_s = Size([5, 6]), Size([4]), Size([10, 11]), Size([3, 2])
        sample_dims = [-1, -2]
        param1 = torch.randn([5, 6, 4])
        part1 = [torch.randn(10, 3), torch.randn(11, 2)]
        w1 = torch.rand(5, 6, 10, 11)
        l1 = [-torch.rand(10), -torch.rand(11)]
        msg = Message(MessageType.Particles, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=param1, particles=part1, weight=w1, log_densities=l1)
        # In this case, both the weight and log density tensor would be normalized.
        # Check that they are proportional to the original tensor and that their statistics are correct
        # Check parameter
        assert_equal_within_error(msg.parameter, param1)
        # Check particle value
        for p1, p2 in zip(msg.particles, part1):
            assert_equal_within_error(p1, p2)
        # Check weight
        assert_proportional_within_error(msg.weight, w1, dims=sample_dims)
        assert_equal_within_error(msg.weight.sum(dim=sample_dims), torch.tensor([1.]))
        # Check density
        for d1, d2 in zip(msg.log_densities, l1):
            assert_proportional_within_error(d1.exp(), d2.exp(), dims=[0])
            assert_equal_within_error(d1.sum(), torch.tensor([0.]))

    def test_id_no_batch_shape_parameter_type(self):
        # Identity with no batch shape specified
        # Parameter type
        p_s = Size([3])
        msg = Message(MessageType.Parameter, param_shape=p_s, parameter=0)

        assert msg.isid
        assert msg.parameter == 0
        assert msg.b_shape == Size([])

    def test_id_no_batch_shape_particles_type(self):
        # Identity with no batch shape specified
        # Particles type
        s_s, e_s = Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        l1 = [-torch.rand(10)]
        msg = Message(MessageType.Particles, sample_shape=s_s, event_shape=e_s, particles=p1, weight=1,
                      log_densities=l1)

        assert msg.isid
        assert msg.weight == 1
        assert msg.b_shape == Size([])

    def test_id_no_batch_shape_dual_type(self):
        # Identity with no batch shape specified
        # Dual type
        p_s = Size([3])
        s_s, e_s = Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        l1 = [-torch.rand(10)]
        msg = Message(MessageType.Dual, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=0, particles=p1, weight=1, log_densities=l1)

        assert msg.isid
        assert msg.parameter == 0
        assert msg.weight == 1
        assert msg.b_shape == Size([])

    def test_id_with_batch_shape_parameter_type(self):
        # Identity with batch shape specified
        # Parameter type
        b_s = Size([5, 6])
        p_s = Size([3])
        msg = Message(MessageType.Parameter, batch_shape=Size([5, 6]), param_shape=p_s, parameter=0)

        assert msg.isid
        assert msg.parameter == 0
        assert msg.b_shape == b_s

    def test_id_with_batch_shape_particles_type(self):
        # Identity with batch shape specified
        # Parameter type
        b_s = Size([5, 6])
        s_s, e_s = Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        l1 = [-torch.rand(10)]
        msg = Message(MessageType.Particles, batch_shape=Size([5, 6]), sample_shape=s_s, event_shape=e_s,
                particles=p1, weight=1, log_densities=l1)

        assert msg.isid
        assert msg.weight == 1
        assert msg.b_shape == b_s

    def test_id_with_batch_shape_dual_type(self):
        # Identity with batch shape specified
        # Parameter type
        b_s = Size([5, 6])
        p_s = Size([3])
        s_s, e_s = Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        l1 = [-torch.rand(10)]
        msg = Message(MessageType.Dual, batch_shape=Size([5, 6]), param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=0, particles=p1, weight=1, log_densities=l1)

        assert msg.isid
        assert msg.parameter == 0
        assert msg.weight == 1
        assert msg.b_shape == b_s

    @cuda_only
    def test_with_gpu(self):
        # Test correct init with GPU tensors
        device = torch.cuda.current_device()
        b_s, p_s, s_s, e_s = Size([5]), Size([3]), Size([10]), Size([3])
        t1 = torch.randn([5, 3])
        p1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg = Message(MessageType.Particles, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=t1, particles=p1, weight=w1, log_densities=l1, device=device)
        assert msg.parameter.get_device() == device
        assert msg.weight.get_device() == device
        for p in msg.particles:
            assert p.get_device() == device
        for d in msg.log_densities:
            assert d.get_device() == device

    def test_wrong_shape_parameter_type_1(self):
        with pytest.raises(AssertionError):
            t1 = torch.randn([5, 3])
            msg = Message(MessageType.Parameter, batch_shape=Size([2]), param_shape=Size([3]), parameter=t1)

    def test_wrong_shape_parameter_type_2(self):
        with pytest.raises(AssertionError):
            t1 = torch.randn([5, 3])
            msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([2]), parameter=t1)

    def test_wrong_shape_parameter_type_3(self):
        # Empty shapes
        with pytest.raises(AssertionError):
            t1 = torch.randn([5, 3, 2])
            msg = Message(MessageType.Parameter, batch_shape=Size([5, 3, 2]), param_shape=Size([]), parameter=t1)

    def test_wrong_shape_parameter_type_4(self):
        # Empty shapes
        with pytest.raises(AssertionError):
            t1 = torch.randn([5, 3, 2])
            msg = Message(MessageType.Parameter, batch_shape=Size([]), param_shape=Size([5, 3, 2]), parameter=t1)

    def test_wrong_shape_parameter_type_5(self):
        # Param shape longer than 1
        with pytest.raises(AssertionError):
            t1 = torch.randn([5, 3, 2])
            msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3, 2]), parameter=t1)

    def test_wrong_shape_particles_type_1(self):
        # Empty shape
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([]), Size([10]), Size([3])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(10)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_2(self):
        # Empty shape
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([]), Size([3])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(5)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_3(self):
        # Empty shape
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(5)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_4(self):
        # Empty shape
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([]), Size([])
            p1 = []
            w1 = torch.rand(5)
            l1 = []
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_5(self):
        # Mismatched shape
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
            p1 = [torch.randn(10, 4)]
            w1 = torch.rand(5, 10)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_6(self):
        # Mismatched shape
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(5, 11)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_7(self):
        # Mismatched shape
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(5, 10)
            l1 = [-torch.rand(11)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_8(self):
        # Wrong number of particle tensors
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(5, 6, 10, 12)
            l1 = [-torch.rand(10), -torch.rand(12)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_9(self):
        # Wrong number of particle tensors
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3), torch.randn([12, 2]), torch.randn([13, 19])]
            w1 = torch.rand(5, 6, 10, 12)
            l1 = [-torch.rand(10), -torch.rand(12)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_10(self):
        # Wrong number of particle tensors
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3), torch.randn([12, 2])]
            w1 = torch.rand(5, 6)
            l1 = [-torch.rand(10), -torch.rand(12)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_11(self):
        # Wrong number of particle tensors
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3), torch.randn([12, 2])]
            w1 = torch.rand(5, 6, 10, 12, 13, 19)
            l1 = [-torch.rand(10), -torch.rand(12)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_12(self):
        # Wrong number of particle tensors
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3), torch.randn([12, 2])]
            w1 = torch.rand(5, 6, 10, 12)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_shape_particles_type_13(self):
        # Particles in wrong order
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6, 7]), Size([10, 12, 14]), Size([3, 2, 1])
            p1 = [torch.randn(10, 3), torch.randn([14, 1]), torch.randn([12, 2])]
            w1 = torch.rand(5, 6, 7, 10, 14, 12)
            l1 = [-torch.rand(10), -torch.rand(14), -torch.rand(12)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_values(self):
        # Wrong weight value - not necessarily positive
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
            p1 = [torch.randn(10, 3)]
            w1 = torch.randn(5, 10)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_wrong_device(self):
        device = "Some random string"

        with pytest.raises(Exception):
            t1 = torch.randn([5, 3])
            msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1,
                          device=device)


class TestMessageProperties:

    def test_isid(self):
        b_s, p_s, s_s, e_s = Size([5]), Size([3]), Size([10]), Size([3])
        param = torch.randn(5, 3)
        part = [torch.randn(10, 3)]
        w = torch.rand(5, 10)
        l = [-torch.rand(10)]

        # Test against identity messages
        # Parameter
        msg = Message(MessageType.Parameter, param_shape=p_s, parameter=0)
        assert msg.isid

        # Particles
        msg = Message(MessageType.Particles, sample_shape=s_s, event_shape=e_s,
                      particles=part, weight=1, log_densities=l)
        assert msg.isid

        # Both
        # weight and parameter both uniform
        msg = Message(MessageType.Dual, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=0, particles=part, weight=1, log_densities=l)
        assert msg.isid

        # Test against non-identity messages
        # Parameter
        msg = Message(MessageType.Parameter, batch_shape=b_s, param_shape=p_s, parameter=param)
        assert not msg.isid

        # Particles
        msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                      particles=part, weight=w, log_densities=l)
        assert not msg.isid

        # Both
        # weight and parameter both not uniform
        msg = Message(MessageType.Dual, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=param, particles=part, weight=w, log_densities=l)
        assert not msg.isid

        # weight is uniform but parameter not uniform
        msg = Message(MessageType.Dual, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=param, particles=part, weight=1, log_densities=l)
        assert not msg.isid

        # parameter is uniform but weight not uniform
        msg = Message(MessageType.Dual, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=0, particles=part, weight=w, log_densities=l)
        assert not msg.isid


class TestMessageSpecialMethods:
    # Test operator overloading

    def test_eq_parameter_type_equal(self):
        t1 = torch.randn([5, 3])
        t2 = t1.clone()
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2)
        assert msg1 == msg2

    def test_eq_parameter_type_not_equal(self):
        t1 = torch.randn([5, 3])
        t2 = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2)
        assert msg1 != msg2

    def test_eq_particles_type_equal(self):
        b_s, p_s, s_s, e_s = Size([5]), Size([4]), Size([10]), Size([3])
        param1 = torch.randn([5, 4])
        part1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        w2 = w1.clone()
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                       parameter=param1, particles=part1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Particles, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                       parameter=param1, particles=part1, weight=w2, log_densities=l1)
        assert msg1 == msg2

    def test_eq_particles_type_not_equal(self):
        b_s, p_s, s_s, e_s = Size([5]), Size([4]), Size([10]), Size([3])
        param1 = torch.randn([5, 4])
        part1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        w2 = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                       parameter=param1, particles=part1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Particles, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                       parameter=param1, particles=part1, weight=w2, log_densities=l1)
        assert msg1 != msg2

    def test_eq_with_auxiliary_attributes_equal(self):
        # Test equality with auxiliary attributes to the message
        t1 = torch.randn([5, 3])
        t2 = t1.clone()
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1,
                       a=1, b=2, c=3, extra=4)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2,
                       a=1, b=2, c=3, extra=4)
        assert msg1 == msg2

    def test_eq_with_auxiliary_attributes_not_equal(self):
        # Test equality with auxiliary attributes to the message
        t1 = torch.randn([5, 3])
        t2 = t1.clone()
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2,
                       a=1, b=2, c=3, extra=4)
        assert msg1 != msg2

    @cuda_only
    def test_eq_cuda_device(self):
        # Test with different devices
        # Only run this test if GPU is available
        device = torch.cuda.current_device()

        t1 = torch.randn([5, 3])
        t2 = t1.clone()
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2,
                       device=device)
        assert msg1 == msg2

    def test_add_invalid_addend_non_message_1(self):
        # Test add to non-message
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        with pytest.raises(AssertionError):
            msg = msg1 + 1

    def test_add_invalid_addend_non_message_2(self):
        # Test add to non-message
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg = msg2 + "something random"

    def test_add_invalid_addend_incompatible_message_type(self):
        # Test add messages with incompatible type
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

    def test_add_invalid_addend_incompatible_message_shape_1(self):
        # Test add messages with incompatible shapes
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([10]), param_shape=Size([3]),
                       parameter=torch.randn([10, 3]))
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

    def test_add_invalid_addend_incompatible_message_shape_2(self):
        # Test add messages with incompatible shapes
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([6]),
                       parameter=torch.randn([5, 6]))
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

    def test_add_invalid_addend_incompatible_message_shape_3(self):
        # Test add messages with incompatible shapes
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([5]), event_shape=Size([3]),
                       particles=[torch.randn(5, 3)], weight=torch.rand(5, 5), log_densities=[-torch.rand(5)])
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

    def test_add_invalid_addend_incompatible_message_shape_4(self):
        # Test add messages with incompatible shapes
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([6]),
                       particles=[torch.randn(10, 6)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

    def test_add_invalid_addend_incompatible_message_shape_5(self):
        # Test add messages with incompatible shapes
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10, 20]),
                       event_shape=Size([3, 6]),
                       particles=[torch.randn(10, 3), torch.randn(20, 6)], weight=torch.rand(5, 10, 20),
                       log_densities=[-torch.rand(10), -torch.rand(20)])
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

    def test_add_invalid_addend_incompatible_particles_1(self):
        # Test add particles type message with incompatible event / log densities tensor
        part1, part2 = [torch.randn(10, 3)], [torch.randn(10, 3)]
        dens1, dens2 = [-torch.rand(10)], [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=part1, weight=torch.rand(5, 10), log_densities=dens1)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=part2, weight=torch.rand(5, 10), log_densities=dens1)
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

    def test_add_invalid_addend_incompatible_particles_2(self):
        # Test add particles type message with incompatible event / log densities tensor
        part1, part2 = [torch.randn(10, 3)], [torch.randn(10, 3)]
        dens1, dens2 = [-torch.rand(10)], [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=part1, weight=torch.rand(5, 10), log_densities=dens1)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=part1, weight=torch.rand(5, 10), log_densities=dens2)
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

    def test_add_correct_parameter_type(self):
        # Test that the resulting message have correct tensor values
        # Parameter
        t1, t2 = torch.randn([5, 3]), torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2)
        msg = msg1 + msg2
        assert_equal_within_error(msg.parameter, t1 + t2)

    def test_add_correct_particles_type_single_var(self):
        # Test that the resulting message have correct tensor values
        # Particles
        # Single relational and random variables
        b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        w1, w2 = torch.rand(5, 10), torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w2, log_densities=l1)
        msg = msg1 + msg2
        # The sample dimension across ratio should be constant
        assert_proportional_within_error(msg.weight, w1 * w2, dims=[-1])

    def test_add_correct_particles_type_multiple_var(self):
        # Test that the resulting message have correct tensor values
        # Particles
        # Multiple relational and random variables
        b_s, s_s, e_s = Size([5, 6, 7]), Size([10, 12, 14]), Size([3, 2, 1])
        p1 = [torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])]
        w1, w2 = torch.rand(5, 6, 7, 10, 12, 14) + EPS, torch.rand(5, 6, 7, 10, 12, 14) + EPS
        l1 = [-torch.rand(10), -torch.rand(12), -torch.rand(14)]
        msg1 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w2, log_densities=l1)
        msg = msg1 + msg2
        # The sample dimension across ratio should be constant
        assert_proportional_within_error(msg.weight, w1 * w2, dims=[-1, -2, -3])

    def test_add_correct_dual_type(self):
        # Test that the resulting message have correct tensor values
        # Dual
        b_s, p_s, s_s, e_s = Size([5]), Size([4]), Size([10]), Size([3])
        param1, param2 = torch.randn([5, 4]), torch.randn([5, 4])
        part1 = [torch.randn(10, 3)]
        w1, w2 = torch.rand(5, 10), torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Dual, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                       parameter=param1, particles=part1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Dual, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                       parameter=param2, particles=part1, weight=w2, log_densities=l1)
        msg = msg1 + msg2
        assert_equal_within_error(msg.parameter, param1 + param2)
        assert_proportional_within_error(msg.weight, w1 * w2, dims=[-1])

    def test_add_with_identity_same_shape_parameter_type_1(self):
        # Test addition with identity and same shape
        t1 = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=0)
        msg = msg1 + msg2
        assert_equal_within_error(msg.parameter, t1)
        assert msg == msg1

    def test_add_with_identity_same_shape_parameter_type_2(self):
        # Test addition with identity and same shape
        t1 = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=0)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg = msg1 + msg2
        assert_equal_within_error(msg.parameter, t1)
        assert msg == msg2

    def test_add_with_identity_same_shape_particles_type(self):
        # Test addition with identity and same shape
        b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=1, log_densities=l1)
        msg = msg1 + msg2
        assert_proportional_within_error(msg.weight, w1, dims=[-1])
        assert msg == msg1

    def test_add_with_identity_diff_shape_auxiliary_attributes_parameter_type(self):
        # Test addition with identity
        # Different shapes and auxiliary attributes
        t1 = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([10]), param_shape=Size([9]), parameter=0)
        msg = msg1 + msg2
        assert_equal_within_error(msg.parameter, t1)
        assert msg == msg1

    def test_add_with_identity_diff_shape_auxiliary_attributes_particles_type(self):
        # Test addition with identity
        # Different shapes and auxiliary attributes
        b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6]), sample_shape=Size([10, 11]),
                       event_shape=Size([3, 4]),
                       particles=[torch.randn(10, 3), torch.randn(11, 4)], weight=1,
                       log_densities=[-torch.rand(10), -torch.rand(11)])
        msg = msg1 + msg2
        assert_proportional_within_error(msg.weight, w1, dims=[-1])
        assert msg == msg1

    def test_add_builtin_sum(self):
        # Test using builtin sym() on a list of messages
        # Both message
        b_s, p_s, s_s, e_s = Size([5]), Size([4]), Size([10]), Size([3])
        ptcl, dens = [torch.randn(10, 3)], [-torch.rand(10)]
        msg_list = [Message(MessageType.Dual, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                            parameter=torch.randn([5, 4]), particles=ptcl, weight=torch.rand(5, 10), log_densities=dens)
                    for i in range(3)]

        # When using builtin sum(), need to specify start value, otherwise sum() defaults to using 0 which would cause
        #   Type errors
        result = sum(msg_list, Message.identity())
        expected_val = msg_list[0] + msg_list[1] + msg_list[2]
        assert result == expected_val

    def test_iadd_parameter_type(self):
        t1, t2 = torch.randn([5, 3]), torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2)
        msg1 += msg2
        assert_equal_within_error(msg1.parameter, t1 + t2)

    def test_iadd_particles_type(self):
        b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        w1, w2 = torch.rand(5, 10), torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w2, log_densities=l1)
        msg1 += msg2
        assert_proportional_within_error(msg1.weight, w1 * w2, dims=[-1])

    def test_mul_invalid_multiplier_other_message(self):
        # Test multiplying message with another message
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        with pytest.raises(AssertionError):
            msg1 * msg2

    def test_mul_invalid_multiplier_other_type(self):
        # Test multiplying with other data type
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        with pytest.raises(AssertionError):
            msg1 * "some random stuff"
        with pytest.raises(AssertionError):
            msg1 * None

    def test_mul_invalid_multiplier_tensor_wrong_shape(self):
        # Test multiplying with tensor of wrong shape
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        with pytest.raises(AssertionError):
            msg1 * torch.randn(5, 3)

    def test_mul_valid_multiplier_int(self):
        # Test that multiplying with int runs without exception
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand(5, 6, 7, 10, 12, 14),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        msg1 * 2
        msg2 * 2

    def test_mul_valid_multiplier_float(self):
        # Test that multiplying with float runs without exception
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand(5, 6, 7, 10, 12, 14),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        msg1 * 1.5
        msg2 * 1.5

    def test_mul_valid_multiplier_scalar_tensor(self):
        # Test that multiplying with scalar tensor runs without exception
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand(5, 6, 7, 10, 12, 14),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        msg1 * torch.tensor(1.5)
        msg2 * torch.tensor(1.5)
        msg1 * torch.tensor([1.5])
        msg2 * torch.tensor([1.5])

    def test_mul_valid_multiplier_batch_tensor(self):
        # Test that multiplying with tensor with compatible batch shape runs without exception
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand(5, 6, 7, 10, 12, 14),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        msg1 * torch.randn([5])
        msg2 * torch.randn([5, 6, 7])

    def test_mul_correct_values_parameter_type_scalar(self):
        # Test correct return value of multiplying parameter type message with scalar value
        param = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=param)
        result1 = msg1 * 2
        result2 = msg1 * 2.0
        result3 = msg1 * torch.tensor(2.0)
        result4 = msg1 * torch.tensor([2.0])
        assert_equal_within_error(result1.parameter, param * 2)
        assert result1 == result2 == result3 == result4

    def test_mul_correct_values_parameter_type_batch_tensor(self):
        # Test correct return value of multiplying parameter type message with tensor with compatible batch shape
        param = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=param)
        scalar = torch.randn([5])
        result = msg1 * scalar
        assert_equal_within_error(result.parameter, param * scalar.unsqueeze(1))

    def test_mul_correct_values_particles_type_single_var_scalar(self):
        # Test correct return value of multiplying particles type message with scalar value
        # Single relational and random variable
        weight = torch.rand(5, 10)
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=weight, log_densities=[-torch.rand(10)])
        result1 = msg1 * 2
        result2 = msg1 * 2.0
        result3 = msg1 * torch.tensor(2.0)
        result4 = msg1 * torch.tensor([2.0])
        expected_weight = weight ** 2
        assert_proportional_within_error(result1.weight, expected_weight, dims=[-1])
        assert result1 == result2 == result3 == result4

    def test_mul_correct_values_particles_type_single_var_batch_tensor(self):
        # Test correct return value of multiplying particles type message with tensor with compatible batch shape
        # Single relational and random variable
        weight = torch.rand(5, 10)
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=weight, log_densities=[-torch.rand(10)])
        scalar = torch.randn([5])
        scalar_expanded = scalar.unsqueeze(-1)
        result = msg1 * scalar
        expected_weight = weight ** scalar_expanded
        assert_proportional_within_error(result.weight, expected_weight, dims=[-1])

    def test_mul_correct_values_particles_type_multiple_var_scalar(self):
        # Test correct return value of multiplying particles type message with scalar value
        # Multiple relational and random variables
        weight = torch.rand(5, 6, 7, 10, 12, 14)
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=weight,
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        result1 = msg1 * 2
        result2 = msg1 * 2.0
        result3 = msg1 * torch.tensor(2.0)
        result4 = msg1 * torch.tensor([2.0])
        expected_weight = weight ** 2
        assert_proportional_within_error(result1.weight, expected_weight, dims=[-1, -2, -3])
        assert result1 == result2 == result3 == result4

    def test_mul_correct_values_particles_type_multiple_var_batch_tensor(self):
        # Test correct return value of multiplying particles type message with tensor with compatible batch shape
        # Multiple relational and random variables
        weight = torch.rand(5, 6, 7, 10, 12, 14)
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=weight,
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        scalar = torch.randn([5, 6, 7])
        scalar_expanded = scalar.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        result = msg1 * scalar
        expected_weight = weight ** scalar_expanded
        assert_proportional_within_error(result.weight, expected_weight, dims=[-1, -2, -3])

    def test_mul_correct_values_dual_type_scalar(self):
        # Test correct return value of multiplying dual type message with tensor with compatible batch shape
        b_s, p_s, s_s, e_s = Size([5]), Size([4]), Size([10]), Size([3])
        param = torch.randn([5, 4])
        part = [torch.randn(10, 3)]
        weight = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Dual, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                       parameter=param, particles=part, weight=weight, log_densities=l1)
        result1 = msg1 * 2
        # Check parameter
        assert_equal_within_error(result1.parameter, param * 2)
        # Check particles component
        assert_proportional_within_error(result1.weight, weight ** 2, dims=[-1])

    def test_mul_identity_parameter_type(self):
        # Test multiplying with identity message
        # Parameter type message
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=0)
        result1 = msg1 * 2.0
        assert msg1 == result1

    def test_mul_identity_particles_type(self):
        # Test multiplying with identity message
        # Parameter type message
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=1,
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        result2 = msg2 * 5.0
        assert msg2 == result2

    def test_mul_identity_dual_type(self):
        msg3 = Message(MessageType.Dual,
                       batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                       parameter=0, particles=[torch.randn(10, 3)], weight=1,
                       log_densities=[-torch.rand(10)])
        result3 = msg3 * 10.0
        assert msg3 == result3


class TestMessageStaticMethods:
    # Test message class static methods

    def test_compose_invalid_arguments_same_message_type_1(self):
        # Test composing message with the same message type
        # Test 1: both are parameter message
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        with pytest.raises(AssertionError):
            msg = Message.compose(msg1, msg2)

    def test_compose_invalid_arguments_same_message_type_2(self):
        # Test composing message with the same message type
        # Test 2: both are particles message
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        with pytest.raises(AssertionError):
            Message.compose(msg1, msg2)

    def test_compose_invalid_arguments_same_message_type_3(self):
        # Test composing message with the same message type
        # Test 3: both are dual type message
        msg1 = Message(MessageType.Dual,
                       batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                       parameter=torch.randn([5, 4]), particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]),
                       log_densities=[-torch.rand(10)])
        msg2 = Message(MessageType.Dual,
                       batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                       parameter=torch.randn([5, 4]), particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]),
                       log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            Message.compose(msg1, msg2)

    def test_compose_invalid_arguments_incompatible_message_type_1(self):
        # Test composing message with different message types but these types are incompatible
        # Test 1: Parameter with Dual
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Dual,
                       batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                       parameter=torch.randn([5, 4]), particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]),
                       log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            Message.compose(msg1, msg2)

    def test_compose_invalid_arguments_incompatible_message_type_2(self):
        # Test composing message with different message types but these types are incompatible
        # Test 2: Particles with Dual
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        msg2 = Message(MessageType.Dual,
                       batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                       parameter=torch.randn([5, 4]), particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]),
                       log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            Message.compose(msg1, msg2)

    def test_compose_invalid_arguments_wrong_shape(self):
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([6]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(6, 10), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            Message.compose(msg1, msg2)

    def test_compose_invalid_arguments_identity_message_1(self):
        # Test compose with one identity message
        # Test 1: with an identity Parameter type message
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=0)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg = Message.compose(msg1, msg2)

    def test_compose_invalid_arguments_identity_message_2(self):
        # Test compose with one identity message
        # Test 1: with an identity Particles type message
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=1, log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg = Message.compose(msg1, msg2)

    def test_compose_correct_values(self):
        # Test correct return values
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        msg = Message.compose(msg1, msg2)
        assert msg.type is MessageType.Dual
        assert_equal_within_error(msg.parameter, msg1.parameter)
        assert_equal_within_error(msg.weight, msg2.weight)

    def test_compose_correct_values_attr_dict_overwrite(self):
        # Test that the auxiliary attribute dictionary is overwritten in the correct way
        attr1 = {"a": 1, "b": 2, "c": 3}
        attr2 = {"a": 4, "d": 5}
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]),
                       **attr1)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)],
                       **attr2)
        msg = Message.compose(msg1, msg2)
        assert msg.attr["a"] == 4 and msg.attr["b"] == 2 and msg.attr["c"] == 3 and msg.attr["d"] == 5

    def test_identity_parameter_type(self):
        # Test parameter type
        id_msg = Message.identity(MessageType.Parameter)
        assert id_msg.type is MessageType.Parameter
        assert id_msg.isid
        assert id_msg.b_shape == Size([]) and id_msg.p_shape == Size([]) and id_msg.s_shape == Size([]) and \
               id_msg.e_shape == Size([])
        assert id_msg.attr == {}

    def test_identity_particles_type(self):
        # Test particles type
        id_msg = Message.identity(MessageType.Particles)
        assert id_msg.type is MessageType.Particles
        assert id_msg.isid
        assert id_msg.b_shape == Size([]) and id_msg.p_shape == Size([]) and id_msg.s_shape == Size([]) and \
               id_msg.e_shape == Size([])
        assert id_msg.attr == {}

    def test_identity_dual_type(self):
        # Test dual type
        id_msg = Message.identity(MessageType.Dual)
        assert id_msg.type is MessageType.Dual
        assert id_msg.isid
        assert id_msg.b_shape == Size([]) and id_msg.p_shape == Size([]) and id_msg.s_shape == Size([]) and \
               id_msg.e_shape == Size([])
        assert id_msg.attr == {}


class TestMessageGeneralMethods:
    # Test message class general methods

    def test_size_parameter_type(self):
        msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                      parameter=torch.randn([5, 3]))
        assert msg.size() == (Size([5]), Size([3]), Size([]), Size([]))

    def test_size_particles_type(self):
        msg = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                      event_shape=Size([3, 2, 1]),
                      particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                      weight=torch.rand([5, 6, 7, 10, 12, 14]),
                      log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        assert msg.size() == (Size([5, 6, 7]), Size([]), Size([10, 12, 14]), Size([3, 2, 1]))

    def test_size_dual_type(self):
        msg = Message(MessageType.Dual,
                      batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                      parameter=torch.randn([5, 4]), particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]),
                      log_densities=[-torch.rand(10)])
        assert msg.size() == (Size([5]), Size([4]), Size([10]), Size([3]))

    def test_same_particle_as_same_particles(self):
        part1 = [torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])]
        part2 = list(p.clone() for p in part1)
        log1 = [-torch.rand(10), -torch.rand(12), -torch.rand(14)]
        log2 = list(l.clone() for l in log1)
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=part1,
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=log1)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=part2,
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=log2)
        assert msg1.same_particles_as(msg2)
        assert msg2.same_particles_as(msg1)

    def test_same_particle_as_different_particles_same_densities(self):
        part1 = [torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])]
        part2 = [torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])]
        log1 = [-torch.rand(10), -torch.rand(12), -torch.rand(14)]
        log2 = list(l.clone() for l in log1)
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=part1,
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=log1)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=part2,
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=log2)
        assert not msg1.same_particles_as(msg2)
        assert not msg2.same_particles_as(msg1)

    def test_same_particle_as_same_particles_different_densities(self):
        part1 = [torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])]
        part2 = list(p.clone() for p in part1)
        log1 = [-torch.rand(10), -torch.rand(12), -torch.rand(14)]
        log2 = [-torch.rand(10), -torch.rand(12), -torch.rand(14)]
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=part1,
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=log1)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=part2,
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=log2)
        assert not msg1.same_particles_as(msg2)
        assert not msg2.same_particles_as(msg1)

    def test_same_particle_as_not_particles_type(self):
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        assert not msg1.same_particles_as(msg2)
        assert not msg2.same_particles_as(msg1)

    def test_same_particle_as_identity(self):
        # NOTE: semantic change: now a non-identity message will NOT have same particles as an identity message.
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        msg2 = Message.identity(MessageType.Particles)
        assert not msg1.same_particles_as(msg2)
        assert not msg2.same_particles_as(msg1)

    @cuda_only
    def test_diff_param_different_devices(self):
        # Only run this test if GPU is available
        device = torch.cuda.current_device()
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]), device=device)
        with pytest.raises(AssertionError):
            msg1.diff_param(msg2)

    def test_diff_param_wrong_type(self):
        # Test with types other than parameter
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg1.diff_param(msg2)

    def test_diff_param_diff_batch_shape(self):
        # Different batch shape
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([6]), param_shape=Size([3]),
                       parameter=torch.randn([6, 3]))
        with pytest.raises(AssertionError):
            msg1.diff_param(msg2)

    def test_diff_param_diff_param_shape(self):
        # Different batch shape
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([4]),
                       parameter=torch.randn([5, 4]))
        with pytest.raises(AssertionError):
            msg1.diff_param(msg2)

    def test_diff_param_ad_hoc_values_1(self):
        # Test 1: ad-hoc parameter values
        msg1 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([2]),
                       parameter=torch.tensor([[3., 4.]]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([2]),
                       parameter=torch.tensor([[0., 0.]]))
        assert msg1.diff_param(msg2) == msg2.diff_param(msg1) == 5

    def test_diff_param_ad_hoc_values_2(self):
        # Test 1: ad-hoc parameter values
        msg1 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([2]),
                       parameter=torch.tensor([[4., 6.]]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([2]),
                       parameter=torch.tensor([[1., 2.]]))
        assert msg1.diff_param(msg2) == msg2.diff_param(msg1) == 5

    def test_diff_param_ad_hoc_values_3(self):
        # Test 1: ad-hoc parameter values
        msg1 = Message(MessageType.Parameter, batch_shape=Size([2]), param_shape=Size([2]),
                       parameter=torch.tensor([[4., 6.], [6., 8.]]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([2]), param_shape=Size([2]),
                       parameter=torch.tensor([[1., 2.], [0., 0.]]))
        assert msg1.diff_param(msg2) == msg2.diff_param(msg1) == 7.5

    def test_diff_param_random_values_single_batch(self):
        # Test 1: random single batch
        param1, param2 = torch.rand([1, 10]), torch.rand([1, 10])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([10]),
                       parameter=param1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([10]),
                       parameter=param2)
        diff_param = param1 - param2
        assert torch.equal(msg1.diff_param(msg2), msg2.diff_param(msg1))
        assert torch.equal(msg1.diff_param(msg2), diff_param.norm())

    def test_diff_param_random_values_multiple_batch(self):
        # Test 2: random multiple batch
        param1, param2 = torch.rand([5, 6, 10]), torch.rand([5, 6, 10])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5, 6]), param_shape=Size([10]),
                       parameter=param1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5, 6]), param_shape=Size([10]),
                       parameter=param2)
        diff_param = param1 - param2
        diff_param_norm = diff_param.norm(dim=-1)
        diff_param_norm_mean = diff_param_norm.mean()
        assert torch.equal(msg1.diff_param(msg2), msg2.diff_param(msg1))
        assert torch.equal(msg1.diff_param(msg2), diff_param_norm_mean)

    def test_diff_param_one_identity(self):
        param1 = torch.rand([5, 6, 10])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5, 6]), param_shape=Size([10]),
                       parameter=param1)
        msg2 = Message.identity(MessageType.Parameter)
        diff_norm = param1.norm(dim=-1)
        diff_norm_mean = diff_norm.mean()
        assert torch.equal(msg1.diff_param(msg2), msg2.diff_param(msg1))
        assert torch.equal(msg1.diff_param(msg2), diff_norm_mean)

    def test_diff_param_both_identities(self):
        msg1 = Message.identity(MessageType.Parameter)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=0)
        assert msg1.diff_param(msg2) == 0
        assert msg2.diff_param(msg1) == 0

    @cuda_only
    def test_diff_weight_different_device(self):
        # Only run this test if GPU is available

        device = torch.cuda.current_device()
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)],
                       device=device)
        with pytest.raises(AssertionError):
            msg1.diff_weight(msg2)

    def test_diff_weight_wrong_type(self):
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=0)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([3, 4]), event_shape=Size([6, 7]),
                       particles=[torch.randn([3, 6]), torch.randn([4, 7])],
                       weight=torch.rand([5, 3, 4]),
                       log_densities=[-torch.rand(3), -torch.rand([4])])
        with pytest.raises(AssertionError):
            msg1.diff_weight(msg2)

    def test_diff_weight_diff_shape(self):
        # Different batch shape
        ptcl = [torch.randn([3, 6]), torch.randn([4, 7])]
        dsty = [-torch.rand(3), -torch.rand([4])]
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([3, 4]),
                       event_shape=Size([6, 7]),
                       particles=ptcl,
                       weight=torch.rand([5, 3, 4]),
                       log_densities=dsty)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([3, 4]),
                       event_shape=Size([6, 7]),
                       particles=ptcl,
                       weight=torch.rand([5, 6, 7, 3, 4]),
                       log_densities=dsty)
        with pytest.raises(AssertionError):
            msg1.diff_weight(msg2)

    def test_diff_weight_diff_particles(self):
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([3, 4]),
                       event_shape=Size([6, 7]),
                       particles=[torch.randn([3, 6]), torch.randn([4, 7])],
                       weight=torch.rand([5, 3, 4]),
                       log_densities=[-torch.rand(3), -torch.rand([4])])
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([3, 4]),
                       event_shape=Size([6, 7]),
                       particles=[torch.randn([3, 6]), torch.randn([4, 7])],
                       weight=torch.rand([5, 3, 4]),
                       log_densities=[-torch.rand(3), -torch.rand([4])])
        with pytest.raises(AssertionError):
            msg1.diff_weight(msg2)

    def test_diff_weight_random_weight_single_batch(self):
        # Single batch
        ptcl = [torch.randn([2, 6]), torch.randn([5, 7])]
        dsty = [-torch.rand(2), -torch.rand([5])]
        w1, w2 = torch.rand([1, 2, 5]), torch.rand([1, 2, 5])
        msg1 = Message(MessageType.Particles, batch_shape=Size([1]), sample_shape=Size([2, 5]),
                       event_shape=Size([6, 7]),
                       particles=ptcl,
                       weight=w1,
                       log_densities=dsty)
        msg2 = Message(MessageType.Particles, batch_shape=Size([1]), sample_shape=Size([2, 5]),
                       event_shape=Size([6, 7]),
                       particles=ptcl,
                       weight=w2,
                       log_densities=dsty)
        assert msg1.diff_weight(msg2) == msg2.diff_weight(msg1)
        assert msg1.diff_weight(msg2) == torch.nn.functional.l1_loss(msg1.weight, msg2.weight, reduction='mean')

    def test_diff_weight_random_weight_multiple_batch(self):
        # Multiple batches
        ptcl = [torch.randn([2, 6]), torch.randn([5, 7])]
        dsty = [-torch.rand(2), -torch.rand([5])]
        w1, w2 = torch.rand([10, 2, 5]), torch.rand([10, 2, 5])
        msg1 = Message(MessageType.Particles, batch_shape=Size([10]), sample_shape=Size([2, 5]),
                       event_shape=Size([6, 7]),
                       particles=ptcl,
                       weight=w1,
                       log_densities=dsty)
        msg2 = Message(MessageType.Particles, batch_shape=Size([10]), sample_shape=Size([2, 5]),
                       event_shape=Size([6, 7]),
                       particles=ptcl,
                       weight=w2,
                       log_densities=dsty)
        assert msg1.diff_weight(msg2) == msg2.diff_weight(msg1)
        assert msg1.diff_weight(msg2) == torch.nn.functional.l1_loss(msg1.weight, msg2.weight, reduction='mean')

    def test_diff_weight_one_identity(self):
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([3, 4]),
                       event_shape=Size([6, 7]),
                       particles=[torch.randn([3, 6]), torch.randn([4, 7])],
                       weight=torch.rand([5, 3, 4]),
                       log_densities=[-torch.rand(3), -torch.rand([4])])
        msg2 = Message.identity(MessageType.Particles)
        uniform_weight = torch.ones([5, 3, 4]) / 60
        assert msg1.diff_weight(msg2) == torch.nn.functional.l1_loss(msg1.weight, uniform_weight, reduction='mean')

    def test_diff_weight_both_identities(self):
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([3, 4]),
                       event_shape=Size([6, 7]),
                       particles=[torch.randn([3, 6]), torch.randn([4, 7])],
                       weight=1,
                       log_densities=[-torch.rand(3), -torch.rand([4])])
        msg2 = Message.identity(MessageType.Particles)
        assert msg1.diff_weight(msg2) == 0

    def test_reduce_type_wrong_target_type_1(self):
        # Test reduce from dual to dual
        target_type = MessageType.Dual
        msg = Message(MessageType.Dual,
                      batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                      parameter=torch.randn([5, 4]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg.reduce_type(target_type)

    def test_reduce_type_wrong_target_type_2(self):
        # Test reduce from parameter to dual
        target_type = MessageType.Dual
        msg = Message(MessageType.Parameter,
                      batch_shape=Size([5]), param_shape=Size([4]),
                      parameter=torch.randn([5, 4]))
        with pytest.raises(AssertionError):
            msg.reduce_type(target_type)

    def test_reduce_type_wrong_target_type_3(self):
        # Test reduce from particles to dual
        target_type = MessageType.Dual
        msg = Message(MessageType.Particles,
                      batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg.reduce_type(target_type)

    def test_reduce_type_wrong_target_type_4(self):
        # Test reduce from particles to parameter
        target_type = MessageType.Parameter
        msg = Message(MessageType.Particles,
                      batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg.reduce_type(target_type)

    def test_reduce_type_wrong_target_type_5(self):
        # Test reduce from parameter to particles
        target_type = MessageType.Particles
        msg = Message(MessageType.Parameter,
                      batch_shape=Size([5]), param_shape=Size([4]),
                      parameter=torch.randn([5, 4]))
        with pytest.raises(AssertionError):
            msg.reduce_type(target_type)

    def test_reduce_type_correct_contents_1(self):
        test_attr = {"a": 1, "b": 2, "c": 3}
        # Test particles reduce to particles
        msg = Message(MessageType.Particles,
                      batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)],
                      **test_attr)
        reduced_msg = msg.reduce_type(MessageType.Particles)
        assert msg == reduced_msg

    def test_reduce_type_correct_contents_2(self):
        test_attr = {"a": 1, "b": 2, "c": 3}
        # Test Dual reduce to particles
        msg = Message(MessageType.Dual,
                      batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                      parameter=torch.randn([5, 4]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)],
                      **test_attr)
        reduced_msg = msg.reduce_type(MessageType.Particles)
        assert msg.b_shape == reduced_msg.b_shape and msg.e_shape == reduced_msg.e_shape and \
               msg.s_shape == reduced_msg.s_shape
        assert reduced_msg.p_shape == Size([])
        assert_equal_within_error(msg.weight, reduced_msg.weight)
        for p, rp in zip(msg.particles, reduced_msg.particles):
            assert_equal_within_error(p, rp)
        for d, rd in zip(msg.log_densities, reduced_msg.log_densities):
            assert_equal_within_error(d, rd)
        assert reduced_msg.parameter == 0

    def test_reduce_type_correct_contents_3(self):
        test_attr = {"a": 1, "b": 2, "c": 3}
        # Test parameter reduce to parameter
        msg = Message(MessageType.Parameter,
                      batch_shape=Size([5]), param_shape=Size([4]), parameter=torch.randn([5, 4]),
                      **test_attr)
        reduced_msg = msg.reduce_type(MessageType.Parameter)
        assert msg == reduced_msg

    def test_reduce_type_correct_contents_4(self):
        test_attr = {"a": 1, "b": 2, "c": 3}
        # Test dual reduce to parameter
        msg = Message(MessageType.Dual,
                      batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                      parameter=torch.randn([5, 4]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)],
                      **test_attr)
        reduced_msg = msg.reduce_type(MessageType.Parameter)
        assert msg.b_shape == reduced_msg.b_shape and msg.p_shape == reduced_msg.p_shape
        assert reduced_msg.s_shape == Size([]) and reduced_msg.e_shape == Size([])
        assert_equal_within_error(reduced_msg.parameter, msg.parameter)
        assert reduced_msg.weight == 1

    def test_clone_parameter_type(self):
        test_attr = {"a": 1, "b": 2, "c": 3}
        # Test parameter messages
        msg = Message(MessageType.Parameter,
                      batch_shape=Size([5]), param_shape=Size([4]), parameter=torch.randn([5, 4]),
                      **test_attr)
        cloned_msg = msg.clone()
        assert msg == cloned_msg
        assert msg.parameter is not cloned_msg.parameter

    def test_clone_particles_type(self):
        test_attr = {"a": 1, "b": 2, "c": 3}
        # Test particles message
        msg = Message(MessageType.Particles,
                      batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)],
                      **test_attr)
        cloned_msg = msg.clone()
        assert msg == cloned_msg
        for p, rp in zip(msg.particles, msg.particles):
            assert_equal_within_error(p, rp)
        for d, rd in zip(msg.log_densities, msg.log_densities):
            assert_equal_within_error(d, rd)
        assert msg.weight is not cloned_msg.weight

    @cuda_only
    def test_clone_with_gpu_device(self):
        # Carry out this test only if system has cuda
        device = torch.cuda.current_device()

        # Test dual type
        test_attr = {"a": 1, "b": 2, "c": 3}
        msg = Message(MessageType.Dual,
                      batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                      parameter=torch.randn([5, 4]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)],
                      **test_attr)
        cuda_msg = msg.to_device(device)
        back_msg = cuda_msg.to_device('cpu')
        assert cuda_msg.parameter.get_device() == device and cuda_msg.weight.get_device() == device
        assert all(p.get_device() == device for p in cuda_msg.particles)
        assert all(d.get_device() == device for d in cuda_msg.log_densities)
        assert back_msg == msg


class TestMessageBatchMethods:
    # Test message batch-manipulation methods

    def test_batch_permute_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: Positive dim
        perm_order = [2, 0, 1]
        result = msg.batch_permute(perm_order)

        # Check shape
        assert result.parameter.shape == Size([3, 1, 2, 1])
        assert result.weight.shape == Size([3, 1, 2, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.permute([2, 0, 1, 3]))
        assert_equal_within_error(result.weight, msg.weight.permute([2, 0, 1, 3, 4, 5]))

    def test_batch_permute_negative_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: Positive dim
        perm_order = [-1, -3, -2]
        result = msg.batch_permute(perm_order)

        # Check shape
        assert result.parameter.shape == Size([3, 1, 2, 1])
        assert result.weight.shape == Size([3, 1, 2, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.permute([2, 0, 1, 3]))
        assert_equal_within_error(result.weight, msg.weight.permute([2, 0, 1, 3, 4, 5]))

    def test_batch_unsqueeze_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = 2
        result = msg.batch_unsqueeze(dim)

        # Check shape
        assert result.parameter.shape == Size([1, 2, 1, 3, 1])
        assert result.weight.shape == Size([1, 2, 1, 3, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.unsqueeze(2))
        assert_equal_within_error(result.weight, msg.weight.unsqueeze(2))

    def test_batch_unsqueeze_negative_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = -2
        result = msg.batch_unsqueeze(dim)

        # Check shape
        assert result.parameter.shape == Size([1, 2, 1, 3, 1])
        assert result.weight.shape == Size([1, 2, 1, 3, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.unsqueeze(2))
        assert_equal_within_error(result.weight, msg.weight.unsqueeze(2))

    def test_batch_index_select_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([11, 22, 33]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = 2
        index = torch.tensor([5, 10, 15, 20, 25, 30], dtype=torch.long)
        result = msg.batch_index_select(dim, index)

        # Check shape
        assert result.parameter.shape == Size([11, 22, 6, 1])
        assert result.weight.shape == Size([11, 22, 6, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.index_select(dim, index))
        assert_equal_within_error(result.weight, msg.weight.index_select(dim, index))

    def test_batch_index_select_negative_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([11, 22, 33]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = -1
        index = torch.tensor([5, 10, 15, 20, 25, 30], dtype=torch.long)
        result = msg.batch_index_select(dim, index)

        # Check shape
        assert result.parameter.shape == Size([11, 22, 6, 1])
        assert result.weight.shape == Size([11, 22, 6, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.index_select(2, index))
        assert_equal_within_error(result.weight, msg.weight.index_select(2, index))

    def test_batch_index_put_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = 0
        index = torch.tensor([5, 4, 3], dtype=torch.long)
        result = msg.batch_index_put(dim, index)

        # Check shape
        assert result.parameter.shape == Size([6, 4, 5, 1])
        assert result.weight.shape == Size([6, 4, 5, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter[:3], torch.zeros(3, 4, 5, 1))
        assert_equal_within_error(result.weight[:3], torch.ones(3, 4, 5, 4, 5, 6) / (4 * 5 * 6))
        for i, ri in enumerate(index):
            assert_equal_within_error(result.parameter[ri], msg.parameter[i])
        for i, ri in enumerate(index):
            assert_equal_within_error(result.weight[ri], msg.weight[i])

    def test_batch_index_put_negative_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = -3
        index = torch.tensor([5, 4, 3], dtype=torch.long)
        result = msg.batch_index_put(dim, index)

        # Check shape
        assert result.parameter.shape == Size([6, 4, 5, 1])
        assert result.weight.shape == Size([6, 4, 5, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter[:3], torch.zeros(3, 4, 5, 1))
        assert_equal_within_error(result.weight[:3], torch.ones(3, 4, 5, 4, 5, 6) / (4 * 5 * 6))
        for i, ri in enumerate(index):
            assert_equal_within_error(result.parameter[ri], msg.parameter[i])
        for i, ri in enumerate(index):
            assert_equal_within_error(result.weight[ri], msg.weight[i])

    def test_batch_diagonal_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim1, dim2 = 0, 2
        result = msg.batch_diagonal(dim1, dim2)

        # Check shape
        assert result.parameter.shape == Size([4, 3, 1])
        assert result.weight.shape == Size([4, 3, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.diagonal(0, 0, 2).transpose(-1, -2))
        assert_equal_within_error(result.weight, msg.weight.diagonal(0, 0, 2).permute(0, 4, 1, 2, 3))

    def test_batch_diagonal_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim1, dim2 = -3, -1
        result = msg.batch_diagonal(dim1, dim2)

        # Check shape
        assert result.parameter.shape == Size([4, 3, 1])
        assert result.weight.shape == Size([4, 3, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.diagonal(0, 0, 2).transpose(-1, -2))
        assert_equal_within_error(result.weight, msg.weight.diagonal(0, 0, 2).permute(0, 4, 1, 2, 3))

    def test_batch_diag_embed_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        diag_dim, target_dim1, target_dim2 = 1, 1, 3
        result = msg.batch_diag_embed(diag_dim, target_dim1, target_dim2)

        # Check shape
        assert result.parameter.shape == Size([3, 4, 5, 4, 1])
        assert result.weight.shape == Size([3, 4, 5, 4, 4, 5, 6])

        # Check contents
        assert_equal_within_error(result.parameter.diagonal(0, 1, 3).permute([0, 3, 1, 2]), msg.parameter)
        assert_equal_within_error(result.weight.diagonal(0, 1, 3).permute([0, 5, 1, 2, 3, 4]), msg.weight)

    def test_batch_diag_embed_negative_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        diag_dim, target_dim1, target_dim2 = -2, -4, -2
        result = msg.batch_diag_embed(diag_dim, target_dim1, target_dim2)

        # Check shape
        assert result.parameter.shape == Size([4, 3, 4, 5, 1])
        assert result.weight.shape == Size([4, 3, 4, 5, 4, 5, 6])

        # Check contents
        assert_equal_within_error(result.parameter.diagonal(0, 0, 2).permute([0, 3, 1, 2]), msg.parameter)
        assert_equal_within_error(result.weight.diagonal(0, 0, 2).permute([0, 5, 1, 2, 3, 4]), msg.weight)

    def test_batch_narrow_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim, length = 2, 3
        result = msg.batch_narrow(dim, length)

        # Check shape
        assert result.parameter.shape == Size([3, 4, 3, 1])
        assert result.weight.shape == Size([3, 4, 3, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter[:, :, :3, :])
        assert_equal_within_error(result.weight, msg.weight[:, :, :3, :, :, :])

    def test_batch_narrow_negative_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim, length = -1, 3
        result = msg.batch_narrow(dim, length)

        # Check shape
        assert result.parameter.shape == Size([3, 4, 3, 1])
        assert result.weight.shape == Size([3, 4, 3, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter[:, :, :3, :])
        assert_equal_within_error(result.weight, msg.weight[:, :, :3, :, :, :])

    def test_batch_broaden_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim, length = 0, 6
        result = msg.batch_broaden(dim, length)

        # Check shape
        assert result.parameter.shape == Size([6, 4, 5, 1])
        assert result.weight.shape == Size([6, 4, 5, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter[:3], msg.parameter)
        assert_equal_within_error(result.parameter[3:], torch.zeros([3, 4, 5, 1]))
        assert_equal_within_error(result.weight[:3], msg.weight)
        assert_equal_within_error(result.weight[3:], torch.ones([3, 4, 5, 4, 5, 6]) / (4 * 5 * 6))

    def test_batch_broaden_negative_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim, length = -3, 6
        result = msg.batch_broaden(dim, length)

        # Check shape
        assert result.parameter.shape == Size([6, 4, 5, 1])
        assert result.weight.shape == Size([6, 4, 5, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter[:3], msg.parameter)
        assert_equal_within_error(result.parameter[3:], torch.zeros([3, 4, 5, 1]))
        assert_equal_within_error(result.weight[:3], msg.weight)
        assert_equal_within_error(result.weight[3:], torch.ones([3, 4, 5, 4, 5, 6]) / (4 * 5 * 6))

    def test_batch_summarize_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = 0
        result = msg.batch_summarize(dim)

        # Check shape
        assert result.parameter.shape == Size([4, 5, 1])
        assert result.weight.shape == Size([4, 5, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.mean(dim, keepdim=False))
        weight_product = msg.weight[0]
        for i in range(1, b_shape[dim]):
            weight_product *= msg.weight[i]
        weight_product /= weight_product.sum(dim=[-1, -2, -3], keepdim=True)
        assert_equal_within_error(result.weight, weight_product)

    def test_batch_summarize_negative_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 2: negative dim
        dim = 0
        neg_dim = -3
        result = msg.batch_summarize(neg_dim)

        # Check shape
        assert result.parameter.shape == Size([4, 5, 1])
        assert result.weight.shape == Size([4, 5, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.mean(dim, keepdim=False))
        weight_product = msg.weight[0]
        for i in range(1, b_shape[dim]):
            weight_product *= msg.weight[i]
        weight_product /= weight_product.sum(dim=[-1, -2, -3], keepdim=True)
        assert_equal_within_error(result.weight, weight_product)

    def test_batch_flatten_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dims = [0, 1]
        result = msg.batch_flatten(dims)

        # Check shape
        assert result.parameter.shape == Size([5, 12, 1])
        assert result.weight.shape == Size([5, 12, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.view(-1, 5, 1).permute(1, 0, 2))
        assert_equal_within_error(result.weight, msg.weight.view(-1, 5, 4, 5, 6).permute(1, 0, 2, 3, 4))

    def test_batch_flatten_negative_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dims = [-3, -2]
        result = msg.batch_flatten(dims)

        # Check shape
        assert result.parameter.shape == Size([5, 12, 1])
        assert result.weight.shape == Size([5, 12, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.view(-1, 5, 1).permute(1, 0, 2))
        assert_equal_within_error(result.weight, msg.weight.view(-1, 5, 4, 5, 6).permute(1, 0, 2, 3, 4))

    def test_batch_flatten_default(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        result = msg.batch_flatten()

        # Check shape
        assert result.parameter.shape == Size([60, 1])
        assert result.weight.shape == Size([60, 4, 5, 6])

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.view(-1, 1))
        assert_equal_within_error(result.weight, msg.weight.view(-1, 4, 5, 6))

    def test_batch_reshape(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        new_batch_shape = Size([2, 3, 10])
        result = msg.batch_reshape(new_batch_shape)

        # Check shape
        assert result.parameter.shape == new_batch_shape + p_shape
        assert result.weight.shape == new_batch_shape + s_shape

        # Check content
        assert_equal_within_error(result.parameter, msg.parameter.reshape(new_batch_shape + p_shape))
        assert_equal_within_error(result.weight, msg.weight.reshape(new_batch_shape + s_shape))

    def test_batch_expand(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 4, 1]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        new_batch_shape = Size([3, -1, 5])
        result = msg.batch_expand(new_batch_shape)

        # Check shape
        assert result.parameter.shape == Size([3, 4, 5, 1])
        assert result.weight.shape == Size([3, 4, 5, 4, 5, 6])

        # Check content
        for i in range(3):
            assert_equal_within_error(result.parameter[i:i + 1, :, :1, :], msg.parameter)
        for i in range(3):
            assert_equal_within_error(result.parameter[:1, :, i:i + 1, :], msg.parameter)
        for i in range(3):
            assert_equal_within_error(result.weight[i:i + 1, :, :1, :, :, :], msg.weight)
        for i in range(3):
            assert_equal_within_error(result.weight[:1, :, i:i + 1, :, :, :], msg.weight)


class TestMessageEventMethods:
    # Test Message class event-related methods

    def test_event_transform_event_dims_0(self):
        # Test transformation with 0 event dimension
        b_shape, p_shape, s_shape, e_shape = Size([3]), Size([]), Size([4, 5, 6]), Size([1, 2, 3])

        # Create three distributions of corresponding event shape and sample three list of particles
        dist = [D.MultivariateNormal(torch.zeros(i), torch.eye(i)) for i in range(1, 4)]
        ptcl = [d.sample([j]) for d, j in zip(dist, range(4, 7))]
        dens = [d.log_prob(p) for d, p in zip(dist, ptcl)]
        msg = Message(MessageType.Particles,
                      b_shape, p_shape, s_shape, e_shape,
                      particles=ptcl, log_densities=dens, weight=torch.rand([3, 4, 5, 6]))

        # Forward transform
        trans = T.ExpTransform()
        result = msg.event_transform(trans)

        # Check content
        # Check particle values
        for p, rp in zip(msg.particles, result.particles):
            assert_equal_within_error(rp, trans(p))
        # Check log densities
        transformed_dens = [D.TransformedDistribution(d, [trans]).log_prob(trans(p)) for d, p in zip(dist, ptcl)]
        for rd, td in zip(result.log_densities, transformed_dens):
            td = td - td.sum() / td.numel()     # Normalize log densities so that mean is 0
            assert_equal_within_error(rd, td)

    def test_event_transform_event_dims_1(self):
        # Test transformation with 1 event dimension
        b_shape, p_shape, s_shape, e_shape = Size([3]), Size([]), Size([4, 5, 6]), Size([1, 2, 3])

        # Create three distributions of corresponding event shape and sample three list of particles
        dist = [D.MultivariateNormal(torch.zeros(i), torch.eye(i)) for i in range(1, 4)]
        ptcl = [d.sample([j]) for d, j in zip(dist, range(4, 7))]
        dens = [d.log_prob(p) for d, p in zip(dist, ptcl)]
        msg = Message(MessageType.Particles,
                      b_shape, p_shape, s_shape, e_shape,
                      particles=ptcl, log_densities=dens, weight=torch.rand([3, 4, 5, 6]))

        # Forward transform
        trans = T.AffineTransform(1, 10, event_dim=1)
        result = msg.event_transform(trans)

        # Check content
        # Check particle values
        for p, rp in zip(msg.particles, result.particles):
            assert_equal_within_error(rp, trans(p))
        # Check log densities
        transformed_dens = [D.TransformedDistribution(d, [trans]).log_prob(trans(p)) for d, p in zip(dist, ptcl)]
        for rd, td in zip(result.log_densities, transformed_dens):
            td = td - td.sum() / td.numel()  # Normalize log densities so that mean is 0
            assert_equal_within_error(rd, td)

    def test_event_transform_log_normal(self):
        # Test transformation with 0 event dimension
        b_shape, p_shape, s_shape, e_shape = Size([3]), Size([]), Size([5]), Size([1])

        # Create three distributions of corresponding event shape and sample three list of particles
        loc, scale = torch.randn(1), torch.rand(1)
        dist, expected_dist = D.Normal(loc, scale), D.LogNormal(loc, scale)
        ptcl = [dist.sample(s_shape)]
        dens = [dist.log_prob(ptcl[0]).squeeze(-1)]
        msg = Message(MessageType.Particles,
                      b_shape, p_shape, s_shape, e_shape,
                      particles=ptcl, log_densities=dens, weight=torch.rand([3, 5]))

        # Forward transform
        trans = T.ExpTransform()
        result = msg.event_transform(trans)

        # Check content
        # Check particle values
        for p, rp in zip(msg.particles, result.particles):
            assert_equal_within_error(rp, trans(p))
        # Check resulting log densities against transformed densities
        transformed_dens = [D.TransformedDistribution(dist, [trans]).log_prob(trans(p)).squeeze(-1) for p in ptcl]
        for rd, td in zip(result.log_densities, transformed_dens):
            td = td - td.sum() / td.numel()  # Normalize log densities so that mean is 0
            assert_equal_within_error(rd, td)
        # Check resulting log densities against real LogNormal densities
        expected_dens = [expected_dist.log_prob(tp).squeeze(-1) for tp in result.particles]
        for rd, ed in zip(result.log_densities, expected_dens):
            ed = ed - ed.sum() / ed.numel()  # Normalize log densities so that mean is 0
            assert_equal_within_error(rd, ed)

    def test_event_reweight(self):
        b_shape, p_shape, s_shape, e_shape = Size([4]), Size([1]), Size([4, 5, 6]), Size([1, 2, 3])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        target_log_prob = torch.randn([4, 4, 5, 6])
        result = msg.event_reweight(target_log_prob)

        # Check content
        dens_expanded = [
            msg.log_densities[0].unsqueeze(1).unsqueeze(2).expand(-1, 5, 6),
            msg.log_densities[1].unsqueeze(0).unsqueeze(2).expand(4, -1, 6),
            msg.log_densities[2].unsqueeze(0).unsqueeze(1).expand(4, 5, -1),
        ]
        joint_log_dens = sum(dens_expanded)
        expected_log_ratio = target_log_prob - joint_log_dens.unsqueeze(0)
        expected_weight = torch.exp(expected_log_ratio)
        expected_weight /= expected_weight.sum(dim=[-1, -2, -3], keepdims=True)

        assert_equal_within_error(result.weight, expected_weight)

    def test_event_marginalize_positive_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([10]), Size([1]), Size([4, 5, 6]), Size([1, 2, 3])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        event_dim = 1

        dens_expanded = [
            msg.log_densities[0].unsqueeze(1).unsqueeze(2).expand(-1, 5, 6),
            msg.log_densities[1].unsqueeze(0).unsqueeze(2).expand(4, -1, 6),
            msg.log_densities[2].unsqueeze(0).unsqueeze(1).expand(4, 5, -1),
        ]
        joint_log_dens = sum(dens_expanded)
        target_prob = msg.weight * torch.exp(joint_log_dens).unsqueeze(0)

        marginalized_target_prob = target_prob.sum(dim=2, keepdim=False)

        marg_dens_expanded = [
            msg.log_densities[0].unsqueeze(1).expand(-1, 6),
            msg.log_densities[2].unsqueeze(0).expand(4, -1)
        ]
        joint_marg_log_dens = sum(marg_dens_expanded)

        expected_weight = marginalized_target_prob / torch.exp(joint_marg_log_dens).unsqueeze(0)
        expected_weight /= expected_weight.sum(dim=[-1, -2], keepdim=True)

        result = msg.event_marginalize(event_dim)

        # Check shape
        assert result.weight.shape == Size([10, 4, 6])
        assert result.particles[0].shape == Size([4, 1]) and result.particles[1].shape == Size([6, 3])
        assert result.log_densities[0].shape == Size([4]) and result.log_densities[1].shape == Size([6])

        # Check content
        assert_equal_within_error(result.weight, expected_weight)

    def test_event_marginalize_negative_dims(self):
        b_shape, p_shape, s_shape, e_shape = Size([10]), Size([1]), Size([4, 5, 6]), Size([1, 2, 3])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        event_dim = -2

        dens_expanded = [
            msg.log_densities[0].unsqueeze(1).unsqueeze(2).expand(-1, 5, 6),
            msg.log_densities[1].unsqueeze(0).unsqueeze(2).expand(4, -1, 6),
            msg.log_densities[2].unsqueeze(0).unsqueeze(1).expand(4, 5, -1),
        ]
        joint_log_dens = sum(dens_expanded)
        target_prob = msg.weight * torch.exp(joint_log_dens).unsqueeze(0)

        marginalized_target_prob = target_prob.sum(dim=2, keepdim=False)

        marg_dens_expanded = [
            msg.log_densities[0].unsqueeze(1).expand(-1, 6),
            msg.log_densities[2].unsqueeze(0).expand(4, -1)
        ]
        joint_marg_log_dens = sum(marg_dens_expanded)

        expected_weight = marginalized_target_prob / torch.exp(joint_marg_log_dens).unsqueeze(0)
        expected_weight /= expected_weight.sum(dim=[-1, -2], keepdim=True)

        result = msg.event_marginalize(event_dim)

        # Check shape
        assert result.weight.shape == Size([10, 4, 6])
        assert result.particles[0].shape == Size([4, 1]) and result.particles[1].shape == Size([6, 3])
        assert result.log_densities[0].shape == Size([4]) and result.log_densities[1].shape == Size([6])

        # Check content
        assert_equal_within_error(result.weight, expected_weight)

    def test_event_concatenate_shape_1_positive_dims(self):
        # Test 2: flatten 2 out of 4 RVs into 3 RVs
        b_shape, p_shape, s_shape, e_shape = Size([10]), Size([1]), Size([4, 5, 6, 7]), Size([1, 2, 3, 4])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # scenario 1: positive dims
        result = msg.event_concatenate([0, 2], 1)

        # Check shape
        assert result.s_shape == Size([5, 24, 7]) and result.e_shape == Size([2, 4, 4])
        assert result.weight.shape == Size([10, 5, 24, 7])
        assert result.particles[0].shape == Size([5, 2]) and \
               result.particles[1].shape == Size([24, 4]) and \
               result.particles[2].shape == Size([7, 4])
        assert result.log_densities[0].shape == Size([5]) and \
               result.log_densities[1].shape == Size([24]) and \
               result.log_densities[2].shape == Size([7])

    def test_event_concatenate_shape_1_negative_dims(self):
        # Test 2: flatten 2 out of 4 RVs into 3 RVs
        b_shape, p_shape, s_shape, e_shape = Size([10]), Size([1]), Size([4, 5, 6, 7]), Size([1, 2, 3, 4])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # scenario 1: positive dims
        result = msg.event_concatenate([-4, -2], -2)

        # Check shape
        assert result.s_shape == Size([5, 24, 7]) and result.e_shape == Size([2, 4, 4])
        assert result.weight.shape == Size([10, 5, 24, 7])
        assert result.particles[0].shape == Size([5, 2]) and \
               result.particles[1].shape == Size([24, 4]) and \
               result.particles[2].shape == Size([7, 4])
        assert result.log_densities[0].shape == Size([5]) and \
               result.log_densities[1].shape == Size([24]) and \
               result.log_densities[2].shape == Size([7])

    def test_event_concatenate_shape_2_positive_dims(self):
        # Test 2: flatten 2 out of 5 RVs into 4 RVs
        b_shape, p_shape, s_shape, e_shape = Size([10]), Size([1]), Size([4, 5, 6, 7, 8]), Size([1, 2, 3, 4, 5])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # scenario 1: positive dims
        result = msg.event_concatenate([0, 3], 1)

        # Check shape
        assert result.s_shape == Size([5, 28, 6, 8]) and result.e_shape == Size([2, 5, 3, 5])
        assert result.weight.shape == Size([10, 5, 28, 6, 8])
        assert result.particles[0].shape == Size([5, 2]) and \
               result.particles[1].shape == Size([28, 5]) and \
               result.particles[2].shape == Size([6, 3]) and \
               result.particles[3].shape == Size([8, 5])
        assert result.log_densities[0].shape == Size([5]) and \
               result.log_densities[1].shape == Size([28]) and \
               result.log_densities[2].shape == Size([6]) and \
               result.log_densities[3].shape == Size([8])

    def test_event_concatenate_shape_2_negative_dims(self):
        # Test 2: flatten 2 out of 5 RVs into 4 RVs
        b_shape, p_shape, s_shape, e_shape = Size([10]), Size([1]), Size([4, 5, 6, 7, 8]), Size([1, 2, 3, 4, 5])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # scenario 1: positive dims
        result = msg.event_concatenate([-5, -2], -3)

        # Check shape
        assert result.s_shape == Size([5, 28, 6, 8]) and result.e_shape == Size([2, 5, 3, 5])
        assert result.weight.shape == Size([10, 5, 28, 6, 8])
        assert result.particles[0].shape == Size([5, 2]) and \
               result.particles[1].shape == Size([28, 5]) and \
               result.particles[2].shape == Size([6, 3]) and \
               result.particles[3].shape == Size([8, 5])
        assert result.log_densities[0].shape == Size([5]) and \
               result.log_densities[1].shape == Size([28]) and \
               result.log_densities[2].shape == Size([6]) and \
               result.log_densities[3].shape == Size([8])

    def test_event_concatenate_shape_3_positive_dims(self):
        # Test 3: flatten 3 out of 5 RVs into 3 RVs
        b_shape, p_shape, s_shape, e_shape = Size([10]), Size([1]), Size([4, 5, 6, 7, 8]), Size([1, 2, 3, 4, 5])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # scenario 1: positive dims
        result = msg.event_concatenate([0, 1, 3], 0)

        # Check shape
        assert result.s_shape == Size([140, 6, 8]) and result.e_shape == Size([7, 3, 5])
        assert result.weight.shape == Size([10, 140, 6, 8])
        assert result.particles[0].shape == Size([140, 7]) and \
               result.particles[1].shape == Size([6, 3]) and \
               result.particles[2].shape == Size([8, 5])
        assert result.log_densities[0].shape == Size([140]) and \
               result.log_densities[1].shape == Size([6]) and \
               result.log_densities[2].shape == Size([8])

    def test_event_concatenate_shape_3_negative_dims(self):
        # Test 3: flatten 3 out of 5 RVs into 3 RVs
        b_shape, p_shape, s_shape, e_shape = Size([10]), Size([1]), Size([4, 5, 6, 7, 8]), Size([1, 2, 3, 4, 5])
        msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)

        # Scenario 2: negative dims
        result = msg.event_concatenate([-5, -4, -2], -3)

        # Check shape
        assert result.s_shape == Size([140, 6, 8]) and result.e_shape == Size([7, 3, 5])
        assert result.weight.shape == Size([10, 140, 6, 8])
        assert result.particles[0].shape == Size([140, 7]) and \
               result.particles[1].shape == Size([6, 3]) and \
               result.particles[2].shape == Size([8, 5])
        assert result.log_densities[0].shape == Size([140]) and \
               result.log_densities[1].shape == Size([6]) and \
               result.log_densities[2].shape == Size([8])

    def test_event_concatenate_ad_hoc_1(self):
        # Test 1: simple 2 RVs flatten into a single RV
        b_shape, s_shape, e_shape = Size([10]), Size([4, 5]), Size([1, 2])
        ptcl = [
            torch.tensor(
                [[0.],
                 [1.],
                 [2.],
                 [3.]]
            ),
            torch.tensor(
                [[10., 20.],
                 [11., 21.],
                 [12., 22.],
                 [13., 23.],
                 [14., 24.]]
            )
        ]
        dens = [
            torch.tensor([0., .1, .2, .3]),
            torch.tensor([0., .1, .2, .3, .4])
        ]
        weight = torch.rand([10, 4, 5])
        msg = Message(MessageType.Particles,
                      batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                      particles=ptcl, weight=weight, log_densities=dens)

        result = msg.event_concatenate([0, 1])

        # Check shape
        assert result.s_shape == Size([20]) and result.e_shape == Size([3])
        assert result.weight.shape == Size([10, 20])
        assert result.particles[0].shape == Size([20, 3])
        assert result.log_densities[0].shape == Size([20])

        # Check content
        expected_ptcl_raw = []
        expected_dens_raw = []
        for i in range(4):
            for j in range(5):
                expected_ptcl_raw.append([i, 10 + j, 20 + j])
                expected_dens_raw.append(i / 10 + j / 10)
        expected_ptcl = torch.tensor(expected_ptcl_raw, dtype=torch.float)
        expected_dens = torch.tensor(expected_dens_raw, dtype=torch.float)
        expected_dens -= expected_dens.sum() / expected_dens.numel()
        expected_weight = weight.view([10, 20])

        assert_equal_within_error(result.particles[0], expected_ptcl)
        assert_equal_within_error(result.log_densities[0], expected_dens)
        assert_proportional_within_error(result.weight, expected_weight, dims=[-1])

    def test_event_deconcatenate_full_decat_1(self):
        # Original message has only 1 rv.
        # Decat that random variable into 2 marginal rvs
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([]), Size([10, 20]), Size([6, 7])
        msg = random_message(MessageType.Particles, b_shape, p_shape, s_shape, e_shape)

        cat_msg = msg.event_concatenate([0, 1], 0)

        return_msg = cat_msg.event_deconcatenate(0, s_shape, e_shape)

        assert msg.type is MessageType.Particles
        # The decat message shall equal to the pre-cat message
        assert return_msg == msg

    def test_event_deconcatenate_full_decat_2(self):
        # Original message has only 1 rv.
        # Decat that random variable into 4 marginal rvs
        # Message has auxiliary attributes
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([]), Size([2, 3, 4, 5]), Size([6, 7, 8, 9])
        msg = random_message(MessageType.Particles, b_shape, p_shape, s_shape, e_shape)
        msg.attr = {'a': 1, 'b': 2, 'c': 3}

        cat_msg = msg.event_concatenate([0, 1, 2, 3], 0)

        return_msg = cat_msg.event_deconcatenate(0, s_shape, e_shape)

        assert msg.type is MessageType.Particles
        # The decat message shall equal to the pre-cat message
        assert return_msg == msg

    def test_event_deconcatenate_partial_decat_1(self):
        # Original message has 2 rvs.
        # Decat one of the many random variable into 2 marginal rvs
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([]), Size([10, 15, 20]), Size([6, 7, 8])
        msg = random_message(MessageType.Particles, b_shape, p_shape, s_shape, e_shape)

        # Concat the first and third rv, and put as the last rv
        cat_msg = msg.event_concatenate([0, 2], -1)

        return_msg = cat_msg.event_deconcatenate(-1, [10, 20], [6, 8])

        assert msg.type is MessageType.Particles
        # The decat message shall equal to the pre-cat message with events permuted
        expected_msg = msg.event_permute([1, 0, 2])
        assert return_msg == expected_msg

    def test_event_deconcatenate_partial_decat_2(self):
        # Original message has 3 rvs.
        # Decat one of the many random variable into 3 marginal rvs
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([]), Size([2, 3, 4, 5, 6]), Size([7, 8, 9, 10, 11])
        msg = random_message(MessageType.Particles, b_shape, p_shape, s_shape, e_shape)

        # Concat the first and third rv, and put as the last rv
        cat_msg = msg.event_concatenate([0, 2, 3], -1)

        return_msg = cat_msg.event_deconcatenate(-1, [2, 4, 5], [7, 9, 10])

        assert msg.type is MessageType.Particles
        # The decat message shall equal to the pre-cat message with events permuted
        expected_msg = msg.event_permute([1, 4, 0, 2, 3])
        assert return_msg == expected_msg

    def test_event_deconcatenate_value_error_1(self):
        # Test that a value error is raised if the message can not be de-concatenated
        # Test 1: make a random message
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([]), Size([200]), Size([13])
        msg = random_message(MessageType.Particles, b_shape, p_shape, s_shape, e_shape)

        with pytest.raises(ValueError):
            return_msg = msg.event_deconcatenate(0, [10, 20], [6, 7])

    def test_event_deconcatenate_value_error_2(self):
        # Test that a value error is raised if the message can not be de-concatenated
        # Test 2: decat a already concatenated message, but the event order is wrong
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([]), Size([10, 20]), Size([6, 7])
        msg = random_message(MessageType.Particles, b_shape, p_shape, s_shape, e_shape)

        cat_msg = msg.event_concatenate([0, 1], 0)

        with pytest.raises(ValueError):
            return_msg = cat_msg.event_deconcatenate(0, [20, 10], [7, 6])

    def test_event_permute_single_var(self):
        # Single random variable
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([]), Size([10]), Size([3])
        msg = random_message(MessageType.Particles, b_shape, p_shape, s_shape, e_shape)

        return_msg = msg.event_permute([0])

        # In this case, the return message should equal the original message
        assert msg.type is MessageType.Particles
        assert return_msg == msg

    def test_event_permute_multiple_var(self):
        # Multiple random variables
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([]), Size([10, 15, 20]), Size([3, 6, 9])
        msg = random_message(MessageType.Particles, b_shape, p_shape, s_shape, e_shape)

        perm_order = [2, 0, 1]
        return_msg = msg.event_permute(perm_order)

        assert msg.type is MessageType.Particles
        # Check particles
        for i, pi in enumerate(perm_order):
            assert_equal_within_error(msg.particles[pi], return_msg.particles[i])
        # Check weight
        expected_weight = msg.weight.permute([0, 1, 4, 2, 3])
        assert_equal_within_error(return_msg.weight, expected_weight)
        # Check densities
        for i, pi in enumerate(perm_order):
            assert_equal_within_error(msg.log_densities[pi], return_msg.log_densities[i])

    def test_event_permute_identity(self):
        # Test with identity message
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([]), Size([10, 15, 20]), Size([3, 6, 9])
        msg = Message(MessageType.Particles,
                      b_shape, p_shape, s_shape, e_shape,
                      particles=[torch.randn([10, 3]), torch.randn([15, 6]), torch.randn([20, 9])],
                      weight=1,
                      log_densities=[torch.randn([10]), torch.randn([15]), torch.randn([20])])

        return_msg = msg.event_permute([2, 0, 1])

        assert msg == return_msg

