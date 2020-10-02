"""
    Unit tests for Message class
"""

import pytest
import torch
from torch import Size
from pysigma.defs import Message, MessageType


class TestMessage:

    def test_correct_init(self):
        # Test correct normal init
        # Normal none-identity messages
        # Parameter type
        # One relational variable
        t1 = torch.randn([5, 3])
        msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        assert torch.equal(msg.parameter, t1)

        # Two relational variable
        t1 = torch.randn([5, 6, 3])
        msg = Message(MessageType.Parameter, batch_shape=Size([5, 6]), param_shape=Size([3]), parameter=t1)
        assert torch.equal(msg.parameter, t1)

        # Particles type
        # One relational and random variable
        b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                      particles=p1, weight=w1, log_densities=l1)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(msg.particles, p1))
        assert torch.equal(msg.weight, w1)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(msg.log_densities, l1))

        # Two relational and random variables
        b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
        p1 = [torch.randn(10, 3), torch.randn([12, 2])]
        w1 = torch.rand(5, 6, 10, 12)
        l1 = [-torch.rand(10), -torch.rand(12)]
        msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                      particles=p1, weight=w1, log_densities=l1)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(msg.particles, p1))
        assert torch.equal(msg.weight, w1)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(msg.log_densities, l1))

        # Multiple relational and random variables
        b_s, s_s, e_s = Size([5, 6, 7]), Size([10, 12, 14]), Size([3, 2, 1])
        p1 = [torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])]
        w1 = torch.rand(5, 6, 7, 10, 12, 14)
        l1 = [-torch.rand(10), -torch.rand(12), -torch.rand(14)]
        msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                      particles=p1, weight=w1, log_densities=l1)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(msg.particles, p1))
        assert torch.equal(msg.weight, w1)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(msg.log_densities, l1))

        # Both type
        # One relational variable and random variable
        b_s, p_s, s_s, e_s = Size([5]), Size([4]), Size([10]), Size([3])
        param1 = torch.randn([5, 4])
        part1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg = Message(MessageType.Particles, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=param1, particles=part1, weight=w1, log_densities=l1)
        assert torch.equal(msg.parameter, param1)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(msg.particles, part1))
        assert torch.equal(msg.weight, w1)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(msg.log_densities, l1))

        # Multiple relational variable and random variable
        b_s, p_s, s_s, e_s = Size([5, 6]), Size([4]), Size([10, 11]), Size([3, 2])
        param1 = torch.randn([5, 6, 4])
        part1 = [torch.randn(10, 3), torch.randn(11, 2)]
        w1 = torch.rand(5, 6, 10, 11)
        l1 = [-torch.rand(10), -torch.rand(11)]
        msg = Message(MessageType.Particles, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=param1, particles=part1, weight=w1, log_densities=l1)
        assert torch.equal(msg.parameter, param1)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(msg.particles, part1))
        assert torch.equal(msg.weight, w1)
        assert all(torch.equal(t1, t2) for t1, t2 in zip(msg.log_densities, l1))

        # Identity with no batch shape specified
        p_s = Size([3])
        Message(MessageType.Parameter, param_shape=p_s, parameter=0)

        s_s, e_s = Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        l1 = [-torch.rand(10)]
        Message(MessageType.Particles, sample_shape=s_s, event_shape=e_s, particles=p1, weight=1, log_densities=l1)

        Message(MessageType.Both, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                parameter=0, particles=p1, weight=1, log_densities=l1)

        # Identity with batch shape specified
        p_s = Size([3])
        Message(MessageType.Parameter, batch_shape=Size([5, 6]), param_shape=p_s, parameter=0)

        s_s, e_s = Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        l1 = [-torch.rand(10)]
        Message(MessageType.Particles, batch_shape=Size([5, 6]), sample_shape=s_s, event_shape=e_s,
                particles=p1, weight=1, log_densities=l1)

        Message(MessageType.Both, batch_shape=Size([5, 6]), param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                parameter=0, particles=p1, weight=1, log_densities=l1)

    def test_correct_init_with_device(self):
        # Only run this test if GPU is available
        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        # Parameter type
        t1 = torch.randn([5, 3])
        msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1, device=device)
        assert msg.parameter.get_device() == device

        # Particles type
        b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                      particles=p1, weight=w1, log_densities=l1, device=device)
        assert all(t.get_device() == device for t in msg.particles)
        assert msg.weight.get_device() == device
        assert all(t.get_device() == device for t in msg.log_densities)

    def test_incorrect_init_wrong_shape(self):
        # Test init with incorrect arguments
        # Parameter type
        # Mismatched shape
        with pytest.raises(AssertionError):
            t1 = torch.randn([5, 3])
            msg = Message(MessageType.Parameter, batch_shape=Size([2]), param_shape=Size([3]), parameter=t1)

        with pytest.raises(AssertionError):
            t1 = torch.randn([5, 3])
            msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([2]), parameter=t1)

        # Empty shape
        with pytest.raises(AssertionError):
            t1 = torch.randn([5, 3, 2])
            msg = Message(MessageType.Parameter, batch_shape=Size([5, 3, 2]), param_shape=Size([]), parameter=t1)

        with pytest.raises(AssertionError):
            t1 = torch.randn([5, 3, 2])
            msg = Message(MessageType.Parameter, batch_shape=Size([]), param_shape=Size([5, 3, 2]), parameter=t1)

        # Param shape longer than 1
        with pytest.raises(AssertionError):
            t1 = torch.randn([5, 3, 2])
            msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3, 2]), parameter=t1)

        # Particles type
        # Empty shape
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([]), Size([10]), Size([3])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(10)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([]), Size([3])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(5)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(5)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([]), Size([])
            p1 = []
            w1 = torch.rand(5)
            l1 = []
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        # Mismatched shape
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
            p1 = [torch.randn(10, 4)]
            w1 = torch.rand(5, 10)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(5, 11)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(5, 10)
            l1 = [-torch.rand(11)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        # Wrong number of particle tensors
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3)]
            w1 = torch.rand(5, 6, 10, 12)
            l1 = [-torch.rand(10), -torch.rand(12)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3), torch.randn([12, 2]), torch.randn([13, 19])]
            w1 = torch.rand(5, 6, 10, 12)
            l1 = [-torch.rand(10), -torch.rand(12)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3), torch.randn([12, 2])]
            w1 = torch.rand(5, 6)
            l1 = [-torch.rand(10), -torch.rand(12)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3), torch.randn([12, 2])]
            w1 = torch.rand(5, 6, 10, 12, 13, 19)
            l1 = [-torch.rand(10), -torch.rand(12)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3), torch.randn([12, 2])]
            w1 = torch.rand(5, 6, 10, 12)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6]), Size([10, 12]), Size([3, 2])
            p1 = [torch.randn(10, 3), torch.randn([12, 2])]
            w1 = torch.rand(5, 6, 10, 12)
            l1 = [-torch.rand(10), -torch.rand(12), -torch.rand(113)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        # Particles in wrong order
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5, 6, 7]), Size([10, 12, 14]), Size([3, 2, 1])
            p1 = [torch.randn(10, 3), torch.randn([14, 1]), torch.randn([12, 2])]
            w1 = torch.rand(5, 6, 7, 10, 14, 12)
            l1 = [-torch.rand(10), -torch.rand(14), -torch.rand(12)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_incorrect_init_wrong_values(self):
        # Wrong weight value - not necessarily positive
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
            p1 = [torch.randn(10, 3)]
            w1 = torch.randn(5, 10)
            l1 = [-torch.rand(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

        # Wrong log densities value - not necessarily non-positive
        with pytest.raises(AssertionError):
            b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
            p1 = [torch.rand(10, 3)]
            w1 = torch.rand(5, 10)
            l1 = [torch.randn(10)]
            msg = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                          particles=p1, weight=w1, log_densities=l1)

    def test_incorrect_init_wrong_device(self):
        device = "Some random string"

        with pytest.raises(Exception):
            t1 = torch.randn([5, 3])
            msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1,
                          device=device)

