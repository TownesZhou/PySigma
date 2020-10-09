"""
    Unit tests for Message class
"""

import pytest
import torch
from torch import Size
from pysigma.defs import Message, MessageType


# Numerical accuracy
EPS = 1e-6


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

        # Test init with auxiliary arguments.
        msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=torch.randn([5, 3]),
                      k1=1, k2=2, k3=3)
        assert msg.attr['k1'] == 1 and msg.attr['k2'] == 2 and msg.attr['k3'] == 3

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
        msg = Message(MessageType.Both, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
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
        msg = Message(MessageType.Both, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=param, particles=part, weight=w, log_densities=l)
        assert not msg.isid

        # weight is uniform but parameter not uniform
        msg = Message(MessageType.Both, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=param, particles=part, weight=1, log_densities=l)
        assert not msg.isid

        # parameter is uniform but weight not uniform
        msg = Message(MessageType.Both, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                      parameter=0, particles=part, weight=w, log_densities=l)
        assert not msg.isid

    def test_eq(self):
        # Test the equality testing operator
        # Test equality
        # Parameter
        t1 = torch.randn([5, 3])
        t2 = t1.clone()
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2)
        assert msg1 == msg2

        # Particles
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

        # Test with auxiliary data
        t1 = torch.randn([5, 3])
        t2 = t1.clone()
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1,
                       a=1, b=2, c=3, extra=4)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2,
                       a=1, b=2, c=3, extra=4)
        assert msg1 == msg2

        # Test inequality
        # Parameter
        t1 = torch.randn([5, 3])
        t2 = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2)
        assert msg1 != msg2

        # Particles
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

        # Test with auxiliary data
        t1 = torch.randn([5, 3])
        t2 = t1.clone()
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2,
                       a=1, b=2, c=3, extra=4)
        assert msg1 != msg2

        # Test with different devices
        # Only run this test if GPU is available
        if not torch.cuda.is_available():
            return
        device = torch.cuda.current_device()

        t1 = torch.randn([5, 3])
        t2 = t1.clone()
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2,
                       device=device)
        assert msg1 == msg2

    def test_add_invalid_addend(self):
        # Add to Non-message
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        with pytest.raises(AssertionError):
            msg = msg1 + 1

        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg = msg2 + "something random"

        # Incompatible message type
        # Add Parameter to Particle
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

        # Incompatible message shapes
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([10]), param_shape=Size([3]),
                       parameter=torch.randn([10, 3]))
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([6]),
                       parameter=torch.randn([5, 6]))
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([5]), event_shape=Size([3]),
                       particles=[torch.randn(5, 3)], weight=torch.rand(5, 5), log_densities=[-torch.rand(5)])
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([6]),
                       particles=[torch.randn(10, 6)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10, 20]), event_shape=Size([3, 6]),
                       particles=[torch.randn(10, 3), torch.randn(20, 6)], weight=torch.rand(5, 10, 20),
                       log_densities=[-torch.rand(10), -torch.rand(20)])
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

        # Incompatible event / log densities tensor for Particle addend
        part1, part2 = [torch.randn(10, 3)], [torch.randn(10, 3)]
        dens1, dens2 = [-torch.rand(10)], [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=part1, weight=torch.rand(5, 10), log_densities=dens1)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=part2, weight=torch.rand(5, 10), log_densities=dens1)
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=part1, weight=torch.rand(5, 10), log_densities=dens1)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=part1, weight=torch.rand(5, 10), log_densities=dens2)
        with pytest.raises(AssertionError):
            msg = msg1 + msg2

    def test_add_correct_values(self):
        # Test that the resulting message have correct tensor values
        # Parameter
        t1, t2 = torch.randn([5, 3]), torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2)
        msg = msg1 + msg2
        assert torch.equal(msg.parameter, t1 + t2)

        # Particles
        # one relational and one random variable
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
        ratio = (w1 * w2) / msg.weight
        min_const, _ = ratio.min(dim=-1)
        max_const, _ = ratio.max(dim=-1)
        assert torch.max(max_const - min_const) < EPS

        # Multiple relational and random variables
        b_s, s_s, e_s = Size([5, 6, 7]), Size([10, 12, 14]), Size([3, 2, 1])
        p1 = [torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])]
        w1, w2 = torch.rand(5, 6, 7, 10, 12, 14), torch.rand(5, 6, 7, 10, 12, 14)
        l1 = [-torch.rand(10), -torch.rand(12), -torch.rand(14)]
        msg1 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                      particles=p1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w2, log_densities=l1)
        msg = msg1 + msg2
        # The sample dimension across ratio should be constant
        ratio = (w1 * w2) / msg.weight
        min_const = ratio.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
        max_const = ratio.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
        assert torch.max(max_const - min_const) < EPS

        # Test addition with identity
        # with same shape
        t1 = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=0)
        msg = msg1 + msg2
        assert torch.equal(msg.parameter, t1)
        assert msg == msg1

        t1 = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=0)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg = msg1 + msg2
        assert torch.equal(msg.parameter, t1)
        assert msg == msg2

        b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        w1 = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=1, log_densities=l1)
        msg = msg1 + msg2
        assert torch.equal(msg.weight, w1)
        assert msg == msg1

        # with different shape and attributes
        t1 = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([10]), param_shape=Size([9]), parameter=0)
        msg = msg1 + msg2
        assert torch.equal(msg.parameter, t1)
        assert msg == msg1

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
        assert torch.equal(msg.weight, w1)
        assert msg == msg1

    def test_iadd(self):
        t1, t2 = torch.randn([5, 3]), torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]), parameter=t2)
        msg1 += msg2
        assert torch.equal(msg1.parameter, t1 + t2)

        b_s, s_s, e_s = Size([5]), Size([10]), Size([3])
        p1 = [torch.randn(10, 3)]
        w1, w2 = torch.rand(5, 10), torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Particles, batch_shape=b_s, sample_shape=s_s, event_shape=e_s,
                       particles=p1, weight=w2, log_densities=l1)
        msg1 += msg2
        ratio = (w1 * w2) / msg1.weight
        min_const, _ = ratio.min(dim=-1)
        max_const, _ = ratio.max(dim=-1)
        assert torch.max(max_const - min_const) < EPS

    def test_mul_invalid_multiplier(self):
        # Multiply message with another message
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        with pytest.raises(AssertionError):
            msg1 * msg2

        # Multiply with other data structure
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        with pytest.raises(AssertionError):
            msg1 * "some random stuff"
        with pytest.raises(AssertionError):
            msg1 * None

        # Multiply with Tensor of wrong shape
        with pytest.raises(AssertionError):
            msg1 * torch.randn(5, 3)

    def test_mul_valid_multiplier(self):
        # With int
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                      event_shape=Size([3, 2, 1]),
                      particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                      weight=torch.rand(5, 6, 7, 10, 12, 14),
                      log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        msg1 * 2
        msg2 * 2

        # With float
        msg1 * 1.5
        msg2 * 1.5

        # With singleton tensor
        msg1 * torch.tensor(1.5)
        msg2 * torch.tensor(1.5)
        msg1 * torch.tensor([1.5])
        msg2 * torch.tensor([1.5])

        # With batch shape tensor
        msg1 * torch.randn([5])
        msg2 * torch.randn([5, 6, 7])

    def test_mul_correct_values(self):
        # Parameter
        param = torch.randn([5, 3])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=param)
        # With single scalar
        result1 = msg1 * 2
        result2 = msg1 * 2.0
        result3 = msg1 * torch.tensor(2.0)
        result4 = msg1 * torch.tensor([2.0])
        assert torch.equal(result1.parameter, param * 2)
        assert result1 == result2 == result3 == result4

        # With batch-wise scalar
        scalar = torch.randn([5])
        result = msg1 * scalar
        assert torch.equal(result.parameter, param * scalar.unsqueeze(1))

        # Particles
        # One relational and random variable
        # weight = torch.rand(5, 6, 7, 10, 12, 14)
        # msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
        #                event_shape=Size([3, 2, 1]),
        #                particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
        #                weight=torch.rand(5, 6, 7, 10, 12, 14),
        #                log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        weight = torch.rand(5, 10)
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                      particles=[torch.randn(10, 3)], weight=weight, log_densities=[-torch.rand(10)])
        # With single scalar
        result1 = msg1 * 2
        result2 = msg1 * 2.0
        result3 = msg1 * torch.tensor(2.0)
        result4 = msg1 * torch.tensor([2.0])
        # The ratio across sample dimensions should be constant
        ratio = weight ** 2.0 / result1.weight
        min_const = ratio.min(dim=-1)[0]
        max_const = ratio.max(dim=-1)[0]
        assert torch.max((max_const - min_const) / min_const.norm()) < EPS
        assert result1 == result2 == result3 == result4

        # With batch-wise scalar
        scalar = torch.randn([5])
        scalar_expanded = scalar.unsqueeze(-1)
        result = msg1 * scalar
        expected_weight = weight ** scalar_expanded
        ratio = expected_weight / result.weight
        min_const = ratio.min(dim=-1)[0]
        max_const = ratio.max(dim=-1)[0]
        assert torch.max((max_const - min_const) / min_const.norm()) < EPS

        # Multiple relational and random variables
        weight = torch.rand(5, 6, 7, 10, 12, 14)
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=weight,
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        # With single scalar
        result1 = msg1 * 2
        result2 = msg1 * 2.0
        result3 = msg1 * torch.tensor(2.0)
        result4 = msg1 * torch.tensor([2.0])
        # The ratio across sample dimensions should be constant
        ratio = weight ** 2.0 / result1.weight
        min_const = ratio.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
        max_const = ratio.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
        assert torch.max((max_const - min_const) / min_const.norm()) < EPS
        assert result1 == result2 == result3 == result4

        # With batch-wise scalar
        scalar = torch.randn([5, 6, 7])
        scalar_expanded = scalar.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        result = msg1 * scalar
        expected_weight = weight ** scalar_expanded
        ratio = expected_weight / result.weight
        min_const = ratio.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0]
        max_const = ratio.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
        assert torch.max((max_const - min_const) / min_const.norm()) < EPS

    def test_mul_identity(self):
        # Test multiplication when the message itself is an identity.
        # In this case, the returned message should be the original message itself
        # Parameter
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=0)
        result1 = msg1 * 2.0
        assert msg1 == result1

        # Particles
        msg2 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=1,
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        result2 = msg2 * 5.0
        assert msg2 == result2
