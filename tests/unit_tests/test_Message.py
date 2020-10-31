"""
    Unit tests for Message class
"""

import pytest
import torch
from torch import Size
import torch.distributions as D
import torch.distributions.transforms as T
from pysigma.defs import Message, MessageType


# Numerical accuracy
EPS = 1e-6


class TestMessage:

    # Utility method - test equality with given numerical precision

    # Check if the given two tensors are equal up to a predefined numerical precision
    @staticmethod
    def equal_within_error(tensor_1, tensor_2):
        return torch.max(torch.abs(tensor_1 - tensor_2)) < EPS

    # Check if the given two messages are proportional to each other along the hyperplane spanning the last few
    # dimensions (i.e. messages are proportional in the quotient space) up to a predefined numerical precision
    @staticmethod
    def proportional(tensor_1, tensor_2, num_quotient_dims=1):
        ratio = tensor_1 / tensor_2
        min_const, max_const = ratio, ratio
        for i in range(num_quotient_dims):
            min_const = min_const.min(dim=-1)[0]
            max_const = max_const.max(dim=-1)[0]
        return torch.max(max_const - min_const) < EPS

    @staticmethod
    def random_message(msg_type, b_shape, p_shape, s_shape, e_shape):
        param = torch.randn(b_shape + p_shape) if MessageType.Parameter in msg_type else 0
        weight = torch.rand(b_shape + s_shape) if MessageType.Particles in msg_type else 1
        ptcl = [torch.randn([s, e]) for s, e in zip(list(s_shape), list(e_shape))] \
            if MessageType.Particles in msg_type else None
        dens = [torch.randn(s) for s in zip(list(s_shape))] \
            if MessageType.Particles in msg_type else None
        return Message(msg_type, b_shape, p_shape, s_shape, e_shape,
                       param, ptcl, weight, dens)

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
        msg = Message(MessageType.Both, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
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

        # Both
        b_s, p_s, s_s, e_s = Size([5]), Size([4]), Size([10]), Size([3])
        param1, param2 = torch.randn([5, 4]), torch.randn([5, 4])
        part1 = [torch.randn(10, 3)]
        w1, w2 = torch.rand(5, 10), torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Both, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                       parameter=param1, particles=part1, weight=w1, log_densities=l1)
        msg2 = Message(MessageType.Both, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                       parameter=param2, particles=part1, weight=w2, log_densities=l1)
        msg = msg1 + msg2
        assert torch.equal(msg.parameter, param1 + param2)
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

        # Both
        b_s, p_s, s_s, e_s = Size([5]), Size([4]), Size([10]), Size([3])
        param = torch.randn([5, 4])
        part = [torch.randn(10, 3)]
        weight = torch.rand(5, 10)
        l1 = [-torch.rand(10)]
        msg1 = Message(MessageType.Both, batch_shape=b_s, param_shape=p_s, sample_shape=s_s, event_shape=e_s,
                       parameter=param, particles=part, weight=weight, log_densities=l1)
        # With single scalar
        result1 = msg1 * 2
        # Check parameter component
        assert torch.equal(result1.parameter, param * 2)
        # Check particles component
        ratio = weight ** 2.0 / result1.weight
        min_const = ratio.min(dim=-1)[0]
        max_const = ratio.max(dim=-1)[0]
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

        # Both
        msg3 = Message(MessageType.Both,
                       batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                       parameter=0, particles=[torch.randn(10, 3)], weight=1,
                       log_densities=[-torch.rand(10)])
        result3 = msg3 * 10.0
        assert msg3 == result3

    def test_compose_invalid_arguments_wrong_type(self):
        # Test composing messages with the same type
        # Test 1: both are parameter message
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        with pytest.raises(AssertionError):
            msg = Message.compose(msg1, msg2)

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

        # Test 3: both are Both type message
        msg1 = Message(MessageType.Both,
                       batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                       parameter=torch.randn([5, 4]), particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]),
                       log_densities=[-torch.rand(10)])
        msg2 = Message(MessageType.Both,
                       batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                       parameter=torch.randn([5, 4]), particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]),
                       log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            Message.compose(msg1, msg2)

        # Test composing messages with the different but incompatible type
        # Test 4: Parameter with Both
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Both,
                       batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                       parameter=torch.randn([5, 4]), particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]),
                       log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            Message.compose(msg1, msg2)

        # Test 5: Particles with Both
        msg1 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                      particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        msg2 = Message(MessageType.Both,
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

    def test_compose_invalid_arguments_identity_message(self):
        # Test compose with one identity message
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=0)
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg = Message.compose(msg1, msg2)

        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=1, log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg = Message.compose(msg1, msg2)

    def test_compose_successful_operation(self):
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Particles, batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                       particles=[torch.randn(10, 3)], weight=torch.rand(5, 10), log_densities=[-torch.rand(10)])
        msg = Message.compose(msg1, msg2)
        assert msg.type is MessageType.Both
        assert torch.equal(msg.parameter, msg1.parameter)
        assert torch.equal(msg.weight, msg2.weight)

    def test_compose_successful_operation_attr_dict_overwrite(self):
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

    def test_identity(self):
        # Test parameter type
        id_msg = Message.identity(MessageType.Parameter)
        assert id_msg.type is MessageType.Parameter
        assert id_msg.isid
        assert id_msg.b_shape == Size([]) and id_msg.p_shape == Size([]) and id_msg.s_shape == Size([]) and \
               id_msg.e_shape == Size([])
        assert id_msg.attr == {}

        # Test particles type
        id_msg = Message.identity(MessageType.Particles)
        assert id_msg.type is MessageType.Particles
        assert id_msg.isid
        assert id_msg.b_shape == Size([]) and id_msg.p_shape == Size([]) and id_msg.s_shape == Size([]) and \
               id_msg.e_shape == Size([])
        assert id_msg.attr == {}

        # Test particles type
        id_msg = Message.identity(MessageType.Both)
        assert id_msg.type is MessageType.Both
        assert id_msg.isid
        assert id_msg.b_shape == Size([]) and id_msg.p_shape == Size([]) and id_msg.s_shape == Size([]) and \
               id_msg.e_shape == Size([])
        assert id_msg.attr == {}

    def test_size(self):
        # Test parameter type
        msg = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                      parameter=torch.randn([5, 3]))
        assert msg.size() == (Size([5]), Size([3]), Size([]), Size([]))

        # Test particles type
        msg = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                      event_shape=Size([3, 2, 1]),
                      particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                      weight=torch.rand([5, 6, 7, 10, 12, 14]),
                      log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        assert msg.size() == (Size([5, 6, 7]), Size([]), Size([10, 12, 14]), Size([3, 2, 1]))

        # Test Both type
        msg = Message(MessageType.Both,
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
        msg1 = Message(MessageType.Particles, batch_shape=Size([5, 6, 7]), sample_shape=Size([10, 12, 14]),
                       event_shape=Size([3, 2, 1]),
                       particles=[torch.randn(10, 3), torch.randn([12, 2]), torch.randn([14, 1])],
                       weight=torch.rand([5, 6, 7, 10, 12, 14]),
                       log_densities=[-torch.rand(10), -torch.rand(12), -torch.rand(14)])
        msg2 = Message.identity(MessageType.Particles)
        assert msg1.same_particles_as(msg2)
        assert msg2.same_particles_as(msg1)

    def test_diff_param_different_devices(self):
        # Only run this test if GPU is available
        if not torch.cuda.is_available():
            return

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

    def test_diff_param_diff_shape(self):
        # Different batch shape
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([6]), param_shape=Size([3]),
                       parameter=torch.randn([6, 3]))
        with pytest.raises(AssertionError):
            msg1.diff_param(msg2)

        # Different param shape
        msg1 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([3]),
                       parameter=torch.randn([5, 3]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([5]), param_shape=Size([4]),
                       parameter=torch.randn([5, 4]))
        with pytest.raises(AssertionError):
            msg1.diff_param(msg2)

    def test_diff_param_ad_hoc_values(self):
        # Test 1: ad-hoc parameter values
        msg1 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([2]),
                       parameter=torch.tensor([[3., 4.]]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([2]),
                       parameter=torch.tensor([[0., 0.]]))
        assert msg1.diff_param(msg2) == msg2.diff_param(msg1) == 5

        # Test 2: ad-hoc parameter values
        msg1 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([2]),
                       parameter=torch.tensor([[4., 6.]]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([2]),
                       parameter=torch.tensor([[1., 2.]]))
        assert msg1.diff_param(msg2) == msg2.diff_param(msg1) == 5

        # Test 3: ad-hoc parameter values
        msg1 = Message(MessageType.Parameter, batch_shape=Size([2]), param_shape=Size([2]),
                       parameter=torch.tensor([[4., 6.], [6., 8.]]))
        msg2 = Message(MessageType.Parameter, batch_shape=Size([2]), param_shape=Size([2]),
                       parameter=torch.tensor([[1., 2.], [0., 0.]]))
        assert msg1.diff_param(msg2) == msg2.diff_param(msg1) == 7.5

    def test_diff_param_random_values(self):
        # Test 1: random single batch
        param1, param2 = torch.rand([1, 10]), torch.rand([1, 10])
        msg1 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([10]),
                       parameter=param1)
        msg2 = Message(MessageType.Parameter, batch_shape=Size([1]), param_shape=Size([10]),
                       parameter=param2)
        diff_param = param1 - param2
        assert torch.equal(msg1.diff_param(msg2), msg2.diff_param(msg1))
        assert torch.equal(msg1.diff_param(msg2), diff_param.norm())

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

    def test_diff_weight_different_device(self):
        # Only run this test if GPU is available
        if not torch.cuda.is_available():
            return

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

        # Different sample shapes and event shapes would be notified when checking same_particles_as()

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

    def test_diff_weight_random_weight(self):
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

    def test_reduce_type_wrong_target_type(self):
        # Test Both target type
        target_type = MessageType.Both
        msg = Message(MessageType.Both,
                      batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                      parameter=torch.randn([5, 4]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg.reduce_type(target_type)

        # Test Parameter message
        msg = Message(MessageType.Parameter,
                      batch_shape=Size([5]), param_shape=Size([4]),
                      parameter=torch.randn([5, 4]))
        with pytest.raises(AssertionError):
            msg.reduce_type(target_type)

        # Test Particles message
        msg = Message(MessageType.Particles,
                      batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg.reduce_type(target_type)

        # Test Particles reduce to Parameter
        msg = Message(MessageType.Particles,
                      batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)])
        with pytest.raises(AssertionError):
            msg.reduce_type(MessageType.Parameter)

        # Test Parameter reduce to Particles
        msg = Message(MessageType.Parameter,
                      batch_shape=Size([5]), param_shape=Size([4]),
                      parameter=torch.randn([5, 4]))
        with pytest.raises(AssertionError):
            msg.reduce_type(MessageType.Particles)

    def test_reduce_type_correct_contents(self):
        test_attr = {"a": 1, "b": 2, "c": 3}
        # Test particles reduce to particles
        msg = Message(MessageType.Particles,
                      batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)],
                      **test_attr)
        reduced_msg = msg.reduce_type(MessageType.Particles)
        assert msg == reduced_msg

        # Test Both reduce to particles
        msg = Message(MessageType.Both,
                      batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                      parameter=torch.randn([5, 4]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)],
                      **test_attr)
        reduced_msg = msg.reduce_type(MessageType.Particles)
        assert msg.b_shape == reduced_msg.b_shape and msg.e_shape == reduced_msg.e_shape and \
               msg.s_shape == reduced_msg.s_shape
        assert reduced_msg.p_shape == Size([])
        assert self.equal_within_error(msg.weight, reduced_msg.weight)
        assert all(torch.equal(p, rp) for p, rp in zip(msg.particles, reduced_msg.particles))
        assert all(torch.equal(d, rd) for d, rd in zip(msg.log_densities, reduced_msg.log_densities))
        assert reduced_msg.parameter == 0

        # Test parameter reduce to parameter
        msg = Message(MessageType.Parameter,
                      batch_shape=Size([5]), param_shape=Size([4]), parameter=torch.randn([5, 4]),
                      **test_attr)
        reduced_msg = msg.reduce_type(MessageType.Parameter)
        assert msg == reduced_msg

        # Test Both reduce to parameter
        msg = Message(MessageType.Both,
                      batch_shape=Size([5]), param_shape=Size([4]), sample_shape=Size([10]), event_shape=Size([3]),
                      parameter=torch.randn([5, 4]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)],
                      **test_attr)
        reduced_msg = msg.reduce_type(MessageType.Parameter)
        assert msg.b_shape == reduced_msg.b_shape and msg.p_shape == reduced_msg.p_shape
        assert reduced_msg.s_shape == Size([]) and reduced_msg.e_shape == Size([])
        assert torch.equal(reduced_msg.parameter, msg.parameter)
        assert reduced_msg.weight == 1

    def test_clone(self):
        test_attr = {"a": 1, "b": 2, "c": 3}
        # Test parameter messages
        msg = Message(MessageType.Parameter,
                      batch_shape=Size([5]), param_shape=Size([4]), parameter=torch.randn([5, 4]),
                      **test_attr)
        cloned_msg = msg.clone()
        assert msg == cloned_msg
        assert msg.parameter is not cloned_msg.parameter

        # Test particles message
        msg = Message(MessageType.Particles,
                      batch_shape=Size([5]), sample_shape=Size([10]), event_shape=Size([3]),
                      particles=[torch.randn(10, 3)], weight=torch.rand([5, 10]), log_densities=[-torch.rand(10)],
                      **test_attr)
        cloned_msg = msg.clone()
        assert msg == cloned_msg
        assert all(p is not rp for p, rp in zip(msg.particles, cloned_msg.particles))
        assert msg.weight is not cloned_msg.weight
        assert all(d is not rd for d, rd in zip(msg.log_densities, cloned_msg.log_densities))

    def test_clone(self):
        # Carry out this test only if system has cuda
        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()

        # Test both
        test_attr = {"a": 1, "b": 2, "c": 3}
        msg = Message(MessageType.Both,
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

    def test_batch_permute(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        # Test 1: Positive dim
        perm_order = [2, 0, 1]
        result = msg.batch_permute(perm_order)

        # Check shape
        assert result.parameter.shape == Size([3, 1, 2, 1])
        assert result.weight.shape == Size([3, 1, 2, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.permute([2, 0, 1, 3]))
        assert self.equal_within_error(result.weight, msg.weight.permute([2, 0, 1, 3, 4, 5]))

        # Test 2 : Negative dim
        perm_order = [-1, -3, -2]
        result = msg.batch_permute(perm_order)

        # Check shape
        assert result.parameter.shape == Size([3, 1, 2, 1])
        assert result.weight.shape == Size([3, 1, 2, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.permute([2, 0, 1, 3]))
        assert self.equal_within_error(result.weight, msg.weight.permute([2, 0, 1, 3, 4, 5]))


    def test_batch_unsqueeze(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2, 3]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = 2
        result = msg.batch_unsqueeze(dim)

        # Check shape
        assert result.parameter.shape == Size([1, 2, 1, 3, 1])
        assert result.weight.shape == Size([1, 2, 1, 3, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.unsqueeze(2))
        assert self.equal_within_error(result.weight, msg.weight.unsqueeze(2))

        # Test 2: negative dim
        dim = -2
        result = msg.batch_unsqueeze(dim)

        # Check shape
        assert result.parameter.shape == Size([1, 2, 1, 3, 1])
        assert result.weight.shape == Size([1, 2, 1, 3, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.unsqueeze(2))
        assert self.equal_within_error(result.weight, msg.weight.unsqueeze(2))

    def test_batch_index_select(self):
        b_shape, p_shape, s_shape, e_shape = Size([11, 22, 33]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = 2
        index = torch.tensor([5, 10, 15, 20, 25, 30], dtype=torch.long)
        result = msg.batch_index_select(dim, index)

        # Check shape
        assert result.parameter.shape == Size([11, 22, 6, 1])
        assert result.weight.shape == Size([11, 22, 6, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.index_select(dim, index))
        assert self.equal_within_error(result.weight, msg.weight.index_select(dim, index))

        # Test 2: negative dim
        dim = -1
        index = torch.tensor([5, 10, 15, 20, 25, 30], dtype=torch.long)
        result = msg.batch_index_select(dim, index)

        # Check shape
        assert result.parameter.shape == Size([11, 22, 6, 1])
        assert result.weight.shape == Size([11, 22, 6, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.index_select(2, index))
        assert self.equal_within_error(result.weight, msg.weight.index_select(2, index))

    def test_batch_index_put(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = 0
        index = torch.tensor([5, 4, 3], dtype=torch.long)
        result = msg.batch_index_put(dim, index)

        # Check shape
        assert result.parameter.shape == Size([6, 4, 5, 1])
        assert result.weight.shape == Size([6, 4, 5, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter[:3], torch.zeros(3, 4, 5, 1))
        assert self.equal_within_error(result.weight[:3], torch.ones(3, 4, 5, 4, 5, 6) / (4 * 5 * 6))
        assert all(self.equal_within_error(result.parameter[ri], msg.parameter[i]) for i, ri in enumerate(index))
        assert all(self.equal_within_error(result.weight[ri], msg.weight[i]) for i, ri in enumerate(index))

        # Test 2: negative dim
        dim = -3
        index = torch.tensor([5, 4, 3], dtype=torch.long)
        result = msg.batch_index_put(dim, index)

        # Check shape
        assert result.parameter.shape == Size([6, 4, 5, 1])
        assert result.weight.shape == Size([6, 4, 5, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter[:3], torch.zeros(3, 4, 5, 1))
        assert self.equal_within_error(result.weight[:3], torch.ones(3, 4, 5, 4, 5, 6) / (4 * 5 * 6))
        assert all(self.equal_within_error(result.parameter[ri], msg.parameter[i]) for i, ri in enumerate(index))
        assert all(self.equal_within_error(result.weight[ri], msg.weight[i]) for i, ri in enumerate(index))

    def test_batch_diagonal(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim1, dim2 = 0, 2
        result = msg.batch_diagonal(dim1, dim2)

        # Check shape
        assert result.parameter.shape == Size([4, 3, 1])
        assert result.weight.shape == Size([4, 3, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.diagonal(0, 0, 2).transpose(-1, -2))
        assert self.equal_within_error(result.weight, msg.weight.diagonal(0, 0, 2).permute(0, 4, 1, 2, 3))

        # Test 2: negative dim
        # Test 1: positive dim
        dim1, dim2 = -3, -1
        result = msg.batch_diagonal(dim1, dim2)

        # Check shape
        assert result.parameter.shape == Size([4, 3, 1])
        assert result.weight.shape == Size([4, 3, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.diagonal(0, 0, 2).transpose(-1, -2))
        assert self.equal_within_error(result.weight, msg.weight.diagonal(0, 0, 2).permute(0, 4, 1, 2, 3))

    def test_batch_diag_embed(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        diag_dim, target_dim1, target_dim2 = 1, 1, 3
        result = msg.batch_diag_embed(diag_dim, target_dim1, target_dim2)

        # Check shape
        assert result.parameter.shape == Size([3, 4, 5, 4, 1])
        assert result.weight.shape == Size([3, 4, 5, 4, 4, 5, 6])

        # Check contents
        assert self.equal_within_error(result.parameter.diagonal(0, 1, 3).permute([0, 3, 1, 2]), msg.parameter)
        assert self.equal_within_error(result.weight.diagonal(0, 1, 3).permute([0, 5, 1, 2, 3, 4]), msg.weight)

        # Test 2: negative dim
        diag_dim, target_dim1, target_dim2 = -2, -4, -2
        result = msg.batch_diag_embed(diag_dim, target_dim1, target_dim2)

        # Check shape
        assert result.parameter.shape == Size([3, 4, 5, 4, 1])
        assert result.weight.shape == Size([3, 4, 5, 4, 4, 5, 6])

        # Check contents
        assert self.equal_within_error(result.parameter.diagonal(0, 1, 3).permute([0, 3, 1, 2]), msg.parameter)
        assert self.equal_within_error(result.weight.diagonal(0, 1, 3).permute([0, 5, 1, 2, 3, 4]), msg.weight)

    def test_batch_narrow(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim, length = 2, 3
        result = msg.batch_narrow(dim, length)

        # Check shape
        assert result.parameter.shape == Size([3, 4, 3, 1])
        assert result.weight.shape == Size([3, 4, 3, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter[:, :, :3, :])
        assert self.equal_within_error(result.weight, msg.weight[:, :, :3, :, :, :])

        # Test 2: negative dim
        dim, length = -1, 3
        result = msg.batch_narrow(dim, length)

        # Check shape
        assert result.parameter.shape == Size([3, 4, 3, 1])
        assert result.weight.shape == Size([3, 4, 3, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter[:, :, :3, :])
        assert self.equal_within_error(result.weight, msg.weight[:, :, :3, :, :, :])

    def test_batch_broaden(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim, length = 0, 6
        result = msg.batch_broaden(dim, length)

        # Check shape
        assert result.parameter.shape == Size([6, 4, 5, 1])
        assert result.weight.shape == Size([6, 4, 5, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter[:3], msg.parameter)
        assert self.equal_within_error(result.parameter[3:], torch.zeros([3, 4, 5, 1]))
        assert self.equal_within_error(result.weight[:3], msg.weight)
        assert self.equal_within_error(result.weight[3:], torch.ones([3, 4, 5, 4, 5, 6]) / (4 * 5 * 6))

        # Test 2: negative dim
        dim, length = -3, 6
        result = msg.batch_broaden(dim, length)

        # Check shape
        assert result.parameter.shape == Size([6, 4, 5, 1])
        assert result.weight.shape == Size([6, 4, 5, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter[:3], msg.parameter)
        assert self.equal_within_error(result.parameter[3:], torch.zeros([3, 4, 5, 1]))
        assert self.equal_within_error(result.weight[:3], msg.weight)
        assert self.equal_within_error(result.weight[3:], torch.ones([3, 4, 5, 4, 5, 6]) / (4 * 5 * 6))

    def test_batch_summarize(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dim = 0
        result = msg.batch_summarize(dim)

        # Check shape
        assert result.parameter.shape == Size([4, 5, 1])
        assert result.weight.shape == Size([4, 5, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.mean(dim, keepdim=False))
        weight_product = msg.weight[0]
        for i in range(1, b_shape[dim]):
            weight_product *= msg.weight[i]
        weight_product /= weight_product.sum(dim=[-1, -2, -3], keepdim=True)
        assert self.equal_within_error(result.weight, weight_product)

        # Test 2: negative dim
        neg_dim = -3
        result = msg.batch_summarize(neg_dim)

        # Check shape
        assert result.parameter.shape == Size([4, 5, 1])
        assert result.weight.shape == Size([4, 5, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.mean(dim, keepdim=False))
        weight_product = msg.weight[0]
        for i in range(1, b_shape[dim]):
            weight_product *= msg.weight[i]
        weight_product /= weight_product.sum(dim=[-1, -2, -3], keepdim=True)
        assert self.equal_within_error(result.weight, weight_product)

    def test_batch_flatten(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        # Test 1: positive dim
        dims = [0, 1]
        result = msg.batch_flatten(dims)

        # Check shape
        assert result.parameter.shape == Size([5, 12, 1])
        assert result.weight.shape == Size([5, 12, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.view(-1, 5, 1).permute(1, 0, 2))
        assert self.equal_within_error(result.weight, msg.weight.view(-1, 5, 4, 5, 6).permute(1, 0, 2, 3, 4))

        # Test 2: negative data
        dims = [-3, -2]
        result = msg.batch_flatten(dims)

        # Check shape
        assert result.parameter.shape == Size([5, 12, 1])
        assert result.weight.shape == Size([5, 12, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.view(-1, 5, 1).permute(1, 0, 2))
        assert self.equal_within_error(result.weight, msg.weight.view(-1, 5, 4, 5, 6).permute(1, 0, 2, 3, 4))

    def test_batch_flatten_default(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        result = msg.batch_flatten()

        # Check shape
        assert result.parameter.shape == Size([60, 1])
        assert result.weight.shape == Size([60, 4, 5, 6])

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.view(-1, 1))
        assert self.equal_within_error(result.weight, msg.weight.view(-1, 4, 5, 6))

    def test_batch_reshape(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        new_batch_shape = Size([2, 3, 10])
        result = msg.batch_reshape(new_batch_shape)

        # Check shape
        assert result.parameter.shape == new_batch_shape + p_shape
        assert result.weight.shape == new_batch_shape + s_shape

        # Check content
        assert self.equal_within_error(result.parameter, msg.parameter.reshape(new_batch_shape + p_shape))
        assert self.equal_within_error(result.weight, msg.weight.reshape(new_batch_shape + s_shape))

    def test_batch_expand(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 4, 1]), Size([1]), Size([4, 5, 6]), Size([1, 1, 1])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

        new_batch_shape = Size([3, -1, 5])
        result = msg.batch_expand(new_batch_shape)

        # Check shape
        assert result.parameter.shape == Size([3, 4, 5, 1])
        assert result.weight.shape == Size([3, 4, 5, 4, 5, 6])

        # Check content
        assert all(self.equal_within_error(result.parameter[i:i+1, :, :1, :], msg.parameter) for i in range(3))
        assert all(self.equal_within_error(result.parameter[:1, :, i:i+1, :], msg.parameter) for i in range(5))
        assert all(self.equal_within_error(result.weight[i:i+1, :, :1, :, :, :], msg.weight) for i in range(3))
        assert all(self.equal_within_error(result.weight[:1, :, i:i+1, :, :, :], msg.weight) for i in range(5))

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
        assert all(self.equal_within_error(rp, trans(p)) for p, rp in zip(msg.particles, result.particles))
        # Check log densities
        transformed_dens = [D.TransformedDistribution(d, [trans]).log_prob(trans(p)) for d, p in zip(dist, ptcl)]
        assert all(self.equal_within_error(rd, td) for rd, td in zip(result.log_densities, transformed_dens))

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
        assert all(self.equal_within_error(rp, trans(p)) for p, rp in zip(msg.particles, result.particles))
        # Check log densities
        transformed_dens = [D.TransformedDistribution(d, [trans]).log_prob(trans(p)) for d, p in zip(dist, ptcl)]
        assert all(self.equal_within_error(rd, td) for rd, td in zip(result.log_densities, transformed_dens))

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
        assert all(self.equal_within_error(rp, trans(p)) for p, rp in zip(msg.particles, result.particles))
        # Check log densities
        transformed_dens = [D.TransformedDistribution(dist, [trans]).log_prob(trans(p)).squeeze(-1) for p in ptcl]
        assert all(self.equal_within_error(rd, td) for rd, td in zip(result.log_densities, transformed_dens))

        expected_dens = [expected_dist.log_prob(tp).squeeze(-1) for tp in result.particles]
        assert all(self.equal_within_error(rd, ed) for rd, ed in zip(result.log_densities, expected_dens))

    def test_event_reweight(self):
        b_shape, p_shape, s_shape, e_shape = Size([4]), Size([1]), Size([4, 5, 6]), Size([1, 2, 3])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

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


        assert self.equal_within_error(result.weight, expected_weight)

    def test_event_marginalize(self):
        b_shape, p_shape, s_shape, e_shape = Size([10]), Size([1]), Size([4, 5, 6]), Size([1, 2, 3])
        msg = self.random_message(MessageType.Both, b_shape, p_shape, s_shape, e_shape)

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
        assert self.equal_within_error(result.weight, expected_weight)

        # test 2: negative dim
        event_dim = -2
        result = msg.event_marginalize(event_dim)

        # Check shape
        assert result.weight.shape == Size([10, 4, 6])
        assert result.particles[0].shape == Size([4, 1]) and result.particles[1].shape == Size([6, 3])
        assert result.log_densities[0].shape == Size([4]) and result.log_densities[1].shape == Size([6])

        # Check content
        assert self.equal_within_error(result.weight, expected_weight)


