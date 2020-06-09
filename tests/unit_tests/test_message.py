from pysigma.defs import Message, MessageType
import torch


class TestInitMessage:
    """
        Testing initialization with arguments
        and the size() and same_size_as() functions
        cases includes combinations of different
        vector shapes for batch, param, sample, and event
    """

    def test_bx0_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2]),
                     param_shape=torch.Size([1]),
                     parameters=torch.ones(2)
                     )

        m2 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([4]),
                     param_shape=torch.Size([1]),
                     parameters=torch.ones(4)
                     )
        assert m1.size() == torch.Size([2])
        assert m2.size() == torch.Size([4])

    def test_0xp_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([1]),
                     param_shape=torch.Size([2]),
                     parameters=torch.ones(2)
                     )

        m2 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([1]),
                     param_shape=torch.Size([2]),
                     parameters=torch.zeros(2)
                     )
        assert m1.size() == torch.Size([2])
        assert m2.size() == torch.Size([2])
        assert m1.same_size_as(m2)

    def test_bxp_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2]),
                     param_shape=torch.Size([2]),
                     parameters=torch.ones(2, 2)
                     )

        m2 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3]),
                     param_shape=torch.Size([2]),
                     parameters=torch.zeros(3, 2)
                     )
        assert m1.size() == torch.Size([2, 2])
        assert m2.size() == torch.Size([3, 2])
        assert not m1.same_size_as(m2)

    def test_sx0x0_particles(self):
        m = Message(MessageType.Particles,
                    sample_shape=torch.Size([100]),
                    batch_shape=torch.Size([1]),
                    event_shape=torch.Size([1]),
                    particles=torch.rand(100),
                    weights=torch.Tensor([0.01] * 100),
                    log_density=torch.rand(100)
                    )
        assert m.size() == torch.Size([100])

    def test_0xbx0_particles(self):
        m = Message(MessageType.Particles,
                    sample_shape=torch.Size([1]),
                    batch_shape=torch.Size([10]),
                    event_shape=torch.Size([1]),
                    particles=torch.rand(10),
                    weights=torch.Tensor([0.1] * 10),
                    log_density=torch.rand(10)
                    )
        assert m.size() == torch.Size([10])

    # TODO: Fix this issue
    def test_0x0xe_particles(self):
        m = Message(MessageType.Particles,
                    sample_shape=torch.Size([1]),
                    batch_shape=torch.Size([1]),
                    event_shape=torch.Size([10]),
                    particles=torch.rand(10),
                    weights=torch.ones(1),
                    log_density=torch.rand(1)
                    )
        assert m.size() == torch.Size([10])

    def test_sxbx0_particles(self):
        m = Message(MessageType.Particles,
                    sample_shape=torch.Size([10]),
                    batch_shape=torch.Size([10]),
                    event_shape=torch.Size([1]),
                    particles=torch.rand(10,10),
                    weights=torch.Tensor([[0.1] * 10]*10),
                    log_density=torch.rand(10, 10)
                    )
        assert m.size() == torch.Size([10, 10])

    def test_sxbxe_particles(self):
        m = Message(MessageType.Particles,
                    sample_shape=torch.Size([10]),
                    batch_shape=torch.Size([10]),
                    event_shape=torch.Size([10]),
                    particles=torch.rand(10, 10, 10),
                    weights=torch.Tensor([[0.1] * 10]*10),
                    log_density=torch.rand(10, 10)
                    )
        assert m.size() == torch.Size([10, 10, 10])


class TestMessageArithmetic:
    """
        Testing Message addition and multiplication for
        parameter and particles with cases including:
        simple addition, multiply by int, float,
        singleton tensor, and tensors with same shape
    """

    def test_addition_parameter(self):
        m1 = Message(MessageType.Parameter,
                     param_shape=torch.Size([1]),
                     batch_shape=torch.Size([2]),
                     parameters=torch.Tensor([1, 2])
                     )

        m2 = Message(MessageType.Parameter,
                     param_shape=torch.Size([1]),
                     batch_shape=torch.Size([2]),
                     parameters=torch.Tensor([2, 3])
                     )

        m3 = m1 + m2

        assert m3.size() == m2.size()
        assert torch.all(m3.parameters.eq(torch.Tensor([3, 5])))

    def test_addition_particles(self):
        m1 = Message(MessageType.Particles,
                    sample_shape=torch.Size([2]),
                    batch_shape=torch.Size([3]),
                    event_shape=torch.Size([2]),
                    particles=torch.ones(2,3,2),
                    weights=torch.Tensor([[0.5] * 3] * 2),
                    log_density=torch.Tensor([[1,2, 3],[4,5,6]])
                    )


    def test_multiplication_particle_int(self):
        m1 = Message(MessageType.Particles,
                    sample_shape=torch.Size([2]),
                    batch_shape=torch.Size([3]),
                    event_shape=torch.Size([2]),
                    particles=torch.ones(2,3,2),
                    weights=torch.Tensor([[0.5] * 3] * 2),
                    log_density=torch.Tensor([[1,2, 3],[4,5,6]])
                    )

        m2 = m1 * 5
        print()

    def test_multiplication_particle_float(self):
        pass

    def test_multiplication_particle_tensor_singleton(self):
        pass

    def test_multiplication_particle_tensor_sameshape(self):
        pass


class TestMessageDimensionOperation:
    def test_batch_permute(self):
        pass

    def test_batch_unsqueeze(self):
        pass

    def test_batch_index_select(self):
        pass
