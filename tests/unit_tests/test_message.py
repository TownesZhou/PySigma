from pysigma.defs import Message, MessageType
import torch


class TestInitMessage:

    # is batch_shape necessary for parameter message?

    # torch.Size([]) + torch.Size([]) == torch.Size([])
    # torch.Tensor([]).shape == torch.Size([0]) != torch.Size([])
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
                     parameters=torch.Tensor([1, 2])
                     )

        m2 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([1]),
                     param_shape=torch.Size([2]),
                     parameters=torch.Tensor([2, 1])
                     )
        assert m1.size() == torch.Size([2])
        assert m2.size() == torch.Size([2])
        assert m1.same_size_as(m2)

    def test_bxp_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2]),
                     param_shape=torch.Size([2]),
                     parameters=torch.ones(2,2)
                     )

        m2 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3]),
                     param_shape=torch.Size([2]),
                     parameters=torch.Tensor([2, 1])
                     )
        assert m1.size() == torch.Size([2,2])
        assert m2.size() == torch.Size([2])
        assert m1.same_size_as(m2)

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
        pass

    def test_0x0xe_particles(self):
        pass

    def test_sxbx0_particles(self):
        pass

    def test_sx0xe_particles(self):
        pass

    def test_0xbxe_particles(self):
        pass

    def test_sxbxe_particles(self):
        pass

    def test_parameter_same_size(self):
        m = Message(MessageType.Parameter,
                    param_shape=torch.Size([2]),
                    batch_shape=torch.Size([2]),
                    parameters=torch.Tensor([1, 2])
                    )
        assert m.size() == torch.Size([2])


class TestMessageArithmetic:
    def test_addition(self):
        m1 = Message(MessageType.Parameter,
                     param_shape=torch.Size([1]),
                     batch_shape=torch.Size([2]),
                     parameters=torch.Tensor([1, 2])
                     )

        m2 = Message(MessageType.Parameter,
                     param_shape=torch.Size([1]),
                     batch_shape=torch.Size([2]),
                     parameters=torch.Tensor([1, 2])
                     )

        m1 + m2

    def test_multiplication(self):
        pass


class TestMessageDimensionOperation:
    def test_batch_permute(self):
        pass

    def test_batch_unsqueeze(self):
        pass

    def test_batch_index_select(self):
        pass
