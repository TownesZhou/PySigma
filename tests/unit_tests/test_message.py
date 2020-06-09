from pysigma.defs import Message, MessageType
import torch

class TestInitMessage:

    # is batch_shape necessary for parameter message?

    # torch.Size([]) + torch.Size([]) == torch.Size([])
    # torch.Tensor([]).shape == torch.Size([0]) != torch.Size([])
    def test_simple_parameter(self):
        m = Message(MessageType.Parameter,
                    param_shape=torch.Size([1]),
                    batch_shape=torch.Size([2]),
                    parameters=torch.Tensor([1,2])
        )

    def test_simple_particles(self):
        m = Message(MessageType.Particles,
                    sample_shape=torch.Size([100]),
                    batch_shape=torch.Size([1]),
                    event_shape=torch.Size([1]),
                    particles=torch.rand(100),
                    weights=torch.Tensor([0.01]*100),
                    log_density=torch.rand(100)
        )


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

        m1+m2

    def test_multiplication(self):
        pass


class TestMessageDimensionOperation:
    def test_batch_permute(self):
        pass

    def test_batch_unsqueeze(self):
        pass

    def test_batch_index_select(self):
        pass