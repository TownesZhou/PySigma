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
        ss = str(m1)
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
                    particles=torch.rand(10, 10),
                    weights=torch.Tensor([[0.1] * 10] * 10),
                    log_density=torch.rand(10, 10)
                    )
        assert m.size() == torch.Size([10, 10])

    def test_sxbxe_particles(self):
        m = Message(MessageType.Particles,
                    sample_shape=torch.Size([10]),
                    batch_shape=torch.Size([10]),
                    event_shape=torch.Size([10]),
                    particles=torch.rand(10, 10, 10),
                    weights=torch.Tensor([[0.1] * 10] * 10),
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

    def test_addition_parameter_edge(self):
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
                     particles=torch.ones(2, 3, 2),
                     weights=torch.Tensor([[0.1, 0.5, 0.7], [0.9, 0.5, 0.3]]),
                     log_density=torch.Tensor([[1, 2, 3], [4, 5, 6]])
                     )

        m2 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 3, 2),
                     weights=torch.Tensor([[0.2, 0.3, 0.4], [0.8, 0.7, 0.6]]),
                     log_density=torch.Tensor([[1, 2, 3], [4, 5, 6]])
                     )

        m3 = m1 + m2

    def test_multiplication_particle_int_float(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 3, 2),
                     weights=torch.Tensor([[0.1, 0.5, 0.7], [0.9, 0.5, 0.3]]),
                     log_density=torch.Tensor([[1, 2, 3], [4, 5, 6]])
                     )

        m2 = m1 * 5
        m3 = m2 * 2.4

    # TODO: continue testing this once fixed
    def test_multiplication_particle_tensor_singleton(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 3, 2),
                     weights=torch.Tensor([[0.1, 0.5, 0.7], [0.9, 0.5, 0.3]]),
                     log_density=torch.Tensor([[1, 2, 3], [4, 5, 6]])
                     )

        m2 = m1 * torch.Tensor([4])
        print()

    # def test_multiplication_particle_tensor_sameshape(self):
    #     pass


class TestMessageDimensionOperation:
    """
        Testing Message Operation for
        clone(), batch_permute(),
        batch_unsqueeze(), and batch_index_select()
    """

    def helper_normalize(self, weight):
        return weight / weight.sum(dim=0, keepdim=True)

    def test_clone(self):
        # The clone for parametric message and particle message only have
        # trivial difference, so only one is tested
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 3, 2),
                     weights=torch.Tensor([[0.1, 0.5, 0.7], [0.9, 0.5, 0.3]]),
                     log_density=torch.Tensor([[1, 2, 3], [4, 5, 6]])
                     )

        m2 = m1.clone()

        assert m2.same_size_as(m1)
        assert m2.parameters is None
        assert torch.all(m2.particles.eq(m1.particles))
        assert torch.all(m2.weights.eq(m1.weights))
        assert torch.all(m2.log_density.eq(m1.log_density))

    # TODO: test once fixed
    def test_parameter_batch_permute(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2, 3, 4]),
                     param_shape=torch.Size([1]),
                     parameters=torch.ones(2, 3, 4)
                     )
        m2 = m1.batch_permute([2, 0, 1])

    # TODO test once fixed
    def test_particles_batch_permute(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3,2,3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 3, 2, 3, 2),
                     weights=self.helper_normalize(torch.ones(2, 3, 2, 3)),
                     log_density=torch.ones(2, 3, 2, 3)
                     )

        m2 = m1.batch_permute([2,0,1])

    def test_parameter_batch_unsqueeze(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2, 3, 4]),
                     param_shape=torch.Size([1]),
                     parameters=torch.ones(2, 3, 4)
                     )

        m2 = m1.batch_unsqueeze(2)


    def test_particles_batch_unsqueeze(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 3, 2, 3, 2),
                     weights=self.helper_normalize(torch.ones(2, 3, 2, 3)),
                     log_density=torch.ones(2, 3, 2, 3)
                     )
        m2 = m1.batch_unsqueeze(2)

    # TODO test once fixed
    def test_particles_batch_unsqueeze_edge(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([1]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([1]),
                     particles=torch.ones(3, 2, 3),
                     weights=self.helper_normalize(torch.rand(3, 2, 3)),
                     log_density=torch.ones(3, 2, 3)
                     )
        m2 = m1.batch_unsqueeze(2)

    # TODO test once fixed
    def test_parameter_batch_index_select(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 4]),
                     param_shape=torch.Size([1]),
                     parameters=torch.rand(3, 4)
                     )

        m2 = m1.batch_index_select(0, torch.LongTensor([0, 2]))

    # TODO test once fixed
    def test_particles_batch_index_select(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 3, 2, 3, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 2, 3)),
                     log_density=torch.ones(2, 3, 2, 3)
                     )

        m2 = m1.batch_index_select(0, torch.LongTensor([0, 2]))