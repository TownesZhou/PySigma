from pysigma.defs import Message, MessageType
import torch
import pytest

def message_size(b_shape, p_shape, s_shape, e_shape):
    if s_shape is not None:
        s_shape = torch.Size([s_shape]) if isinstance(s_shape, int) else torch.Size(s_shape)
    else:
        s_shape = torch.Size()
    if b_shape is not None:
        b_shape = torch.Size([b_shape]) if isinstance(b_shape, int) else torch.Size(b_shape)
    else:
        b_shape = torch.Size()
    if p_shape is not None:
        p_shape = torch.Size([p_shape]) if isinstance(p_shape, int) else torch.Size(p_shape)
    else:
        p_shape = torch.Size()
    if e_shape is not None:
        e_shape = torch.Size([e_shape]) if isinstance(e_shape, int) else torch.Size(e_shape)
    else:
        e_shape = torch.Size()
    return b_shape, p_shape, s_shape, e_shape

class TestInitMessage:
    """
        Testing initialization with arguments
        and the size() and same_size_as() functions
        cases includes combinations of different
        vector shapes for batch, param, sample, and event
    """

    def test_bx1_parameter_simple(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2]),
                     param_shape=torch.Size([1]),
                     parameters=torch.tensor([[2], [3]])
                     )

    def test_bx1_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2]),
                     param_shape=torch.Size([1]),
                     parameters=torch.ones(2).unsqueeze(-1)
                     )

        m2 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([4]),
                     param_shape=torch.Size([1]),
                     parameters=torch.ones(4).unsqueeze(-1)
                     )
        assert m1.size() == message_size(2, 1, None, None)
        assert m2.size() == message_size(4, 1, None, None)

        m3 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([4]),
                     param_shape=torch.Size([3]),
                     parameters=torch.ones(4, 3)
                     )

    def test_bxp_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2]),
                     param_shape=torch.Size([2]),
                     parameters=torch.ones(2, 2)
                     )

    def test_bxp_parameter_wrong_dimension(self):
        with pytest.raises(AssertionError):
            m = Message(MessageType.Parameter,
                         batch_shape=torch.Size([2]),
                         param_shape=torch.Size([2, 2]),
                         parameters=torch.ones(2, 2)
                         )


    def test_particles_wrong_dimension(self):
        with pytest.raises(AssertionError):
            m = Message(MessageType.Particles,
                        sample_shape=torch.Size([100]),
                        batch_shape=torch.Size([3]),
                        event_shape=torch.Size([1, 2]),
                        particles=torch.rand(100),
                        weights=1,
                        log_densities=0
                        )

        with pytest.raises(AssertionError):
            m = Message(MessageType.Particles,
                        sample_shape=torch.Size([100, 50]),
                        batch_shape=torch.Size([3, 4]),
                        event_shape=torch.Size([5]),
                        particles=torch.rand(100),
                        weights=1,
                        log_densities=0
                        )

    def test_0xbx0_particles(self):
        m = Message(MessageType.Particles,
                    sample_shape=torch.Size([1]),
                    batch_shape=torch.Size([10]),
                    event_shape=torch.Size([1]),
                    particles=torch.rand(1,1,1),
                    weights=torch.tensor([0.1] * 10).unsqueeze(0),
                    log_densities=torch.rand(1,1)
                    )
        assert m.size() == message_size(10, None, 1, 1)

    def test_sxbx0_particles(self):
        m = Message(MessageType.Particles,
                    sample_shape=torch.Size([10]),
                    batch_shape=torch.Size([10]),
                    event_shape=torch.Size([1]),
                    particles=torch.rand(1, 10, 1),
                    weights=torch.tensor([[0.1] * 10] * 10),
                    log_densities=torch.rand(1,10)
                    )
        assert m.size() == message_size(10, None, 10, 1)

    def test_sxbxe_particles(self):
        m = Message(MessageType.Particles,
                    sample_shape=torch.Size([10]),
                    batch_shape=torch.Size([10]),
                    event_shape=torch.Size([10]),
                    particles=torch.rand(1, 10, 10),
                    weights=torch.tensor([[0.1] * 10] * 10),
                    log_densities=torch.rand(1,10)
                    )
        assert m.size() == message_size(10, None, 10, 10)


class TestMessageArithmetic:
    """
        Testing Message addition and multiplication for
        parameter and particles with cases including:
        simple addition, multiply by int, float,
        singleton tensor, and tensors with same shape
    """
    def helper_normalize(self, weight):
        return weight / weight.sum(dim=0, keepdim=True)

    def test_addition_parameter(self):
        m1 = Message(MessageType.Parameter,
                     param_shape=torch.Size([1]),
                     batch_shape=torch.Size([2]),
                     parameters=torch.tensor([[1], [2]])
                     )

        m2 = Message(MessageType.Parameter,
                     param_shape=torch.Size([1]),
                     batch_shape=torch.Size([2]),
                     parameters=torch.tensor([[2], [3]])
                     )

        m3 = m1 + m2

        assert m3.size() == m2.size()
        # assert torch.all(m3.parameter.eq(torch.tensor([[3], [5]])))

    def test_addition_parameter_edge(self):
        m1 = Message(MessageType.Parameter,
                     param_shape=torch.Size([4]),
                     batch_shape=torch.Size([3, 3]),
                     parameters=torch.rand(3, 3, 4)
                     )

        m2 = Message(MessageType.Parameter,
                     param_shape=torch.Size([4]),
                     batch_shape=torch.Size([3, 3]),
                     parameters=torch.rand(3, 3, 4)
                     )

        m3 = m1 + m2

        assert m3.size() == m2.size()

    def test_addition_particles(self):
        matching_log_density = torch.rand(1,2)

        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(1, 2, 2),
                     weights=1,
                     log_densities=matching_log_density
                     )

        m2 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(1, 2, 2),
                     weights=1,
                     log_densities=matching_log_density
                     )

        m3 = m1 + m2

    def test_unmatch_log_density(self):
        with pytest.raises(AssertionError):
            m1 = Message(MessageType.Particles,
                         sample_shape=torch.Size([2]),
                         batch_shape=torch.Size([3]),
                         event_shape=torch.Size([2]),
                         particles=torch.ones(1, 2, 2),
                         weights=1,
                         log_densities=torch.rand(1,2)
                         )

            m2 = Message(MessageType.Particles,
                         sample_shape=torch.Size([2]),
                         batch_shape=torch.Size([3]),
                         event_shape=torch.Size([2]),
                         particles=torch.ones(1, 2, 2),
                         weights=1,
                         log_densities=torch.rand(1,2)
                         )

            m3 = m1 + m2


    def test_addition_particles_complex(self):
        matching_log_density = torch.rand(1, 2)
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 5]),
                     event_shape=torch.Size([4]),
                     particles=torch.ones(1, 2, 4),
                     weights=1,
                     log_densities=matching_log_density
                     )

        m2 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 5]),
                     event_shape=torch.Size([4]),
                     particles=torch.ones(1, 2, 4),
                     weights=1,
                     log_densities=matching_log_density
                     )

        m3 = m1 + m2

    def test_multiplication_parameter_int_float(self):
        m1 = Message(MessageType.Parameter,
                     param_shape=torch.Size([3]),
                     batch_shape=torch.Size([2,3]),
                     parameters=torch.rand(2, 3, 3)
                     )

        m2 = m1 * 5
        m3 = m2 * 2.4

    def test_multiplication_particle_int_float(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(1, 2, 2),
                     weights=torch.tensor([[0.1, 0.5, 0.7], [0.9, 0.5, 0.3]]),
                     log_densities=torch.rand(1,2)
                     )

        m2 = m1 * 5
        m3 = m2 * 2.4

    def test_multiplication_particle_int_float_high_dim(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3,4,5]),
                     event_shape=torch.Size([2]),
                     particles=torch.rand(1, 2, 2),
                     weights=1,
                     log_densities=torch.rand(1,2)
                     )

        m2 = m1 * 5
        m3 = m2 * 2.4

    def test_multiplication_particle_tensor_singleton(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(1, 2, 2),
                     weights=torch.tensor([[0.1, 0.5, 0.7], [0.9, 0.5, 0.3]]),
                     log_densities=torch.rand(1,2)
                     )

        m2 = m1 * torch.tensor(4)

    def test_multiplication_particle_tensor_sameshape(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(1, 2, 2),
                     weights=torch.tensor([[0.1, 0.5, 0.7], [0.9, 0.5, 0.3]]),
                     log_densities=torch.rand(1,2)
                     )

        t = torch.tensor([1, 2, 3])
        m2 = m1 * t

    def test_multiplication_particle_tensor_sameshape_uniform_weight(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(1, 2, 2),
                     weights=1,
                     log_densities=torch.rand(1,2)
                     )

        t = torch.tensor([1, 2, 3])
        m2 = m1 * t

    def test_multiplication_particle_tensor_sameshape_uniform_weight_edge(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([1]),
                     batch_shape=torch.Size([3]),
                     event_shape=torch.Size([1]),
                     particles=torch.ones(1, 1, 1),
                     weights=1,
                     log_densities=torch.rand(1,1)
                     )

        t = torch.tensor([1, 2, 3])
        m2 = m1 * t


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
                     particles=torch.ones(1, 2, 2),
                     weights=torch.tensor([[0.1, 0.5, 0.7], [0.9, 0.5, 0.3]]),
                     log_densities=torch.rand(1,2)
                     )

        m2 = m1.clone()

        assert m2.same_size_as(m1)
        assert m2.parameters is None
        assert torch.all(m2.particles.eq(m1.particles))
        assert torch.all(m2.weights.eq(m1.weights))
        assert torch.all(m2.log_densities.eq(m1.log_densities))

    def test_parameter_batch_permute(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2, 3, 4]),
                     param_shape=torch.Size([2]),
                     parameters=torch.ones(2, 3, 4, 2)
                     )
        m2 = m1.batch_permute([2, 0, 1])

    def test_particles_batch_permute(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3,2,3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.ones(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )

        m2 = m1.batch_permute([2,0,1])
        assert m2.s_shape == torch.Size([2])
        assert m2.b_shape == torch.Size([3, 3, 2])
        assert m2.e_shape == torch.Size([2])

    def test_parameter_batch_unsqueeze(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2, 3, 4]),
                     param_shape=torch.Size([1]),
                     parameters=torch.ones(2, 3, 4, 1)
                     )

        m2 = m1.batch_unsqueeze(2)
        assert m2.p_shape == torch.Size([1])
        assert m2.b_shape == torch.Size([2, 3, 1, 4])


    def test_particles_batch_unsqueeze(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.ones(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )
        m2 = m1.batch_unsqueeze(3)
        assert m2.s_shape == torch.Size([2])
        assert m2.b_shape == torch.Size([3, 2, 3, 1])
        assert m2.e_shape == torch.Size([2])

    def test_particles_batch_unsqueeze_edge(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([1]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([1]),
                     particles=torch.ones(1, 1),
                     weights=self.helper_normalize(torch.ones(1, 3, 2, 3)),
                     log_densities=torch.ones(1)
                     )
        m2 = m1.batch_unsqueeze(3)
        assert m2.s_shape == torch.Size([1])
        assert m2.b_shape == torch.Size([3, 2, 3, 1])
        assert m2.e_shape == torch.Size([1])

    def test_particles_batch_unsqueeze_zero_dim(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.ones(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )
        m2 = m1.batch_unsqueeze(0)
        assert m2.s_shape == torch.Size([2])
        assert m2.b_shape == torch.Size([1, 3, 2, 3])
        assert m2.e_shape == torch.Size([2])

    def test_particles_batch_unsqueeze_negative_dim(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.ones(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )
        m2 = m1.batch_unsqueeze(-1)
        assert m2.s_shape == torch.Size([2])
        assert m2.b_shape == torch.Size([3, 2, 3, 1])
        assert m2.e_shape == torch.Size([2])

    def test_particles_batch_unsqueeze_edge_2(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([1]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([1]),
                     particles=torch.ones(1, 1),
                     weights=self.helper_normalize(torch.rand(1, 3, 2, 3)),
                     log_densities=torch.ones(1)
                     )
        m2 = m1.batch_unsqueeze(2)
        assert m2.s_shape == torch.Size([1])
        assert m2.b_shape == torch.Size([3, 2, 1, 3])
        assert m2.e_shape == torch.Size([1])

    def test_parameter_batch_index_select(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 4]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(3, 4, 2)
                     )

        m2 = m1.batch_index_select(0, torch.LongTensor([0, 2]))

    def test_particles_batch_index_select(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )

        m2 = m1.batch_index_select(0, torch.LongTensor([0, 2]))

    def test_index_put_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 3]),
                     param_shape=torch.Size([1]),
                     parameters=torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32).unsqueeze(-1)
                     )

        m2 = m1.batch_index_put(0, torch.LongTensor([2, 0, 5]))
        assert m2.size() == message_size(None, [6, 3], 1, None)

    def test_index_put_parameter_more(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 6]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(3, 6, 2)
                     )

        m2 = m1.batch_index_put(0, torch.LongTensor([0, 2, 7]))
        assert m2.size() == message_size(None, [8, 6], 2, None)

        m3 = m1.batch_index_put(1, torch.LongTensor([4, 1, 8, 7, 5, 9]))
        assert m3.size() == message_size(None, [3, 10], 2, None)

    def test_index_put_particles(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )

        m2 = m1.batch_index_put(0, torch.LongTensor([0, 2, 3]))
        assert m2.size() == message_size(2, [4, 2, 3], None, 2)

    def test_batch_diagonal_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 4]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(3, 4, 2)
                     )

        m2 = m1.batch_diagonal()
        assert m2.size() == message_size(None, 3, 2, None)

        m3 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([4, 3]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(4, 3, 2)
                     )

        m4 = m3.batch_diagonal()
        assert m2.size() == message_size(None, 3, 2, None)

    def test_batch_diagonal_parameter_edge(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 4]),
                     param_shape=torch.Size([1]),
                     parameters=torch.rand(3, 4, 1)
                     )

        m2 = m1.batch_diagonal()
        assert m2.size() == message_size(None, 3, 1, None)

    def test_batch_diagonal_particle(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )

        m2 = m1.batch_diagonal(1, 2)

    def test_batch_diag_embed_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 4]),
                     param_shape=torch.Size([1]),
                     parameters=torch.rand(3, 4, 1)
                     )

        m2 = m1.batch_diag_embed()
        m3 = m1.batch_diag_embed(0)
        m4 = m1.batch_diag_embed(1, 1, 2)

    def test_batch_diag_embed_particles(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )

        m2 = m1.batch_diag_embed()
        m3 = m1.batch_diag_embed(0)
        m4 = m1.batch_diag_embed(1, 1, 2)

    def test_batch_narrow_parameter(self):
        para = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 3]),
                     param_shape=torch.Size([1]),
                     parameters=para.unsqueeze(-1)
                     )

        m2 = m1.batch_narrow(0, 2)
        assert m2.size() == message_size(None, [2, 3], 1, None)
        assert torch.equal(m2.parameters.squeeze(-1), torch.narrow(para, 0, 0, 2))

        m3 = m1.batch_narrow(1, 2)
        assert m3.size() == message_size(None, [3, 2], 1, None)

    def test_batch_narrow_particles(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )
        m2 = m1.batch_narrow(0, 2)
        assert m2.size() == message_size(2, [2, 2, 3], None, 2)
        m3 = m1.batch_narrow(2, 1)
        assert m3.size() == message_size(2, [3, 2, 1], None, 2)

    def test_batch_broaden_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 3]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(3, 3, 2)
                     )

        m2 = m1.batch_broaden(0, 5)
        assert m2.size() == message_size(None, [5, 3], 2, None)

    def test_batch_broaden_parameter_edge(self):
        para = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 3]),
                     param_shape=torch.Size([1]),
                     parameters=para.unsqueeze(-1)
                     )

        m2 = m1.batch_broaden(0, 5)
        assert m2.size() == message_size(None, [5, 3], 1, None)

    def test_batch_broaden_particles(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )
        m2 = m1.batch_broaden(1, 5)
        assert m2.size() == message_size(2, [3, 5, 3], None, 2)

        m2 = m1.batch_broaden(-3, 5)
        assert m2.size() == message_size(2, [5, 2, 3], None, 2)

    def test_batch_summarize_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 4]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(3, 4, 2)
                     )

        m2 = m1.batch_summarize(1)
        assert m2.size() == message_size(None, 3, 2, None)

        m3 = m1.batch_summarize(-1)
        assert m3.size() == message_size(None, 3, 2, None)

        m4 = m1.batch_summarize(0)
        assert m4.size() == message_size(None, 4, 2, None)

    def test_batch_summarize_particles(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )

        m2 = m1.batch_summarize(0)
        assert m2.size() == message_size(2, [2, 3], None, 2)

        m3 = m1.batch_summarize(2)
        assert m3.size() == message_size(2, [3, 2], None, 2)

        m4 = m1.batch_summarize(-1)
        assert m4.size() == message_size(2, [3, 2], None, 2)

    def test_batch_flatten_parameter_default(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 3]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(3, 3, 2)
                     )

        m2 = m1.batch_flatten(None)
        assert m2.size() == message_size(None, 9, 2, None)

        m3 = m2.batch_flatten(None)
        assert m3.size() == message_size(None, 9, 2, None)

    def test_batch_flatten_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([2, 3]),
                     param_shape=torch.Size([5]),
                     parameters=torch.rand(2, 3, 5)
                     )

        m2 = m1.batch_flatten([0])
        assert m2.size() == message_size(None, [3, 2], 5, None)

    def test_batch_flatten_parameter_no_zero(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 3, 4, 5]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(3, 3, 4, 5, 2)
                     )

        m2 = m1.batch_flatten([1, 2, 3])
        assert m2.size() == message_size(None, [3, 60], 2, None)

        m3 = m1.batch_flatten([-2, -3])
        assert m3.size() == message_size(None, [3, 5, 12], 2, None)

    def test_batch_flatten_particles(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 2, 3]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 2, 3)),
                     log_densities=torch.ones(2)
                     )

        m2 = m1.batch_flatten(None)
        assert m2.size() == message_size(2, 18, None, 2)

        m3 = m1.batch_flatten([0])
        assert m3.size() == message_size(2, [2, 3, 3], None, 2)

        m4 = m1.batch_flatten([1, 2])
        assert m4.size() == message_size(2, [3, 6], None, 2)

    def test_batch_reshape_parameter(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 3, 4, 5]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(3, 3, 4, 5, 2)
                     )

        m2 = m1.batch_reshape([4, 5, 9])
        assert m2.size() == message_size(None, [4, 5, 9], 2, None)

        m3 = m1.batch_reshape(torch.Size([180]))
        assert m3.size() == message_size(None, 180, 2, None)

        m4 = m1.batch_reshape(torch.Size([3, 5, 4, 3, 1]))
        assert m4.size() == message_size(None, [3, 5, 4, 3, 1], 2, None)

        m5 = m1.batch_reshape([1, 3, 2, 1, 3, 5, 2, 1])
        assert m5.size() == message_size(None, [1, 3, 2, 1, 3, 5, 2, 1], 2, None)

    def test_batch_reshape_particles(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([2, 6, 3, 4]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 2, 6, 3, 4)),
                     log_densities=torch.ones(2)
                     )

        m2 = m1.batch_reshape([2, 3, 4, 6, 1])
        assert m2.size() == message_size(2, [2, 3, 4, 6, 1], None, 2)

        m3 = m1.batch_reshape([12, 12])
        assert m3.size() == message_size(2, [12, 12], None, 2)

        m4 = m1.batch_reshape([144])
        assert m4.size() == message_size(2, 144, None, 2)

        m5 = m1.batch_reshape([1, 2, 3, 2, 3, 1, 2, 2, 1])
        assert m5.size() == message_size(2, [1, 2, 3, 2, 3, 1, 2, 2, 1], None, 2)

    def test_batch_expand_parameters(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 1]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(3, 1, 2)
                     )

        m2 = m1.batch_expand([3, 4])
        assert m2.size() == message_size(None, [3, 4], 2, None)

    def test_batch_expand_parameters_negative_1(self):
        m1 = Message(MessageType.Parameter,
                     batch_shape=torch.Size([3, 1]),
                     param_shape=torch.Size([2]),
                     parameters=torch.rand(3, 1, 2)
                     )

        m2 = m1.batch_expand([-1, 4])
        assert m2.size() == message_size(None, [3, 4], 2, None)

    def test_batch_expand_particles(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 1]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 1)),
                     log_densities=torch.ones(2)
                     )

        m2 = m1.batch_expand([3, 4])
        assert m2.size() == message_size(2, [3, 4], None, 2)

    def test_batch_expand_particles_negative_1(self):
        m1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3, 1]),
                     event_shape=torch.Size([2]),
                     particles=torch.ones(2, 2),
                     weights=self.helper_normalize(torch.rand(2, 3, 1)),
                     log_densities=torch.ones(2)
                     )

        m2 = m1.batch_expand([-1, 4])
        assert m2.size() == message_size(2, [3, 4], None, 2)