"""
    Unit tests for WMVN class
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, FactorNode
from pysigma.graphical.predicate_nodes import WMVN
from pysigma.utils import KnowledgeServer as KS

from ...utils import generate_positive_definite


class TestWMVN():

    # Utility method
    def generate_wmvn_1(self, b_shape=Size([4, 5]), p_shape=Size([2]), s_shape=Size([10, 15, 20]),
                        e_shape=Size([1, 2, 3])):
        # Use D.Distribution as stand-in distribution class for KnowledgeServer. In this config most KnowledgeServer
        #   methods would not work.
        name = "test_wmvn"

        dist_class = D.Distribution
        rv_sizes = list(e_shape)
        rv_cstr = [C.real, ] * len(e_shape)
        rv_num_ptcl = list(s_shape)
        ks = KS(dist_class, rv_sizes, rv_cstr, rv_num_ptcl)

        rel_var_list = [Variable("rel_" + str(i), VariableMetatype.Relational, b) for i, b in enumerate(b_shape)]
        param_var = Variable("param", VariableMetatype.Parameter, p_shape[0])
        index_var_list = [Variable("index_" + str(i), VariableMetatype.Indexing, s) for i, s in enumerate(s_shape)]
        ran_var_list = [Variable("ran_" + str(i), VariableMetatype.Random, e, [rv_cstr[i]])
                        for i, e in enumerate(e_shape)]

        wmvn = WMVN(name, ks, rel_var_list, param_var, index_var_list, ran_var_list)

        return wmvn

    def generate_wmvn_2(self, b_shape=Size([4, 5]), p_shape=Size([42]), s_shape=Size([10, 15, 20]),
                        e_shape=Size([1, 2, 3])):
        # Use a real distribution: MultivariateNormal
        name = "test_wmvn"

        dist_class = D.MultivariateNormal
        rv_sizes = list(e_shape)
        rv_cstr = [C.real, ] * len(e_shape)
        rv_num_ptcl = list(s_shape)
        ks = KS(dist_class, rv_sizes, rv_cstr, rv_num_ptcl)

        rel_var_list = [Variable("rel_" + str(i), VariableMetatype.Relational, b) for i, b in enumerate(b_shape)]
        param_var = Variable("param", VariableMetatype.Parameter, p_shape[0])
        index_var_list = [Variable("index_" + str(i), VariableMetatype.Indexing, s) for i, s in enumerate(s_shape)]
        ran_var_list = [Variable("ran_" + str(i), VariableMetatype.Random, e, [rv_cstr[i]])
                        for i, e in enumerate(e_shape)]

        wmvn = WMVN(name, ks, rel_var_list, param_var, index_var_list, ran_var_list)

        return wmvn

    def test_init(self):
        b_shape, p_shape, s_shape, e_shape = Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3])

        name = "test_wmvn"

        dist_class = D.Distribution
        rv_sizes = list(e_shape)
        rv_cstr = [C.real,] * len(e_shape)
        rv_num_ptcl = list(s_shape)
        ks = KS(dist_class, rv_sizes, rv_cstr, rv_num_ptcl)

        rel_var_list = [Variable("rel_" + str(i), VariableMetatype.Relational, b) for i, b in enumerate(b_shape)]
        param_var = Variable("param", VariableMetatype.Parameter, p_shape[0])
        index_var_list = [Variable("index_" + str(i), VariableMetatype.Indexing, s) for i, s in enumerate(s_shape)]
        ran_var_list = [Variable("ran_" + str(i), VariableMetatype.Random, e, [rv_cstr[i]])
                        for i, e in enumerate(e_shape)]

        wmvn = WMVN(name, ks, rel_var_list, param_var, index_var_list, ran_var_list)

        assert wmvn.ks is ks
        assert wmvn._cache == {}

    def test_compute_quiescence_mock_ld(self):
        # Test WMVN quiescence behavior using mocked linkdata
        mock_in_ld, mock_out_ld = MagicMock(spec_set=LinkData), MagicMock(spec_set=LinkData)
        mock_msg = MagicMock(spec_set=Message)
        mock_in_ld.read.return_value = mock_msg

        wmvn = self.generate_wmvn_1()
        wmvn.in_linkdata.append(mock_in_ld)
        wmvn.out_linkdata.append(mock_out_ld)

        # Patch wmvn's quiescence state
        with patch("pysigma.graphical.basic_nodes.Node.quiescence", new_callable=PropertyMock) as mock_quiescence:
            mock_quiescence.return_value = True

            wmvn.compute()
            mock_in_ld.read.assert_not_called()
            mock_out_ld.write.assert_not_called()

    def test_compute_single_in_ld_mock_ld_1(self):
        # Test correct message propagation with a single incoming linkdata using mocked linkdata
        # Scenario 1: in_ld and out_ld are connected to different factor nodes
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        fn_1, fn_2 = MagicMock(spec_set=FactorNode), MagicMock(spec_set=FactorNode)
        mock_msg = MagicMock(spec_set=Message)
        mock_in_ld.read.return_value = mock_msg
        mock_in_ld.new = True
        mock_in_ld.fn = fn_1
        mock_out_ld.fn = fn_2

        wmvn = self.generate_wmvn_1()
        wmvn.in_linkdata.append(mock_in_ld)
        wmvn.out_linkdata.append(mock_out_ld)

        wmvn.compute()

        # Check call arguments
        mock_out_ld.write.assert_called_once_with(mock_msg)

    # Ignore warnings that contains 'WMVN' substring
    @pytest.mark.filterwarnings("ignore:WMVN")
    def test_compute_single_in_ld_mock_ld_2(self):
        # Test correct message propagation with a single incoming linkdata using mocked linkdata
        # Scenario 2: in_ld and out_ld are connected to the same factor nodes. In this case, the message should not be
        #   propagated
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        fn_1= MagicMock(spec=FactorNode)
        fn_1.name = "mock_FactorNode"
        mock_msg = MagicMock(spec_set=Message)
        mock_in_ld.read.return_value = mock_msg
        mock_in_ld.new = True
        mock_in_ld.fn = fn_1
        mock_out_ld.fn = fn_1

        wmvn = self.generate_wmvn_1()
        wmvn.in_linkdata.append(mock_in_ld)
        wmvn.out_linkdata.append(mock_out_ld)

        wmvn.compute()

        # Check call arguments
        mock_out_ld.write.assert_not_called()

    def test_compute_combination_all_parameters_mock_ld(self):
        # Test message combination with all parameter messages using mocked linkdata
        b_shape, p_shape = Size([4, 5]), Size([2])
        num_in_ld = 3
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        in_msg_list = [Message(MessageType.Parameter, batch_shape=b_shape, param_shape=p_shape,
                               parameter=torch.randn(b_shape + p_shape)) for i in range(num_in_ld)]
        for mock_in_ld, in_msg in zip(mock_in_ld_list, in_msg_list):
            mock_in_ld.new = True
            mock_in_ld.fn = MagicMock(spec_set=FactorNode)
            mock_in_ld.read.return_value = in_msg
        mock_oud_ld = MagicMock(spec=LinkData)
        mock_oud_ld.fn = MagicMock(spec_set=FactorNode)

        wmvn = self.generate_wmvn_1()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata.append(mock_oud_ld)

        wmvn.compute()

        # Check call and call arguments
        expected_out_msg = sum(in_msg_list, Message.identity())
        mock_oud_ld.write.assert_called_once()
        assert mock_oud_ld.write.call_args[0][0] == expected_out_msg

    def test_compute_combination_all_particles_mock_ld(self):
        # Test message combination with all particles messages using mocked linkdata
        b_shape, s_shape, e_shape = Size([4, 5]), Size([10, 15, 20]), Size([1, 2, 3])
        num_in_ld = 3
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        ptcl, dens = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)], [torch.randn([s]) for s in s_shape]
        in_msg_list = [Message(MessageType.Particles,
                               batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                               weight=torch.rand(b_shape + s_shape),
                               particles=ptcl, log_densities=dens) for i in range(num_in_ld)]
        for mock_in_ld, in_msg in zip(mock_in_ld_list, in_msg_list):
            mock_in_ld.new = True
            mock_in_ld.fn = MagicMock(spec_set=FactorNode)
            mock_in_ld.read.return_value = in_msg
        mock_oud_ld = MagicMock(spec=LinkData)
        mock_oud_ld.fn = MagicMock(spec_set=FactorNode)

        wmvn = self.generate_wmvn_1()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata.append(mock_oud_ld)

        wmvn.compute()

        # Check call and call arguments
        expected_out_msg = sum(in_msg_list, Message.identity())
        mock_oud_ld.write.assert_called_once()
        assert mock_oud_ld.write.call_args[0][0] == expected_out_msg

    def test_compute_combination_all_both_mock_ld(self):
        # Test message combination with all Both type messages using mocked linkdata
        b_shape, p_shape, s_shape, e_shape = Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3])
        num_in_ld = 3
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        ptcl, dens = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)], [torch.randn([s]) for s in s_shape]
        in_msg_list = [Message(MessageType.Particles,
                               batch_shape=b_shape, param_shape=p_shape, sample_shape=s_shape, event_shape=e_shape,
                               parameter=torch.randn(b_shape + p_shape), weight=torch.rand(b_shape + s_shape),
                               particles=ptcl, log_densities=dens) for i in range(num_in_ld)]
        for mock_in_ld, in_msg in zip(mock_in_ld_list, in_msg_list):
            mock_in_ld.new = True
            mock_in_ld.fn = MagicMock(spec_set=FactorNode)
            mock_in_ld.read.return_value = in_msg
        mock_oud_ld = MagicMock(spec=LinkData)
        mock_oud_ld.fn = MagicMock(spec_set=FactorNode)

        wmvn = self.generate_wmvn_1()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata.append(mock_oud_ld)

        wmvn.compute()

        # Check call and call arguments
        expected_out_msg = sum(in_msg_list, Message.identity())
        mock_oud_ld.write.assert_called_once()
        assert mock_oud_ld.write.call_args[0][0] == expected_out_msg

    def test_compute_combination_parameter_particles_mock_ld_1(self):
        # Test message combination with particles and parameters messages using mocked linkdata
        # Two LinData: 1 sends parameter message and the other one sends particles message.
        # Using MultivariateNormal distribution as example
        b_shape, s_shape, e_shape = Size([4, 5]), Size([10, 15, 20]), Size([1, 2, 3])
        e_size = sum(e_shape)
        p_shape = Size([e_size * (e_size + 1)])

        num_in_ld = 2
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        # Particles message data
        ptcl, dens = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)], [torch.randn([s]) for s in s_shape]
        weight = torch.rand(b_shape + s_shape)
        # Parameter message data
        loc = torch.randn(b_shape + Size([e_size]))
        cov = generate_positive_definite(b_shape, e_size)
        param = torch.cat([loc, cov.view(b_shape + Size([-1]))], dim=-1)
        in_msg_list = [Message(MessageType.Parameter,
                               batch_shape=b_shape, param_shape=p_shape,
                               parameter=param),
                       Message(MessageType.Particles,
                               batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                               weight=weight, particles=ptcl, log_densities=dens)]
        # Set up messages
        for mock_in_ld, in_msg in zip(mock_in_ld_list, in_msg_list):
            mock_in_ld.new = True
            mock_in_ld.fn = MagicMock(spec_set=FactorNode)
            mock_in_ld.read.return_value = in_msg
        mock_oud_ld = MagicMock(spec=LinkData)
        mock_oud_ld.fn = MagicMock(spec_set=FactorNode)

        # Use KS with MultivariateNormal distribution class
        wmvn = self.generate_wmvn_2()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata.append(mock_oud_ld)

        wmvn.compute()

        # Check call and arguments
        mock_oud_ld.write.assert_called_once()
        # Check message type
        received_msg = mock_oud_ld.write.call_args[0][0]
        assert received_msg.type is MessageType.Particles

    def test_compute_combination_parameter_particles_mock_ld_2(self):
        # Test message combination with particles and parameters messages using mocked linkdata
        # Multiple LinData: multiple parameter message and the multiple particles message.
        # Using MultivariateNormal distribution as example
        b_shape, s_shape, e_shape = Size([4, 5]), Size([10, 15, 20]), Size([1, 2, 3])
        e_size = sum(e_shape)
        p_shape = Size([e_size * (e_size + 1)])

        num_param_in_ld, num_ptcl_in_ld = 3, 5
        num_in_ld = num_param_in_ld + num_param_in_ld
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]

        in_msg_list = []
        # Make parameter messages
        for i in range(num_param_in_ld):
            loc = torch.randn(b_shape + Size([e_size]))
            cov = generate_positive_definite(b_shape, e_size)
            param = torch.cat([loc, cov.view(b_shape + Size([-1]))], dim=-1)
            in_msg_list.append(Message(MessageType.Parameter,
                                       batch_shape=b_shape, param_shape=p_shape,
                                       parameter=param))
        # Make particles messages
        ptcl, dens = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)], [torch.randn([s]) for s in s_shape]
        for j in range(num_ptcl_in_ld):
            in_msg_list.append(Message(MessageType.Particles,
                                       batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                                       weight=torch.rand(b_shape + s_shape), particles=ptcl, log_densities=dens))

        # Set up messages
        for mock_in_ld, in_msg in zip(mock_in_ld_list, in_msg_list):
            mock_in_ld.new = True
            mock_in_ld.fn = MagicMock(spec_set=FactorNode)
            mock_in_ld.read.return_value = in_msg
        mock_oud_ld = MagicMock(spec=LinkData)
        mock_oud_ld.fn = MagicMock(spec_set=FactorNode)

        # Use KS with MultivariateNormal distribution class
        wmvn = self.generate_wmvn_2()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata.append(mock_oud_ld)

        wmvn.compute()

        # Check call and arguments
        mock_oud_ld.write.assert_called_once()
        # Check message type
        received_msg = mock_oud_ld.write.call_args[0][0]
        assert received_msg.type is MessageType.Particles

    def test_compute_combination_parameter_both_mock_ld(self):
        # Test message combination with Parameter and Both type messages using mocked linkdata
        # Two LinkData: one sends Parameter type message and the other sends Both type message
        # Using MultivariateNormal distribution class as example
        b_shape, s_shape, e_shape = Size([4, 5]), Size([10, 15, 20]), Size([1, 2, 3])
        e_size = sum(e_shape)
        p_shape = Size([e_size * (e_size + 1)])

        num_in_ld = 2
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        # Particles message data
        ptcl, dens = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)], [torch.randn([s]) for s in s_shape]
        weight = torch.rand(b_shape + s_shape)
        # Parameter message data
        loc = torch.randn(b_shape + Size([e_size]))
        cov = generate_positive_definite(b_shape, e_size)
        param = torch.cat([loc, cov.view(b_shape + Size([-1]))], dim=-1)
        in_msg_list = [Message(MessageType.Parameter,
                               batch_shape=b_shape, param_shape=p_shape,
                               parameter=param),
                       Message(MessageType.Both,
                               batch_shape=b_shape, param_shape=p_shape, sample_shape=s_shape, event_shape=e_shape,
                               parameter=param,
                               weight=weight, particles=ptcl, log_densities=dens)]
        # Set up messages
        for mock_in_ld, in_msg in zip(mock_in_ld_list, in_msg_list):
            mock_in_ld.new = True
            mock_in_ld.fn = MagicMock(spec_set=FactorNode)
            mock_in_ld.read.return_value = in_msg
        mock_oud_ld = MagicMock(spec=LinkData)
        mock_oud_ld.fn = MagicMock(spec_set=FactorNode)

        # Use KS with MultivariateNormal distribution class
        wmvn = self.generate_wmvn_2()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata.append(mock_oud_ld)

        wmvn.compute()

        # Check call and arguments
        mock_oud_ld.write.assert_called_once()
        # Check message type
        received_msg = mock_oud_ld.write.call_args[0][0]
        assert received_msg.type is MessageType.Both

    def test_compute_combination_particles_both_mock_ld(self):
        # Test message combination with Particles and Both type messages using mocked linkdata
        # Two LinkData: one sends Particles type message and the other sends Both type message
        # Using MultivariateNormal distribution class as example
        b_shape, s_shape, e_shape = Size([4, 5]), Size([10, 15, 20]), Size([1, 2, 3])
        e_size = sum(e_shape)
        p_shape = Size([e_size * (e_size + 1)])

        num_in_ld = 2
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        # Particles message data
        ptcl, dens = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)], [torch.randn([s]) for s in s_shape]
        # Parameter message data
        loc = torch.randn(b_shape + Size([e_size]))
        cov = generate_positive_definite(b_shape, e_size)
        param = torch.cat([loc, cov.view(b_shape + Size([-1]))], dim=-1)
        in_msg_list = [Message(MessageType.Particles,
                               batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                               weight=torch.rand(b_shape + s_shape), particles=ptcl, log_densities=dens),
                       Message(MessageType.Both,
                               batch_shape=b_shape, param_shape=p_shape, sample_shape=s_shape, event_shape=e_shape,
                               parameter=param,
                               weight=torch.rand(b_shape + s_shape), particles=ptcl, log_densities=dens)]
        # Set up messages
        for mock_in_ld, in_msg in zip(mock_in_ld_list, in_msg_list):
            mock_in_ld.new = True
            mock_in_ld.fn = MagicMock(spec_set=FactorNode)
            mock_in_ld.read.return_value = in_msg
        mock_oud_ld = MagicMock(spec=LinkData)
        mock_oud_ld.fn = MagicMock(spec_set=FactorNode)

        # Use KS with MultivariateNormal distribution class
        wmvn = self.generate_wmvn_2()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata.append(mock_oud_ld)

        wmvn.compute()

        # Check call and arguments
        mock_oud_ld.write.assert_called_once()
        # Check message type
        received_msg = mock_oud_ld.write.call_args[0][0]
        assert received_msg.type is MessageType.Particles

    def test_compute_combination_all_types_mock_ld(self):
        # Test message combination with Particles, Parameter, and Both type messages using mocked linkdata
        # Three LinkData: one sends each type
        # Using MultivariateNormal distribution class as example
        b_shape, s_shape, e_shape = Size([4, 5]), Size([10, 15, 20]), Size([1, 2, 3])
        e_size = sum(e_shape)
        p_shape = Size([e_size * (e_size + 1)])

        num_in_ld = 3
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        # Particles message data
        ptcl, dens = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)], [torch.randn([s]) for s in s_shape]
        # Parameter message data
        loc = torch.randn(b_shape + Size([e_size]))
        cov = generate_positive_definite(b_shape, e_size)
        param = torch.cat([loc, cov.view(b_shape + Size([-1]))], dim=-1)
        in_msg_list = [Message(MessageType.Parameter,
                               batch_shape=b_shape, param_shape=p_shape,
                               parameter=param),
                       Message(MessageType.Particles,
                               batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                               weight=torch.rand(b_shape + s_shape), particles=ptcl, log_densities=dens),
                       Message(MessageType.Both,
                               batch_shape=b_shape, param_shape=p_shape, sample_shape=s_shape, event_shape=e_shape,
                               parameter=param,
                               weight=torch.rand(b_shape + s_shape), particles=ptcl, log_densities=dens)]
        # Set up messages
        for mock_in_ld, in_msg in zip(mock_in_ld_list, in_msg_list):
            mock_in_ld.new = True
            mock_in_ld.fn = MagicMock(spec_set=FactorNode)
            mock_in_ld.read.return_value = in_msg
        mock_oud_ld = MagicMock(spec=LinkData)
        mock_oud_ld.fn = MagicMock(spec_set=FactorNode)

        # Use KS with MultivariateNormal distribution class
        wmvn = self.generate_wmvn_2()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata.append(mock_oud_ld)

        wmvn.compute()

        # Check call and arguments
        mock_oud_ld.write.assert_called_once()
        # Check message type
        received_msg = mock_oud_ld.write.call_args[0][0]
        assert received_msg.type is MessageType.Particles

    def test_compute_error_incompatible_particles_mock_ld(self):
        # Test that AssertionError is raised if incoming particle messages have incompatible particles
        # Three LinkData: 2 with same particles, 1 other with other random particles
        # Use Distribution class as example
        b_shape, s_shape, e_shape = Size([4, 5]), Size([10, 15, 20]), Size([1, 2, 3])
        num_in_ld = 3
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        ptcl, dens = [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)], [torch.randn([s]) for s in s_shape]
        ptcl_other, dens_other = \
            [torch.randn([s, e]) for s, e in zip(s_shape, e_shape)], [torch.randn([s]) for s in s_shape]
        in_msg_list = [Message(MessageType.Particles,
                               batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                               weight=torch.rand(b_shape + s_shape),
                               particles=ptcl, log_densities=dens),
                       Message(MessageType.Particles,
                               batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                               weight=torch.rand(b_shape + s_shape),
                               particles=ptcl, log_densities=dens),
                       Message(MessageType.Particles,
                               batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                               weight=torch.rand(b_shape + s_shape),
                               particles=ptcl_other, log_densities=dens_other),     # Incompatible message
                       ]
        for mock_in_ld, in_msg in zip(mock_in_ld_list, in_msg_list):
            mock_in_ld.new = True
            mock_in_ld.fn = MagicMock(spec_set=FactorNode)
            mock_in_ld.read.return_value = in_msg
        mock_oud_ld = MagicMock(spec=LinkData)
        mock_oud_ld.fn = MagicMock(spec_set=FactorNode)

        wmvn = self.generate_wmvn_1()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata.append(mock_oud_ld)

        with pytest.raises(AssertionError):
            wmvn.compute()

    def test_compute_exclude_outgoing_link_fn_mock_ld_1(self):
        # Test that when combining messages, message from the incoming linkdata that is connected to the factor node
        #   that the current outgoing linkdata is connected to is excluded from the combination.
        # Scenario 1: 3 incoming LinkData, 2 outgoing LinkData with each of both connect to one distinct FN that
        #   one of the 3 incoming LinkData is connected to
        # All messages are Parameter type. Use stand-in Distribution class
        b_shape, p_shape = Size([4, 5]), Size([2])
        num_in_ld, num_out_ld = 3, 1

        # Initialize incoming message list
        msg_list = [Message(MessageType.Parameter,
                            batch_shape=b_shape, param_shape=p_shape,
                            parameter=torch.randn(b_shape + p_shape)) for i in range(num_in_ld)]

        # Set up mocked linkdata
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        mock_out_ld_list = [MagicMock(spec=LinkData) for i in range(num_out_ld)]
        fn_list = [MagicMock(spec_set=FactorNode) for i in range(num_in_ld)]
        for in_ld, fn, msg in zip(mock_in_ld_list, fn_list, msg_list):
            in_ld.new = True
            in_ld.fn = fn
            in_ld.read.return_value = msg
        expected_out_msg_list = []
        for j, out_ld in enumerate(mock_out_ld_list):
            out_ld.fn = fn_list[j]
            expected_in_msg_list = [msg for i, msg in enumerate(msg_list) if i != j]
            expected_out_msg_list.append(sum(expected_in_msg_list, Message.identity()))

        wmvn = self.generate_wmvn_1()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata = mock_out_ld_list

        wmvn.compute()

        for j, out_ld in enumerate(mock_out_ld_list):
            out_ld.write.assert_called_once()
            call_arg = out_ld.write.call_args[0][0]
            assert call_arg == expected_out_msg_list[j]

    def test_compute_exclude_outgoing_link_fn_mock_ld_2(self):
        # Test that when combining messages, message from the incoming linkdata that is connected to the factor node
        #   that the current outgoing linkdata is connected to is excluded from the combination.
        # Scenario 1: 3 incoming LinkData, 2 outgoing LinkData with none of them connected to a same FactorNode that
        #   the 3 incoming linkdata connect to.
        # All messages are Parameter type. Use stand-in Distribution class
        b_shape, p_shape = Size([4, 5]), Size([2])
        num_in_ld, num_out_ld = 3, 2

        # Initialize incoming message list
        msg_list = [Message(MessageType.Parameter,
                            batch_shape=b_shape, param_shape=p_shape,
                            parameter=torch.randn(b_shape + p_shape)) for i in range(num_in_ld)]

        # Set up mocked linkdata
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        mock_out_ld_list = [MagicMock(spec=LinkData) for i in range(num_out_ld)]
        for in_ld, msg in zip(mock_in_ld_list, msg_list):
            in_ld.new = True
            in_ld.fn = MagicMock(spec_set=FactorNode)
            in_ld.read.return_value = msg
        for j, out_ld in enumerate(mock_out_ld_list):
            out_ld.fn = MagicMock(spec_set=FactorNode)      # All different FNs

        # The message sent to both outgoing LinkData should be the same
        expected_out_msg = sum(msg_list, Message.identity())

        wmvn = self.generate_wmvn_1()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata = mock_out_ld_list

        wmvn.compute()

        for j, out_ld in enumerate(mock_out_ld_list):
            out_ld.write.assert_called_once()
            call_arg = out_ld.write.call_args[0][0]
            assert call_arg == expected_out_msg

    def test_compute_cache(self):
        # Test that cache is properly used when message combination from same group of incoming links is requested
        #   Multiple times, and that the cache is cleared after compute is done.
        # 3 incoming Linkdata, 2 outgoing Linkdata, all linkdata connects to different FactorNodes
        # Expects 1 cache miss and 1 cache hit
        b_shape, p_shape = Size([4, 5]), Size([2])
        num_in_ld, num_out_ld = 3, 2

        # Initialize incoming message list
        msg_list = [Message(MessageType.Parameter,
                            batch_shape=b_shape, param_shape=p_shape,
                            parameter=torch.randn(b_shape + p_shape)) for i in range(num_in_ld)]

        # Set up mocked linkdata
        mock_in_ld_list = [MagicMock(spec=LinkData) for i in range(num_in_ld)]
        mock_out_ld_list = [MagicMock(spec=LinkData) for i in range(num_out_ld)]
        for in_ld, msg in zip(mock_in_ld_list, msg_list):
            in_ld.new = True
            in_ld.fn = MagicMock(spec_set=FactorNode)
            in_ld.read.return_value = msg
        for j, out_ld in enumerate(mock_out_ld_list):
            out_ld.fn = MagicMock(spec_set=FactorNode)  # All different FNs

        # The message sent to both outgoing LinkData should be the same
        expected_out_msg = sum(msg_list, Message.identity())

        wmvn = self.generate_wmvn_1()
        wmvn.in_linkdata = mock_in_ld_list
        wmvn.out_linkdata = mock_out_ld_list

        # Patch wmvn._cache with a mock dict
        mock_cache = MagicMock(spec_set=dict)
        # Manually wraps a dictionary's main methods
        mock_cache.__setitem__.side_effect = wmvn._cache.__setitem__
        mock_cache.keys.side_effect = wmvn._cache.keys
        mock_cache.__getitem__.side_effect = wmvn._cache.__getitem__
        # Replace the original cache dictionary with mock
        wmvn._cache = mock_cache

        wmvn.compute()

        mock_cache.__setitem__.assert_called_once()
        mock_cache.__setitem__.assert_called_once_with(tuple(mock_in_ld_list), expected_out_msg)     # 1 cache miss
        mock_cache.__getitem__.assert_called_once()
        mock_cache.__getitem__.assert_called_once_with(tuple(mock_in_ld_list))     # 1 cache hit

