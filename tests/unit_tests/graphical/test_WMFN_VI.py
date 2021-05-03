"""
    Unit tests for WMFN_VI class
"""
import pytest
from unittest.mock import MagicMock
import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, VariableNode, NodeConfigurationError
from pysigma.graphical.predicate_nodes import WMFN_VI, WMVN
from pysigma.utils import KnowledgeServer as KS
from pysigma.utils import DistributionServer as DS
from ...utils import random_message

from ...utils import generate_positive_definite


class TestWMFN_VI:

    # Utility methods
    def generate_wmfn_vi_1(self, b_shape=Size([4, 5]), p_shape=Size([2]), s_shape=Size([10, 15, 20]),
                         e_shape=Size([1, 2, 3])):
        name = "test_wmfn_vi"

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

        wmfn_vi = WMFN_VI(name, ks, rel_var_list, param_var, index_var_list, ran_var_list)

        return wmfn_vi

    def generate_wmfn_vi_2(self, b_shape=Size([4, 5]), p_shape=Size([42]), s_shape=Size([10, 15, 20]),
                         e_shape=Size([1, 2, 3])):
        # Generate a wmfn_vi carrying a KS with actual distribution class - MultivariateNormal
        name = "test_wmfn_vi"

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

        wmfn_vi = WMFN_VI(name, ks, rel_var_list, param_var, index_var_list, ran_var_list)

        return wmfn_vi

    def generate_wmfn_vi_2(self, b_shape=Size([4, 5]), p_shape=Size([42]), s_shape=Size([10, 15, 20]),
                         e_shape=Size([1, 2, 3])):
        name = "test_wmfn_vi"

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

        wmfn_vi = WMFN_VI(name, ks, rel_var_list, param_var, index_var_list, ran_var_list)

        return wmfn_vi

    def test_init(self):
        wmfn_vi = self.generate_wmfn_vi_1()

        assert wmfn_vi.to_draw
        assert wmfn_vi.msg_cache.isid

    def test_add_link_correct_outgoing_link(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_1(*msg_shape)

        # Mock outgoing link that is connected to a WMVN
        mock_vn = MagicMock(spec=WMVN)          # WMVN class
        mock_vn.name = "test_variable_node"

        ld = LinkData(mock_vn, wmfn_vi, False, msg_shape)  # Outgoing link

        wmfn_vi.add_link(ld)

        assert len(wmfn_vi.out_linkdata) == 1 and wmfn_vi.out_linkdata[0] is ld

    def test_add_link_outgoing_link_not_to_wmvn(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_1(*msg_shape)

        # Mock outgoing link that is not connected to a WMVN
        mock_vn = MagicMock(spec=VariableNode)      # Abstract variable node, not a WMVN
        mock_vn.name = "test_variable_node"

        ld = LinkData(mock_vn, wmfn_vi, False, msg_shape)       # Outgoing link

        with pytest.raises(AssertionError, match="At {}: Attempting to register an outgoing link that is not connected "
                                                 "to a WMVN.".format(wmfn_vi.name)):
            wmfn_vi.add_link(ld)

    def test_add_link_multiple_outgoing_links(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_1(*msg_shape)

        # Mock outgoing link that is connected to a WMVN
        mock_vn_1 = MagicMock(spec=WMVN)  # WMVN class
        mock_vn_1.name = "test_variable_node_1"
        mock_vn_2 = MagicMock(spec=WMVN)  # WMVN class
        mock_vn_2.name = "test_variable_node_2"

        ld_1 = LinkData(mock_vn_1, wmfn_vi, False, msg_shape)  # Outgoing link
        ld_2 = LinkData(mock_vn_2, wmfn_vi, False, msg_shape)  # Outgoing link

        with pytest.raises(AssertionError, match="At {}: Attempting to register more than one outgoing linkdata."
                                                 .format(wmfn_vi.name)):
            wmfn_vi.add_link(ld_1)
            wmfn_vi.add_link(ld_2)

    def test_add_link_missing_type_attribute(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_1(*msg_shape)

        # Mock incoming link that does not declare a "type" keyed attribute
        mock_vn = MagicMock(VariableNode)
        mock_vn.name = "test_variable_node_1"
        ld = LinkData(mock_vn, wmfn_vi, True, msg_shape)      # Incoming link
        ld.attr = {}        # Set empty dictionary

        with pytest.raises(AssertionError, match="At{}: Incoming link to a WMFN_VI must specify 'type' special attribute"
                                                 .format(wmfn_vi.name)):
            wmfn_vi.add_link(ld)

    def test_add_link_wrong_type_attribute(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_1(*msg_shape)

        # Mock incoming link that does not declare a "type" keyed attribute
        mock_vn = MagicMock(VariableNode)
        mock_vn.name = "test_variable_node_1"
        ld = LinkData(mock_vn, wmfn_vi, True, msg_shape)  # Incoming link
        ld.attr = {'type': 'something random'}  # Set wrong value for "type" keyed entry

        with pytest.raises(AssertionError, match="At{}: Incoming link to a WMFN_VI must have 'type' special attribute "
                                                 "with value 'event' or 'param'".format(wmfn_vi.name)):
            wmfn_vi.add_link(ld)

    def test_add_link_event_type_incoming_link_not_to_wmvn(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_1(*msg_shape)

        # Mock incoming link that does not declare a "type" keyed attribute
        mock_vn = MagicMock(VariableNode)       # Some abstract VariableNode
        mock_vn.name = "test_variable_node_1"
        ld = LinkData(mock_vn, wmfn_vi, True, msg_shape)  # Incoming link
        ld.attr = {'type': 'event'}             # Set event type

        with pytest.raises(AssertionError, match="At {}: Attempting to register an event type incoming link that is "
                                                 "not connected to a WMVN".format(wmfn_vi.name)):
            wmfn_vi.add_link(ld)

    def test_add_link_multiple_event_type_incoming_link(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_1(*msg_shape)

        # Mock incoming link that does not declare a "type" keyed attribute
        mock_vn_1 = MagicMock(WMVN)  # WMVN class
        mock_vn_1.name = "test_variable_node_1"
        ld_1 = LinkData(mock_vn_1, wmfn_vi, True, msg_shape)  # Incoming link
        ld_1.attr = {'type': 'event'}  # Set event type

        mock_vn_2 = MagicMock(WMVN)  # WMVN class
        mock_vn_2.name = "test_variable_node_1"
        ld_2 = LinkData(mock_vn_2, wmfn_vi, True, msg_shape)  # Incoming link
        ld_2.attr = {'type': 'event'}  # Set event type

        with pytest.raises(AssertionError, match="At {}: Attempting to register more than one incoming event type "
                                                 "linkdata".format(wmfn_vi.name)):
            wmfn_vi.add_link(ld_1)
            wmfn_vi.add_link(ld_2)

    def test_add_link_correct_incoming_link(self):
        # Test adding one event type link and multiple param type links
        num_param_lds = 5

        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_1(*msg_shape)

        mock_vn = MagicMock(WMVN)  # Some abstract VariableNode
        mock_vn.name = "test_variable_node_1"
        event_ld = LinkData(mock_vn, wmfn_vi, True, msg_shape)  # Incoming link
        event_ld.attr = {'type': 'event'}  # Set event type

        param_lds = []
        for i in range(num_param_lds):
            mock_vn_param = MagicMock(VariableNode)
            mock_vn_param.name = "test_variable_node_for_param_type_ld_{}".format(i)
            param_ld = LinkData(mock_vn_param, wmfn_vi, True, msg_shape)
            param_ld.attr = {'type': 'param'}  # Set param type
            param_lds.append(param_ld)

        wmfn_vi.add_link(event_ld)
        for param_ld in param_lds:
            wmfn_vi.add_link(param_ld)

        assert len(wmfn_vi.in_linkdata) == num_param_lds + 1
        assert [ld for ld in wmfn_vi.in_linkdata if ld.attr['type'] == 'event'][0] is event_ld
        assert set([ld for ld in wmfn_vi.in_linkdata if ld.attr['type'] == 'param']) == set(param_lds)

    def test_modify_wrong_message_type(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_1(*msg_shape)

        param_lds = []
        for i in range(3):
            mock_vn = MagicMock(VariableNode)
            mock_vn.name = "test_variable_node_{}".format(i)
            ld = LinkData(mock_vn, wmfn_vi, True, msg_shape)
            ld.attr = {'type': 'param'}
            param_lds.append(ld)

        param_lds[0].write(random_message(MessageType.Parameter, *msg_shape))
        param_lds[1].write(random_message(MessageType.Dual, *msg_shape))
        param_lds[2].write(random_message(MessageType.Particles, *msg_shape))

        for param_ld in param_lds:
            wmfn_vi.add_link(param_ld)

        with pytest.raises(AssertionError) as excinfo:
            wmfn_vi.modify()

        assert str(excinfo.value) == "At {}: Expect all messages from incoming param type linkdata to contain " \
                                     "parameters, but instead found message types: {} from linkdata {}." \
                                     .format(wmfn_vi.name, [MessageType.Particles], [str(param_lds[2])])

    def test_modify_correct_param_no_draw(self):
        num_param_lds = 3
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_1(*msg_shape)

        param_lds = []
        for i in range(num_param_lds):
            mock_vn = MagicMock(VariableNode)
            mock_vn.name = "test_variable_node_{}".format(i)
            ld = LinkData(mock_vn, wmfn_vi, True, msg_shape)
            ld.attr = {'type': 'param'}
            param_lds.append(ld)

        msgs = [random_message(MessageType.Parameter, *msg_shape) for i in range(num_param_lds)]
        combined_msg = sum(msgs, Message.identity())

        for param_ld, msg in zip(param_lds, msgs):
            param_ld.write(msg)
            wmfn_vi.add_link(param_ld)

        wmfn_vi.set_draw(False)       # Select no draw
        wmfn_vi.modify()

        assert wmfn_vi.msg_cache == combined_msg

    def test_modify_correct_particles_with_draw(self):
        # Use MultivariateNormal distribution as example
        # With natural parameters
        num_param_lds = 3
        msg_shape = (Size([4, 5]), Size([42]), Size([10, 15, 20]), Size([1, 2, 3]))
        wmfn_vi = self.generate_wmfn_vi_2(*msg_shape)

        param_lds = []
        for i in range(num_param_lds):
            mock_vn = MagicMock(VariableNode)
            mock_vn.name = "test_variable_node_{}".format(i)
            ld = LinkData(mock_vn, wmfn_vi, True, msg_shape)
            ld.attr = {'type': 'param'}
            param_lds.append(ld)

        msgs = []
        for i in range(num_param_lds):
            loc = torch.randn(Size([4, 5]) + Size([6]))
            cov = generate_positive_definite(Size([4, 5]), 6)
            p1, p2 = DS._multivariate_normal_param_reg2exp([loc, cov])
            param = torch.cat([p1, p2.view(Size([4, 5]) + Size([-1]))], dim=-1)
            msgs.append(Message(MessageType.Parameter,
                                batch_shape=Size([4, 5]), param_shape=Size([42]),
                                parameter=param))
        combined_msg = sum(msgs, Message.identity())

        for param_ld, msg in zip(param_lds, msgs):
            param_ld.write(msg)
            wmfn_vi.add_link(param_ld)

        wmfn_vi.set_draw(True)  # Select no draw
        wmfn_vi.modify()

        assert Message.reduce_type(wmfn_vi.msg_cache, MessageType.Parameter) == combined_msg

    def test_ill_configuration_1(self):
        # Test that NodeConfigurationError is raised if missing incoming linkdata.
        # Mock only one outgoing linkdata
        mock_out_ld = MagicMock(spec_set=LinkData)

        wmfn_vi = self.generate_wmfn_vi_1()
        wmfn_vi.out_linkdata.append(mock_out_ld)

        with pytest.raises(NodeConfigurationError) as excinfo:
            wmfn_vi.compute()

        assert str(excinfo.value) == "Wrong configuration for node {}: a WMFN_VI expects at least one incoming " \
                                     "linkdata and one outgoing linkdata to be computable. Found 0 registered " \
                                     "incoming linkdata and 1 registered outgoing linkdata".format(wmfn_vi.name)

    def test_ill_configuration_2(self):
        # Test that NodeConfigurationError is raised if missing outgoing linkdata.
        # Mock only one incoming linkdata
        mock_in_ld = MagicMock(spec_set=LinkData)

        wmfn_vi = self.generate_wmfn_vi_1()
        wmfn_vi.in_linkdata.append(mock_in_ld)

        with pytest.raises(NodeConfigurationError) as excinfo:
            wmfn_vi.compute()

        assert str(excinfo.value) == "Wrong configuration for node {}: a WMFN_VI expects at least one incoming " \
                                     "linkdata and one outgoing linkdata to be computable. Found 1 registered " \
                                     "incoming linkdata and 0 registered outgoing linkdata".format(wmfn_vi.name)

    def test_quiescence(self):
        # Test that no message is sent when quiesced
        wmfn_vi = self.generate_wmfn_vi_1()

        # Set visited to True
        wmfn_vi.visited = True

        # Use mock ld
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        wmfn_vi.in_linkdata.append(mock_in_ld)
        wmfn_vi.out_linkdata.append(mock_out_ld)

        wmfn_vi.compute()
        mock_out_ld.write.assert_not_called()

    def test_compute_correct_msg(self):
        # Test that out ld receives correct message
        wmfn_vi = self.generate_wmfn_vi_1()

        mock_msg = MagicMock(spec_set=Message)
        wmfn_vi.msg_cache = mock_msg

        # Use mock ld
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        wmfn_vi.in_linkdata.append(mock_in_ld)
        wmfn_vi.out_linkdata.append(mock_out_ld)

        wmfn_vi.compute()
        mock_out_ld.write.assert_called_once_with(mock_msg)

    def test_compute_once_then_quiesced(self):
        # Test that compute() can only be called once and then will reach quiescence in one decision phase
        wmfn_vi = self.generate_wmfn_vi_1()

        mock_msg = MagicMock(spec_set=Message)
        wmfn_vi.msg_cache = mock_msg

        # Use mock ld
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        wmfn_vi.in_linkdata.append(mock_in_ld)
        wmfn_vi.out_linkdata.append(mock_out_ld)

        # Call twice
        wmfn_vi.compute()
        wmfn_vi.compute()

        assert wmfn_vi.quiescence
        mock_out_ld.write.assert_called_once_with(mock_msg)

