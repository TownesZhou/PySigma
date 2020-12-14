"""
    Unit tests for LTMFN class
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, FactorNode, VariableNode
from pysigma.graphical.predicate_nodes import LTMFN, WMVN
from pysigma.utils import KnowledgeServer as KS

from ...utils import generate_positive_definite


class TestLTMFN():

    # Utility methods
    def generate_ltmfn_1(self, b_shape=Size([4, 5]), p_shape=Size([2]), s_shape=Size([10, 15, 20]),
                         e_shape=Size([1, 2, 3])):
        name = "test_ltmfn"

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

        ltmfn = LTMFN(name, ks, rel_var_list, param_var, index_var_list, ran_var_list)

        return ltmfn

    def test_init(self):
        ltmfn = self.generate_ltmfn_1()

        assert ltmfn.to_draw
        assert ltmfn.msg_cache is None

    def test_add_link_correct_outgoing_link(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        ltmfn = self.generate_ltmfn_1(*msg_shape)

        # Mock outgoing link that is connected to a WMVN
        mock_vn = MagicMock(spec=WMVN)          # WMVN class
        mock_vn.name = "test_variable_node"

        ld = LinkData(mock_vn, ltmfn, False, msg_shape)  # Outgoing link

        ltmfn.add_link(ld)

        assert len(ltmfn.out_linkdata) == 1 and ltmfn.out_linkdata[0] is ld

    def test_add_link_outgoing_link_not_to_wmvn(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        ltmfn = self.generate_ltmfn_1(*msg_shape)

        # Mock outgoing link that is not connected to a WMVN
        mock_vn = MagicMock(spec=VariableNode)      # Abstract variable node, not a WMVN
        mock_vn.name = "test_variable_node"

        ld = LinkData(mock_vn, ltmfn, False, msg_shape)       # Outgoing link

        with pytest.raises(AssertionError, match="At {}: Attempting to register an outgoing link that is not connected "
                                                 "to a WMVN.".format(ltmfn.name)):
            ltmfn.add_link(ld)

    def test_add_link_multiple_outgoing_links(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        ltmfn = self.generate_ltmfn_1(*msg_shape)

        # Mock outgoing link that is connected to a WMVN
        mock_vn_1 = MagicMock(spec=WMVN)  # WMVN class
        mock_vn_1.name = "test_variable_node_1"
        mock_vn_2 = MagicMock(spec=WMVN)  # WMVN class
        mock_vn_2.name = "test_variable_node_2"

        ld_1 = LinkData(mock_vn_1, ltmfn, False, msg_shape)  # Outgoing link
        ld_2 = LinkData(mock_vn_2, ltmfn, False, msg_shape)  # Outgoing link

        with pytest.raises(AssertionError, match="At {}: Attempting to register more than one outgoing linkdata."
                                                 .format(ltmfn.name)):
            ltmfn.add_link(ld_1)
            ltmfn.add_link(ld_2)

    def test_add_link_missing_type_attribute(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        ltmfn = self.generate_ltmfn_1(*msg_shape)

        # Mock incoming link that does not declare a "type" keyed attribute
        mock_vn = MagicMock(VariableNode)
        mock_vn.name = "test_variable_node_1"
        ld = LinkData(mock_vn, ltmfn, True, msg_shape)      # Incoming link
        ld.attr = {}        # Set empty dictionary

        with pytest.raises(AssertionError, match="At{}: Incoming link to a LTMFN must specify 'type' special attribute"
                                                 .format(ltmfn.name)):
            ltmfn.add_link(ld)

    def test_add_link_wrong_type_attribute(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        ltmfn = self.generate_ltmfn_1(*msg_shape)

        # Mock incoming link that does not declare a "type" keyed attribute
        mock_vn = MagicMock(VariableNode)
        mock_vn.name = "test_variable_node_1"
        ld = LinkData(mock_vn, ltmfn, True, msg_shape)  # Incoming link
        ld.attr = {'type': 'something random'}  # Set wrong value for "type" keyed entry

        with pytest.raises(AssertionError, match="At{}: Incoming link to a LTMFN must have 'type' special attribute "
                                                 "with value 'event' or 'param'".format(ltmfn.name)):
            ltmfn.add_link(ld)

    def test_add_link_event_type_incoming_link_not_to_wmvn(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        ltmfn = self.generate_ltmfn_1(*msg_shape)

        # Mock incoming link that does not declare a "type" keyed attribute
        mock_vn = MagicMock(VariableNode)       # Some abstract VariableNode
        mock_vn.name = "test_variable_node_1"
        ld = LinkData(mock_vn, ltmfn, True, msg_shape)  # Incoming link
        ld.attr = {'type': 'event'}             # Set event type

        with pytest.raises(AssertionError, match="At {}: Attempting to register an event type incoming link that is "
                                                 "not connected to a WMVN".format(ltmfn.name)):
            ltmfn.add_link(ld)

    def test_add_link_multiple_event_type_incoming_link(self):
        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        ltmfn = self.generate_ltmfn_1(*msg_shape)

        # Mock incoming link that does not declare a "type" keyed attribute
        mock_vn_1 = MagicMock(WMVN)  # WMVN class
        mock_vn_1.name = "test_variable_node_1"
        ld_1 = LinkData(mock_vn_1, ltmfn, True, msg_shape)  # Incoming link
        ld_1.attr = {'type': 'event'}  # Set event type

        mock_vn_2 = MagicMock(WMVN)  # WMVN class
        mock_vn_2.name = "test_variable_node_1"
        ld_2 = LinkData(mock_vn_2, ltmfn, True, msg_shape)  # Incoming link
        ld_2.attr = {'type': 'event'}  # Set event type

        with pytest.raises(AssertionError, match="At {}: Attempting to register more than one incoming event type "
                                                 "linkdata".format(ltmfn.name)):
            ltmfn.add_link(ld_1)
            ltmfn.add_link(ld_2)

    def test_add_link_correct_outgoing_link(self):
        # Test adding one event type link and multiple param type links
        num_param_lds = 5

        msg_shape = (Size([4, 5]), Size([2]), Size([10, 15, 20]), Size([1, 2, 3]))
        ltmfn = self.generate_ltmfn_1(*msg_shape)

        mock_vn = MagicMock(WMVN)  # Some abstract VariableNode
        mock_vn.name = "test_variable_node_1"
        event_ld = LinkData(mock_vn, ltmfn, True, msg_shape)  # Incoming link
        event_ld.attr = {'type': 'event'}  # Set event type

        param_lds = []
        for i in range(num_param_lds):
            mock_vn_param = MagicMock(VariableNode)
            mock_vn_param.name = "test_variable_node_for_param_type_ld_{}".format(i)
            param_ld = LinkData(mock_vn_param, ltmfn, True, msg_shape)
            param_ld.attr = {'type': 'param'}  # Set param type
            param_lds.append(param_ld)

        ltmfn.add_link(event_ld)
        for param_ld in param_lds:
            ltmfn.add_link(param_ld)

        assert len(ltmfn.in_linkdata) == num_param_lds + 1
        assert [ld for ld in ltmfn.in_linkdata if ld.attr['type'] == 'event'][0] is event_ld
        assert set([ld for ld in ltmfn.in_linkdata if ld.attr['type'] == 'param']) == set(param_lds)

