"""
    Unit tests for GVN class
"""
import pytest
from unittest.mock import MagicMock
from typing import List, Type

import torch
import torch.distributions.constraints as C
import torch.distributions as D
import torch.distributions.transforms as T
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, DVN, VariableNode, FactorNode, NodeConfigurationError
from pysigma.graphical.gamma_node import GVN
from ...utils import random_message, assert_equal_within_error, assert_proportional_within_error, \
    generate_positive_definite


def gen_ld(fn, vn, msg_shape, incoming: bool, inward: bool, name=''):
    # Generate mocked linkdata
    mock_ld = MagicMock(spec=LinkData)
    mock_ld.fn, mock_ld.vn = fn, vn
    mock_ld.msg_shape = msg_shape
    mock_ld.to_fn = not incoming
    mock_ld.attr = {'direction': 'inward' if inward else 'outward'}
    # repr
    mock_ld.__repr__ = MagicMock(return_value=name)
    return mock_ld


class TestGVN_Init:

    def test_incorrect_multiple_ran_vars(self):
        # Test AssertionError is raised if multiple random variables are given
        # Test Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
            Variable("ran_var_2", VariableMetatype.Random, 3, (C.real,)),
            Variable("ran_var_3", VariableMetatype.Random, 3, (C.real,)),
        ]

        with pytest.raises(AssertionError) as excinfo:
            gvn = GVN("test_gvn", rel_var_list, ran_var_list)
        assert str(excinfo.value) == "GVN can admit only one random variable. Given random variables: {}"\
            .format(tuple(ran_var_list))

    def test_correct(self):
        # Test Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]

        gvn = GVN("test_gvn", rel_var_list, ran_var_list)

        assert len(gvn.labeled_ld_group) == 0


class TestGVN_AddLink:

    def test_missing_special_attribute(self):
        # Test AssertionError is raised if linkdata misses the special attribute
        # Test Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)
        # Mock Factor node
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_fn"
        # Test linkdata
        test_ld = LinkData(gvn, mock_fn, False, (Size([5, 5, 5]), Size([]), Size([]), Size([3])))

        with pytest.raises(AssertionError) as excinfo:
            gvn.add_link(test_ld)
        assert str(excinfo.value) == "In test_gvn: For a Gamma Variable Node, the linkdata '{}' should specify a " \
                                     "special attribute named 'direction', with value either 'inward' or 'outward'." \
                                     .format(test_ld)

    def test_wrong_special_attribute(self):
        # Test AssertionError is raised if linkdata's special attribute value was wrong
        # Test Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)
        # Mock Factor node
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_fn"
        # Test linkdata
        test_ld = LinkData(gvn, mock_fn, False, (Size([5, 5, 5]), Size([]), Size([]), Size([3])),
                           direction="something random")    # Wrong special attribute value

        with pytest.raises(AssertionError) as excinfo:
            gvn.add_link(test_ld)
        assert str(excinfo.value) == "In test_gvn: For a Gamma Variable Node, the linkdata '{}' should specify a " \
                                     "special attribute named 'direction', with value either 'inward' or 'outward'." \
            .format(test_ld)

    def test_incorrect_duplicate_inward_outgoing_linkdata(self):
        # Test AssertionError is raised if multiple inward outgoing linkdata are registered
        # Test AssertionError is raised if linkdata's special attribute value was wrong
        # Test Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        msg_shape = (Size([5, 5, 5]), Size([]), Size([]), Size([3]))
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)
        # Mock Factor node
        mock_fn_1, mock_fn_2 = MagicMock(spec=FactorNode), MagicMock(spec=FactorNode)
        mock_fn_1.name, mock_fn_2.name = "test_fn_1", "test_fn_2"
        # Test linkdata
        test_ld_1 = LinkData(gvn, mock_fn_1, True, msg_shape, direction="inward")
        test_ld_2 = LinkData(gvn, mock_fn_2, True, msg_shape, direction="inward")     # Duplicate

        gvn.add_link(test_ld_1)
        with pytest.raises(AssertionError) as excinfo:
            gvn.add_link(test_ld_2)
        assert str(excinfo.value) == "In test_gvn: A GVN can admit at most one inward outgoing linkdata. Linkdata {} " \
                                     "with the same direction is already registered." \
                                     .format(test_ld_1)

    def test_incorrect_duplicate_outward_incoming_linkdata(self):
        # Test AssertionError is raised if multiple inward outgoing linkdata are registered
        # Test AssertionError is raised if linkdata's special attribute value was wrong
        # Test Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        msg_shape = (Size([5, 5, 5]), Size([]), Size([]), Size([3]))
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)
        # Mock Factor node
        mock_fn_1, mock_fn_2 = MagicMock(spec=FactorNode), MagicMock(spec=FactorNode)
        mock_fn_1.name, mock_fn_2.name = "test_fn_1", "test_fn_2"
        # Test linkdata
        test_ld_1 = LinkData(gvn, mock_fn_1, False, msg_shape, direction="outward")
        test_ld_2 = LinkData(gvn, mock_fn_2, False, msg_shape, direction="outward")     # Duplicate

        gvn.add_link(test_ld_1)
        with pytest.raises(AssertionError) as excinfo:
            gvn.add_link(test_ld_2)
        assert str(excinfo.value) == "In test_gvn: A GVN can admit at most one outward incoming linkdata. Linkdata {} "\
                                     "with the same direction is already registered." \
                                     .format(test_ld_1)

    def test_correct(self):
        # Test correct add_link() and check registered linkdata group data structure
        # Test Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        msg_shape = (Size([5, 5, 5]), Size([]), Size([]), Size([3]))
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)
        # Mock Factor node
        mock_fn_1, mock_fn_2, mock_fn_3, mock_gfn = MagicMock(spec=FactorNode), MagicMock(spec=FactorNode), \
                                                    MagicMock(spec=FactorNode), MagicMock(spec=FactorNode)
        mock_fn_1.name, mock_fn_2.name, mock_fn_3.name, mock_gfn.name = "test_fn_1", "test_fn_2", "test_fn_3", \
                                                                        "test_gfn"
        # Test linkdata
        inward_in_ld_1, inward_in_ld_2, inward_in_ld_3 = \
            LinkData(gvn, mock_fn_1, False, msg_shape, direction="inward"), \
            LinkData(gvn, mock_fn_2, False, msg_shape, direction="inward"), \
            LinkData(gvn, mock_fn_3, False, msg_shape, direction="inward")
        inward_out_ld = LinkData(gvn, mock_gfn, True, msg_shape, direction="inward")
        outward_in_ld = LinkData(gvn, mock_gfn, False, msg_shape, direction="outward")
        outward_out_ld_1, outward_out_ld_2, outward_out_ld_3 = \
            LinkData(gvn, mock_fn_1, True, msg_shape, direction="outward"), \
            LinkData(gvn, mock_fn_2, True, msg_shape, direction="outward"), \
            LinkData(gvn, mock_fn_3, True, msg_shape, direction="outward")

        gvn.add_link(inward_in_ld_1)
        gvn.add_link(inward_in_ld_2)
        gvn.add_link(inward_in_ld_3)
        gvn.add_link(inward_out_ld)
        gvn.add_link(outward_in_ld)
        gvn.add_link(outward_out_ld_1)
        gvn.add_link(outward_out_ld_2)
        gvn.add_link(outward_out_ld_3)

        assert dict(gvn.labeled_ld_group['inward']) == {
            True: [inward_out_ld],
            False: [inward_in_ld_1, inward_in_ld_2, inward_in_ld_3]
        }
        assert dict(gvn.labeled_ld_group['outward']) == {
            True: [outward_out_ld_1, outward_out_ld_2, outward_out_ld_3],
            False: [outward_in_ld]
        }


class TestGVN_PrecomputeCheck:

    def test_error_no_linkdata(self):
        # Test NodeConfigurationError is raised if no linkdata is registered
        # Test Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        msg_shape = (Size([5, 5, 5]), Size([]), Size([]), Size([3]))
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)

        with pytest.raises(NodeConfigurationError) as excinfo:
            gvn.compute()

        assert str(excinfo.value) == "In test_gvn: No linkdata is registered."

    def test_error_incomplete_linkdata_group_1(self):
        # Test NodeConfigurationError is raised if in some group, only incoming or outgoing linkdata is registered
        # Test only inward incoming
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        msg_shape = (Size([5, 5, 5]), Size([]), Size([]), Size([3]))
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)

        # Only register one inward incoming linkdata
        # Mock Factor node
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_fn"
        # Test linkdata
        test_ld = LinkData(gvn, mock_fn, False, msg_shape, direction="inward")

        gvn.add_link(test_ld)

        with pytest.raises(NodeConfigurationError) as excinfo:
            gvn.compute()
        assert str(excinfo.value) == "In test_gvn: In the inward directional linkdata group, expect both incoming and "\
                                     "outgoing linkdata. However, found only incoming linkdata are registered. "

    def test_error_incomplete_linkdata_group_2(self):
        # Test NodeConfigurationError is raised if in some group, only incoming or outgoing linkdata is registered
        # Test only outward outgoing
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        msg_shape = (Size([5, 5, 5]), Size([]), Size([]), Size([3]))
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)

        # Only register one outward outgoing linkdata
        # Mock Factor node
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_fn"
        # Test linkdata
        test_ld = LinkData(gvn, mock_fn, True, msg_shape, direction="outward")

        gvn.add_link(test_ld)

        with pytest.raises(NodeConfigurationError) as excinfo:
            gvn.compute()
        assert str(excinfo.value) == "In test_gvn: In the outward directional linkdata group, expect both incoming " \
                                     "and outgoing linkdata. However, found only outgoing linkdata are registered. "


class TestGVN_Compute:

    def test_inward_combine(self):
        # Test inward message propagation. Message shall be combined arithmetically
        # Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        msg_shape = (Size([5, 5, 5]), Size([2]), Size([15]), Size([3]))
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)
        # Mock FactorNode
        mock_fn_1, mock_fn_2, mock_fn_3 = \
            MagicMock(spec=FactorNode), MagicMock(spec=FactorNode), MagicMock(spec=FactorNode)
        mock_gfn = MagicMock(spec=FactorNode)
        # Mock linkdata
        inward_in_ld_1, inward_in_ld_2, inward_in_ld_3 = \
            gen_ld(mock_fn_1, gvn, msg_shape, True, True, 'inward_in_ld_1'), \
            gen_ld(mock_fn_2, gvn, msg_shape, True, True, 'inward_in_ld_2'), \
            gen_ld(mock_fn_3, gvn, msg_shape, True, True, 'inward_in_ld_3')
        inward_out_ld = gen_ld(mock_gfn, gvn, msg_shape, False, True, 'inward_out_ld')
        inward_in_ld_1.new, inward_in_ld_2.new, inward_in_ld_3.new = True, True, True
        # Mock message
        test_ptcl = [torch.randn(15, 3)]
        test_dens = [torch.randn(15)]
        in_msg_1, in_msg_2, in_msg_3 = \
            random_message(MessageType.Dual, *msg_shape), \
            random_message(MessageType.Dual, *msg_shape), \
            random_message(MessageType.Dual, *msg_shape)
        in_msg_1.particles, in_msg_1.log_densities = test_ptcl, test_dens
        in_msg_2.particles, in_msg_2.log_densities = test_ptcl, test_dens
        in_msg_3.particles, in_msg_3.log_densities = test_ptcl, test_dens
        in_msg_1, in_msg_2, in_msg_3 = in_msg_1.clone(), in_msg_2.clone(), in_msg_3.clone()  # Use clone to normalize
        inward_in_ld_1.read.return_value, inward_in_ld_2.read.return_value, inward_in_ld_3.read.return_value = \
            in_msg_1, in_msg_2, in_msg_3

        expected_out_msg = in_msg_1 + in_msg_2 + in_msg_3

        gvn.add_link(inward_in_ld_1)
        gvn.add_link(inward_in_ld_2)
        gvn.add_link(inward_in_ld_3)
        gvn.add_link(inward_out_ld)

        gvn.compute()

        # Check outgoing linkdata called
        inward_out_ld.write.assert_called_once()
        # Check outgoing message
        out_msg = inward_out_ld.write.call_args[0][0]
        assert out_msg == expected_out_msg

    def test_outward_relay(self):
        # Test outward message propagation. Message shall be relayed
        # Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        msg_shape = (Size([5, 5, 5]), Size([2]), Size([15]), Size([3]))
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)
        # Mock FactorNode
        mock_fn_1, mock_fn_2, mock_fn_3 = \
            MagicMock(spec=FactorNode), MagicMock(spec=FactorNode), MagicMock(spec=FactorNode)
        mock_gfn = MagicMock(spec=FactorNode)
        # Mock linkdata
        outward_out_ld_1, outward_out_ld_2, outward_out_ld_3 = \
            gen_ld(mock_fn_1, gvn, msg_shape, False, False, 'outward_out_ld_1'), \
            gen_ld(mock_fn_2, gvn, msg_shape, False, False, 'outward_out_ld_2'), \
            gen_ld(mock_fn_3, gvn, msg_shape, False, False, 'outward_out_ld_3')
        outward_in_ld = gen_ld(mock_gfn, gvn, msg_shape, True, False, 'outward_in_ld')
        outward_in_ld.new = True
        # Mock message
        test_ptcl = [torch.randn(15, 3)]
        test_dens = [torch.randn(15)]
        in_msg_1 = random_message(MessageType.Dual, *msg_shape)
        in_msg_1.particles, in_msg_1.log_densities = test_ptcl, test_dens
        in_msg_1 = in_msg_1.clone()  # Use clone to normalize
        outward_in_ld.read.return_value = in_msg_1

        gvn.add_link(outward_out_ld_1)
        gvn.add_link(outward_out_ld_2)
        gvn.add_link(outward_out_ld_3)
        gvn.add_link(outward_in_ld)

        gvn.compute()

        # Check outgoing linkdata called
        outward_out_ld_1.write.assert_called_once()
        outward_out_ld_1.write.assert_called_once()
        outward_out_ld_1.write.assert_called_once()
        # Check outgoing message
        out_msg_1 = outward_out_ld_1.write.call_args[0][0]
        out_msg_2 = outward_out_ld_2.write.call_args[0][0]
        out_msg_3 = outward_out_ld_3.write.call_args[0][0]
        assert out_msg_1 == in_msg_1
        assert out_msg_2 == in_msg_1
        assert out_msg_3 == in_msg_1

    def test_quiescence_no_compute(self):
        # Test that when quiesced, no computation is carried out
        # Test with both inward and outward linkdata
        # Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        msg_shape = (Size([5, 5, 5]), Size([2]), Size([15]), Size([3]))
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)
        # Mock FactorNode
        mock_fn_1, mock_fn_2, mock_fn_3 = \
            MagicMock(spec=FactorNode), MagicMock(spec=FactorNode), MagicMock(spec=FactorNode)
        mock_gfn = MagicMock(spec=FactorNode)
        # Mock linkdata
        inward_in_ld = gen_ld(mock_fn_1, gvn, msg_shape, True, True, 'inward_in_ld')
        inward_out_ld = gen_ld(mock_gfn, gvn, msg_shape, False, True, 'inward_out_ld')
        outward_out_ld = gen_ld(mock_fn_1, gvn, msg_shape, False, False, 'outward_out_ld')
        outward_in_ld = gen_ld(mock_gfn, gvn, msg_shape, True, False, 'outward_in_ld')
        # Set new of all linkdata to False
        inward_in_ld.new = False
        inward_out_ld.new = False
        outward_out_ld.new = False
        outward_in_ld.new = False
        # Mock messages
        in_msg_1, in_msg_2 = random_message(MessageType.Dual, *msg_shape), random_message(MessageType.Dual, *msg_shape)
        inward_in_ld.read.return_value, outward_in_ld.read.return_value = in_msg_1, in_msg_2

        gvn.add_link(inward_in_ld)
        gvn.add_link(inward_out_ld)
        gvn.add_link(outward_out_ld)
        gvn.add_link(outward_in_ld)

        gvn.compute()

        # Check that no read and write were called
        inward_in_ld.read.assert_not_called()
        outward_in_ld.read.assert_not_called()
        inward_out_ld.write.assert_not_called()
        outward_out_ld.write.assert_not_called()

    def test_quiescence_selective_compute(self):
        # Test that when not quiesced, only affect linkdata group is computed
        # Test with both inward and outward linkdata
        # Test with inward direction break quiescence, but not outward direction
        # Variables
        rel_var_list = [
            Variable("rel_var_1", VariableMetatype.Relational, 5),
            Variable("rel_var_2", VariableMetatype.Relational, 5),
            Variable("rel_var_3", VariableMetatype.Relational, 5),
        ]
        ran_var_list = [
            Variable("ran_var_1", VariableMetatype.Random, 3, (C.real,)),
        ]
        msg_shape = (Size([5, 5, 5]), Size([2]), Size([15]), Size([3]))
        # GVN
        gvn = GVN("test_gvn", rel_var_list, ran_var_list)
        # Mock FactorNode
        mock_fn_1, mock_fn_2, mock_fn_3 = \
            MagicMock(spec=FactorNode), MagicMock(spec=FactorNode), MagicMock(spec=FactorNode)
        mock_gfn = MagicMock(spec=FactorNode)
        # Mock linkdata
        inward_in_ld = gen_ld(mock_fn_1, gvn, msg_shape, True, True, 'inward_in_ld')
        inward_out_ld = gen_ld(mock_gfn, gvn, msg_shape, False, True, 'inward_out_ld')
        outward_out_ld = gen_ld(mock_fn_1, gvn, msg_shape, False, False, 'outward_out_ld')
        outward_in_ld = gen_ld(mock_gfn, gvn, msg_shape, True, False, 'outward_in_ld')
        # Set new of all linkdata to False
        inward_in_ld.new = True     # Has new message
        inward_out_ld.new = False
        outward_out_ld.new = False
        outward_in_ld.new = False
        # Mock messages
        in_msg_1, in_msg_2 = random_message(MessageType.Dual, *msg_shape), random_message(MessageType.Dual, *msg_shape)
        inward_in_ld.read.return_value, outward_in_ld.read.return_value = in_msg_1, in_msg_2

        gvn.add_link(inward_in_ld)
        gvn.add_link(inward_out_ld)
        gvn.add_link(outward_out_ld)
        gvn.add_link(outward_in_ld)

        gvn.compute()

        # Check that no read and write were called
        inward_in_ld.read.assert_called_once()      # Inward direction propagated
        inward_out_ld.write.assert_called_once()
        outward_in_ld.read.assert_not_called()      # Outward direction quiesced
        outward_out_ld.write.assert_not_called()
