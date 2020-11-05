"""
    Unit tests for DFN class
"""
import pytest
from unittest.mock import patch, PropertyMock, MagicMock
from torch import Size
from torch.distributions.constraints import real

from pysigma.defs import Message
from pysigma.graphical.basic_nodes import DFN, LinkData
from pysigma.defs import Variable, VariableMetatype
from .test_VariableNode import VariableNodeForTest


class TestDFN():

    def test_add_link_more_than_one_incoming_link(self):
        fn = DFN("test_dfn")
        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3),
                        Variable("rel2", VariableMetatype.Relational, 4),
                        Variable("rel3", VariableMetatype.Relational, 5)]
        vn = VariableNodeForTest("test_vn", rel_var_list)
        msg_shape = (Size([3, 4, 5]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(vn, fn, True, msg_shape)

        fn.add_link(ld)
        with pytest.raises(AssertionError):
            fn.add_link(ld)

    def test_add_link_incompatible_variables(self):
        # Test 1: incompatible batch shape
        fn = DFN("test_dfn")
        rel_var_list_1 = [Variable("rel1", VariableMetatype.Relational, 3)]
        rel_var_list_2 = [Variable("rel2", VariableMetatype.Relational, 4),
                          Variable("rel3", VariableMetatype.Relational, 5)]
        vn_1 = VariableNodeForTest("test_vn_1", rel_var_list_1)
        vn_2 = VariableNodeForTest("test_vn_2", rel_var_list_2)
        msg_shape_1 = (Size([3]), Size([]), Size([]), Size([]))
        msg_shape_2 = (Size([4, 5]), Size([]), Size([]), Size([]))
        ld_1 = LinkData(vn_1, fn, True, msg_shape_1)
        ld_2 = LinkData(vn_2, fn, False, msg_shape_2)

        fn.add_link(ld_1)
        with pytest.raises(AssertionError):
            fn.add_link(ld_2)

    def test_add_link_compatible_variables(self):
        # add linkdata with different part of variables shape defined
        fn = DFN("test_dfn")

        rel_var_list_1 = [Variable("rel1", VariableMetatype.Relational, 3)]
        param_var = Variable("p1", VariableMetatype.Parameter, 10)
        vn_1 = VariableNodeForTest("test_vn_1", rel_var_list_1, param_var)
        msg_shape_1 = (Size([3]), Size([10]), Size([]), Size([]))
        ld_1 = LinkData(vn_1, fn, True, msg_shape_1)

        rel_var_list_2 = [Variable("rel2", VariableMetatype.Relational, 3)]
        index_var_list = [Variable("i1", VariableMetatype.Indexing, 6)]
        ran_var_list = [Variable("ran1", VariableMetatype.Random, 1, (real,))]
        vn_2 = VariableNodeForTest("test_vn_2", rel_var_list_2,
                                   index_var_list=index_var_list, ran_var_list=ran_var_list)
        msg_shape_2 = (Size([3]), Size([]), Size([6]), Size([1]))
        ld_2 = LinkData(vn_2, fn, False, msg_shape_2)

        fn.add_link(ld_1)
        fn.add_link(ld_2)

    def test_compute_quiescence(self):
        # Test no message is sent when quiesced
        # Mock linkdata and message
        mock_in_ld, mock_out_ld = MagicMock(spec_set=LinkData), MagicMock(spec_set=LinkData)
        mock_msg = MagicMock(spec_set=Message)
        mock_in_ld.read = MagicMock(return_value=mock_msg)

        fn = DFN("test_dfn")
        fn.in_linkdata.append(mock_in_ld)
        fn.out_linkdata.append(mock_out_ld)

        with patch("pysigma.graphical.basic_nodes.DFN.quiescence", new_callable=PropertyMock) as mock_quiescence:
            mock_quiescence.return_value = True
            fn.compute()
            mock_in_ld.read.assert_not_called()
            mock_out_ld.write.assert_not_called()

    def test_compute_do(self):
        # Test when not quiesced and compute is carried out.
        mock_in_ld = MagicMock(spec_set=LinkData)
        mock_msg = MagicMock(spec_set=Message)
        mock_in_ld.read = MagicMock(return_value=mock_msg)

        mock_out_ld_list = [MagicMock(spec_set=LinkData) for i in range(5)]

        fn = DFN("test_dfn")
        fn.in_linkdata.append(mock_in_ld)
        fn.out_linkdata = mock_out_ld_list

        # Use patch to make sure the quiescence property returns False
        with patch("pysigma.graphical.basic_nodes.DFN.quiescence", new_callable=PropertyMock) as mock_quiescence:
            mock_quiescence.return_value = False
            fn.compute()
            for mock_out_ld in mock_out_ld_list:
                mock_in_ld.read.assert_called_once()
                mock_out_ld.write.assert_called_once_with(mock_msg)
