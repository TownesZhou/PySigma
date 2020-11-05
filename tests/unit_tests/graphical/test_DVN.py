"""
    Unit tests for DVN class
"""
import pytest
from unittest.mock import patch, PropertyMock, MagicMock
from torch import Size
from torch.distributions.constraints import real

from pysigma.defs import Message
from pysigma.graphical.basic_nodes import DVN, FactorNode, VariableNode, LinkData
from pysigma.defs import Variable, VariableMetatype
from .test_VariableNode import VariableNodeForTest


class TestDVN():

    def test_add_link_more_than_one_incoming(self):
        # Test add more than one incoming linkdata
        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3)]
        vn = DVN("test_dvn", rel_var_list)

        # Mock linkdata
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([3]), Size([10]), Size([6, 7, 10]), Size([1, 2, 3]))
        ld_1 = LinkData(vn, mock_fn, False, msg_shape)
        ld_2 = LinkData(vn, mock_fn, False, msg_shape)

        vn.add_link(ld_1)
        with pytest.raises(AssertionError):
            vn.add_link(ld_2)

    def test_compute_quiesced(self):
        # Test no message is sent when quiesced
        # Mock linkdata and message
        mock_in_ld, mock_out_ld = MagicMock(spec_set=LinkData), MagicMock(spec_set=LinkData)
        mock_msg = MagicMock(spec_set=Message)
        mock_in_ld.read = MagicMock(return_value=mock_msg)

        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3)]
        dvn = DVN("test_dvn", rel_var_list)
        dvn.in_linkdata.append(mock_in_ld)
        dvn.out_linkdata.append(mock_out_ld)

        with patch("pysigma.graphical.basic_nodes.DVN.quiescence", new_callable=PropertyMock) as mock_quiescence:
            mock_quiescence.return_value = True
            dvn.compute()
            mock_in_ld.read.assert_not_called()
            mock_out_ld.write.assert_not_called()

    def test_compute_do(self):
        # Test when not quiesced and compute is carried out.
        mock_in_ld = MagicMock(spec_set=LinkData)
        mock_msg = MagicMock(spec_set=Message)
        mock_in_ld.read = MagicMock(return_value=mock_msg)

        mock_out_ld_list = [MagicMock(spec_set=LinkData) for i in range(5)]

        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3)]
        dvn = DVN("test_dvn", rel_var_list)
        dvn.in_linkdata.append(mock_in_ld)
        dvn.out_linkdata = mock_out_ld_list

        # Use patch to make sure the quiescence property returns False
        with patch("pysigma.graphical.basic_nodes.DVN.quiescence", new_callable=PropertyMock) as mock_quiescence:
            mock_quiescence.return_value = False
            dvn.compute()
            for mock_out_ld in mock_out_ld_list:
                mock_in_ld.read.assert_called_once()
                mock_out_ld.write.assert_called_once_with(mock_msg)
