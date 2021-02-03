"""
    Unit tests for FactorNode class
"""
import pytest
from unittest.mock import MagicMock
from torch import Size

from pysigma.graphical.basic_nodes import FactorNode, VariableNode, LinkData


# Subclass abstract base FactorNode to carry out tests
class FactorNodeForTest(FactorNode):
    def compute(self):
        super(FactorNodeForTest, self).compute()


class TestFactorNode:

    def test_add_link_not_connected_to_self(self):
        # Test add a link where the factor node is not self
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        fn = FactorNodeForTest("test_fn")
        with pytest.raises(AssertionError):
            fn.add_link(ld)

    def test_add_link_correct_duplicate(self):
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))

        fn = FactorNodeForTest("test_fn")
        ld = LinkData(mock_vn, fn, True, msg_shape)
        fn.in_linkdata.append(ld)

        fn.add_link(ld)
        assert len(fn.in_linkdata) == 1 and fn.in_linkdata[0] is ld

    def test_add_link_correct_incoming(self):
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))

        fn = FactorNodeForTest("test_fn")
        ld = LinkData(mock_vn, fn, True, msg_shape)

        fn.add_link(ld)
        assert len(fn.in_linkdata) == 1 and fn.in_linkdata[0] is ld
        assert len(fn.out_linkdata) == 0

    def test_add_link_correct_outgoing(self):
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))

        fn = FactorNodeForTest("test_fn")
        ld = LinkData(mock_vn, fn, False, msg_shape)

        fn.add_link(ld)
        assert len(fn.out_linkdata) == 1 and fn.out_linkdata[0] is ld
        assert len(fn.in_linkdata) == 0


