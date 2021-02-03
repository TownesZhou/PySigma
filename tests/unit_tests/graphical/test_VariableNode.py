"""
    Unit test for VariableNode class
"""
import pytest
from unittest.mock import MagicMock
from torch import Size
from torch.distributions.constraints import real

from pysigma.graphical.basic_nodes import FactorNode, VariableNode, LinkData
from pysigma.defs import Variable, VariableMetatype


# Subclassing abstract base VariableNode class for testing
class VariableNodeForTest(VariableNode):
    def compute(self):
        super(VariableNodeForTest, self).compute()


class TestVariableNode:

    def test_init_correct_batch_shape(self):
        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3),
                        Variable("rel2", VariableMetatype.Relational, 4),
                        Variable("rel3", VariableMetatype.Relational, 5)]
        vn = VariableNodeForTest("test_vn", rel_var_list)

        assert vn.b_shape == Size([3, 4, 5])
        assert vn.p_shape == Size([]) and vn.s_shape == Size([]) and vn.e_shape == Size([])

    def test_init_correct_param_shape(self):
        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3),
                        Variable("rel2", VariableMetatype.Relational, 4),
                        Variable("rel3", VariableMetatype.Relational, 5)]
        param_var = Variable("p1", VariableMetatype.Parameter, 10)
        vn = VariableNodeForTest("test_vn", rel_var_list, param_var=param_var)

        assert vn.p_shape == Size([10])
        assert vn.s_shape == Size([]) and vn.e_shape == Size([])

    def test_init_correct_sample_shape_and_event_shape(self):
        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3),
                        Variable("rel2", VariableMetatype.Relational, 4),
                        Variable("rel3", VariableMetatype.Relational, 5)]
        index_var_list = [Variable("i1", VariableMetatype.Indexing, 6),
                          Variable("i2", VariableMetatype.Indexing, 7),
                          Variable("i3", VariableMetatype.Indexing, 8)]
        ran_var_list = [Variable("ran1", VariableMetatype.Random, 1, (real,)),
                        Variable("ran2", VariableMetatype.Random, 2, (real,)),
                        Variable("ran3", VariableMetatype.Random, 3, (real,))]
        vn = VariableNodeForTest("test_vn", rel_var_list, index_var_list=index_var_list, ran_var_list=ran_var_list)

        assert vn.s_shape == Size([6, 7, 8]) and vn.e_shape == Size([1, 2, 3])

    def test_add_link_not_connected_to_self(self):
        # Test adding a linkdata that is not connected to self
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([3, 4, 5]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3),
                        Variable("rel2", VariableMetatype.Relational, 4),
                        Variable("rel3", VariableMetatype.Relational, 5)]
        vn = VariableNodeForTest("test_vn", rel_var_list)

        with pytest.raises(AssertionError):
            vn.add_link(ld)

    def test_add_link_incompatible_shape(self):
        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3),
                        Variable("rel2", VariableMetatype.Relational, 4),
                        Variable("rel3", VariableMetatype.Relational, 5)]
        param_var = Variable("p1", VariableMetatype.Parameter, 10)
        index_var_list = [Variable("i1", VariableMetatype.Indexing, 6),
                          Variable("i2", VariableMetatype.Indexing, 7),
                          Variable("i3", VariableMetatype.Indexing, 8)]
        ran_var_list = [Variable("ran1", VariableMetatype.Random, 1, (real,)),
                        Variable("ran2", VariableMetatype.Random, 2, (real,)),
                        Variable("ran3", VariableMetatype.Random, 3, (real,))]
        vn = VariableNodeForTest("test_vn", rel_var_list, param_var, index_var_list, ran_var_list)

        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([3, 4, 5]), Size([10]), Size([6, 7, 10]), Size([1, 2, 3]))
        ld = LinkData(vn, mock_fn, True, msg_shape)

        with pytest.raises(AssertionError):
            vn.add_link(ld)

    def test_add_link_compatible_default_shape(self):
        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3),
                        Variable("rel2", VariableMetatype.Relational, 4),
                        Variable("rel3", VariableMetatype.Relational, 5)]
        vn = VariableNodeForTest("test_vn", rel_var_list)

        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([3, 4, 5]), Size([10]), Size([6, 7, 10]), Size([1, 2, 3]))
        ld = LinkData(vn, mock_fn, True, msg_shape)

        vn.add_link(ld)

    def test_add_link_duplicate(self):
        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3),
                        Variable("rel2", VariableMetatype.Relational, 4),
                        Variable("rel3", VariableMetatype.Relational, 5)]
        vn = VariableNodeForTest("test_vn", rel_var_list)

        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([3, 4, 5]), Size([10]), Size([6, 7, 10]), Size([1, 2, 3]))
        ld = LinkData(vn, mock_fn, True, msg_shape)

        vn.out_linkdata.append(ld)

        vn.add_link(ld)
        assert len(vn.out_linkdata) == 1 and vn.out_linkdata[0] is ld

    def test_add_link_correct_incoming(self):
        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3),
                        Variable("rel2", VariableMetatype.Relational, 4),
                        Variable("rel3", VariableMetatype.Relational, 5)]
        vn = VariableNodeForTest("test_vn", rel_var_list)

        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([3, 4, 5]), Size([10]), Size([6, 7, 10]), Size([1, 2, 3]))
        ld = LinkData(vn, mock_fn, False, msg_shape)

        vn.add_link(ld)
        assert len(vn.in_linkdata) == 1 and vn.in_linkdata[0] is ld

    def test_add_link_correct_outgoing(self):
        rel_var_list = [Variable("rel1", VariableMetatype.Relational, 3),
                        Variable("rel2", VariableMetatype.Relational, 4),
                        Variable("rel3", VariableMetatype.Relational, 5)]
        vn = VariableNodeForTest("test_vn", rel_var_list)

        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([3, 4, 5]), Size([10]), Size([6, 7, 10]), Size([1, 2, 3]))
        ld = LinkData(vn, mock_fn, True, msg_shape)

        vn.add_link(ld)
        assert len(vn.out_linkdata) == 1 and vn.out_linkdata[0] is ld
