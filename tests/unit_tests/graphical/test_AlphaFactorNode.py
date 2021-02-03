"""
    Unit test for AlphaFactorNode class
"""
import pytest
from unittest.mock import MagicMock
import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, VariableNode, FactorNode
from pysigma.graphical.alpha_beta_nodes import AlphaFactorNode
from pysigma.utils import KnowledgeServer as KS
from pysigma.utils import DistributionServer as DS
from ...utils import random_message

from ...utils import generate_positive_definite


# Concrete subclass for testing
class AlphaFactorNodeForTest(AlphaFactorNode):

    def inward_compute(self, in_ld, out_ld):
        super(AlphaFactorNodeForTest, self).inward_compute(in_ld, out_ld)

    def outward_compute(self, in_ld, out_ld):
        super(AlphaFactorNodeForTest, self).outward_compute(in_ld, out_ld)


class TestAlphaFactorNode:

    def test_init(self):
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        assert afn.name == "test_alpha_factor_node"
        assert len(afn.labeled_ld_pair) == 0
        assert afn.ran_vars is None

    def test_add_link_missing_special_attribute(self):
        # Test that AssertionError is raised if the 'direction' special attribute is not specified
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        # Mock linkdata with missing special attribute
        ld = MagicMock(spec=LinkData)
        ld.attr = {}        # Empty special attribute dict
        ld.__repr__ = MagicMock(return_value="test_ld")     # Mock __repr__ to give the mock ld a mocked name

        with pytest.raises(AssertionError) as excinfo:
            afn.add_link(ld)

        assert str(excinfo.value) == "In test_alpha_factor_node: For an Alpha Factor Node, the linkdata test_ld " \
                                     "should specify a special attribute named 'direction', with value either 'inward' " \
                                     "or 'outward'."

    def test_add_link_wrong_special_attribute(self):
        # Test that AssertionError is raised if the 'direction' special attribute has wrong value
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        # Mock linkdata with missing special attribute
        ld = MagicMock(spec=LinkData)
        ld.attr = {'direction': 'something random'}  # Some random value of the special attribute
        ld.__repr__ = MagicMock(return_value="test_ld")  # Mock __repr__ to give the mock ld a mocked name

        with pytest.raises(AssertionError) as excinfo:
            afn.add_link(ld)

        assert str(excinfo.value) == "In test_alpha_factor_node: For an Alpha Factor Node, the linkdata test_ld " \
                                     "should specify a special attribute named 'direction', with value either 'inward' " \
                                     "or 'outward'."

    def test_add_link_conflicting_random_variables(self):
        # Test that AssertionError is raised if two linkdata added have different random variables.
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        # Mock random variables
        rv_1, rv_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        rv_1.__repr__, rv_2.__repr__ = MagicMock(return_value="test_random_variable_1"), \
                                       MagicMock(return_value="test_random_variable_1")
        # Mock vn
        vn_1, vn_2 = MagicMock(spec=VariableNode), MagicMock(spec=VariableNode)
        vn_1.ran_vars, vn_2.ran_vars = (rv_1, ), (rv_1, rv_2)
        # Mock linkdata
        ld_1, ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        ld_1.vn, ld_2.vn = vn_1, vn_2
        ld_1.fn, ld_2.fn = afn, afn
        ld_1.to_fn, ld_2.to_fn = True, False
        ld_1.attr, ld_2.attr = {'direction': 'inward'}, {'direction': 'inward'}
        ld_1.__repr__, ld_2.__repr__ = MagicMock(return_value="test_ld_1"), MagicMock(return_value="test_ld_2")

        with pytest.raises(AssertionError) as excinfo:
            afn.add_link(ld_1)
            afn.add_link(ld_2)

        assert str(excinfo.value) == "In test_alpha_factor_node: linkdata test_ld_2 has conflicting random variables. " \
                                     "This Alpha Factor Node infers random variables (test_random_variable_1,) from " \
                                     "other previously registered linkdata, but found different random variables " \
                                     "(test_random_variable_1, test_random_variable_1) in this linkdata."

    def test_add_link_duplicate_incoming_link(self):
        # Test that AssertionError is raised if multiple incoming links with same direction is registered
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        # Mock random variables
        rv_1, rv_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        rv_1.__repr__, rv_2.__repr__ = MagicMock(return_value="test_random_variable_1"), \
                                       MagicMock(return_value="test_random_variable_1")
        # Mock vn
        vn_1, vn_2 = MagicMock(spec=VariableNode), MagicMock(spec=VariableNode)
        vn_1.ran_vars, vn_2.ran_vars = (rv_1, rv_2), (rv_1, rv_2)
        # Mock linkdata
        ld_1, ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        ld_1.vn, ld_2.vn = vn_1, vn_2
        ld_1.fn, ld_2.fn = afn, afn
        ld_1.to_fn, ld_2.to_fn = True, True
        ld_1.attr, ld_2.attr = {'direction': 'inward'}, {'direction': 'inward'}
        ld_1.__repr__, ld_2.__repr__ = MagicMock(return_value="test_ld_1"), MagicMock(return_value="test_ld_2")

        with pytest.raises(AssertionError) as excinfo:
            afn.add_link(ld_1)
            afn.add_link(ld_2)

        assert str(excinfo.value) == "In test_alpha_factor_node: Attempting to register an incoming linkdata " \
                                     "test_ld_2 with inward conditional message propagation direction, while already " \
                                     "having another incoming linkdata test_ld_1 with the same direction."

    def test_add_link_duplicate_outgoing_link(self):
        # Test that AssertionError is raised if multiple incoming links with same direction is registered
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        # Mock random variables
        rv_1, rv_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        rv_1.__repr__, rv_2.__repr__ = MagicMock(return_value="test_random_variable_1"), \
                                       MagicMock(return_value="test_random_variable_1")
        # Mock vn
        vn_1, vn_2 = MagicMock(spec=VariableNode), MagicMock(spec=VariableNode)
        vn_1.ran_vars, vn_2.ran_vars = (rv_1, rv_2), (rv_1, rv_2)
        # Mock linkdata
        ld_1, ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        ld_1.vn, ld_2.vn = vn_1, vn_2
        ld_1.fn, ld_2.fn = afn, afn
        ld_1.to_fn, ld_2.to_fn = False, False
        ld_1.attr, ld_2.attr = {'direction': 'outward'}, {'direction': 'outward'}
        ld_1.__repr__, ld_2.__repr__ = MagicMock(return_value="test_ld_1"), MagicMock(return_value="test_ld_2")

        with pytest.raises(AssertionError) as excinfo:
            afn.add_link(ld_1)
            afn.add_link(ld_2)

        assert str(excinfo.value) == "In test_alpha_factor_node: Attempting to register an outgoing linkdata " \
                                     "test_ld_2 with outward conditional message propagation direction, while already " \
                                     "having another outgoing linkdata test_ld_1 with the same direction."

    def test_add_link_correct_execution_single_pair(self):
        pass


