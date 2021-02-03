"""
    Unit test for AlphaFactorNode class
"""
import pytest
from unittest.mock import MagicMock

from pysigma.defs import Message, Variable
from pysigma.graphical.basic_nodes import LinkData, VariableNode, NodeConfigurationError
from pysigma.graphical.alpha_beta_nodes import AlphaFactorNode


# Concrete subclass for testing
class AlphaFactorNodeForTest(AlphaFactorNode):

    def inward_compute(self, in_ld, out_ld):
        # Ad-hoc behavior for testing. Simply relay in_ld message to out_ld
        msg = in_ld.read()
        out_ld.write(msg)

    def outward_compute(self, in_ld, out_ld):
        # Ad-hoc behavior for testing. Simply relay in_ld message to out_ld
        msg = in_ld.read()
        out_ld.write(msg)


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

    def test_add_link_correct_single_pair(self):
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
        ld_1.to_fn, ld_2.to_fn = True, False
        ld_1.attr, ld_2.attr = {'direction': 'inward'}, {'direction': 'inward'}
        ld_1.__repr__, ld_2.__repr__ = MagicMock(return_value="test_ld_1"), MagicMock(return_value="test_ld_2")

        afn.add_link(ld_1)
        afn.add_link(ld_2)

        # Incoming link should be the first linkdata in the tuple
        assert len(tuple(afn.labeled_ld_pair.items())) == 1 and 'inward' in afn.labeled_ld_pair.keys()
        assert afn.labeled_ld_pair['inward'] == (ld_1, ld_2)

    def test_add_link_correct_double_pair(self):
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
        ld_3, ld_4 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        ld_1.vn, ld_2.vn = vn_1, vn_2
        ld_3.vn, ld_4.vn = vn_1, vn_2
        ld_1.fn, ld_2.fn = afn, afn
        ld_3.fn, ld_4.fn = afn, afn
        ld_1.to_fn, ld_2.to_fn = True, False
        ld_3.to_fn, ld_4.to_fn = False, True
        ld_1.attr, ld_2.attr = {'direction': 'inward'}, {'direction': 'inward'}
        ld_3.attr, ld_4.attr = {'direction': 'outward'}, {'direction': 'outward'}
        ld_1.__repr__, ld_2.__repr__ = MagicMock(return_value="test_ld_1"), MagicMock(return_value="test_ld_2")
        ld_3.__repr__, ld_4.__repr__ = MagicMock(return_value="test_ld_3"), MagicMock(return_value="test_ld_4")

        afn.add_link(ld_1)
        afn.add_link(ld_2)
        afn.add_link(ld_3)
        afn.add_link(ld_4)

        # Incoming link should be the first linkdata in the tuple
        assert len(tuple(afn.labeled_ld_pair.items())) == 2 and \
               'inward' in afn.labeled_ld_pair.keys() and \
               'outward' in afn.labeled_ld_pair.keys()
        assert afn.labeled_ld_pair['inward'] == (ld_1, ld_2) and afn.labeled_ld_pair['outward'] == (ld_4, ld_3)

    def test_ill_configuration_1(self):
        # Test that NodeConfigurationError is raised if no linkdata is registered
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        with pytest.raises(NodeConfigurationError) as excinfo:
            afn.compute()

        assert str(excinfo.value) == "Wrong configuration for node test_alpha_factor_node: no registered linkdata."

    def test_ill_configuration_2(self):
        # Test that NodeConfigurationError is raised if linkdata pair is incomplete
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        # Test with complete inward pair but incomplete outward pair
        # Mock random variables
        rv_1, rv_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        rv_1.__repr__, rv_2.__repr__ = MagicMock(return_value="test_random_variable_1"), \
                                       MagicMock(return_value="test_random_variable_1")
        # Mock vn
        vn_1, vn_2 = MagicMock(spec=VariableNode), MagicMock(spec=VariableNode)
        vn_1.ran_vars, vn_2.ran_vars = (rv_1, rv_2), (rv_1, rv_2)
        # Mock linkdata
        ld_1, ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        ld_3 = MagicMock(spec=LinkData)
        ld_1.vn, ld_2.vn = vn_1, vn_2
        ld_3.vn = vn_1
        ld_1.fn, ld_2.fn = afn, afn
        ld_3.fn = afn
        ld_1.to_fn, ld_2.to_fn = True, False
        ld_3.to_fn = False
        ld_1.attr, ld_2.attr = {'direction': 'inward'}, {'direction': 'inward'}
        ld_3.attr = {'direction': 'outward'}
        ld_1.__repr__, ld_2.__repr__ = MagicMock(return_value="test_ld_1"), MagicMock(return_value="test_ld_2")
        ld_3.__repr__ = MagicMock(return_value="test_ld_3")

        afn.add_link(ld_1)
        afn.add_link(ld_2)
        afn.add_link(ld_3)

        with pytest.raises(NodeConfigurationError) as excinfo:
            afn.compute()

        assert str(excinfo.value) == "Wrong configuration for node test_alpha_factor_node: incomplete pair of " \
                                     "linkdata. Found 2 inward linkdata and 1 outward linkdata."

    def test_quiescence_no_compute(self):
        # Test that compute is not carried out when quiesced
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        inward_in_ld, inward_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        outward_in_ld, outward_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)

        afn.in_linkdata = [inward_in_ld, outward_in_ld]
        afn.out_linkdata = [inward_out_ld, outward_out_ld]
        afn.labeled_ld_pair['inward'] = (inward_in_ld, inward_out_ld)
        afn.labeled_ld_pair['outward'] = (outward_in_ld, outward_out_ld)

        # Set quiescence condition: incoming linkdata has no new message
        inward_in_ld.new, outward_in_ld.new = False, False

        afn.compute()

        # Check no read
        inward_in_ld.read.assert_not_called()
        inward_out_ld.write.assert_not_called()
        outward_in_ld.read.assert_not_called()
        outward_out_ld.write.assert_not_called()

    def test_inward_compute(self):
        # Test that when having new message, inward_compute() is correctly called
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        # Mock linkdata
        inward_in_ld, inward_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        outward_in_ld, outward_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        # Mock message
        mock_msg = MagicMock(spec=Message)
        inward_in_ld.read.return_value = mock_msg
        outward_in_ld.read.return_value = mock_msg

        afn.in_linkdata = [inward_in_ld, outward_in_ld]
        afn.out_linkdata = [inward_out_ld, outward_out_ld]
        afn.labeled_ld_pair['inward'] = (inward_in_ld, inward_out_ld)
        afn.labeled_ld_pair['outward'] = (outward_in_ld, outward_out_ld)

        # Set quiescence state
        inward_in_ld.new, outward_in_ld.new = True, False

        afn.compute()

        # Check correct execution and message propagation
        inward_in_ld.read.assert_called_once()
        inward_out_ld.write.assert_called_once_with(mock_msg)
        outward_in_ld.read.assert_not_called()
        outward_out_ld.write.assert_not_called()

    def test_outward_compute(self):
        # Test that when having new message, inward_compute() is correctly called
        afn = AlphaFactorNodeForTest("test_alpha_factor_node")

        # Mock linkdata
        inward_in_ld, inward_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        outward_in_ld, outward_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        # Mock message
        mock_msg = MagicMock(spec=Message)
        inward_in_ld.read.return_value = mock_msg
        outward_in_ld.read.return_value = mock_msg

        afn.in_linkdata = [inward_in_ld, outward_in_ld]
        afn.out_linkdata = [inward_out_ld, outward_out_ld]
        afn.labeled_ld_pair['inward'] = (inward_in_ld, inward_out_ld)
        afn.labeled_ld_pair['outward'] = (outward_in_ld, outward_out_ld)

        # Set quiescence state
        inward_in_ld.new, outward_in_ld.new = False, True

        afn.compute()

        # Check correct execution and message propagation
        inward_in_ld.read.assert_not_called()
        inward_out_ld.write.assert_not_called()
        outward_in_ld.read.assert_called_once()
        outward_out_ld.write.assert_called_once_with(mock_msg)
