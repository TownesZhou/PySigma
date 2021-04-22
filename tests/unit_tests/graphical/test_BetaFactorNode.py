"""
    Unit test for abstract BetaFactorNode class
"""
import pytest
from typing import List
from unittest.mock import MagicMock

from pysigma.defs import Message, Variable
from pysigma.graphical.basic_nodes import LinkData, VariableNode, NodeConfigurationError
from pysigma.graphical.beta_nodes import BetaFactorNode


# Concrete subclass for testing
class BetaFactorNodeForTest(BetaFactorNode):

    def __init__(self, name):
        super(BetaFactorNodeForTest, self).__init__(name)
        # Testing purpose only attribute
        # For i-th out_ld, the msg_map[i]-th in_ld's message will be relayed to it
        self.inward_msg_map: List[int] = []
        self.outward_msg_map: List[int] = []

    def inward_compute(self, in_lds: List[LinkData], out_lds: List[LinkData]):
        for out_ld_id, in_ld_id in enumerate(self.inward_msg_map):
            out_lds[out_ld_id].write(in_lds[in_ld_id].read())

    def outward_compute(self, in_lds: List[LinkData], out_lds: List[LinkData]):
        for out_ld_id, in_ld_id in enumerate(self.outward_msg_map):
            out_lds[out_ld_id].write(in_lds[in_ld_id].read())


class TestBetaFactorNode:

    def test_init(self):
        bfn = BetaFactorNodeForTest('test_bfn')

        # Check attributes
        assert isinstance(bfn.labeled_ld_group, dict)
        assert len(bfn.labeled_ld_group) == 0
        assert bfn.rel_vars is None

    def test_add_link_missing_special_attribute(self):
        # Test an AssertionError is raised if linkdata does not have the special attribute
        bfn = BetaFactorNodeForTest('test_bfn')
        # Mock linkdata
        ld = MagicMock(spec=LinkData)
        ld.attr = {}    # Give an empty special attribute dict
        ld.__repr__ = MagicMock(return_value='test_mock_ld')        # Mock __repr__ to give the mock ld a mocked name

        with pytest.raises(AssertionError) as excinfo:
            bfn.add_link(ld)

        assert str(excinfo.value) == "In test_bfn: For a Beta Factor Node, the linkdata 'test_mock_ld' should specify "\
                                     "a special attribute named 'direction', with value either 'inward' or 'outward'."

    def test_add_link_wrong_special_attribute_value(self):
        # Test an AssertionError is raised if linkdata does not have the special attribute
        bfn = BetaFactorNodeForTest('test_bfn')
        # Mock linkdata
        ld = MagicMock(spec=LinkData)
        ld.attr = {'direction': 'something random'}  # Give an special attribute dict with wrong value
        ld.__repr__ = MagicMock(return_value='test_mock_ld')  # Mock __repr__ to give the mock ld a mocked name

        with pytest.raises(AssertionError) as excinfo:
            bfn.add_link(ld)

        assert str(excinfo.value) == "In test_bfn: For a Beta Factor Node, the linkdata 'test_mock_ld' should specify "\
                                     "a special attribute named 'direction', with value either 'inward' or 'outward'."

    def test_add_link_conflicting_relational_variables(self):
        # Test adding linkdata that connects to variable nodes with different relational variables
        bfn = BetaFactorNodeForTest('test_bfn')
        # Mock relational variables
        mock_relvar_1, mock_relvar_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        mock_relvar_1.__repr__, mock_relvar_2.__repr__ = MagicMock(return_value='mock_relvar_1'), \
                                                         MagicMock(return_value='mock_relvar_2')
        # Mock variable nodes
        vn_1, vn_2 = MagicMock(spec=VariableNode), MagicMock(spec=VariableNode)
        vn_1.rel_vars, vn_2.rel_vars = (mock_relvar_1, ), (mock_relvar_1, mock_relvar_2)
        # Mock linkdata
        ld_1, ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        ld_1.vn, ld_2.vn = vn_1, vn_2
        ld_1.fn, ld_2.fn = bfn, bfn
        ld_1.to_fn, ld_2.to_fn = True, False
        ld_1.attr, ld_2.attr = {'direction': 'inward'}, {'direction': 'inward'}
        ld_1.__repr__, ld_2.__repr__ = MagicMock(return_value="test_ld_1"), MagicMock(return_value="test_ld_2")

        with pytest.raises(AssertionError) as excinfo:
            bfn.add_link(ld_1)
            bfn.add_link(ld_2)

        assert str(excinfo.value) == "In test_bfn: linkdata 'test_ld_2' has conflicting relational variables. This " \
                                     "Beta Factor Node infers relational variables (mock_relvar_1,) from other " \
                                     "previously registered linkdata, but found different relational variables " \
                                     "(mock_relvar_1, mock_relvar_2) in this linkdata."

    def test_add_link_correct_single_group(self):
        # Test adding a single directional group
        bfn = BetaFactorNodeForTest('test_bfn')
        # Mock relational variables
        mock_relvar_1, mock_relvar_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        mock_relvar_1.__repr__, mock_relvar_2.__repr__ = MagicMock(return_value='mock_relvar_1'), \
                                                         MagicMock(return_value='mock_relvar_2')
        # Mock variable nodes\
        vn_list = [MagicMock(spec=VariableNode) for i in range(4)]
        for vn in vn_list:
            vn.rel_vars = (mock_relvar_1, mock_relvar_2)
        # Mock linkdata
        # 1 incoming, 3 outgoing
        in_ld_1 = MagicMock(spec=LinkData)
        out_ld_1, out_ld_2, out_ld_3 = MagicMock(spec=LinkData), MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld_1.vn = vn_list[0]
        out_ld_1.vn, out_ld_2.vn, out_ld_3.vn = vn_list[1], vn_list[2], vn_list[3]
        in_ld_1.fn = bfn
        out_ld_1.fn, out_ld_2.fn, out_ld_3.fn = bfn, bfn, bfn
        in_ld_1.to_fn = True
        out_ld_1.to_fn, out_ld_2.to_fn, out_ld_3.to_fn = False, False, False
        in_ld_1.attr = {'direction': 'inward'}
        out_ld_1.attr, out_ld_2.attr, out_ld_3.attr = {'direction': 'inward'}, {'direction': 'inward'}, \
                                                      {'direction': 'inward'}
        in_ld_1.__repr__ = MagicMock(return_value="test_in_ld_1")
        out_ld_1.__repr__, out_ld_2.__repr__, out_ld_3.__repr__ = MagicMock(return_value="test_out_ld_1"), \
                                                                  MagicMock(return_value="test_out_ld_1"), \
                                                                  MagicMock(return_value="test_out_ld_1")

        bfn.add_link(in_ld_1)
        bfn.add_link(out_ld_1)
        bfn.add_link(out_ld_2)
        bfn.add_link(out_ld_3)

        assert len(bfn.labeled_ld_group) == 1
        assert len(bfn.labeled_ld_group['inward']) == 2
        assert len(bfn.labeled_ld_group['inward'][True]) == 1 and len(bfn.labeled_ld_group['inward'][False]) == 3
        assert bfn.labeled_ld_group['inward'][True] == [in_ld_1]
        assert bfn.labeled_ld_group['inward'][False] == [out_ld_1, out_ld_2, out_ld_3]

    def test_add_link_correct_double_groups(self):
        # Adding two directional groups
        bfn = BetaFactorNodeForTest('test_bfn')
        # Mock relational variables
        mock_relvar_1, mock_relvar_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        mock_relvar_1.__repr__, mock_relvar_2.__repr__ = MagicMock(return_value='mock_relvar_1'), \
                                                         MagicMock(return_value='mock_relvar_2')
        # Mock variable nodes\
        vn_list = [MagicMock(spec=VariableNode) for i in range(5)]
        for vn in vn_list:
            vn.rel_vars = (mock_relvar_1, mock_relvar_2)
        # Mock linkdata
        # 1 incoming, 3 outgoing
        in_ld_1, in_ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        out_ld_1, out_ld_2, out_ld_3 = MagicMock(spec=LinkData), MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld_1.vn, in_ld_2.vn = vn_list[0], vn_list[4]
        out_ld_1.vn, out_ld_2.vn, out_ld_3.vn = vn_list[1], vn_list[2], vn_list[3]
        in_ld_1.fn, in_ld_2.fn = bfn, bfn
        out_ld_1.fn, out_ld_2.fn, out_ld_3.fn = bfn, bfn, bfn
        in_ld_1.to_fn, in_ld_2.to_fn = True, True
        out_ld_1.to_fn, out_ld_2.to_fn, out_ld_3.to_fn = False, False, False
        in_ld_1.attr, in_ld_2.attr = {'direction': 'outward'}, {'direction': 'inward'}
        out_ld_1.attr, out_ld_2.attr, out_ld_3.attr = {'direction': 'outward'}, {'direction': 'outward'}, \
                                                      {'direction': 'inward'}
        in_ld_1.__repr__, in_ld_2.__repr__ = MagicMock(return_value="test_in_ld_1"), \
                                             MagicMock(return_value="test_in_ld_2")
        out_ld_1.__repr__, out_ld_2.__repr__, out_ld_3.__repr__ = MagicMock(return_value="test_out_ld_1"), \
                                                                  MagicMock(return_value="test_out_ld_1"), \
                                                                  MagicMock(return_value="test_out_ld_1")

        bfn.add_link(in_ld_1)
        bfn.add_link(out_ld_3)
        bfn.add_link(out_ld_1)
        bfn.add_link(in_ld_2)
        bfn.add_link(out_ld_2)

        assert len(bfn.labeled_ld_group) == 2
        assert bfn.labeled_ld_group['inward'][True] == [in_ld_2]
        assert bfn.labeled_ld_group['inward'][False] == [out_ld_3]
        assert bfn.labeled_ld_group['outward'][True] == [in_ld_1]
        assert bfn.labeled_ld_group['outward'][False] == [out_ld_1, out_ld_2]

    def test_ill_configuration_no_directional_group(self):
        # Test an ill-configured compute: no directional group has been registered.
        bfn = BetaFactorNodeForTest('test_bfn')

        with pytest.raises(NodeConfigurationError) as excinfo:
            bfn.compute()

        assert str(excinfo.value) == "Wrong configuration for node test_bfn: no registered linkdata."

    def test_ill_configuration_no_incoming_linkdata(self):
        # Test an ill-configured compute: no incoming linkdata in one of the directional groups
        bfn = BetaFactorNodeForTest('test_bfn')
        # Mock relational variables
        mock_relvar_1, mock_relvar_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        mock_relvar_1.__repr__, mock_relvar_2.__repr__ = MagicMock(return_value='mock_relvar_1'), \
                                                         MagicMock(return_value='mock_relvar_2')
        # Mock variable nodes\
        vn_list = [MagicMock(spec=VariableNode) for i in range(4)]
        for vn in vn_list:
            vn.rel_vars = (mock_relvar_1, mock_relvar_2)
        # Mock linkdata
        # 1 incoming, 3 outgoing
        out_ld_1 = MagicMock(spec=LinkData)
        out_ld_1.vn = vn_list[1]
        out_ld_1.fn = bfn
        out_ld_1.to_fn = False
        out_ld_1.attr = {'direction': 'inward'}
        out_ld_1.__repr__ = MagicMock(return_value="test_out_ld_1")

        bfn.add_link(out_ld_1)

        with pytest.raises(NodeConfigurationError) as excinfo:
            bfn.compute()

        assert str(excinfo.value) == "Wrong configuration for node test_bfn: incomplete group of linkdata. For the " \
                                     "group of linkdata with message propagation direction 'inward', there is no " \
                                     "incoming linkdata."

    def test_ill_configuration_no_outgoing_linkdata(self):
        # Test an ill-configured compute: no outgoing linkdata in one of the directional groups
        bfn = BetaFactorNodeForTest('test_bfn')
        # Mock relational variables
        mock_relvar_1, mock_relvar_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        mock_relvar_1.__repr__, mock_relvar_2.__repr__ = MagicMock(return_value='mock_relvar_1'), \
                                                         MagicMock(return_value='mock_relvar_2')
        # Mock variable nodes\
        vn_list = [MagicMock(spec=VariableNode) for i in range(4)]
        for vn in vn_list:
            vn.rel_vars = (mock_relvar_1, mock_relvar_2)
        # Mock linkdata
        # 1 incoming, 3 outgoing
        in_ld_1 = MagicMock(spec=LinkData)
        in_ld_1.vn = vn_list[1]
        in_ld_1.fn = bfn
        in_ld_1.to_fn = True
        in_ld_1.attr = {'direction': 'outward'}
        in_ld_1.__repr__ = MagicMock(return_value="test_out_ld_1")

        bfn.add_link(in_ld_1)

        with pytest.raises(NodeConfigurationError) as excinfo:
            bfn.compute()

        assert str(excinfo.value) == "Wrong configuration for node test_bfn: incomplete group of linkdata. For the " \
                                     "group of linkdata with message propagation direction 'outward', there is no " \
                                     "outgoing linkdata."

    def test_quiescence_no_compute(self):
        # Test that no compute is carried out when the node is in quiescence state
        # inward group: 3 incoming and 1 outgoing links
        # outward group: 2 incoming and 1 outgoing links
        bfn = BetaFactorNodeForTest('test_bfn')
        num_inward_in, num_outward_in = 3, 2
        # Mock relational variables
        mock_relvar_1, mock_relvar_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        mock_relvar_1.__repr__, mock_relvar_2.__repr__ = MagicMock(return_value='mock_relvar_1'), \
                                                         MagicMock(return_value='mock_relvar_2')
        # Mock variable nodes
        inward_vn_list, outward_vn_list = [MagicMock(spec=VariableNode) for i in range(num_inward_in + 1)], \
                                          [MagicMock(spec=VariableNode) for i in range(num_outward_in + 1)]
        for vn in inward_vn_list:
            vn.rel_vars = (mock_relvar_1, mock_relvar_2)
        for vn in outward_vn_list:
            vn.rel_vars = (mock_relvar_1, mock_relvar_2)
        # Mock linkdata
        inward_incoming_lds, outward_incoming_lds = [MagicMock(spec=LinkData) for i in range(num_inward_in)], \
                                                    [MagicMock(spec=LinkData) for i in range(num_outward_in)]
        inward_outgoing_ld, outward_outgoing_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        for inward_in_ld, in_vn in zip(inward_incoming_lds, inward_vn_list[:-1]):
            inward_in_ld.vn = in_vn
            inward_in_ld.fn = bfn
            inward_in_ld.to_fn = True
            inward_in_ld.attr = {'direction': 'inward'}
            inward_in_ld.__repr__ = MagicMock(return_value='test_inward_in_ld')
            inward_in_ld.new = False        # Set new message state
        for outward_in_ld, in_vn in zip(outward_incoming_lds, outward_vn_list[:-1]):
            outward_in_ld.vn = in_vn
            outward_in_ld.fn = bfn
            outward_in_ld.to_fn = True
            outward_in_ld.attr = {'direction': 'outward'}
            outward_in_ld.__repr__ = MagicMock(return_value='test_outward_in_ld')
            outward_in_ld.new = False       # Set new message state
        inward_outgoing_ld.vn, outward_outgoing_ld.vn = inward_vn_list[-1], outward_vn_list[-1]
        inward_outgoing_ld.fn, outward_outgoing_ld.fn = bfn, bfn
        inward_outgoing_ld.to_fn, outward_outgoing_ld.to_fn = False, False
        inward_outgoing_ld.attr, outward_outgoing_ld.attr = {'direction': 'inward'}, {'direction': 'outward'}
        inward_outgoing_ld.__repr__, outward_outgoing_ld.__repr__ = MagicMock(return_value='test_inward_in_ld'), \
                                                                    MagicMock(return_value='test_outward_in_ld')

        for ld in inward_incoming_lds:
            bfn.add_link(ld)
        for ld in outward_incoming_lds:
            bfn.add_link(ld)
        bfn.add_link(inward_outgoing_ld)
        bfn.add_link(outward_outgoing_ld)

        # Set compute message map
        bfn.inward_msg_map, bfn.outward_msg_map = [0], [0]  # Take the first incoming linkdata's message

        bfn.compute()

        # Check no read
        for ld in inward_incoming_lds:
            ld.read.assert_not_called()
        for ld in outward_incoming_lds:
            ld.read.assert_not_called()

    def test_inward_compute(self):
        # Test that inward_compute() is called when there are new inward incoming messages
        # Check that the outward_compute() is not called when there is no new outward incoming message
        bfn = BetaFactorNodeForTest('test_bfn')
        num_inward_in, num_outward_in = 3, 2
        # Mock message
        mock_msgs = [MagicMock(spec=Message) for i in range(num_inward_in)]
        # Mock relational variables
        mock_relvar_1, mock_relvar_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        mock_relvar_1.__repr__, mock_relvar_2.__repr__ = MagicMock(return_value='mock_relvar_1'), \
                                                         MagicMock(return_value='mock_relvar_2')
        # Mock variable nodes
        inward_vn_list, outward_vn_list = [MagicMock(spec=VariableNode) for i in range(num_inward_in + 1)], \
                                          [MagicMock(spec=VariableNode) for i in range(num_outward_in + 1)]
        for vn in inward_vn_list:
            vn.rel_vars = (mock_relvar_1, mock_relvar_2)
        for vn in outward_vn_list:
            vn.rel_vars = (mock_relvar_1, mock_relvar_2)
        # Mock linkdata
        inward_incoming_lds, outward_incoming_lds = [MagicMock(spec=LinkData) for i in range(num_inward_in)], \
                                                    [MagicMock(spec=LinkData) for i in range(num_outward_in)]
        inward_outgoing_ld, outward_outgoing_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        for inward_in_ld, in_vn in zip(inward_incoming_lds, inward_vn_list[:-1]):
            inward_in_ld.vn = in_vn
            inward_in_ld.fn = bfn
            inward_in_ld.to_fn = True
            inward_in_ld.attr = {'direction': 'inward'}
            inward_in_ld.__repr__ = MagicMock(return_value='test_inward_in_ld')
            inward_in_ld.new = True  # Set new message state
        # Set the return message for inward incoming linkdata
        for inward_in_ld, mock_msg in zip(inward_incoming_lds, mock_msgs):
            inward_in_ld.read = MagicMock(return_value=mock_msg)
        for outward_in_ld, in_vn in zip(outward_incoming_lds, outward_vn_list[:-1]):
            outward_in_ld.vn = in_vn
            outward_in_ld.fn = bfn
            outward_in_ld.to_fn = True
            outward_in_ld.attr = {'direction': 'outward'}
            outward_in_ld.__repr__ = MagicMock(return_value='test_outward_in_ld')
            outward_in_ld.new = False  # Set new message state
        inward_outgoing_ld.vn, outward_outgoing_ld.vn = inward_vn_list[-1], outward_vn_list[-1]
        inward_outgoing_ld.fn, outward_outgoing_ld.fn = bfn, bfn
        inward_outgoing_ld.to_fn, outward_outgoing_ld.to_fn = False, False
        inward_outgoing_ld.attr, outward_outgoing_ld.attr = {'direction': 'inward'}, {'direction': 'outward'}
        inward_outgoing_ld.__repr__, outward_outgoing_ld.__repr__ = MagicMock(return_value='test_inward_in_ld'), \
                                                                    MagicMock(return_value='test_outward_in_ld')

        for ld in inward_incoming_lds:
            bfn.add_link(ld)
        for ld in outward_incoming_lds:
            bfn.add_link(ld)
        bfn.add_link(inward_outgoing_ld)
        bfn.add_link(outward_outgoing_ld)

        # Set compute message map
        bfn.inward_msg_map, bfn.outward_msg_map = [0], [0]  # Take the first incoming linkdata's message

        bfn.compute()

        # Check that inward direction propagate correctly
        inward_incoming_lds[0].read.assert_called_once()
        inward_outgoing_ld.write.assert_called_once_with(mock_msgs[0])

        # Check that outward direction is not called at all
        for in_ld in outward_incoming_lds:
            in_ld.read.assert_not_called()
        outward_outgoing_ld.read.assert_not_called()

    def test_outward_compute(self):
        # Test that outward_compute() is called when there are new outward incoming messages
        # Check that the inward_compute() is not called when there is no new inward incoming message
        bfn = BetaFactorNodeForTest('test_bfn')
        num_inward_in, num_outward_in = 3, 2
        # Mock message
        mock_msgs = [MagicMock(spec=Message) for i in range(num_outward_in)]
        # Mock relational variables
        mock_relvar_1, mock_relvar_2 = MagicMock(spec=Variable), MagicMock(spec=Variable)
        mock_relvar_1.__repr__, mock_relvar_2.__repr__ = MagicMock(return_value='mock_relvar_1'), \
                                                         MagicMock(return_value='mock_relvar_2')
        # Mock variable nodes
        inward_vn_list, outward_vn_list = [MagicMock(spec=VariableNode) for i in range(num_inward_in + 1)], \
                                          [MagicMock(spec=VariableNode) for i in range(num_outward_in + 1)]
        for vn in inward_vn_list:
            vn.rel_vars = (mock_relvar_1, mock_relvar_2)
        for vn in outward_vn_list:
            vn.rel_vars = (mock_relvar_1, mock_relvar_2)
        # Mock linkdata
        inward_incoming_lds, outward_incoming_lds = [MagicMock(spec=LinkData) for i in range(num_inward_in)], \
                                                    [MagicMock(spec=LinkData) for i in range(num_outward_in)]
        inward_outgoing_ld, outward_outgoing_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        for inward_in_ld, in_vn in zip(inward_incoming_lds, inward_vn_list[:-1]):
            inward_in_ld.vn = in_vn
            inward_in_ld.fn = bfn
            inward_in_ld.to_fn = True
            inward_in_ld.attr = {'direction': 'inward'}
            inward_in_ld.__repr__ = MagicMock(return_value='test_inward_in_ld')
            inward_in_ld.new = False  # Set new message state
        # Set the return message for inward incoming linkdata
        for outward_in_ld, in_vn in zip(outward_incoming_lds, outward_vn_list[:-1]):
            outward_in_ld.vn = in_vn
            outward_in_ld.fn = bfn
            outward_in_ld.to_fn = True
            outward_in_ld.attr = {'direction': 'outward'}
            outward_in_ld.__repr__ = MagicMock(return_value='test_outward_in_ld')
            outward_in_ld.new = True  # Set new message state
        for outward_in_ld, mock_msg in zip(outward_incoming_lds, mock_msgs):
            outward_in_ld.read = MagicMock(return_value=mock_msg)
        inward_outgoing_ld.vn, outward_outgoing_ld.vn = inward_vn_list[-1], outward_vn_list[-1]
        inward_outgoing_ld.fn, outward_outgoing_ld.fn = bfn, bfn
        inward_outgoing_ld.to_fn, outward_outgoing_ld.to_fn = False, False
        inward_outgoing_ld.attr, outward_outgoing_ld.attr = {'direction': 'inward'}, {'direction': 'outward'}
        inward_outgoing_ld.__repr__, outward_outgoing_ld.__repr__ = MagicMock(return_value='test_inward_in_ld'), \
                                                                    MagicMock(return_value='test_outward_in_ld')

        for ld in inward_incoming_lds:
            bfn.add_link(ld)
        for ld in outward_incoming_lds:
            bfn.add_link(ld)
        bfn.add_link(inward_outgoing_ld)
        bfn.add_link(outward_outgoing_ld)

        # Set compute message map
        bfn.inward_msg_map, bfn.outward_msg_map = [0], [1]  # Take the first incoming linkdata's message

        bfn.compute()

        # Check that outward direction propagate correctly
        outward_incoming_lds[1].read.assert_called_once()
        outward_outgoing_ld.write.assert_called_once_with(mock_msgs[1])

        # Check that outward direction is not called at all
        for in_ld in inward_incoming_lds:
            in_ld.read.assert_not_called()
        inward_outgoing_ld.read.assert_not_called()

