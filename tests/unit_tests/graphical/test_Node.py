"""
    Unit tests for abstract base Node class
"""
import pytest
import torch
from unittest.mock import MagicMock, PropertyMock, patch
from pysigma.graphical.basic_nodes import Node, NodeConfigurationError
from tests.utils import cuda_only


# Since Node class is an abstract base class, we need a make a concrete subclass of it and test the subclass
class NodeForTest(Node):
    
    def __init__(self, *args, **kwargs):
        super(NodeForTest, self).__init__(*args, **kwargs)

        # Ad-hoc switch to indicate if node is correctly configured for testing the precompute_check() method
        self.computable = True
    
    def add_link(self, linkdata):
        super(NodeForTest, self).add_link(linkdata)

    def precompute_check(self):
        if not self.computable:
            raise NodeConfigurationError("Test NodeConfigurationError")

    @Node.compute_control
    def compute(self):
        pass


class TestNode:

    def test_init(self):
        name = "test_name"
        node = NodeForTest(name)

        assert node.name == name
        assert node.device == torch.device('cpu')
        assert not node.visited
        assert isinstance(node.in_linkdata, list) and len(node.in_linkdata) == 0
        assert isinstance(node.out_linkdata, list) and len(node.out_linkdata) == 0

    @cuda_only
    def test_init_cuda(self):
        # Run this test method only if the host system has cuda enabled
        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        name = "test_name"
        node = NodeForTest(name, device)

        assert node.device == device

    def test_str(self):
        name = "test_node"
        node = NodeForTest(name)

        assert str(node) == name

    def test_quiescence(self):
        node = NodeForTest("test_node")

        # Test 1: all incoming LinkData do not contain new message
        node.in_linkdata = [MagicMock(new=False)] * 3
        assert node.quiescence

        # Test 2: some incoming LinkData contains new message
        node.in_linkdata = [MagicMock(new=False)] * 3 + [MagicMock(new=True)]
        assert not node.quiescence

    def test_configuration_error(self):
        # Test that correct exception is thrown if node is ill-configured
        with patch("pysigma.graphical.basic_nodes.Node.quiescence", new_callable=PropertyMock) as mock_quiescence:
            mock_quiescence.return_value = True        # Exception should be raised even if node is quiesced
            node = NodeForTest("test_node")
            node.computable = False     # Set the flag. This should trigger NodeForTest instance to fail the check

            with pytest.raises(NodeConfigurationError) as excinfo:
                node.compute()

            # Check exception value
            assert str(excinfo.value) == "Test NodeConfigurationError"
            # Check that node state is not changed
            assert not node.visited

    def test_compute(self):
        # Test 1: quiescence
        # Mock the node's quiescence property
        with patch("pysigma.graphical.basic_nodes.Node.quiescence", new_callable=PropertyMock) as mock_quiescence:
            mock_quiescence.return_value = True
            node = NodeForTest("test_node")
            node.compute()
            assert not node.visited

        # Test 1: not quiescence
        # Mock the node's quiescence property
        with patch("pysigma.graphical.basic_nodes.Node.quiescence", new_callable=PropertyMock) as mock_quiescence:
            mock_quiescence.return_value = False
            node = NodeForTest("test_node")
            node.compute()
            assert node.visited

    def test_reset_state(self):
        node = NodeForTest("test_node")
        node.visited = True
        node.reset_state()
        assert not node.visited
