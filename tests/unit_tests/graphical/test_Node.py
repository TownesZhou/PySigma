"""
    Unit tests for abstract base Node class
"""

import pytest
import torch
from unittest.mock import MagicMock, PropertyMock, patch
from pysigma.graphical.basic_nodes import Node
from ..utils import cuda_only


# Since Node class is an abstract base class, we need a make a concrete subclass of it and test the subclass
class NodeForTest(Node):
    
    def add_link(self, linkdata):
        super(NodeForTest, self).add_link(linkdata)

    @Node.compute_control
    def compute(self):
        pass


class TestNode():

    def test_init(self):
        name = "test_name"
        node = NodeForTest(name)

        assert node.name == name
        assert node.device == 'cpu'
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
