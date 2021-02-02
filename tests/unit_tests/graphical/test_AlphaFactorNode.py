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
from pysigma.graphical.basic_nodes import LinkData, VariableNode
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


