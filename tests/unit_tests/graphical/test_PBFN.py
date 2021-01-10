"""
    Unit tests for PBFN class
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, FactorNode, VariableNode
from pysigma.graphical.predicate_nodes import WMFN, WMVN
from pysigma.utils import KnowledgeServer as KS
from pysigma.utils import DistributionServer as DS
from ..test_Message import TestMessage

from ...utils import generate_positive_definite


class TestPBFN:

    def test_init(self):
        pass
