"""
    Unit tests for ESFN class
"""
import pytest
from unittest.mock import MagicMock
import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, VariableNode, NodeConfigurationError
from pysigma.graphical.predicate_nodes import WMFN_VI, WMVN
from pysigma.utils import KnowledgeServer as KS
from pysigma.utils import DistributionServer as DS
from ...utils import random_message

from ...utils import generate_positive_definite


class TestESFN:

    def test_init(self):
        pass
