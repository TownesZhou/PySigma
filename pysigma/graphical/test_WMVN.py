"""
    Unit tests for WMVN class
"""
import pytest
from unittest.mock import MagicMock, patch
import torch
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.predicate_nodes import WMVN


class TestWMVN():

    def test_init(self):
        pass
