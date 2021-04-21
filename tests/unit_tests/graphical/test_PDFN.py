"""
    Unit tests for PDFN class
"""
import pytest
from unittest.mock import MagicMock

import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, DVN
from pysigma.graphical.beta_nodes import PDFN
from pysigma.pattern_structures.variable_map import VariableMapCallable
from pysigma.pattern_structures.builtin_sum_op import sum_op_combination
from ...utils import random_message, assert_equal_within_error, assert_proportional_within_error




