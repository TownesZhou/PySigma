"""
    All nodes related to the Conditional Beta subgraph
"""
from __future__ import annotations      # For postponed evaluation of typing annotations
from typing import Optional, List, Tuple, Dict
from typing import Iterable as IterableType
import copy
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
import torch
from ..defs import Variable
from ..pattern_structures.variable_map import VariableMapCallable
from .basic_nodes import LinkData, FactorNode, NodeConfigurationError
from ..pattern_structures.summarization import SummarizationClass, SummarizationError


class PDFN(FactorNode):

    pass


class PRFN(FactorNode):

    pass
