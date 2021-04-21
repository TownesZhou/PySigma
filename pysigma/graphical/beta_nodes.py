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
    """
        Particle Dispatch Factor Node
    """
    def __init__(self, name):
        super(PDFN, self).__init__(name)




class PRFN(FactorNode):
    """
        Particle Reweight Factor Node
    """
    def __init__(self, name):
        super(PRFN, self).__init__(name)


