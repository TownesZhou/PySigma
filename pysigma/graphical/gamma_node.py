"""
    The gamma node, the peak of a conditional subgraph.
"""
from .basic_nodes import FactorNode, VariableNode


class GTVN(VariableNode):
    """Gamma Terminal Variable Node

    """
    pass


class GFN(FactorNode):
    """
        Gamma Factor Node

        Carry out general-inference message computation at the PGM factor node.

        Induce a message computation task for each of the outgoing link.
    """
    pass
