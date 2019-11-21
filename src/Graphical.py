"""
    Structures and Processes that drive the Sigma graphical architecture, which is the bottom part of the cognitive
    hourglass.

    In this Python implementation of Sigma, the representation and processing of messages are based on PyTorch tensors,
        and the representation and processing of the factor graph are based on NetworkX.

    Author of this file: Jincheng Zhou, University of Southern California
    Reference: Paul S. Rosenbloom, Abram Demski & Volkan Ustun, 2016, The Sigma Cognitive Architecture and System:
        Toward Functionally Elegant Grand Unification
"""

import torch
import networkx


class Message(torch.Tensor):
    """
        A subclass of pytorch tensor. Stores the actual content as a tensor and can be directly manipulated as a regular
            tensor, but stores extra bookkeeping information for message processing mechanism in Sigma.
    """
    def __init__(self):
        super(Message, self).__init__()
        # TODO

class Node:
    """
        The super class of `FactorNode` and `VariableNode`. It declares common attributes between `FactorNode` and
            `VariableNode`, for example a list of working memory variables. During construction of the graph, its
            instance will be passed to `NetworkX` methods to instantiate a node.
    """
    def __init__(self):
        # TODO
        pass


class LinkData:
    """
        Identify the *data* of a ***directed*** link between a factor node and a variable node. Stores intermediate
            messages in the **message memory**. Two of such links should be specified with opposite directions to
            represent a bidirectional link between a factor node and a variable node, typically in the case of condacts.
        During construction of the graph, its instance will be passed to `NetworkX` methods as the edge data to
            instantiate an edge.
    """
    def __init__(self):
        # TODO
        self.memory = None


class FactorNode(Node):
    """
        Specify a **factor node**, with pytorch tensor as optional factor node function (default to a constant function 
            of 1 everywhere effectively). It is the super class of factor nodes of various subtypes, such as alpha, 
            beta, delta, gamma nodes. 
    """
    def __init__(self):
        super(FactorNode, self).__init__()
        # TODO


class VariableNode(Node):
    """
        Specify a **variable node**.
    """
    def __init__(self, name, variables):
        """
            Decalre a VariableNode
        :param name:
        :param variables:
        """
        super(VariableNode, self).__init__()
        # TODO


class Graph(networkx.DiGraph):
    """
        The actual graph instance representing a compiled graphical architecture. Can be manipulated directly as a
            `networkx.DiGraph` but store extra bookkeeping information. use `networkx.DiGraph` as super class to support
            directed links.
    """
    def __init__(self):
        super(Graph, self).__init__()
        # TODO

    def add_unilink(self, node1, node2):
        """
            Add a unidirectional link from node1 to node2. Note that one of the nodes should be a variable node and the
                other a factor node
        """
        # TODO: add unidirectional link
        pass

    def add_bilink(self, node1, node2):
        """
            Add a bidrectional link between node 1 and node2. Note that one of the nodes should be a variable node and
                the other a factor node
        """
        # TODO: add bidirectional link
        pass

    def solve(self):
        """
            One phase of graph solution of message passing until quiescence is reached.
        """
        # TODO
        pass

    def modify(self):
        """
            One phase of graph modification on existing graph memories.
        """
        # TODO
        pass
