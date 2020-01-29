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
        A subclass of pytorch tensor. Stores message plm as tensors and can be directly manipulated, but keep extra
            bookkeeping information for message processing mechanism in Sigma.
    """
    def __init__(self):
        super(Message, self).__init__()
        # TODO


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

    def set(self, new, epsilon):
        """
            Set the link message memory to the new message arriving at this link. Implement the optimization so that
                memory content is not changed if new message differs from existing content by less than epsilon
        :param new:
        :param epsilon:
        :return:
        """
        # TODO:


class Node:
    """
        The super class of `FactorNode` and `VariableNode`. It declares common attributes between `FactorNode` and
            `VariableNode`, for example a list of working memory variables. During construction of the graph, its
            instance will be passed to `NetworkX` methods to instantiate a node.
    """
    def __init__(self):
        # TODO
        self._quiescence = False        # Flag indicating whether quiescence reached in current cycle. If so, no 
                                        #   sum-product local processing needed at this node. 


class FactorNode(Node):
    """
        Specify a **factor node**, with pytorch tensor as optional factor node function (default to a constant function 
            of 1 everywhere effectively). It is the super class of factor nodes of various subtypes, such as alpha, 
            beta, delta, gamma nodes. 
    """
    def __init__(self):
        super(FactorNode, self).__init__()
        # TODO


class PBFN(FactorNode):
    """
        Perception Buffer Factor Node
    """

    # TODO
    def __init__(self):
        super(PBFN, self).__init__()


class LTMFN(FactorNode):
    """
        Long-Term Memory Factor Node
    """

    # TODO
    def __init__(self):
        super(LTMFN, self).__init__()


class WMFN(FactorNode):
    """
        Working Memory Factor Node
    """

    # TODO
    def __init__(self):
        super(WMFN, self).__init__()


class ACFN(FactorNode):
    """
        Action-Combination Factor Node
    """

    # TODO
    def __init__(self):
        super(ACFN, self).__init__()
        
        
class FFN(FactorNode):
    """
        Filter Node
    """
    
    # TODO
    def __init__(self):
        super(FFN, self).__init__()


class ADFN(FactorNode):
    """
        Affine Delta Factor Node
    """
    
    # TODO
    def __init__(self):
        super(ADFN, self).__init__()


class VariableNode(Node):
    """
        Specify a **variable node**.
    """
    def __init__(self, name, variables):
        """
            Decalre a VariableNode
        :param name: name of the variable node
        :param variables: list of symbols representing the variables of this variable nodes
        """
        super(VariableNode, self).__init__()
        self.name = name
        self.variables = variables


class WMVN(VariableNode):
    """
        Working Memory Variable Node
    """

    # TODO
    def __init__(self):
        super(WMVN, self).__init__()


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
