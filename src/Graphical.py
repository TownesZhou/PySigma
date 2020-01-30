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
    def __init__(self, vn, var_list):
        """

        :param vn:      name of the variable node that this link is incident to
        :param vars:    list of variables of the adjacent variable node
        """
        # TODO
        # Link message memory
        self.memory = None
        # Whether this message is a new one just sent by adjacent node and haven't been processed
        self.new = False
        # The name of the variable node to which this link is incident
        self.vn = vn
        # List of variables of the adjacent variable node
        self.vars = var_list

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
            `VariableNode`, for example a flag indicating whether this node has reach quiescence, i.e., whether no new
            message shall be sent via adjacent link in curretn cycle. During construction of the graph, its instance
            will be passed to `NetworkX` methods to instantiate a node.
    """
    def __init__(self):
        # TODO
        # Flag indicating whether quiescence reached in current cycle. If so, no sum-product local processing needed at
        #   this node.
        self._quiescence = False


class FactorNode(Node):
    """
        Specify a **factor node**, with pytorch tensor as optional factor node function (default to a constant function 
            of 1 everywhere effectively). It is the super class of factor nodes of various subtypes, such as alpha, 
            beta, delta, gamma nodes.
        Implement default message processing mechanism of a factor node:
            1. Product: broadcast the messages from all adjacent variable nodes except the output one to full dimension
                (i.e., high dimensional tensor with all adjacent nodes' variables), and time them with this factor
                node's function)
            2. Summary: Take the result from step 1., compute the marginals for the output variable node's variables by
                means of summary (taking sum/integral or max over other variables' dimensions)

        Note: All implementation considers dynamic structure, i.e., the possibility of adding new link and nodes on the
            fly. Therefore:
                - Variable dimension alignment for each incoming link message is computed dynamically
    """
    def __init__(self):
        super(FactorNode, self).__init__()

        # Factor Node function. Usually specified as a tensor. Default to None, effectively 1 everywhere when broadcast
        #   to full vairable dimension
        self._function = None
        # List of adjacent variable node names
        self._vn_list = []
        # Dictionary of adjacent variable nodes' variable names, indexed by variable nodes' names
        self._vn2var = {}
        # List of variable names from all adjacent variable nodes. Used for dimension order of this factor node's
        #   sum-product processing. In other words a flattened version of vn_var_dict
        self._var_list = []
        # List of LinkData of those link connecting to this factor node, from which we retrieve messages
        self._linkdata = []
        self._vn2linkdata = {}

    def add_vn(self, linkdata):
        """
            Register the name and variables of a newly added variable node by registering the linkdata that connect to
                that variable node. New variables, if not already exist in _var_list, will be appended.
        :param linkdata:    the linkdata of the link that connects to the newly added variable node
        :return:
        """
        self._linkdata.append(linkdata)
        self._vn2linkdata[linkdata.vn] = linkdata
        self._vn_list.append(linkdata.vn)
        self._vn2var[linkdata.vn] = linkdata.vars
        for v in linkdata.vars:
            if v not in self._var_list:
                self._var_list.append(v)

    def align(self, msg, var_list):
        """
            Compute and return the bradcasted message with its dimension properly aligned with the factor node variable
                order.

            Note: MAY BE REPLACED LATER WITH PYTORCH'S BUILT-IN DIMENSION REARRANGEMENT FOR NAMED TENSORS.
        :param msg:  unbroadcasted message from a linkdata
        :param vars: list of variables of the unbroadcasted message
        :return:    broadcasted message with dimension aligned
        """
        # First take a view that append extra dimensions to the message
        view_dim = msg.shape + ((1,) * (len(self._var_list) - len(msg.shape)))
        # Find indices of the variables in _var_list
        var_id = [self._var_list.index(var) for var in var_list]
        # Compute the permutation
        perm = [-1] * len(self._var_list)   # -1 is dummy value
        for i, id in enumerate(var_id):
            perm[id] = i
        rest = len(var_list)
        for i in range(len(self._var_list)):
            if i not in var_id:
                perm[i] = rest
                rest += 1
        # Permute message dimension and return
        aligned_msg = msg.view(view_dim).permute(perm)
        return aligned_msg

    def compute(self, out_vn):
        """
            Compute the message for the outgoing variable node whose name specified by out_vn

            Note: this method should be overriden by special purpose factor node subclasses that comes with unique
                message processing mechanism.
        :param out_vn:  outgoing variable node name
        :return:
        """



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
