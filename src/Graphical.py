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


class Variable:
    """
        Variable as in variable nodes in the graphical architecture. Store information about this variable such as
            whether it is unique or universal.
        The equality of variables' identity is determined by the equality of their name
    """
    def __init__(self, name, type):
        """
        :param name:    variable name
        :param type:    'unique' or 'universal'
        """
        self.name = name
        assert type in ['unique', 'universal'], "Variable type must be either unique or universal"
        self.type = type

    def __eq__(self, other):
        # override so '==' operator test the 'name' field
        return self.name == other.name

    def __ne__(self, other):
        # override so '!=' operator test the 'name' field
        return self.name != other.name


class LinkData:
    """
        Identify the *data* of a ***directed*** link between a factor node and a variable node. Stores intermediate
            messages in the **message memory**.
        Note that links are directional, and two of such links should be specified with opposite directions to
            represent a bidirectional link between a factor node and a variable node, typically in the case of condacts.
        During construction of the graph, its instance will be passed to `NetworkX` methods as the edge data to
            instantiate an edge.
    """
    def __init__(self, vn, var_list, to_fn):
        """
        :param vn:      name of the variable node that this link is incident to
        :param var_list:    list of variables of the adjacent variable node
        :param to_fn:   True/False indicating whether this link is pointing toward a factor node
        """
        # TODO
        # Link message memory
        self.memory = None
        # Whether this message is a new one just sent by adjacent node and haven't been read by the other node
        self.new = False

        self.vn = vn
        self.var_list = var_list
        self.to_fn = to_fn

    def set(self, new, epsilon):
        """
            Set the link message memory to the new message arriving at this link. Implement the optimization so that
                memory content is not changed if new message differs from existing content by less than epsilon
        :param new:
        :param epsilon:
        :return:
        """
        # TODO:

    def read(self):
        """
            Return the current content stored in memory. Set new to False to indicate this link message have been read
                since current cycle
        :return:    message content
        """
        self.new = False
        return self.memory


class Node:
    """
        The super class of `FactorNode` and `VariableNode`. It declares common attributes between `FactorNode` and
            `VariableNode`, for example a flag indicating whether this node has reach quiescence, i.e., whether no new
            message shall be sent via adjacent link in curretn cycle. During construction of the graph, its instance
            will be passed to `NetworkX` methods to instantiate a node.
    """
    def __init__(self, epsilon=10e-7):
        # TODO
        # Flag indicating whether quiescence reached in current cycle. If so, no sum-product local processing needed at
        #   this node.
        self._quiescence = False
        self._epsilon = epsilon

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
    def __init__(self, epsilon=None):
        super(FactorNode, self).__init__()

        # Factor Node function. Usually specified as a tensor. Default to None, effectively 1 everywhere when broadcast
        #   to full vairable dimension
        self._function = None
        # List of variables corresponding to the factor function
        self._func_var_list = None

        # List of adjacent variable node names, the ones with incoming links and the ones with outgoing links
        # self._in_vn_list = []
        # self._out_vn_list = []
        # Dictionary of adjacent variable nodes' variable names, indexed by variable nodes' names
        # self._vn2var = {}

        # List of variable names from all adjacent variable nodes. Used for dimension order of this factor node's
        #   sum-product processing. In other words a flattened version of vn_var_dict
        self._var_list = []
        # List of LinkData of those link connecting to this factor node, from which we retrieve messages
        # self._linkdata = []
        self._in_linkdata = []
        self._out_linkdata = []

        # Minimum difference between messages to initiate message passing
        if epsilon is not None:
            self._epsilon = epsilon

    def set_function(self, function, var_list):
        """
            (Re)set the factor node function.
        :param func:        A pytorch tensor, or Message, with dimension ordered by var_list
        :param var_list:    list of variable names corresponding to dimensions of function
        """
        self._function = function
        self._func_var_list = var_list

    def add_vn(self, linkdata):
        """
            Register the name and variables of a newly added variable node by registering the linkdata that connect to
                that variable node. New variables, if not already exist in _var_list, will be appended.
            Note: if the link between the variable node and this factor node is bidirectional, then this method should
                be called twice, once with linkdata of each direction.
        :param linkdata:    the linkdata of the link that connects to the newly added variable node
        """
        if linkdata in self._in_linkdata or linkdata in self._out_linkdata:
            return

        if linkdata.to_fn:
            self._in_linkdata.append(linkdata)
        else:
            self._out_linkdata.append(linkdata)

        for v in linkdata.var_list:
            if v not in self._var_list:
                self._var_list.append(v)

    def align(self, msg, var_list):
        """
            Compute and return the bradcasted message with its dimension properly aligned with the factor node variable
                order.

            Note: MAY BE REPLACED LATER WITH PYTORCH'S BUILT-IN DIMENSION REARRANGEMENT FOR NAMED TENSORS.
        :param msg:  unbroadcasted message from a linkdata
        :param var_list: list of variables of the unbroadcasted message
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

    def sp_product(self, msg1, msg2):
        """
            The default implementation of the product part of the sum-product algorithm. Return the product of two
                message. May be override by special
                purpose factor node
            Note: Assume msg1 and msg2 are properly aligned and already broadcastable
        """
        result = msg1 * msg2
        return result

    def sp_sum(self, msg, dims):
        """
            The default implementation of the summary part of the sum-product algorithm by taking sum/integral over
                specified dimensions. Returns the summed message
        :param msg:     tensor or Message whose selected variable dimensions to be summed
        :param dims:    a list of index specifying the dimension to be summed over
        """
        result = torch.sum(msg, dims)
        return result

    def sp_max(self, msg, dims):
        """
            The default implementation of the summary part of the sum-product algorithm by taking maximum over
                specified dimensions. Returns the maxed over message
        :param msg:     tensor or Message whose selected variable dimensions to be maxed
        :param dims:    a list of index specifying the dimension to be maxed over
        """
        result = torch.max(msg, dims)
        return result

    def compute(self):
        """
            Compute messages from incoming nodes and send the message toward all outgoing variable nodes.
            Default behavior is to time together aligned message from all incomming variable nodes except perhaps the
                target one with the factor function

            Note: this method should be overriden by special purpose factor node subclass that comes with unique message
                processing mechanism, such as Affine-Delta factor node
        :param out_vn:  outgoing variable node name
        :return:    result of sum-product of
        """

        for out_ld in self._out_linkdata:
            out_vn = out_ld.vn
            buf = 1 if self._function is None else self.align(self._function, self._func_var_list)

            # Product
            for in_ld in self._in_linkdata:
                in_vn = in_ld.vn
                if in_vn is out_vn:     # Here use 'is' operator to test vn's identity
                    continue
                msg = self.align(in_ld.read(), in_ld.var_list)
                buf = self.sp_product(buf, msg)

            # Summary
            unique_reduce = [i for i, var in enumerate(self._var_list)
                                if var.type == 'unique' or var in out_ld.var_list]
            universal_reduce = [i for i, var in enumerate(self._var_list)
                                if var.type == 'universal' or var in out_ld.var_list]
            buf = self.sp_sum(buf, unique_reduce)
            buf = self.sp_max(buf, universal_reduce)

            # Send message
            out_ld.set(buf, self._epsilon)


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
