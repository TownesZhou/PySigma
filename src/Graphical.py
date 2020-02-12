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

    def __init__(self, *args, **kwargs):
        super(Message, self).__init__(*args, **kwargs)
        # TODO


class Variable:
    """
        Variable as in variable nodes in the graphical architecture. Store information about this variable such as
            whether it is unique or universal.
        Note: The equality of variables' identity is determined by the equality of their name, regardless of the values
            of other fields
    """

    def __init__(self, name, size, unique=True, selection=False):
        """
        :param name:        Variable name
        :param size:        The size, or maximum number of regions, of this variable dimension
        :param unique:      True/False indicating whether 'unique' or 'universal'
        :param selection:   Selection method
        """
        self.name = name
        self.size = size
        self.unique = unique
        self.selection = selection

    def __eq__(self, other):
        # override so '==' operator test the 'name' field
        return self.name == other.name

    def __ne__(self, other):
        # override so '!=' operator test the 'name' field
        return self.name != other.name

    def __str__(self):
        # override to provide the name as the string representation
        return self.name


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
        # Link message memory
        self.memory = None
        # Whether this message is a new one just sent by adjacent node and haven't been read by the other node
        self.new = False

        # Following fields should correspond to the ones in the incident variable node
        self.vn = vn
        self.var_list = var_list

        # Whether this link is pointing toward a factor node
        self.to_fn = to_fn

        # Record the dimensions of link message. Use to check potential dimension mismatch
        self._dims = [var.size for var in self.var_list]

    def set(self, new, epsilon):
        """
            Set the link message memory to the new message arriving at this link. Implement the optimization so that
                memory content is not changed if new message differs from existing content by less than epsilon
            Default to check the absolute maximum difference between the old and new messages
        :param new:         new arriving message
        :param epsilon:     epsilon criterion
        """
        # Check dimension mismatch
        size = list(new.shape)
        if size != self._dims:
            raise ValueError("The new message's dimension '{}' does not match the link memory's preset dimension '{}'. "
                             "Target variable node: '{}', link direction: toward factor node = '{}'"
                             .format(size, self._dims, str(self.vn), self.to_fn))

        diff = torch.max(torch.abs(self.memory - new))
        if diff < epsilon:
            return

        self.memory = new
        self.new = True

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

    def __init__(self, name, epsilon=10e-5):
        # Flag indicating whether quiescence reached in current cycle. If so, no sum-product local processing needed at
        #   this node.
        self.name = name
        self._epsilon = epsilon
        self.quiescence = False

        # Global logging info
        self.log = {}
        self.pretty_log = {}        # Log info to display at the GUI

    def __str__(self):
        # override to provide the node's name as its string representation
        return self.name


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

    def __init__(self, name, function=None, func_var_list=None, epsilon=None):
        """
        :param name:            Name of this factor node
        :param function:        An int, float, or torch.tensor. or Message, with dimension ordered by func_var_list. If
                                None, set to default value (1). If torch.tensor, change type to Message
        :param func_var_list:   Variables of the function. Default to None
        :param epsilon:         The epsilon no-change criterion used for comparing quiesced message
        """
        super(FactorNode, self).__init__(name)

        # Set function
        self._function = None
        self._func_var_list = None
        self.set_function(function, func_var_list)

        # set custom epsilon
        if epsilon is not None:
            self._epsilon = epsilon

        # List of Variables from all adjacent variable nodes. Used for dimension order of this factor node's
        #   sum-product processing. In other words a flattened version of vn_var_dict
        self._var_list = []
        # List of LinkData of those link connecting to this factor node, incoming and outgoing ones respectively, from
        #   which we retrieve messages
        self._in_linkdata = []
        self._out_linkdata = []

        # pretty log
        self.pretty_log["all variables"] = []       # init

    def set_function(self, function, var_list):
        """
            (Re)set the factor node function.
        :param function:    int, float, or a torch.tensor, or Message, with dimension ordered by var_list. If None, set
                            to default value (1). If torch.tensor, change type to Message
        :param var_list:    list of variable names corresponding to dimensions of function
        """
        var_size = [var.size for var in var_list] if var_list is not None else None
        if function is None:
            self._function = 1
        elif type(function) is torch.Tensor:
            # first check if provided tensor's size is correct
            assert list(function.shape) == var_size, "The dimensions of the function variables (in the specified order)" \
                                                     " is {}, However, the dimensions of the provided function tensor " \
                                                     "is {}. Function dimensions must agree with its variables' " \
                                                     "dimensions".format(var_size, list(function.shape))
            self._function = Message(function)
        else:
            self._function = function
        self._func_var_list = var_list

        # update pretty log
        self.pretty_log["function type"] = "tensor" if isinstance(function, torch.Tensor) else "constant"
        self.pretty_log["function variables"] = [var.name for var in self._func_var_list] \
            if self._func_var_list is not None else None
        self.pretty_log["function size"] = var_size

    def get_function(self):
        return self._function, self._func_var_list

    def add_link(self, linkdata):
        """
            Register the name and variables of a newly added variable node by registering the linkdata that connect to
                that variable node. New variables, if not already exist in _var_list, will be appended.
            If a variable from the linkdata already exists in _var_list, check that their size agrees, otherwise raise
                an error.

            Note: if the link between the variable node and this factor node is bidirectional, then this method should
                be called twice, once with linkdata of each direction.
        :param linkdata:    the linkdata of the link that connects to the newly added variable node
        """
        if linkdata in self._in_linkdata or linkdata in self._out_linkdata:
            return

        # Check variable size agrees
        for var in linkdata.var_list:
            for old_var in self._var_list:
                if var == old_var:
                    assert var.size == old_var.size, "The variable '{}' from the variable node '{}' that attempts to " \
                                                     "link to this factor node has size '{}', which does not match the " \
                                                     "size '{}' of the variable with the same name already present in " \
                                                     "the variable lists of this factor node."\
                        .format(str(var), str(linkdata.vn), var.size, old_var.size)

        if linkdata.to_fn:
            self._in_linkdata.append(linkdata)
        else:
            self._out_linkdata.append(linkdata)

        for v in linkdata.var_list:
            if v not in self._var_list:
                self._var_list.append(v)

                # update pretty log
                self.pretty_log["all variables"].append(v.name)

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
        perm = [-1] * len(self._var_list)  # -1 is dummy value
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
                messages. May be override by special purpose factor node
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
            Implement the optimization so that no new message is computed and sent if messages from all incoming link
                are not new.

            Note: this method should be overriden by special purpose factor node subclass that comes with unique message
                processing mechanism, such as Affine-Delta factor node
        """
        # First loop through incoming linkdata once to check whether messages from each incoming link is not new to
        # determine whether this node has reached quiescence
        # TODO: Normalization step
        quiesced = True
        for in_ld in self._in_linkdata:
            if in_ld.new:
                quiesced = False
                break
        if quiesced:
            self.quiescence = True
            return

        # If this node has not reached quiescence, compute and send new messages
        for out_ld in self._out_linkdata:
            out_vn = out_ld.vn
            # If function is an int or float, keep as a scalar. Otherwise expand to full dim
            buf = self._function if type(self._function) in [int, float] \
                else self.align(self._function, self._func_var_list)

            # Product
            for in_ld in self._in_linkdata:
                in_vn = in_ld.vn
                if in_vn is out_vn:  # Here use 'is' operator to test variable node's identity
                    continue
                msg = self.align(in_ld.read(), in_ld.var_list)
                buf = self.sp_product(buf, msg)

            # Summary
            # Perform sum reduce if variable is unique, otherwise perform max reduce
            sum_reduce = [i for i, var in enumerate(self._var_list)
                          if var.unique and var not in out_ld.var_list]
            max_reduce = [i for i, var in enumerate(self._var_list)
                          if not var.unique and var not in out_ld.var_list]
            buf = self.sp_sum(buf, sum_reduce)
            buf = self.sp_max(buf, max_reduce)

            # Send message
            out_ld.set(buf, self._epsilon)


class PBFN(FactorNode):
    """
        Perception Buffer Factor Node
    """

    # TODO
    def __init__(self, name, function=None, func_var_list=None):
        super(PBFN, self).__init__(name, function, func_var_list)
        self.pretty_log["node type"] = "Perceptual Buffer Function Node"


class LTMFN(FactorNode):
    """
        Long-Term Memory Factor Node
    """

    # TODO
    def __init__(self, name, function=None, func_var_list=None):
        super(LTMFN, self).__init__(name, function, func_var_list)
        self.pretty_log["node type"] = "Long-Term Memory Function Node"


class WMFN(FactorNode):
    """
        Working Memory Factor Node
    """

    # TODO
    def __init__(self, name, function=None, func_var_list=None):
        super(WMFN, self).__init__(name, function, func_var_list)
        self.pretty_log["node type"] = "Working Memory Function Node"


class ACFN(FactorNode):
    """
        Action-Combination Factor Node
    """

    # TODO: Special implementation of sum-product to implement message conjunction
    def __init__(self, name,):
        super(ACFN, self).__init__(name, function=None, func_var_list=None)
        self.pretty_log["node type"] = "Action Combination Function Node"


class FFN(FactorNode):
    """
        Filter Node
    """

    # TODO
    def __init__(self, name, function=None, func_var_list=None):
        super(FFN, self).__init__(name, function, func_var_list)
        self.pretty_log["node type"] = "Filter Factor Node"


class NLFN(FactorNode):
    """
        Nonlinearity Factor Node. Element-wise nonlinearity "filters"
    """

    # TODO:
    def __init__(self, name, nonlinear):
        """
        :param name:        Name of the node
        :param nonlinear:  'str' or function objects, specifying the type of element-wise nonlinearity
        """
        super(NLFN, self).__init__(name, function=None, func_var_list=None)
        self.nonlinear = nonlinear
        self.pretty_log["node type"] = "Nonlinearity Factor Node"
        self.pretty_log["nonlinearity"] = str(nonlinear)


class ADFN(FactorNode):
    """
        Affine Delta Factor Node, generally for matching between WM variables and Pattern variables
        Performs delta function masking and variable swapping
        Essential component of the Rete's Alpha network's within pattern variable matching

        Note: ADFN only admits one incoming link and one outgoing link.

        Note: Special implementation needed for resolving potential conflicts between pattern variables and wm variables,
                especially mismatch in sizes.
    """

    # TODO
    def __init__(self, name, pt_var_info):
        super(ADFN, self).__init__(name, function=None, func_var_list=None)
        self.pretty_log["node type"] = "Affine Delta Factor Node"
        self._pt_var_info = pt_var_info


class ATFN(FactorNode):
    """
        Affine Transformation Factor Node
    """

    # TODO
    def __init__(self, name, affine):
        """
        :param name:    Name of the node
        :param affine:  Affine transformation specification
        """
        super(ATFN, self).__init__(name, function=None, func_var_list=None)
        self.affine = affine
        self.pretty_log["node type"] = "Affine Transformation Factor Node"


class BJFN(FactorNode):
    """
        Beta-Join Factor Node
    """

    # TODO
    def __init__(self, name):
        super(BJFN, self).__init__(name, function=None, func_var_list=None)
        self.pretty_log["node type"] = "Beta-Join Factor Node"


class GFFN(FactorNode):
    """
        Gamma Function Factor Node
    """

    # TODO
    def __init__(self, name, function, func_var_list):
        super(GFFN, self).__init__(name, function=function, func_var_list=func_var_list)
        self.pretty_log["node type"] = "Gamma Function Factor Node"


class VariableNode(Node):
    """
        Specify a **variable node**.
    """

    def __init__(self, name, var_list, epsilon=None):
        """
            Decalre a VariableNode
        :param name:        name of the variable node
        :param variables:   list of Variables representing the variables of this variable nodes
        :param epsilon:     The epsilon no-change criterion used for comparing quiesced message
        """
        super(VariableNode, self).__init__(name)
        self.var_list = var_list

        # List of LinkData of those links connecting to this variable nodes, incoming and outgoing ones respectively.
        self._in_linkdata = []
        self._out_linkdata = []

        # set custom epsilon
        if epsilon is not None:
            self._epsilon = epsilon

        # Pretty log
        self.pretty_log["variable names"] = [var.name for var in self.var_list]
        self.pretty_log["variable dimensions"] = [var.size for var in self.var_list]
        self.pretty_log["variable uniqueness"] = [var.unique for var in self.var_list]
        self.pretty_log["variable selection"] = [var.selection for var in self.var_list]

    def add_link(self, linkdata):
        """
            Register the LinkData connecting a factor node to this variable node.
            Note: the 'vn' and 'var_list' fields should conform to the ones as specified in this variable node.
        """
        if linkdata in self._in_linkdata or linkdata in self._out_linkdata:
            return

        if linkdata.to_fn:
            self._out_linkdata.append(linkdata)
        else:
            self._in_linkdata.append(linkdata)

    def sp_product(self, msg1, msg2):
        """
            The default implementation of the product part of the sum-product algorithm. Return the product of two
                messages. May be override by special purpose factor node.
        """
        result = msg1 * msg2
        return result

    def compute(self):
        """
            Take the product of the messages from incoming nodes and send the result toward all outgoing nodes.
            Implement the optimization so that no new meesage is computed and sent if message from all incoming link are
                not new.
        """
        # First loop through incoming linkdata once to check whether messages from each incoming link is not new to
        #   determine whether this node has reached quiescence
        # TODO: Normalization step
        quiesced = True
        for in_ld in self._in_linkdata:
            if in_ld.new:
                quiesced = False
                break
        if quiesced:
            self.quiescence = True
            return

        # If not reached quiescence, compute and send new messages
        for out_ld in self._out_linkdata:
            out_vn = out_ld.vn
            buf = 1

            # Taking products
            for in_ld in self._in_linkdata:
                in_vn = in_ld.vn
                if in_vn is out_vn:     # Here use 'is' operator to test variable node's identity
                    continue
                msg = in_ld.read()
                buf = self.sp_product(buf, msg)

            # Send message
            out_ld.set(buf, self._epsilon)


class WMVN(VariableNode):
    """
        Working Memory Variable Node
    """

    # TODO
    def __init__(self, name, var_list):
        super(WMVN, self).__init__(name, var_list)
        self.pretty_log["node type"] = "Working Memory Variable Node"


class Graph(networkx.DiGraph):
    """
        The actual graph instance representing a compiled graphical architecture. Can be manipulated directly as a
            `networkx.DiGraph` but store extra bookkeeping information. use `networkx.DiGraph` as super class to support
            directed links.
    """

    def __init__(self):
        super(Graph, self).__init__()

        # node type name2type dictionary
        self._name2type = \
            {PBFN.__name__: PBFN,
             LTMFN.__name__: LTMFN,
             WMFN.__name__: WMFN,
             ACFN.__name__: ACFN,
             FFN.__name__: FFN,
             NLFN.__name__: NLFN,
             ADFN.__name__: ADFN,
             ATFN.__name__: ATFN,
             BJFN.__name__: BJFN,
             GFFN.__name__: GFFN,
             WMVN.__name__:WMVN,
             }

    def new_node(self, node_type, *args, **kwargs):
        """
            A node factory method. Generate new node based on node_type, add it to DiGraph, and return it.
        :param node_type:   str or class. Specifying the node type, such as 'WMVN' or WMVN
        :return:            the newly generated node instance
        """
        assert node_type in self._name2type.keys() or node_type in self._name2type.values(), \
            "first argument {} not a known node type".format(node_type)

        if node_type in self._name2type.keys():
            node_type = self._name2type[node_type]
        node = node_type(*args, **kwargs)

        # Add node to DiGraph
        self.add_node(node)

        return node

    def add_unilink(self, node1, node2):
        """
            Add a unidirectional link FROM node1 TO node2, and create and register corresponding LinkData
            Note that one of the nodes should be a variable node and the other a factor node
        """
        assert (isinstance(node1, VariableNode) and isinstance(node2, FactorNode)) or \
               (isinstance(node1, FactorNode) and isinstance(node2, VariableNode)), \
            "One of the nodes must be a variable node and the other one a factor node"

        # Create a LinkData and set its attributes
        vn = node1 if isinstance(node1, VariableNode) else node2
        var_list = vn.var_list
        to_fn = True if isinstance(node2, FactorNode) else False
        linkdata = LinkData(vn, var_list, to_fn)

        # Create edge in graph. The LinkData exists in the 'data' field in an edge of the NetworkX graph
        self.add_edge(node1, node2, data=linkdata)

        # register LinkData in both nodes
        node1.add_link(linkdata)
        node2.add_link(linkdata)

    def add_bilink(self, node1, node2):
        """
            Add a bidrectional link between node 1 and node2. Note that one of the nodes should be a variable node and
                the other a factor node
        """
        assert (isinstance(node1, VariableNode) and isinstance(node2, FactorNode)) or \
               (isinstance(node1, FactorNode) and isinstance(node2, VariableNode)), \
            "One of the nodes must be a variable node and the other one a factor node"

        self.add_unilink(node1, node2)
        self.add_unilink(node2, node1)

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
