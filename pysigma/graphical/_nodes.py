"""
    All kinds of nodes in the graphical architecture
"""
import torch
from abc import ABC, abstractmethod

# from ._structures import Message


class Node(ABC):
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
        self.pretty_log = {}  # Log info to display at the GUI

    def __str__(self):
        # override to provide the node's name as its string representation
        return self.name

    @abstractmethod
    def compute(self):
        pass


class FactorNode(Node, ABC):
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
        self.pretty_log["all variables"] = []  # init

    def set_function(self, function, var_list):
        """
            (Re)set the factor node function.
        :param function:    int, float, or a torch.tensor, or Message, with dimension ordered by var_list. If None, set
                            to default value (1). If torch.tensor, change type to Message
        :param var_list:    list of variable names corresponding to dimensions of function
        """
        var_size = [var.size for var in var_list] if var_list is not None else None

        self._function = 1 if function is None else function
        if isinstance(function, torch.Tensor):
            assert list(function.shape) == var_size, "The dimensions of the function variables (in the specified order)" \
                                                     " is {}, However, the dimensions of the provided function tensor " \
                                                     "is {}. Function dimensions must agree with its variables' " \
                                                     "dimensions".format(var_size, list(function.shape))
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
        assert linkdata not in self._in_linkdata and linkdata not in self._out_linkdata, "Adding duplicate link"
        if linkdata.to_fn:
            assert linkdata.vn not in [ld.vn for ld in self._in_linkdata], \
                "Adding duplicate link away from the variable node '{}'".format(linkdata.vn)
        else:
            assert linkdata.vn not in [ld.vn for ld in self._out_linkdata], \
                "Adding duplicate link to the variable node '{}'".format(linkdata.vn)


        # Check variable size agrees
        for var in linkdata.var_list:
            for old_var in self._var_list:
                if var == old_var:
                    assert var.size == old_var.size, "The variable '{}' from the variable node '{}' that attempts to " \
                                                     "link to this factor node has size '{}', which does not match the " \
                                                     "size '{}' of the variable with the same name already present in " \
                                                     "the variable lists of this factor node." \
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
            Compute and return the bradcasted message with its dimension AND shape properly aligned with the factor node
                variable order. This means the variable dimensions of the message will be first permuted, and then the
                message will be manually broadcasted to full shape.
            Manual broadcast is enforced here to prevent dimension misalignment.

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
        # Permute message dimension
        aligned_msg = msg.view(view_dim).permute(perm)
        # Manual broadcast by expanding dimension
        full_shape = (var.size for var in self._var_list)
        expanded_msg = aligned_msg.expand(full_shape)
        return expanded_msg

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


class VariableNode(Node, ABC):
    """
        Specify a **variable node**.
    """

    def __init__(self, name, var_list, epsilon=None):
        """
            Decalre a VariableNode
        :param name:        name of the variable node
        :param var_list:   list of Variables representing the variables of this variable nodes
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
                if in_vn is out_vn:  # Here use 'is' operator to test variable node's identity
                    continue
                msg = in_ld.read()
                buf = self.sp_product(buf, msg)

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
    def __init__(self, name, ):
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

        Note: ADFN only admits at most two pairs of incoming / outgoing links. (one pair in condition / action pattern,
            two pairs in condact pattern)

        Note: Special implementation needed for resolving potential conflicts between pattern variables and wm variables,
                especially mismatch in sizes.

        pt_var_info schema:
            { wm_var_name :
                  { "name" : pt_var_name
                    "type" : "var" or "const"
                    "vals" : int/str values if type is const or None otherwisea
                    "rel"  : relation, if specified, otherwise None } }
    """
    def __init__(self, name, pt_var_info):
        super(ADFN, self).__init__(name, function=None, func_var_list=None)
        self.pretty_log["node type"] = "Affine Delta Factor Node"
        self._pt_var_info = pt_var_info

    # Override to make sure only admit at most two pairs of incoming / outgoing links
    def add_link(self, linkdata):
        assert (not linkdata.to_fn or len(self._in_linkdata) <= 1), "Attempting to add more than two incoming links"
        assert (linkdata.to_fn or len(self._out_linkdata) <= 1), "Attempting to add more than two outgoing links"
        super(ADFN, self).add_link(linkdata)

    # Override compute() to implement tensor manipulation
    def compute(self):
        """
            Three cases for each variable dimension:
                1. If associated pattern var shared among multiple wm var:
                    - If message goes inward, take diagonal values and shrink dimension
                    - If message goes outward, expand dimensions, put values on diagonals and put 0 everywhere else
                2. If associated pattern var is constant,
                    - If message goes inward, take the slices and concatenate, slices taken in the order specified by 'vals'
                    - If message goes outward, split into slices and put in place. Put 0 everywhere else
                3. If associated pattern var is a variable but is distinct, then simply swap variable.

            In cases 1 and 3, need to account for mismatch in variable dimension sizes when swapping variables:
                - If need to enlarge dimension size, append necessary slices of 0's at the tail of that dimension
                - If need to shrink dimension size, truncate necessary slices at the tail of that dimension

            # TODO: Accomodate for within-pattern variable relations in future iterations
                        E.g.  pred[ (arg1 v) (arg2 (v + 3)) ]
                    Or more generally,
                        E.g.  pred[ (arg1 v) (arg2 f(v)) ]
                    Where f(v) is a custom mapping embodying the relation
        """
        # Check that there are equal number of incoming links and outgoing links
        assert len(self._in_linkdata) == len(self._out_linkdata), \
            "The number of of incoming links ({}) do not match the number of outgoing links ({}). " \
                .format(len(self._in_linkdata), len(self._out_linkdata))

        for in_ld in self._in_linkdata:
            out_ld = [ld for ld in self._out_linkdata if ld.vn is not in_ld.vn][0]

            # Compare var_list against pt_var_info and determin wm vars and pt vars
            # Use set to discard orders for comparison
            in_ld_varnames, out_ld_varnames = set(var.name for var in in_ld.var_list), set(var.name for var in out_ld.var_list)
            wm_varnames, pt_varnames = set(self._pt_var_info.keys()), set(v["names"] for v in self._pt_var_info.values())
            assert (in_ld_varnames == wm_varnames and out_ld_varnames == pt_varnames) or \
                   (in_ld_varnames == pt_varnames and out_ld_varnames == wm_varnames), \
                "wm vars and pattern vars provided by pt_var_info does not match that as specified in the linkdata. " \
                "In pt_var_info: wm vars - '{}', pattern vars - '{}'. " \
                "In linkdata: incoming link vars - '{}', outgoing link vars - '{}'." \
                .format(wm_varnames, pt_varnames, in_ld_varnames, out_ld_varnames)

            # Determine direction of this pair of links. Inward or outward, w.r.t. beta network
            inward = True if in_ld_varnames == wm_varnames else False

            in_varname_list = list(var.name for var in in_ld.var_list)
            out_varname_list = list(var.name for var in out_ld.var_list)
            # in_var2dim = {v.name: k for k, v in enumerate(in_ld.var_list)}
            # out_var2dim = {v.name: k for k, v in enumerate(out_ld.var_list)}
            in_dim = [var.size for var in in_ld.var_list]
            out_dim = [var.size for var in out_ld.var_list]

            msg = in_ld.read()      # read message
            assert isinstance(msg, torch.Tensor)
            trace_varnames = list(var.name for var in in_ld.var_list)  # Trace the change of variable dimensions of msg

            # Build ptvar -> wmvar lookup table to distinguish shared pt vars from distinct ones
            ptvar2wmvar = {}
            for wm_varname, v in self._pt_var_info.items():
                pt_varname = v['name']
                if pt_varname not in ptvar2wmvar.keys():
                    ptvar2wmvar[pt_varname] = [wm_varname]
                else:
                    ptvar2wmvar[pt_varname].append(wm_varname)

            for wm_varname, v in self._pt_var_info.items():
                pt_varname = v['name']
                # First do case 2, constant variables. Throughout this step, variable order should not be altered
                if v['type'] == "const":
                    vals = v['vals']
                    # If inward, simply index select
                    if inward:
                        dim = in_varname_list.index(wm_varname)
                        msg = msg.index_select(dim=dim, index=torch.LongTensor(vals))
                        trace_varnames[dim] = pt_varname
                    # If outward, then need to take slice separately, create empty slices, and concatenate
                    else:
                        dim = in_varname_list.index(pt_varname)
                        assert in_dim[dim] == len(vals)     # This should always be correct since size of const pt vars are determined by len of vals
                        slice_shape = list(msg.shape)
                        slice_shape[dim] = 1
                        dim_target_size = out_dim[out_varname_list.index(wm_varname)]
                        # Slice messages
                        sliced_msg = msg.split(1, dim=dim)      # Split into tuple of tensors with shape slice_shape
                        assert all(list(t.shape) == slice_shape for t in sliced_msg)
                        # Create target message
                        target_slices = list(torch.zeros(size=slice_shape)
                                             if i not in vals else sliced_msg[vals.index(i)]
                                             for i in range(dim_target_size))
                        # Concatenate to form new message
                        msg = torch.cat(target_slices, dim=dim)
                        trace_varnames[dim] = wm_varname

                # Second do case 3, swap distinct pt vars and account for mismatch in size
                elif len(ptvar2wmvar[pt_varname]) == 1:
                    if inward:      # inward should be enlarging size, if any
                        dim = in_varname_list.index(wm_varname)
                        wm_size = in_dim[dim]
                        pt_size = out_dim[out_varname_list.index(pt_varname)]
                        assert wm_size <= pt_size
                        trace_varnames[dim] = pt_varname
                        # Enlarge dimension size with zero tensor if needed
                        if pt_size > wm_size:
                            extra_shape = list(msg.shape)
                            extra_shape[dim] = pt_size - wm_size
                            msg = torch.cat((msg, torch.zeros(size=extra_shape)), dim=dim)
                    else:           # outward should be shrinking size, if any
                        dim = in_varname_list.index(pt_varname)
                        pt_size = in_dim[dim]
                        wm_size = out_dim[out_varname_list.index(wm_varname)]
                        assert wm_size <= pt_size
                        trace_varnames[dim] = wm_varname
                        # Shrink dimension size by truncating tail is needed
                        if pt_size > wm_size:
                            msg = torch.split(msg, wm_size, dim=dim)[0]     # Only take the first chunk

            # Finally case 1, where variable order may be changed due to taking the diagonal
            for pt_varname, wm_varname_list in ptvar2wmvar.items():
                if len(wm_varname_list) > 1:
                    # If message goes inward, take diagonal values and shrink dimension
                    if inward:
                        # Diagonalize first pair
                        wm_varname1 = wm_varname_list[0]
                        wm_varname2 = wm_varname_list[1]
                        dim1 = trace_varnames.index(wm_varname1)    # use trace_varnames because var order may have
                        dim2 = trace_varnames.index(wm_varname2)    # changed
                        msg = torch.diagonal(msg, dim1=dim1, dim2=dim2)
                        trace_varnames.remove(wm_varname1)
                        trace_varnames.remove(wm_varname2)
                        trace_varnames.append(pt_varname)       # Diagonalized dimension is appended as the last dim

                        # If there are more than two wm vars sharing the same pt vars
                        for i in range(2, len(wm_varname_list)):
                            wm_varname = wm_varname_list[i]
                            dim1 = trace_varnames.index(wm_varname)     # use trace_varnames because var order changed
                            msg = torch.diagonal(msg, dim1=dim1, dim2=-1)
                            trace_varnames.remove(wm_varname)

                        # Accomodate for mismatch in pt var dimension size
                        tar_dim = out_dim[out_varname_list.index(pt_varname)]
                        cur_dim = msg.shape[-1]
                        if tar_dim > cur_dim:
                            extra_shape = list(msg.shape)
                            extra_shape[-1] = tar_dim - cur_dim
                            msg = torch.cat((msg, torch.zeros(size=extra_shape)), dim=-1)

                    # If message goes outward, expand dimensions, put values on diagonals and put 0 everywhere else
                    else:
                        # Will permute the working dimension to the front for easy indexing
                        dim = trace_varnames.index(pt_varname)
                        perm = (dim,) + tuple(i for i in range(msg.dim()) if i is not dim)  # index permutation for msg
                        perm_msg = msg.permute(perm)        # working pt var dimension now put at front
                        wm_dim_sizes = tuple(out_dim[out_varname_list.index(wm_varname)]
                                             for wm_varname in wm_varname_list)
                        tar_shape = wm_dim_sizes + tuple(perm_msg.size())[1:]
                        buf = torch.zeros(tar_shape)        # zero tensor where diag val will be put on first two dim

                        # Account for mismatch in size
                        for i in range(min(wm_dim_sizes)):
                            buf[(i,) * len(wm_dim_sizes)] += perm_msg[i]
                        msg = buf

                        # bookkeeping stuff
                        trace_varnames.remove(pt_varname)
                        trace_varnames = wm_varname_list + trace_varnames

            # Final step: permute variable dimension to align with out_varname_list and check dim against out_dim
            assert all(var in out_varname_list for var in trace_varnames)
            assert all(var in trace_varnames for var in out_varname_list)

            perm = tuple(trace_varnames.index(var) for var in out_varname_list)
            msg = msg.permute(perm)
            assert list(msg.size()) == out_dim

            out_ld.set(msg, self._epsilon)


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


class WMVN(VariableNode):
    """
        Working Memory Variable Node
    """

    # TODO
    def __init__(self, name, var_list):
        super(WMVN, self).__init__(name, var_list)
        self.pretty_log["node type"] = "Working Memory Variable Node"
