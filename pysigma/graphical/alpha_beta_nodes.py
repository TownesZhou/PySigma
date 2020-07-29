"""
    All nodes related to a conditional subgraph
"""
import copy
from abc import ABC, abstractmethod
import torch
from torch.distributions import Transform
from torch.distributions.constraints import Constraint
from defs import VariableMetatype, Variable, MessageType, Message, NP_EPSILON
from graphical.basic_nodes import LinkData, VariableNode, FactorNode
from utils import KnowledgeServer
from structures import VariableMap, Summarization


class AlphaFactorNode(FactorNode, ABC):
    """Abstract base class for any factor node belonging to an alpha subgraph.

    Captures the commonalities of all alpha subgraph nodes:

    * **Topology**: an alpha node accepts up to two pairs of incoming and outgoing linkdata, with one pair propagating
      messages inward toward the Gamma Factor Node, and the other pair propagating messages outward toward the predicate
      Working Memory Variable Node.
    * **Admissible variable nodes**: an alpha node treats the relational variables and manipulates messages' batch
      dimensions only, and leaves untouched the random variables and corresponding event dimensions. Therefore,
      incident variable nodes should have the same tuple of random variables.
    * **Compute pattern**: an alpha node computes outgoing messages for each pair of linkdata individually. In other
      words, for instance, the outgoing message to an inward outgoing link is solely dependent on the message received
      from the inward incoming link. Accordingly, the `compute()` method is subdivided into an `inward_compute()` and an
      `outward_compute()` method.
    * **Quiescence state**: an alpha node as a whole reaches quiescence if and only if **all** incoming linkdata do not
      contain new message. However, for the two subdivided method `inward_compute()` and `outward_compute()`, each of
      them should only be carried out if its incoming linkdata of interest contains new message.
    """

    def __init__(self, name, **kwargs):
        super(AlphaFactorNode, self).__init__(name, **kwargs)

        # Pairs of incoming and outgoing linkdata labeled with their directionality w.r.t. the alpha structure
        self.labeled_ld_pair = {}
        self.ran_vars = None

    def add_link(self, linkdata):
        """An alpha factor node admits at least one but no more than two pairs of incoming and outgoing linkdata. The
        incident variable nodes should have the same tuple of random variables. Furthermore, a ``"direction"`` key-ed
        attribute should be included in the linkdata's optional attribute map with value ``"inward"`` or ``"outward"``
        to indicate the message propagation direction of the linkdata. The two pairs of linkdata should have different
        message propagation directions.

        Parameters
        ----------
        linkdata : LinkData
            The linkdata to be registered. Must specify a ``"direction"`` attribute in its optional attribute map
            ``linkdata.attr``.
        """
        assert isinstance(linkdata, LinkData)
        # Check random variables
        if self.ran_vars is None:
            self.ran_vars = linkdata.vn.ran_vars
        else:
            assert self.ran_vars == linkdata.vn.ran_vars

        assert 'direction' in linkdata.attr and linkdata.attr['direction'] in ['inward', 'outward']

        if linkdata.to_fn:
            assert len(self.in_linkdata) == 0 or linkdata.attr['direction'] != self.in_linkdata[0].attr['direction']
            assert len(self.in_linkdata) <= 1
        else:
            assert len(self.out_linkdata) == 0 or linkdata.attr['direction'] != self.out_linkdata[0].attr['direction']
            assert len(self.out_linkdata) <= 1
        super(AlphaFactorNode, self).add_link(linkdata)

        # If the other ld of this ld pair has not been added, then temporarily register this ld instance directly
        direction = linkdata.attr['direction']
        if direction not in self.labeled_ld_pair.keys():
            self.labeled_ld_pair[direction] = linkdata
        # Otherwise, take out the other ld of this ld pair from the dict and replace entry with a tuple
        #   Make sure that incoming ld is the first element of the tuple and outgoing ld the second element
        else:
            other_ld = self.labeled_ld_pair[direction]
            self.labeled_ld_pair[direction] = (linkdata, other_ld) if linkdata.to_fn else (other_ld, linkdata)

    def compute(self):
        """Carries out `inward_compute()` and `outward_compute()` individually if their corresponding incoming linkdata
        contains new message.
        """
        super(AlphaFactorNode, self).compute()
        assert len(self.in_linkdata) == len(self.out_linkdata) and len(self.in_linkdata) > 0

        # Carry out directional computation
        for direction, (in_ld, out_ld) in self.labeled_ld_pair.items():
            if direction == 'inward' and in_ld.new:
                self.inward_compute(in_ld, out_ld)
            if direction == 'outward' and out_ld.new:
                self.outward_compute(in_ld, out_ld)

    @abstractmethod
    def inward_compute(self, in_ld, out_ld):
        """Inward message computation. To be implemented by child class.

        """
        raise NotImplementedError

    @abstractmethod
    def outward_compute(self, in_ld, out_ld):
        """Outward message computation. To be implemented by child class.

        """
        raise NotImplementedError


class ESFN(AlphaFactorNode):
    """Expansion / Summarization Factor Node

    This node should be connected to two DVNs that share the same set of random variables but perhaps different
    relational variables, with the DVN on the outward propagation side having a set of relational variables that is a
    subset of the set of relational variables of the DVN on the inward propagation side. The task of this node is to
    manipulate the incoming message's batch dimensions so that they align with the relational variable dimensions of the
    target DVN. Specifically:

    * For the inward propagation, since the target DVN may include relational variables that do not present in the
      source DVN, this procedure consists of **expansion** of the message's batch dimensions to make space for these
      missing relational variables, as well as **permutation** of the expanded dimensions so that they are in the same
      order as demanded by the target DVN.
    * For the outward propagation, it's the exact opposite. **Summarization** across the message batch dimensions
      associated with those un-referenced relational variables is first carried out, followed by the same
      **permutation** procedure.

    The summarization step can be thought of as a search or optimization problem, for which one finds a single
    distribution instance that best "summarizes" the behaviors of an entire (finite) space of distribution instances,
    where the dimensions of the space are spanned by the un-referenced relational variables. The semantics of such
    summarization can vary depending on the use case. The default summarization procedure implements the semantics of
    "combining" distributions, i.e.::

        p(x) = p_1(x) * p_2(x) * ... * p_k(x)

    but a custom summarization procedure can also be declared using the `sum_op` argument to implement other semantics.

    Parameters
    ----------
    name : str
        Name of this node
    sum_op : Summarization, optional
        The summarization operation functor to be called during outward propagation. Defaults to ``None``, in which case
        the "combination" semantic is implemented.

    Attributes
    ----------
    sum_op
    """

    def __init__(self, name, sum_op=None, **kwargs):
        """
            Necessary data structure:

            :param sum_op:      None or a Summarization instance. Default is None.
        """
        super(ESFN, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Expansion / Summarization Factor Node"

        assert sum_op is None or isinstance(sum_op, Summarization)
        if sum_op is not None:
            raise NotImplementedError("Summarization operation using Summarization instance is not yet implemented.")

        self.sum_op = sum_op

        self.pretty_log["node type"] = "Expansion / Summarization Factor Node"

    def inward_compute(self, in_ld, out_ld):
        """Expands and permutes the incoming message's relational variable dimensions to match the target outgoing
        relational variable dimensions.

        Raises
        ------
        AssertionError
            If the set of source DVN's relational variables is not a subset of the set of target DVN's relational
            variables.
        """
        msg = in_ld.read()
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        assert isinstance(msg, Message)
        in_rel_vars, out_rel_vars = in_ld.vn.rel_vars, out_ld.vn.rel_vars
        # Check that the set of relational variables of incoming message is a subset of that of outgoing message
        assert set(in_rel_vars).issubset(set(out_rel_vars)), \
            "At {}: during inward propagation, expect the set of source DVN's relational variables being a subset of " \
            "the set of target DVN's relational variables. Found variables {} from incoming linkdata, but variables " \
            "{} from outgoing linkdata.".format(self.name, in_rel_vars, out_rel_vars)

        # Keep a running list of variables
        mapped_var_list = list(in_rel_vars)

        # For every relational variable in out_rel_var_list that is not in in_rel_var_list, unsqueeze a dimension of the
        #   message as the last dimension.
        for rel_var in out_rel_vars:
            if rel_var not in in_rel_vars:
                # Unsqueeze message
                msg = msg.batch_unsqueeze(dim=-1)
                # Append the pt_var to the end of running var list
                mapped_var_list.append(rel_var)

        assert set(mapped_var_list) == set(out_rel_vars)

        # Permute message batch dimension so that it matches the order given by out_rel_var_list
        perm_order = list(mapped_var_list.index(v) for v in out_rel_vars)
        msg = msg.batch_permute(perm_order)

        # Expand to full relational variable dimension shape
        target_shape = torch.Size([v.size for v in out_rel_vars])
        msg = msg.batch_expand(target_shape)

        # Send message
        out_ld.write(msg)

    def outward_compute(self, in_ld, out_ld):
        """Summarizes over incoming message's un-referenced relational variable dimensions and permute to match the
        target outgoing relational variable dimensions.

        The summarization semantic is defined by the sum_op specified during initialization. If sum_op is None,
        uses default summarization semantic defined at the Message level.

        Raises
        ------
        AssertionError
            If the set of source DVN's relational variables is not a subset of the set of target DVN's relational
            variables.
        """
        msg = in_ld.read()
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        assert isinstance(msg, Message)
        in_rel_vars, out_rel_vars = in_ld.vn.rel_vars, out_ld.vn.rel_vars
        # Check that the set of relational variables of outgoing message is a subset of that of incoming message
        assert set(in_rel_vars).issubset(set(out_rel_vars))

        # Keep a running list of variables
        mapped_var_list = list(in_rel_vars)

        # Summarize using custom sum_op
        if self.sum_op is not None:
            # Flatten both the group of dimensions to be summarized over and the group of other dimensions. Put the
            #   former as the first dimension and the latter as the last batch dimension
            sum_dims = list(dim for dim, v in enumerate(in_rel_vars) if v not in out_rel_vars)
            other_dims = list(dim for dim, v in enumerate(in_rel_vars) if v in out_rel_vars)
            if len(sum_dims) > 0:
                # First flatten sum_dims, then other_dims, so that flattened sum_dims will be the first dim
                msg = msg.batch_flatten(sum_dims)
                msg = msg.batch_flatten(other_dims)
                # Process using the sum_op
                msg = self.sum_op(msg, self.ran_vars)
                # Reshape
                msg = msg.batch_reshape(other_dims)

        # Otherwise if sum_op is None, carry out default summarization
        else:
            # Iterate over all relational variables not referenced by out_rel_var_list
            for pt_var in in_rel_vars:
                if pt_var not in out_rel_vars:
                    dim = mapped_var_list.index(pt_var)
                    # Summarize over the message batch dimension
                    msg = msg.batch_summarize(dim)
                    # Remove the variable from the running list
                    mapped_var_list.remove(pt_var)

        assert set(mapped_var_list) == set(out_rel_vars)

        # Permute message dimension so that it matches the outgoing relational variable order
        perm_order = list(mapped_var_list.index(v) for v in out_rel_vars)
        msg = msg.batch_permute(perm_order)

        # Send message
        out_ld.write(msg)


class RMFN(AlphaFactorNode):
    """Relation Variable Mapping Node

    Converts predicate arguments from and to pattern variables. Applies relational variable's `VariableMap` (if
    specified) by manipulating message batch dimensions. This node can thus implements the semantic of inner-pattern
    relational variable matching.

    Mappings between the source relational arguments and target relational variables as well as that from source
    relational arguments to `VariableMap` instances need to be provided.

    Parameters
    ----------
    name : str
        Name of this node
    arg2var : dict
        Dictionary mapping predicate relational argument `Variable` instance to pattern relational variable `Variable`
        instance.
    var2arg : dict
        Dictionary mapping pattern relational variable `Variable` instance to LIST of predicate relational argument
        `Variable` instances. The list length would be longer than 1 if the relational variable is being referenced
        by multiple arguments.
    arg2var_map : dict
        Dictionary mapping predicate relational argument `Variable` instance to a `VariableMap` instance if one is
        specified for this argument. If an argument is not associated with a `VariableMap`, it should not appear in this
        dictionary.

    Attributes
    ----------
    arg2var
    var2arg
    arg2var_map
    arg_set : set
        The set of predicate argument `Variable` instances. Obtained from the keys of `arg2var`.
    var_set : set
        The set of pattern variable `Variable` instances. Obtained from the keys of `var2arg`.
    arg2var_map_tuple : dict
        Obtained from `arg2var_map`. Mapping from predicate argument to the mapping dictionary (if a `VariableMap` is
        specified).
    arg2var_map_inv_tuple : dict
        Obtained from `arg2var_map`. Mapping from predicate argument to the INVERSE mapping dictionary (if a
        `VariableMap` is specified).
    """

    def __init__(self, name, arg2var, var2arg, arg2var_map, **kwargs):
        super(RMFN, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Relation Variable Mapping Node"
        assert isinstance(name, str)
        assert isinstance(arg2var, dict) and all(isinstance(k, Variable) for k in arg2var.keys()) and \
               all(isinstance(v, Variable) for v in arg2var.values())
        assert isinstance(var2arg, dict) and all(isinstance(k, Variable) for k in var2arg.keys()) and \
               all(isinstance(v, list) and all(isinstance(arg, Variable) for arg in v) for v in var2arg.values())
        assert isinstance(arg2var_map, dict) and all(isinstance(k, Variable) for k in arg2var_map.keys()) and \
               all(isinstance(v, VariableMap) for v in arg2var_map.values())

        self.arg2var = arg2var
        self.var2arg = var2arg
        self.arg2var_map = arg2var_map

        self.arg_set = set(self.arg2var.keys())
        self.var_set = set(self.var2arg.keys())

        # Sanity check whether the variables given in the three dicts are compatible.
        assert set(self.arg2var.values()) == self.var_set
        arg_set = set()
        for arg_list in self.var2arg.values():
            arg_set.union(set(arg_list))
        assert arg_set == self.arg_set
        assert set(self.arg2var_map.keys()).issubset(self.arg_set)

        # Obtain mapping dictionary and inverse mapping dictionary
        self.arg2var_map_tuple = {arg: var_map.get_map() for arg, var_map in self.arg2var_map.items()}
        self.arg2var_map_inv_tuple = {arg: var_map.get_inverse_map() for arg, var_map in self.arg2var_map.items()}

        self.pretty_log["node type"] = "Relation Variable Mapping Node"

    def inward_compute(self, in_ld, out_ld):
        """Inward computation. Converts predicate relational arguments to pattern relational variables. Applies mappings
        to relational variable's values, if specified.

        For inward direction, we are assuming this is used in condition or condact patterns. Accordingly, the inverse
        mapping should be used to map predicate arguments to pattern variables.

        Will check anyway if domain and image of the inverse map agree with the size range of the predicate argument
        and pattern variable respectively. However to be user friendly this should be checked beforehand by compiler.

        Note that domain should be a subset of predicate argument size range, but image should be exactly equal to
        the pattern variable size range

        The computations to be carried out can be summarized in three steps: map/broaden, diagonalize, & permute
        """
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        msg = in_ld.read()
        assert isinstance(msg, Message)
        in_rel_vars, out_rel_vars = in_ld.vn.rel_vars, out_ld.vn.rel_vars

        # Check that given data structures agree with variable lists of the incident variable node
        assert set(in_rel_vars) == self.arg_set
        assert set(out_rel_vars) == self.var_set
        # Check that mapping's domain and image agree with variables' sizes
        #   Note that for inward computation we are using the inverse map
        for arg, var_map_tuple in self.arg2var_map_inv_tuple.items():
            pat_var = self.arg2var[arg]
            _, domain, image = var_map_tuple
            assert domain.issubset(set(range(arg.size))), \
                "At {}: The VariableMap declared on the predicate argument {} would induce an image that exceeds the " \
                "argument's value range. The argument's value range is {}, but found image {}." \
                    .format(self.name, arg.name, set(range(arg.size)), domain)
            assert image == set(range(pat_var.size)), \
                "At {}: The VariableMap declared on the predicate argument {} should have a domain equal to the value " \
                "range of the associated pattern variable {}. The pattern variable's value range is {}, but found " \
                "VariableMap domain {}" \
                    .format(self.name, arg.name, pat_var.name, set(range(pat_var.size)), image)

        # 1. First, translate predicate arguments to pattern variables. This step involves broadening the variable
        #    dimension if predicate argument size is smaller than pattern variable size, or map predicate argument
        #    values to pattern variable values if a VariableMap is specified for the given predicate argument.
        #
        #    For the mapping we should use original forward mapping in combination with batch_index_select(), because we
        #    are selecting "image" to place in "domain", in the order mandated by domain.
        #
        #    For example, if the VariableMap is defined as  2 * var + 3 |-> arg , and `var` range is [0, 2], we will be
        #    selecting from the incoming message, the slices with indices 3, 5, 8, to place onto indices 0, 1, 2 of the
        #    outgoing message. To use batch_index_select(), we gather the indices [3, 5, 8] as the argument, which are
        #    obtained from the forward mapping dictionary.
        #
        #    Note that we have guaranteed that forward mapping's domain is equal to pattern variable's size range
        #    A running list of variables is maintained to keep track of variable dimensions
        mapped_var_list = list(in_rel_vars)
        for dim, pred_arg in enumerate(mapped_var_list):
            pat_var = self.arg2var[pred_arg]
            # Apply map if VariableMap is specified
            if pred_arg in self.arg2var_map_tuple.keys():
                map_dict, _, _ = self.arg2var_map_tuple[pred_arg]
                indices = torch.tensor(list(map_dict[i] for i in range(pat_var.size)), dtype=torch.long,
                                       device=self.device)
                msg = msg.batch_index_select(dim, indices)
            # Broaden the variable dimension size if currently it is smaller than the pattern variable's size
            if msg.b_shape[dim] < pat_var.size:
                msg = msg.batch_broaden(dim, pat_var.size)
            # Change predicate argument to pattern variable Variable instance
            mapped_var_list[dim] = pat_var

        # 2. The step above guarantees that for any dimensions that share the same pattern variable, their axis values
        #       are semantically identical to the pattern variable's value.
        #    Now we should collapse the dimensions that share the same pattern variable, by selecting the diagonal
        #       entries across these dimensions.
        for pt_var in mapped_var_list:
            # Current position of pt_var in the moving mapped_var_list. Using enumerate() iterator will not return
            #   index of value w.r.t. a moving list
            dim1 = mapped_var_list.index(pt_var)
            # Find any repetition in later part of the list
            if pt_var in mapped_var_list[dim1 + 1:]:
                # Current index of the first repetition
                dim2 = mapped_var_list[dim1 + 1:].index(pt_var) + dim1 + 1
                # Remove these two entries from the list
                mapped_var_list.remove(pt_var)
                mapped_var_list.remove(pt_var)
                # Append one to the end
                mapped_var_list.append(pt_var)
                # Selecting diagonal entries in message
                msg = msg.batch_diagonal(dim1, dim2)

        # Check that we have produced all the target relational variables, albeit with a different order perhaps
        assert set(mapped_var_list) == set(out_rel_vars)

        # 3. With all predicate argument dimension converted to pattern variable dimensions and all repetitions
        #       diagonalized, we guarantee that all predicate variable appears in mapped_var_list.
        #    The last thing to do is to permute the batch dimensions so that the processed message's dimension match
        #       exactly with out_rel_var_list
        perm_order = list(mapped_var_list.index(pt_var) for pt_var in out_rel_vars)
        msg = msg.batch_permute(perm_order)
        assert msg.b_shape == torch.Size([v.size for v in out_rel_vars])

        # Send message
        out_ld.write(msg)

    def outward_compute(self, in_ld, out_ld):
        """Outward computation. Converts pattern relational variables to predicate relational arguments. Applies
        mappings to relational variable's values, if specified.

        For outward direction, we are assuming this is used in action or condact patterns. Accordingly, the
        original forward mapping should be used to map pattern variables to predicate arguments.

        Will check anyway if domain and image of the forward map agree with the size range of the predicate argument
        and pattern variable respectively. However to be user friendly this should be checked beforehand by
        compiler.

        Note that image of the map should be a subset of predicate argument size range, but its domain should be
        exactly equal to the pattern variable size range.

        The computations to be carried out can be summarized in three steps: un-diagonalize, map/narrow, & permute
        """
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        msg = in_ld.read()
        assert isinstance(msg, Message)
        in_rel_vars, out_rel_vars = in_ld.vn.rel_vars, out_ld.vn.rel_vars

        # Check that given data structures agree with variable lists of the incident variable node
        assert set(in_rel_vars) == self.var_set
        assert set(out_rel_vars) == self.arg_set
        # Check that mapping's domain and image agree with variables' sizes
        #   For outward computation we are using the forward map
        for arg, var_map_tuple in self.arg2var_map_tuple.items():
            pat_var = self.arg2var[arg]
            _, domain, image = var_map_tuple
            assert image.issubset(set(range(arg.size))), \
                "At {}: The VariableMap declared on the predicate argument {} would induce an image that exceeds the " \
                "argument's value range. The argument's value range is {}, but found image {}." \
                    .format(self.name, arg.name, set(range(arg.size)), image)
            assert domain == set(range(pat_var.size)), \
                "At {}: The VariableMap declared on the predicate argument {} should have a domain equal to the value " \
                "range of the associated pattern variable {}. The pattern variable's value range is {}, but found " \
                "VariableMap domain {}" \
                    .format(self.name, arg.name, pat_var.name, set(range(pat_var.size)), domain)

        # 1. First, translate pattern variables to predicate arguments. This step involves unbinding the predicate
        #       variables that are referenced by multiple predicate arguments.
        #    Computationally, this is achieved by un-diagonalize, or embed entries along that predicate variables'
        #       dimension into a 2D plane (higher dimensional space if there are more than 2 such predicate arguments).
        #    A running list of variables is maintained to keep track of variable dimensions
        mapped_var_list = copy.deepcopy(in_rel_vars)
        for pt_var in mapped_var_list:
            # Look up how many predicate arguments reference this single pattern variable. If only 1, simply change
            #   variable. Otherwise, need to do more
            # Use .index() to get the position of current variable, instead of from enumerate() iterator, because the
            #   the latter does not keep up with a running list.
            dim1 = mapped_var_list.index(pt_var)
            args = self.var2arg[pt_var]
            num_shared = len(args)
            assert num_shared > 0
            # Switch pattern variable to predicate argument in-place anyway. If the pattern variable is associated with
            #   multiple predicate arguments, process further.
            mapped_var_list[dim1] = args[0]
            if num_shared > 1:
                # Iterate over rest of the associated predicate arguments
                for j, arg in enumerate(args):
                    # Only arguments from the second one will be processed. First associated argument stays in-place.
                    if j == 0:
                        continue
                    # Un-diagonalize the message by embedding the entries along the i-th dimension of the original
                    #   message into a 2D plane specified by the i-th and the last dimension of the new message
                    msg = msg.batch_diag_embed(diag_dim=dim1, target_dim1=dim1, target_dim2=-1)
                    # This corresponds to append the other dimension to the end, so we append the currently referenced
                    #   predicate argument Variable instance to the running variable list
                    mapped_var_list.append(arg)

        assert set(mapped_var_list) == set(out_rel_vars)

        # 2. The step above guarantees a symbolic one-to-one mapping between message dimensions and predicate arguments'
        #       variable dimensions. We now need to narrow the variable dimension if the predicate argument's size is
        #       smaller than the associated pattern variable's size, or to map pattern variable's value to predicate
        #       argument's value if a VariableMap is specified.
        for dim, pred_arg in enumerate(mapped_var_list):
            pat_var = self.arg2var[pred_arg]
            # Apply map if VariableMap is specified
            if pred_arg in self.arg2var_map_tuple.keys():
                map_dict, _, _ = self.arg2var_map_tuple[pred_arg]
                indices = torch.tensor(list(map_dict[i] for i in range(pat_var.size)), dtype=torch.long,
                                       device=self.device)
                msg = msg.batch_index_put(dim, indices)
            # Broaden the variable dimension size if currently it is smaller than the pattern variable's size
            if msg.b_shape[dim] > pred_arg.size:
                msg = msg.batch_narrow(dim, pred_arg.size)

        # 3. Finally, with all pattern variables converted to predicate arguments and the values are now with respect to
        #       the predicate arguments, the last thing to do is to permute the batch dimensions so that the processed
        #       message's dimensions match exactly with out-_rel_var_list
        perm_order = list(mapped_var_list.index(pred_arg) for pred_arg in out_rel_vars)
        msg = msg.batch_permute(perm_order)
        assert msg.b_shape == torch.Size([v.size for v in out_rel_vars])

        # Send message
        out_ld.write(msg)


class BetaFactorNode(FactorNode, ABC):
    """Abstract base class for nodes belonging to the beta subgraph of a conditional.

    Captures the commonalities of beta subgraph nodes:

    * **Topology**: a beta node is typically connected to up to two pairs of *lists* of incoming and outgoing links,
      with each pair designated a unique message propagation direction. In other words, contrast to an alpha node, a
      beta node for example may be connected to multiple inward incoming links. However, the message propagation
      directions of the links are well grouped in that, if there is inward incoming link, then there must be a inward
      outgoing link.
    * **Admissible variable nodes**: a beta factor node treats the random variables only, and leave untouched the
      relational variables that should have been processed by alpha factor nodes. Thus, incident variable nodes should
      have the same tuple of relational variables.
    * **Compute pattern**: similar to an alpha factor node. However, since there may be multiple incoming links with
      same message propagation direction, each subdivided compute method, i.e., `inward_compute()` and
      `outward_compute()`, would be executed if **all** of its incoming linkdata contain new messages.
    * **Quiescence state**: similar to an alpha factor node. `inward_compute()` and `outward_compute()` will only be
      carried out if all of their corresponding incoming linkdata contain new messages.
    """

    def __init__(self, name, **kwargs):
        super(BetaFactorNode, self).__init__(name, **kwargs)

        # Pairs of incoming and outgoing linkdata list with their messaging direction w.r.t. the beta structure
        self.labeled_ld_list_pair = {
            'inward': ([], []),  # First list contains incoming links, and second outgoing links
            'outward': ([], [])
        }
        self.ran_vars = None

    def add_link(self, linkdata):
        """A Beta Factor Node admits at least one but no more than two pairs of lists of incoming and outgoing linkdata.
        The incident variable nodes should have the same tuple of random variables. Furthermore, a ``"direction"``
        key-ed attribute should be included in the linkdata's optional attribute map with value ``"inward"`` or
        ``"outward"`` to indicate the message propagation direction of the linkdata. The two pairs of linkdata should
        have different message propagation directions.

        Parameters
        ----------
        linkdata : LinkData
            The linkdata to be registered. Must specify a ``"direction"`` attribute in its optional attribute map
            ``linkdata.attr``.
        """
        assert isinstance(linkdata, LinkData)
        # Check random variables
        if self.ran_vars is None:
            self.ran_vars = linkdata.vn.ran_vars
        else:
            assert self.ran_vars == linkdata.vn.ran_vars

        assert 'direction' in linkdata.attr and linkdata.attr['direction'] in ['inward', 'outward']

        if linkdata.to_fn:
            self.labeled_ld_list_pair[linkdata.attr['direction']][0].append(linkdata)
        else:
            self.labeled_ld_list_pair[linkdata.attr['direction']][1].append(linkdata)

        super(BetaFactorNode, self).add_link(linkdata)

    def compute(self):
        """Carries out `inward_compute()` and `outward_compute()` separately only if all their corresponding incoming
        linkdata contains new message.
        """
        super(BetaFactorNode, self).compute()

        for direction, (in_ld_list, out_ld_list) in self.labeled_ld_list_pair.items():
            if len(in_ld_list) > 0 and len(out_ld_list) > 0:
                if direction == 'inward' and all(in_ld.new for in_ld in in_ld_list):
                    self.inward_compute(in_ld_list, out_ld_list)
                if direction == 'outward' and all(in_ld.new for in_ld in in_ld_list):
                    self.outward_compute(in_ld_list, out_ld_list)

    @abstractmethod
    def inward_compute(self, in_ld_list, out_ld_list):
        """
            Inward message computation. To be implemented by child class.
        """
        raise NotImplementedError

    @abstractmethod
    def outward_compute(self, in_ld_list, out_ld_list):
        """
            Outward message computation. To be implemented by child class.
        """
        raise NotImplementedError


class CMTN(BetaFactorNode):
    """Concatenation, Marginalization, & Transformation Node

    Transforms the joint predicate pattern messages from and to a list of univariate pattern element messages
    representative of the events of each of the referenced pattern random variables.

    By default, outgoing messages that consists of only particles contents will be marked with a special attribute
    ``preemptive == True``. Alternatively, a `preemptive` attribute can be set at the node level during the
    initialization of this node, in which case when set to ``True`` the outgoing message will always be marked with
    ``preemptive == True``.

    Marking a message with ``preemptive == True`` indicates that the event particle values of this message should be
    taken as the universal event values for the corresponding pattern variable during this cognitive cycle. Any other
    predicate pattern that references this pattern variable should be forced to adopt this list of event values.
    Therefore, at most one CNTN among all those that reference the same pattern variable can send preemptive messages,
    otherwise an exception will be raised.

    Parameters
    ----------
    name : str
        Name of this node
    args2var : dict
        The dictionary mapping either a single predicate random argument `Variable` instance, or a `tuple` of predicate
        random argument `Variable` instances, to a single pattern random variable `Variable` instance.
    var2trans : dict
        The dictionary mapping a pattern random variable `Variable` instance to a `Transform` instance, if a
        transformation is declared for this pattern variable. Accordingly, for each pattern element declared in the
        predicate pattern, there must be a corresponding FVN connected to this CMTN.
    preemptive : bool, optional
        Whether the outgoing messages to all downstream nodes will always be marked with ``preemptive == True``.
        Defaults to ``False``.

    Attributes
    ----------
    args2var
    var2trans
    preemptive
    """

    def __init__(self, name, args2var, var2trans, preemptive=False, **kwargs):
        super(CMTN, self).__init__(name, **kwargs)
        assert isinstance(args2var, dict) and \
               all((isinstance(k, Variable) and k.metatype is VariableMetatype.Random) or
                   (isinstance(k, tuple) and all((isinstance(arg, Variable) and arg.metatype is VariableMetatype.Random
                                                  for arg in k)))
                   for k in args2var.keys()) and \
               all((isinstance(v, Variable) and v.metatype is VariableMetatype.Random) for v in args2var.values())
        assert isinstance(var2trans, dict) and \
               all(isinstance(k, Variable) and k.metatype is VariableMetatype.Random for k in var2trans.keys()) and \
               all(isinstance(v, Transform) for v in var2trans.values())

        self.args2var = args2var
        self.var2trans = var2trans
        self.preemptive = preemptive

        self.pretty_log["node type"] = "Concatenation, Marginalization, & Transformation Node"

    def add_link(self, linkdata):
        """Some additional checks for CMTN

        """
        super(CMTN, self).add_link(linkdata)

        # If inward and outgoing, or outward and incoming, check that connects to a FVN and with valid ran var list size
        if (linkdata.to_fn and linkdata.attr['direction'] == 'outward') or \
                (not linkdata.to_fn and linkdata.attr['direction'] == 'inward'):
            assert isinstance(linkdata.vn, FVN) and len(linkdata.vn.ran_vars) == 1

    def inward_compute(self, in_ld_list, out_ld_list):
        """For each pattern element in the predicate pattern, generate a univariate message that is representative of
        the marginal events of the associated pattern variable. Sends this message to the FVN that corresponds to this
        pattern element.

        The above message generation step is done by the following procedure:

        1. Marginalize over any un-referenced predicate arguments 
        2. For each random pattern element:

           1. Compute the outgoing message's particles component:

              1. marginalize the message over all other predicate random arguments that are not referenced by this
                 pattern element.
              2. If this pattern element references multiple predicate random arguments in a list, then concatenate
                 these dimensions. The resulting message should consist of a single event dimension.
              3. If a transformation is declared on the associated pattern random variable, then apply the INVERSE of
                 the transformation on the message.

           2. Annotate the outgoing message's parameter component (if included in incoming message):

              1. Annotate the message with ``param_event_mismatch == False`` label to indicate the particles above
                 approximates the same distribution the parameters are encoding, only if there is only one pattern
                 element, and it references all predicate random arguments in the correct order (the original order when
                 these arguments were declared)
              2. In any other case, annotate the message with ``event_space=True`` label.

           3. If the message do not contain parameters components, or if ``self.preemptive == True``, annotate the
              outgoing message with special attribute ``preemptive == True``. Otherwise, annotate with
              ``preemptive == False``.

           3. Send this message to the corresponding outgoing linkdata
        """
        assert len(in_ld_list) == 1
        in_msg = in_ld_list[0].read()
        assert isinstance(in_msg, Message)
        ran_args = list(in_ld_list[0].vn.ran_vars)  # Obtain full random variable list
        ran_vars = list(out_ld.vn.ran_vars[0] for out_ld in out_ld_list)
        running_args = copy.deepcopy(ran_args)

        # Pre-compute check: same set of random arguments
        assert all((isinstance(k, Variable) and k in ran_args) or
                   (isinstance(k, tuple) and set(k).issubset(set(ran_args))) for k in self.args2var.keys())
        # Pre-compute check: same set of random variables
        assert set(self.args2var.values()) == set(ran_vars)

        # 1. Marginalize over any un-referenced predicate arguments 
        ref_args = []
        for k in self.args2var.keys():
            if isinstance(k, tuple):
                ref_args += list(k)
            else:
                ref_args.append(k)
        for arg in ran_args:
            if arg not in ref_args:
                in_msg = in_msg.event_marginalize(running_args.index(arg))
                running_args.remove(arg)

        # 2. For each random pattern element
        for args, pt_var in self.args2var:
            args = tuple(args) if not isinstance(args, tuple) else args
            out_ld = out_ld_list[ran_vars.index(pt_var)]
            out_msg = in_msg
            ele_running_args = copy.deepcopy(running_args)

            # 2.1.1 Marginalize message over all other arguments
            for arg in running_args:
                if arg not in args:
                    out_msg = out_msg.event_marginalize(ele_running_args.index(arg))
                    ele_running_args.remove(arg)

            # 2.1.2 If multiple arguments, then concatenate all
            if len(args) > 1:
                out_msg = out_msg.event_concatenate(tuple(range(len(args))))

            # 2.1.3 Apply INVERSE transformation on events if declared
            if pt_var in self.var2trans.keys():
                trans = self.var2trans[pt_var]
                trans_msg = out_msg.event_transform(trans.inv)
                # Re-compose the message transformed events and parameters
                out_msg = Message.compose(out_msg.reduce_type(MessageType.Parameter), trans_msg)

            # 2.2 Annotate message parameter if included
            if MessageType.Parameter in out_msg.type:
                # Annotate COMPLETE if args referenced by this pattern element is/are all the predicate arguments, and
                # in correct order
                if args == ran_args:
                    out_msg.attr['param_event_mismatch'] = 'False'
                else:
                    out_msg.attr['param_event_mismatch'] = 'True'

            # 2.3 Annotate preemptive
            if MessageType.Parameter not in out_msg.type or self.preemptive:
                out_msg.attr['preemptive'] = True
            else:
                out_msg.attr['preemptive'] = False

            # 3. send message to corresponding linkdata
            out_ld.write(out_msg)

    def outward_compute(self, in_ld_list, out_ld_list):
        """

        """
        raise NotImplementedError


class FVN(VariableNode):
    """Filter Variable Node

    Compares incoming message's particle values against the constraints declared for the random variable. Filters out
    the particles if they do not meet the constraints, and raise an exception in this case if the message does not
    contain parameters.

    Parameters
    ----------
    preemptive : bool, optional
        If ``True``, the outgoing message will always be marked with the special attribute ``preemptive == True``.
        Defaults to ``False``.

    Attributes
    ----------
    ran_var : Variable
        The single random variable this FVN is representing.
    val_constraints : set of torch.distributions.constraints.Constraint
        The value constraints extracted from `ran_var`.
    """

    def __init__(self, name, **kwargs):
        super(FVN, self).__init__(name, **kwargs)
        assert 'ran_var_list' in kwargs.keys() and len(list(kwargs['ran_var_list'])) == 1

        # Extract the random variable and value constraints
        self.ran_var = self.ran_vars[0]
        self.val_constraints = self.ran_var.constraints

        self.pretty_log["node type"] = "Filter Variable Node"

    def add_link(self, linkdata):
        """FVN only admits one incoming link, but can admit multiple outgoing links

        """
        super(FVN, self).add_link(linkdata)
        assert linkdata.to_fn or len(self.in_linkdata) <= 1

    def compute(self):
        """Filters particles if the particle values do not meet the pre-specified value constraints of the random
        random variables.

        Raises
        ------
        ValueError
            If the message being filtered does not contain parameters; this means the outgoing message would be None
            had we sent out the filtered message.
        """
        super(FVN, self).compute()

        assert len(self.in_linkdata) > 0
        in_ld = self.in_linkdata[0]
        in_msg = in_ld.read()
        assert isinstance(in_msg, Message)

        # Check event values
        passed = True
        for cstr in self.val_constraints:
            if not torch.all(cstr.check(in_msg.particles[0])):
                passed = False

        # Raise an exception if not passed and message does not contain any parameters
        if not passed and MessageType.Parameter not in in_msg.type:
            raise ValueError("At {}: Filtered a message with incompatible particle values that does not contain "
                             "parameters. The pre-specified value constraints are {}"
                             .format(self.name, self.val_constraints))

        # Otherwise, reduce message to pur particles type
        if not passed:
            out_msg = in_msg.reduce_type(MessageType.Particles)
        else:
            out_msg = in_msg

        # Send messages to each outgoing link
        for out_ld in self.out_linkdata:
            out_ld.send(out_msg)


class EAFN(BetaFactorNode):
    """Event Aggregation Factor Node

    An EAFN aggregates, for a represented conditional pattern random variable, a universal list of event particle values
    that should be adopted and evaluated by all predicate patterns who reference this pattern variable. In other words,
    the particles message sent by this node can be viewed as an augmented list of particles that both necessarily meets
    the value constraint of this pattern variable and sufficiently covers the important regions in the event space
    of each predicate pattern prior distribution.

    Connected from Filter Variable Nodes, an EAFN expects incoming messages to contain at least either particles or
    parameters, preferably both. In addition, messages should be marked with a boolean attribute ``preemptive``. A
    message with ``preemptive == True`` indicates its list of event particle values will be taken directly as the final
    universal list of event particle values, ignoring all other messages. An EAFN therefore expects to receive only
    one preemptive message, otherwise an exception will be raised. If all messages are not preemptive, then the lists of
    event values from all of them (if applicable, i.e., if said message contains particles) will be taken and
    concatenated to form the universal list.

    Parameters
    ----------
    name : str
        Name of this node.
    ran_var : Variable
        The pattern random variable this EAFN is representing.

    Attributes
    ----------
    name
    ran_var
    e_shape : torch.Size
        The event shape. Inffered from `ran_var`.
    """
    def __init__(self, name, ran_var, **kwargs):
        assert isinstance(ran_var, Variable) and ran_var.metatype is VariableMetatype.Random
        super(EAFN, self).__init__(name, **kwargs)

        self.ran_var = ran_var
        self.e_shape = torch.Size([self.ran_var.size])

        self.pretty_log["node type"] = "Event Aggregation Factor Node"

    def add_link(self, linkdata):
        """All connected variable node should only represent one single random variable, and it must be `self.ran_var`.

        """
        super(EAFN, self).add_link(linkdata)
        assert len(linkdata.vn.ran_vars) == 1 and linkdata.vn.ran_vars[0] == self.ran_var

    def inward_compute(self, in_ld_list, out_ld_list):
        """In no incoming message is preemptive, then aggregate particle event values and log sampling densities.
        Otherwise, take the particle event values and log sampling densities form the preemptive message. If there are
        more than one preemptive message, raise an exception.

        Outgoing message will be an identity Particles message.

        Raises
        ------
        ValueError
            If found multiple preemptive messages.
        """
        # For inward direction, there should be only one outgoing node
        assert len(out_ld_list) == 1

        in_msgs = [in_ld.read() for in_ld in in_ld_list]
        assert all(isinstance(msg, Message) and 'preemptive' in msg.attr.keys() for msg in in_msgs)

        # Check preemptiveness
        preempt_msgs = [msg for msg in in_msgs if msg.attr['preemptive']]
        preempt_lds = [in_ld for in_ld in in_ld_list if in_ld.read()['preemptive']]
        if len(preempt_msgs) > 1:
            raise ValueError("At {}: Found multiple preemptive messages sent from these linkdata: {}. "
                             "Please check if the associated predicate patterns are sending particles-only messages, "
                             "or if more than one of them have been set to send preemptive messages."
                             .format(self.name, preempt_lds, self.ran_var.name))
        elif len(preempt_msgs) == 1:
            # Found one preemptive message. Take it
            preempt_msg = preempt_msgs[0]
            particles, log_densities = preempt_msg.particles, preempt_msg.log_densities
            out_msg = Message(MessageType.Particles,
                              sample_shape=preempt_msg.sample_shape, event_shape=self.e_shape,
                              particles=particles, log_densities=log_densities, weight=1,
                              device=self.device)
        else:
            # Otherwise concatenate the particles from all messages that contain particles
            ptcl_msgs = [msg for msg in in_msgs if MessageType.particles in msg.type]
            particles_list = [msg.particles for msg in ptcl_msgs]
            densities_list = [msg.log_densities for msg in ptcl_msgs]
            cat_particles = torch.cat(particles_list, dim=0)
            cat_densities = torch.cat(densities_list, dim=0)
            cat_s_shape = torch.Size(sum(msg.s_shape[0] for msg in ptcl_msgs))
            out_msg = Message(MessageType.Particles,
                              sample_shape=cat_s_shape, event_shape=self.e_shape,
                              particles=cat_particles, log_densities=cat_densities, weight=1,
                              device=self.device)

        # Send message
        out_ld_list[0].write(out_msg)

    def outward_compute(self, in_ld_list, out_ld_list):
        raise NotImplementedError


class ERFN(BetaFactorNode):
    """Event Resolution Factor Node

    An ERFN resolves incompatible event particle values for a pattern random variable and sends messages that represent
    a predicate pattern's prior beliefs regarding the distributions of the said pattern random variable.

    Why this node exists:

    Messages propagating at this stage of the Beta subgraph still represents a predicate's own marginal prior belief
    regarding a conditional pattern variable. The next step of processing happened at Event Combination Factor Node
    combines messages from multiple predicate pattern branches, when these predicates match the same conditional pattern
    variable.

    For parameters, combination is carried out by vector addition. However, for particles, due to each predicate drawing
    their own list of event values at the start of the cognitive cycle, messages sent by each predicate may contain
    entirely different event values, which hinders direct combination.

    Therefore, an ERFN should step in and resolve the conflicts, hence the name "Event Resolution Factor Node". This is
    done by aggregating messages from all other pattern branches that matches the specified pattern variable and
    inspecting these messages for any conflicts. When conflicts exist, it **"borrows"** the event values from other
    messages and queries the probability densities of these borrowed event values w.r.t. the predicate's prior belief
    distribution encoded by parameters, thus forming the so-called "surrogate particle list".

    These augmented list of particles is then sent to ECFN, at which point it is guaranteed that all incoming messages
    would share the same list of event values.

    """
