"""
    All nodes related to a conditional subgraph
"""
from __future__ import annotations      # For postponed evaluation of typing annotations
from typing import Union, Optional, List, Tuple, Dict
from typing import Iterable as IterableType
import copy
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
import torch
from torch.distributions import Transform
from torch.distributions.constraints import Constraint
from ..defs import VariableMetatype, Variable, MessageType, Message, NP_EPSILON
from ..pattern_structures.variable_map import VariableMapCallable
from .basic_nodes import LinkData, VariableNode, FactorNode, NodeConfigurationError
from ..utils import KnowledgeServer
from ..structures import VariableMap
from ..pattern_structures.summarization import SummarizationClass, SummarizationError


class AlphaFactorNode(FactorNode, ABC):
    """Abstract base class for any factor node belonging to an alpha subgraph.

    Captures the commonalities of all alpha subgraph nodes:

    * **Topology**: an alpha node accepts up to two pairs of incoming and outgoing linkdata, with one pair propagating
      messages inward toward the Gamma Factor Node, and the other pair propagating messages outward toward the predicate
      Working Memory Variable Node.
    * **Admissible variable nodes**: an alpha node works on the relational variables and manipulates messages' batch
      dimensions only. It does not tamper with the random variables and corresponding event dimensions. Therefore,
      incident variable nodes should have the same tuple of random variables.
    * **Compute pattern**: an alpha node computes outgoing messages for each pair of linkdata individually. In other
      words, for instance, the outgoing message to an inward outgoing link is solely dependent on the message received
      from the inward incoming link. Accordingly, the `compute()` method is divided into an `inward_compute()` and an
      `outward_compute()` subroutine.
    * **Quiescence state**: an alpha node as a whole reaches quiescence if and only if **all** incoming linkdata do not
      contain new message. However, for the two subroutines `inward_compute()` and `outward_compute()`, either one
      should be carried out only if its incoming linkdata contains new message.
    """

    def __init__(self, name: str, **kwargs):
        super(AlphaFactorNode, self).__init__(name, **kwargs)

        # Pairs of incoming and outgoing linkdata labeled with their directionality w.r.t. the alpha structure
        self.labeled_ld_pair: Dict[str, Tuple[LinkData, LinkData]] = {}
        self.ran_vars: Optional[List[Variable]] = None

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
        assert 'direction' in linkdata.attr and linkdata.attr['direction'] in ['inward', 'outward'], \
            "In {}: For an Alpha Factor Node, the linkdata {} should specify a special attribute named 'direction', " \
            "with value either 'inward' or 'outward'.".format(self.name, linkdata)

        # Check random variables
        if self.ran_vars is None:
            self.ran_vars = linkdata.vn.ran_vars
        else:
            assert self.ran_vars == linkdata.vn.ran_vars, \
                "In {}: linkdata {} has conflicting random variables. This Alpha Factor Node infers random variables " \
                "{} from other previously registered linkdata, but found different random variables {} in this linkdata."\
                .format(self.name, linkdata, self.ran_vars, linkdata.vn.ran_vars)

        if linkdata.to_fn:
            assert len(self.in_linkdata) == 0 or linkdata.attr['direction'] != self.in_linkdata[0].attr['direction'], \
                "In {}: Attempting to register an incoming linkdata {} with {} conditional message propagation " \
                "direction, while already having another incoming linkdata {} with the same direction."\
                .format(self.name, linkdata, linkdata.attr['direction'], self.in_linkdata[0])
        else:
            assert len(self.out_linkdata) == 0 or linkdata.attr['direction'] != self.out_linkdata[0].attr['direction'],\
                "In {}: Attempting to register an outgoing linkdata {} with {} conditional message propagation " \
                "direction, while already having another outgoing linkdata {} with the same direction."\
                .format(self.name, linkdata, linkdata.attr['direction'], self.out_linkdata[0])
        super(AlphaFactorNode, self).add_link(linkdata)

        # If the other ld of this ld pair has not been added, then temporarily register this ld instance directly
        direction = linkdata.attr['direction']
        if direction not in self.labeled_ld_pair.keys():
            self.labeled_ld_pair[direction] = (linkdata,)
        # Otherwise, take out the other ld of this ld pair from the dict and replace entry with a tuple
        #   Make sure that incoming ld is the first element of the tuple and outgoing ld the second element
        else:
            other_ld = self.labeled_ld_pair[direction][0]
            self.labeled_ld_pair[direction] = (linkdata, other_ld) if linkdata.to_fn else (other_ld, linkdata)

    def precompute_check(self):
        """The basic computable condition for a AlphaFactorNode is that there is at least pair of incoming and outgoing
        linkdata of the same message propagation direction.

        This method shall be extended by an AlphaNode subclass if it has specific computable conditions.
        """
        if len(list(self.labeled_ld_pair.keys())) == 0:
            raise NodeConfigurationError("Wrong configuration for node {}: no registered linkdata.".format(self.name))
        if not all(len(pair) == 2 for pair in self.labeled_ld_pair.values()):
            raise NodeConfigurationError("Wrong configuration for node {}: incomplete pair of linkdata. Found {} "
                                         "inward linkdata and {} outward linkdata."
                                         .format(self.name,
                                                 len(self.labeled_ld_pair['inward']),
                                                 len(self.labeled_ld_pair['outward'])))

    @FactorNode.compute_control
    def compute(self):
        """Carries out `inward_compute()` and `outward_compute()` individually if their corresponding incoming linkdata
        contains new message.
        """
        # Carry out directional computation
        for direction, (in_ld, out_ld) in self.labeled_ld_pair.items():
            if in_ld.new:
                if direction == 'inward':
                    self.inward_compute(in_ld, out_ld)
                else:
                    self.outward_compute(in_ld, out_ld)

    @abstractmethod
    def inward_compute(self, in_ld: LinkData, out_ld: LinkData):
        """Inward message computation. To be implemented by child class.

        """
        raise NotImplementedError

    @abstractmethod
    def outward_compute(self, in_ld: LinkData, out_ld: LinkData):
        """Outward message computation. To be implemented by child class.

        """
        raise NotImplementedError


class ESFN(AlphaFactorNode):
    """Expansion / Summarization Factor Node

    This node should be connected to two DVNs that share the same set of random variables but perhaps different
    relational variables。The set of relational variables of the DVN on the outward message propagation side should be a
    subset of the relational variables of the inward-side DVN.

    The task of this node is to manipulate the incoming message's batch dimensions so that they align with the
    relational variable dimensions of the target DVN. Specifically:

    * For the inward propagation, since the target DVN may include relational variables that do not present in the
      source DVN, this procedure may **expand/un-squeeze** the incoming messages' batch dimensions to make space for
      these missing relational variables, and then **permute** the expanded dimensions so that they are in the
      correct order as demanded by the target DVN.
    * For the outward propagation, the inverse procedure is carried out. Specifically, it may **summarize** across the
      message batch dimensions associated with those un-referenced relational variables, and then, same as the
      inward direction, **permute** the summarized dimensions so they are correctly aligned.

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

    def __init__(self, name: str, sum_op: SummarizationClass, **kwargs):
        """
            Necessary data structure:

            :param sum_op:      None or a Summarization instance. Default is None.
        """
        super(ESFN, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Expansion / Summarization Factor Node"

        assert isinstance(sum_op, SummarizationClass)
        self.sum_op = sum_op

    def precompute_check(self):
        """For ESFN, check that the outward DVN's relational variable set is a subset of the inward DVN's relational
        variable set, and that these two DVNs have identical random variables.
        """
        super(ESFN, self).precompute_check()

        # If there's an inward message pathway, then incoming ld's vn is the Outward-side node, and outgoing ld's vn
        #   is the Inward-side node
        # If an outward message pathway, the vice versa.
        if 'inward' in self.labeled_ld_pair.keys():
            in_ld, out_ld = self.labeled_ld_pair['inward']
            out_dvn, in_dvn = in_ld.vn, out_ld.vn
        else:
            in_ld, out_ld = self.labeled_ld_pair['outward']
            in_dvn, out_dvn = in_ld.vn, out_ld.vn

        in_rel_vars, out_rel_vars = in_dvn.rel_vars, out_dvn.rel_vars

        assert set(out_rel_vars).issubset(set(in_rel_vars)), \
            "In {}: The outward DVN's relational variable set should be an subset of the inward DVN's relational " \
            "variable set. Instead, found inward DVN random variables: {}, and outward DVN random variables: {}"\
            .format(self.name, list(in_rel_vars), list(out_rel_vars))

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
        in_rel_vars, out_rel_vars = in_ld.vn.rel_vars, out_ld.vn.rel_vars

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

        The summarization semantic is defined by the sum_op specified during initialization.
        """
        msg = in_ld.read()
        in_rel_vars, out_rel_vars = in_ld.vn.rel_vars, out_ld.vn.rel_vars

        # Skip summarization if identical set of relational variables
        rest_vars = in_rel_vars
        if not set(in_rel_vars) == set(out_rel_vars):

            # Flatten both the group of dimensions to be summarized over and the group of other dimensions. Put the
            #   former as the first dimension and the latter as the last batch dimension
            sum_dims = list(dim for dim, v in enumerate(in_rel_vars) if v not in out_rel_vars)
            other_dims = list(dim for dim, v in enumerate(in_rel_vars) if v in out_rel_vars)
            # First flatten sum_dims, then other_dims, so that flattened sum_dims will be the first dim
            msg = msg.batch_flatten(sum_dims)
            msg = msg.batch_flatten(other_dims)
            # Process using the provided sum op
            try:
                msg = self.sum_op(msg)
            except SummarizationError as err:
                # Append node name to the exception message
                raise type(err)('In {}: '.format(self.name) + str(err)).with_traceback(sys.exc_info()[2])
            # Reshape
            rest_vars = [in_rel_vars[dim] for dim in other_dims]
            rest_shape = [v.size for v in rest_vars]
            msg = msg.batch_reshape(rest_shape)

            assert set(rest_vars) == set(out_rel_vars)

        # Permute message dimension so that it matches the outgoing relational variable order
        perm_order = list(rest_vars.index(v) for v in out_rel_vars)
        msg = msg.batch_permute(perm_order)

        # Send message
        out_ld.write(msg)


class RMFN(AlphaFactorNode):
    """Relation Variable Mapping Node

    Converts predicate arguments from and to pattern variables. Applies relational variable's `VariableMap` (if
    specified) by manipulating message batch dimensions. This node can thus implements the semantic of inner-pattern
    relational variable matching.

    Caller must provide one `VariableMapCallable` for each of the pattern element, i.e. the binding of a predicate
    argument and pattern variable, of this predicate pattern.

    Parameters
    ----------
    name : str
    variable_maps : Iterable of VariableMapCallable
        The VariableMapCallable instances specifying the bindings and mappings between the predicate arguments and
        pattern variables. There should be one VariableMapCallable for each of the pattern element.

    Attributes
    ----------
    variable_maps
    vm_pred_args
    vm_pat_vars
    pred_args2var_map
    pat_var2pred_args
    """
    def __init__(self, name: str, variable_maps: IterableType[VariableMapCallable], **kwargs):
        super(RMFN, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Relation Variable Mapping Node"

        assert isinstance(variable_maps, Iterable) and all(isinstance(vm, VariableMapCallable) for vm in variable_maps)

        self.variable_maps = variable_maps
        # Infer the set of predicate arguments and pattern variables from the variable maps
        self.vm_pred_args = set(vm.pred_arg for vm in self.variable_maps)
        self.vm_pat_vars = set(vm.pat_var for vm in self.variable_maps)
        # Dictionary mapping predicate arguments to their associated variable maps
        self.pred_arg2var_map = {vm.pred_arg: vm for vm in self.variable_maps}
        # Dictionary mapping pattern variables to the list of associated (mapped) predicate arguments
        self.pat_var2pred_args = {pat_var: [vm.pred_arg for vm in self.variable_maps if vm.pat_var == pat_var]
                                  for pat_var in self.vm_pat_vars}

        self.pretty_log['node type'] = "Relation Variable Mapping Node"

    def add_link(self, linkdata: LinkData):
        """
            Check if `linkdata`'s relational variables are compatible with the predicate arguments or pattern variables
            we inferred from the variable mappings during init.
        """
        super(RMFN, self).add_link(linkdata)

        # Check if the relational variables indicated by the linkdata is consistent with the predicate arguments and
        #   pattern variables inferred from the given variable mappings.
        ld_rel_vars = set(linkdata.vn.rel_vars)
        if (linkdata.attr['direction'] == 'inward' and linkdata.to_fn) or \
                (linkdata.attr['direction'] == 'outward' and not linkdata.to_fn):
            assert ld_rel_vars == self.vm_pred_args, \
                "In {}: the linkdata {} to add has incompatible predicate arguments. Expect set of predicate " \
                "arguments {} (inferred from the given variable mappings), but found {} in the linkdata." \
                .format(self.name, linkdata, self.vm_pred_args, ld_rel_vars)
        else:
            assert ld_rel_vars == self.vm_pat_vars, \
                "In {}: the linkdata {} to add has incompatible pattern variables. Expect set of pattern " \
                "variables {} (inferred from the given variable mappings), but found {} in the linkdata." \
                .format(self.name, linkdata, self.vm_pat_vars, ld_rel_vars)

    def inward_compute(self, in_ld, out_ld):
        """Inward computation. Converts predicate relational arguments to pattern relational variables. Applies mappings
        to relational variable's values, if specified.

        For inward direction, we are assuming this is used in condition or condact patterns.

        The computations to be carried out can be summarized in three steps: map/broaden, diagonalize, & permute
        """
        msg = in_ld.read()
        pred_args, pat_vars = list(in_ld.vn.rel_vars), list(out_ld.vn.rel_vars)

        # 1. First, translate predicate arguments to pattern variables.
        #
        #    For each predicate argument, we consults with its variable mapping to obtain the index map tuple. The
        #    tuple should indicate the index mapping pat_var_index -> pred_arg_index. Therefore, we first use
        #    batch_index_select() with pred_arg_index to select those referenced message slices, then use
        #    batch_index_put() with pat_var_index to put those slices at the appropriate places with identity message
        #    slices filled in those unreferenced pattern variable indices.
        for dim, pred_arg in enumerate(pred_args):
            varmap: VariableMapCallable = self.pred_arg2var_map[pred_arg]
            pat_var_ids, pred_arg_ids = varmap()
            tmp_msg = msg.batch_index_select(dim=dim, index=pred_arg_ids)
            msg = tmp_msg.batch_index_put(dim=dim, index=pat_var_ids)

            # Broaden the message dimension so that it matches with the target pattern variable size. This is in case
            #   pat_var_ids does not include the largest possible index, so the last line above would return a message
            #   with a smaller dimension size than desired. 
            if msg.b_shape[dim] < varmap.pat_var.size:
                msg = msg.batch_broaden(dim=dim, length=varmap.pat_var.size)

        # 2. The step above translate each predicate argument to its mapped pattern variable. Now, we need to collapse
        #    the dimensions that are mapped to the same pattern variables, by using batch_diagonal() to select the
        #    diagonal entries.
        #
        #    Since this step changes the dimension ordering, we use a running_vars to keep track of the dimensional
        #    location of each variable (predicate arguments before translation, and pattern variables after).
        running_vars = copy.deepcopy(pred_args)
        for pat_var, mapped_pred_args in self.pat_var2pred_args.items():
            # Before diagonalization, translate the first mapped pred arg in running_vars to its mapped pat var.
            running_vars[running_vars.index(mapped_pred_args[0])] = pat_var

            # Now, if mapped_pred_args has more than 1 elements, then we need to diagonalize
            if len(mapped_pred_args) > 1:
                for next_pred_arg in mapped_pred_args[1:]:      # Exclude the first pred arg in list
                    # Find current and next variable's dimensional location
                    cur_id, next_id = running_vars.index(pat_var), running_vars.index(next_pred_arg)
                    # Diagonalize the message
                    msg = msg.batch_diagonal(dim1=cur_id, dim2=next_id)
                    # The new diagonalized dimension is appended to the end, so we remove the current variable and the
                    #   next variable from the running list, and append the pat var to the end of it
                    running_vars.remove(pat_var)
                    running_vars.remove(next_pred_arg)
                    running_vars.append(pat_var)

        # Sanity Check: we should have produced all the target relational variables, albeit with a different order
        assert set(running_vars) == set(pat_vars)

        # 3. With all predicate argument dimension translated to pattern variable dimensions and all repetitions
        #       diagonalized, we guarantee that all predicate variable appears in running_vars.
        #    The last thing to do is to permute the batch dimensions so that the processed message's dimension match
        #       exactly with out_rel_var_list
        perm_order = [running_vars.index(pat_var) for pat_var in pat_vars]
        msg = msg.batch_permute(target_dims=perm_order)

        # Send message
        out_ld.write(msg)

    def outward_compute(self, in_ld: LinkData, out_ld: LinkData):
        """Outward computation. Converts pattern relational variables to predicate relational arguments. Applies
        mappings to relational variable's values, if specified.

        For outward direction, we are assuming this is used in action or condact patterns.

        The computations to be carried out can be summarized in three steps: un-diagonalize, map/narrow, & permute
        """
        msg = in_ld.read()
        pat_vars, pred_args = list(in_ld.vn.rel_vars), list(out_ld.vn.rel_vars)

        # 1. First, unbind pattern variables. For each of the pattern variables that are mapped to multiple predicate
        #    arguments, we unbind the dimension by using batch_diag_embed() to put the pattern variable dimension onto
        #    the diagonal of the target 2D plane corresponding to two of the mapped predicate arguments.
        #
        #    Since this step changes the dimension ordering, we keep a running list of variables (pattern variables
        #    before translation, and predicate arguments after).
        running_vars = copy.deepcopy(pat_vars)
        for pat_var, mapped_pred_args in self.pat_var2pred_args.items():
            # if mapped_pred_args has more than 1 elements, perform diagonal embedding.
            if len(mapped_pred_args) > 1:
                for next_pred_arg in mapped_pred_args[1:]:       # Skip the first pred arg
                    # Find the directional location of the pattern variable
                    pat_var_dim = running_vars.index(pat_var)
                    # Diagonally embed the pat var dimension into a 2D plane. Keep the pat var dimension as is, and make
                    # the new message's last dimension as the second dimension of the 2D plane
                    msg = msg.batch_diag_embed(diag_dim=pat_var_dim, target_dim1=pat_var_dim, target_dim2=-1)
                    # Append the next predicate argument to the running variable list
                    running_vars.append(next_pred_arg)

            # Translate the pat var in running_vars to the first pred arg in mapped_pred_args
            running_vars[running_vars.index(pat_var)] = mapped_pred_args[0]

        # Sanity check: we should have translated all pattern variables to predicate arguments
        assert set(running_vars) == self.vm_pred_args

        # 2. The step above unbinds pattern variable dimensions and translate pattern variables to associated predicate
        #    arguments. Now, apply mapping to each predicate argument dimension using the index mapping tuple
        #    (pat_var_index, pred_arg_index) obtained from the variable maps
        #
        #    First, use batch_index_select() with pat_var_index to select the referenced slices, then use
        #    batch_index_put() to put the slices to correct locations
        for dim, pred_arg in enumerate(running_vars):
            varmap: VariableMapCallable = self.pred_arg2var_map[pred_arg]
            pat_var_ids, pred_arg_ids = varmap()
            tmp_msg = msg.batch_index_select(dim=dim, index=pat_var_ids)
            msg = tmp_msg.batch_index_put(dim=dim, index=pred_arg_ids)

            # Broaden the message dimension so that it matches with the target predicate argument size. This is in case
            #   pred_arg_ids does not include the largest possible index, so the last line above would return a message
            #   with a smaller dimension size than desired.
            if msg.b_shape[dim] < varmap.pred_arg.size:
                msg = msg.batch_broaden(dim=dim, length=varmap.pred_arg.size)

        # 3. Finally, permute the message dimensions so that it conforms to the correct predicate argument order
        perm_order = [running_vars.index(pred_arg) for pred_arg in pred_args]
        msg = msg.batch_permute(target_dims=perm_order)

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

    An ERFN receives messages containing the universal event values determined by EAFN for each of the pattern variable
    referenced by the represented predicate pattern, as well as another one message that is the original unprocessed
    message sent by the alpha subgraph terminal node (ESFN), whose event values are unprocessed. This node then proceeds
    to compare the universal event values of each pattern variable against the original event values of the associated
    predicate arguments from the unprocessed message. If all such comparisons are equal, then the original message would
    be sent, otherwise the universal event values will be taken as the surrogate particle values in place of the
    original ones, followed by a step of importance re-weighting to yield an augmented message.

    Parameters
    ----------
    ks : KnowledgeServer
        The KnowledgeServer instance for the represented predicate pattern.
    args2var : dict
        The dictionary mapping either a single predicate random argument `Variable` instance, or a `tuple` of predicate
        random argument `Variable` instances, to a single pattern random variable `Variable` instance.
    var2trans : dict
        The dictionary mapping a pattern random variable `Variable` instance to a `Transform` instance, if a
        transformation is declared for this pattern variable. Accordingly, for each pattern element declared in the
        predicate pattern, there must be a corresponding FVN connected to this CMTN.

    Attributes
    ----------
    ks
    args2var
    var2trans


    """
    def __init__(self, name, ks, args2var, var2trans, **kwargs):
        super(ERFN, self).__init__(name, **kwargs)
        assert isinstance(ks, KnowledgeServer)
        assert isinstance(args2var, dict) and \
               all((isinstance(k, Variable) and k.metatype is VariableMetatype.Random) or
                   (isinstance(k, tuple) and all((isinstance(arg, Variable) and arg.metatype is VariableMetatype.Random
                                                  for arg in k)))
                   for k in args2var.keys()) and \
               all((isinstance(v, Variable) and v.metatype is VariableMetatype.Random) for v in args2var.values())
        assert isinstance(var2trans, dict) and \
               all(isinstance(k, Variable) and k.metatype is VariableMetatype.Random for k in var2trans.keys()) and \
               all(isinstance(v, Transform) for v in var2trans.values())

        self.ks = ks
        self.args2var = args2var
        self.var2trans = var2trans

        self._orig_in_ld = None
        self._ran_args = None
        self._ran_vars = ()
        self._var2in_ld = {}

        self.pretty_log["node type"] = "Event Resolution Factor Node"

    def add_link(self, linkdata):
        """ERFN admits only one outgoing linkdata connecting to a DVN with full predicate arguments as random variables.

        ERFN can admit multiple incoming linkdata, but each must be labeled with a boolean special attribute
        ``"original"``. Each of the ``original==False`` linkdata must connect to a DVN that is one-to-one corresponding
        to one of the pattern variable referenced by the represented predicate pattern. On the other hand, only one
        ``original==True`` linkdata can be admitted, and it must be a DVN with full predicate arguments as random
        variables, similar to the outgoing DVN node.
        """
        super(ERFN, self).add_link(linkdata)
        if linkdata.to_fn:
            assert "original" in linkdata.attr.keys() and type(linkdata.attr["original"]) is bool, \
                "At {}: incoming linkdata must have boolean special attribute `original`.".format(self.name)
            if linkdata.attr["original"]:
                assert self._orig_in_ld is None, "At {}: Multiple incoming `original` linkdata.".format(self.name)
                self._orig_in_ld = linkdata
                assert self._ran_args is None or linkdata.vn.ran_vars == self._ran_args, \
                    "At {}: The tuple of random arguments inferred from incoming `original` linkdata and that inferred " \
                    "from the outgoing linkdata do not coincide. The registered tuple of random predicate arguments are " \
                    "{}, and found {} in the new linkdata {}."\
                    .format(self.name, self._ran_args, linkdata.vn.ran_vars, linkdata)
                self._ran_args = linkdata.vn.ran_vars if self._ran_args is None else self._ran_args
            else:
                assert len(linkdata.vn.ran_vars) == 1, \
                    "At {}: Incoming non-`original` linkdata must connect to a DVN with only one random variable that " \
                    "represents one of the predicate pattern's pattern variable. Found linkdata {} connecting to a DVN " \
                    "with {} random variables.".format(self.name, linkdata, len(linkdata.vn.ran_vars))
                assert linkdata.vn.ran_vars[0] in self.args2var.values(), \
                    "At {}: Incoming non-`original` linkdata must connect to a DVN with only one random variable that " \
                    "represents one of the predicate pattern's pattern variable. Found linkdata {} connecting to a DVN " \
                    "with a random variable {}, which cannot be recognized from the value set {} in `args2var` " \
                    "specified during initialization."\
                    .format(self.name, linkdata, linkdata.vn.ran_vars[0], set(self.args2var.values()))
                assert linkdata.vn.ran_vars[0] not in self._ran_vars, \
                    "At {}: Incoming non-`original` linkdata cannot connect to DVNs with the same random variable. " \
                    "Found linkdata {} with duplicate random variable {}."\
                    .format(self.name, linkdata, linkdata.vn.ran_vars[0])
                self._ran_vars += (linkdata.vn.ran_vars[0],)
                self._var2in_ld[linkdata.vn.ran_vars[0]] = linkdata
        else:
            assert len(self.in_linkdata) <= 1, "At {}: At most one outgoing linkdata.".format(self.name)
            assert self._ran_args is None or linkdata.vn.ran_vars == self._ran_args, \
                "At {}: The tuple of random arguments inferred from incoming `original` linkdata and that inferred " \
                "from the outgoing linkdata do not coincide. The registered tuple of random predicate arguments are " \
                "{}, and found {} in the new linkdata {}."\
                .format(self.name, self._ran_args, linkdata.vn.ran_vars, linkdata)
            self._ran_args = linkdata.vn.ran_vars if self._ran_args is None else self._ran_args

    def inward_compute(self, in_ld_list, out_ld_list):
        """The following procedure is carried out:

        1. For each pattern element (args & var pair in ``self.args2var``):

            1. Gather the universal event values of the pattern variable from corresponding incoming linkdata
            2. If a transformation is declared on this pattern element, then apply the FORWARD (inverse of inverse)
               transformation on the universal event values
            3. Save the (transformed) event values as part of the surrogate particles.

        2. Retrieve the message parameter from the "original" message
        3. Use the surrogate particles to create an augmented Particles message, combine with original parameters,
           and perform importance re-weighting. Send the resulting message
        """
        out_ld = out_ld_list[0]
        assert self._orig_in_ld is not None
        assert set(self._var2in_ld.keys()) == set(self.args2var.values())

        same = True
        orig_msg = self._orig_in_ld.read()
        assert isinstance(orig_msg, Message)
        alt_ptcl, alt_dens, index_map = [], [], {}
        # 1. For each pattern element
        for i, (args, pat_var) in enumerate(self.args2var.items()):
            # 1.1
            universal_msg = self._var2in_ld[pat_var].read()
            assert isinstance(universal_msg, Message)
            # 1.2
            if pat_var in self.var2trans.keys():
                universal_msg = universal_msg.event_transform(self.var2trans[pat_var])
            universal_ptcl, universal_dens = universal_msg.particles[0], universal_msg.log_densities[0]
            # 1.3,
            alt_ptcl.append(universal_ptcl)
            alt_dens.append(universal_dens)
            index_map[i] = self._ran_args.index[args] if isinstance(args, Variable) else \
                [self._ran_args.index(arg) for arg in args]
        # 2
        if same:
            out_msg = orig_msg
        else:
            # Get surrogate log prob
            log_prob = self.ks.surrogate_log_prob(orig_msg.parameter, alt_particles=alt_ptcl, index_map=index_map)
            # Generate a identity message with the surrogate particles

        raise NotImplementedError






    def outward_compute(self, in_ld_list, out_ld_list):
        """

        """
        raise NotImplementedError