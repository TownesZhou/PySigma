"""
    All nodes related to a conditional subgraph
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
