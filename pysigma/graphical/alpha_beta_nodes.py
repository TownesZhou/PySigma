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

    Captures the commonality of all alpha subgraph nodes:
    The commonality of all alpha subgraph factor nodes is that they all only admit up to two paris of incoming and
        outgoing link. Additionally, links must declare a special attribute 'direction' with value 'inward' or
        'outward' to indicate whether it is pointing toward the conditional gamma factor node or not.

        Such link check is implemented in add_link() to be inherited by concrete alpha factor node class. Also
            implemented in this method is the registration of labeled pairs of linkdata in self.labeled_ld_pair

        compute() is implemented so that it execute inward_compute() and/or outward_compute() based on the presence of
            linkdata pairs.

        inward_compute() and outward_compute() are now abstract methods that must be implemented by child classes, but
            compute() should not be override.
    """
    def __init__(self, name):
        super(AlphaFactorNode, self).__init__(name)

        # Pairs of incoming and outgoing linkdata labeled with their directionality w.r.t. the alpha structure
        self.labeled_ld_pair = {}

    def add_link(self, linkdata):
        assert isinstance(linkdata, LinkData)
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
        super(AlphaFactorNode, self).compute()
        assert len(self.in_linkdata) == len(self.out_linkdata) and len(self.in_linkdata) > 0

        # Carry out directional computation based on presence of link in self.labeled_ld_pair
        for direction, (in_ld, out_ld) in self.labeled_ld_pair.items():
            if direction == 'inward':
                self.inward_compute(in_ld, out_ld)
            else:
                self.outward_compute(in_ld, out_ld)

    @abstractmethod
    def inward_compute(self, in_ld, out_ld):
        """
            Inward message computation. To be implemented by child class.
        """
        pass

    @abstractmethod
    def outward_compute(self, in_ld, out_ld):
        """
            Outward message computation. To be implemented by child class.
        """
        pass


class RMFN(AlphaFactorNode):
    """
        Relation Variable Mapping Node

        Convert between predicate arguments and pattern variables. Apply relational variable's VariableMap (if declared)
            by selecting and placing entries among the message batch dimensions. This node can thus carry out
            inner-pattern relational variable matching by itself.

        This node is a component of the alpha conditionial subgraph, so admits up to two pairs of incoming and outgoing
            links. Link must declare special attribute 'direction' with value 'inward' or 'outward' to indicate whether
            it is pointing toward the conditional gamma factor node or not.

        For inward direction, inner-pattern relational variable matching is handled by selecting entries on the
            diagonals from the incoming message. For outward direction, this is handled by placing incoming message onto
            the diagonals of a larger message tensor.
    """
    def __init__(self, name, arg2var, var2arg, arg2var_map):
        """
            Necessary data structure:

            :param  arg2var:    dictionary mapping predicate argument Variable instance to pattern variable Variable
                                    instance
            :param  var2arg:    dictionary mapping pattern variable Variable instance to LIST of predicate argument
                                    Variable instance(s)
            :param  arg2var_map:    dictionary mapping predicate argument Variable instance to VariableMap instance
        """
        super(RMFN, self).__init__(name)
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

        # Obtain mapping dictionary and inverse mapping dictionary
        self.arg2var_map_tuple = {arg: var_map.get_map() for arg, var_map in self.arg2var_map.items()}
        self.arg2var_map_inv_tuple = {arg: var_map.get_inverse_map() for arg, var_map in self.arg2var_map.items()}

    def inward_compute(self, in_ld, out_ld):
        """
            Inward computation. Convert predicate relational arguments to pattern relational variables. Apply mappings
                to relational variable's values, if specified.

            For inward direction, we are assuming this is used in condition or condact patterns. Accordingly, the
                inverse mapping should be used to map predicate arguments to pattern variables.

            Will check anyway if domain and image of the inverse map agree with the size range of the predicate argument
                and pattern variable respectively. However to be user friendly this should be checked beforehand by
                compiler.

            Note that domain should be a subset of predicate argument size range, but image should be exactly equal to
                the pattern variable size range

            The computations to be carried out can be summarized in three steps: map/broaden, diagonalize, & permute
        """
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        msg = in_ld.read()
        assert isinstance(msg, Message)
        in_rel_var_list, out_rel_var_list = in_ld.vn.rel_var_list, out_ld.vn.rel_var_list

        # Check that given data structures agree with variable lists of the incident variable node
        assert set(self.arg2var.keys()) == set(in_rel_var_list)
        assert set(self.var2arg.keys()) == set(out_rel_var_list)
        # Check that mapping's domain and image agree with variables' sizes
        #   Note that for inward computation we are using the inverse map
        for arg, var_map_tuple in self.arg2var_map_inv_tuple.items():
            pat_var = self.arg2var[arg]
            _, domain, image = var_map_tuple
            assert domain.issubset(set(range(arg.size)))
            assert image == set(range(pat_var.size))

        # 1. First, translate predicate arguments to pattern variables. This step involves broadening the variable
        #       dimension if predicate argument size is smaller than pattern variable size, or map predicate argument
        #       values to pattern variable values if a VariableMap is specified for the given predicate argument.
        #    For the mapping we should use original forward mapping, because we are selecting "image" to place in
        #       "domain", in the order mandated by domain.
        #    Note that we have guaranteed that forward mapping's domain is equal to pattern variable's size range
        #    A running list of variables is maintained to keep track of variable dimensions
        mapped_var_list = copy.deepcopy(in_rel_var_list)
        for dim, pred_arg in enumerate(mapped_var_list):
            pat_var = self.arg2var[pred_arg]
            # Apply map if VariableMap is specified
            if pred_arg in self.arg2var_map_tuple.keys():
                map_dict, _, _ = self.arg2var_map_tuple[pred_arg]
                indices = torch.tensor(list(map_dict[i] for i in range(pat_var.size)), dtype=torch.long)
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

        assert set(mapped_var_list) == set(out_rel_var_list)

        # 3. With all predicate argument dimension converted to pattern variable dimensions and all repetitions
        #       diagonalized, we guarantee that all predicate variable appears in mapped_var_list.
        #    The last thing to do is to permute the batch dimensions so that the processed message's dimension match
        #       exactly with out_rel_var_list
        perm_order = list(mapped_var_list.index(pt_var) for pt_var in out_rel_var_list)
        msg = msg.batch_permute(perm_order)
        assert msg.b_shape == torch.Size([v.size for v in out_rel_var_list])

        # Send message
        out_ld.write(msg)

    def outward_compute(self, in_ld, out_ld):
        """
            Outward computation. Convert pattern relational variables to predicate relational arguments. Apply mappings
                to relational variable's values, if specified.

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
        in_rel_var_list, out_rel_var_list = in_ld.vn.rel_var_list, out_ld.vn.rel_var_list

        # Check that given data structures agree with variable lists of the incident variable node
        assert set(self.var2arg.keys()) == set(in_rel_var_list)
        assert set(self.arg2var.keys()) == set(out_rel_var_list)
        # Check that mapping's domain and image agree with variables' sizes
        #   For outward computation we are using the forward map
        for arg, var_map_tuple in self.arg2var_map_tuple.items():
            pt_var = self.arg2var[arg]
            _, domain, image = var_map_tuple
            assert image.issubset(set(range(arg.size)))
            assert domain == set(range(pt_var.size))

        # 1. First, translate pattern variables to predicate arguments. This step involves unbinding the predicate
        #       variables that are referenced by multiple predicate arguments.
        #    Computationally, this is achieved by un-diagonalize, or embed entries along that predicate variables'
        #       dimension into a 2D plane (higher dimensional space if there are more than 2 such predicate arguments).
        #    A running list of variables is maintained to keep track of variable dimensions
        mapped_var_list = copy.deepcopy(in_rel_var_list)
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

        assert set(mapped_var_list) == set(out_rel_var_list)

        # 2. The step above guarantees a symbolic one-to-one mapping between message dimensions and predicate arguments'
        #       variable dimensions. We now need to narrow the variable dimension if the predicate argument's size is
        #       smaller than the associated pattern variable's size, or to map pattern variable's value to predicate
        #       argument's value if a VariableMap is specified.
        for dim, pred_arg in enumerate(mapped_var_list):
            pat_var = self.arg2var[pred_arg]
            # Apply map if VariableMap is specified
            if pred_arg in self.arg2var_map_tuple.keys():
                map_dict, _, _ = self.arg2var_map_tuple[pred_arg]
                indices = torch.tensor(list(map_dict[i] for i in range(pat_var.size)), dtype=torch.long)
                msg = msg.batch_index_put(dim, indices)
            # Broaden the variable dimension size if currently it is smaller than the pattern variable's size
            if msg.b_shape[dim] > pred_arg.size:
                msg = msg.batch_narrow(dim, pred_arg.size)

        # 3. Finally, with all pattern variables converted to predicate arguments and the values are now with respect to
        #       the predicate arguments, the last thing to do is to permute the batch dimensions so that the processed
        #       message's dimensions match exactly with out-_rel_var_list
        perm_order = list(mapped_var_list.index(pred_arg) for pred_arg in out_rel_var_list)
        msg = msg.batch_permute(perm_order)
        assert msg.b_shape == torch.Size([v.size for v in out_rel_var_list])

        # Send message
        out_ld.write(msg)


class ESFN(AlphaFactorNode):
    """
        Expansion / Summarization Node

        This node is a component of the alpha conditional subgraph, so admits up to two pairs of incoming and outgoing
            links. Link must declare special attribute 'direction' with value 'inward' or 'outward' to indicate whether
            it is pointing toward the conditional gamma factor node or not.

        For inward direction, it expands and permutes the incoming message's relational variable dimensions to match the
            full relational variable dimensions determined by the conditional. For outward direction, it summarizes
            over irrelevant relational variables and permute the dimensions to match the relational variable dimensions
            of this pattern.

        Note that the expanded dimensions will be of size 1, so that the expanded tensor is broadcastable along this
            dimension.

        The summarization step can be thought of as a search or optimization problem, where one finds a single
            distribution instance that best "summarizes" the behaviors of an entire space of distribution instances,
            where the dimensions of the space is defined and spanned by the irrelevant relational variables. Depending
            on the user-specified summarization criteria, different semantics can be interpreted for this step.

        A sum_op should be specified during initialization to specify special summarization semantics, such as Max
            Product semantics or searching. If not specified, will default to Sum-Product alike summarization. Please
            refer to Message class documentation for more information.
    """
    def __init__(self, name, sum_op=None):
        """
            Necessary data structure:

            :param sum_op:      None or a Summarization instance. Default is None.
        """
        super(ESFN, self).__init__(name)
        self.pretty_log["node type"] = "Expansion Summarization Factor Node"

        assert sum_op is None or isinstance(sum_op, Summarization)
        if sum_op is not None:
            raise NotImplementedError("Summarization operation using Summarization instance is not yet implemented.")

        self.sum_op = sum_op

    def inward_compute(self, in_ld, out_ld):
        """
            Expansion operation. Expand and permutes the incoming message's relational variable dimensions to match the
                outgoing relational relational variable dimensions.
        """
        msg = in_ld.read()
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        assert isinstance(msg, Message)
        in_rel_var_list, out_rel_var_list = in_ld.vn.rel_var_list, out_ld.vn.rel_var_list
        # Check that the set of relational variables of incoming message is a subset of that of outgoing message
        assert set(in_rel_var_list).issubset(set(out_rel_var_list))

        # Keep a running list of variables
        mapped_var_list = copy.deepcopy(in_rel_var_list)

        # For every relational variable in out_rel_var_list that is not in in_rel_var_list, unsqueeze a dimension of the
        #   message as the last dimension.
        for pt_var in out_rel_var_list:
            if pt_var not in in_rel_var_list:
                # Unsqueeze message
                msg = msg.batch_unsqueeze(dim=-1)
                # Append the pt_var to the end of running var list
                mapped_var_list.append(pt_var)

        assert set(mapped_var_list) == set(out_rel_var_list)

        # Permute message dimension so that it matches the order given by out_rel_var_list
        perm_order = list(mapped_var_list.index(v) for v in out_rel_var_list)
        msg = msg.batch_permute(perm_order)

        # Expand to full relational variable dimension shape
        target_shape = torch.Size([v.size for v in out_rel_var_list])
        msg = msg.batch_expand(target_shape)

        # Send message
        out_ld.write(msg)

    def outward_compute(self, in_ld, out_ld):
        """
            Summarization operation. Summarize over incoming message's relational variable dimensions that are not
                referenced by outgoing message's relational variables.

            The summarization semantic is defined by the sum_op specified during initialization. If sum_op is None,
                uses default summarization semantic defined at the Message level.
        """
        msg = in_ld.read()
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        assert isinstance(msg, Message)
        in_rel_var_list, out_rel_var_list = in_ld.vn.rel_var_list, out_ld.vn.rel_var_list
        # Check that the set of relational variables of outgoing message is a subset of that of incoming message
        assert set(out_rel_var_list).issubset(set(in_rel_var_list))

        # Keep a running list of variables
        mapped_var_list = copy.deepcopy(in_rel_var_list)

        # Summarize using custom sum_op
        if self.sum_op is not None:
            # Flatten both the group of dimensions to be summarized over and the group of other dimensions. Put the
            #   former as the last dimension and the latter as the first batch dimension
            sum_dims = list(dim for dim, v in enumerate(in_rel_var_list) if v not in out_rel_var_list)
            other_dims = list(dim for dim, v in enumerate(in_rel_var_list) if v in out_rel_var_list)
            if len(sum_dims) > 0:
                # First flatten other_dims, then sum_dims, so that flattened sum_dims will be the last dim
                msg = msg.batch_flatten(other_dims)
                msg = msg.batch_flatten(sum_dims)
                # Process using the sum_op
                msg = self.sum_op.process(msg)
                # Reshape
                msg = msg.batch_reshape(other_dims)

        # Otherwise if sum_op is None, carry out default summarization
        else:
            # Iterate over all relational variables not referenced by out_rel_var_list
            for pt_var in in_rel_var_list:
                if pt_var not in out_rel_var_list:
                    dim = mapped_var_list.index(pt_var)
                    # Summarize over the message batch dimension
                    msg = msg.batch_summarize(dim)
                    # Remove the variable from the running list
                    mapped_var_list.remove(pt_var)

        assert set(mapped_var_list) == set(out_rel_var_list)

        # Permute message dimension so that it matches the order given by out_rel_var_list
        perm_order = list(mapped_var_list.index(v) for v in out_rel_var_list)
        msg = msg.batch_permute(perm_order)

        # Send message
        out_ld.write(msg)


class BetaFactorNode(FactorNode, ABC):
    """
            Abstract base class for nodes belonging to the beta subgraph of a conditional.

            Captures the commonality of Beta factor nodes, including link connectivity and inward & outward message
                propagation pattern:
                - Groups links in terms of whether the messages moves inward to the Gamma Factor node or not.
                - During compute, perform inward and outward computation separately in turn by calling inward_compute()
                    and outward_compute()

            Different from Alpha nodes, there's no restriction on the number of incoming or outgoing linkdata, as long as
                they can be identified in terms of their messaging directionality.

            Need to specify "direction" attribute in linkdata
        """

    def __init__(self, name):
        super(BetaFactorNode, self).__init__(name)

        # Pairs of incoming and outgoing linkdata list with their messaging direction w.r.t. the beta structure
        self.labeled_ld_list_pair = {
            'inward': ([], []),
            'outward': ([], [])
        }

    def add_link(self, linkdata):
        assert isinstance(linkdata, LinkData)
        assert 'direction' in linkdata.attr and linkdata.attr['direction'] in ['inward', 'outward']

        if linkdata.to_fn:
            self.labeled_ld_list_pair[linkdata.attr['direction']][0].append(linkdata)
        else:
            self.labeled_ld_list_pair[linkdata.attr['direction']][1].append(linkdata)

        super(BetaFactorNode, self).add_link(linkdata)

    def compute(self):
        super(BetaFactorNode, self).compute()

        for direction, (in_ld_list, out_ld_list) in self.labeled_ld_list_pair.items():
            if len(in_ld_list) > 0 and len(out_ld_list) > 0:
                if direction == 'inward':
                    self.inward_compute(in_ld_list, out_ld_list)
                else:
                    self.outward_compute(in_ld_list, out_ld_list)

    @abstractmethod
    def inward_compute(self, in_ld_list, out_ld_list):
        """
            Inward message computation. To be implemented by child class.
        """
        pass

    @abstractmethod
    def outward_compute(self, in_ld_list, out_ld_list):
        """
            Outward message computation. To be implemented by child class.
        """
        pass


class ESCFN(BetaFactorNode):
    """
        Event Split / Combination Node

        add link:
            - check that for inward direction, there's only one incoming link; for outward direction, only one outgoing
                link.

        Inward direction compute:
            1. Flatten batch dimensions into a single dimension
            2. If there are multiple referenced random variables, ensure that incoming message contains Particles,
                otherwise raise an alert.
            3. Translate event particles from PyTorch format to Cognitive format. If multiple pattern random variables
                are referenced, the events will/should be split in accordance to the size of each pattern r.v.
            4. If 'reference' is True, then will cache the incoming message after each inward compute.
            5. If there is only one referenced pattern random variable, send translated message as is to the outgoing
                link. Otherwise, send split event messages to each outgoing link with corresponding pattern r.v.
                respectively.

        Outward direction compute:
            1. If there is inward propagation direction, check that the cache is not None. Check that incoming messages
                holds the same particles as the cached message. Otherwise, raise an alert.
            2. If there are multiple incoming links, check that messages from all incoming links have Particles. In
                this case, if there is cached message, combine incoming messages' event particles with alignment to
                the cached message's event particles. Otherwise, combine incoming messages' event particles randomly /
                without alignment.
            3. Reshape the single batch dimensions into full conditional batch dimensions.
    """
    def __init__(self, name, reference=True):
        super(ESCFN, self).__init__(name)

        assert isinstance(reference, bool)
        # Whether to cache referenced event particle message
        self.reference = reference
        self.cache = None

    def add_link(self, linkdata):
        super(ESCFN, self).add_link(linkdata)
        if linkdata.attr['direction'] == 'inward' and linkdata.to_fn:
            assert len(self.labeled_ld_list_pair['inward'][0]) <= 1
        elif linkdata.attr['direction'] == 'outward' and not linkdata.to_fn:
            assert len(self.labeled_ld_list_pair['outward'][1]) <= 1

    def inward_compute(self, in_ld_list, out_ld_list):
        assert len(in_ld_list) == 1
        in_ld = in_ld_list[0]
        ran_var_list = in_ld.vn.ran_var_list
        # Check that the number of outgoing links equal the number of random variables from the incoming link, and that
        # there's one-to-one correspondence
        assert len(out_ld_list) == len(ran_var_list) and \
            set(out_ld.vn.ran_var_list[0] for out_ld in out_ld_list) == set(ran_var_list)

        msg = in_ld_list[0].read()
        assert isinstance(msg, Message)

        # 1. Flatten batch dimension
        msg = msg.batch_flatten()

        # 2. Ensure incoming message contain particles if multiple referenced r.v.
        if len(ran_var_list) > 1:
            assert MessageType.Particles in msg.type, \
                "In {}: Pattern has multiple referenced random variables: {}, however the incoming message {} does " \
                "not contain particles. "

        # 3. Translate event format
        pass


class RTFN(AlphaFactorNode):
    """
        Random Variable Transformation Node

        Carry out three core functionality:

        1. Manipulate batch dimensions:
            - For inward direction: flatten batch dimensions into a single dimension
            - for outward direction: reshape the flattened batch dimension into full batch dimensions of the conditional

        2. Apply transformation on events:
            - For inward direction: apply pre-specified transformation on event.
            - For outward direction: apply the INVERSE of the pre-specified transformation on events.

        3. Check if event values meets the pre-specified constraints.
            - This step will be automatically skipped if it's outward direction and/or the message does not contain
                event particles.
            - If constraints not satisfied and replaceable == False, raise an alert.
            - If constraints not satisfied and replaceable == True, check if the incoming message type is Both.
                - If yes, then reduce the message into Parameter message and send it to outgoing link.
                - Otherwise, raise an alert.

    """
    def __init__(self, name, trans, constraints, replaceable=True):
        """
            :param trans:       torch.distributions.transforms.Transform. The transformation functor
            :param constraints  a set of torch.distributions.constraints.Constraint. The value constraints of the target
                                    conditional's pattern random variable.
        """
        super(RTFN, self).__init__(name)
        self.pretty_log["node type"] = "Random Variable Transformation Node"

        assert isinstance(trans, Transform)
        assert isinstance(constraints, set) and all(isinstance(c, Constraint) for c in constraints)
        assert isinstance(replaceable, bool)

        self.trans = trans
        self.constraints = constraints
        self.replaceable = replaceable

    def inward_compute(self, in_ld, out_ld):
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        msg = in_ld.read()
        assert isinstance(msg, Message)

        # 1. Flatten batch dimensions into a single dimension
        msg = msg.batch_flatten()

        # 2. Apply forward transformation.
        msg = msg.event_transform(self.trans)

        # 3. Check value constraints only if message involves particles
        if MessageType.Particles in msg.type:
            valid = True
            for constraint in self.constraints:
                valid *= constraint.check(msg.particles).all()

            # Raise an alert if not valid and particles not replaceable.
            assert valid or self.replaceable, \
                "At {}: It has been specified that particles in incoming messages are not replaceable, but encountered " \
                "a message where the particle values do not meet pre-specified constraints: {}"\
                .format(self.name, self.constraints)

            # Check message type if not valid but replaceable is True
            if not valid and self.replaceable:
                assert MessageType.Parameter in msg.type, \
                    "At {}: Message must contain parameter if it's particles are to be replaced when its own particles " \
                    "do not meet the constraints. Instead, found one incoming message of type {} whose particles do " \
                    "not meet the constraints.".format(self.name, msg.type)
                # Reduce the message into only parameters
                msg = msg.reduce_type(MessageType.Parameter)

        # Send message
        out_ld.write(msg)

    def outward_compute(self, in_ld, out_ld):
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        out_rel_var_list = out_ld.vn.rel_var_list
        msg = in_ld.read()
        assert isinstance(msg, Message)

        # 1. Reshape batch dimension into full conditional relational pattern variable dimensions
        out_dims = list(v.size for v in out_rel_var_list)
        assert len(msg.b_shape) == 1
        msg = msg.batch_reshape(out_dims)

        # 2. Apply inverse transformation.
        msg = msg.transform(self.trans.inv)

        # Send message
        out_ld.write(msg)