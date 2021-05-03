"""
    All nodes related to the Conditional Beta subgraph
"""
from __future__ import annotations      # For postponed evaluation of typing annotations
from typing import Optional, List, Tuple, Dict
from typing import Iterable as IterableType
import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
import torch

from ..utils import KnowledgeServer
from ..pattern_structures.event_transform import EventTransform
from ..defs import Variable, Message, MessageType
from .basic_nodes import LinkData, FactorNode, NodeConfigurationError


class BetaFactorNode(FactorNode, ABC):
    """
        Abstract base class for any factor node belonging to the Conditional Beta subgraph.

        Captures the commonalities of all Beta subgraph factor nodes:

        * **Topology**: a Beta Factor Node accepts one and up to two groups of linkdata, with one group propagating
          messages inward to the Gamma Factor Node, and another group propagating messages outward to the predicate
          subgraph. In each group, there should be at least one incoming and outgoing linkdata, but unlike an Alpha
          Factor Node, there is no restriction on the maximum number of linkdata in each group.
        * **Admissible variable nodes**: a Beta factor node works on the random variables, and does not touch the batch
          dimensions of the messages. Thus, incident variable nodes should have the same tuple of relational variables.
        * **Compute pattern**: a Beta factor node computes outgoing messages for each group of linkdata individually.
          In other words, for instance, the outgoing message to an inward outgoing link is solely dependent on the
          message received from the inward incoming links. Accordingly, the `compute()` method is divided into an
          `inward_compute()` and an `outward_compute()` subroutine.
        * **Quiescence state**: a Beta factor node as a whole reaches quiescence if and only if **all** incoming
          linkdata do not contain new message. However, for the two subroutines `inward_compute()` and
          `outward_compute()`, either one should not be called once its incoming linkdata do not contain new message.

        Attributes
        ----------
        labeled_ld_group : dict
            Dictionary of groups of incoming and outgoing linkdata labeled by their message propagation direction.
            Structure:
                {
                    [ linkdata.attr['direction] ] : {
                            [ linkdata.to_fn ] : linkdata
                        }
                }
        rel_vars: tuple of Variable
            The relational variables of all incident variable node.
    """
    def __init__(self, name: str, **kwargs):
        super(BetaFactorNode, self).__init__(name, **kwargs)

        # Groups of incoming and outgoing linkdata labeled with their directionality w.r.t. the Beta structure
        # A dictionary mapping direction to dictionaries mapping incoming/outgoing indicator to list of Linkdata
        #   instances.
        # Use a default dict
        self.labeled_ld_group: Dict[str, Dict[bool, List[LinkData]]] = defaultdict(
            lambda: defaultdict(
                lambda: []
            )
        )
        # Cache the relational variables when add_link() is called to ensure that incident variable nodes have the
        #   same tuple of relational variables.
        self.rel_vars: Optional[Tuple[Variable]] = None

    def add_link(self, linkdata: LinkData):
        """
            A Beta factor node accepts at least one but no more than two groups of incoming and outgoing linkdata. The
            incident variable nodes should have the same tuple of relational variables. Furthermore, a ``"direction"``
            key-ed attribute should be included in the linkdata's optional attribute map with value ``"inward"`` or
            ``"outward"`` to indicate its message propagation direction. The two groups of linkdata should have
            different message propagation directions.

            Parameters
            ----------
            linkdata : LinkData
                The linkdata to be registered. Must specify a ``"direction"`` attribute in its optional attribute map
                ``linkdata.attr``.
        """
        assert isinstance(linkdata, LinkData)
        assert 'direction' in linkdata.attr.keys() and linkdata.attr['direction'] in ['inward', 'outward'], \
            "In {}: For a Beta Factor Node, the linkdata '{}' should specify a special attribute named 'direction', " \
            "with value either 'inward' or 'outward'.".format(self.name, linkdata)

        # Check relational variables
        # Cache the first linkdata's relational variables every registered
        if self.rel_vars is None:
            self.rel_vars = linkdata.vn.rel_vars
        else:
            assert self.rel_vars == linkdata.vn.rel_vars, \
                "In {}: linkdata '{}' has conflicting relational variables. This Beta Factor Node infers relational " \
                "variables {} from other previously registered linkdata, but found different relational variables {} " \
                "in this linkdata." \
                .format(self.name, linkdata, self.rel_vars, linkdata.vn.rel_vars)

        super(BetaFactorNode, self).add_link(linkdata)

        # Register the linkdata in the paired group based on its direction.
        # Because we are using a default dict, we can directly register
        self.labeled_ld_group[linkdata.attr['direction']][linkdata.to_fn].append(linkdata)

    def precompute_check(self):
        """
            Basic computable requirement for a Beta factor node:

                - It has at least one directional group of linkdata, and in each group, there should be at least one
                  incoming and one outgoing linkdata.

            This method shall be extended by a Beta Factor Node subclass if it has more specific requirements.
        """
        if len(self.labeled_ld_group) == 0:
            raise NodeConfigurationError(
                "Wrong configuration for node {}: no registered linkdata.".format(self.name)
            )
        for direction, ld_group in self.labeled_ld_group.items():
            if len(ld_group[True]) == 0:    # No incoming linkdata
                raise NodeConfigurationError(
                    "Wrong configuration for node {}: incomplete group of linkdata. For the group of linkdata with "
                    "message propagation direction '{}', there is no incoming linkdata."
                    .format(self.name, direction)
                )
            if len(ld_group[False]) == 0:   # No outgoing linkdata
                raise NodeConfigurationError(
                    "Wrong configuration for node {}: incomplete group of linkdata. For the group of linkdata with "
                    "message propagation direction '{}', there is no outgoing linkdata."
                    .format(self.name, direction)
                )

    @FactorNode.compute_control
    def compute(self):
        """
            Calls `inward_compute()` and `outward_compute()` individually if any of their corresponding incoming
            linkdata contains new message.
        """
        for direction, ld_group in self.labeled_ld_group.items():
            if any(ld.new for ld in ld_group[True]):
                if direction == 'inward':
                    self.inward_compute(ld_group[True], ld_group[False])
                else:
                    self.outward_compute(ld_group[True], ld_group[False])

    @abstractmethod
    def inward_compute(self, in_lds: List[LinkData], out_lds: List[LinkData]):
        """
            Inward message computation. To be implemented by child class.
        """
        raise NotImplementedError

    @abstractmethod
    def outward_compute(self, in_lds: List[LinkData], out_lds: List[LinkData]):
        """
            Outward message computation. To be implemented by child class.
        """
        raise NotImplementedError


class PDFN(BetaFactorNode):
    """
        Particle Dispatch Factor Node

        This node is one of the two Beta factor nodes that is responsible for dispatching univariate marginal messages
        to each of the connecting outgoing variable nodes in the inward message propagation direction, as well as
        aggregating multiple marginal messages into a joint message in the outward direction. Specifically, it carries
        out the following steps of message manipulation:

        Inward direction:

            1. **Concatenation**:
               For each of the pattern element, If it involves multiple predicate arguments, than concatenate the event
               dimensions in the incoming message corresponding to these predicate arguments into a single event
               dimension.
               Input: one multivariate message. Output: one multivariate message.
            2. **Marginalization**
               For each of the pattern element, marginalize out every other event dimensions of the multivariate message
               from last step that does not correspond to the pattern variable in that pattern element.
               Note that during this step, any predicate argument that is not referenced in any given pattern element
               will be marginalized out in all of the output univariate messages. Its particles, however, will be cached
               and reserved for outward computation if specified.
               Input: one multivariate message. Output: multiple univariate message.
            3. **Transformation**
               For each of the pattern element and its associated univariate message from last step, apply the forward
               transformation on the particles if such a transformation is declared in the pattern element.
               Input: multiple univariate messages. Output: multiple univariate messages.

        Outward:

            1. **Transformation**
               For each of the pattern element and its associated marginal message, apply the inverse (backward)
               transformation on the particles if such a transformation is declared in the pattern element.
               Input: multiple univariate messages. Output: multiple univariate messages.
            2. **Cross product**
               Take the matrix-vector cross product of the marginal messages from the last step one by one to obtain
               a multivariate joint message.
               Note that if there is any predicate argument that is not referenced in any given pattern element, there
               is no way to retrieve its particles from the incoming messages because none of the incoming messages'
               event dimension correspond to it. Therefore, in this case it is imperative to cache its particles during
               the inward computation and retrieve them herein.
               Input: multiple univariate messages. Output one multivariate message.
            3. **De-concatenation**
               For any event dimension in the message from last step, if it corresponds to a concatenation of multiple
               predicate arguments, then de-concatenate this event dimension into multiple smaller event dimensions,
               with each one corresponding to one of the original predicate argument.
               Note that in order to carry out the operation, it is necessary to know the original sample size of each
               original predicate argument. This information can be inferred from the particles cache which is also used
               in the step above.

        In general, the particles cache must be used whenever:

            1. Both the inward and outward computation needs to be carried out
            2. Some of the predicate argument is not referenced in any given pattern element
            3. Some pattern element involves a concatenation of predicate arguments.

        Note that it is possible to have multiple inward-side linkdata that connects to the same Gamma Variable Node if
        there are multiple pattern elements that associate the same pattern variable (i.e. inner-pattern binding).
        In general, there should be a one-to-one mapping between an inward outgoing linkdata and an EventTransform
        instance that represents a pattern element, and similarly a one-to-one mapping between an outward incoming
        linkdata and an EventTransform instance.

        Parameters
        ----------
        event_transforms : Iterable of EventTransform
            The random variable pattern elements represented by EventTransform instances. Each pattern element
            associates a predicate argument, or a list of predicate arguments to one pattern variable, along with an
            optional event transformation.
        to_cache : bool
            Whether to cache the particles (value & densities) of the outward-incoming message during each cognitive
            cycle. This should be turned on when there are both inward and outward groups of linkdata, and some
            predicate argument is missing in the pattern elements, and/or some predicate arguments are concatenated
            in some pattern element. Defaults to True.
    """
    def __init__(self, name: str, event_transforms: IterableType[EventTransform], to_cache=True, **kwargs):
        super(PDFN, self).__init__(name, **kwargs)
        self.pretty_log['node type'] = "Particle Dispatch Factor Node"

        # Validate event_transforms
        assert isinstance(event_transforms, Iterable) and all(isinstance(et, EventTransform) for et in event_transforms)
        assert all(et.finalized for et in event_transforms)     # Make sure all pattern element is finalized

        # Internal mappings inferred from given arguments
        self.to_cache = to_cache
        self.event_transforms = tuple(event_transforms)
        self.et_pred_args = set(et.pred_arg for et in self.event_transforms)
        self.et_pat_vars = set(et.pat_var for et in self.event_transforms)
        self.pat_var2event_transforms = defaultdict(lambda: [])
        for et in self.event_transforms:
            self.pat_var2event_transforms[et.pat_var].append(et)
        self.et_pred_args_flattened = set()
        for pred_arg in self.et_pred_args:
            if isinstance(pred_arg, tuple):
                self.et_pred_args_flattened = self.et_pred_args_flattened.union(set(pred_arg))
            else:
                self.et_pred_args_flattened.add(pred_arg)

        # Other mappings that can be inferred by registered linkdata in add_link()
        # First key EventTransform, second key boolean (incoming/outgoing), and value linkdata
        self.event_transform2ld: Dict[EventTransform, Dict[bool, LinkData]] = defaultdict(lambda: {})
        self.ld_pred_args: Optional[Tuple[Variable]] = None    # Will be inferred from registered linkdata

        # Cache for compute
        self.ptcl_cache: Optional[List[torch.Tensor]] = None  # Particle value cache
        self.dens_cache: Optional[List[torch.Tensor]] = None  # Particle density cache

    def add_link(self, linkdata: LinkData):
        """
            PDFN has following restrictions for linkdata apart from those defined by BetaFactorNode:

                1. There can be at most one inward incoming linkdata and one outward outgoing linkdata. Both of them
                   must connect to the same variable node.
                2. The set of predicate arguments inferred from the inward incoming linkdata / outward outgoing linkdata
                   must be a super set of the set of predicate arguments inferred from the pattern elements given during
                   init.
                3. Every inward outgoing linkdata and outward incoming linkdata must connect to a univariate variable
                   node that is already declared by the pattern elements.
                4. Both the inward outgoing linkdata and the outward incoming linkdata must have an one-to-one
                   correspondence to the pattern elements given during init.
        """
        super(PDFN, self).add_link(linkdata)

        # Check requirement 1 and requirement 2
        if (linkdata.attr['direction'] == 'inward' and linkdata.to_fn) or \
           (linkdata.attr['direction'] == 'outward' and not linkdata.to_fn):
            # Check requirement 1
            assert len(self.labeled_ld_group[linkdata.attr['direction']][linkdata.to_fn]) == 1, \
                "In {}: PDFN can accept at most one inward incoming linkdata and one outward outgoing linkdata." \
                .format(self.name)
            # Check requirement 2
            if self.ld_pred_args is None:
                self.ld_pred_args = linkdata.vn.ran_vars
                assert set(self.ld_pred_args).issuperset(self.et_pred_args_flattened), \
                    "In {}: The predicate arguments {} inferred from linkdata '{}' is not a super set of those " \
                    "predicate arguments {} inferred from the pattern elements given during initialization." \
                    .format(self.name, self.ld_pred_args, linkdata, tuple(self.et_pred_args_flattened))
            else:
                assert self.ld_pred_args == linkdata.vn.ran_vars, \
                    "In {}: attempts to register an outward-side linkdata with conflicting predicate arguments. " \
                    "Inferred predicate arguments {} from other linkdata, but found {} in linkdata '{}'." \
                    .format(self.name, self.ld_pred_args, linkdata.vn.ran_vars, linkdata)
        else:
            # Check requirement 3
            assert len(linkdata.vn.ran_vars) == 1, \
                "In {}: Every inward outgoing or outward incoming linkdata must connect to a univariate variable " \
                "node. Found linkdata '{}' connecting to a variable node with random variables {}." \
                .format(self.name, linkdata, linkdata.vn.ran_vars)
            pat_var = linkdata.vn.ran_vars[0]
            assert pat_var in self.et_pat_vars, \
                "In {}: The variable node that any inward outgoing or outward incoming linkdata connects to must have "\
                "the random variable that is declared as one of the pattern variables by the pattern elements. " \
                "Expect one of {}, but found {} in linkdata '{}'." \
                .format(self.name, self.et_pat_vars, linkdata.vn.ran_vars[0], linkdata)
            # Check requirement 4
            # Match a pattern element for this linkdata
            found = False
            for et in self.pat_var2event_transforms[pat_var]:
                if et not in self.event_transform2ld.keys() or linkdata.to_fn not in self.event_transform2ld[et].keys():
                    self.event_transform2ld[et][linkdata.to_fn] = linkdata
                    found = True
                    break
            assert found, \
                "In {}: Cannot associate a pattern element to the inward-side linkdata '{}'." \
                .format(self.name, linkdata)

    def precompute_check(self):
        """
            Additional checks for PDFN:

                1. Check that all pattern elements have been matched with at least one linkdata
                2. Check that if both inward linkdata group and outward linkdata group are registered, then there are
                   exactly one outward incoming or inward outgoing linkdata matched for each pattern element.
        """
        super(PDFN, self).precompute_check()

        # Check requirement 1
        assert set(self.event_transform2ld.keys()) == set(self.event_transforms), \
            "In {}: Not all pattern elements have been matched with a inward-side linkdata.".format(self.name)
        # Check requirement 2
        if len(self.labeled_ld_group) == 2:
            assert all(len(ld_pair) == 2 for ld_pair in self.event_transform2ld.values()), \
                "In {}: All pattern elements must be matched with a pair of inward-outgoing and outward-incoming " \
                "linkdata if both inward and outward directional linkdata are registered.".format(self.name)

    def reset_state(self):
        """
            Clear the cache before beginning of each cognitive cycle
        """
        super(PDFN, self).reset_state()
        self.ptcl_cache = None
        self.dens_cache = None

    def inward_compute(self, in_lds: List[LinkData], out_lds: List[LinkData]):
        """
            Inward computation steps:

                1. Concatenation
                2. Marginalization
                3. Transformation

            Cache incoming message particles if specified.
        """
        msg = in_lds[0].read()
        running_vars = list(copy.deepcopy(self.ld_pred_args))

        # Cache particle values and log densities if required. Their order should be in line with self.ld_pred_args
        if self.to_cache:
            self.ptcl_cache = tuple(msg.particles)
            self.dens_cache = tuple(msg.log_densities)

        # Step 1: Iterate through all pattern elements and concatenate predicate arguments
        for pattern in self.event_transforms:
            if isinstance(pattern.pred_arg, tuple):
                # Concatenate these event dimensions and place as the last dimension
                cat_event_dims = list(running_vars.index(arg) for arg in pattern.pred_arg)
                msg = msg.event_concatenate(cat_event_dims)
                # Update running_vars: remove the individual args, and replace them with their tuple inserted at the
                #   back
                for arg in pattern.pred_arg:
                    running_vars.remove(arg)
                running_vars.append(pattern.pred_arg)

        # Step 2: Iterate through all pattern elements, marginalize message
        event_transform2marg_msg = {}
        for pattern in self.event_transforms:
            event_dim = running_vars.index(pattern.pred_arg)
            marg_msg = msg.event_marginalize(event_dim)
            event_transform2marg_msg[pattern] = marg_msg

        # Step 3: Iterate through all pattern elements, and apply forward transformation if specified
        for pattern in self.event_transforms:
            if pattern.has_trans:
                marg_msg = event_transform2marg_msg[pattern]
                marg_msg = marg_msg.event_transform(pattern.forward_trans)
                event_transform2marg_msg[pattern] = marg_msg

        # Final step: send marginal message to each outgoing linkdata
        for pattern, ld_group in self.event_transform2ld.items():
            out_ld = ld_group[False]    # Outgoing linkdata
            marg_msg = event_transform2marg_msg[pattern]
            out_ld.write(marg_msg)

    def outward_compute(self, in_lds: List[LinkData], out_lds: List[LinkData]):
        """
            Outward computation steps:

                1. Transformation
                2. Cross product
                3. De-concatenation

            Raise ValueError if particles cache is needed but not available.
        """
        in_ld2msg: Dict[LinkData, Message] = {in_ld: in_ld.read() for in_ld in in_lds}

        # Step 1: iterate through all pattern elements and apply backward transformation if specified
        for pattern, ld_group in self.event_transform2ld.items():
            if pattern.has_trans:
                in_ld = ld_group[True]  # Incoming linkdata
                msg = in_ld2msg[in_ld]
                msg = msg.event_transform(pattern.backward_trans)
                in_ld2msg[in_ld] = msg

        # Step 2: Take cross product of all marginal message
        # 2.1: First take the cross product of those that are bound with a pattern variable
        running_vars = []
        marg_msg_list = []
        for pattern, ld_group in self.event_transform2ld.items():
            in_ld = ld_group[True]  # incoming linkdata
            marg_msg_list.append(in_ld2msg[in_ld])
            running_vars.append(pattern.pred_arg)
        joint_msg = Message.cross_product(marg_msg_list)
        # 2.2: Check if any predicate arguments in the outward-side linkdata is not referenced in any pattern element.
        if set(self.ld_pred_args) != self.et_pred_args_flattened:
            rest_args = set(self.ld_pred_args) - self.et_pred_args_flattened
            # Now need to access cache. If cache unavailable, raise value error
            if self.ptcl_cache is None or self.dens_cache is None:
                raise ValueError(
                    "In {}: Outward computation needs to access particle cache to take the cross product of marginal "
                    "particles, but the cache is None."
                    .format(self.name)
                )
            # Append these particles by taking the cross product of the previous joint message with new identity message
            for arg in rest_args:
                event_dim = self.ld_pred_args.index(arg)
                ptcl, dens = self.ptcl_cache[event_dim], self.dens_cache[event_dim]
                s_shape, e_shape = torch.Size([ptcl.shape[0]]), torch.Size([ptcl.shape[1]])
                new_marg_msg = Message(MessageType.Particles,
                                       batch_shape=joint_msg.b_shape, sample_shape=s_shape, event_shape=e_shape,
                                       particles=[ptcl], weight=1, log_densities=[dens],
                                       device=self.device)
                # Append the particle into the joint message
                joint_msg = joint_msg.event_cross_product(new_marg_msg)
                # Update running_vars
                running_vars.append(arg)

        # Step 3: Iterate through all pattern elements, de-concatenate the joint particles
        for pattern in self.event_transforms:
            if isinstance(pattern.pred_arg, tuple):
                # Check that the cache is available, otherwise raise value error
                if self.ptcl_cache is None or self.dens_cache is None:
                    raise ValueError(
                        "In {}: Outward computation needs to access particle cache to de-concatenate joint particles, "
                        "but the cache is None."
                        .format(self.name)
                    )
                # Deconcatenate the joint event. New marginal event dimensions will be appended at the end
                decat_event_dim = running_vars.index(pattern.pred_arg)
                orig_event_dims = [self.ld_pred_args.index(arg) for arg in pattern.pred_arg]
                target_event_nums = [self.ptcl_cache[dim].shape[0] for dim in orig_event_dims]
                target_event_sizes = [self.ptcl_cache[dim].shape[1] for dim in orig_event_dims]
                joint_msg = joint_msg.event_deconcatenate(decat_event_dim, target_event_nums, target_event_sizes)
                # Update running_vars accordingly
                running_vars += list(pattern.pred_arg)
                running_vars.remove(pattern.pred_arg)

        # Finally, reorder the event dimensions of the joint message and send to the outgoing linkdata
        # Sanity check: all predicate arguments have been recovered
        assert set(running_vars) == set(self.ld_pred_args)
        event_perm_order = [running_vars.index(arg) for arg in self.ld_pred_args]
        out_msg = joint_msg.event_permute(event_perm_order)

        out_lds[0].write(out_msg)


class PRFN(BetaFactorNode):
    """
        Particle Reweight Factor Node

        This node is one of the two Conditional Beta Factor Nodes that is responsible for importance-reweighting the
        predicate's knowledge based on external particles borrowed from other predicate patterns, who binds with
        this predicate pattern on some random pattern variables and declares itself as the MASTER branch for serving
        particles.

        PRFN only exists in the condition part of a Beta subgraph, and therefore it only accepts inward-directional
        linkdata. Its computation can be summarized as follows:

        Inward computation:

            1. **Retrieval**:
               Every outward-side linkdata shall be tagged in their special attribute dictionary if it is the HOME
               branch or an ALIEN branch. The home linkdata connects from the Alpha terminal of this predicate
               branch, whereas an ALIEN linkdata connects from a PDFN from other predicate branches. The HOME branch
               provides the original multivariate message from the predicate with its event dimensions corresponding to
               the predicate's random arguments, whereas an ALIEN linkdata provides a univariate marginal message with
               its single event dimension corresponding to a pattern variable that this predicate binds on.

            2. **Transformation**:
               If an event transformation is declared on the pattern element for which an ALIEN message is retrieved
               in the step above, then apply the *backward* version of the transformation on this ALIEN message. This
               is done so that the transformed message would represent the predicate argument instead of the pattern
               variable.

            3. **Validation**
               Check that all transformed alien particle values satisfy the value constraints of the corresponding
               predicate arguments. Raise ValueError if the constraints is not met.

            4. **Reweighting**:
               Use the alien particles, together with original particles for other predicate arguments, to reweight
               based on the distribution parameters included in the incoming message from the HOME branch linkdata.

            5. **Reordering**:
               Reorder the reweighted message's event dimensions so that they align with the predicate argument order
               of the outgoing linkdata

        Since importance reweighting requires distribution parameters to instantiate a distribution instance, the
        incoming message from the Home branch must be of type ``MessageType.Dual``.

        Parameters
        ----------
        event_transforms : Iterable of EventTransform
            The non-master pattern elements of this predicate whose associated pattern variable are also bound by other
            predicate patterns.
        knowledge_server : KnowledgeServer
            The knowledge server for performing importance reweighting.
    """
    def __init__(self, name: str, event_transforms: IterableType[EventTransform], knowledge_server: KnowledgeServer,
                 **kwargs):
        super(PRFN, self).__init__(name, **kwargs)
        assert isinstance(event_transforms, Iterable) and all(isinstance(et, EventTransform) for et in event_transforms)
        assert all(et.finalized for et in event_transforms)     # All patterns must be finalized
        assert isinstance(knowledge_server, KnowledgeServer)

        self.event_transforms = tuple(event_transforms)
        self.knowledge_server = knowledge_server

        # Internal data structures
        self.et_pat_vars = tuple(et.pat_var for et in self.event_transforms)
        self.et_flattened_pred_args = []
        for pattern in self.event_transforms:
            if isinstance(pattern.pred_arg, tuple):
                self.et_flattened_pred_args += list(pattern.pred_arg)
            else:
                self.et_flattened_pred_args.append(pattern.pred_arg)
        self.et_surrogate_pred_args = [pattern.surrogate_pred_arg for pattern in self.event_transforms]

        # Internal data structures that will be populated after add_link() is called
        self.home_ld_pred_args: Tuple[Variable] = tuple()
        self.out_ld_pred_args: Tuple[Variable] = tuple()
        self.et2alien_ld: Dict[EventTransform, LinkData] = {}
        self.alien_ld_pat_vars: List[Variable] = []
        self.home_ld: LinkData = None
        self.alien_lds: List[LinkData] = []
        
    def add_link(self, linkdata: LinkData):
        """
            PRFN has the following additional requirements regarding acceptable linkdata:

                1. Only inward-directional linkdata
                2. Only one outgoing linkdata
                    a. The predicate arguments inferred from the outgoing linkdata must be a superset of the set of
                       surrogate predicate arguments inferred from the event transforms.
                3. Incoming linkdata must have special attribute 'alien' with value True/False
                    a. Only one 'alien=False' linkdata, i.e. the Home linkdata
                        i. Its predicate arguments must be a superset of the (flattened) predicate arguments inferred
                           from the event transforms declared during init.
                    b. Every 'alien=True' linkdata, i.e., the alien linkdata, must connect from a univariate variable
                       node.
                        i. The variable must be distinct for each alien linkdata.
                        ii. The variable must be referenced by at least one of the event transforms declared during
                            init.

        """
        super(PRFN, self).add_link(linkdata)

        # Check requirement 1
        assert linkdata.attr['direction'] == 'inward', \
            "In {}: PRFN only accept inward-directional linkdata.".format(self.name)
        # Check requirement 2
        if not linkdata.to_fn:
            assert len(self.labeled_ld_group['inward'][False]) == 1, \
                "In {}: PRFN only accept one inward outgoing linkdata.".format(self.name)
            self.out_ld_pred_args = linkdata.vn.ran_vars
            # Check requirement 2.a
            assert set(linkdata.vn.ran_vars).issuperset(set(self.et_surrogate_pred_args)), \
                "In {}: The predicate arguments inferred from the outgoing linkdata must be a superset of the set of " \
                "surrogate predicate arguments inferred from the event transforms. Expect a superset of arguments {}, "\
                "but instead found arguments {} in linkdata '{}'." \
                .format(self.name, self.et_surrogate_pred_args, linkdata.vn.ran_vars, linkdata)
        # Check requirement 3
        if linkdata.to_fn:
            assert 'alien' in linkdata.attr.keys() and isinstance(linkdata.attr['alien'], bool), \
                "In {}: Incoming linkdata to a PRFN must be tagged with `alien` special attribute, whose value " \
                "must be boolean.".format(self.name)
            if not linkdata.attr['alien']:
                # Check requirement 3.a
                assert self.home_ld is None, \
                    "In {}: PRFN can only accept one HOME branch incoming linkdata. Linkdata '{}' is already " \
                    "registered as the HOME incoming linkdata."\
                    .format(self.name, self.home_ld)
                self.home_ld = linkdata
                # Check requirement 3.a.i
                assert set(linkdata.vn.ran_vars).issuperset(set(self.et_flattened_pred_args)), \
                    "In {}: The set of predicate arguments inferred from the HOME branch incoming linkdata must be " \
                    "a superset of the set of predicate arguments inferred from the event transforms given during " \
                    "initialization. Expect a superset of the subset {}, however found {}." \
                    .format(self.name, set(self.et_flattened_pred_args), set(linkdata.vn.ran_vars))
                self.home_ld_pred_args = linkdata.vn.ran_vars
            else:
                self.alien_lds.append(linkdata)
                # Check requirement 3.b
                assert len(linkdata.vn.ran_vars) == 1, \
                    "In {}: Every ALIEN incoming linkdata to a PRFN must connect from a univariate variable node. " \
                    "Found linkdata '{}' is connected from a node with random variables {}." \
                    .format(self.name, linkdata, linkdata.vn.ran_vars)
                # Check requirement 3.b.i
                ld_ran_var = linkdata.vn.ran_vars[0]
                assert ld_ran_var not in self.alien_ld_pat_vars, \
                    "In {}: Every ALIEN incoming linkdata to a PRFN must connect from a distinct variable node. " \
                    "Found linkdata '{}' is connected from a node with random variable '{}' that is already " \
                    "registered." \
                    .format(self.name, linkdata, ld_ran_var)
                self.alien_ld_pat_vars.append(ld_ran_var)
                # Check requirement 3.b.ii
                matched_ets = [et for et in self.event_transforms if et.pat_var == ld_ran_var]
                assert len(matched_ets) > 0, \
                    "In {}: Every ALIEN incoming linkdata to a PRFN must connect from a variable node whose random " \
                    "variable must be referenced as the pattern variable by at least one event transforms declared " \
                    "during initialization. Found linkdata '{}' with unknown pattern variable '{}'." \
                    .format(self.name, linkdata, ld_ran_var)
                for matched_et in matched_ets:
                    self.et2alien_ld[matched_et] = linkdata

    def precompute_check(self):
        """
            PRFN has the following additional requirements regarding computable config:

                1. One outgoing linkdata is registered. Check omitted since BetaFactorNode will check this for us.
                2. One Home linkdata is registered
                3. At least one Alien linkdata is registered
                4. All pattern variables that are referenced by the event transforms must also be referenced by the
                   Alien linkdata
                5. The set of outgoing predicate arguments (the variables of the vn connected to by the outgoing
                   linkdata) must be equal to:
                   set(incoming predicate arguments) - set(event transform flattened predicate arguments) +
                   set(event transform surrogate predicate arguments)
        """
        super(PRFN, self).precompute_check()

        # Check requirement 2
        assert self.home_ld is not None, \
            "In {}: One HOME incoming linkdata must be registered to start computation.".format(self.name)
        # Check requirement 3
        assert len(self.alien_lds) > 0, \
            "In {}: At least one ALIEN incoming linkdata must be registered to start computation.".format(self.name)
        # Check requirement 4
        assert set(self.et_pat_vars) == set(self.alien_ld_pat_vars), \
            "In {}: All pattern variables that are referenced by the event transforms must also be referenced by the " \
            "ALIEN incoming linkdata. Inferred pattern variables {} from the event transforms, while inferred {} from "\
            "the ALIEN linkdata." \
            .format(self.name, self.et_pat_vars, self.alien_ld_pat_vars)
        # Check requirement 5
        expected_out_pred_args = (set(self.home_ld_pred_args) - set(self.et_flattened_pred_args))\
            .union(set(et.surrogate_pred_arg for et in self.event_transforms))
        assert set(self.out_ld_pred_args) == expected_out_pred_args, \
            "In {}: Unexpected set of outgoing predicate arguments found in the outgoing linkdata. Expect set of " \
            "predicate arguments {}, instead found {}"\
            .format(self.name, list(expected_out_pred_args), self.out_ld_pred_args)

    def inward_compute(self, in_lds: List[LinkData], out_lds: List[LinkData]):
        """
            Inward computation:

            1. Retrieval
            2. Transformation
            3. Validation
            4. Reweighting
            5. Reordering
        """
        # Step 1
        home_msg = self.home_ld.read()
        # Check that the home message contains both particles and parameter
        if home_msg.type is not MessageType.Dual:
            raise ValueError(
                "In {}: The HOME message sent to a PRFN must be of type `MessageType.Dual`, i.e., containing both "
                "particles and parameter. However, found the HOME message having type '{}'."
                .format(self.name, home_msg.type)
            )

        # Step 2
        # Iterate through patterns and retrieve the alien message for that pattern
        # Apply BACKWARD transformation if declared
        pattern2msg = {}
        for pattern, alien_ld in self.et2alien_ld.items():
            alien_msg = alien_ld.read()
            if pattern.has_trans:
                alien_msg = alien_msg.event_transform(pattern.backward_trans)
            pattern2msg[pattern] = alien_msg

        # Step 3: Validation
        # Iterate all patterns and check that the candidate particles meet the constraints
        # Use the surrogate predicate argument's constraints to check
        for pattern, alien_msg in pattern2msg.items():
            cstr_list = pattern.surrogate_pred_arg.constraints
            ptcl = alien_msg.particles[0]
            for cstr in cstr_list:
                if not torch.all(cstr.check(ptcl)):
                    raise ValueError(
                        "In {}: The particle values of the incoming ALIEN message for pattern '{}' does not meet the "
                        "value constraint '{}' declared for the pattern's predicate argument."
                        .format(self.name, pattern, cstr)
                    )

        # Step 4: Reweighting
        alt_ptcl, alt_dens, index_map = [], [], {}
        running_vars = []
        # Original particles from home message
        home_ptcl, home_dens = home_msg.particles, home_msg.log_densities
        # First, pick up the particles corresponding to the predicate arguments unreferenced by any pattern
        for arg_id, pred_arg in enumerate(self.home_ld_pred_args):
            if pred_arg not in self.et_flattened_pred_args:
                alt_ptcl.append(home_ptcl[arg_id])
                alt_dens.append(home_dens[arg_id])
                index_map[len(alt_ptcl) - 1] = arg_id
                running_vars.append(pred_arg)
        # Then, iterate through patterns and push the alien particles
        for pattern, alien_msg in pattern2msg.items():
            alt_ptcl.append(alien_msg.particles[0])
            alt_dens.append(alien_msg.log_densities[0])
            running_vars.append(pattern.surrogate_pred_arg)
            # If a list of predicate arguments, map to a list of indices
            if isinstance(pattern.pred_arg, Iterable):
                arg_ids = [self.home_ld_pred_args.index(arg) for arg in pattern.pred_arg]
                index_map[len(alt_ptcl) - 1] = arg_ids
            # Otherwise if single predicate argument, map to a single index
            else:
                arg_id = self.home_ld_pred_args.index(pattern.pred_arg)
                index_map[len(alt_ptcl) - 1] = arg_id

        # Obtain surrogate log prob
        srg_log_prob = self.knowledge_server.surrogate_log_prob(home_msg.parameter, alt_ptcl, index_map)

        # Now generate the reweighed message
        # Generate a new Particles message with home particles substituted by alien particles
        new_s_shape, new_e_shape = torch.Size([ptcl.shape[0] for ptcl in alt_ptcl]), \
                                   torch.Size([ptcl.shape[1] for ptcl in alt_ptcl])
        sub_msg = Message(MessageType.Particles,
                          batch_shape=home_msg.b_shape, sample_shape=new_s_shape, event_shape=new_e_shape,
                          particles=alt_ptcl, weight=1, log_densities=alt_dens)
        # Time to reweight the message
        reweight_msg = sub_msg.event_reweight(srg_log_prob)

        # Step 5:
        # First sanity check that we indeed have all the desired outgoing predicate arguments
        assert set(running_vars) == set(self.out_ld_pred_args)
        # Permute the event dimensions to align with outgoing predicate arguments
        perm_order = [running_vars.index(arg) for arg in self.out_ld_pred_args]
        out_msg = reweight_msg.event_permute(perm_order)

        # Write message
        out_lds[0].write(out_msg)

    def outward_compute(self, in_lds: List[LinkData], out_lds: List[LinkData]):
        """
            PRFN does not have outward computation.
        """
        raise NotImplementedError
