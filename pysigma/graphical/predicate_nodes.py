"""
    All nodes related to a predicate subgraph
"""
from __future__ import annotations      # For postponed evaluation of typing annotations
from typing import Union, Optional, Tuple, Dict
from typing import Iterable as IterableType
import warnings
from abc import ABC
from collections.abc import Iterable
import torch
from torch.nn import Parameter
import torch.distributions as D
from ..defs import VariableMetatype, Variable, MessageType, Message, NP_EPSILON
from .basic_nodes import LinkData, VariableNode, FactorNode, NodeConfigurationError
from ..utils import compatible_shape, KnowledgeServer


class WMVN(VariableNode):
    """Working Memory Variable Node.

    Gate node connecting predicate structure to conditionals.

    WMVN will attempt to combine incoming messages, regardless of whether they come from alpha terminals in Conditional
    subgraphs, or from other nodes in the Predicate subgraph. The combination generally yields a message that is
    representative of the marginal belief from a certain part of the graphical model. It is then sent to downstream
    nodes for further processing.

    A KnowledgeServer instance associated with the belonging Predicate is required because occasionally log prob of
    particles needs to be queried.

    WMVN quiescence state:
        A WMVN reaches quiescence state if and only if **all** incoming linkdata do not contain new message. Same as
        default quiescence state definition.

    It is defined as such so that, although it may not be efficient that WMVN has to fire multiple times while
    sending partially complete messages, it is guaranteed that no new arriving message would be blocked herein simply
    because other messages were blocked elsewhere and did not arrive at this node, consequently blocking all downstream
    processing.

    Parameters
    ----------
    name : str
        Name of this variable node.
    ks : KnowledgeServer
        The KnowledgeServer instance associated with the belonging Predicate.
    rel_var_list : iterable of Variable
        Iterable of relational variables. Corresponds to the batch dimensions. Used to check ``b_shape`` attribute of
        incoming messages.
    param_var : Variable, optional
        The parameter variable. Corresponds to the parameter dimension. Used to check ``p_shape`` attribute of incoming
        messages.
    index_var_list : iterable of Variable, optional
        Iterable of indexing variables. Corresponds to the sample dimensions. Used to check ``s_shape`` attribute of
        incoming messages. Must specify if `ran_var_list` is specified.
    ran_var_list : iterable of Variable, optional
        Iterable of random variables. Corresponds to the event dimensions. Used to check ``e_shape`` attribute of
        incoming messages. Must specify if `index_var_list` is specified.

    Attributes
    ----------
    ks : KnowledgeServer
        The KnowledgeServer instance associated with the belonging Predicate.


    """

    def __init__(self,
                 name: str,
                 ks: KnowledgeServer,
                 rel_var_list: IterableType[Variable],
                 param_var: Variable = None,
                 index_var_list: IterableType[Variable] = None,
                 ran_var_list: IterableType[Variable] = None,
                 **kwargs):
        assert isinstance(ks, KnowledgeServer)
        super(WMVN, self).__init__(name, rel_var_list, param_var, index_var_list, ran_var_list, **kwargs)
        self.pretty_log["node type"] = "Working Memory Variable Node"

        # Distribution class the Predicate self belongs to is assuming
        self.ks = ks
        # Cache for temporarily saving computation result for combination
        self._cache: Dict[Tuple[LinkData, ...], Message] = {}

    def precompute_check(self):
        """The computable condition for a WMVN is that there is at least one incoming link. Not having an outgoing link
        would not cause trouble in compute(), although in this case no WMVN's internal state is changed and no outgoing
        message is propagated
        """
        if len(self.in_linkdata) == 0:
            raise NodeConfigurationError("Wrong configuration for node {}: a WMVN expects at least one incoming "
                                         "linkdata. Found no registered incoming linkdata.".format(self.name))

    @VariableNode.compute_control
    def compute(self):
        """Combine incoming message to this Predicate subgraph.

        Will attempt to combine incoming messages if there are multiple incoming links, subsuming the functionality of
        FAN node in Lisp Sigma. Combination can be carried out if messages are all Parameter type, or if there exist
        Particles type messages but all of them are compatible (sharing the same particle values as well as sampling
        log densities).

        Raises
        ------
        AssertionError
            If found that not all incoming messages contain either parameter or particles, or both.
        AssertionError
            If the ``MessageType.Particles`` type messages in the incoming links do not share the same particles
            (including particle value tensors and/or particle log sampling density tensors)

        Warns
        -----
        UserWarning
            If there is only one incoming link, but this link is connected to a factor node that is also connected from
            this WMVN via an outgoing link. This means, per Sum-Product exclusion principle, that this outgoing link
            will also be neglected during compute() and no new message will ever be propagated via this link.

        Notes
        -----
        Following combination procedure is carried out to conform to the standard of all inference methods

        1. If incoming messages all contains parameter, then these parameters will be combined. The outgoing message
           will contain the combined parameter. Otherwise if any incoming message does not contain parameter, this
           procedure will not be performed and the outgoing message will not contain parameter, but rather a combined
           particle list will be computed. See the followings.

           For the parameter message combination procedure and accompanying assumptions, see
           :ref:`Message class notes on arithmetic structures<message-arithmetic-structures-notes>` for more details.

        2. If any incoming message also contains particles, then it is imperative that all such messages contain the
           same particle values as well as particle log sampling densities. The particle weights will be gathered from
           these messages and combined. For all other parameter messages that only contain parameters, the particles
           from these particles messages will be used as the surrogate particles to generate a particle message as
           approximation, which will take part in the particles combination procedure.

        In short, here is a summary listing the correspondence between incoming message types and outgoing message
        types:

        +----------------------------------+--------------------------+
        |   Incoming Message Types         |    Outgoing Message Type |
        +==================================+==========================+
        |   Parameter                      |    Parameter             |
        +----------------------------------+--------------------------+
        |   Particles                      |    Particles             |
        +----------------------------------+--------------------------+
        |   Both                           |    Both                  |
        +----------------------------------+--------------------------+
        |   Parameter + Particles          |    Particles             |
        +----------------------------------+--------------------------+
        |   Parameter + Both               |    Both                  |
        +----------------------------------+--------------------------+
        |   Particles + Both               |    Particles             |
        +----------------------------------+--------------------------+
        |   Parameter + Particles + Both   |    Particles             |
        +----------------------------------+--------------------------+

        Or, logically speaking, the outgoing message will contain particles if **any** incoming message also contains
        particles, but it will contain parameter only if **all** incoming messages contain parameters.

        Note that in any case, incoming message can not be ``MessageType.Undefined`` type, in which case an exception
        will be raised.

        When combining messages, will exclude message from the link to which the combined message is to send to (if
        such a bidirected link exists). This implements the Sum-Product algorithm's variable node semantics, if
        this WMVN is served as both WMVN_IN and WMVN_OUT, i.e., if the predicate is of memory-less vector type.

        Optimization is implemented by caching the combination result for each outgoing link. If two outgoing links
        share the same set of incoming links that provide the messages, previously computed result will be reused
        """
        # Relay message if only one incoming link
        if len(self.in_linkdata) == 1:
            in_ld = self.in_linkdata[0]
            msg = in_ld.read()
            for out_ld in self.out_linkdata:
                # Throw a warning if the outgoing link is connected to the same factor node that the only incoming
                #   link is connected to, since in such case no message would be sent to that factor node
                if out_ld.fn is in_ld.fn:
                    warnings.warn("WMVN '{}' is connected to factor node '{}', while its only incoming link is also "
                                  "connected from the same factor node. In this case no message would be sent out to "
                                  "the factor node. Please check if the model is properly defined"
                                  .format(self.name, out_ld.fn.name))
                else:
                    out_ld.write(msg)
        # Otherwise, combine messages
        else:
            for out_ld in self.out_linkdata:
                # The tuple of all incoming linkdata that are not connected to the factor node the selected outgoing
                #   linkdata is connected to.
                # Use tuple here because tuple is hashable and we will use it as keys to cache dictionary
                in_lds = tuple(in_ld for in_ld in self.in_linkdata if in_ld.fn is not out_ld.fn)

                # Check if there's cached data. If yes, use cached result
                if in_lds in self._cache.keys():
                    out_msg = self._cache[in_lds]
                # Otherwise, compute combined message
                else:
                    combined_param_msg, combined_ptcl_msg = None, None
                    in_msgs = tuple(in_ld.read() for in_ld in in_lds)

                    assert all(MessageType.Parameter in msg.type or MessageType.Particles in msg.type
                               for msg in in_msgs), \
                        "At {}: Expect all incoming messages to contain either parameter or particles, or both, but " \
                        "the types of the incoming messages are: {}" \
                            .format(self.name, list(msg.type for msg in in_msgs))

                    # Only if all incoming messages contain parameters should we combine the parameters
                    if all(MessageType.Parameter in msg.type for msg in in_msgs):
                        param_msgs = tuple(msg.reduce_type(MessageType.Parameter) for msg in in_msgs)
                        combined_param_msg = sum(param_msgs, Message.identity())

                    # If any incoming message contains particles, we should proceed to combine them
                    if any(MessageType.Particles in msg.type for msg in in_msgs):
                        # 1.a. Ensure all particle lists are homogeneous
                        particle_msgs = tuple(msg.reduce_type(MessageType.Particles) for msg in in_msgs
                                              if MessageType.Particles in msg.type)
                        particle_lds = tuple(ld for ld in in_lds if MessageType.Particles in ld.read().type)
                        tmp_msg, tmp_ld = particle_msgs[0], particle_lds[0]
                        for msg, in_ld in zip(particle_msgs, particle_lds):
                            assert tmp_msg.same_particles_as(msg), \
                                "At WMVN '{}': When attempting to combine incoming messages, found that incoming " \
                                "Particle message's particles (particle value tensors and/or particle log sampling " \
                                "density tensors) from  linkdata '{}' does not agree with that of incoming Particle " \
                                "message from linkdata '{}'" \
                                    .format(self.name, in_ld, tmp_ld)

                        # 1.b Find message that only contains parameter. If they exist, use the particles from the above
                        # messages as surrogate particle list and query its log prob w.r.t. the parameter.
                        param_msgs = tuple(msg for msg in in_msgs if MessageType.Particles not in msg.type)
                        particles = tmp_msg.particles

                        # 2.b Compute particle weights w.r.t. distributions induced by the Parameter type messages
                        candidate_msgs = list(particle_msgs)
                        for param_msg in param_msgs:
                            target_log_prob = self.ks.surrogate_log_prob(param_msg.parameter, particles)
                            surrogate_msg = tmp_msg.event_reweight(target_log_prob)
                            candidate_msgs.append(surrogate_msg)

                        # Combine messages
                        combined_ptcl_msg = sum(candidate_msgs, Message.identity())

                    # Compose components
                    if combined_param_msg is not None and combined_ptcl_msg is not None:
                        out_msg = Message.compose(combined_param_msg, combined_ptcl_msg)
                    elif combined_param_msg is not None:
                        out_msg = combined_param_msg
                    else:
                        out_msg = combined_ptcl_msg

                    # Cache result
                    self._cache[in_lds] = out_msg
                # Send message
                out_ld.write(out_msg)

            # Clear cache
            self._cache.clear()


class WMFN(FactorNode, ABC):
    """The abstract base class for all types of Working Memory Factor Nodes

    """
    def __init__(self, name: str, **kwargs):
        super(WMFN, self).__init__(name, **kwargs)

    @property
    def quiescence(self) -> bool:
        """For WMFN, quiescence is reached if messages stored in cache has been sent to the nearby node.

        """
        return self.visited


class WMFN_MCMC(WMFN):
    """WMFN-MCMC implementing the message update step of the Markov Chain Monte Carlo (MCMC) inference procedure.

    Metropolis-Hastings algorithm is implemented herein for the MCMC procedure.

    Due to the nature of the M-H algorithm, the candidate particles generated by the random walk distribution needs to
    be evaluated against the target function. The evaluated values of the candidate particles are then compared to the
    values of the posterior particles from last cycle representing the previous posterior belief. A coin is then flipped
    with a probability equal to the ratio of the candidate particle's weight to the posterior particle's weight to
    determine if the posterior particle is to be replaced by the candidate particle. Otherwise, the current particle is
    retained. This process will yield a posterior particle list of this cycle representing the current belief. Then,
    a step of random walk is performed on the yielded posterior particles to generate the candidate particles to be
    evaluated in the next cycle.

    To accommodate to this procedure, in any particle-based MCMC predicate, the WMFN is connected to TWO groups of
    WMVN_IN and WMVN_OUT, named with suffix `_EVAL` and `_POST` respectively. The `WMVN_IN_EVAL` and `WMVN_OUT_EVAL`
    together is responsible for sending and receiving the candidate message (message that contains the randomly walked
    candidate particles) to and from the Gamma node to be evaluated by the target functions. On the other hand,
    `WMVN_IN_POST` and `WMVN_OUT_POST` together will send the true posterior belief messages, as well as receiving the
    marginal posterior from other part of the graph evaluated on the true posterior belief particles.

    Since the candidate particles and the posterior particles need to be evaluated against the same set of target
    functions, both groups of WMVN_IN and WMVN_OUT should always connect to the same set of Conditionals where this
    predicate appears as an action or condact pattern. However, since only `WMVN_OUT_POST` sends the true posterior
    belief messages, it is `WMVN_OUT_POST` rather than `WMVN_OUT_EVAL` that should connect to the condition subgraph in
    order to propagate the posterior belief to other parts of the graph.

    In the condition subgraph, both `WMVN_OUT_POST` and `WMVN_OUT_EVAL` should be associated with the same set of
    pattern random variables to ensure message exclusion.

    To be more specific:

        - If the predicate appears only as a condition pattern, then `WMVN_OUT_POST` serves as the only message source
          and it is the only node that is connected to the corresponding conditional alpha subgraph.
        - If the predicate appears only as an action pattern, then both `WMVN_IN_POST` and `WMVN_IN_EVAL` should be
          connected from the action subgraphs *with the same random variables*. `WMVN_OUT_POST` and `WMVN_OUT_EVAL`
          should skip-connect to the action beta subgraph to directly provide particles for evaluation purpose.
        - If the predicate appears as a condact pattern, then it is the combination of the above two. In other words,
          `WMVN_OUT_POST` will both connect to the condition alpha subgraph as well as to the action beta subgraph.

    By default, the MultivariateNormal distribution with identity covariance matrix is chosen as the random walk
    distribution.

    Parameters
    ----------
    name : str
        Name of this factor node
    index_var_list : Iterable of Variable
        Iterable of indexing variables. Corresponds to the sample dimensions. Used to check ``s_shape`` attribute of
        incoming messages. Note that the indexing variables corresponds to the dimensions of the true posterior belief
        particle messages. It is expected that the sample dimension size of the messages from WMVN_IN and to
        WMVN_OUT_EVAL will be twice as large as the size of the indexing variables declared herein.
    ran_var_list : Iterable of Variable
        Iterable of random variables. Corresponds to the event dimensions. Used to check ``e_shape`` attribute of
        incoming messages.

    Attributes
    ----------
    eval_msg_cache : Message
        The message cache containing the message for evaluation, which contains the candidate particles. It is updated
        during modification phases, and used during decision phases.
    post_msg_cache : Message
        The message cache containing the true posterior belief particles. It is updated during modification phases,
        and used during decision phases.

    """

    def __init__(self,
                 name: str,
                 index_var_list: IterableType[Variable],
                 ran_var_list: IterableType[Variable],
                 **kwargs):
        assert isinstance(index_var_list, Iterable) and \
               (isinstance(v, Variable) and v.metatype is VariableMetatype.Indexing for v in index_var_list)
        assert isinstance(ran_var_list, Iterable) and \
               (isinstance(v, Variable) and v.metatype is VariableMetatype.Random for v in ran_var_list)
        super(WMFN_MCMC, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Markov Chain Monte Carlo Working Memory Factor Node"

        self.index_vars = tuple(index_var_list)
        self.ran_vars = tuple(ran_var_list)
        self.s_shape: torch.Size = torch.Size([v.size for v in self.index_vars])
        self.e_shape: torch.Size = torch.Size([v.size for v in self.ran_vars])
        self.b_shape: torch.Size = torch.Size([])       # This will be inferred during runtime from incoming messages

        self.eval_msg_cache = Message.identity()  # Initialize to identity
        self.post_msg_cache = Message.identity()  # Initialize to identity
        self.ld_in_eval: Optional[LinkData] = None
        self.ld_out_eval: Optional[LinkData] = None
        self.ld_in_post: Optional[LinkData] = None
        self.ld_out_post: Optional[LinkData] = None

        # Default to using MultivariateNormal distributions with identity covariance matrix as the random walk
        # distributions. One random walk distribution for one random variable.
        self.walk_dist = [D.MultivariateNormal(
            loc=torch.zeros(s_size, e_size, device=self.device),  # Default to 0. Will change each modification phase.
            covariance_matrix=torch.eye(e_size, device=self.device).unsqueeze(dim=0).expand(s_size, -1, -1)
        ) for s_size, e_size in zip(self.s_shape, self.e_shape)]

    def add_link(self, linkdata: LinkData):
        """WMFN admits two groups of WMVN_IN and WMVN_OUT, for posterior messages and evaluation messages respectively.

        `linkdata` must specify a special attribute, either `type=posterior` or `type=evaluation`, in `linkdata.attr`
        dictionary.

        Will check if the message shape declared in `linkdata` is compatible with what is declared for this node.

        Parameters
        ----------
        linkdata : LinkData
            The linkdata to be registered.
        """
        assert isinstance(linkdata.vn, WMVN), "In {}: WMFN_MCMC can only be connected to WMVN.".format(self.name)
        assert 'type' in linkdata.attr.keys() and linkdata.attr['type'] in ['evaluation', 'posterior'], \
            "In {}: Linkdata must specify a special attribute `type` with value `posterior` or `evaluation`."\
            .format(self.name)
        if linkdata.to_fn:
            if linkdata.attr['type'] == 'evaluation':
                assert self.ld_in_eval is None, "In {}: Duplicate connection to WMVN_IN_EVAL.".format(self.name)
                self.ld_in_eval = linkdata
            else:
                assert self.ld_in_post is None, "In {}: Duplicate connection to WMVN_IN_POST.".format(self.name)
                self.ld_in_post = linkdata
        else:
            if linkdata.attr['type'] == 'evaluation':
                assert self.ld_out_eval is None, "In {}: Duplicate connection to WMVN_OUT_EVAL.".format(self.name)
                self.ld_out_eval = linkdata
            else:
                assert self.ld_out_post is None, "In {}: Duplicate connection to WMVN_OUT_POST.".format(self.name)
                self.ld_out_post = linkdata

        # Check shape
        _, _, s_shape, e_shape = linkdata.msg_shape
        assert s_shape == self.s_shape, \
            "In {}: linkdata should have the same sample shape as what is declared for this node. Expect {}, but " \
            "found {}." \
            .format(self.name, self.s_shape, s_shape)
        assert e_shape == self.e_shape, \
            "In {}: linkdata should have the same event shape as what is declared for this node. Expect {}, but " \
            "found {}." \
            .format(self.name, self.e_shape, e_shape)

        super(WMFN_MCMC, self).add_link(linkdata)

    def _random_walk(self):
        """
        Private subroutine. Take particles from `self.post_msg_cache` and use `self.walk_dist` to perform a random walk
        to generate new particles. The new particles will be encapsulated in a new message saved in
        `self.eval_msg_cache`. The new message will have uniform weight and uniform log densities.
        """
        new_eval_ptcl = []
        for post_ptcl, walk_dist in zip(self.post_msg_cache.particles, self.walk_dist):
            walk_dist.loc = post_ptcl  # Set posterior particles as the mean
            eval_ptcl = walk_dist.sample()  # Draw one sample
            new_eval_ptcl.append(eval_ptcl)
        # Instantiate a uniform weight tensor here instead of using integer 1 because the latter will render the message
        #   an identity message, which may halt message propagation.
        uniform_log_densities = [torch.zeros(s_size, device=self.device) for s_size in self.s_shape]
        uniform_weight = torch.ones(self.b_shape + self.s_shape, device=self.device)
        self.eval_msg_cache = Message(MessageType.Particles,
                                      batch_shape=self.b_shape, sample_shape=self.s_shape, event_shape=self.e_shape,
                                      particles=new_eval_ptcl, weight=uniform_weight,
                                      log_densities=uniform_log_densities)

    def init_particles(self, init_ptcl_msg: Message):
        """
        Initialize the WMFN with an initial particles message to start the random walk procedure.

        Parameters
        ----------
        init_ptcl_msg: Message
            The initial particles message to start with.

        Raises
        ------
        ValueError
            If `init_ptcl_msg` has incompatible sample shape and event shape.
        """
        assert isinstance(init_ptcl_msg, Message) and MessageType.Particles in init_ptcl_msg.type
        if init_ptcl_msg.s_shape != self.s_shape or init_ptcl_msg.e_shape != self.e_shape:
            raise ValueError("In {}: `init_ptcl_msg`'s sample shape and event shape are incompatible. Expecting sample "
                             "shape {} and event shape {}, but found {} and {}."
                             .format(self.name, self.s_shape, self.e_shape, init_ptcl_msg.s_shape, init_ptcl_msg.e_shape))
        # Infer batch shape
        self.b_shape = init_ptcl_msg.b_shape
        # Set message cache
        self.post_msg_cache = init_ptcl_msg
        # Take one random walk to generate the initial candidate particles message
        self._random_walk()

    def modify(self):
        """
        Carries out the message update of the Metropolis-Hastings algorithm. Update eval_msg_cache and post_msg_cache.
        """
        # Get message from incoming link and check shape compatibility
        in_eval_msg, in_post_msg = self.ld_in_eval.read(), self.ld_in_post.read()
        assert isinstance(in_eval_msg, Message) and isinstance(in_post_msg, Message)
        # If either message is empty, then messages haven't propagated here yet. No modification and defer to next cycle
        if in_eval_msg.isid or in_post_msg.isid:
            return
        # Not empty message, check type
        assert MessageType.Particles in in_eval_msg.type and MessageType.Particles in in_post_msg.type, \
            "In {}: Expect all incoming messages to be Particles type, but instead found type {}, {} for in_eval_msg " \
            "and in_post_msg respectively.".format(self.name, in_eval_msg.type, in_post_msg.type)

        # infer the batch shape
        assert in_eval_msg.b_shape == in_post_msg.b_shape, "In {}: Incoming posterior and evaluation messages have " \
                                                           "different batch shape: found {} and {} respectively."\
            .format(self.name, in_post_msg.b_shape, in_eval_msg.b_shape)
        self.b_shape = in_eval_msg.b_shape
        # Step 1: Generate new posterior belief
        # 1.1 Summarize (by multiplying) weights over batch dimensions
        # in_eval_weight_sum = in_eval_msg.weight.log().sum(dim=list(range(len(self.b_shape)))).exp()
        # in_post_weight_sum = in_post_msg.weight.log().sum(dim=list(range(len(self.b_shape)))).exp()
        in_eval_weight_sum = in_eval_msg.weight.sum(dim=list(range(len(self.b_shape))))
        in_post_weight_sum = in_post_msg.weight.sum(dim=list(range(len(self.b_shape))))
        # 1.2 Marginalize (by summing) weight w.r.t. each random variable respectively
        in_eval_var_weights, in_post_var_weights = [], []
        if len(self.s_shape) > 1:
            for dim in range(len(self.s_shape)):
                dims = list(range(len(self.s_shape)))
                dims.remove(dim)
                in_eval_var_weights.append(in_eval_weight_sum.sum(dim=dims))
                in_post_var_weights.append(in_post_weight_sum.sum(dim=dims))
        else:
            in_eval_var_weights, in_post_var_weights = [in_eval_weight_sum], [in_post_weight_sum]
        # 1.3 Compute ratio for each random variable's marginal particles
        var_ratio = [(eval_val / post_val).clamp(0., 1.)    # Clamp values in range [0, 1]
                     for eval_val, post_val in zip(in_eval_var_weights, in_post_var_weights)]
        # Extra step: in extreme cases var_ratio will contain NaN elements. In such we replace NaN with value 1 so that
        #   the sampled mask will keep replacing the particles
        for ratio in var_ratio:
            ratio[torch.isnan(ratio)] = 1
        # 1.4 Instantiate Bernoulli distribution to flip coins for each random variable
        update_dist = [D.Bernoulli(ratio) for ratio in var_ratio]
        # 1.5 Flip coins to obtain binary mask for updating particle values
        # Unsqueeze an event dimension for the mask so that it is broadcastable to the particles
        update_mask = [dist.sample().unsqueeze(dim=-1) for dist in update_dist]
        update_mask_inv = [mask * -1 + 1 for mask in update_mask]       # Inverse mask
        # 1.6 Apply mask to obtain new posterior particles
        new_post_ptcl = [eval_ptcl * mask + post_ptcl * inv_mask
                         for eval_ptcl, post_ptcl, mask, inv_mask
                         in zip(in_eval_msg.particles, in_post_msg.particles, update_mask, update_mask_inv)]
        # 1.7 Instantiate new posterior belief message
        # MCMC particles message will have uniform weights and log sampling densities
        # Instantiate a uniform weight tensor here instead of using integer 1 because the latter will render the message
        #   an identity message, which may halt message propagation.
        uniform_log_densities = [torch.zeros(s_size, device=self.device) for s_size in self.s_shape]
        uniform_weight = torch.ones(self.b_shape + self.s_shape, device=self.device)
        self.post_msg_cache = Message(MessageType.Particles,
                                      batch_shape=self.b_shape, sample_shape=self.s_shape, event_shape=self.e_shape,
                                      particles=new_post_ptcl, weight=uniform_weight,
                                      log_densities=uniform_log_densities)

        # Step 2. Take a random walk step to generate new evaluation (candidate) particles message based on the above
        #   new posterior message
        self._random_walk()

    def precompute_check(self):
        if self.ld_in_post is None or \
           self.ld_in_eval is None or\
           self.ld_out_post is None or \
           self.ld_out_eval is None:
            raise NodeConfigurationError("Wrong configuration for node {}: All four of the linkdata must be specified "
                                         "for a WMFN-MCMC node. Only {} of them are specified."
                                         .format(self.name,
                                                 (self.ld_in_post is not None) +
                                                 (self.ld_in_eval is not None) +
                                                 (self.ld_out_post is not None) +
                                                 (self.ld_out_eval is not None)))

    @WMFN.compute_control
    def compute(self):
        """Send contents in cache to WMVN_OUT_POST and WMVN_OUT_EVAL respectively.

        """
        assert self.ld_out_post is not None and self.ld_out_eval is not None
        self.ld_out_post.write(self.post_msg_cache)
        self.ld_out_eval.write(self.eval_msg_cache)


class WMFN_VI(WMFN):
    """The Variational Inference Working Memory Factor Node.

    Memorizes and updates the predicate's knowledge across decision cycles. Hosts and maintains the associated
    KnowledgeServer instance to provide service to downstream nodes.

    Admits one incoming link from `WMVN_IN` that sends combined action message toward this predicate by the end of
    the decision cycle, as well as arbitrary number of other incoming links from parameter feed nodes and/or other parts
    of the graph that sends parameter messages. Special attribute therefore needs to be declared in the LinkData's
    attribute dictionary to distinguish which one sends "event" message from `WMVN_IN` and which ones send "param"
    messages from parameter feeds.

    If there are multiple incoming "param" labeled links, then the received parameter messages will be combined, using
    the default Message summation operation. See
    :ref:`Message class notes on arithmetic structures<message-arithmetic-structures-notes>`
    for more details.

    `modify()` should be called during modification phase of a cognitive cycle so that the message to be sent to
    downstream nodes during the decision phase of next cognitive cycle is computed herein. This includes gathering new
    parameters that are ready to be read from the incoming `param` linkdata at the end of the previous decision phase,
    as well as optionally drawing importance weighted particles w.r.t. the batched distributions that are instantiated
    from the newly gathered parameters. The latter behavior can be set by calling `set_draw()` method.

    In most cases, it is better to send as much information as possible to the connected conditional subgraph.
    Therefore, by default particles-drawing mode is on. However, there are also circumstances in which this behavior
    should be avoided, for instance when the Predicate is perceiving observations / evidence from PBFN, where the
    particle values should be determined by the observation feed rather than be drawn here at the LTMFN.

    Message cache is initialized to identity message.

    Parameters
    ----------
    name : str
        Name of this node.
    ks : KnowledgeServer
        The KnowledgeServer instance associated with the predicate.
    rel_var_list : iterable of Variable
        Iterable of relational variables, corresponding to the predicate's relational arguments.
    param_var : Variable
        The parameter variable.
    index_var_list : iterable of Variable
        Iterable of indexing variables.
    ran_var_list : iterable of Variable
        Iterable of random variables, corresponding to the predicate's random arguments.
    to_draw : bool, optional
        Initialize whether this LTMFN should be drawing particles in `init_msg()`. Defaults to ``True``.

    Attributes
    ----------
    ks
    rel_var_list
    param_var
    index_var_list
    ran_var_list
    to_draw
    b_shape : torch.Size
        The batch shape.
    p_shape : torch.Size
        The parameter shape (size).
    s_shape : torch.Size
        The sample shape.
    e_shape : torch.Size
        The event shape.
    msg_cache : Message
        The message cache. Set during modification phase, and sent during decision phase of the next cognitive cycle.
    """

    def __init__(self,
                 name: str,
                 ks: KnowledgeServer,
                 rel_var_list: IterableType[Variable],
                 param_var: Variable,
                 index_var_list: IterableType[Variable],
                 ran_var_list: IterableType[Variable],
                 to_draw: bool = True,
                 **kwargs):
        super(WMFN_VI, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Variational Inference Working Memory Factor Node Node"

        assert isinstance(ks, KnowledgeServer)
        assert isinstance(rel_var_list, Iterable) and \
               all(isinstance(v, Variable) and v.metatype is VariableMetatype.Relational for v in rel_var_list)
        assert isinstance(param_var, Variable) and param_var.metatype is VariableMetatype.Parameter
        assert isinstance(index_var_list, Iterable) and \
               all(isinstance(v, Variable) and v.metatype is VariableMetatype.Indexing for v in index_var_list)
        assert isinstance(ran_var_list, Iterable) and \
               all(isinstance(v, Variable) and v.metatype is VariableMetatype.Random for v in ran_var_list)

        self.ks = ks
        self.rel_var_list = tuple(rel_var_list)
        self.param_var = param_var
        self.index_var_list = tuple(index_var_list)
        self.ran_var_list = tuple(ran_var_list)
        self.to_draw = to_draw

        self.b_shape: torch.Size = torch.Size([v.size for v in self.rel_var_list])
        self.p_shape: torch.Size = torch.Size([self.param_var.size])
        self.s_shape: torch.Size = torch.Size([v.size for v in self.index_var_list])
        self.e_shape: torch.Size = torch.Size([v.size for v in self.ran_var_list])

        # Message cache
        self.msg_cache = Message.identity()

    def add_link(self, linkdata: LinkData):
        """Only admits one incoming and one outgoing event message link, the former should be connected from `WMVN_IN`
        and the later from `WMVN_OUT` (can be the same WMVN instance). However can admit multiple incoming
        parameter message link.

        In the incoming message links' attribute dictionary, a `"type"` keyed entry should be declared with value either
        `"param"` or '"event"'.

        Parameters
        ----------
        linkdata : LinkData
            The linkdata to be registered. The ``'type'`` key-ed attribute must present in the linkdata's attribute
            dict.

        Raises
        ------
        AssertionError
            If the new linkdata does not have identical batch shape and param shape.
        AssertionError
            If attempting to register more than one outgoing link.
        AssertionError
            If ``'type'`` key does not exist in the linkdata's attribute dict, or if the associated value is neither
            ``'event'`` nor ``'param'``.
        AssertionError
            If attempting to register more than one `event` incoming link.
        """
        # Check that the linkdata has correct batch shape and param shape.
        assert isinstance(linkdata, LinkData) and linkdata.msg_shape[:2] == (self.b_shape, self.p_shape)

        # Only admit one outgoing link and that must be WMVN. Check dimensions to be compatible with event message
        if not linkdata.to_fn:
            assert isinstance(linkdata.vn, WMVN), \
                "At {}: Attempting to register an outgoing link that is not connected to a WMVN.".format(self.name)
            assert len(self.out_linkdata) == 0, \
                "At {}: Attempting to register more than one outgoing linkdata.".format(self.name)
        # Can admit multiple incoming links. Check that link has special attribute declared.
        #   Check dimension for parameter link and event link respectively
        else:
            assert 'type' in linkdata.attr.keys(), \
                "At{}: Incoming link to a WMFN_VI must specify 'type' special attribute".format(self.name)
            assert linkdata.attr['type'] in ['event', 'param'], \
                "At{}: Incoming link to a WMFN_VI must have 'type' special attribute with value 'event' or 'param'" \
                    .format(self.name)
            if linkdata.attr['type'] == 'event':
                assert isinstance(linkdata.vn, WMVN), \
                    "At {}: Attempting to register an event type incoming link that is not connected to a WMVN" \
                        .format(self.name)
                assert len(list(ld for ld in self.in_linkdata if ld.attr['type'] == 'event')) == 0, \
                    "At {}: Attempting to register more than one incoming event type linkdata".format(self.name)
        super(WMFN_VI, self).add_link(linkdata)

    def set_draw(self, to_draw: bool):
        """Sets whether this LTMFN should draw particles in `init_msg()` and send `MessageType.Dual` type message, or
        not draw particles and send `MessageType.Parameter` message

        Parameters
        ----------
        to_draw : bool
            Whether to draw particles or not.
        """
        self.to_draw = to_draw

    def modify(self):
        """Draws particles and instantiate new message for next cognitive cycle.

        This method should be called during the modification phase. Parameter will be gathered from incoming `param`
        linkdata, and in the case of multiple incoming `param` linkdata the parameter tensors gathered will be combined.
        A new list of particles will then be drawn in the cognitive format by querying the given KnowledgeServer.

        Raises
        ------
        AssertionError
            If no `param` type incoming linkdata is found.
        AssertionError
            If messages read from incoming linkdata do not all contain parameters.
        """
        # Obtain parameters from incoming 'param' link.
        param_lds = list(ld for ld in self.in_linkdata if ld.attr['type'] == 'param')

        param_msgs = tuple(ld.read() for ld in param_lds)
        assert all(MessageType.Parameter in msg.type for msg in param_msgs), \
            "At {}: Expect all messages from incoming param type linkdata to contain parameters, but instead found " \
            "message types: {} from linkdata {}." \
                .format(self.name,
                        list(msg.type for msg in param_msgs if MessageType.Parameter not in msg.type),
                        list(
                            str(ld) for ld, msg in zip(param_lds, param_msgs) if MessageType.Parameter not in msg.type))

        # Combine parameter messages and extract the parameter tensor
        param = sum(param_msgs, Message.identity()).parameter

        if self.to_draw:
            # Query KnowledgeServer to extract components of a particle list.
            particles, log_densities = self.ks.draw_particles(param, self.b_shape, update_cache=True)
            log_prob = self.ks.surrogate_log_prob(param)

            # Instantiate a temporary message with uniform weight and use Message method to obtain re-weighted message
            tmp_msg = Message(MessageType.Dual,
                              batch_shape=self.b_shape, param_shape=self.p_shape,
                              sample_shape=self.s_shape, event_shape=self.e_shape,
                              parameter=param, particles=particles, weight=1, log_densities=log_densities)
            new_msg = tmp_msg.event_reweight(log_prob)
        else:
            # If not to draw particles, simply cache a Parameter message
            new_msg = Message(MessageType.Parameter,
                              batch_shape=self.b_shape, param_shape=self.p_shape,
                              parameter=param)

        self.msg_cache = new_msg

    def precompute_check(self):
        """The computable condition for a WMFN_VI is that there are at least one incoming linkdata and one outgoing
        linkdata.

        """
        if len(self.in_linkdata) == 0 or len(self.out_linkdata) == 0:
            raise NodeConfigurationError("Wrong configuration for node {}: a WMFN_VI expects at least one incoming "
                                         "linkdata and one outgoing linkdata to be computable. Found {} registered "
                                         "incoming linkdata and {} registered outgoing linkdata"
                                         .format(self.name, len(self.in_linkdata), len(self.out_linkdata)))

    @WMFN.compute_control
    def compute(self):
        """
        Send message in ``self.msg_cache`` to the connected `WMVN_OUT` node.

        Raises
        ------
        AssertionError
            If there are no connected outgoing linkdata.
        AssertionError
            If ``self.msg_cache`` is None. This means `init_msg()` were not called prior to the current decision phase
            which calls this method.
        """
        out_ld = self.out_linkdata[0]
        out_ld.write(self.msg_cache)


class PBFN(FactorNode):
    """Perception Buffer Factor Node.

    Receives perception / observation / evidence as particle list from `perceive()` and sends particles message to WMVN.

    Does not admit any incoming link. Only admits one outgoing link connecting to a WMVN.

    Perception is buffered, and will be latched to next cycle if no new observation is specified. An identity message
    can be used to flush the buffer without introducing actual perceptions.

    PBFN quiescence state:
        A PBFN reaches quiescence state if and only if it has been visited.

    Parameters
    ----------
    name : str
        Name of this node
    rel_var_list : iterable of Variable
        Used to infer the batch shape of the Predicate's knowledge. In a PBFN this is solely used to align the particle
        weight tensor in the outgoing message to the correct shape.
    event_shape : iterable of Variable
        Used to infer the event shape of any observation / evidence event particles, except for ``None`` observation.
        Its length should match the number of predicate random arguments. See more details in following `perceive()`
        method.

    Attributes
    ----------
    buffer : torch.Tensor
        The perceptual buffer.
    b_shape : torch.Size
        Set by `batch_shape`.
    e_shape : torch.Size
        Set by `event_shape`.
    """

    def __init__(self,
                 name: str,
                 rel_var_list: IterableType[Variable],
                 ran_var_list: IterableType[Variable],
                 **kwargs):
        assert isinstance(rel_var_list, Iterable) and \
               all(isinstance(v, Variable) and v.metatype is VariableMetatype.Relational for v in rel_var_list)
        assert isinstance(ran_var_list, Iterable) and \
               all(isinstance(v, Variable) and v.metatype is VariableMetatype.Random for v in ran_var_list)
        super(PBFN, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Perceptual Buffer Factor Node"

        self.b_shape: torch.Size = torch.Size([v.size for v in rel_var_list])
        self.e_shape: torch.Size = torch.Size([v.size for v in ran_var_list])
        # Perceptual buffer. Initialize to identity message
        self.buffer = Message.identity(MessageType.Dual)

    def add_link(self, linkdata: LinkData):
        """For PBFN, only one linkdata can be admitted, and it should be an outgoing linkdata connecting a WMVN node.

        """
        # Ensure that no incoming link and only one outgoing link connecting to a WMVN
        assert isinstance(linkdata, LinkData)
        assert not linkdata.to_fn, \
            "In {}: PBFN can only admit outgoing linkdata that connects to a WMVN. The given linkdata {} is incoming."\
            .format(self.name, linkdata)
        assert len(self.out_linkdata) == 0, \
            "In {}: PBFN only accepts one linkdata. Linkdata {} is already registered."\
            .format(self.name, self.out_linkdata[0])
        assert isinstance(linkdata.vn, WMVN), \
            "In {}: PBFN only admits linkdata that connects to a WMVN. The given linkdata {} connects to a {}."\
            .format(self.name, linkdata, type(linkdata.vn))
        assert linkdata.msg_shape[0] == self.b_shape and linkdata.msg_shape[3] == self.e_shape, \
            "In {}: Attempting to add a linkdata with incompatible message shapes. Expect batch shape {} and event " \
            "shape {}, but found {} and {} in the given linkdata respectively."\
            .format(self.name, self.b_shape, self.e_shape, linkdata.msg_shape[0], linkdata.msg_shape[3])
        super(PBFN, self).add_link(linkdata)

    def perceive(self,
                 obs: Union[torch.Tensor, IterableType[torch.Tensor], None] = None,
                 weight: Union[torch.Tensor, IterableType[torch.Tensor], None] = None,
                 mode: str = 'joint'):
        """Perceives a new piece of observation / evidence particle events, specified by `obs`, with optional weight
        specified by `weight`. instantiate and store the perception message in the perceptual buffer and send by
        `compute()`.

        If `obs` is ``None``, a ``MessageType.Dual`` type identity message will be instantiated. Otherwise, it is a
        ``MessageType.Particles`` message with particle values from `obs`, particles weight reflecting `weight` (uniform
        if `weight` is ``None``), and uniform log sampling densities.

        The particle weight tensor will be copied and expanded to include full batch dimension shape ``self.b_shape``.

        There are two perception mode: `joint` or `marginal`, specified by `mode`. This distinction makes a difference
        only for predicates with multiple random arguments:

        * When in `joint` mode, the observations should be list of joint particle events. Accordingly, `obs` must be a
          2D tensor with the last dimension being the joint event dimension having a size equal to the sum of all random
          variables' sizes (sum of ``self.e_shape``), and the first dimension being the sample (indexing) dimension.
          'weight' must a 1D tensor with its length equal to the size of `obs` 's first dimension.

          Internally, in order to conform to standard message format, this joint event tensor `obs` will be split
          into chunks along the event dimension according to the sizes of the random variables. Each chunk thus
          represents a list of marginal event values, corresponding to one of the random variables, on an axis of a
          high-dimensional event lattice in the joint event space. A weight tensor of the same dimensional shape will be
          created to annotate this event lattice, with entries equal to values found in `weight` for the slots that
          corresponds to those joint events in `obs`, and other entries set to NP_EPSILON (representing numerically
          stable 0 weight).

        * When in `marginal` mode, the observations are tuple of marginal events for each random variable, and it is
          assumed that these marginal events for each random variable are mutually independent. Accordingly,
          `obs` must be an ITERABLE of 2D tensors, with the last dimension size of each entry equal to the size of the
          corresponding random variable, in the order specified by ``self.e_shape``. Similarly, `weight` must also be
          an ITERABLE with the same length as `obs`, containing 1D tensors.

        The weights for duplicate events would be added together. However, this behavior should not be counted on and it
        is recommended to avoid duplicate entries in `obs` and instead use `weight` to signal the relative importance
        of each observation.

        Note that it is not necessary that `weight` is normalized and sums to 1; they will be automatically normalized
        when the outgoing message is instantiated. However `weight` must contain only positive values.

        This method should be called prior to the decision phase of a cognitive cycle for the perceived observation
        be sent to downstream nodes during the decision phase.

        Parameters
        ----------
        obs : iterable of torch.Tensor, torch.Tensor, or None. optional
            If not ``None``, must be a 2D tensor if `mode` is ``joint``, or an iterable of 2D tensors if `mode` is
            ``marginal``. Defaults to ``None``.
        weight : iterable of torch.Tensor, torch.Tensor, or None. optional
            If not ``None``, must be a 1D tensor if `mode` is ``joint``, or an iterable of 1D tensors if `mode` is
            ``marginal``. If `obs` is ``None``, this value will be ignored. Defaults to ``None``.
        mode : {"joint", "marginal"}
            The perception mode. Defaults to ``"joint"``.
        """
        assert mode in ['joint', 'marginal']
        if mode == 'joint':
            assert obs is None or (isinstance(obs, torch.Tensor) and obs.dim() == 2), \
                "In {}: when perceiving 'joint' observations, if specified, `obs` should be a 2-dimensional tensor."\
                .format(self.name)
            assert obs is None or obs.shape[-1] == sum(self.e_shape), \
                "In {}: in 'joint' perception mode, the size of the observation's event dimension must match the " \
                "sum of random variable sizes. Expect size {}, but found size {}." \
                .format(self.name, sum(self.e_shape), obs.shape[-1])

            assert weight is None or (isinstance(weight, torch.Tensor) and weight.dim() == 1), \
                "In {}: when perceiving 'joint' observations, if specified, `weight` should be a 1-dimensional tensor."\
                .format(self.name)
            assert weight is None or obs is None or weight.shape[0] == obs.shape[0], \
                "In {}: in 'joint' perception mode, when specified, the weight tensor must have same length as the " \
                "observation tensor's first dimension. Found weight length {}, and observation tensor's first " \
                "dimension size {}." \
                .format(self.name, weight.shape[0], obs.shape[0])
            assert weight is None or torch.all(weight > 0), \
                "In {}: in 'joint' perception mode, if specified, 'weight' must be a positive tensor. " \
                "Found minimum value {}."\
                .format(self.name, weight.min())

        else:
            assert obs is None or \
                (isinstance(obs, Iterable) and all(isinstance(o, torch.Tensor) and o.dim() == 2 for o in obs)), \
                "In {}: when perceiving 'marginal' observations, if specified, `obs` should be an iterable of " \
                "2-dimensional tensors.".format(self.name)
            assert obs is None or len(obs) == len(self.e_shape), \
                "In {}: in 'marginal' perception mode, the number of observations must match the number of random " \
                "variables. Found {} entries in `obs` but the predicate has {} random variables." \
                .format(self.name, len(obs), len(self.e_shape))
            assert obs is None or all(o.shape[-1] == self.e_shape[i] for i, o in enumerate(obs)), \
                "In {}: in 'marginal' perception mode, the size of each marginal observation's event dimension must " \
                "match the size of the corresponding random variable. Expect event sizes {}, but found sizes {}." \
                .format(self.name, list(self.e_shape), list(o.shape[-1] for o in obs))

            assert weight is None or \
                (isinstance(weight, Iterable) and all(isinstance(w, torch.Tensor) and w.dim() == 1 for w in weight)), \
                "In {}: when perceiving 'marginal' observations, if specified, `weight` should be an iterable of " \
                "1-dimensional tensors.".format(self.name)
            assert weight is None or len(weight) == len(self.e_shape), \
                "In {}: in 'marginal' perception mode, the number of weights must match the number of random " \
                "variables. Found {} entries in `weight` but the predicate has {} random variables." \
                .format(self.name, len(weight), len(self.e_shape))
            assert obs is None or weight is None or all(o.shape[0] == w.shape[0] for o, w in zip(obs, weight)), \
                "In {}: the first dimension size of each observation tensor in `obs` should match the length of the " \
                "corresponding weight tensor in `weight`. Found observation first dimension sizes {}, and weight " \
                "lengths {}." \
                .format(self.name, list(o.shape[0] for o in obs), list(w.shape[0] for w in weight))
            assert weight is None or all(torch.all(w > 0) for w in weight), \
                "In {}: in 'marginal' observation mode, if specified, all entries in the `weight` must be positive " \
                "tensors. Found the minimum values for each of the entry in `weight`: {}." \
                .format(self.name, list(w.min() for w in weight))

        # Set buffer to identity message and return directly if obs is None
        if obs is None:
            self.buffer = Message(MessageType.Dual, batch_shape=self.b_shape, parameter=0, weight=1)
            return

        obs = tuple(obs) if mode == 'marginal' else obs
        weight = tuple(weight) if mode == 'marginal' and weight is not None else weight

        # If mode is 'joint', split joint events and create sparse weight lattice
        if mode == 'joint':
            # split and find unique marginal event values (to get rid of duplicate event values)
            split_ptcl = torch.split(obs, self.e_shape, dim=-1)
            unique_ptcl, inverse_ids = zip(*tuple(torch.unique(p, return_inverse=True, dim=0) for p in split_ptcl))
            # Infer actual sample shape from unique particles
            s_shape = torch.Size([ptcl.shape[0] for ptcl in unique_ptcl])

            # Create lattice weight, sample shape only.
            # If weight is None, create a uniform weight list
            weight = torch.ones(obs.shape[0], dtype=torch.float, device=self.device) if weight is None else weight
            stacked_ids = torch.stack(inverse_ids, dim=1)
            ptcl_ids = tuple(torch.squeeze(i) for i in torch.split(stacked_ids, 1, dim=0))
            ptcl_weight = torch.ones(s_shape, device=self.device) * NP_EPSILON      # Init to zeros, with Eps for numerical stability
            for i, ptcl_id in enumerate(ptcl_ids):
                # If the index is a multi-dimensional index (if multiple rv), then cast the index tensor to a tuple
                #   in order to actually index a tensor
                ptcl_id = tuple(ptcl_id) if ptcl_id.numel() > 1 else ptcl_id
                ptcl_weight[ptcl_id] += weight[i]

            # Expand weight to include full batch dims
            ptcl_weight = ptcl_weight.view(torch.Size([1] * len(self.b_shape)) + s_shape).expand(self.b_shape + s_shape)
            # Clone the weight to itself to ensure every element of the tensor refers to a distinct memory location.
            ptcl_weight = ptcl_weight.clone()

            # Uniform log densities
            log_densities = tuple(torch.zeros(s_size, dtype=torch.float, device=self.device) for s_size in s_shape)

            perceptual_msg = Message(MessageType.Particles,
                                     batch_shape=self.b_shape, sample_shape=s_shape, event_shape=self.e_shape,
                                     particles=unique_ptcl, weight=ptcl_weight, log_densities=log_densities)

        # If mode is 'marginal', take cross product of weights if not None and init msg directly
        else:
            s_shape = torch.Size([o.shape[0] for o in obs])
            if weight is None:
                ptcl_weight = torch.ones(self.b_shape + s_shape, dtype=torch.float, device=self.device)
            else:
                expanded_log_weight = []
                for i in range(len(self.e_shape)):
                    view_dim = [1] * (len(self.e_shape) - 1)
                    view_dim.insert(i, -1)
                    expanded_log_weight.append(torch.log(weight[i].view(view_dim)))
                sum_log_weight = sum(expanded_log_weight)
                ptcl_weight = torch.exp(sum_log_weight)
                # Prepend batch dimensions
                ptcl_weight = ptcl_weight.view(torch.Size([1] * len(self.b_shape)) + s_shape)\
                    .expand(self.b_shape + s_shape)
                ptcl_weight = ptcl_weight.contiguous()

            # Uniform log densities
            log_densities = tuple(torch.zeros(s_size, dtype=torch.float) for s_size in s_shape)

            perceptual_msg = Message(MessageType.Particles,
                                     batch_shape=self.b_shape, sample_shape=s_shape, event_shape=self.e_shape,
                                     particles=obs, weight=ptcl_weight, log_densities=log_densities)

        # set buffer
        self.buffer = perceptual_msg

    @property
    def quiescence(self) -> bool:
        """Overrides default behavior so now PBFN's quiescence is determined by whether `compute()` has been called.

        """
        return self.visited

    @FactorNode.compute_control
    def compute(self):
        """Sends the contents in perceptual buffer to the connected WMVN.

        """
        assert len(self.out_linkdata) > 0
        out_ld = self.out_linkdata[0]
        out_ld.write(self.buffer)


class PSFN(FactorNode):
    # TODO: implement and test
    """Parameter Store Factor Node

    Stores the batched distribution parameters of the Predicate's knowledge and feeds them to LTMFN via DVN. Admits
    no incoming links and only one outgoing link to a DVN.

    This node is typically used in combination with a LTMFN, where a PSFN handles the actual storage and updates of the
    distribution parameter, and LTMFN uses this parameter to derive the event messages to be sent to WMVN gate node.

    By default, the parameter tensor is stored using a torch.nn.Parameter wrapper, so that any downstream processing
    and derived tensors automatically turns on gradient tracing.

    PSFN quiescence state:
        A PSFN reaches quiescence state if and only if it has been visited.

    The `quiescence` property is therefore overridden to conform to this definition.

    Parameters
    ----------
    batch_shape : torch.Size
        The batch shape of the distribution parameter.
    param_shape : torch.Size
        The parameter shape of the distribution parameter.
    init_param : torch.Tensor or None
        The initialized parameter tensor. If not None, should be a tensor of shape (batch_shape + param_shape). Defaults
        to None.

    Attributes
    ----------
    b_shape
    p_shape
    param : torch.nn.Parameter
        The parameter buffer. Should be a tensor of shape ``(batch_shape + param_shape)``. Value defaults to a zero
        tensor, when `init_param` is None during initialization.
    """

    def __init__(self, name, batch_shape, param_shape, init_param=None, **kwargs):
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(param_shape, torch.Size)
        assert init_param is None or \
               (isinstance(init_param, torch.Tensor) and init_param.shape == batch_shape + param_shape)
        super(PSFN, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Parameter Store Factor Node"

        self.b_shape = batch_shape
        self.p_shape = param_shape

        # Parameter buffer
        self.param = Parameter(init_param, requires_grad=True) if init_param is not None else \
            Parameter(torch.zeros(self.b_shape + self.p_shape, dtype=torch.float, device=self.device),
                      requires_grad=True)

        self.pretty_log["node type"] = "Parameter Store Factor Node"

    def reset_param(self, param):
        """Resets the parameter tensor.

        Parameters
        ----------
        param : torch.Tensor
            The tensor to be set as the parameter value. Should have shape ``(self.b_shape + self.p_shape)``.
        """
        assert isinstance(param, torch.Tensor) and param.shape == self.b_shape + self.p_shape

        self.param = Parameter(param, requires_grad=True)

    def add_link(self, linkdata):
        """For PSFN, only one outgoing link is admitted.

        """
        assert not linkdata.to_fn and len(self.out_linkdata) == 0
        super(PSFN, self).add_link(linkdata)

    def compute(self):
        """Instantiates and sends a Parameter message.

        """
        super(PSFN, self).compute()
        assert len(self.out_linkdata) > 0
        out_msg = Message(MessageType.Parameter, batch_shape=self.b_shape, param_shape=self.p_shape,
                          parameter=self.param)
        self.out_linkdata[0].write(out_msg)

    @property
    def quiescence(self):
        """Overrides so that PSFN's quiescence state is equivalent to its visited state

        """
        return self.visited


class LTMFN(FactorNode):
    # TODO: implement
    pass
