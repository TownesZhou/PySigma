"""
    All nodes related to a predicate subgraph
"""

import warnings
from collections.abc import Iterable
import torch
from torch.nn import Parameter
from defs import VariableMetatype, Variable, MessageType, Message, NP_EPSILON
from graphical.basic_nodes import LinkData, VariableNode, FactorNode
from utils import compatible_shape, KnowledgeServer


class WMVN(VariableNode):
    """Working Memory Variable Node.

    Gate node connecting predicate structure to conditionals.

    WMVN will attempt to combine incoming messages, regardless of whether they come from alpha terminals in Conditional
    subgraphs, or from other nodes in the Predicate subgraph. The combined message generally yields the semantics of
    marginal belief coming from a certain part of the graphical model, and is sent to downstream nodes for further
    processing.

    A KnowledgeServer instance associated with the belonging Predicate is required because occasionally log prob of
    particles needs to be queried.

    WMVN quiescence state:
        A WMVN reaches quiescence state if and only if **all** incoming linkdata do not contain new message.

    It is defined as such so that, although inefficiency may be induced due to WMVN having to fire multiple times while
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
    def __init__(self, name, ks, rel_var_list, param_var=None, index_var_list=None, ran_var_list=None, **kwargs):
        assert isinstance(ks, KnowledgeServer)
        super(WMVN, self).__init__(name, rel_var_list, param_var, index_var_list, ran_var_list, **kwargs)
        self.pretty_log["node type"] = "Working Memory Variable Node"

        # Distribution class the Predicate self belongs to is assuming
        self.ks = ks
        # Cache for temporarily saving computation result for combination
        self._cache = {}

    def compute(self):
        """Combine incoming message to this Predicate subgraph.

        Will attempt to combine incoming messages if there are multiple incoming links, subsuming the functionality of
        FAN node in Lisp Sigma. Combination can be carried out if messages are all Parameter type, or if there exist
        Particles type messages but all of them are homogeneous (sharing the same particle values as well as sampling
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
        super(WMVN, self).compute()
        # Relay message if only one incoming link
        if len(self.in_linkdata) == 1:
            in_ld = self.in_linkdata[0]
            msg = in_ld.read()
            assert isinstance(msg, Message)
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
                    param_msg, ptcl_msg = None, None
                    in_msgs = tuple(in_ld.read() for in_ld in in_lds)

                    assert all(MessageType.Parameter in msg.type or MessageType.Particles in msg.type
                               for msg in in_msgs), \
                        "At {}: Expect all incoming messages to contain either parameter or particles, or both, but " \
                        "the types of the incoming messages are: {}"\
                        .format(self.name, list(msg.type for msg in in_msgs))

                    # Only if all incoming messages contain parameters should we combine the parameters
                    if all(MessageType.Parameter in msg.type for msg in in_msgs):
                        param_msgs = tuple(msg.reduce_type(MessageType.Parameter) for msg in in_msgs)
                        param_msg = sum(param_msgs)

                    # If any incoming message contains particles, we should proceed to combine them
                    if any(MessageType.Particles in msg.type for msg in in_msgs):
                        # 1.a. Ensure all particle lists are homogeneous
                        particle_msgs = tuple(msg.reduce_type(MessageType.Particles) for msg in in_msgs
                                              if MessageType.Particles in msg.type)
                        particle_lds = tuple(ld for ld in in_lds if MessageType.Particles in ld.read().type)
                        tmp_msg, tmp_ld = particle_msgs[0], particle_lds[0]
                        assert isinstance(tmp_msg, Message)
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
                        ptcl_msg = sum(candidate_msgs)

                    # Compose components
                    if param_msg is not None and ptcl_msg is not None:
                        out_msg = Message.compose(param_msg, ptcl_msg)
                    elif param_msg is not None:
                        out_msg = param_msg
                    else:
                        out_msg = ptcl_msg

                # Cache result
                self._cache[in_lds] = out_msg
                # Send message
                out_ld.write(out_msg)

            # Clear cache
            self._cache = {}


class LTMFN(FactorNode):
    """Long-Term Memory Factor Node.

    Memorizes and updates the predicate's knowledge across decision cycles. Hosts and maintains the associated
    KnowledgeServer instance to provide service to downstream nodes.

    Admits one incoming link from `WMVN_IN` that contains combined action message toward this predicate by the end of
    the decision cycle, as well as arbitrary number of incoming links from parameter feeds and/or `WMFN` that contains
    parameter messages. Special attribute therefore needs to be declared in the linkdata's attribute dictionary to
    distinguish which one sends "event" message from `WMVN_IN` and which ones send "param" messages from parameter
    feeds.

    If there are multiple incoming "param" labeled links, then combination will be carried out by taking summation over
    the parameters. See
    :ref:`Message class notes on arithmetic structures<message-arithmetic-structures-notes>`
    for more details.

    `init_msg()` should be called during modification phase of a cognitive cycle so that the message to be sent to
    downstream nodes during the next cognitive cycle is prepared herein. This includes gathering new parameters that
    are ready to be read from the incoming `param` linkdata at the end of the previous decision phase, as well as
    optionally drawing importance weighted particles w.r.t. the batched distributions that are instantiated from the
    newly gathered parameters. The latter behavior can be set by calling `toggle_draw()` method. In general, it is
    expected to include as much information as possible in the outgoing message, and so drawing mode should be turned
    on, but there are also circumstances in which this behavior should be avoided, for instance when the Predicate is
    perceiving observations / evidence from PBFN, where the particle values should be determined by the observation feed
    rather than be drawn here at the LTMFN.

    Particles can optionally be drawn during modification phase of each cognitive cycle by calling `init_msg()` method,
    which internally calls the corresponding method of the KnowledgeServer instance to perform the Gibbs sampling
    procedure.

    .. todo::  Define LTMFN's quiescence behavior.

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
    def __init__(self, name, ks, rel_var_list, param_var, index_var_list, ran_var_list, to_draw=True, **kwargs):
        super(LTMFN, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Long-Term Memory Factor Node"

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

        self.b_shape = torch.Size([v.size for v in self.rel_var_list])
        self.p_shape = torch.Size([self.param_var.size])
        self.s_shape = torch.Size([v.size for v in self.index_var_list])
        self.e_shape = torch.Size([v.size for v in self.ran_var_list])

        # Message cache
        self.msg_cache = None

    def add_link(self, linkdata):
        """Only admits one incoming and one outgoing event message link, the former should be connected from `WMVN_IN`
        and the later from `WMVN_OUT` (can be the same WMVN instance). However can admit multiple incoming
        parameter message link.

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
            assert len(self.out_linkdata) == 0 and isinstance(linkdata.vn, WMVN), \
                "Attempting to register more than one outgoing linkdata."
        # Can admit multiple incoming links. Check that link has special attribute declared.
        #   Check dimension for parameter link and event link respectively
        else:
            assert 'type' in linkdata.attr.keys(), \
                "At{}: Incoming link to a LTMFN must specify 'type' special attribute".format(self.name)
            assert linkdata.attr['type'] in ['event', 'param'], \
                "At{}: Incoming link to a LTMFN must have 'type' special attribute with value 'event' or 'param'"\
                .format(self.name)
            if linkdata.attr['type'] == 'event':
                assert len(list(ld for ld in self.in_linkdata if ld.attr['type'] == 'event')) == 0,\
                    "At {}: Attempting to register more than one incoming event type linkdata"
        super(LTMFN, self).add_link(linkdata)

    def toggle_draw(self, to_draw):
        """Sets whether this LTMFN should draw particles in `init_msg()` and send `MessageType.Both` type message, or
        not draw particles and send `MessageType.Parameter` message

        Parameters
        ----------
        to_draw : bool
            Whether to draw particles or not.
        """
        self.to_draw = to_draw

    def init_msg(self):
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
        assert len(param_lds) > 0, \
            "At {}: Attempting to gather parameters, but no incoming param type linkdata found."

        param_msgs = tuple(ld.read() for ld in param_lds)
        assert all(MessageType.Parameter in msg.type for msg in param_msgs), \
            "At {}: Expect all messages from incoming param type linkdata to contain parameters, but instead found " \
            "message types: {} from linkdata {}."\
            .format(self.name, list(msg.type for msg in param_msgs), list(str(ld) for ld in param_lds))

        # Combine parameter messages and extract the parameter tensor
        param = sum(param_msgs).parameter

        if self.to_draw:
            # Query KnowledgeServer to extract components of a particle list.
            particles, log_densities = self.ks.draw_particles(param, self.b_shape, update_cache=True)
            log_prob = self.ks.surrogate_log_prob(param)

            # Instantiate a temporary message with uniform weight and use Message method to obtain re-weighted message
            tmp_msg = Message(MessageType.Both,
                              batch_shape=self.b_shape, param_shape=self.p_shape,
                              sample_shape=self.s_shape, event_shape=self.e_shape,
                              parameter=param, particles=particles, weight=1, log_densities=log_densities,
                              dist_info=self.ks.dist_info)
            new_msg = tmp_msg.event_reweight(log_prob)
        else:
            # If not to draw particles, simply cache a Parameter message
            new_msg = Message(MessageType.Parameter,
                              batch_shape=self.b_shape, param_shape=self.p_shape,
                              parameter=param,
                              dist_ino=self.ks.dist_info)

        self.msg_cache = new_msg

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
        super(LTMFN, self).compute()
        assert len(self.out_linkdata) > 0
        assert self.msg_cache is not None, \
            "At {}: No cached message at this LTMFN node to be send outward. init_msg() should first be called prior " \
            "to calling this method."
        out_ld = self.out_linkdata[0]
        out_ld.write(self.msg_cache)


class PSFN(FactorNode):
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
            Parameter(torch.zeros(self.b_shape + self.p_shape, dtype=torch.float), requires_grad=True)

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


class PBFN(FactorNode):
    """Perception Buffer Factor Node.

    Receives perception / observation / evidence as particle list from `perceive()` and sends particles message to WMVN.

    Does not admit any incoming link. Only admits one outgoing link connecting to a WMVN.

    Perception is buffered, and will be latched to next cycle if no new observation is specified. To cancel out the
    previously buffered observation, a ``None`` observation needs to be perceived.

    PBFN quiescence state:
        A PBFN reaches quiescence state if and only if it has been visited.

    The `quiescence` property is therefore overridden to conform to this definition.

    Parameters
    ----------
    name : str
        Name of this node
    batch_shape : torch.Size
        The batch shape of the Predicate's knowledge. In a PBFN this is solely used to align the particle weight tensor
        in the outgoing message to the correct shape.
    event_shape : torch.Size
        The event shape of any observation / evidence event particles, except for ``None`` observation. Its length
        should match the number of predicate random arguments. See more details in following `perceive()` method.

    Attributes
    ----------
    buffer : torch.Tensor
        The perceptual buffer. It is a 2D tensor whose last dimension is the event dimension with size equal to
        ``self.e_shape``.
    b_shape : torch.Size
        Set by `batch_shape`.
    e_shape : torch.Size
        Set by `event_shape`.
    """
    def __init__(self, name, batch_shape, event_shape, **kwargs):
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(event_shape, torch.Size)
        super(PBFN, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Perceptual Buffer Function Node"

        self.b_shape = batch_shape
        self.e_shape = event_shape
        # Perceptual buffer. Initialize to identity message
        self.buffer = Message(MessageType.Both, batch_shape=self.b_shape, parameter=0, weight=1)

    def perceive(self, obs=None, weight=None, mode='joint'):
        """Perceives a new piece of observation / evidence particle events, specified by `obs`, with optional weight
        specified by `weight`. instantiate the perception message to be sent by `compute()` and store it in the
        perceptual buffer.

        If `obs` is ``None``, a ``MessageType.Both`` type identity message will be instantiated. Otherwise, it is a
        ``MessageType.Particles`` message with particle values from `obs`, particles weight reflecting `weight` (uniform
        if `weight` is ``None``), and uniform log sampling densities.

        The particle weight tensor will be copied and expanded to include full batch dimension shape ``self.b_shape``.

        There are two perception mode: `joint` or `marginal`, specified by `mode`. This distinction makes a difference
        mostly for predicates with multiple random arguments:

        * When in `joint` mode, the observations should be list of joint particle events. Accordingly, `obs` must be a
          2D tensor with the last dimension being the joint event dimension having a size equal to the sum of all random
          variables' sizes (sum of ``self.e_shape``), and the first dimension being the sample (indexing) dimension.
          'weight' must a 1D tensor with its length equal to the size of `obs` 's first dimension.

          Internally, in order to conform to standard message format, this joint event tensor `obs` will be broken up
          into chunks along the event dimension according to the sizes of the random variables. Each chunk thus
          represents a list of marginal event values, corresponding to one of the random variables, on an axis of a
          high-dimensional event lattice in the joint event space. A weight tensor of the same dimensional shape will be
          created to annotate this event lattice, with entries equal to values found in `weight` for the slots that
          corresponds to those joint events in `obs`, and other entries set to NP_EPSILON (representing numerically
          stable 0 weight).

        * When in `marginal` mode, the observations are tuple of marginal events for each random variable, and the
          assumption is taken that these marginal events for each random variable are mutually independent. Accordingly,
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
        assert obs is None or (mode == 'joint' and isinstance(obs, torch.Tensor) and obs.dim() == 2) or \
            (mode == 'marginal' and isinstance(obs, Iterable) and
             all(isinstance(o, torch.Tensor) and o.dim() == 2 for o in obs))
        assert weight is None or \
            (mode == 'joint' and isinstance(weight, torch.Tensor) and weight.dim() == 1 and torch.all(weight > 0)) or \
            (mode == 'marginal' and isinstance(weight, Iterable) and
             all(isinstance(w, torch.Tensor) and w.dim() == 1 and torch.all(w > 0) for w in weight))

        # Set buffer to identity message and return directly if obs is None
        if obs is None:
            self.buffer = Message(MessageType.Both, batch_shape=self.b_shape, parameter=0, weight=1)
            return

        obs = tuple(obs) if isinstance(obs, Iterable) else obs
        weight = tuple(weight) if isinstance(weight, Iterable) else weight
        # Check event size
        if mode == 'joint':
            assert obs.shape[-1] == sum(self.e_shape), \
                "At {}: in 'joint' perception mode, the size of the observation's event dimension must match the " \
                "sum of random variable sizes. Expect {}, but encountered {}."\
                .format(self.name, sum(self.e_shape), obs.shape[-1])
            assert weight is None or weight.shape[0] == obs.shape[0], \
                "At {}: in 'joint' perception mode, when specified, the weight tensor must have same length as the " \
                "observation tensor's first dimension. Found weight length {}, and observation tensor's first " \
                "dimension size {}"\
                .format(self.name, weight.shape[0], obs.shape[0])
        else:
            assert len(obs) == len(self.e_shape), \
                "At {}: in 'marginal' perception mode, the number of observations must match the number of random " \
                "variables. Found {} entries in `obs` but {} entries in `self.e_shape`." \
                .format(self.name, len(obs), len(self.e_shape))
            assert len(obs) == len(weight), \
                "At {}: in 'marginal' perception mode, the number of observations must match the number of weights. " \
                "Found {} entries in `obs` but {} entries in `weight`." \
                .format(self.name, len(obs), len(weight))
            assert all(o.shape[-1] == self.e_shape[i] for i, o in enumerate(obs)), \
                "At {}: in 'marginal' perception mode, the size of each marginal observation's event dimension must " \
                "match the size of the corresponding random variable. Expect event sizes {}, but encountered {}."\
                .format(self.name, list(self.e_shape), list(o.shape[-1] for o in obs))
            assert all(o.shape[0] == w.shape[0] for o, w in zip(obs, weight)), \
                "At {}: the first dimension size of each observation tensor in `obs` should match the length of the " \
                "corresponding weight tensor in `weight`. Found observation first dimension sizes {}, and weight " \
                "lengths {}."\
                .format(self.name, list(o.shape[0] for o in obs), list(w.shape[0] for w in weight))

        # If mode is 'joint', split joint events and create sparse weight lattice
        if mode == 'joint':
            s_shape = torch.Size([obs.shape[0]] * len(self.e_shape))
            # split and find unique marginal event values
            split_ptcl = torch.split(obs, self.e_shape, dim=-1)
            unique_ptcl, inverse_ids = zip(*tuple(torch.unique(p, return_inverse=True, dim=0) for p in split_ptcl))

            # Create lattice weight, sample shape only.
            # If weight is None, create a uniform weight list
            weight = torch.ones(obs.shape[0], dtype=torch.float) if weight is None else weight
            stacked_ids = torch.stack(inverse_ids, dim=1)
            ptcl_ids = tuple(torch.squeeze(i) for i in torch.split(stacked_ids, 1, dim=0))
            ptcl_weight = torch.ones(s_shape) * NP_EPSILON
            for i, ptcl_id in enumerate(ptcl_ids):
                ptcl_weight[ptcl_id] = weight[i]

            # Expand weight to include full batch dims
            ptcl_weight = ptcl_weight.view(torch.Size([1] * len(self.b_shape)) + s_shape).expand(self.b_shape + s_shape)

            # Uniform log densities
            log_densities = tuple(torch.zeros(obs.shape[0], dtype=torch.float),) * len(self.e_shape)

            perceptual_msg = Message(MessageType.Particles,
                                     batch_shape=self.b_shape, sample_shape=s_shape, event_shape=self.e_shape,
                                     particles=unique_ptcl, weight=ptcl_weight, log_densities=log_densities)

        # If mode is 'marginal', take cross product of weights if not None and init msg directly
        else:
            s_shape = torch.Size([o.shape[0] for o in obs])
            if weight is None:
                ptcl_weight = 1
            else:
                expanded_log_weight = []
                for i in range(len(self.e_shape)):
                    view_dim = [1] * (len(self.e_shape) - 1)
                    view_dim.insert(i, -1)
                    expanded_log_weight.append(torch.log(weight[i].view(view_dim)))
                sum_log_weight = sum(expanded_log_weight)
                ptcl_weight = torch.exp(sum_log_weight)

            # Uniform log densities
            log_densities = tuple(torch.zeros(obs.shape[0], dtype=torch.float), ) * len(self.e_shape)

            perceptual_msg = Message(MessageType.Particles,
                                     batch_shape=self.b_shape, sample_shape=s_shape, event_shape=self.e_shape,
                                     particles=obs, weight=ptcl_weight, log_densities=log_densities)

        # set buffer
        self.buffer = perceptual_msg

    def add_link(self, linkdata):
        """For PBFN, only one linkdata can be admitted, and it should be an outgoing linkdata connecting a WMVN node.

        """
        # Ensure that no incoming link and only one outgoing link connecting to a WMVN
        assert isinstance(linkdata, LinkData)
        assert not linkdata.to_fn
        assert len(self.out_linkdata) == 0
        assert isinstance(linkdata.vn, WMVN)
        super(PBFN, self).add_link(linkdata)

    def compute(self):
        """Sends the contents in perceptual buffer to the connected WMVN.

        """
        super(PBFN, self).compute()

        assert len(self.out_linkdata) > 0
        out_ld = self.out_linkdata[0]
        out_ld.write(self.buffer)

    @property
    def quiescence(self):
        """Overrides default behavior so now PBFN's quiescence is determined by whether `compute()` has been called.

        """
        return self.visited


class WMFN(FactorNode):
    """Working Memory Factor Node.

    Effectively a buffer node that contains a memory buffer, WMFN mixes the incoming message with its stored memory by
    taking a weight sum during the modification phase, updates its memory with the result, and sends this updated memory
    during the decision phase at the next cognitive cycle. The first two steps are performed by `update_memory()`,
    whereas sending the message is, as always, performed by `compute()`.

    The weighted sum mixture behavior can be described as follows::

        new_memory = new_msg + (1 - decay_rate) * old_memory

    where ``decay_rate`` is a real number in range [0, 1]. The vector addition and scalar multiplication for messages
    of different types are mathematically defined by the Message class. See
    :ref:`Message class notes on arithmetic structures<message-arithmetic-structures-notes>` for more details.

    The incoming message will always be cloned before weighted sum update is performed. This is to prevent
    any components of the memory message from in-place change by some parts elsewhere in the graph.

    Admits only one incoming and one outgoing links. Note that WMFN does not check the message shape of messages and
    memory contents. These should be guaranteed compatible by linkdata and neighboring nodes.

    The `check_quiesce()` method is overridden so that PSFN's quiescence state is determined by whether this node is
    visited during the decision phase, i.e., whether `compute()` is called.

    Parameters
    ----------
    name : str
        The name of this node
    decay_rate : float
        The decay rate of the memory contents.

    Attributes
    ----------
    decay_rate
    memory
    """
    def __init__(self, name, decay_rate=1, **kwargs):
        assert isinstance(decay_rate, (float, int)) and 0 <= decay_rate <= 1
        super(WMFN, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Working Memory Function Node"

        self.decay_rate = decay_rate
        # memory buffer. Initialized to a universal identity message
        self.memory = Message.identity()

    def add_link(self, linkdata):
        """WMFN only admits one incoming link and one outgoing link.

        """
        if linkdata.to_fn:
            assert len(self.in_linkdata) == 0
        else:
            assert len(self.out_linkdata) == 0

        super(WMFN, self).add_link(linkdata)

    def update_memory(self):
        """Updates the content in memory using message from incoming link.

        This step should be called during the modification phase.
        """
        assert len(self.in_linkdata) > 0
        in_ld = self.in_linkdata[0]
        # Clone incoming message
        new_msg = in_ld.read().clone()
        assert compatible_shape(self.memory, new_msg), \
            "At {}: found incompatible message shapes. The new message has shape {}, whereas the current working " \
            "memory content has shape {}".format(self.name, new_msg.shape, self.memory.shape)

        # perform weighted sum update
        self.memory = new_msg + self.memory * (1 - self.decay_rate)

    def compute(self):
        """Sends memory content toward outgoing link

        """
        super(WMFN, self).compute()
        assert len(self.out_linkdata) > 0
        self.out_linkdata[0].write(self.memory)

    def check_quiesce(self):
        """Overrides so that quiescence for WMFN is equivalent to visited

        """
        self.quiescence = self.visited
        return self.quiescence