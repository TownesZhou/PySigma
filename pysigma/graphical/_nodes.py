"""
    All kinds of nodes in the graphical architecture
"""
import copy
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
import torch
from torch.nn import Parameter
from torch.distributions import Transform
from torch.distributions.constraints import Constraint
from defs import VariableMetatype, Variable, MessageType, Message, NP_EPSILON
from utils import KnowledgeServer, compatible_shape
from structures import VariableMap, Summarization


"""
    Basic data structures and abstract node classes
        - LinkData
        - Node
        - FactorNode
        - VariableNode
"""


class LinkData:
    """Identifies the data of a directed link between a factor node and a variable node. Stores intermediate messages in
    its message memory.

    Note that links are directional, and two of such links should be specified with opposite directions to represent
    a bidirectional link between a factor node and a variable node.

    During construction of the graph, its instance should be passed to NetworkX methods as the edge data to instantiate
    an edge.

    Parameters
    ----------
    vn : VariableNode
        VariableNode instance that this link is incident to.
    fn : FactorNode
        FactorNode instance that this link is incident to.
    to_fn : bool
        True if this link is pointing toward the factor node.
    msg_shape : tuple of torch.Size
        The shape of the message to carry. Used for sanity check of message shapes. Should be in the format
        ``(batch_shape, param_shape, sample_shape, event_shape)``. An empty shape ``torch.Size([])`` should be used as
        the default none shape.
    epsilon : float, optional
        Epsilon upper bound for checking message difference.

    Attributes
    ----------
    memory : Message or None
        The message memory buffer.
    new : bool
        Indicates if this link-data has received a new message in the current decision phase.
    vn : VariableNode
        The incident variable node.
    fn : FactorNode
        The incident factor node.
    msg_shape : tuple of torch.Size
        The allowable message shape, in the format ``(batch_shape, param_shape, sample_shape, event_shape)``.
    to_fn : bool
        Indicates if this link-data is pointing towards a factor node.
    epsilon : float
        Epsilon upper bound for checking message difference.
    attr : dict
        Additional special attributes specified via `kwargs` in the constructor.
    pretty_log : dict
        Pretty logging for front-end visualization.
    """
    def __init__(self, vn, fn, to_fn, msg_shape, epsilon=10e-5, **kwargs):
        """
        :param vn:      VariableNode instance that this link is incident to
        :param fn:      FactorNode instance that this link is incident to
        :param to_fn:   True/False indicating whether this link is pointing toward a factor node
        :param msg_shape    Fixed message shape that this linkdata will carry. Used for checking dimensions
                            For Parameter message, should be (batch_shape + param_shape)
                            For particles message, should be (sample_shape + batch_shape + event_shape)
        :param epsilon:     epsilon upper bound for checking message difference using KL divergence
        """
        assert isinstance(vn, VariableNode)
        assert isinstance(fn, FactorNode)
        assert isinstance(msg_shape, tuple) and len(msg_shape) == 4 and \
            all(isinstance(s, torch.Size) for s in msg_shape)
        assert isinstance(to_fn, bool)
        assert isinstance(epsilon, float)

        # Message memory. Initialized to the identity message
        self.memory = Message.identity()
        # Whether this message is new, and haven't been read by recipient node. Of type bool
        self.new = False
        # Incident nodes and their variable list
        self.vn = vn                # Of type VariableNode
        self.fn = fn                # Of type FactorNode
        self.msg_shape = msg_shape
        # Link direction. Whether pointing toward a factor node. Of type bool
        self.to_fn = to_fn
        # Threshold of KL-divergence metric to measure whether a candidate message is different from the existing one
        self.epsilon = epsilon
        # Reserved field for additional attributes. Arbitrary type
        self.attr = kwargs
        # Pretty log
        self.pretty_log = {}

    def __str__(self):
        # Override for pretty debugging and printing
        fn_name = self.fn.name
        vn_name = self.vn.name
        if self.to_fn:
            return vn_name + " --> " + fn_name
        return fn_name + " --> " + vn_name

    def reset_shape(self, msg_shape):
        """Reset shape for the Message

        Parameters
        ----------
        msg_shape : tuple of torch.Size
            The target message shape, in the format ``(batch_shape, param_shape, sample_shape, event_shape)``.  An empty
            shape ``torch.Size([])`` should be used as the default none shape.

        Warnings
        --------
        This method will clear the memory buffer ``self.memory`` and set ``self.new`` to False.
        """
        assert isinstance(msg_shape, tuple) and len(msg_shape) == 4 and \
            all(isinstance(s, torch.Size) for s in msg_shape)
        self.msg_shape = msg_shape
        self.memory = None
        self.new = False

    def write(self, new_msg, check_diff=True, clone=False):
        """Writes to the link message memory with the new message specified via `new_msg`. Once a new message is
        written, ``self.new`` will be set to ``True``.

        The message shape ``new_msg.shape`` will first be checked against ``self.msg_shape`` to ensure that the message
        is compatible in shape. See `compatible_shape()` for more details.

        If `check_diff` is ``True``, will check if the new message is different from the existing one before
        replacing the existing with the new one.

        If `clone` is ``True``, then will first clone `new_msg` and store the cloned message in the memory buffer.

        Parameters
        ----------
            new_msg : Message
                The new message to be stored in this link-data.
            check_diff : bool, optional
                Whether to compare the difference between stored message against `new_msg` and decide whether to receive
                the new message and set ``self.new`` to ``True``.
            clone : bool, optional
                Whether to clone `new_msg` before storing it in the memory buffer.

        Raises
        ------
        AssertionError
            If new message's shape is not compatible.

        Notes
        -----
        Messages will be deemed different in the following cases:

            1. If they are of different types,
            2. If new message has ``MessageType.Undefined`` type,
            3. If they both have parameters and the batch average L2 distance between the two parameter tensors
               is larger than ``epsilon``,
            4. If they both have particles and either their particle value tensors or their particle log sampling
               tensors are different.
            5. If they both have particles, and they possess the same particles value tensors and same sampling log
               density tensors, but the batch average cosine similarity distance between the two particle weight tensors
               is larger than ``epsilon``.

        .. note::

            When `self` and `other` have type ``MessageType.Both``, the parameters will be chosen over the particles to
            compare message difference.

        .. note::

           If want to set a new message of a different message type than the current memory, make sure reset_shape()
           is first called so that shape check works for the new message.
        """
        assert isinstance(new_msg, Message)
        # Check new message shape
        assert compatible_shape(self.msg_shape, new_msg.shape), \
            "At {}: Attempt to write a message with incompatible shape. Expect message shape {}, instead encountered " \
            "{}".format(str(self), self.msg_shape, new_msg.shape)

        # Will replace the memory immediately if any one of the following conditions is met:
        #   - self.memory is None
        #   - check_diff is False
        #   - new message has different type
        #   - new message has Undefined type
        #   - messages have particles and new message has different particle values and/or sampling log densities
        if self.memory is None or check_diff is False or new_msg.type != self.memory.type or \
                new_msg.type == MessageType.Undefined or\
                (MessageType.Particles in new_msg.type and not self.memory.same_particles_as(new_msg)):
            self.memory = new_msg.clone() if clone is True else new_msg
            self.new = True
            return

        # Otherwise, check difference by KL-divergence
        if MessageType.Parameter in self.memory.type:
            # For Parameter message, compare batch average L2 distance
            # Parameter tensor has shape (batch_shape + param_shape), with param_shape of length 1
            # L2 distance is computed along param_shape dimension, i.e., the -1 dimension
            val = self.memory.diff_param(new_msg)
        else:
            # For Particles message, compare batch average cosine similarity distance
            # Particle weights has shape (batch_shape + sample_shape)
            # cosine similarity distance is computed along sample_shape dimension. i.e., the 0 dimension
            val = self.memory.diff_weight(new_msg)

        # Compare distance value with epsilon
        if val > self.epsilon:
            self.memory = new_msg.clone() if clone is True else new_msg
            self.new = True

    def read(self, clone=False):
        """Returns the current content stored in memory. Set ``self.new`` to ``False`` to indicate this link message
        has been read in the current decision phase.

        Parameters
        ----------
        clone : bool
            Whether to return a cloned message of the memory.

        Returns
        -------
            The current memory message.
        """
        self.new = False
        msg = self.memory.clone() if clone is True else self.memory
        return msg


class Node(ABC):
    """The base class for all nodes in PySigma graphical architecture.

    Declares common attributes between `FactorNode` and `VariableNode`, for example a flag indicating whether this node
    has reach quiescence, i.e., whether no new message shall be sent via adjacent links in current cycle. During
    construction of the graph, its instance should be passed to `NetworkX` methods to instantiate a node.

    Parameters
    ----------
    name : str
        Name of the node.

    Attributes
    ----------
    name : str
        Name of the node.
    visited : bool
        Indicates whether this node has been visited at all, i.e., `compute()` method being called at the current cycle.
        Default to ``False`` at the start of the cycle.
    in_linkdata : list of LinkData
        List of incoming linkdata.
    out_linkdata : list of LinkData
        List of outgoing linkdata.
    log : dict
        Internal log
    pretty_log : dict
        Pretty log for beautiful front-end visualization.
    """

    def __init__(self, name):
        self.name = name
        # Flag indicating whether this node has been visited (compute() method called) during a decision cycle
        self.visited = False

        # List of LinkData of those link connecting to this node, incoming and outgoing ones respectively, from
        #   which we retrieve messages
        self.in_linkdata = []
        self.out_linkdata = []

        # Global logging info
        self.log = {}
        self.pretty_log = {}  # Log info to display at the GUI

    def __str__(self):
        # override to provide the node's name as its string representation
        return self.name

    @property
    def quiescence(self):
        """Indicates whether this node has reached quiescence state, and no further message update computations, i.e.,
        `compute()`, is necessary at the decision phase of the current cognitive cycle. This property thus plays an
        important role in deciding the node traversing schedule of the graphical architecture.

        The default behavior is to check whether any incoming linkdata contains a new unread message. If so, it is
        decided that this node has breached quiescence and should carry out `compute()`, otherwise this nodes stays
        quiesced.

        This property should be overridden by subclass that desires a different quiescence checking behavior. For
        example for certain nodes, `compute()` may only be called once *all* incoming linkdata contain new messages, not
        *any*, and for other nodes their quiescence states depend only on whether `compute()` has ever been called
        within the current decision phase.
        """
        return any(in_ld.new for in_ld in self.in_linkdata)

    @abstractmethod
    def add_link(self, linkdata):
        """Adding linkdata connecting to this node.

        """
        raise NotImplementedError

    @abstractmethod
    def compute(self):
        """Compute method to be called to propagate messages during decision phases.

        Note that ``super()`` must be called within `compute()` method in any child class, because all abstract
        node-level statistics logging is taken care of herein.

        The default quiescence behavior for `compute()` is to return directly if `self.quiescence` is ``True``,
        without logging anything or performing any further computation. Note that such behavior may or may not be
        desired by child node class.
        """
        # Return directly if quiesced
        if self.quiescence:
            return
        # General logging regarding node computation statistics to be added here
        self.visited = True

    def reset_state(self):
        """Clears and resets the node's message propagation statistics to prepare logging for the imminent decision
        phase computation.

        .. note::  This method should be called prior to the first call to compute() at every cognitive cycle.
        """
        # Clears visited state
        self.visited = False


class FactorNode(Node, ABC):
    """Factor node abstract base class.

    Guarantees that all incident nodes are variable nodes.
    """
    def add_link(self, linkdata):
        """Add a linkdata connecting to a variable node

        Parameters
        ----------
        linkdata : LinkData
            The incident linkdata to be registered.
        """
        assert isinstance(linkdata, LinkData)
        assert linkdata.fn is self
        if linkdata in self.in_linkdata + self.out_linkdata:
            return

        if linkdata.to_fn:
            self.in_linkdata.append(linkdata)
        else:
            self.out_linkdata.append(linkdata)


class VariableNode(Node, ABC):
    """Variable node abstract base class.

    Guarantees that all incident nodes are factor nodes

    Parameters
    ----------
    name : str
        Name of this variable node.
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
    rel_vars : tuple of Variable
        Tuple of relational variables, specified by `rel_var_list`.
    param_var : Variable
        Parameter variable, specified by `param_var`. Defaults to ``None`` if `param_var` is not specified.
    index_vars : tuple of Variable
        Tuple of indexing variables, specified by `index_var_list`. Defaults to ``None`` if `index_var_list` is not
        specified.
    ran_vars : tuple of Variable
        Tuple of random variables, specified by `ran_var_list`. Default sto ``None`` if `ran_var_list` is not specified.
    b_shape : torch.Size
        Batch dimension sizes. Inferred from `rel_vars`.
    p_shape : torch.Size
        Parameter dimension sizes. Inferred from `param_var`. Defaults to ``torch.Size([])``.
    s_shape : torch.Size
        Sample dimension sizes. Inferred from `index_vars`. Defaults to ``torch.Size([])``.
    e_shape : torch.Size
        Event dimension sizes. Inferred from `ran_vars`. Defaults to ``torch.Size([])``.
    """
    def __init__(self, name, rel_var_list, param_var=None, index_var_list=None, ran_var_list=None):
        super(VariableNode, self).__init__(name)
        assert isinstance(rel_var_list, Iterable) and \
            all(isinstance(v, Variable) and v.metatype is VariableMetatype.Relational for v in rel_var_list)
        assert param_var is None or \
            (isinstance(param_var, Variable) and param_var.metatype == VariableMetatype.Parameter)
        assert index_var_list is None or \
            (isinstance(index_var_list, Iterable) and
             all(isinstance(v, Variable) and v.metatype == VariableMetatype.Indexing) for v in index_var_list)
        assert ran_var_list is None or \
            (isinstance(ran_var_list, Iterable) and
             all(isinstance(v, Variable) and v.metatype == VariableMetatype.Random) for v in ran_var_list)
        assert (index_var_list is None) is (ran_var_list is None)

        self.rel_vars = tuple(rel_var_list)
        self.param_var = param_var
        self.index_vars = tuple(index_var_list) if index_var_list is not None else None
        self.ran_vars = tuple(ran_var_list) if ran_var_list is not None else None

        self.b_shape = torch.Size([v.size for v in self.rel_vars])
        self.p_shape = torch.Size([self.param_var.size]) if self.param_var is not None else torch.Size([])
        self.s_shape = torch.Size([v.size for v in self.index_vars]) if self.index_vars is not None else torch.Size([])
        self.e_shape = torch.Size([v.size for v in self.ran_vars]) if self.ran_vars is not None else torch.Size([])

    def add_link(self, linkdata):
        """Register the LinkData connecting a factor node to this variable node.

        Checks that the preset message shape specified in `linkdata` agrees with the inferred message shape at this
        variable node.

        Parameters
        ----------
        linkdata : LinkData
            The incident linkdata to be registered.

        Raises
        ------
        AssertionError
            If the linkdata to be added does not impose the same message shape restriction as this variable node does.
        """
        assert isinstance(linkdata, LinkData)
        assert linkdata.vn is self
        assert linkdata.msg_shape == (self.b_shape, self.p_shape, self.s_shape, self.e_shape), \
            "At {}: The linkdata to be registered with the current node does not impose the same message shape " \
            "restriction as this node does. Current node's shape: (batch_shape, param_shape, sample_shape, " \
            "event_shape) = {}. Found linkdata's shape: {}"\
            .format(self.name, (self.b_shape, self.p_shape, self.s_shape, self.e_shape), linkdata.msg_shape)
        if linkdata in self.in_linkdata + self.out_linkdata:
            return

        if linkdata.to_fn:
            self.out_linkdata.append(linkdata)
        else:
            self.in_linkdata.append(linkdata)


"""
    Utility structural nodes
        - DFN
        - DVN
"""


class DFN(FactorNode):
    """Default (Dummy) Factor Node.

    No special computation. Simply relays message to one or multiple incident variable nodes. Requires that incident
    variable nodes share the same variables. Only admits one incoming link but can connect with multiple outgoing links.

    Since all incident variable nodes should share the same variables, these variables will also be registered in the
    attributes.

    Parameters
    ----------
    name : str
        Name of this node.

    Attributes
    ----------
    rel_vars : tuple of Variable
        Tuple of relational variables.
    param_var : Variable
        Parameter variable.
    index_vars : tuple of Variable
        Tuple of indexing variables.
    ran_vars : tuple of Variable
        Tuple of random variables.
    """
    def __init__(self, name):
        super(DFN, self).__init__(name)
        self.pretty_log["node type"] = "Default Factor Node"

        # Since all incident nodes should have the same variable list, we can therefore log it here as an attribute
        self.rel_vars = None
        self.param_var = None
        self.index_vars = None
        self.ran_vars = None

    def add_link(self, linkdata):
        """Checks that all variable nodes on the other side of the linkdata share the same set of variables. Infer
        attribute values from the connected variable nodes' variables.

        .. note::

           Only one incoming linkdata can be registered.

        Parameters
        ----------
        linkdata : LinkData
            The linkdata to be registered.

        Raises
        ------
        AssertionError
            If attempting to register more than one incoming linkdata.
        AssertionError
            If the variable node on the other side of the specified linkdata do not share the same set of variables as
            other variable nodes in those linkdata that are already registered.
        """
        assert isinstance(linkdata, LinkData)
        # Make sure no more than on incoming link
        assert not linkdata.to_fn or len(self.in_linkdata) == 0
        # Also make sure incident variable nodes' var_list agree with each other
        if self.rel_vars is None:
            self.param_var = linkdata.vn.param_var
            self.index_vars = linkdata.vn.rel_vars
            self.ran_vars = linkdata.vn.ran_vars
        else:
            assert self.rel_vars == linkdata.vn.rel_vars
            assert self.param_var == linkdata.vn.param_var
            assert self.index_vars == linkdata.vn.index_vars
            assert self.ran_vars == linkdata.vn.ran_vars

        super(DFN, self).add_link(linkdata)

    def compute(self):
        """Relay untempered message to downstream variable nodes.

        """
        super(DFN, self).compute()
        in_ld = self.in_linkdata[0]
        msg = in_ld.read()
        for out_ld in self.out_linkdata:
            out_ld.write(msg)


class DVN(VariableNode):
    """Default (Dummy) Variable Node.

    No special computation. Simply relay message to one or multiple factor nodes. Only admits one incoming link but can
    connect with multiple outgoing links.

    Parameters
    ----------
    name : str
        Name of this variable node.
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
    """
    def __init__(self, name, rel_var_list, param_var=None, index_var_list=None, ran_var_list=None):
        super(DVN, self).__init__(name, rel_var_list, param_var, index_var_list, ran_var_list)
        self.pretty_log["node type"] = "Default Variable Node"

    def add_link(self, linkdata):
        """Guarantees that no more than on incoming link is registered.

        Parameters
        ----------
        linkdata : LinkData
            The linkdata to be registered.

        Raises
        ------
        AssertionError
            If attempting to register more than one incoming linkdata.
        """
        assert isinstance(linkdata, LinkData)
        # Make sure no more than on incoming link
        assert linkdata.to_fn or len(self.in_linkdata) == 0

        super(DVN, self).add_link(linkdata)

    def compute(self):
        """Relay untempered message to downstream factor nodes.

        """
        super(DVN, self).compute()
        in_ld = self.in_linkdata[0]
        msg = in_ld.read()
        for out_ld in self.out_linkdata:
            out_ld.write(msg)


"""
    Nodes relating to Predicate subgraph structures
        - WMVN
        - LTMFN
        - PBFN
        - WMFN
"""


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
        A WMVN reaches quiescence state if and only if **any** incoming linkdata contains new message.

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
    def __init__(self, name, ks, rel_var_list, param_var=None, index_var_list=None, ran_var_list=None):
        assert isinstance(ks, KnowledgeServer)
        super(WMVN, self).__init__(name, rel_var_list, param_var, index_var_list, ran_var_list)
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
    def __init__(self, name, ks, rel_var_list, param_var, index_var_list, ran_var_list, to_draw=True):
        super(LTMFN, self).__init__(name)
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
    def __init__(self, name, batch_shape, param_shape, init_param=None):
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(param_shape, torch.Size)
        assert init_param is None or \
            (isinstance(init_param, torch.Tensor) and init_param.shape == batch_shape + param_shape)
        super(PSFN, self).__init__(name)
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
    def __init__(self, name, batch_shape, event_shape):
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(event_shape, torch.Size)
        super(PBFN, self).__init__(name)
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
    def __init__(self, name, decay_rate=1):
        assert isinstance(decay_rate, (float, int)) and 0 <= decay_rate <= 1
        super(WMFN, self).__init__(name)
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


"""
    Nodes relating to Conditional Alpha subgraph structures
        - AlphaFactorNode: RMFN, ESFN
"""


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


"""
    Nodes relating to Conditional Beta subgraph Structures
        - BetaFactorNode: ESCFN, RVTFN, BJTFN
"""


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


class RVTFN(BetaFactorNode):
    pass


class BJTFN(BetaFactorNode):
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




class EDVN(BetaFactorNode, VariableNode):
    """
        Event Dispatch Variable Node.

        For inward direction:
            - Assume one incoming link and multiple outgoing links.
            - If the incoming message has particles, transform the events from PyTorch distribution event format to
                PySigma cognitive level event format. If there are multiple referenced pattern random variables, events
                will be split w.r.t. the random variable sizes, and each split message sent to the corresponding PCFN
                that processes the corresponding random variable.
            - If the incoming message does not have particles, simply send the Parameter message to all connected PCFN.
        For outgoing direction:
            - Assume one outgoing link and multiple incoming links.
            - If the incoming links have particles,

    """
    pass


class PCFN(BetaFactorNode, FactorNode):
    pass


class GFN:
    """
        Gamma Factor Node

        Carry out general-inference message computation at the PGM factor node.

        Induce a message computation task for each of the outgoing link.
    """
    pass

