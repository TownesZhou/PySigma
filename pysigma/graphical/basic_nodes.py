"""
    Basic data structures and abstract node classes
        - LinkData
        - Node
        - FactorNode
        - VariableNode
"""
from __future__ import annotations      # For postponed evaluation of typing annotations
from typing import List, Tuple
from typing import Iterable as IterableType
from abc import ABC, abstractmethod
from collections.abc import Iterable
import torch
from ..defs import VariableMetatype, Variable, MessageType, Message, NP_EPSILON
from ..utils import compatible_shape

# Define typing aliases
MessageShape = Tuple[torch.Size, torch.Size, torch.Size, torch.Size]


class LinkData:
    """Identifies the data of a directed link between a factor node and a variable node. Stores intermediate messages in
    its message memory.

    Note that links are directional, and two of such links should be specified with opposite directions to represent
    a bidirectional link between a factor node and a variable node.

    During construction of the graph, its instance would be passed to NetworkX methods as the edge data to instantiate
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
        The shape of the message to carry. Used for sanity check of message shapes. Should be a 4-tuple of
        ``torch.Size()`` in the format ``(batch_shape, param_shape, sample_shape, event_shape)``. An empty shape
        ``torch.Size([])`` represents the default none shape.
    epsilon : float, optional
        Epsilon upper bound for checking message difference. Defaults to ``defs.NP_EPSILON``.

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
    def __init__(self,
                 vn: VariableNode,
                 fn: FactorNode,
                 to_fn: bool,
                 msg_shape: MessageShape,
                 epsilon: float = NP_EPSILON,
                 **kwargs):
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
        self.msg_shape: MessageShape = msg_shape
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

    def __repr__(self):
        return str(self)

    def reset_shape(self, msg_shape: MessageShape):
        """Reset shape for the Message

        Parameters
        ----------
        msg_shape : tuple of torch.Size
            The target message shape, in the format ``(batch_shape, param_shape, sample_shape, event_shape)``.  An empty
            shape ``torch.Size([])`` should be used as the default none shape.

        Warnings
        --------
        This method will reset the memory buffer ``self.memory`` to initial identity value and set ``self.new`` to
        False.
        """
        assert isinstance(msg_shape, tuple) and len(msg_shape) == 4 and \
            all(isinstance(s, torch.Size) for s in msg_shape)
        self.msg_shape = msg_shape
        self.memory = Message.identity()
        self.new = False

    def write(self, new_msg: Message, check_diff: bool = True, clone: bool = False):
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

            When `self` and `other` have type ``MessageType.Dual``, the parameters will be chosen over the particles to
            compare message difference.

        .. note::

           If want to set a new message of a different message shape than the current memory, make sure reset_shape()
           is first called so that shape check works for the new message.
        """
        assert isinstance(new_msg, Message)
        # Check new message shape
        assert compatible_shape(self.msg_shape, new_msg.shape), \
            "At {}: Attempt to write a message with incompatible shape. Expect message shape {}, instead encountered " \
            "{}".format(str(self), self.msg_shape, new_msg.shape)

        # Will replace the memory immediately if any one of the following conditions is met:
        #   - check_diff is False
        #   - new message has different type
        #   - new message has Undefined type
        #   - messages have particles and new message has different particle values and/or sampling log densities
        if check_diff is False or new_msg.type != self.memory.type or \
                new_msg.type == MessageType.Undefined or\
                (MessageType.Particles in new_msg.type and not self.memory.same_particles_as(new_msg)):
            self.memory = new_msg.clone() if clone is True else new_msg
            self.new = True
            return

        # Otherwise, check difference
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

    def read(self, clone: bool = False) -> Message:
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
    device : str
        The device on which the node should be processing the messages. Defaults to 'cpu'

    Attributes
    ----------
    name
    device
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

    def __init__(self, name: str, device: torch.device = torch.device('cpu')):
        self.name = name
        self.device = device
        # Flag indicating whether this node has been visited (compute() method called) during a decision cycle
        self.visited = False

        # List of LinkData of those link connecting to this node, incoming and outgoing ones respectively, from
        #   which we retrieve messages
        self.in_linkdata: List[LinkData] = []
        self.out_linkdata: List[LinkData] = []

        # Global logging info
        self.log = {}
        self.pretty_log = {}  # Log info to display at the GUI

    def __str__(self):
        # override to provide the node's name as its string representation
        return self.name

    @property
    def quiescence(self) -> bool:
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
        return all(not in_ld.new for in_ld in self.in_linkdata)

    @abstractmethod
    def add_link(self, linkdata: LinkData):
        """Adding linkdata connecting to this node.

        """
        raise NotImplementedError

    def modify(self):
        """Modification method to be called during modification phase

        Since not every nodes necessarily needs to have its contents modified during modification phase, this method
        is not marked as abstract, and the default behavior is simply to do nothing.
        """
        pass

    @abstractmethod
    def compute(self):
        """Compute method to be called to propagate messages during decision phases.

        Note
        ----
        Note that in a subclass, when overriding this method, it must be decorated with
        ``@parent_class.compute_control`` where ``parent_class`` is the parent class of said subclass so that
        the message propagation process can be controlled and relevant stats be set.
        """
        raise NotImplementedError

    @staticmethod
    def compute_control(compute_func):
        """Decorator that controls the message processing of this node depending on the quiescence state and set
        corresponding stats when the processing is completed. This decorator **must** always be used with the
        `compute()` method in any Node subclass.

        Basic controls:
            - Return directly and prevent computation if reached quiescence.
            - Set ``visited`` to true after the computation.
        """
        assert callable(compute_func)

        def wrapper(self):
            # Return directly if quiesced. avoid any computation
            if self.quiescence:
                return
            # Call the actual compute method
            compute_func(self)
            # Set stats aftermath
            self.visited = True

        return wrapper

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
    def add_link(self, linkdata: LinkData):
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
    def __init__(self,
                 name: str,
                 rel_var_list: IterableType[Variable],
                 param_var: Variable = None,
                 index_var_list: IterableType[Variable] = None,
                 ran_var_list: IterableType[Variable] = None,
                 **kwargs):
        super(VariableNode, self).__init__(name, **kwargs)
        assert isinstance(rel_var_list, Iterable) and \
            all(isinstance(v, Variable) and v.metatype is VariableMetatype.Relational for v in rel_var_list)
        assert param_var is None or \
            (isinstance(param_var, Variable) and param_var.metatype == VariableMetatype.Parameter)
        assert index_var_list is None or \
            (isinstance(index_var_list, Iterable) and
             all(isinstance(v, Variable) and v.metatype == VariableMetatype.Indexing for v in index_var_list))
        assert ran_var_list is None or \
            (isinstance(ran_var_list, Iterable) and
             all(isinstance(v, Variable) and v.metatype == VariableMetatype.Random for v in ran_var_list))
        assert (index_var_list is None) is (ran_var_list is None)

        self.rel_vars: Tuple[Variable, ...] = tuple(rel_var_list)
        self.param_var: Variable = param_var
        self.index_vars: Tuple[Variable, ...] = tuple(index_var_list) if index_var_list is not None else None
        self.ran_vars: Tuple[Variable, ...] = tuple(ran_var_list) if ran_var_list is not None else None

        self.b_shape = torch.Size([v.size for v in self.rel_vars])
        self.p_shape = torch.Size([self.param_var.size]) if self.param_var is not None else torch.Size([])
        self.s_shape = torch.Size([v.size for v in self.index_vars]) if self.index_vars is not None else torch.Size([])
        self.e_shape = torch.Size([v.size for v in self.ran_vars]) if self.ran_vars is not None else torch.Size([])

    def add_link(self, linkdata: LinkData):
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
        assert compatible_shape(linkdata.msg_shape, (self.b_shape, self.p_shape, self.s_shape, self.e_shape)), \
            "At {}: The message shape of the linkdata to be registered is not compatible with the predefined message " \
            "shape of the current variable node. Current node's shape: (batch_shape, param_shape, sample_shape, " \
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
    variable nodes have compatible variable shapes. Only admits one incoming link but can connect with multiple outgoing
    links.

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
    def __init__(self, name: str, **kwargs):
        super(DFN, self).__init__(name, **kwargs)
        self.pretty_log["node type"] = "Default Factor Node"

        # Since all incident nodes should have the same variable list, we can therefore log it here as an attribute
        self.rel_vars = None
        self.param_var = None
        self.index_vars = None
        self.ran_vars = None

        self.b_shape = torch.Size([])
        self.p_shape = torch.Size([])
        self.s_shape = torch.Size([])
        self.e_shape = torch.Size([])

    def add_link(self, linkdata: LinkData):
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
        # Check that the variable shape of the newly connected variable node is compatible with the rest
        if self.rel_vars is not None:
            assert compatible_shape((self.b_shape, self.p_shape, self.s_shape, self.e_shape), linkdata.msg_shape), \
                "In DFN {}: the message shape of the linkdata is not compatible with the variable shape of already " \
                "existing variable nodes. Expecting message shape compatible with {}, but found {}"\
                .format(self.name, (self.b_shape, self.p_shape, self.s_shape, self.e_shape), linkdata.msg_shape)

        # Register variables and shapes
        if self.rel_vars is None:
            self.rel_vars = linkdata.vn.rel_vars
            self.b_shape = linkdata.vn.b_shape
        if self.param_var is None:
            self.param_var = linkdata.vn.param_var
            self.p_shape = linkdata.vn.p_shape
        if self.index_vars is None:
            self.index_vars = linkdata.vn.index_vars
            self.s_shape = linkdata.vn.s_shape
        if self.ran_vars is None:
            self.ran_vars = linkdata.vn.ran_vars
            self.e_shape = linkdata.vn.e_shape

        super(DFN, self).add_link(linkdata)

    @FactorNode.compute_control
    def compute(self):
        """Relay untempered message to downstream variable nodes.

        """
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
    def __init__(self,
                 name: str,
                 rel_var_list: IterableType[Variable],
                 param_var: Variable = None,
                 index_var_list: IterableType[Variable] = None,
                 ran_var_list: IterableType[Variable] = None,
                 **kwargs):
        super(DVN, self).__init__(name, rel_var_list, param_var, index_var_list, ran_var_list, **kwargs)
        self.pretty_log["node type"] = "Default Variable Node"

    def add_link(self, linkdata: LinkData):
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

    @VariableNode.compute_control
    def compute(self):
        """Relay untempered message to downstream factor nodes.

        """
        in_ld = self.in_linkdata[0]
        msg = in_ld.read()
        for out_ld in self.out_linkdata:
            out_ld.write(msg)
