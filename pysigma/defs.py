"""
    Basic structures in the graphical architecture
"""

from copy import deepcopy
from enum import Enum, Flag, auto
from collections.abc import Iterable
import torch
from torch.distributions import Transform
from torch.distributions.constraints import Constraint
from torch.nn.functional import l1_loss
import numpy as np
from .utils import KnowledgeServer

"""
    All PySigma global variables / constants should be declared below.
"""
# Global numerical precision
NP_EPSILON = 1e-6


# Variable Metatypes and Variable for general inference
class VariableMetatype(Enum):
    """Enum class for Variable metatypes.

    """
    Indexing = 0  # Particle indexing variable, first dimension of a message
    Relational = 1  # Relational variable, second set of dimensions of a message
    Random = 2  # Random variable, last set of dimensions of a message
    Parameter = 3  # Parameter variable, last dimensions of a distribution's parameter tensor


class Variable:
    """Variable represented by variable nodes in the graphical architecture. Stores information such as a variable's
    meta-type, dimension size, and value constraints (if the variable has Random meta-type).

    The equality testing is used for matching variables in Alpha-Beta subgraphs. Two variables are equal if and only
    if ALL of their fields are equal.

    Parameters
    ----------
    name : str
        The name of the variable.
    metatype : {``VariableMetatype.Indexing``, ``VariableMetatype.Relational``, ``VariableMetatype.Random``, ``VariableMetatype.Parameter``}
        The meta-type of this variable.
    size : int
        The size of the message dimension this variable corresponds to.
    value_constraints : iterable of torch.distributions.constraints.Constraint
        The set of value constraints that determine the value range (support) of this variable. Specify if and only if
        variable's metatype is ``VariableMetatype.Random``.

    Warning
    -------
    Since ``torch.distributions.constraints.Constraint`` and its concrete classes do not overwrite the default equality
    check method ``__eq__()``, two constraint object are equal if and only if they are the same object instance. This
    means two `Variable` instances will be regarded different if their `value_constraints` fields contains different
    constraint instances, even if these constraint instances semantically refer to the same constraint.

    Attributes
    ----------
    name : str
        Variable name.
    metatype : {``VariableMetatype.Indexing``, ``VariableMetatype.Relational``, ``VariableMetatype.Random``, ``VariableMetatype.Parameter``}
        The meta-type of this variable.
    size : int
        The size of the message dimension this variable corresponds to.
    constraints : set of torch.distributions.constraints.Constraint
        The set of value constraints that determine the value range (support) of this variable.
    """

    def __init__(self, name, metatype, size, value_constraints=None):
        assert isinstance(name, str)
        assert isinstance(metatype, VariableMetatype)
        assert isinstance(size, int)
        assert value_constraints is None or (isinstance(value_constraints, Iterable) and
                                             all(isinstance(c, Constraint) for c in value_constraints))
        assert (value_constraints is not None) is (metatype is VariableMetatype.Random)

        # Variable name, its identity. Used for variable matching. Of type str
        self.name = name
        # Variable meta-type, of type VariableMetatype
        self.metatype = metatype
        # Variable size. Size of the dimension that the variable corresponds to. Of type int
        self.size = size
        # List of value constraints if the Variable is of Random metatype.
        #   Useful at Beta-join to select globally valid particle values
        self.constraints = set(value_constraints) if value_constraints is not None else None

    def __eq__(self, other):
        # override so '==' operator test variable equality
        assert isinstance(other, Variable)
        val = self.name == other.name and \
              self.metatype == other.metatype and \
              self.size == other.size and \
              self.constraints == other.constraints

        return val

    def __ne__(self, other):
        # override so '!=' operator test variable inequality
        return not self.__eq__(other)

    def __str__(self):
        # override to provide the name as the string representation
        return self.name

    def __repr__(self):
        return str(self)

    def __repr__(self):
        return self.name

    def __hash__(self):
        # override so that hash value of the string representation of the variable is used
        return hash(self.name + str(self.metatype) + str(self.size) + str(self.constraints))


# Generalized message type and message representation
class MessageType(Flag):
    """Enum class to represent message types

        The True-valued boolean relationship between types, using the ``in`` operator::

            Undefined in Undefined == Undefined in Parameter == Undefined in Particles == Undefined in Both == True
            Parameter in Parameter == Undefined in Both == True
            Particles in Particles == Undefined in Both == True

        All other relations are False.
    """
    Undefined = 0
    Parameter = auto()
    Particles = auto()
    Both = Parameter | Particles


class Message:
    """Message to be propagated between nodes in the graphical architecture.

    The `Message` class is the most fundamental data structure in PySigma that carries the knowledge of a batch of
    distributions to be processed by downstream graphs.

    Parameters
    ----------
    msg_type : {``MessageType.Undefined``, ``MessageType.Parameter``, ``MessageType.Particles``, ``MessageType.Both``}
        The type of this message.
    batch_shape : torch.Size, optional
        The size of the batch dimensions. Must specify and be a shape of **at least** length 1, unless the message is
        representing an identity. See `Notes` section below for more details on identity message.
    param_shape : torch.Size, optional
        The size of the parameter dimension of `parameter`. Must specify if `msg_type` is ``MessageType.Parameter`` with
        a length of **exactly** 1. Default to an empty shape ``torch.Size([])``.
    sample_shape : torch.Size, optional
        The size of the sample dimensions of each particle tensor in `particles` respectively in order. Must specify if
        message type is ``MessageType.Particles``, with a length equal to the number of particle tensors. Default to an
        empty shape ``torch.Size([])``.
    event_shape : torch.Size, optional
        The size of the event dimensions of each particle tensor in `particles` respectively in order. Must specify if
        message type is ``MessageType.Particles``, with a length equal to the number of particle tensors. Default to an
        empty shape ``torch.Size([])``.
    parameter : torch.Tensor or an int of 0, optional
        The parameter tensor to the batch of distributions this message is encoding. Must specify if the message type is
        ``MessageType.Parameter``. A torch.Tensor of shape ``batch_shape + param_shape`` if the parameters do not
        represent the identity in the parameter vector space. Alternatively, can be an int of 0 to specify the identity,
        in which case it is not necessary to specify `batch_shape`. Default to an int of 0.
    particles : iterable of torch.Tensor, optional
        The list of particles representing events w.r.t. each random variable respectively whose collective joint
        distribution this message is encoding. Must specify if the message type is ``MessageType.Particles``, unless
        `weight` is 1, in which case the message represents a universal identity in the particles space. The jth
        entry of the iterable should have shape ``sample_shape[j] + event_shape[j]``.
    weight : torch.Tensor or an int of 1, optional
        The importance weight tensor that, when multiplied with the exponential of the cross product of the log sampling
        densities in `log_densities`, yields the pdf of each combined particle w.r.t. the target distribution that this
        message is encoding. Must specify if the message type is ``MessageType.Particles``. If the weights are
        non-uniform, must be a **positively valued** tensor of shape ``batch_shape + sample_shape``. The supplied tensor
        will be normalized during initialization so that it sums to 1 across the subspace spanned by the sample
        dimensions. Alternatively, can be an int of 1 to specify the identity (uniform weight), in which case it is not
        necessary to specify `batch_shape`. Default to 1.
    log_densities : iterable of torch.Tensor, optional
        The jth entry in the iterable represents the log pdf of the jth particle in `particles` w.r.t. the (marginal)
        sampling distribution from which the jth particle was originally drawn. Must specify if the message type is
        ``MessageType.Particles``, unless `weight` is 1, in which case the message represents a universal identity in
        the particles space. If specified, all density tensors must be **non-positively valued**, since a particle's
        sampling probability density cannot be greater than 1. The jth entry must have shape ``sample_shape[j]``.
    device : str, optional
        The device where the tensor components are hosted. ``.to(device)`` will be called on the tensor arguments
        during initialization. Defaults to 'cpu'.
    **kwargs
        Other keyword arguments that specify special attributes of the message. Will be deep copied when the message is
        cloned. Note that any `dist_info` required by DistributionServer regarding the specification of the parameters
        should be associated with the key ``"dist_info"``.

    Attributes
    ----------
    type : {``MessageType.Undefined``, ``MessageType.Parameter``, ``MessageType.Particles``, ``MessageType.Both``}
        Message type.
    b_shape : torch.Size
        Batch shape.
    p_shape : torch.Size
        Parameter shape.
    s_shape : torch.Size
        Sample shape.
    e_shape : torch.Size
        Event shape.
    parameter : torch.Tensor or None
        Parameter tensor
    particles : list of torch.Tensor or None
        Tuple of particle value tensors
    weight : torch.Tensor or None
        Particle weight tensor
    log_densities : list of torch.Tensor or None
        Tuple of particles log sampling tensors
    num_rvs : int
        The number of random variables. Inferred from the length of `particles`.
    device : str
        The device where the tensor components are hosted.
    attr : dict
        Miscellaneous optional attributes, specified by **kwargs in the constructor.

    Notes
    _____
    In PySigma Graphical Architecture, a message can represent not only a single joint distribution w.r.t. multiple
    random variables, but a *batch* of such joint distribution instances. The distribution instances in the batch are
    mutually independent, but may or may not be identically distributed. This batch is managed and indexed by the batch
    dimensions, specified by `batch_shape`.

    Depending on how each of the distribution instance is represented, a message can be roughly categorized into two
    types: *Parameter* type or *Particles* type.

    1. *Parameter* type: a message of this type encodes a batch of distributions by holding their parameter tensors. The
       semantics of the parameters depends on the context, e.g. whether they are natural parameters to exponential
       family distributions or conventional parameters to PyTorch distribution class. For the latter one, the semantics
       may even be distribution class dependent.

       Specifying the `parameter` argument only in the constructor is sufficient in terms of the message contents.

    2. *Particles* type: a message of this type encodes a batch of distributions by a particle list, with the particles
       being importantly weighted to correctly reflect their pdf w.r.t. to each of the target distribution in the
       distribution batch. In other words, conceptually, each entry in the particle list is a 3-tuple:
       ``(x, w_x, log_p(x))`` where ``x`` is the event value, ``log_p(x)`` is the log pdf of ``x`` w.r.t. its sampling
       distribution ``P(x)``, and ``w_x`` is defined as the ratio of ``Q(x)``, the target distribution pdf, over
       ``P(x)``. Therefore, the target pdf of ``x`` can be recovered by::

           Q(x) = w_x * exp(log_p(x))
           log Q(x) = log(w_X) + log_p(x)

       Note that a message uses a single list of particles to encode and approximate each and every distribution in the
       batch. In other words, the set of event values used to represent each distribution instance is the same, but the
       importance weights assigned to each event value by different distribution instances are different. This is the
       reason that `weight` tensor should include batch dimensions, whereas particle tensors in `particles` and log
       sampling density tensors in `log_densities` should not.

       When there are multiple random variables, each distribution instance in the batch is a joint distribution
       over all random variables. In this case, each of the entry in the provided `particles` are events w.r.t. each
       random variable *only*. To represent the joint distributions, a list of *joint* particles will be formed by
       concatenating the event tensors in `particles` combinatorially, or so to speak, by taking the tensor product.
       Accordingly, the log sampling density vectors in `log_densities` will be taken cross product to form a higher
       dimensional sampling density tensors. In this way, the joint particles are effectively arranged in a lattice in
       the joint event space, therefore easing the marginalization process because we can simply *summarize* over one
       dimension to achieve the effect of marginalizing over the corresponding random variable.

       To support the above semantics and computations, all of the arguments `particles`, `weight`, and `log_densities`
       must be specified in the constructor.

    A message can encode both type of contents, in which case the message type is ``MessageType.Both``.

    .. _message-arithmetic-structures-notes:

    Both types of messages are assumed to reside in certain vector space, and thus the appropriate arithmetic
    operations -- *Addition* and *Scalar Multiplication* -- are defined and implemented:

    * For Parameter messages,

        * *Addition* operation is defined as arithmetic addition on the parameter tensors.
        * *Scalar multiplication* is defined as arithmetic scalar multiplication with the parameter tensors.
        * 0 is treated as the identity element.

    * For Particles messages:

        * The following two operations are defined as operations on the particle weights, and meaningful only
          for Particle messages that share the same particle values and the same sampling log densities of the
          particles, **except for the message that represents particle identity** (See more below). In addition, results
          from these two operations are normalized so that the weight tensor sums to 1 across the sample dimensions.
        * *Addition* operation is defined as element-wise multiplication of particle weights tensors, up to a
          normalization factor.
        * *Scalar Multiplication* is defined as taking elements of the particle weights tensor to the power
          of the scalar, up to a normalization factor.
        * 1 is treated as the identity element for the operations.
        * Note that it is provably correct that the weighs with above operations form a vector space. The proof idea is
          to consider the log quotient space over one dimension, which reduces to standard real space with one less
          dimension.

    Regarding the identity messages:

    * The ``MessageType.Parameter`` type identity message is one whose ``parameter`` field is 0.
    * The ``MessageType.Particles`` type identity message is one whose ``weight`` field is 1, **regardless of its
      particle values ``particles`` or sampling log densities ``log_densities``.
    * The ``MessageType.Both`` type identity message is the composition of the above two identity messages. In other
      words, a ``MessageType.Both`` type message is identity if and only if both ``parameter`` field is 0 and
      ``weight`` field is 1.

    Accordingly, the '+' and '*' operator are overloaded according the to the specifications above.
    """
    def __init__(self, msg_type,
                 batch_shape=torch.Size([]),
                 param_shape=torch.Size([]), sample_shape=torch.Size([]), event_shape=torch.Size([]),
                 parameter=0, particles=None, weight=1, log_densities=None, device='cpu', **kwargs):
        assert isinstance(msg_type, MessageType)
        assert isinstance(batch_shape, torch.Size) and \
               ((MessageType.Parameter in msg_type and not isinstance(parameter, torch.Tensor)) or
                (MessageType.Particles in msg_type and not isinstance(weight, torch.Tensor)) or
                len(batch_shape) >= 1), \
            "`batch_shape` must be a torch.Size of length at least 1, unless the message is an identity."
        assert isinstance(param_shape, torch.Size) and len(param_shape) <= 1, \
            "`param_shape` must be a torch.Size of length at most 1. Found {}".format(param_shape)
        assert isinstance(sample_shape, torch.Size)
        assert isinstance(event_shape, torch.Size) and len(event_shape) == len(sample_shape), \
            "`sample_shape` and `event_shape` must both be torch.Size with the same length. Found " \
            "(sample_shape, event_shape) = ({}, {})".format(sample_shape, event_shape)

        assert isinstance(parameter, torch.Tensor) or parameter == 0
        assert particles is None or (isinstance(particles, Iterable) and
                                     all(isinstance(p, torch.Tensor) for p in particles))
        assert isinstance(weight, torch.Tensor) or weight == 1
        assert log_densities is None or (isinstance(log_densities, Iterable) and
                                         all(isinstance(d, torch.Tensor) for d in log_densities))

        # Message type, of type MessageType
        self.type = msg_type
        # Device
        self.device = device
        # Parameter
        self.parameter = parameter.to(device=self.device) if isinstance(parameter, torch.Tensor) else parameter
        # Particle list
        self.particles = tuple(p.to(device=self.device) for p in particles) if particles is not None else None
        self.weight = weight.to(device=self.device) if isinstance(weight, torch.Tensor) else weight
        self.log_densities = tuple(d.to(device=self.device) for d in log_densities) if log_densities is not None else \
            None
        # Additional important attributes
        self.attr = kwargs
        # Shapes.
        self.b_shape = batch_shape
        self.p_shape = param_shape
        self.s_shape = sample_shape
        self.e_shape = event_shape
        # Number of random variables
        self.num_rvs = len(self.particles) if particles is not None else 0

        # Check whether necessary arguments are provided
        if MessageType.Parameter in self.type:
            assert self.isid or len(self.p_shape) == 1, \
                "For a Parameter message, the length of the parameter shape must be exactly 1. Found param_shape = {}" \
                .format(self.p_shape)
            assert self.parameter is not None, \
                "Must specify `parameter` for a Parameter message."
        if MessageType.Particles in self.type:
            assert self.isid or len(self.s_shape) >= 1, \
                "For a Particles message, the length of the sample / event shape must be at least 1. Found " \
                "sample_shape = {}".format(self.s_shape)
            # does not care whether particles is specified if it's a particles identity
            assert self.isid or self.particles is not None, \
                "Must specify the particle values tensor via `particles` for a Particles message, unless the message " \
                "represents the identity."
            assert self.weight is not None, \
                "Must specify the particle weight tensor via `weight` for a Particles message, unless the message " \
                "represents the identity."
            # similarly does not care if it's identity
            assert self.isid or self.log_densities is not None, \
                "Must specify the particles log density tensor via `log_densities` for a Particles message, unless " \
                "the message represents the identity."

        # Check shape and values. Adjust if necessary
        if isinstance(self.parameter, torch.Tensor):
            # Parameter tensor should have shape (b_shape + p_shape)
            assert self.b_shape + self.p_shape == self.parameter.shape, \
                "When specified, the parameter tensor should have shape (batch_shape + param_shape). Expect {}, but " \
                "instead found {}.".format(self.b_shape + self.p_shape, self.parameter.shape)
        if particles is not None:
            # jth particle tensor should have shape (s_shape[j] + e_shape[j])
            assert self.num_rvs == len(self.s_shape) == len(self.e_shape), \
                "When specified, the number of entries in the iterable `particles` must equal to the length of " \
                "`sample_shape` and `event_shape`. Expect {}, but instead found {} entries." \
                .format(len(self.s_shape), self.num_rvs)
            assert all(torch.Size([self.s_shape[j], self.e_shape[j]]) == p.shape
                       for j, p in enumerate(self.particles)), \
                "When specified, The j-th particle tensor specified in the iterable `particles` should have shape " \
                "(sample_shape[j] + event_shape[j]). Expect shapes {}, but instead found particle tensor shapes {}." \
                .format(list(torch.Size([self.s_shape[j], self.e_shape[j]]) for j in range(self.num_rvs)),
                        list(p.shape for p in self.particles))
        if isinstance(self.weight, torch.Tensor):
            # Weights tensor should have shape (b_shape + s_shape)
            assert self.b_shape + self.s_shape == self.weight.shape, \
                "When specified, the particles weight should have shape (batch_shape + sample_shape). Expect {}, but " \
                "intead found {}.".format(self.b_shape + self.s_shape, self.weight.shape)
            # Check that values are positive
            assert torch.all(self.weight >= 0), "Found negative values in particle weights. Minimum value: {}" \
                .format(torch.min(self.weight))
            # Normalize the values so that weights sum to 1 across sample dimension
            sample_dims = list(range(len(self.b_shape), len(self.b_shape) + len(self.s_shape)))
            self.weight /= self.weight.sum(dim=sample_dims, keepdim=True)
        if self.log_densities is not None:
            # jth log density tensor vector should have shape (s_shape[j])
            assert len(self.log_densities) == self.num_rvs, \
                "When specified, the iterable `log_densities` must have the same number of entries as `particles`. " \
                "Found {} entries in `particles`, but {} entries in `log_densities`" \
                .format(self.num_rvs, len(self.log_densities))
            assert all(torch.Size([self.s_shape[j]]) == d.shape for j, d in enumerate(self.log_densities)), \
                "When specified, the j-th log density tensor specified in the iterable `log_densities` should have " \
                "shape (sample_shape[j]). Expect shapes {}, but instead found log density tensor shapes {}." \
                .format(list(self.s_shape[j] for j in range(self.num_rvs)), list(d.shape for d in self.log_densities))

    """
        Member properties
    """
    @property
    def isid(self):
        """Whether `self` is an identity message.

        """
        if not MessageType.Parameter in self.type and not MessageType.Particles in self.type:
            return False

        if self.type is MessageType.Parameter:
            return not isinstance(self.parameter, torch.Tensor)

        if self.type is MessageType.Particles:
            return not isinstance(self.weight, torch.Tensor)

        if self.type is MessageType.Both:
            return not isinstance(self.parameter, torch.Tensor) and not isinstance(self.weight, torch.Tensor)

    @property
    def shape(self):
        """Shape of the message. Equivalent to calling size()

        """
        return self.size()

    """
        Overload arithmetic operators
    """
    def __eq__(self, other):
        """Overrides equality testing operation ``==``

        Two messages are equal if and only if all of its contents are equal, including contents in the auxiliary
        dictionary ``self.attr``.

        Note
        ----
        Messages with same contents but on different devices are considered equal. Therefore, ``device`` field is not
        taken into consideration when testing equality.
        """
        assert isinstance(other, Message)
        if self.type != other.type:
            return False
        if self.b_shape != other.b_shape or self.p_shape != other.p_shape or self.s_shape != other.s_shape or \
           self.e_shape != other.e_shape:
            return False

        # If two messages' devices are different, compare self to a clone of other that is transferred to the device
        # that self is on
        if self.device != other.device:
            other = other.to_device(self.device)

        # Consider numerical precision when comparing tensors
        if isinstance(self.parameter, torch.Tensor) != isinstance(other.parameter, torch.Tensor):
            return False
        elif isinstance(self.parameter, torch.Tensor) and isinstance(other.parameter, torch.Tensor) and \
                not torch.max(torch.abs(self.parameter - other.parameter)) < NP_EPSILON:
            return False

        if (self.particles is None) != (other.particles is None):
            return False
        elif self.particles is not None and other.particles is not None:
            if len(self.particles) != len(other.particles):
                return False
            elif not all(torch.max(torch.abs(p1 - p2)) < NP_EPSILON for p1, p2 in zip(self.particles, other.particles)):
                return False

        if isinstance(self.weight, torch.Tensor) != isinstance(other.weight, torch.Tensor):
            return False
        elif isinstance(self.weight, torch.Tensor) and isinstance(other.weight, torch.Tensor) and \
                not torch.max(torch.abs(self.weight - other.weight)) < NP_EPSILON:
            return False

        if (self.log_densities is None) != (other.log_densities is None):
            return False
        elif self.log_densities is not None and other.log_densities is not None:
            if len(self.log_densities) != len(other.log_densities):
                return False
            elif not all(torch.max(torch.abs(d1 - d2)) < NP_EPSILON for d1, d2 in
                         zip(self.log_densities, other.log_densities)):
                return False

        if not self.attr == other.attr:
            return False

        return True

    def __ne__(self, other):
        """Overrides inequality testing operation ``!=``

        See Also
        --------
        __eq__
        """
        return not self == other

    def __add__(self, other):
        """Overrides the addition operation ``+``.

        Implements the semantics of addition operation as in vector spaces. The computational operations used to
        implement the semantics are different for different message contents. See
        :ref:`Message class notes on arithmetic structures<message-arithmetic-structures-notes>`
        for more details.

        Only messages with compatible types can be added. This means a ``MessageType.Parameter`` type message can only
        be added with one of type ``MessageType.Parameter`` or ``MessageType.Both``, and similarly a
        ``MessageType.Particles`` type message can only be added with one of type ``MessageType.Particles`` or
        ``MessageType.Both``. ``MessageType.Both`` type message can be added with any other type except
        ``MessageType.Undefined``, and in any case a ``MessageType.Undefined`` type message cannot be added.

        There are more restrictions for ``MessageType.Particles`` type messages. Messages of such type can only be
        added together if their ``particles`` and ``log_densities`` fields are equal, unless one (or both) is the
        identity Particles message.

        If two messages with compatible but not identical types are added together, the resulting message will have the
        smaller type, meaning only the common components will be added. For example, the result of adding a
        ``MessageType.Parameter`` type message with a ``MessageType.Both`` type message is a ``MessageType.Parameter``
        type message. But if two ``MessageType.Both`` type messages are added, the resulting message will also have
        type ``MessageType.Both``, containing both parameter and particles components.

        Note that the identity messages (Parameter message with ``parameter == 0``, Particles message with
        ``weight == 1``, or Both message with both conditions) are assumed universal, i.e., they can be added with
        any other message that has a compatible type but may or may not have a compatible shape. The resulting message
        will be the other message itself. If both `self` and `other` are identity messages, the returning message will
        be the identity message with the larger type.

        Parameters
        ----------
        other : Message
            The other message instance to be added together with `self`. It should have a compatible message type with
            `self`.

        Returns
        -------
        Message
            The new message as a result of the summation.

        Raises
        ------
        AssertionError
            If `other`'s message type is incompatible with `self`.
        AssertionError
            If either `self` or `other`'s message type is ``MessageType.Undefined``.
        AssertionError
            If contents of `self` and `other` have conflicting shapes, when both `self` and `other` are not identity
            messages.
        AssertionError
            If `self` and `other` have particles message contents to be added, but their particle values do not match,
            or their log sampling density tensors do not match.

        Warnings
        --------
        The attribute dictionaries ``self.attr`` and ``other.attr`` from the two messages will be merged. However, if
        there exist conflicting entries, some would be overwritten by the other. In general, it is the last operand
        in the expression, i.e., `other`, whose attribute entries persist, but this behavior should not be counted on.
        """
        assert isinstance(other, Message), "Message can only be added with another Message"
        assert self.device == other.device, \
            "Messages not residing on the same device. Found devices {} and {}".format(self.device, other.device)
        assert self.type in other.type or other.type in self.type, \
            "Only compatible types of messages can be added. First operand has type '{}',  while the second one has " \
            "type '{}'".format(self.type, other.type)
        # Get the small type and large type
        if self.type in other.type:
            s_type, l_type = self.type, other.type
        else:
            s_type, l_type = other.type, self.type
        # Undefined type cannot be added
        assert s_type is not MessageType.Undefined, \
            "Message of undefined type cannot be added. First operand has type '{}', while the second one has type " \
            "'{}'".format(self.type, other.type)

        # Check if either self or other is an identity message. If so, return the other message directly.
        if self.isid and other.isid:
            if self.type in other.type:
                return other
            return self
        if self.isid:
            return other
        if other.isid:
            return self

        param_msg = None
        ptcl_msg = None
        # Addition for Parameter type
        if MessageType.Parameter in s_type:
            assert self.b_shape == other.b_shape and self.p_shape == other.p_shape, \
                "Only Messages with the same shape can be added together. The messages being added are of Parameter " \
                "type. Found first message with (batch_shape, param_shape) = '{}', and second message with " \
                "(batch_shape, param_shape) = '{}'".format((self.b_shape, self.p_shape), (other.b_shape, other.p_shape))
            # Tensor addition
            new_parameter = self.parameter + other.parameter

            param_msg = Message(MessageType.Parameter, batch_shape=self.b_shape, param_shape=self.p_shape,
                                parameter=new_parameter, device=self.device)

        # Addition for Particles type
        if MessageType.Particles in s_type:
            assert self.b_shape == other.b_shape and self.s_shape == other.s_shape and self.e_shape == other.e_shape, \
                "Only Messages with the same shape can be added together. The messages being added are of Particles " \
                "type. Found first message with (batch_shape, sample_shape, event_shape) = '{}', and second message " \
                "with (batch_shape, sample_shape, event_shape) = '{}'" \
                    .format((self.b_shape, self.s_shape, self.e_shape), (other.b_shape, other.s_shape, other.e_shape))
            assert all(torch.equal(self_p, other_p) for self_p, other_p in zip(self.particles, other.particles)), \
                "For particle messages, only ones with matching particle values can be added together. "
            assert all(torch.equal(self_d, other_d) for self_d, other_d in
                       zip(self.log_densities, other.log_densities)), \
                "For particle messages, only ones with matching log sampling densities can be added together"

            # Take element-wise product
            new_weights = self.weight * other.weight
            # Clone self tensor contents
            cloned_particles = tuple(p.clone() for p in self.particles)
            cloned_log_densities = tuple(d.clone() for d in self.log_densities)
            ptcl_msg = Message(MessageType.Particles,
                               batch_shape=self.b_shape, sample_shape=self.s_shape, event_shape=self.e_shape,
                               particles=cloned_particles, weight=new_weights, log_densities=cloned_log_densities,
                               device=self.device)

        # Compose if we are adding two Both type messages, otherwise return the proper one
        if param_msg is not None and ptcl_msg is not None:
            new_msg = Message.compose(param_msg, ptcl_msg)
        elif param_msg is not None:
            new_msg = param_msg
        else:
            new_msg = ptcl_msg

        # Merge and set attributes
        new_msg.attr = {**self.attr, **other.attr}

        return new_msg

    def __iadd__(self, other):
        """Overrides self-addition operator ``+=``.

        See Also
        --------
        __add__
        """
        return self.__add__(other)

    def __mul__(self, other):
        """Overrides multiplication operator ``*``.

        Implements the semantics of scalar multiplication operation as in vector spaces. The computational operations
        used to implement the semantics are different for different message contents. See
        :ref:`Message class notes regarding arithmetic structures<message-arithmetic-structures-notes>`
        for more details.

        Message of type ``MessageType.Undefined`` cannot be scalar multiplied.

        If `self` is an identity message, returns `self` unchanged directly.

        Parameters
        ----------
        other : int, float, or torch.Tensor
            The scalar to the multiplication. If a torch.Tensor, can be a singleton tensor representing a single scalar,
            or a tensor of shape ``batch_shape`` representing a batched scalars, assigning a different scalar value to
            each distribution instance in the batch.

        Returns
        -------
        Message
            The new message as a result of the scalar multiplication.

        Raises
        ------
        AssertionError
            Attempting to scalar multiply a message of type ``MessageType.Undefined``.
        """
        assert isinstance(other, (int, float, torch.Tensor)), \
            "Message can only be multiplied with a scalar. The scalar can be of int, float or torch.Tensor type. " \
            "Instead found: '{}'".format(type(other))
        if isinstance(other, torch.Tensor):
            assert other.shape in [torch.Size([]), torch.Size([1]), self.b_shape], \
                "If the scalar is a torch.Tensor, must be either a singleton tensor or a tensor with the same shape " \
                "as the Message's batch shape: '{}'. Instead found: '{}'".format(self.b_shape, other.shape)
            # If other is a tensor with a singleton dimension, squeeze into pure scalar
            if other.shape == torch.Size([1]):
                other = other.squeeze(0)
        # Undefined type cannot be scalar multiplied
        assert self.type is not MessageType.Undefined, \
            "Message of undefined type cannot be scalar multiplied. The message has type '{}'" \
            .format(self.type)

        # If self is identity, return directly.
        if self.isid:
            return self

        # Expand scalar tensor dimension if it is a batched scalars
        b_p_other = other
        b_s_other = other
        if isinstance(other, torch.Tensor) and other.dim() > 0:
            if MessageType.Parameter in self.type:
                b_p_other = torch.unsqueeze(b_p_other, dim=-1)

            if MessageType.Particles in self.type:
                for i in range(len(self.s_shape)):
                    b_s_other = torch.unsqueeze(b_s_other, dim=-1)

        # Scalar multiplication for Parameter messages
        param_msg = None
        ptcl_msg = None
        if MessageType.Parameter in self.type:
            new_parameter = b_p_other * self.parameter
            param_msg = Message(MessageType.Parameter,
                                batch_shape=self.b_shape, param_shape=self.p_shape, parameter=new_parameter,
                                device=self.device)

        # Scalar multiplication for Particles messages
        if MessageType.Particles in self.type:
            # The result of scalar multiplication with uniform weights is still uniform, so only process non-uniform
            #   weights
            new_weight = self.weight
            if isinstance(new_weight, torch.Tensor):
                # Extract int/float from singleton scalar tensor
                if isinstance(b_s_other, torch.Tensor) and b_s_other.dim() == 0:
                    b_s_other = b_s_other.item()
                # Take weights tensor to the power of the scalar
                new_weight = torch.pow(new_weight, b_s_other)

            # Clone tensor contents
            cloned_particles = tuple(p.clone() for p in self.particles)
            cloned_log_densities = tuple(d.clone() for d in self.log_densities)
            ptcl_msg = Message(MessageType.Particles,
                               batch_shape=self.b_shape, sample_shape=self.s_shape, event_shape=self.e_shape,
                               particles=cloned_particles, weight=new_weight, log_densities=cloned_log_densities,
                               device=self.device, **self.attr)

        # Compose if we are multiplying a Both type messages, otherwise return the proper one
        if param_msg is not None and ptcl_msg is not None:
            new_msg = Message.compose(param_msg, ptcl_msg)
        elif param_msg is not None:
            new_msg = param_msg
        else:
            new_msg = ptcl_msg

        return new_msg

    def __imul__(self, other):
        """Overrides self-multiplication operator ``*=``.

        See Also
        --------
        __mul__
        """
        return self.__mul__(other)

    def __str__(self):
        # if self.type == MessageType.Parameter:
        #     b_shape_str = str(list(self.b_shape))
        #     p_shape_str = str(list(self.p_shape))
        #     parameters_str = str(list(self.parameter.tolist()))
        #
        #     return f"Type: Parameter\nBatch_Shape: {b_shape_str}\nParameter_Shape: {p_shape_str}\n" \
        #            f"Parameters{parameters_str}"
        #
        # else:
        #     s_shape_str = str(list(self.s_shape))
        #     b_shape_str = str(list(self.b_shape))
        #     e_shape_str = str(list(self.e_shape))
        #     particles_str = str(list(self.particles.tolist()))
        #     weights_str = str(list(self.weights.tolist()))
        #     log_density_str = str(list(self.log_density.tolist()))
        #
        #     return f"Type: Particles\nSample_Shape: {s_shape_str}\nBatch_Shape: {b_shape_str}\n" \
        #            f"Event_Shape: {e_shape_str}\nParticles: {particles_str}\n" \
        #            f"Weights: {weights_str}\nLog_Density: {log_density_str}"
        raise NotImplementedError("String representation for Message instance is yet to be implemented.")

    """
        Message class static utility methods 
    """
    @staticmethod
    def compose(msg1, msg2):
        """Composes a ``MessageType.Particles`` message with a ``MessageType.Parameters`` message to return a
        ``MessageType.Both`` message that contain all components from both messages.

        Both `msg1` and `msg2` cannot be identity messages.

        Parameters
        ----------
        msg1 : Message
            The first message to be composed. Its type must be either `MessageType.Particles`` or
            ``MessageType.Parameters``, but must be different from that of `msg2`.
        msg2 : Message
            The second message to be composed. Its type must be either `MessageType.Particles`` or
            ``MessageType.Parameters``, but must be different from that of `msg1`.

        Returns
        -------
        Message
            A message with type ``MessageType.Both`` that contains all components from both `msg1` and `msg2`.

        Raises
        ------
        AssertionError
            If `msg1` and `msg2` have conflicting attributes, such as batch shape.

        Warnings
        --------
        The attribute dictionaries ``msg1.attr`` and ``msg2.attr`` will be merged. If there exists conflicting entries
        (key-value pairs with same key but different values), those from ``msg2.attr`` will overwrite those from
        ``msg1.attr``.
        """
        assert isinstance(msg1, Message) and isinstance(msg2, Message)
        assert msg1.device == msg2.device, \
            "Messages not residing on the same device. Found devices {} and {}".format(msg1.device, msg2.device)
        assert {msg1.type, msg2.type} == {MessageType.Parameter, MessageType.Particles}
        assert not msg1.isid and not msg2.isid, \
            "`msg1` and `msg2` both cannot be identity messages when composing new messages."

        param_msg = msg1 if msg1.type is MessageType.Parameter else msg2
        ptcl_msg = msg1 if msg1.type is MessageType.Particles else msg2
        # Check consistency of other Message attributes
        assert param_msg.b_shape == ptcl_msg.b_shape, \
            "Attempting to compose a parameter message with a particles message, but found conflicting batch shapes: " \
            "{}".format([msg1.b_shape, msg2.b_shape])

        new_msg = Message(MessageType.Both,
                          batch_shape=param_msg.b_shape, param_shape=param_msg.p_shape,
                          sample_shape=ptcl_msg.s_shape, event_shape=ptcl_msg.e_shape,
                          parameter=param_msg.parameter,
                          particles=ptcl_msg.particles, weight=ptcl_msg.weight, log_densities=ptcl_msg.log_densities,
                          device=msg1.device, **{**msg1.attr, **msg2.attr})
        return new_msg

    @staticmethod
    def identity(msg_type=MessageType.Both):
        """Returns a minimum identity message (without declaration of shapes) of the specified type.

        Parameters
        ----------
        msg_type : MessageType
            Target message type. Defaults to ``MessageType.Both``.

        Returns
        -------
        Message
            The identity message.
        """
        assert MessageType.Parameter in msg_type or MessageType.Particles in msg_type
        if msg_type is MessageType.Both:
            return Message(MessageType.Both, parameter=0, weight=1)
        if msg_type is MessageType.Parameter:
            return Message(MessageType.Parameter, parameter=0)
        return Message(MessageType.Particles, weight=1)

    """
        General utility member methods
    """
    def size(self):
        """Returns a tuple of the message's shapes: ``(batch_shape, param_shape, sample_shape, event_shape)``

        Returns
        -------
        tuple of torch.Size
            A tuple of the message's shapes
        """
        return self.b_shape, self.p_shape, self.s_shape, self.e_shape

    # def same_size_as(self, other):
    #     """Check if self has the same shape as the other message.
    #
    #     Returns
    #     -------
    #     bool
    #         True if self has the same shape as the other message.
    #     """
    #     assert isinstance(other, Message)
    #     return self.size() == other.size()

    def same_particles_as(self, other):
        """Check if `self` has the same particles as the other message. This include checking the list of particle value
        tensors as well as checking the list of particle log sampling density tensors.

        .. note::

           Will always return ``False`` if `self` or `other` is not Particles message.

        .. note::

           Will always return ``True`` if both `self` and `other` are Particles message and one (or both) is the
           identity.

        Parameters
        ----------
        other : Message
            The other message.

        Returns
        -------
        bool
            True if `self` has the same particles as the `other` message.
        """
        assert isinstance(other, Message)
        # Return False is message is not event Particles message
        if MessageType.Particles not in self.type or MessageType.Particles not in other.type:
            return False
        # Return True if one is the identity
        if self.isid or other.isid:
            return True
        # Otherwise, if number of random variables are different, return False directly
        if self.num_rvs != other.num_rvs:
            return False

        same = True
        # Check list of particle value tensors
        for s_p, o_p in zip(self.particles, other.particles):
            if not torch.equal(s_p, o_p):
                same = False
        # Check list of sampling log density tensors
        for s_d, o_d in zip(self.log_densities, other.log_densities):
            if not torch.equal(s_d, o_d):
                same = False

        return same

    def diff_param(self, other):
        """Compute the difference between the parameters of `self` and `other`.

        Returns a batch average L2 distance between the two parameters. Since parameters have shape
        ``(batch_shape, param_shape)``, with ``param_shape`` of exactly length 1, the L2 distance is calculated along
        ``dim=-1``.

        Parameters
        ----------
        other : Message
            The other message.

        Returns
        -------
        torch.Tensor or 0
            The batch average L2 distance.

        Raises
        ------
        AssertionError
            If `self` and `other` are not on the same device.
        AssertionError
            If `self` and/or `other` are/is not Parameter type.
        AssertionError
            If `self` and `other` are not identities and have different batch shapes or parameter shapes.

        See Also
        --------
        The L2 norm computation:
        `torch.norm() <https://pytorch.org/docs/stable/torch.html#torch.norm>`_.
        """
        assert isinstance(other, Message)
        assert self.device == other.device, \
            "Messages not residing on the same device. Found devices {} and {}".format(self.device, other.device)
        assert MessageType.Parameter in self.type and MessageType.Parameter in other.type
        assert self.isid or other.isid or (self.b_shape, self.p_shape) == (other.b_shape, other.p_shape)

        # Returns 0 if both are identity
        if self.isid and other.isid:
            return 0

        # Parameter has shape (batch_shape, param_shape), with param_shape of exactly length one
        diff = self.parameter - other.parameter
        val = diff.norm(dim=-1).mean()
        return val

    def diff_weight(self, other):
        """Compute the difference between the weight of `self` and `other`.

        Returns a mean element-wise absolute value difference between the two weight tensors.

        Note that calculating the difference of weights only makes sense if both messages have the *same* particle value
        tensors and particle log sampling density tensors. Therefore, `same_particles_as()` will first be called for a
        sanity check. An assertion error will be raised if `same_particles_as()` returns ``False``.

        Parameters
        ----------
        other : Message
            The other message

        Returns
        -------
        torch.Tensor or 0
            The batch average cosine similarity.

        Raises
        ------
        AssertionError
            If `self` and `other` are not on the same device.
        AssertionError
            If `self` and/or `other` are/is not Particles type.
        AssertionError
            If `self` and `other` do not have the same batch shapes, sample shapes, or event shapes.
        AssertionError
            If `self` does not have the same particle values and log sampling densities as `other`.

        See Also
        --------
        The cosine similarity computation:
        `torch.nn.functional.cosine_similarity() <https://pytorch.org/docs/stable/nn.functional.html?highlight=cosine#torch.nn.functional.cosine_similarity>`_.
        """
        assert isinstance(other, Message)
        assert self.device == other.device, \
            "Messages not residing on the same device. Found devices {} and {}".format(self.device, other.device)
        assert MessageType.Particles in self.type and MessageType.Particles in other.type
        assert self.isid or other.isid or \
               (self.b_shape, self.s_shape, self.e_shape) == (other.b_shape, other.s_shape, other.e_shape)
        assert self.same_particles_as(other)

        # Returns 0 if both are identity
        if self.isid and other.isid:
            return 0
        # If one is identity, create a uniform weight tensor of the same size as the other message's weight tensor
        x1 = self.weight if not self.isid else torch.ones_like(other.weight, device=self.device) / other.weight.numel()
        x2 = other.weight if not other.isid else torch.ones_like(self.weight, device=self.device) / self.weight.numel()

        val = l1_loss(x1, x2, reduction='mean')
        return val

    def reduce_type(self, msg_type):
        """Returns a reduced `msg_type` type message from `self`, where irrelevant components w.r.t. 'msg_type' in
        `self` is removed, and only relevant components are retained and cloned.

        The target message type must be either ``MessageType.Parameter`` or ``MessageType.Particles``.

        parameters
        ----------
        msg_type : {``MessageType.Parameter``, ``MessageType.Particles``}
            The message type of the returned reduced message.

        Returns
        -------
        Message
            The reduced message from `self`.

        Raises
        ------
        AssertionError
            If target message type specified by `msg_type` is not compatible with `self` type.
        """
        assert msg_type in self.type, \
            "Target message type '{}' is not compatible with self message type '{}'".format(msg_type, self.type)
        assert msg_type is MessageType.Parameter or msg_type is MessageType.Particles, \
            "The target message type can only be Parameter or Particles. "

        if msg_type == self.type:
            return self

        # First clone content
        cloned_msg = self.clone()
        if msg_type == MessageType.Parameter:
            new_msg = Message(msg_type, batch_shape=cloned_msg.b_shape, param_shape=cloned_msg.p_shape,
                              parameter=cloned_msg.parameter, device=self.device, **cloned_msg.attr)
        else:
            new_msg = Message(msg_type, batch_shape=cloned_msg.b_shape, sample_shape=cloned_msg.s_shape,
                              event_shape=cloned_msg.e_shape, particles=cloned_msg.particles,
                              weight=cloned_msg.weight, log_densities=cloned_msg.log_densities,
                              device=self.device, **cloned_msg.attr)
        return new_msg

    def clone(self):
        """Return a cloned message from self.

        Guarantees that every content is deep-copied. Tensors will be cloned and dictionaries will be deep-copied.

        Returns
        -------
        Message
            A cloned and deep-copied message of `self`.
        """
        parameters = self.parameter
        particles = self.particles
        weight = self.weight
        log_densities = self.log_densities
        attr = self.attr

        if isinstance(parameters, torch.Tensor):
            parameters = parameters.clone()
        if particles is not None:
            particles = tuple(p.clone() for p in particles)
        if isinstance(weight, torch.Tensor):
            weight = weight.clone()
        if log_densities is not None:
            log_densities = tuple(d.clone() for d in log_densities)
        if self.attr is not None:
            attr = deepcopy(self.attr)

        new_msg = Message(self.type,
                          self.b_shape, self.p_shape, self.s_shape, self.e_shape,
                          parameters, particles, weight, log_densities, device=self.device, **attr)
        return new_msg

    def to_device(self, device):
        """Returns a version of `self` where the tensor components are hosted on the specified `device`.

        Per PyTorch design, the original tensors will be returned without copying if target `device` is the current
        device, otherwise a copied version will be returned.

        .. note::

           Any tensor stored in the optional attribute dictionary ``self.attr`` will NOT be inspected and be moved to
           the target device.

        Parameters
        ----------
        device : str
            The target device

        Returns
        -------
        Message
            `self` on target `device`.
        """
        new_parameter = self.parameter.to(device=device) if isinstance(self.parameter, torch.Tensor) else self.parameter
        new_particles = tuple(p.to(device=device) for p in self.particles) if self.particles is not None else None
        new_weight = self.weight.to(device=device) if isinstance(self.weight, torch.Tensor) else self.weight
        new_densities = tuple(d.to(device=device) for d in self.log_densities) \
            if self.log_densities is not None else None

        new_msg = Message(self.type,
                          self.b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_densities,
                          device=device, **self.attr)
        return new_msg

    """
        Methods for batch dimension manipulations. 
    """

    def batch_permute(self, target_dims):
        """Returns a permuted message whose tensor contents that include batch dimensions (e.g. parameters and particle
        values) are permuted w.r.t. `target_dims`.

        The dimensions specified in `target_dims` are relative to the batch dimensions only. Its values should be in
        range ``[-len(batch_shape), len(batch_shape) - 1]``

        `contiguous() <https://pytorch.org/docs/stable/tensors.html?highlight=contiguous#torch.Tensor.contiguous>`_
        will be called so that the returning message's tensor contents are contiguous

        Parameters
        ----------
        target_dims : list of ints
            The desired ordering of the target batch dimensions. Must have the same length as the message's batch shape.

        Returns
        -------
        Message
            The permuted message from `self`.

        See Also
        --------
        This method is a mimic of
        `torch.Tensor.permute() <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute>`_.
        """
        assert isinstance(target_dims, list) and all(isinstance(i, int) for i in target_dims)
        assert len(target_dims) == len(self.b_shape) and \
            all(-len(self.b_shape) <= i <= len(self.b_shape) - 1 for i in target_dims)

        # Translate negative dims to non-negative value
        pos_target_dims = list(len(target_dims) + i if i < 0 else i for i in target_dims)
        # Permuted batch shape
        new_b_shape = torch.Size(list(self.b_shape[pos_target_dims[i]] for i in range(len(self.b_shape))))
        # Permute order for batch and parameter dimensions together
        b_p_dims = pos_target_dims + [len(self.b_shape)]
        # Permute order for sample and batch dimensions together.
        b_s_dims = pos_target_dims + list(range(len(self.b_shape), len(self.b_shape) + len(self.s_shape)))

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameter = new_parameter.permute(b_p_dims)
            new_parameter = new_parameter.contiguous()
        if isinstance(new_weight, torch.Tensor):
            # weight has shape (b_shape + s_shape)
            new_weight = new_weight.permute(b_s_dims)
            new_weight = new_weight.contiguous()

        new_msg = Message(self.type,
                          new_b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_unsqueeze(self, dim):
        """Returns a new message with a dimension of size one inserted at the target batch dimension specified by `dim`.

        The target dimension is relative to the batch dimensions only. It should be in range
        ``[-len(batch_shape) - 1, len(batch_shape) + 1]``.

        Parameters
        ----------
        dim : int
            The position where the new singleton dimension (a dim of size 1) is to be inserted.

        Returns
        -------
        Message
            The unsqueezed message from `self`.

        See Also
        --------
        This method is a mimic of
        `torch.unsqueeze() <https://pytorch.org/docs/stable/torch.html?highlight=unsqueeze#torch.unsqueeze>`_
        """
        assert isinstance(dim, int) and -len(self.b_shape) - 1 <= dim <= len(self.b_shape) + 1

        # Translate dim to positive value if it is negative
        if dim < 0:
            dim = len(self.b_shape) + dim + 1
        # Get new batch shape
        new_b_shape = self.b_shape[:dim] + torch.Size([1]) + self.b_shape[dim:]

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameter = torch.unsqueeze(new_parameter, dim)
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            new_weight = torch.unsqueeze(new_weight, dim)

        new_msg = Message(self.type,
                          new_b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_index_select(self, dim, index):
        """Returns a message that is a concatenation of the slices from `self` along the `dim` batch dimension and
        indexed by `index`.

        In other words, along `dim` dimension, the ``i`` th slice of the returned message is the ``index[i]`` th slice
        of `self`. Consequently, the size of the `dim` dimension of the returned message equals the length of `index`
        array.

        A `dim` value within the range ``[-len(batch_shape), len(batch_shape) - 1]`` can be used. Note that `dim` is
        relative to the batch dimension only.

        Parameters
        ----------
        dim : int
            The dimension along which entries will be selected according to `index`.
        index : torch.LongTensor
            The array of indices of entries along `dim` to be selected. Entries must be non-negative.

        Returns
        -------
        Message
            The returned index-selected and concatenated message from `self`.

        See Also
        --------
        This method is a mimic of
        `torch.index_select() <https://pytorch.org/docs/stable/torch.html?highlight=index_select#torch.index_select>`_
        """
        assert isinstance(dim, int) and -len(self.b_shape) <= dim <= len(self.b_shape) - 1
        assert isinstance(index, torch.LongTensor) and index.dim() == 1

        # Translate dim to positive value if it is negative
        if dim < 0:
            dim = len(self.b_shape) + dim
        # Get new batch shape
        new_b_shape = self.b_shape[:dim] + index.shape + self.b_shape[dim + 1:]

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameter = torch.index_select(new_parameter, dim, index)
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            new_weight = torch.index_select(new_weight, dim, index)

        new_msg = Message(self.type,
                          new_b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_index_put(self, dim, index):
        """Returns a message whose entries along the dimension `dim` are slices from self message and are put into
        the positions along the axis specified by indices in `index`.

        In other words, along `dim` dimension, the ``index[i]`` th slice of the returned message is the ``i`` th slice
        of `self`. Consequently, the size of the `dim` dimension of the returned message equals the maximum value plus 1
        in the `index` array.

        For slices in the new message not referenced by `index`, they will be filled with identity values. For parameter
        tensor, the identity value is 0, and for particle weight tensor, the identity value is a positive uniform
        constant such that the sum across the sample dimensions is 1.

        A `dim` value within the range ``[-len(batch_shape), len(batch_shape) - 1]`` can be used. Note that `dim` is
        relative to the batch dimension only.

        Parameters
        ----------
        dim : int
            The dimension along which entries will be put according to `index`.
        index : torch.LongTensor
            The array of indices of entries along `dim` to be put. Entries must be non-negative.

        Returns
        -------
        Message
            The returned index-put message of `self`.

        See Also
        --------
        batch_index_select() :
            The inverse of batch_index_put(). There is no direct counterpart to this method in PyTorch.
        """
        assert isinstance(dim, int) and -len(self.b_shape) <= dim <= len(self.b_shape) - 1
        assert isinstance(index, torch.LongTensor) and index.dim() == 1 and torch.all(index >= 0)

        # Translate dim value to positive if it's negative
        dim = len(self.b_shape) + dim if dim < 0 else dim
        # Get new batch shape. The size of dimension dim is determined by the maximum value in index
        new_b_shape = self.b_shape[:dim] + torch.Size([torch.max(index) + 1]) + self.b_shape[dim + 1:]

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        # To access tensor slice more easily, we swap the target dim with first dim, perform slicing and assignment on
        #   this new first dim, and swap it back
        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            # Identity value tensor
            to_fill = torch.zeros(new_b_shape + self.p_shape, device=self.device)
            # Transpose target dimension with the first dimension
            to_fill = torch.transpose(to_fill, dim0=0, dim1=dim)
            t_param = torch.transpose(new_parameter, dim0=0, dim1=dim)
            # Slice and assign
            to_fill[index] = t_param
            # Transpose back to get result
            new_parameter = torch.transpose(to_fill, dim0=0, dim1=dim)
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            # Identity value tensor. Use ones here because we assume Message constructor will take care of normalization
            to_fill = torch.ones(new_b_shape + self.s_shape, device=self.device)
            # Transpose target dimension with the first dimension
            to_fill = torch.transpose(to_fill, dim0=0, dim1=dim)
            t_weight = torch.transpose(new_weight, dim0=0, dim1=dim)
            # Slice and assign
            to_fill[index] = t_weight
            # Transpose back to get result
            new_weight = torch.transpose(to_fill, dim0=0, dim1=dim)

        new_msg = Message(self.type,
                          new_b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_diagonal(self, dim1=0, dim2=1):
        """Returns a partial view of self with the its diagonal elements with respect to `dim1` and `dim2` appended as
        a dimension at the end of the shape. Note that the number of dimensions of the returned message is 1 minus
        that of the original message, because `dim1` and `dim2` are reduced to the new diagonal dimension.

        dim values in the range ``[-len(batch_shape), len(batch_shape) - 1]`` can be used. Note that `dim1` and `dim2`
        are relative to the batch dimensions only. The appended dimension will be placed as the last batch dimension,
        but before any sample or param dimensions.

        `contiguous() <https://pytorch.org/docs/stable/tensors.html?highlight=contiguous#torch.Tensor.contiguous>`_
        will be called so that the returning content tensors are contiguous.

        Parameters
        ----------
        dim1 : int, optional
            The first dimension of the 2D subspace where diagonal entries will be taken. Defaults to 0, the first batch
            dimension.
        dim2 : int, optional
            The second dimension of the 2D subspace where diagonal entries will be taken. Defaults to 1, the second
            batch dimension.

        Returns
        -------
        Message
            The diagonalized message of `self`.

        See Also
        --------
        This method is a mimic of
        `torch.diagonal() <https://pytorch.org/docs/stable/torch.html?highlight=diagonal#torch.diagonal>`_
        , with `offset` set to 0
        """
        assert isinstance(dim1, int) and -len(self.b_shape) <= dim1 <= len(self.b_shape) - 1
        assert isinstance(dim2, int) and -len(self.b_shape) <= dim2 <= len(self.b_shape) - 1

        # Translate dim value to positive if it's negative
        dim1 = len(self.b_shape) + dim1 if dim1 < 0 else dim1
        dim2 = len(self.b_shape) + dim2 if dim2 < 0 else dim2
        # Get new batch shape. The size of the appended diagonalized dimension should be the min of dim1 and dim2
        new_b_shape = self.b_shape[:min(dim1, dim2)] + self.b_shape[min(dim1, dim2) + 1: max(dim1, dim2)] + \
            self.b_shape[max(dim1, dim2) + 1:] + torch.Size([min(self.b_shape[dim1], self.b_shape[dim2])])

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameter = torch.diagonal(new_parameter, dim1=dim1, dim2=dim2)
            # Swap param dimension and appended diagonal batch dimension
            new_parameter = torch.transpose(new_parameter, dim0=-1, dim1=-2)
            new_parameter = new_parameter.contiguous()
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            new_weight = torch.diagonal(new_weight, dim1=dim1, dim2=dim2)
            # Permute the appended diagonal batch dimension to the end of the existing batch dimensions
            new_b_dims = len(self.b_shape) - 2
            perm_order = list(range(new_b_dims)) + [new_weight.dim() - 1] + \
                list(range(new_b_dims, new_b_dims + len(self.s_shape)))
            new_weight = new_weight.permute(perm_order)
            new_weight = new_weight.contiguous()

        new_msg = Message(self.type,
                          new_b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_diag_embed(self, diag_dim=-1, target_dim1=-2, target_dim2=-1):
        """Returns a message whose diagonals of certain 2D planes (dimensions specified by `target_dim1` and
        `target_dim2`) are filled by slices of self along the dimension `diag_dim` of the self message).

        The last dimension of self is chosen by default as the diagonal entries to be filled, and the last two
        dimensions of the new message are chosen by default as the 2D planes where the diagonal entries will be filled
        in.

        The 2D planes will be shaped as square matrices, with the size of each dimension matches the size of the
        `diag_dim` dimension in self.

        The length of returned message's batch shape will be the length of original message's batch shape plus 1.

        For slots not on the diagonals of the resulting message, they will be filled with identity values. For parameter
        tensor, the identity value is 0, and for particle weight tensor, the identity value is a positive uniform
        constant such that the sum across the sample dimensions is 1.

        `contiguous() <https://pytorch.org/docs/stable/tensors.html?highlight=contiguous#torch.Tensor.contiguous>`_
        will be called so that the returning content tensors are contiguous.

        Parameters
        ----------
        diag_dim : int, optional
            The dimension of `self` along which slices will be selected. Defaults to -1.
        target_dim1 : int, optional
            The first dimension of the target 2D planes in the target message. Defaults to -2.
        target_dim2 : int, optional
            The second dimension of the target 2D planes in the target message. Defaults to -1.

        Returns
        -------
        Message
            The diagonally embedded message from `self`.

        See Also
        --------
        This method is a mimic of
        `torch.diag_embed() <https://pytorch.org/docs/stable/torch.html?highlight=diag_embed#torch.diag_embed>`_
        , with `offset` default to 0 plus an additional diag_dim argument.
        """
        assert isinstance(diag_dim, int) and -len(self.b_shape) <= diag_dim <= len(self.b_shape) - 1
        assert isinstance(target_dim1, int) and -len(self.b_shape) - 1 <= target_dim1 <= len(self.b_shape)
        assert isinstance(target_dim2, int) and -len(self.b_shape) - 1 <= target_dim2 <= len(self.b_shape)

        # Translate dim value to positive if it's negative
        # For target_dim1 and target_dim2, need to compensate for 1 more dimension in the returned message
        diag_dim = len(self.b_shape) + diag_dim if diag_dim < 0 else diag_dim
        target_dim1 = len(self.b_shape) + 2 + target_dim1 if target_dim1 < 0 else target_dim1
        target_dim2 = len(self.b_shape) + 2 + target_dim2 if target_dim2 < 0 else target_dim2
        # Get new batch shape. The size of target_dim1 and target_dim2 is determined by the size of diag_dim
        diag_size = self.b_shape[diag_dim]
        other_shape = list(self.b_shape[:diag_dim] + self.b_shape[diag_dim + 1:])
        first_new_dim, second_new_dim = min(target_dim1, target_dim2), max(target_dim1, target_dim2)
        other_shape.insert(first_new_dim, diag_size)
        other_shape.insert(second_new_dim, diag_size)
        new_b_shape = torch.Size(other_shape)

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        # Tensors fist need to have the diagonal entries dimension (diag_dim) permuted to the last dimension so that it
        #   will be picked up by torch.diag_embed()
        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            perm_order = list(range(len(self.b_shape + self.p_shape)))
            perm_order.remove(diag_dim)
            perm_order.append(diag_dim)
            new_parameter = new_parameter.permute(perm_order)
            new_parameter = torch.diag_embed(new_parameter, dim1=target_dim1, dim2=target_dim2)
            new_parameter = new_parameter.contiguous()
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            # For weights, the default entries to be filled in places other than the diagonal should be 1's, so we
            #   will first fill the log of input into the diagonal and then take exponential. 0's filled by
            #   torch.diag_embed() will be transformed to 1. Note that for these uniform entries the weights will be
            #   normalized across sample dimension during initialization so no worries.
            log_weight = torch.log(new_weight)
            perm_order = list(range(len(self.b_shape + self.s_shape)))
            perm_order.remove(diag_dim)
            perm_order.append(diag_dim)
            log_weight = log_weight.permute(perm_order)
            log_weight = torch.diag_embed(log_weight, dim1=target_dim1, dim2=target_dim2)
            new_weight = torch.exp(log_weight)
            new_weight = new_weight.contiguous()

        new_msg = Message(self.type,
                          new_b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_narrow(self, dim, length):
        """Returns a message that is a narrowed version of `self` along the dimension specified by `dim`.

        Effectively, this method selects the chunk spanning ``[:length]`` along the dimension `dim` of `self`. The
        returned message and `self` share the same underlying storage.

        `contiguous() <https://pytorch.org/docs/stable/tensors.html?highlight=contiguous#torch.Tensor.contiguous>`_
        will be called so that the returning content tensors are contiguous.

        Parameters
        ----------
        dim : int
            The dimension of along which `self` will be narrowed.
        length : int
            The length of the message chunk to select. It must be no greater than the size of the `dim` dimension in
            `self`.

        Returns
        -------
        Message
            A narrowed message of `self`.

        See Also
        --------
        This method is a mimic of
        `torch.narrow() <https://pytorch.org/docs/stable/torch.html?highlight=narrow#torch.narrow>`_
        , with `start` default to 0.
        """
        assert isinstance(dim, int) and -len(self.b_shape) <= dim <= len(self.b_shape) - 1
        assert isinstance(length, int) and 0 <= length <= self.b_shape[dim] - 1

        # Translate dim value to positive if it's negative
        dim = len(self.b_shape) + dim if dim < 0 else dim
        # Get new batch shape.
        new_b_shape = self.b_shape[:dim] + torch.Size([length]) + self.b_shape[dim + 1:]

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameter = torch.narrow(new_parameter, dim=dim, start=0, length=length)
            new_parameter = new_parameter.contiguous()
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            new_weight = torch.narrow(new_weight, dim=dim, start=0, length=length)
            new_weight = new_weight.contiguous()

        new_msg = Message(self.type,
                          new_b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_broaden(self, dim, length):
        """Returns a message that is a broadened version of `self` along the dimension specified by `dim`, with identity
        values filled in ``[dim_size + 1: length]`` along the dimension `dim` in the returned message.

        In other words, this method is concatenating an identity message to `self` along dimension `dim` so that the
        resulting dimension size is `length`.

        For parameter tensor, the identity value is 0, and for particle weight tensor, the identity value is a positive
        uniform constant such that the sum across the sample dimensions is 1.

        `contiguous() <https://pytorch.org/docs/stable/tensors.html?highlight=contiguous#torch.Tensor.contiguous>`_
        will be called so that the returning content tensors are contiguous.

        Parameters
        ----------
        dim : int
            The dimension of `self` which will be broadened in the returned message.
        length : int
            The length of the broadened dimension of the returned message. It must be greater than the size of the `dim`
            dimension in `self`.

        Returns
        -------
        Message
            A broadened message of `self`.

        See Also
        --------
        batch_narrow() :
            The inverse of batch_broaden(). There is no direct counterpart to this method in PyTorch.
        """
        assert isinstance(dim, int) and -len(self.b_shape) <= dim <= len(self.b_shape) - 1
        assert isinstance(length, int) and length > self.b_shape[dim]

        # Translate dim value to positive if it's negative
        dim = len(self.b_shape) + dim if dim < 0 else dim
        # Get new batch shape.
        new_b_shape = self.b_shape[:dim] + torch.Size([length]) + self.b_shape[dim + 1:]

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            to_concat_shape = self.b_shape[:dim] + torch.Size([length - self.b_shape[dim]]) + \
                              self.b_shape[dim + 1:] + self.p_shape
            to_concat = torch.zeros(to_concat_shape, device=self.device)
            new_parameter = torch.cat([new_parameter, to_concat], dim=dim)
            new_parameter = new_parameter.contiguous()
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            to_concat_shape = self.b_shape[:dim] + torch.Size([length - self.b_shape[dim]]) + \
                              self.b_shape[dim + 1:] + self.s_shape
            to_concat = torch.ones(to_concat_shape, device=self.device)
            new_weight = torch.cat([new_weight, to_concat], dim=dim)
            new_weight = new_weight.contiguous()

        new_msg = Message(self.type,
                          new_b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_summarize(self, dim):
        """Implements the default Sum-Product summarization semantics. Summarizes over the batch dimension specified by
        `dim`. Returns a message with one less dimension.

        For Parameter message, the summarization is realized by taking the mean of the parameter tensor along dimension
        `dim`. For particles message, this is realized by taking addition defined for particle weights along dimension
        `dim`, a.k.a. factor product.

        Parameters
        ----------
        dim : int
            The dimension of `self` to be summarized over.

        Returns
        -------
        Message
            The summarized message from `self`.
        """
        assert isinstance(dim, int) and -len(self.b_shape) <= dim <= len(self.b_shape) - 1

        # Translate dim value to positive if it's negative
        dim = len(self.b_shape) + dim if dim < 0 else dim
        # Get new batch shape.
        new_b_shape = self.b_shape[:dim] + self.b_shape[dim + 1:]

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameter = torch.mean(new_parameter, dim=dim)
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            # For weights, since factor product is taken, we first convert weight values to log scale, perform summation
            #   across the batch dimension, then convert back to exponential scale.
            # The normalization of resulting weights will be taken care of by message initialization
            log_weight = torch.log(new_weight)
            log_weight = torch.sum(log_weight, dim=dim)
            new_weight = torch.exp(log_weight)

        new_msg = Message(self.type,
                          new_b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_flatten(self, dims=None):
        """Flattens the set of batch dimensions specified by `dims` and append the flattened dimension as the last
        batch dimension. If `dims` is ``None``, will flatten all batch dimensions.

        `contiguous() <https://pytorch.org/docs/stable/tensors.html?highlight=contiguous#torch.Tensor.contiguous>`_
        will be called so that the returning content tensors are contiguous.

        Parameters
        ----------
        dims : iterable of ints, optional
            The set of batch dimensions to be flattened. Defaults to ``None``.

        Returns
        -------
        Message
            The flattened message of `self`.
        """
        assert dims is None or (isinstance(dims, Iterable) and
                                all(isinstance(dim, int) and
                                    -len(self.b_shape) <= dim <= len(self.b_shape) - 1 for dim in dims))

        # Translate dim value to positive if it's negative
        dims = list(len(self.b_shape) + dim if dim < 0 else dim for dim in dims) if dims is not None else \
            list(range(len(self.b_shape)))
        other_dims = list(i for i in range(len(self.b_shape)) if i not in dims)
        # Get new batch shape.
        new_b_shape = torch.Size(list(self.b_shape[i] for i in other_dims)) + \
            torch.Size([np.prod(np.array(self.b_shape)[dims])])

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            perm_order = other_dims + dims + [len(self.b_shape)]
            new_parameter = new_parameter.permute(perm_order)
            new_parameter = torch.flatten(new_parameter, start_dim=len(other_dims), end_dim=len(self.b_shape) - 1)
            new_parameter = new_parameter.contiguous()
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            perm_order = other_dims + dims + list(range(len(self.b_shape), len(self.b_shape) + len(self.s_shape)))
            new_weight = new_weight.permute(perm_order)
            new_weight = torch.flatten(new_weight, start_dim=len(other_dims), end_dim=len(self.b_shape) - 1)
            new_weight = new_weight.contiguous()

        new_msg = Message(self.type,
                          new_b_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_reshape(self, new_batch_shape):
        """Returns a message with the same underlying data as self, but with the specified `new_batch_shape`.

        Parameters
        ----------
        new_batch_shape : iterable of int, or torch.Size
            The target batch shape.

        Returns
        --------
        Message
            A reshaped message from `self` with new batch shape.

        See Also
        --------
        This method is a mimic of
        `torch.reshape() <https://pytorch.org/docs/stable/torch.html?highlight=reshape#torch.reshape>`_
        """
        assert isinstance(new_batch_shape, torch.Size) or \
            (isinstance(new_batch_shape, Iterable) and all(isinstance(s, int) for s in new_batch_shape))

        new_batch_shape = torch.Size(list(new_batch_shape)) if not isinstance(new_batch_shape, torch.Size) else \
            new_batch_shape

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameter = torch.reshape(new_parameter, new_batch_shape + self.p_shape)
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            new_weight = torch.reshape(new_weight, new_batch_shape + self.s_shape)

        new_msg = Message(self.type,
                          new_batch_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def batch_expand(self, new_batch_shape):
        """Returns a new view of `self` with singleton batch dimensions expanded to a larger size.

        Passing a -1 as the size for a batch dimension means not changing the size of that batch dimension.

        Expanding `self` would not allocate new memory for `self`'s tensor contents, but would create a new view on the
        existing tensors. Any dimension of size 1 can be expanded to an arbitrary value without allocating new memory.

        Note that `new_batch_shape` is relative to the batch dimensions only.

        `contiguous() <https://pytorch.org/docs/stable/tensors.html?highlight=contiguous#torch.Tensor.contiguous>`_
        will be called so that the returning content tensors are contiguous.

        Parameters
        ----------
        new_batch_shape : iterable of int, or torch.Size
            The target expanded batch shape. Must have the same length as `self`'s current batch shape.

        Returns
        -------
        Message
            An expanded message from `self`.

        See Also
        --------
        This method is a mimic of
        `torch.Tensor.expand() <https://pytorch.org/docs/stable/tensors.html?highlight=expand#torch.Tensor.expand>`_.
        """
        assert isinstance(new_batch_shape, torch.Size) or \
            (isinstance(new_batch_shape, Iterable) and all(isinstance(s, int) for s in new_batch_shape))
        assert len(list(new_batch_shape)) == len(self.b_shape)

        new_batch_shape = torch.Size(list(new_batch_shape)) if not isinstance(new_batch_shape, torch.Size) else \
            new_batch_shape

        # Replace any -1 entry in new_batch_shape with size of the corresponding dimension of self
        new_batch_shape = torch.Size([s if s != -1 else self.b_shape[i] for i, s in enumerate(new_batch_shape)])

        new_parameter = self.parameter
        new_particles = self.particles
        new_weight = self.weight
        new_log_densities = self.log_densities

        if isinstance(new_parameter, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_shape = new_batch_shape + self.p_shape
            new_parameter = new_parameter.expand(new_shape).contiguous()
        if isinstance(new_weight, torch.Tensor):
            # weights has shape (b_shape + s_shape)
            new_shape = new_batch_shape + self.s_shape
            new_weight = new_weight.expand(new_shape).contiguous()

        new_msg = Message(self.type,
                          new_batch_shape, self.p_shape, self.s_shape, self.e_shape,
                          new_parameter, new_particles, new_weight, new_log_densities,
                          device=self.device, **self.attr)
        return new_msg

    """
        Methods for Operations on Message Events 
    """
    def event_transform(self, trans):
        """Applies a transformation on the `self`'s event values. Returns the transformed message.

        `self` contents will be cloned before being passed to the transformed message.

        .. note::

            For now, only Particles message support transformations. `reduce_type()` will first be called to eliminate
            the parameter components before performing the transformation.

        The adjustment made to the particle values and log sampling densities:

        * Apply the transformation directly on the particle tensors in ``self.particles``.
        * Log sampling density tensors in ``self.log_densities`` will be adjusted by adding the log absolute determinant
          of the Jacobian of the transformation::

             log P(Y) = log P(X) + log |det (dX / dY)|

        * Weights are kept the same, but the tensor will be cloned.

        The transformation's event dimensions should be equal to or less than 1, i.e., ``trans.event_dim <= 1``.

        Parameters
        ----------
        trans : torch.distributions.transforms.Transform
            The transformation object

        Returns
        -------
        Message
            The transformed message.

        See Also
        --------
        `torch.distributions.Transform <https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.Transform>`_
        """
        assert isinstance(trans, Transform)
        assert trans.event_dim <= 1
        assert MessageType.Particles in self.type

        # First clone and reduce
        cloned_msg = self.reduce_type(MessageType.Particles)
        new_particles = tuple(trans(p) for p in cloned_msg.particles)

        new_log_densities = []
        for i, d in enumerate(cloned_msg.log_densities):
            new_d = d - trans.log_abs_det_jacobian(cloned_msg.particles[i], new_particles[i]).sum(-1) \
                if trans.event_dim == 0 else d - trans.log_abs_det_jacobian(cloned_msg.particles[i], new_particles[i])
            new_log_densities.append(new_d)
        # new_log_densities = tuple(d - trans.log_abs_det_jacobian(cloned_msg.particles[i], new_particles[i]).sum(-1)
        #                           if trans.event_dim == 0 else d - trans.log_abs_det_jacobian(cloned_msg.particles[i],
        #                                                                                       new_particles[i])
        #                           for i, d in enumerate(cloned_msg.log_densities))

        new_msg = Message(cloned_msg.type,
                          batch_shape=cloned_msg.b_shape,
                          sample_shape=cloned_msg.s_shape, event_shape=cloned_msg.e_shape,
                          particles=new_particles, weight=cloned_msg.weight,
                          log_densities=new_log_densities,
                          device=self.device, **cloned_msg.attr)
        return new_msg

    def event_reweight(self, target_log_prob):
        """Returns a new message with the same type of `self` with the same particle values and log sampling densities
        as `self`, but a different weight tensor, derived from importance weighting `target_log_prob` against stored
        log sampling density tensors in ``self.log_densities``.

        `self` 's type must be either ``MessageType.Particles`` or ``MessageType.Both`` to support this method.

        Parameters
        ----------
        target_log_prob : torch.Tensor
            The batched log pdf of the `self` particles w.r.t. to the batched target distributions the new message is
            to encode. Should have shape ``(self.b_shape + self.s_shape)``.

        Returns
        -------
        Message
            A new importance-reweighted message with the same type and components as `self` except the importance
            weight.

        Raises
        ------
        AssertionError
            If `self` has neither type `MessageType.Particles`` nor type ``MessageType.Both``.

        Notes
        -----
        The importance weighting procedure can be summarized in two steps::

            log_ratio = target_log_pdf - joint_log_density
            new_weight = normalize(exp(log_ratio))

        Some remarks:

        * ``joint_log_density`` here refers to the joint log sampling density of the combinatorially concatenated
          marginal event particles in ``self.particles``. Therefore, if there are multiple random variables, this
          quantity is derived by first expanding each marginal log sampling density tensor in ``self.log_densities``
          to the full sampling dimensions, then taking the sum over all such expanded log density tensor.
        * The last step guarantees that ``new_weight`` sums to 1 across sampling dimensions. Note that this step is not
          explicitly implemented in this method; we assume it is taken care of by Message class constructor.
        """
        assert MessageType.Particles in self.type
        assert isinstance(target_log_prob, torch.Tensor) and target_log_prob.shape == self.b_shape + self.s_shape

        # Obtain joint sampling density. Should have shape (self.s_shape)
        if len(self.s_shape) == 1:
            joint_density = self.log_densities[0]
        else:
            exp_den = []
            for i, d in enumerate(self.log_densities):
                dims = ([1] * (len(self.s_shape) - 1))
                dims.insert(i, -1)
                exp_den.append(d.view(dims).expand(self.s_shape))
            joint_density = sum(exp_den)

        # Make joint_density broadcastable by prepending batch dimensions
        dims = [1] * len(self.b_shape) + list(self.s_shape)
        joint_density = joint_density.view(dims)
        new_weight = torch.exp(target_log_prob - joint_density)    # Unweighted

        new_msg = Message(self.type,
                          batch_shape=self.b_shape, param_shape=self.p_shape,
                          sample_shape=self.s_shape, event_shape=self.e_shape,
                          parameter=self.parameter,
                          particles=self.particles, weight=new_weight, log_densities=self.log_densities,
                          device=self.device, **self.attr)
        return new_msg

    def event_marginalize(self, event_dim):
        """Returns a message from `self` where the event dimension specified bv `event_dim` is marginalized,
        corresponding to marginalizing the corresponding random variable.

        Only messages with particles support this operation. If `self`'s message type is ``MessageType.Both``, a
        ``MessageType.Particles`` type message will be returned, where the parameter of `self` is discarded.

        Parameters
        ----------
        event_dim : int
            Which event dimension / random variable to be marginalized over. Can accept a value in the range
            ``[-len(event_shape), len(event_shape) - 1]``.

        Returns
        -------
        Message
            A ``MessageType.Particles`` type message where the `event_dim` th event dimension is marginalized over.

        Raises
        ------
        AssertionError
            If `self` does not contain particles.
        AssertionError
            If `self`'s ``len(event_shape)`` is 1, i.e., currently only one event dimension, but still this method is
            called to marginalize the only left event dimension.

        Notes
        -----
        Regarding the implementation:

        Marginalization of the particles is implemented by simply discarding the target particle value tensor as well as
        its corresponding log sampling density tensor, and summing over the target prob tensor over the event dimension.
        The target prob tensor is recovered by multiplying the weight tensor with the exponential of the cross product
        of all of the marginal log sampling density tensor.

        Note that the target prob tensor recovered in this way is **NOT** the
        actual probability w.r.t. the target distributions, but one that is proportional to that up to a normalization
        constant factor.
        """
        assert isinstance(event_dim, int) and -len(self.e_shape) <= event_dim <= len(self.e_shape) - 1
        assert MessageType.Particles in self.type, \
            "Only message with particles can be marginalized over. Marginalization over distribution parameter is not" \
            " well defined."
        assert len(self.e_shape) > 1, \
            "Attempting to marginalize over a message with one a single event dimension."

        # Convert event_dim to positive if it's negative
        event_dim = len(self.e_shape) + event_dim if event_dim < 0 else event_dim

        # Discard the target particle
        new_particles = tuple(list(self.particles[:event_dim]) + list(self.particles[event_dim + 1:]))
        new_densities = tuple(list(self.log_densities[:event_dim]) + list(self.log_densities[event_dim + 1:]))
        new_s_shape = self.s_shape[:event_dim] + self.s_shape[event_dim + 1:]
        new_e_shape = self.e_shape[:event_dim] + self.e_shape[event_dim + 1:]

        # Recover target prob
        # First take cross product of all marginal sampling density
        expand_log_den = []
        for j, d in enumerate(self.log_densities):
            # view_dim = [-1] * (len(self.e_shape) - 1)
            # view_dim.insert(j, self.s_shape[j])
            # expand_log_den.append(d.view(view_dim))
            expanded_d = d
            for i in range(j):
                expanded_d = expanded_d.unsqueeze(0)
            for i in range(j + 1, self.num_rvs):
                expanded_d = expanded_d.unsqueeze(-1)
            expand_log_den.append(expanded_d)

        # Take joint sum and exponentialize, which is equivalent to cross product.
        joint_density = torch.exp(sum(expand_log_den))
        # Now expand dimensions even more to full batch dimensions. Resulting shape should be (b_shape + s_shape)
        view_dim = [1] * len(self.b_shape) + list(self.s_shape)
        joint_density = joint_density.view(view_dim)

        # Recover target_prob, and sum over event_dim.
        target_prob = joint_density * self.weight
        summed_prob = torch.sum(target_prob, dim=len(self.b_shape) + event_dim)

        # Obtain new weight
        mar_expand_log_den = expand_log_den[:event_dim] + expand_log_den[event_dim + 1:]
        mar_joint_density = torch.exp(sum(mar_expand_log_den))
        view_dim = [1] * len(self.b_shape) + list(new_s_shape)
        mar_joint_density = mar_joint_density.view(view_dim)
        new_weight = summed_prob / mar_joint_density

        new_msg = Message(MessageType.Particles,
                          batch_shape=self.b_shape, sample_shape=new_s_shape, event_shape=new_e_shape,
                          particles=new_particles, weight=new_weight, log_densities=new_densities,
                          device=self.device, **self.attr)
        return new_msg

    def event_concatenate(self, cat_event_dims, target_event_dim=-1):
        """Concatenate the particle events corresponding to the event dimensions specified by `cat_event_dims`. The new
        concatenated events will be placed at `target_event_dim` dimension.

        To concatenate events means to

        1. combinatorially concatenate the particle value tensors,
        2. take the cross product of associated marginal sampling density tensors and flatten it,
        3. reshape the weight tensor into correct flattened shape.

        Note that the event dimensions will be concatenated in the order given by `cat_event_dims`.

        Only messages with particles support this operation. If `self`'s message type is ``MessageType.Both``, a
        ``MessageType.Parameter`` type message will be returned, where the parameter of `self` is discarded.

        ``.contiguous()`` will be called on the tensors to make sure the resulting particle, log density, and weight
        tensors are contiguous in memory.

        Parameters
        ----------
        cat_event_dims : iterable of int
            The list of event dimensions to be concatenated. Must have length at least 2 and at most the length of
            self's event shape (number of random variables). Each should be in range
            ``[-len(event_shape), len(event_shape) - 1]``.
        target_event_dim : int
            The target event dimension where the concatenated event will be placed. Should be in range
            ``[-len(event_shape) + k - 1, len(event_shape) - k]``, where ``k`` equals to ``len(cat_event_dims)``.

        Returns
        -------
        Message
            A ``Message.Particles`` type message where the specified event dimensions are concatenated.

        Raises
        ------
        AssertionError
            If `self` does not contain particles.
        """
        assert isinstance(cat_event_dims, Iterable)
        cat_event_dims = list(cat_event_dims)
        assert 2 <= len(cat_event_dims) <= self.num_rvs
        assert all(isinstance(d, int) and -self.num_rvs <= d <= self.num_rvs - 1 for d in cat_event_dims)
        assert isinstance(target_event_dim, int) and \
               -self.num_rvs + len(cat_event_dims) - 1 <= target_event_dim <= self.num_rvs - len(cat_event_dims)
        assert MessageType.Particles in self.type, \
            "Only message with particles can concatenate event dimensions. Such operation on distribution parameter is " \
            "not well defined."

        # Convert dims to positive values if they are negative
        cat_event_dims = [len(self.e_shape) + d if d < 0 else d for d in cat_event_dims]
        target_event_dim = len(self.e_shape) - len(cat_event_dims) + 1 + target_event_dim \
            if target_event_dim < 0 else target_event_dim

        # Collect elements to be concatenated, in the order given by cat_event_dims. Also the pre-flattened shape
        cat_particles = list(self.particles[i] for i in cat_event_dims)
        cat_densities = list(self.log_densities[i] for i in cat_event_dims)
        cat_s_shape = torch.Size([self.s_shape[i] for i in cat_event_dims])
        # Collect the residues that are not to be concatenated
        res_particles = list(self.particles[i] for i in range(len(self.s_shape)) if i not in cat_event_dims)
        res_densities = list(self.log_densities[i] for i in range(len(self.s_shape)) if i not in cat_event_dims)
        # New shapes
        new_s_shape = [self.s_shape[i] for i in range(len(self.s_shape)) if i not in cat_event_dims]
        new_s_shape.insert(target_event_dim, np.prod([self.s_shape[j] for j in cat_event_dims]))
        new_s_shape = torch.Size(new_s_shape)
        new_e_shape = [self.e_shape[i] for i in range(len(self.e_shape)) if i not in cat_event_dims]
        new_e_shape.insert(target_event_dim, sum([self.e_shape[j] for j in cat_event_dims]))
        new_e_shape = torch.Size(new_e_shape)

        # Combinatorially concatenate particle values. Flatten result and insert to the rest to form new particle tuple
        comb_particles = KnowledgeServer.combinatorial_cat(cat_particles)
        assert comb_particles.shape[:-1] == cat_s_shape
        flat_particles = comb_particles.view(-1, comb_particles.shape[-1]) .contiguous()       # flatten
        new_particles = res_particles[:target_event_dim] + [flat_particles] + res_particles[target_event_dim:]

        # Take cross product of marginal sampling densities and flatten.
        expand_log_den = []
        for j, d in enumerate(cat_densities):
            for i in range(j):
                d = d.unsqueeze(0)
            for i in range(j + 1, len(cat_densities)):
                d = d.unsqueeze(-1)
            expand_log_den.append(d)

        joint_log_den = sum(expand_log_den)
        assert joint_log_den.shape == cat_s_shape
        flat_log_den = joint_log_den.view(-1).contiguous()       # flatten
        new_densities = res_densities[:target_event_dim] + [flat_log_den] + res_densities[target_event_dim:]

        # Reshape weight tensor into correct flattened shape
        # First permute cat_event_dims to the last dimensions
        new_weight = self.weight
        if isinstance(new_weight, torch.Tensor):
            # of shape (b_shape + sample_shape)
            b_cat_event_dims = list(len(self.b_shape) + d for d in cat_event_dims)     # Account for batch dims at front
            b_target_event_dim = len(self.b_shape) + target_event_dim

            perm_order = list(i for i in range(len(self.b_shape + self.s_shape)) if i not in b_cat_event_dims) + \
                         b_cat_event_dims
            new_weight = new_weight.permute(perm_order).contiguous()

            # for dim in b_cat_event_dims:
            #     perm_order = list(d for d in range(new_weight.dim()) if d != dim) + [dim]
            #     new_weight = new_weight.permute(perm_order)
            # assert new_weight.shape[-len(cat_event_dims):] == cat_s_shape

            new_weight = new_weight.view(new_weight.shape[:-len(cat_event_dims)] + torch.Size([-1]))     # flatten

            perm_order = list(range(new_weight.dim() - 1))
            perm_order.insert(b_target_event_dim, -1)
            new_weight = new_weight.permute(perm_order).contiguous()         # Permute flattened dims to target_event_dim

        new_msg = Message(MessageType.Particles,
                          batch_shape=self.b_shape, sample_shape=new_s_shape, event_shape=new_e_shape,
                          particles=new_particles, weight=new_weight, log_densities=new_densities,
                          device=self.device, **self.attr)

        return new_msg



# TODO: Enum class of all the inference method
class InferenceMethod(Enum):
    BP = 0
    PARTICLE_BP = 1
    VMP = 2
    PARTICLE_VMP = 3
    EP = 4
    PARTICLE_EP = 5
