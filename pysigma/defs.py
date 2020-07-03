"""
    Basic structures in the graphical architecture
"""
import torch
from torch.distributions import Transform
from torch.distributions.constraints import Constraint
from enum import Enum, Flag, auto
from collections.abc import Iterable
import numpy as np
from copy import deepcopy
from utils import DistributionServer, KnowledgeTranslator


# Variable Metatypes and Variable for general inference
class VariableMetatype(Enum):
    Indexing = 0  # Particle indexing variable, first dimension of a message
    Relational = 1  # Relational variable, second set of dimensions of a message
    Random = 2  # Random variable, last set of dimensions of a message
    Parameter = 3  # Parameter variable, last dimensions of a distribution's parameter tensor


class Variable:
    """
        Variable as in variable nodes in the graphical architecture. Store information about this variable such as
            its meta-type and dimension size
        The equality testing is used for matching variables in Alpha-Beta graphs. Two variables are equal if and only
            if ALL of the fields are equal.

    """

    def __init__(self, name, metatype, size, value_constraints=None):
        """
            Instantiate a variable. Optionally indicates a set of value constraints if and only if variable is Random
                metatype.

            :param name:                a str. The name of this variable
            :param metatype:            VariableMetatype. The metatype of this variable: Indexing, Relational, Random,
                                            or Parameter
            :param size:                an int. The size of the dimension this variable corresponds to
            :param value_constraints:   an Iterable of torch.distributions.constraints.Constraint instances. The set of
                                            value constraints that determine the value range (support) of this random
                                            variable. Should specify if and only if metatype is Random.
        """
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
        self.constraints = value_constraints

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

    def __hash__(self):
        # override so that hash value of the string representation of the variable is used
        return hash(self.name + str(self.metatype) + str(self.size) + str(self.constraints))


# Generalized message type and message representation
class MessageType(Flag):
    """
        Enum class to represent message types

        The True-valued boolean relationship between types, using the 'in' operator:
            - Undefined in Undefined == Undefined in Parameter == Undefined in Particles == Undefined in Both == True
            - Parameter in Parameter == Undefined in Both == True
            - Particles in Particles == Undefined in Both == True
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
    batch_shape : torch.Size
        The size of the batch dimensions. Must be a shape of **at least** length 1.
    param_shape : torch.Size, optional
        The size of the parameter dimension of `parameter`. Must specify if `msg_type` is ``MessageType.Parameter``.
        Must be a shape of **exactly** length 1.
    sample_shape : torch.Size, optional
        The size of the sample dimensions of each particle tensor in `particles` respectively in order. Must specify if
        message type is ``MessageType.Particle``. Must be a shape of **at least** length 1.
    event_shape : torch.Size, optional
        The size of the event dimensions of each particle tensor in `particles` respectively in order. Must specify if
        message type is ``MessageType.Particle``. Must be a shape of **at least** length 1.
    parameter : torch.Tensor or an int of 0, optional
        The parameter tensor to the batch of distributions this message is encoding. Must specify if the message type is
        ``MessageType.Parameter``. A torch.Tensor of shape ``batch_shape + param_shape`` if the parameters do not
        represent the identity in the parameter vector space. Alternatively, can be an int of 0 to specify the identity.
        Default to an int of 0.
    particles : iterable of torch.Tensor, optional
        The list of particles representing events w.r.t. each random variable respectively whose collective joint
        distribution this message is encoding. Must specify if the message type is ``MessageType.Particles``. The jth
        entry of the iterable should have shape ``sample_shape[j] + event_shape[j]``.
    weights : torch.Tensor or an int of 1, optional
        The importance weight tensor that, when multiplied with the exponential of the cross product of the log sampling
        densities in `log_densities`, yields the pdf of each combined particle w.r.t. the target distribution that this
        message is encoding. Must specify if the message type is ``MessageType.Particles``. If the weights are
        non-uniform, must be a **positively valued** tensor of shape ``batch_shape + sample_shape``. The supplied tensor
        will be normalized during initialization so that it sums to 1 across the subspace spanned by the sample
        dimensions. Alternatively, can be an int of 1 to specify the uniform weight. Default to 1.
    log_densities : iterable of torch.Tensor, optional
        The jth entry in the iterable represents the log pdf of the jth particle in `particles` w.r.t. the (marginal)
        sampling distribution from which the jth particle was originally drawn. Must specify if the message type is
        ``MessageType.Particles``. The jth entry must have shape ``sample_shape[j]``.
    **kwargs
        Other keyword arguments that specify special attributes of the message. Will be deep copied when the message is
        cloned.

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
        List of particle value tensors
    weight : torch.Tensor or None
        Particle weight tensor
    log_densities : list of torch.Tensor or None
        List of particles log sampling tensors
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
          particles. In addition, results from these two operations are normalized so that the weight tensor
          sums to 1 across the sample dimensions.
        * *Addition* operation is defined as element-wise multiplication of particle weights tensors, up to a
          normalization factor.
        * *Scalar Multiplication* is defined as taking elements of the particle weights tensor to the power
          of the scalar, up to a normalization factor.
        * 1 is treated as the identity element for the operations.
        * Note that it is provably correct that the weighs with above operations form a vector space. The proof idea is
          to consider the log quotient space over one dimension, which reduces to standard real space with one less
          dimension.

    Accordingly, the '+' and '*' operator are overloaded according the to the specifications above.
    """
    def __init__(self, msg_type,
                 batch_shape=None, param_shape=None, sample_shape=None, event_shape=None,
                 parameter=0, particles=None, weight=1, log_densities=None, **kwargs):
        assert isinstance(msg_type, MessageType)
        assert batch_shape is None or isinstance(batch_shape, torch.Size) and len(batch_shape) >= 1
        assert param_shape is None or isinstance(param_shape, torch.Size) and len(param_shape) == 1
        assert sample_shape is None or isinstance(sample_shape, torch.Size) and len(sample_shape) >= 1
        assert event_shape is None or isinstance(event_shape, torch.Size) and len(event_shape) == len(sample_shape)

        assert parameter == 0 or isinstance(parameter, torch.Tensor)
        assert particles is None or (isinstance(particles, Iterable) and
                                     all(isinstance(p, torch.Tensor) for p in particles))
        assert weight == 1 or isinstance(weight, torch.Tensor)
        assert log_densities is None or (isinstance(log_densities, Iterable) and
                                         all(isinstance(d, torch.Tensor) for d in log_densities))

        # Message type, of type MessageType
        self.type = msg_type
        # Parameter
        self.parameter = parameter
        # Particle list
        self.particles = list(particles) if particles is not None else None
        self.weight = weight
        self.log_densities = list(log_densities) if log_densities is not None else None
        # Additional important attributes
        self.attr = kwargs
        # Shapes.
        self.b_shape = batch_shape
        self.p_shape = param_shape
        self.s_shape = sample_shape
        self.e_shape = event_shape

        # Check whether necessary arguments are provided
        if MessageType.Parameter in self.type:
            assert self.b_shape is not None and self.p_shape is not None
            assert self.parameter is not None
        if MessageType.Particles in self.type:
            assert self.s_shape is not None and self.b_shape is not None and self.e_shape is not None
            assert self.particles is not None
            assert self.weight is not None
            assert self.log_densities is not None

        # Check shape and values. Adjust if necessary
        if isinstance(self.parameter, torch.Tensor):
            # Parameter tensor should have shape (b_shape + p_shape)
            assert self.b_shape + self.p_shape == self.parameter.shape
        if particles is not None:
            # jth particle tensor should have shape (s_shape[j] + e_shape[j])
            assert len(self.particles) == len(self.s_shape) == len(self.e_shape)
            assert all(self.s_shape[j] + self.e_shape[j] == p.shape for j, p in enumerate(self.particles))
        if isinstance(self.weight, torch.Tensor):
            # Weights tensor should have shape (b_shape + s_shape)
            assert self.b_shape + self.s_shape == self.weight.shape
            # Check that values are non-negative
            assert torch.all(self.weight > 0), "Found negative values in particle weights. Minimum value: {}" \
                .format(torch.min(self.weight))
            # Normalize the values so that weights sum to 1 across sample dimension
            sample_dims = list(range(len(self.b_shape), len(self.b_shape) + len(self.s_shape)))
            self.weight /= self.weight.sum(dim=sample_dims, keepdim=True)
        if self.log_densities is not None:
            # jth log density tensor vector should have shape (s_shape[j])
            assert len(self.log_densities) == len(self.s_shape) == len(self.e_shape)
            assert all(self.s_shape[j] == d.shape for j, d in enumerate(self.log_densities))

    """
        Overload arithmetic operators
    """

    def __add__(self, other):
        """  Overloads the addition operation '+'

        Implements the semantics of addition operation as in vector spaces, but the computational operations used to
        implement the semantics are different for different message contents. See
        :ref:`Message class notes regarding arithmetic structures<message-arithmetic-structures-notes>`
        for more details.

        Only messages with compatible types can be added. This means a ``MessageType.Parameter`` type message can only
        be added with one of type ``MessageType.Parameter`` or ``MessageType.Both``, and similarly a
        ``MessageType.Particles`` type message can only be added with one of type ``MessageType.Particles`` or
        ``MessageType.Both``. ``MessageType.Both`` type message can be added with any other type except
        ``MessageType.Undefined``, and in any case a ``MessageType.Undefined`` type message cannot be added.

        There are more restrictions for ``MessageType.Particles`` type messages. Messages of such type can only be
        added together if their ``particles`` and ``log_densities`` fields are equal.

        All contents are first cloned before taking the operations and returning the result.

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
            If contents of `self` and `other` have conflicting shapes.
        AssertionError
            If `self` and `other` have particles message contents to be added, but their particle values do not match,
            or their log sampling density tensors do not match.

        Warnings
        --------
        Note that all auxiliary attributes stored in ``attr``, supplied via additional keyword arguments in the Message
        class constructor, of both `self` and `other` will be discarded in the returning message.
        """
        assert isinstance(other, Message), "Message can only be added with another Message"
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

        # Addition for Parameter type
        new_msg = None
        if MessageType.Parameter in s_type:
            assert self.b_shape == other.b_shape and self.p_shape == other.p_shape, \
                "Only Messages with the same shape can be added together. The messages being added are of Parameter " \
                "type. Found first message with (batch_shape, param_shape) = '{}', and second message with " \
                "(batch_shape, param_shape) = '{}'".format((self.b_shape, self.p_shape), (other.b_shape, other.p_shape))
            # Tensor addition
            new_parameters = self.parameter + other.parameter

            new_msg = Message(self.type, batch_shape=self.b_shape, param_shape=self.p_shape, parameters=new_parameters)

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
            cloned_particles = list(p.clone() for p in self.particles)
            cloned_log_densities = list(d.clone() for d in self.log_densities)

            new_msg = Message(self.type,
                              sample_shape=self.s_shape, batch_shape=self.b_shape, event_shape=self.e_shape,
                              particles=cloned_particles, weights=new_weights, log_density=cloned_log_densities)

        return new_msg

    def __iadd__(self, other):
        """
            Overloading self-addition operator '+='
        """
        return self.__add__(other)

    def __mul__(self, other):
        """
            Overloading multiplication operator '*'. Implements scalar multiplication semantics.

            The scalar can be of type int, float, or torch.Tensor. If it is a torch.Tensor, can be a singleton tensor
                representing a single scalar, or a tensor of shape batch_shape representing a batched scalars.
        """
        assert isinstance(other, (int, float, torch.Tensor)), \
            "Message can only be multiplied with a scalar. The scalar can be of int, float or torch.Tensor type. " \
            "Instead found: '{}'".format(type(other))
        if isinstance(other, torch.Tensor):
            assert other.shape == torch.Size([]) or other.shape == self.b_shape, \
                "If the scalar is a torch.Tensor, must be either a singleton tensor or a tensor with the same shape " \
                "as the Message's batch shape: '{}'. Instead found: '{}'".format(self.b_shape, other.shape)
        # Undefined type cannot be scalar multiplied
        assert self.type is not MessageType.Undefined, \
            "Message of undefined type cannot be scalar multiplied. The message has type '{}'" \
            .format(self.type, other.type)

        # Expand scalar tensor dimension if it is a batched scalars
        b_p_other = other
        s_b_other = other
        if isinstance(other, torch.Tensor) and other.dim() > 0:
            if self.type is MessageType.Parameter:
                b_p_other = torch.unsqueeze(b_p_other, dim=-1)

            if self.type is MessageType.Particles:
                s_b_other = torch.unsqueeze(s_b_other, dim=0)

        # Scalar multiplication for Parameter messages
        new_msg = None
        if MessageType.Parameter in self.type:
            new_parameters = b_p_other * self.parameters
            new_msg = Message(self.type, batch_shape=self.b_shape, param_shape=self.p_shape, parameters=new_parameters,
                              **self.attr)

        # Scalar multiplication for Particles messages
        if MessageType.Particles in self.type:
            # The result of scalar multiplication with uniform weights is still uniform, so only process non-uniform
            #   weights
            new_weights = self.weights
            if isinstance(new_weights, torch.Tensor):
                # Extract int/float from singleton scalar tensor
                if isinstance(s_b_other, torch.Tensor) and s_b_other.dim() == 0:
                    s_b_other = s_b_other.item()
                # Take weights tensor to the power of the scaler
                new_weights = torch.pow(new_weights, s_b_other)

            new_msg = Message(self.type,
                              sample_shape=self.s_shape, batch_shape=self.b_shape, event_shape=self.e_shape,
                              particles=self.particles, weights=new_weights, log_density=self.log_density, **self.attr)

        return new_msg

    def __imul__(self, other):
        """
            Overloading self-multiplication operator '*='
        """
        return self.__mul__(other)

    def __str__(self):
        if self.type == MessageType.Parameter:
            b_shape_str = str(list(self.b_shape))
            p_shape_str = str(list(self.p_shape))
            parameters_str = str(list(self.parameters.tolist()))

            return f"Type: Parameter\nBatch_Shape: {b_shape_str}\nParameter_Shape: {p_shape_str}\n" \
                   f"Parameters{parameters_str}"

        else:
            s_shape_str = str(list(self.s_shape))
            b_shape_str = str(list(self.b_shape))
            e_shape_str = str(list(self.e_shape))
            particles_str = str(list(self.particles.tolist()))
            weights_str = str(list(self.weights.tolist()))
            log_density_str = str(list(self.log_density.tolist()))

            return f"Type: Particles\nSample_Shape: {s_shape_str}\nBatch_Shape: {b_shape_str}\n" \
                   f"Event_Shape: {e_shape_str}\nParticles: {particles_str}\n" \
                   f"Weights: {weights_str}\nLog_Density: {log_density_str}"

    def size(self):
        """
            Returns a tuple of the message's shapes:
                (sample_shape, batch_shape, param_shape, event_shape)
        """
        return self.s_shape, self.b_shape, self.p_shape, self.e_shape

    def same_size_as(self, other):
        """
            Check if self has the same shape as the other message. Return True if so.
        """
        assert isinstance(other, Message)
        return self.size() == other.size()

    """
            Methods for operations on the message instance itself.
        """

    def reduce_type(self, msg_type):
        """
            Return a 'msg_type' type reduced self message, where irrelevant components w.r.t. 'msg_type' is removed, and
                the relevant components are retained and cloned.

            Return self and do nothing if msg_type is self type.

            :param msg_type:        MessageType.Parameter or MessageType.Particles.
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
            new_msg = Message(cloned_msg.type, param_shape=cloned_msg.p_shape, batch_shape=cloned_msg.b_shape,
                              parameters=cloned_msg.parameters, **cloned_msg.attr)
        else:
            new_msg = Message(cloned_msg.type, sample_shape=cloned_msg.s_shape, batch_shape=cloned_msg.b_shape,
                              event_shape=cloned_msg.e_shape, particles=cloned_msg.particles,
                              weights=cloned_msg.weights, log_density=cloned_msg.log_density, **cloned_msg.attr)
        return new_msg

    def clone(self):
        """
            Return a cloned message from self. Guarantees that every content is deep-copied. Tensors will be cloned and
                dictionaries will be deep-copied.
        """
        parameters = self.parameters
        particles = self.particles
        weights = self.weights
        log_density = self.log_density
        attr = self.attr

        if isinstance(parameters, torch.Tensor):
            parameters = parameters.clone()
        if isinstance(particles, torch.Tensor):
            particles = particles.clone()
        if isinstance(weights, torch.Tensor):
            weights = weights.clone()
        if isinstance(log_density, torch.Tensor):
            log_density = log_density.clone()
        if self.attr is not None:
            attr = deepcopy(self.attr)

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, self.b_shape, self.e_shape,
                          parameters, particles, weights, log_density, **attr)
        return new_msg

    """
        Methods for batch dimension manipulations. 
    """

    def batch_permute(self, target_dims):
        """
            Returns a permuted message whose tensor attributes are permuted from the original ones w.r.t. 'target_dims'
                in the batch dimensions.
            Note that target_dims is relative to the batch dimension. Its values should be within the range
                    [-len(batch_shape), len(batch_shape) - 1]

            contiguous() will be called before return to make sure the resulting content tensors are contiguous

            This method is a mimic of torch.Tensor.permute()

            :param target_dims:     list of ints. The desired ordering batch dimensions
        """
        assert isinstance(target_dims, list) and all(isinstance(i, int) for i in target_dims)
        assert len(target_dims) == len(self.b_shape) and \
               all(-len(self.b_shape) <= i <= len(self.b_shape) - 1 for i in target_dims)

        # Translate negative dims to nonnegative value
        pos_target_dims = list(len(target_dims) + i if i < 0 else i for i in target_dims)
        # Permuted batch shape
        new_b_shape = torch.Size(list(self.b_shape[pos_target_dims[i]] for i in range(len(self.b_shape))))
        # Permute order for sample and batch dimensions together.
        #   Add 1 to values in pos_target_dims because there's a single sample dimensions at front
        s_b_dims = [0, ] + list(i + 1 for i in pos_target_dims)
        # Permute order for batch and parameter dimensions together
        b_p_dims = pos_target_dims + [len(self.b_shape)]

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameters = self.parameters.permute(b_p_dims)
            new_parameters = new_parameters.contiguous()
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = self.weights.permute(s_b_dims)
            new_weights = new_weights.contiguous()

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_unsqueeze(self, dim):
        """
            Returns a new message with a dimension of size one inserted at the specified batch dimension, similar to
                torch.unsqueeze().
            A 'dim' value within the range [-len(batch_shape) - 1, len(batch_shape) + 1] can be used. Note that 'dim'
                is relative to the batch dimension only.

            This method is a mimic of torch.unsqueeze()

            :param dim:     an int. The place where the new dimension of size one will be inserted at.
        """
        assert isinstance(dim, int) and -len(self.b_shape) - 1 <= dim <= len(self.b_shape) + 1

        # Translate dim to positive value if it is negative
        if dim < 0:
            dim = len(self.b_shape) + dim + 1
        # For message contents who has a sample dimension at front, add 1 to dim
        s_dim = dim + 1
        # Get new batch shape
        new_b_shape = self.b_shape[:dim] + torch.Size([1]) + self.b_shape[dim:]

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameters = torch.unsqueeze(self.parameters, dim)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = torch.unsqueeze(self.weights, s_dim)

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_index_select(self, dim, index):
        """
            Returns a new Message which indexes the input message along batch dimension dim using the entries in index
                which is a LongTensor.
            A 'dim' value within the range [-len(batch_shape), len(batch_shape) - 1] can be used. Note that 'dim'
                is relative to the batch dimension only.

            This method is a mimic of batch_index_select()

            :param dim:     an int. The dimension along which entries will be selected according to 'index'
            :param index:   torch.LongTensor. The index of entries along 'dim' to be selected
        """
        assert isinstance(dim, int) and -len(self.b_shape) <= dim <= len(self.b_shape) - 1
        assert isinstance(index, torch.LongTensor) and index.dim() == 1

        # Translate dim to positive value if it is negative
        if dim < 0:
            dim = len(self.b_shape) + dim
        # For message contents who has a sample dimension at front, add 1 to dim
        s_dim = dim + 1
        # Get new batch shape
        new_b_shape = self.b_shape[:dim] + index.shape + self.b_shape[dim + 1:]

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameters = torch.index_select(self.parameters, dim, index)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = torch.index_select(self.weights, s_dim, index)

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_index_put(self, dim, index):
        """
            Returns a new Message whose entries along the dimension 'dim' are slices from self message and are indexed
                by 'index'. Effectively, along the dimension 'dim':
                    result_msg[..., index[i], ...] = val[i]

            For slices in the new message not referenced by 'index', they will be filled with identity values. For
                Parameter type message, the identity value is 0; for Particles type message, the identity value is 1,
                up to a normalization factor.

            This method is the inverted version of batch_index_select(). There is no direct counterpart to this method
                in PyTorch.

            :param dim:     an int. Specifying a dimension of the message. Should be in range
                                        [-len(batch_shape), len(batch_shape) - 1]
            :param index:   a LongTensor. Specifying the indices along the specified dimension of the returned message.
                                Entries must be non-negative.
        """
        assert isinstance(dim, int) and -len(self.b_shape) <= dim <= len(self.b_shape) - 1
        assert isinstance(index, torch.LongTensor) and index.dim() == 1 and torch.all(index >= 0)

        # Translate dim value to positive if it's negative
        dim = len(self.b_shape) + dim if dim < 0 else dim
        # For message contents who has a sample dimension at front, add 1 to dim
        s_dim = dim + 1
        # Get new batch shape. The size of dimension dim is determined by the maximum value in index
        new_b_shape = self.b_shape[:dim] + torch.Size([torch.max(index) + 1]) + self.b_shape[dim + 1:]

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        # To access tensor slice more easily, we swap the target dim with first dim, perform slicing and assignment on
        #   this new first dim, and swap it back
        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            # Identity value tensor
            to_fill = torch.zeros(new_b_shape + self.p_shape)
            # Transpose target dimension with the first dimension
            to_fill = torch.transpose(to_fill, dim0=0, dim1=dim)
            t_param = torch.transpose(new_parameters, dim0=0, dim1=dim)
            # Slice and assign
            to_fill[index] = t_param
            # Transpose back to get result
            new_parameters = torch.transpose(to_fill, dim0=0, dim1=dim)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            # Identity value tensor
            to_fill = torch.ones(self.s_shape + new_b_shape)
            # Transpose target dimension with the first dimension
            to_fill = torch.transpose(to_fill, dim0=0, dim1=s_dim)
            t_weights = torch.transpose(new_weights, dim0=0, dim1=s_dim)
            # Slice and assign
            to_fill[index] = t_weights
            # Transpose back to get result
            new_weights = torch.transpose(to_fill, dim0=0, dim1=s_dim)

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_diagonal(self, dim1=0, dim2=1):
        """
            Returns a partial view of self with the its diagonal elements with respect to 'dim1' and 'dim2' appended as
                a dimension at the end of the shape.
            dim values within the range [-len(batch_shape), len(batch_shape) - 1] can be used.
            Note that 'dim1' and 'dim2' are relative to the batch dimension. The appended dimension will be placed as
                the last batch dimension, but before any event or param dimension.

            contiguous() will be called before return to make sure the resulting content tensors are contiguous

            This method is a mimic of torch.diagonal(), with offset default to 0

            :param dim1:    an int. Should be in range [-len(batch_shape), len(batch_shape) - 1]
            :param dim2.    Same as 'dim1'
        """
        assert isinstance(dim1, int) and -len(self.b_shape) <= dim1 <= len(self.b_shape) - 1
        assert isinstance(dim2, int) and -len(self.b_shape) <= dim2 <= len(self.b_shape) - 1

        # Translate dim value to positive if it's negative
        dim1 = len(self.b_shape) + dim1 if dim1 < 0 else dim1
        dim2 = len(self.b_shape) + dim2 if dim2 < 0 else dim2
        # For message contents who has a sample dimension at front, add 1 to dim
        s_dim1 = dim1 + 1
        s_dim2 = dim2 + 1
        # Get new batch shape. The size of the appended diagonalized dimension should be the min of dim1 and dim2
        new_b_shape = self.b_shape[:min(dim1, dim2)] + self.b_shape[min(dim1, dim2) + 1: max(dim1, dim2)] + \
                      self.b_shape[max(dim1, dim2) + 1:] + torch.Size([min(self.b_shape[dim1], self.b_shape[dim2])])

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameters = torch.diagonal(new_parameters, dim1=dim1, dim2=dim2)
            # Swap param dimension and appended diagonal batch dimension
            new_parameters = torch.transpose(new_parameters, dim0=-1, dim1=-2)
            new_parameters = new_parameters.contiguous()
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = torch.diagonal(new_weights, dim1=s_dim1, dim2=s_dim2)
            new_weights = new_weights.contiguous()

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_diag_embed(self, diag_dim=-1, target_dim1=-2, target_dim2=-1):
        """
            Creates a message whose diagonals of certain 2D planes (dimensions specified by 'target_dim1' and
                'target_dim2') are filled by vectors of self (dimension specified by 'diag_dim'). The last dimension of
                self is chosen by default as the diagonal entries to be filled, and the last two dimensions of the new
                message are chosen by default as the 2D planes where the diagonal entries will be filled in.

            The 2D planes will be shaped as square matrices, with the size of each dimension matches the size of the
                diag_dim in self.

            The length of returned message's batch shape will be the length of original message's batch shape plus 1.

            For slots not on the diagonal of the resulting message, they will be filled with identity values. For
                Parameter type message, the identity value is 0 w.r.t. the parameter tensor, and for Particles type
                message, the identity value is 1 w.r.t. the weights tensor up to a normalization factor.

            contiguous() will be called before return to make sure the resulting content tensors are contiguous

            This method is a mimic of torch.diag_embed(), with offset default to 0 plus an additional diag_dim argument.

            :param diag_dim:        an int. Specifying a dimension of the original message. Should be in range
                                        [-len(batch_shape), len(batch_shape) - 1]
            :param target_dim1:     an int. Specifying a dimension of the returned message. Should be in range
                                        [-len(batch_shape) - 1, len(batch_shape)]
            :param target_dim2:     Same as 'target_dim1'
        """
        assert isinstance(diag_dim, int) and -len(self.b_shape) <= diag_dim <= len(self.b_shape) - 1
        assert isinstance(target_dim1, int) and -len(self.b_shape) - 1 <= target_dim1 <= len(self.b_shape)
        assert isinstance(target_dim2, int) and -len(self.b_shape) - 1 <= target_dim2 <= len(self.b_shape)

        # Translate dim value to positive if it's negative
        diag_dim = len(self.b_shape) + diag_dim if diag_dim < 0 else diag_dim
        target_dim1 = len(self.b_shape) + 1 + target_dim1 if target_dim1 < 0 else target_dim1
        target_dim2 = len(self.b_shape) + 1 + target_dim2 if target_dim2 < 0 else target_dim2
        # For message contents who has a sample dimension at front, add 1 to dim
        s_diag_dim = diag_dim + 1
        s_target_dim1 = target_dim1 + 1
        s_target_dim2 = target_dim2 + 1
        # Get new batch shape. The size of target_dim1 and target_dim2 is determined by the size of diag_dim
        diag_size = self.b_shape[diag_dim]
        other_shape = list(self.b_shape[:diag_dim] + self.b_shape[diag_dim + 1:])
        first_new_dim, second_new_dim = min(target_dim1, target_dim2), max(target_dim1, target_dim2)
        other_shape.insert(first_new_dim, diag_size)
        other_shape.insert(second_new_dim, diag_size)
        new_b_shape = torch.Size(other_shape)

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        # Tensors fist need to have the diagonal entries dimension (diag_dim) permuted to the last dimension so that it
        #   will be picked up by torch.diag_embed()
        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            perm_order = list(range(len(self.b_shape + self.p_shape)))
            perm_order.remove(diag_dim)
            perm_order.append(diag_dim)
            new_parameters = new_parameters.permute(perm_order)
            new_parameters = torch.diag_embed(new_parameters, dim1=target_dim1, dim2=target_dim2)
            new_parameters = new_parameters.contiguous()
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            # For weights, the default entries to be filled in places other than the diagonal should be 1's, so we
            #   will first fill the log of input into the diagonal and then take exponential. 0's filled by
            #   torch.diag_embed() will be transformed to 1. Note that for these uniform entries the weights will be
            #   normalized across sample dimension during initialization so no worries.
            log_weights = torch.log(new_weights)
            perm_order = list(range(len(self.s_shape + self.b_shape)))
            perm_order.remove(s_diag_dim)
            perm_order.append(s_diag_dim)
            log_weights = log_weights.permute(perm_order)
            log_weights = torch.diag_embed(log_weights, dim1=s_target_dim1, dim2=s_target_dim2)
            new_weights = torch.exp(log_weights)
            new_weights = new_weights.contiguous()

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_narrow(self, dim, length):
        """
            Returns a new message that is a narrowed version of input tensor along the dimension specified by 'dim'.
                Effectively, this method is selecting the chunk spanning [:length] along the dimension 'dim' of the
                original message. The returned message and input message share the same underlying storage.

            contiguous() will be called before return to make sure the resulting content tensors are contiguous

            This method is a mimic of torch.narrow(), with start default to 0.

            :param dim      an int. Specifying a dimension of the original message. Should be in range
                                        [-len(batch_shape), len(batch_shape) - 1]
            :param length   an int. Specifying the length of the message chunk to select. Should be in range
                                        [0, dim_size - 1]
        """
        assert isinstance(dim, int) and -len(self.b_shape) <= dim <= len(self.b_shape) - 1
        assert isinstance(length, int) and 0 <= length <= self.b_shape[dim] - 1

        # Translate dim value to positive if it's negative
        dim = len(self.b_shape) + dim if dim < 0 else dim
        # For message contents who has a sample dimension at front, add 1 to dim
        s_dim = dim + 1
        # Get new batch shape.
        new_b_shape = self.b_shape[:dim] + torch.Size([length]) + self.b_shape[dim + 1:]

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameters = torch.narrow(new_parameters, dim=dim, start=0, length=length)
            new_parameters = new_parameters.contiguous()
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = torch.narrow(new_weights, dim=s_dim, start=0, length=length)
            new_weights = new_weights.contiguous()

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_broaden(self, dim, length):
        """
            Returns a new message that is a broadened version of the input tensor along the dimension specified by
                'dim', with identity values filled in [dim_size + 1: length] along the dimension 'dim' of the original
                message. In other words, this method is concatenating an identity message to the original message along
                the dimension 'dim' so that the resulting dimension size is 'length'.

            For Parameter type message, the identity values are 0. For Particles type message, the identity values are 1
                up to a normalization factor.

            contiguous() will be called before return to make sure the resulting content tensors are contiguous

            This method is the inverted version of batch_narrow(). There is no direct counterpart to this method in
                PyTorch.

            :param dim      an int. Specifying a dimension of the original message. Should be in range
                                        [-len(batch_shape), len(batch_shape) - 1]
            :param length   an int. Specifying the length of the message chunk to select. Should be greater than the
                                current size of dimension 'dim'
        """
        assert isinstance(dim, int) and -len(self.b_shape) <= dim <= len(self.b_shape) - 1
        assert isinstance(length, int) and length > self.b_shape[dim]

        # Translate dim value to positive if it's negative
        dim = len(self.b_shape) + dim if dim < 0 else dim
        # For message contents who has a sample dimension at front, add 1 to dim
        s_dim = dim + 1
        # Get new batch shape.
        new_b_shape = self.b_shape[:dim] + torch.Size([length]) + self.b_shape[dim + 1:]

        if self.type == MessageType.Parameter:
            to_concat_shape = self.b_shape[:dim] + torch.Size([length - self.b_shape[dim]]) + \
                              self.b_shape[dim + 1:] + self.p_shape
        else:
            to_concat_shape = self.s_shape + self.b_shape[:dim] + torch.Size([length - self.b_shape[dim]]) + \
                              self.b_shape[dim + 1:]

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            to_concat = torch.zeros(to_concat_shape)
            new_parameters = torch.cat([new_parameters, to_concat], dim=dim)
            new_parameters = new_parameters.contiguous()
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            to_concat = torch.ones(to_concat_shape)
            new_weights = torch.cat([new_weights, to_concat], dim=s_dim)
            new_weights = new_weights.contiguous()

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_summarize(self, dim):
        """
            Implements the default Sum-Product summarization semantics. Summarizes over the batch dimension specified by
                'dim'. Returns a message with one less dimension.

            For Parameter message, the summarization is realized by taking the mean value of the batched parameters
                across dimension 'dim'. For particles message, this is realized by taking joint addition defined for
                particle weights, a.k.a. factor product.

            :param dim:     an int. Specifying a dimension of the original message. Should be in range
                                        [-len(batch_shape), len(batch_shape) - 1]
        """
        assert isinstance(dim, int) and -len(self.b_shape) <= dim <= len(self.b_shape) - 1

        # Translate dim value to positive if it's negative
        dim = len(self.b_shape) + dim if dim < 0 else dim
        # For message contents who has a sample dimension at front, add 1 to dim
        s_dim = dim + 1
        # Get new batch shape.
        new_b_shape = self.b_shape[:dim] + self.b_shape[dim + 1:]

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameters = torch.mean(new_parameters, dim=dim)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            # For weights, since factor product is taken, we first convert weight values to log scale, perform summation
            #   across the batch dimension, then convert back to exponential scale.
            # The normalization of resulting weights will be taken care of by message initialization
            log_weights = torch.log(new_weights)
            log_weights = torch.sum(log_weights, dim=s_dim)
            new_weights = torch.exp(log_weights)

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_flatten(self, dims=None):
        """
            Flattens the set of batch dimensions specified by 'dims' and append the flattened dimension as the last
                dimension. If 'dims' is None, will flatten all batch dimensions into a single dimension.

            contiguous() will be called before return to make sure the resulting content tensors are contiguous

            :param dims:    None or an Iterable of ints. Specifying the set of dimensions to be flattened. If given,
                                each value should be in range   [-len(batch_shape), len(batch_shape) - 1]
        """
        assert dims is None or (isinstance(dims, Iterable) and all(isinstance(dim, int) and
                                -len(self.b_shape) <= dim <= len(self.b_shape) - 1 for dim in dims))

        # Translate dim value to positive if it's negative
        dims = list(len(self.b_shape) + dim if dim < 0 else dim for dim in dims) if dims is not None else \
               list(range(len(self.b_shape)))
        other_dims = list(i for i in range(len(self.b_shape)) if i not in dims)
        # For message contents who has a sample dimension at front, add 1 to dim
        s_dims = list(dim + 1 for dim in dims)
        s_other_dims = list(dim + 1 for dim in other_dims)
        # Get new batch shape.
        new_b_shape = torch.Size(list(self.b_shape[i] for i in range(len(self.b_shape)) if i not in dims)) + \
                      torch.Size([np.prod(np.array(self.b_shape)[dims])])

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            perm_order = other_dims + dims + [len(self.b_shape)]
            new_parameters = new_parameters.permute(perm_order)
            new_parameters = torch.flatten(new_parameters, start_dim=len(other_dims), end_dim=len(self.b_shape) - 1)
            new_parameters = new_parameters.contiguous()
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            perm_order = s_other_dims + s_dims
            new_weights = new_weights.permute(perm_order)
            new_weights = torch.flatten(new_weights, start_dim=len(s_other_dims), end_dim=-1)
            new_weights = new_weights.contiguous()

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_reshape(self, new_batch_shape):
        """
            Returns a message with the same data as self, but with the specified 'new_batch_shape'.

            This method is a mimic of torch.reshape()

            :param new_batch_shape:     Iterable of python ints, or torch.Size. The target batch shape.
        """
        assert isinstance(new_batch_shape, torch.Size) or \
               (isinstance(new_batch_shape, Iterable) or all(isinstance(s, int) for s in new_batch_shape))

        new_batch_shape = torch.Size(list(new_batch_shape)) if not isinstance(new_batch_shape, torch.Tensor) else \
            new_batch_shape

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameters = torch.reshape(new_parameters, new_batch_shape + self.p_shape)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = torch.reshape(new_weights, self.s_shape + new_batch_shape)

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_batch_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    def batch_expand(self, new_batch_shape):
        """
            Returns a new view of the self message with singleton batch dimensions expanded to a larger size.

            Passing a -1 as the size for a dimension means not changing the size of that dimension.

            Expanding a message would not allocate new memory for the tensor contents, but create a new view on the
                existing tensor. Any dimension of size 1 can be expanded to an arbitrary value without allocating new
                memory.

            Note that more than one element of an expanded message may refer to a single memory location. As a result,
                in-place operations may result in incorrect behavior. Clone first before needing to write in-place to
                the message tensor contents.

            This method is a mimic of torch.expand()

            :param new_batch_shape:     Iterable of python ints, or torch.Size. The target expanded batch shape. Must
                                            have the same length as self's current batch shape.
        """
        assert isinstance(new_batch_shape, torch.Size) or \
               (isinstance(new_batch_shape, Iterable) and all(isinstance(s, int) for s in new_batch_shape))
        assert len(list(new_batch_shape)) == len(self.b_shape)

        new_batch_shape = torch.Size(list(new_batch_shape)) if not isinstance(new_batch_shape, torch.Tensor) else \
            new_batch_shape

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_shape = new_batch_shape + self.p_shape
            new_parameters = new_parameters.expand(new_shape).contiguous()
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_shape = self.s_shape + new_batch_shape
            new_weights = new_weights.expand(new_shape).contiguous()

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_batch_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density, **self.attr)
        return new_msg

    """
        Methods for Manipulations on message events 
    """
    def event_transform(self, trans):
        """
            Apply a transformation on the event values. Return the transformed message.

            Message contents will be cloned.

            For particles:
                - Apply the transformation directly on the particle values
                - Log sampling densities will be adjusted by adding the log abs determinant of the Jacobian of the
                    transformation:
                            log P(Y) = log P(X) + log |det (dX / dY)|
                - Weights are kept the same, but the tensor will be cloned.

            For parameters:
                - Raise an alert if 'dist_class' attribute is missing in self.attr
                - Query DistributionServer to obtained the transformed parameter.

            :param trans:     torch.distributions.transforms.Transform. The transformation functor.
        """
        assert isinstance(trans, Transform)

        # First clone
        cloned_msg = self.clone()
        new_parameters = cloned_msg.parameters
        new_particles = cloned_msg.particles
        new_log_density = cloned_msg.log_density

        if MessageType.Parameter in cloned_msg.type:
            assert 'dist_class' in cloned_msg.attr, \
                "Missing 'dist_class' message attribute when transforming a message that contains parameters."
            new_parameters = DistributionServer.transform_param(new_parameters, cloned_msg.attr['dist_class'], trans)
        if MessageType.Particles in cloned_msg.type:
            new_particles = trans(new_particles)
            new_log_density += trans.log_abs_det_jacobian(cloned_msg.particles, new_particles)

        new_msg = Message(cloned_msg.type,
                          cloned_msg.p_shape, cloned_msg.s_shape, cloned_msg.b_shape, cloned_msg.e_shape,
                          new_parameters, new_particles, cloned_msg.weights, new_log_density, **cloned_msg.attr)
        return new_msg

    @staticmethod
    def event_translate_2pred(msg, translator):
        """
            Translate msg's particles from PyTorch format to Cognitive format, using the given translator.

            If there are multiple r.v., will return a tuple of translated messages, elements corresponding to each r.v.
                specified in the translator. In this case, the particle weights and log sampling densities will be
                copied for each split event particles to form new messages.

            :param msg:             a Message instance. The message to be translated.
            :param translator:      a KnowledgeTranslator instance.
        """
        assert isinstance(msg, Message)
        assert isinstance(translator, KnowledgeTranslator)
        assert MessageType.Particles in msg.type

        result_particles = translator.event2pred_event(msg.particles)

        result_msgs = []
        if not isinstance(result_particles, tuple):
            assert isinstance(result_particles, torch.Tensor)
            result_particles = tuple([result_particles])
        for particles in result_particles:
            # Shape check
            assert isinstance(particles, torch.Tensor) and particles.dim() == 2 and particles.shape[0] == msg.s_shape
            # Clone message
            cloned_msg = msg.clone()
            new_msg = Message(cloned_msg.type,
                              cloned_msg.p_shape, cloned_msg.s_shape, cloned_msg.b_shape, particles.shape[1],
                              cloned_msg.parameters, particles, cloned_msg.weights, cloned_msg.log_density,
                              **cloned_msg.attr)
            result_msgs.append(new_msg)

        return tuple(result_msgs)

    @staticmethod
    def event_translate_2torch(msgs, translator):
        """
            Translate provided iterable of messages' particles from Cognitive format to form a message with particles in
                PyTorch format, using the given translator.

            The message order in the given iterable should conform to the random variable order specified in the
                translator.

            Return a single message.

            :param msgs:        an iterable of Message instances. Order should be compatible with the order of random
                                    variables specified in the given translator.
            :param translator:  a KnowledgeTranslator instance.
        """
        assert isinstance(msgs, Iterable) and all(isinstance(msg, Message) for msg in msgs)
        assert isinstance(translator, KnowledgeTranslator)
        assert all(MessageType.Particles in msg.type for msg in msgs)

        result_particles = translator.event2torch_event(tuple(msgs))






# TODO: Enum class of all the inference method
class InferenceMethod(Enum):
    BP = 0
    PARTICLE_BP = 1
    VMP = 2
    PARTICLE_VMP = 3
    EP = 4
    PARTICLE_EP = 5
