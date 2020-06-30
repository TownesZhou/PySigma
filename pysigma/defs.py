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
    """
        Message structure to support general inference.

        Two basic message types:
            - Parameter: parameters to a batch of distributions that this message is encoding
                - Contains a batched parameter tensor, of shape (batch_shape + param_shape)

            - Particles: Lists of marginally drawn particles w.r.t. each single random variable. The combination of
                particles approximates the batch of joint target distributions this message is encoding via importance
                weighting. Comprise of the following components:
                - Particle Value Tensors List:
                    Each one corresponds to particles marginally drawn w.r.t. one random variable.
                    With shape [sample_size_j, event_size_j] for the jth random variable.
                - Weight Tensor:
                    A single, positively valued tensor that encodes the importance weighting of each joint combination
                        of particles w.r.t. each joint distribution in the batch. The values should sum up to 1 across
                        the tensor subspace spanned by all sample dimensions.
                    If all weights are uniform, can use an int of 1 as the abbreviation.
                    With shape (batch_shape + sample_shape) where batch_shape is the concatenation of of batch sizes,
                        and sample_shape is the concatenation of all sample sizes.
                - Log Sampling Density Tensors List:
                    Each one corresponds to the log sampling density of the corresponding particles in the Particle
                        Value Tensors List.
                    If a particle value tensor was drawn uniformly, its corresponding log sampling density can be
                        abbreviated by an int of 1.
                    With shape [sample_size_j] for the jth random variable.

        Message content shape (when they are torch tensors), assuming there are N relational variables and M random
            variables:
            - Parameter:                    [batch_size_0, ..., batch_size_N, param_size]
            - Particle Value Tensor:        each of shape [sample_size_j, event_size_j]
            - Weight Tensor:                [batch_size_0, ..., batch_size_N, sample_size_0, ..., sample_size_M]
            - Log Sampling Density Tensor:  each of shape [sample_size_j]

        Message shape constraints:
            - sample_shape must have AT LEAST length 1, the same length as event_shape.
            - batch_shape must have AT LEAST length 1.
            - event_shape must have AT LEAST length 1, the same length as sample_shape.
            - param_shape must have EXACTLY length 1.

        The semantics of a Message is determined not only by its type, but by its context as well. In other words,
            which distribution or distribution class a Message represents, or whether a Parameter type message
            represents a Natural parameter to an exponential distribution or a distribution-class specific parameter to
            PyTorch's distribution class interface, is of no concern to the Message structure itself.

        Both types of messages are endowed with certain arithmetic structures:
            - For Parameter messages,
                - Addition operation is defined as addition on the parameter tensors.
                - Scalar multiplication is defined as scalar multiplication with the parameter tensors
                - 0 is treated as the identity element.
                - Parameter message structure therefore constructs and defines the "parameter space".

            - For Particles messages:
                - The following two operations are defined as operations on the particle weights, and meaningful only
                    for Particle messages that share the same particle values and the same sampling log density of the
                    particles. In addition, results from these two operations are normalized so that the weight tensor
                    sum to 1 across the sample dimension.
                - "Addition" operation is defined as element-wise multiplication of particle weights tensors.
                - "Scalar multiplication" is defined as taking elements of the particle weights tensor to the power
                    of the scalar.
                - 1 is treated as the identity element for the element-wise multiplication operation.
                - It turns out that the "addition" and "scalar multiplication" as defined above satisfy associativity
                    and distributivity. With the set of possible weights closed under both operations, a vector space
                    is therefore constructed and defined.

        Accordingly, the '+' and '*' operator are overloaded according the to the specifications above.
    """

    def __init__(self, msg_type: MessageType,
                 param_shape: torch.Size = None,
                 sample_shape: torch.Size = None, batch_shape: torch.Size = None, event_shape: torch.Size = None,
                 parameters: torch.Tensor = None,
                 particles: torch.Tensor = None, weights: [torch.Tensor, int] = None,
                 log_density: [torch.Tensor, int] = None, **kwargs):
        """
            Instantiate a message. An empty shape (i.e. torch.Size([]) ) is equivalent to a shape of 1.

            :param msg_type:    one of MessageType
            :param param_shape:     torch.Size. Must specify if message type is Parameter. Must be a shape of length 1.
            :param sample_shape:    torch.Size. Must specify if message type is Particles. Must be a shape of length 1.
            :param batch_shape:     torch.Size. Must specify if message type is Particles. Must be a shape of at least
                                        length 1.
            :param event_shape:     torch.Size. Must specify if message type is Particles. Must be a shape of length 1.
            :param parameters:  torch.Tensor. Of shape (batch_shape + param_shape) if the parameters do not represent
                                    the identity in the parameter vector space. Alternatively, can be an int of 0 to
                                    represent the identity. Must present if message type is Parameters.
            :param particles:   torch.Tensor. Of shape (sample_shape + event_shape). Must present if message type is
                                    Particles.
            :param weights:     torch.Tensor. Of shape (sample_shape + batch_shape) if weights are not uniform. In this
                                    case the weights tensor will be normalize over sample_shape dimension so that it
                                    sums to 1 over this dimension. Alternatively, can be an int of 1 to represent
                                    uniform weights. Must present if message type is Particles.
            :param log_density: torch.Tensor. Of shape (sample_shape) if log densities are not uniform, i.e. if the
                                    particles were not drawn from a uniform distribution. Alternatively, it can be an
                                    int of 0 to represent uniform densities. Note that this field generally should not
                                    be changed at any time during message propagation after the particles are drawn,
                                    since they directly represent the original sampling distribution from which the
                                    particles were originally drawn. Must present if message type is Particles.
            :param kwargs:      Additional optional attributes
        """
        assert isinstance(msg_type, MessageType)
        assert param_shape is None or isinstance(param_shape, torch.Size) and len(param_shape) == 1
        assert sample_shape is None or isinstance(sample_shape, torch.Size) and len(sample_shape) == 1
        assert batch_shape is None or isinstance(batch_shape, torch.Size) and len(batch_shape) >= 1
        assert event_shape is None or isinstance(event_shape, torch.Size) and len(event_shape) == 1

        assert parameters is None or (isinstance(parameters, int) and parameters == 0) or \
               isinstance(parameters, torch.Tensor)
        assert particles is None or isinstance(particles, torch.Tensor)
        assert weights is None or (isinstance(weights, int) and weights == 1) or isinstance(weights, torch.Tensor)
        assert log_density is None or (isinstance(log_density, int) and log_density == 0) or \
               isinstance(log_density, torch.Tensor)

        # Message type, of type MessageType
        self.type = msg_type
        # Parameter
        self.parameters = parameters
        # Particle list
        self.particles = particles
        self.weights = weights
        self.log_density = log_density
        # Additional important attributes
        self.attr = kwargs
        # Shapes.
        self.p_shape = param_shape
        self.s_shape = sample_shape
        self.b_shape = batch_shape
        self.e_shape = event_shape

        # Check whether necessary arguments are provided
        if MessageType.Parameter in self.type:
            assert self.b_shape is not None and self.p_shape is not None
            assert self.parameters is not None
        if MessageType.Particles in self.type:
            assert self.s_shape is not None and self.b_shape is not None and self.e_shape is not None
            assert self.particles is not None
            assert self.weights is not None
            assert self.log_density is not None

        # Check shape and values. Adjust if necessary
        if isinstance(self.parameters, torch.Tensor):
            # Parameter tensor should have shape (b_shape + p_shape)
            assert self.b_shape + self.p_shape == self.parameters.shape
        if isinstance(self.particles, torch.Tensor):
            # Particles tensor should have shape (s_shape + e_shape)
            assert self.s_shape + self.e_shape == self.particles.shape
        if isinstance(self.weights, torch.Tensor):
            # Weights tensor should have shape (s_shape + b_shape)
            assert self.s_shape + self.b_shape == self.weights.shape
            # Check that values are non-negative
            assert torch.all(self.weights > 0), "Found negative values in particle weights. Minimum value: {}" \
                .format(torch.min(self.weights))
            # Normalize the values so that weights sum to 1 across sample dimension
            weights_sum = self.weights.sum(dim=0, keepdim=True)
            self.weights /= weights_sum
        if isinstance(self.log_density, torch.Tensor):
            # Log density tensor array should have shape (s_shape)
            assert self.s_shape == self.log_density.shape

    """
        Overload arithmetic operators
    """

    def __add__(self, other):
        """
            Overloading addition operation '+'
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
            new_parameters = self.parameters + other.parameters

            new_msg = Message(self.type, batch_shape=self.b_shape, param_shape=self.p_shape, parameters=new_parameters,
                              **self.attr)

        # Addition for Particles type
        if MessageType.Particles in s_type:
            assert self.s_shape == other.s_shape and self.b_shape == other.b_shape and self.e_shape == other.e_shape, \
                "Only Messages with the same shape can be added together. The messages being added are of Particles " \
                "type. Found first message with (sample_shape, batch_shape, event_shape) = '{}', and second message " \
                "with (sample_shape, batch_shape, event_shape) = '{}'" \
                    .format((self.s_shape, self.b_shape, self.e_shape), (other.s_shape, other.b_shape, other.e_shape))
            assert torch.equal(self.particles, other.particles), \
                "For particle messages, only ones with matching particle values can be added together. "
            assert type(self.log_density) is type(other.log_density) and \
                   ((isinstance(self.log_density, int) and self.log_density == other.log_density) or
                    (isinstance(self.log_density, torch.Tensor) and torch.equal(self.log_density, other.log_density))), \
                "For particle messages, only ones with matching log sampling densities can be added together"

            # Take element-wise product
            new_weights = self.weights * other.weights

            new_msg = Message(self.type,
                              sample_shape=self.s_shape, batch_shape=self.b_shape, event_shape=self.e_shape,
                              particles=self.particles, weights=new_weights, log_density=self.log_density, **self.attr)

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
