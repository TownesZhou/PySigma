"""
    Basic structures in the graphical architecture
"""
import torch
from torch.distributions.constraints import Constraint
from enum import Enum


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
            :param value_constraints:   a set of torch.distributions.constraints.Constraint instances. The set of value
                                            constraints that determine the value range (domain) of this variable
        """
        assert isinstance(name, str)
        assert isinstance(metatype, VariableMetatype)
        assert isinstance(size, int)
        assert value_constraints is None or (isinstance(value_constraints, set) and
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
class MessageType(Enum):
    """
        Enum class to represent message types
    """
    Parameter = 0
    Particles = 1


class Message:
    """
        Message structure to support general inference.

        Two basic message types:
            - Parameter: parameters to some batched distributions
                - Contains a batched parameter tensor, of shape (batch_shape + param_shape)

            - Particles: batched particle lists representing events drawn from some batched distributions
                - Contains three components: particle value, particle weight, and sampling log density
                - particle value is a batched tensor with shape (sample_shape + batch_shape + event_shape)
                - particle weight is a batched tensor with shape (sample_shape + batch_shape) if the weights are not
                    uniform. In this case, the entries should be non-negative values, and sum to 1 across 'sample_shape'
                    dimension. Otherwise if the weights are uniform, it can be represented by an int of 1.
                - sample log density is a batched tensor with shape (sample_shape + batch_shape) if the log densities
                    are not uniform. Otherwise if the log densities are uniform, it can be represented by an int of 0.

        Message shape constraints:
            - sample_shape must have exactly length 1.
            - batch_shape can have arbitrary positive length, but the size of each dimension must be at least 2.
            - event_shape must have exactly length 1.
            - param_shape must have exactly length 1.

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
            :param param_shape:     torch.Size. Must specify if message type is Parameter. Can be a shape of arbitrary
                                        length, but the size of each dimension must be at least 2.
            :param sample_shape:    torch.Size. Must specify if message type is Particles. Must be a shape of length 1.
            :param batch_shape:     torch.Size. Must specify if message type is Particles. Must be a shape of length 1.
            :param event_shape:     torch.Size. Must specify if message type is Particles. Must be a shape of length 1.
            :param parameters:  torch.Tensor. Of shape (batch_shape + param_shape) if the parameters do not represent
                                    the identity in the parameter vector space. Alternatively, can be an int of 0 to
                                    represent the identity. Must present if message type is Parameters.
            :param particles:   torch.Tensor. Of shape (sample_shape + batch_shape + event_shape). Must present if
                                    message type is Particles.
            :param weights:     torch.Tensor. Of shape (sample_shape + batch_shape) if weights are not uniform. In this
                                    case the weights tensor will be normalize over sample_shape dimension so that it
                                    sums to 1 over this dimension. Alternatively, can be an int of 1 to represent
                                    uniform weights. Must present if message type is Particles.
            :param log_density: torch.Tensor. Of shape (sample_shape + batch_shape) if log densities are not uniform,
                                    i.e. if the particles were not drawn from an uniform distribution. Alternatively,
                                    can be an int of 0 to represent uniform densities. Note that this field generally
                                    should not be changed at any time during message propagation after the particles are
                                    drawn, since they directly represent the original sampling distribution from which
                                    the particles were originally drawn. Must present if message type is Particles.
            :param kwargs:      Additional optional attributes
        """
        assert isinstance(msg_type, MessageType)
        assert param_shape is None or isinstance(param_shape, torch.Size) and len(param_shape) == 1
        assert sample_shape is None or isinstance(sample_shape, torch.Size) and len(sample_shape) == 1
        assert batch_shape is None or isinstance(batch_shape, torch.Size) and all(d >= 2 for d in batch_shape)
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

        # Shapes. Collapse the shape if it is a singleton (because PyTorch's distribution will collapse it anyhow)
        self.p_shape = None
        self.s_shape = None
        self.b_shape = None
        self.e_shape = None
        if param_shape is not None:
            self.p_shape = param_shape if param_shape != torch.Size([1]) else torch.Size([])
        if sample_shape is not None:
            self.s_shape = sample_shape if sample_shape != torch.Size([1]) else torch.Size([])
        if batch_shape is not None:
            self.b_shape = batch_shape if batch_shape != torch.Size([1]) else torch.Size([])
        if event_shape is not None:
            self.e_shape = event_shape if event_shape != torch.Size([1]) else torch.Size([])

        # Check whether necessary arguments are provided
        if self.type is MessageType.Parameter:
            assert self.b_shape is not None and self.p_shape is not None
            assert self.parameters is not None
        if self.type is MessageType.Particles:
            assert self.s_shape is not None and self.b_shape is not None and self.e_shape is not None
            assert self.particles is not None
            assert self.weights is not None
            assert self.log_density is not None

        # Check shape and values. Adjust if necessary
        if isinstance(self.parameters, torch.Tensor):
            assert self.b_shape + self.p_shape == self.parameters.shape
        if isinstance(self.particles, torch.Tensor):
            assert self.s_shape + self.b_shape + self.e_shape == self.particles.shape
        if isinstance(self.weights, torch.Tensor):
            assert self.s_shape + self.b_shape == self.weights.shape
            # Check that values are non-negative
            assert torch.all(self.weights > 0), "Found negative values in particle weights"
            # Normalize the values so that weights sum to 1 across sample dimension
            weights_sum = self.weights.sum(dim=0, keepdim=True)
            self.weights *= (1 / weights_sum)
        if isinstance(self.log_density, torch.Tensor):
            assert self.s_shape + self.b_shape == self.log_density.shape

        if isinstance(self.weights, int) and self.weights == 1:
            self.weights = torch.tensor(self.weights)

    """
        Overload arithmetic operators
    """

    def __add__(self, other):
        """
            Overloading addition operation '+'
        """
        assert isinstance(other, Message), "Message can only be added with another Message"
        assert self.type == other.type, "Only the same type of messages can be added. First operand has type '{}', " \
                                        "while the second one has type '{}'".format(self.type, other.type)
        # Addition for Parameter type
        if self.type == MessageType.Parameter:
            assert self.b_shape == other.b_shape and self.p_shape == other.p_shape, \
                "Only Messages with the same shape can be added together. The messages being added are of Parameter " \
                "type. Found first message with (batch_shape, param_shape) = '{}', and second message with " \
                "(batch_shape, param_shape) = '{}'".format((self.b_shape, self.p_shape), (other.b_shape, other.p_shape))
            # Tensor addition
            new_parameters = self.parameters + other.parameters

            # edge case when parameter shape is made to Size([]) before, and here to make it back to Size([1])
            if self.p_shape == torch.Size([]):
                self.p_shape = torch.Size([1])
            new_msg = Message(self.type, batch_shape=self.b_shape, param_shape=self.p_shape, parameters=new_parameters)

        # Addition for Particles type
        else:
            assert self.s_shape == other.s_shape and self.b_shape == other.b_shape and self.e_shape == other.e_shape, \
                "Only Messages with the same shape can be added together. The messages being added are of Particles " \
                "type. Found first message with (sample_shape, batch_shape, event_shape) = '{}', and second message " \
                "with (sample_shape, batch_shape, event_shape) = '{}'" \
                    .format((self.s_shape, self.b_shape, self.e_shape), (other.s_shape, other.b_shape, other.e_shape))
            assert torch.all(self.particles.eq(other.particles)), \
                "For particle messages, only ones with matching particle values can be added together. "
            assert isinstance(self.log_density, int) and isinstance(other.log_density, int) and \
                   self.log_density == other.log_density == 0 or \
                   torch.all(self.log_density.eq(other.log_density)), \
                "For particle messages, only ones with matching log sampling densities can be added together"

            # Take element-wise product
            # only if both weights are tensor(1), then the new_weights will be tensor(1)
            new_weights = self.weights * other.weights

            # case when the new weight is uniform
            if isinstance(new_weights, torch.Tensor) and new_weights.size() == torch.Size([]) and \
                    bool(new_weights == torch.tensor(1)):
                new_msg = Message(self.type, sample_shape=self.s_shape, batch_shape=self.b_shape,
                                  event_shape=self.e_shape,
                                  particles=self.particles, weights=1, log_density=self.log_density)
            else:
                new_msg = Message(self.type, sample_shape=self.s_shape, batch_shape=self.b_shape,
                                  event_shape=self.e_shape, particles=self.particles,
                                  weights=new_weights, log_density=self.log_density)

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

        # Expand scalar tensor dimension if it is a batched scalars
        b_p_other = other
        s_b_other = other
        if isinstance(other, torch.Tensor) and other.dim() > 0:
            if self.type is MessageType.Parameter:
                for i in range(len(self.p_shape)):
                    b_p_other = torch.unsqueeze(b_p_other, dim=-1)

            if self.type is MessageType.Particles:
                s_b_other = torch.unsqueeze(s_b_other, dim=0)

        # Scalar multiplication for Parameter messages
        if self.type == MessageType.Parameter:
            new_parameters = b_p_other * self.parameters
            new_msg = Message(self.type, batch_shape=self.b_shape, param_shape=self.p_shape, parameters=new_parameters)

        # Scalar multiplication for Particles messages
        else:
            # if weight uniform
            if self.weights.shape == torch.Size([]) and self.weights == torch.tensor(1):
                # if scalar is Tensor and non-singleton
                if isinstance(other, torch.Tensor) and other.dim() > 0:
                    # Creates uniform weights that matches
                    # edge case when sample is empty
                    if self.s_shape == torch.Size([]):
                        new_weights = torch.pow(torch.ones(*(self.b_shape)), other)
                    else:
                        new_weights = torch.pow(torch.ones(*(self.s_shape + self.b_shape)), s_b_other)
                    new_weights *= torch.tensor(1.) / new_weights.sum(dim=0, keepdim=True)

                    # edge case when sample shape is made to Size([]) before, and here to make it back to Size([1])
                    if self.s_shape == torch.Size([]):
                        self.s_shape = torch.Size([1])

                    # edge case when event shape is made to Size([]) before, and here to make it back to Size([1])
                    if self.e_shape == torch.Size([]):
                        self.e_shape = torch.Size([1])
                    new_msg = Message(self.type, sample_shape=self.s_shape, batch_shape=self.b_shape,
                                      event_shape=self.e_shape,
                                      particles=self.particles, weights=new_weights, log_density=self.log_density)

                # if scalar is int, float, or singleton Tensor
                else:
                    new_msg = Message(self.type, sample_shape=self.s_shape, batch_shape=self.b_shape,
                                      event_shape=self.e_shape,
                                      particles=self.particles, weights=1, log_density=self.log_density)
            # if weight not uniform
            else:
                # Take weights tensor to the power of the scalar
                new_weights = torch.pow(self.weights, s_b_other)
                new_msg = Message(self.type,
                                  sample_shape=self.s_shape, batch_shape=self.b_shape, event_shape=self.e_shape,
                                  particles=self.particles, weights=new_weights, log_density=self.log_density)

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
            Return the shape of the mssage.
            For Parameter message, returns  (batch_shape + param_shape)
            For Particles message, returns  (sample_shape + batch_shape + event_shape)
        """
        if self.type == MessageType.Parameter:
            return self.b_shape + self.p_shape
        else:
            return self.s_shape + self.b_shape + self.e_shape

    def same_size_as(self, other):
        """
            Check if self has the same shape as the other message. Return True if so.
        """
        assert isinstance(other, Message)
        if self.type != other.type:
            return False
        elif self.type == MessageType.Parameter:
            return self.b_shape == other.b_shape and self.p_shape == other.p_shape
        else:
            return self.s_shape == other.s_shape and self.b_shape == other.b_shape and self.e_shape == other.e_shape

    def clone(self):
        """
            Return a cloned message from self. Guarantees that every tensor that constitutes self is cloned
        """
        parameters = None
        particles = None
        weights = None
        log_density = None

        if isinstance(self.parameters, torch.Tensor):
            parameters = self.parameters.clone()
        if isinstance(self.particles, torch.Tensor):
            particles = self.particles.clone()
        if isinstance(self.weights, torch.Tensor):
            weights = self.weights.clone()
        if isinstance(self.log_density, torch.Tensor):
            log_density = self.log_density.clone()

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, self.b_shape, self.e_shape,
                          parameters, particles, weights, log_density)
        return new_msg

    """
        Tensor manipulation utility methods
    """

    def batch_permute(self, target_dims):
        """
            Returns a permuted message whose tensor attributes are permuted from the original ones w.r.t. 'target_dims'
                in the batch dimensions.
            Note that target_dims is relative to the batch dimension. Its values should be within the range
                    [-len(batch_shape), len(batch_shape) - 1]

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

        if self.type == MessageType.Particles:
            # Permute order for sample, batch, and event dimensions altogether
            s_b_e_dims = s_b_dims + list(i + len(s_b_dims) for i in range(len(self.e_shape)))

        if self.type == MessageType.Parameter:
            # Permute order for batch and parameter dimensions together
            b_p_dims = pos_target_dims + list(i + len(pos_target_dims) for i in range(len(self.p_shape)))

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameters = self.parameters.permute(b_p_dims)
        if isinstance(self.particles, torch.Tensor):
            # particles has shape (s_shape + b_shape + e_shape)
            new_particles = self.particles.permute(s_b_e_dims)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = self.weights.permute(s_b_dims)
        if isinstance(self.log_density, torch.Tensor):
            # log_density has shape (s_shape + b_shape)
            new_log_density = self.log_density.permute(s_b_dims)

        # edge case when parameter shape is made to Size([]) before, and here to make it back to Size([1])
        if self.p_shape == torch.Size([]):
            self.p_shape = torch.Size([1])

        # edge case when sample shape is made to Size([]) before, and here to make it back to Size([1])
        if self.s_shape == torch.Size([]):
            self.s_shape = torch.Size([1])

        # edge case when event shape is made to Size([]) before, and here to make it back to Size([1])
        if self.e_shape == torch.Size([]):
            self.e_shape = torch.Size([1])
        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density)
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
        if self.s_shape != torch.Size([]):
            s_dim = dim + 1
        else:
            s_dim = dim
        # Get new batch shape
        new_b_shape = self.b_shape[:dim] + torch.Size([1]) + self.b_shape[dim:]

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameters = torch.unsqueeze(self.parameters, dim)
        if isinstance(self.particles, torch.Tensor):
            # particles has shape (s_shape + b_shape + e_shape)
            new_particles = torch.unsqueeze(self.particles, s_dim)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = torch.unsqueeze(self.weights, s_dim)
        if isinstance(self.log_density, torch.Tensor):
            # log_density has shape (s_shape + b_shape)
            new_log_density = torch.unsqueeze(self.log_density, s_dim)

        # edge case when parameter shape is made to Size([]) before, and here to make it back to Size([1])
        if self.p_shape == torch.Size([]):
            self.p_shape = torch.Size([1])

        # edge case when sample shape is made to Size([]) before, and here to make it back to Size([1])
        if self.s_shape == torch.Size([]):
            self.s_shape = torch.Size([1])

        # edge case when event shape is made to Size([]) before, and here to make it back to Size([1])
        if self.e_shape == torch.Size([]):
            self.e_shape = torch.Size([1])

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density)
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
        assert isinstance(index, torch.LongTensor) and len(index.shape) == 1

        # Translate dim to positive value if it is negative
        if dim < 0:
            dim = len(self.b_shape) + dim
        # For message contents who has a sample dimension at front, add 1 to dim
        s_dim = dim + 1
        # Get new batch shape
        new_b_shape = self.b_shape[:dim] + index.shape + self.b_shape[dim:]

        new_parameters = self.parameters
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density

        if isinstance(self.parameters, torch.Tensor):
            # parameters has shape (b_shape + p_shape)
            new_parameters = torch.index_select(self.parameters, dim, index)
        if isinstance(self.particles, torch.Tensor):
            # particles has shape (s_shape + b_shape + e_shape)
            new_particles = torch.index_select(self.particles, s_dim, index)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = torch.index_select(self.weights, s_dim, index)
        if isinstance(self.log_density, torch.Tensor):
            # log_density has shape (s_shape + b_shape)
            new_log_density = torch.index_select(self.log_density, s_dim, index)

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density)
        return new_msg

    def batch_diagonal(self, dim1, dim2):
        """
            Returns a partial view of self with the its diagonal elements with respect to 'dim1' and 'dim2' appended as
                a dimension at the end of the shape.
            dim values within the range [-len(batch_shape), len(batch_shape) - 1] can be used.
            Note that 'dim1' and 'dim2' are relative to the batch dimension. The appended dimension will be placed as
                the last batch dimension, but before any event or param dimension.

            This method is a mimic of torch.diagonal()
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
        if isinstance(self.particles, torch.Tensor):
            # particles has shape (s_shape + b_shape + e_shape)
            new_particles = torch.diagonal(new_particles, dim1=s_dim1, dim2=s_dim2)
            # Swap event dimension and appended diagonal batch dimension
            new_particles = torch.transpose(new_particles, dim0=-1, dim1=-2)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = torch.diagonal(new_weights, dim1=s_dim1, dim2=s_dim2)
        if isinstance(self.log_density, torch.Tensor):
            # log_density has shape (s_shape + b_shape)
            new_log_density = torch.diagonal(new_log_density, dim1=s_dim1, dim2=s_dim2)

        new_msg = Message(self.type,
                          self.p_shape, self.s_shape, new_b_shape, self.e_shape,
                          new_parameters, new_particles, new_weights, new_log_density)
        return new_msg


# TODO: Enum class of all the inference method
class InferenceMethod(Enum):
    BP = 0
    PARTICLE_BP = 1
    VMP = 2
    PARTICLE_VMP = 3
    EP = 4
    PARTICLE_EP = 5
