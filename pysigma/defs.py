"""
    Basic structures in the graphical architecture
"""
import torch
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from torch.distributions.constraints import Constraint
from enum import Enum

from utils import DistributionServer


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
    Tabular = 0
    Distribution = 1
    Particles = 2
    Parameter = 3

class Message:
    """
        Message structure to support general inference.

        Four basic message type:
            - Tabular factor
            - Parametric distribution
            - Particle list
            - Parameter
        Each of the first three all represents "events" of a batch of distributions, while the last one represents
            parameters to a batch of distributions. For the last one, which class of distribution the parameters are
            describing and whether the parameters are natural parameters to some exponential family, or common
            parameters to PyTorch's distribution class interface, is of no concern to Message itself. Instead, their
            semantic should and will be determined by the context where the message is used.

        Tabular factors will be represented by Categorical distribution, a special case of parametric distribution
            message.
        The message type does not impose restriction on the types of underlying message representations available for
            use, and thus different representations may coexist. For instance, in a Distribution message, a
            torch.distribution may coexists with a particle list, whereas in a Particles message only particle list
            exists.

        Note that there's a distinction between knowledge format compatible with PyTorch's distribution class versus
            that compatible with Sigma's cognitive structures. In general, what is contained in a message are knowledge
            in the PyTorch's format. One should only translate such knowledge using KnowledgeTranslator class (see
            utils.py) after extracting information from a Message instance.

        In general, a message's tensor attributes have shape:
            - distribution parameter:   (b_shape + param_shape)
            - particles:                (s_shape + b_shape + e_shape)
            - weights:                  (s_shape + b_shape)
            - log_density:              (s_shape + b_shape)
    """
    def __init__(self, msg_type: MessageType,
                 sample_shape: torch.Size = None, batch_shape: torch.Size = None, event_shape: torch.Size = None,
                 param_shape: torch.Size = None,
                 dist: Distribution = None, particles: torch.Tensor = None, weights: [torch.Tensor, int] = None,
                 log_density: [torch.Tensor, int] = None,
                 parameters: torch.Tensor = None):
        """
            Instantiate a message. An empty shape (i.e. torch.Size([]) ) is equivalent to a shape of 1.

            :param msg_type:    one of MessageType
            :param sample_shape:    a torch.Size. can be an empty shape
            :param batch_shape:     a torch.Size. can be an empty shape
            :param event_shape:     a torch.Size. can be an empty shape
            :param dist:        A PyTorch distribution instance. If the message is not Particles type
            :param particles:   Particle list. Must present if message is Particles type
            :param weights:     Particle weights. Specify an int of 1 if the weights are uniform. Otherwise, must be a
                                nonnegative torch Tensor of shape (sample_shape + batch_shape). Values along
                                'sample_shape' dimension must add up to 1.
            :param log_density: A PyTorch tensor containing the log probability density (log-pdf) of the particles when
                                they were sampled from the original sampling distribution. Specify an int of 0 if the
                                sampling distribution was uniform. Otherwise, must be a nonnegative torch Tensor of
                                shape (sample_shape + batch_shape). Note that this field generally should not be
                                changed at any time during propagation after the particles are drawn, since they
                                indicates the relative frequency of each particle.
        """
        assert isinstance(msg_type, MessageType)
        assert sample_shape is None or isinstance(sample_shape, torch.Size)
        assert batch_shape is None or isinstance(batch_shape, torch.Size)
        assert event_shape is None or isinstance(event_shape, torch.Size)
        assert param_shape is None or isinstance(param_shape, torch.Size)
        assert dist is None or isinstance(dist, Distribution)
        assert particles is None or isinstance(particles, torch.Tensor)
        assert weights is None or (isinstance(weights, int) and weights == 1) or isinstance(weights, torch.Tensor)
        assert log_density is None or (isinstance(log_density, int) and log_density == 0) or \
               isinstance(log_density, torch.Tensor)
        assert parameters is None or isinstance(parameters, torch.Tensor)

        # Message type, of type MessageType
        self.type = msg_type
        # Distribution
        self.dist = dist
        # Particle list
        self.particles = particles
        self.weights = weights
        self.log_density = log_density
        # Parameter
        self.parameters = parameters
        # Shapes. Collapse the shape if it is a singleton (because PyTorch's distribution will collapse it anyhow)
        self.s_shape = None
        self.b_shape = None
        self.e_shape = None
        self.p_shape = None
        if sample_shape is not None:
            self.s_shape = sample_shape if sample_shape != torch.Size([1]) else torch.Size([])
        if batch_shape is not None:
            self.b_shape = batch_shape if batch_shape != torch.Size([1]) else torch.Size([])
        if event_shape is not None:
            self.e_shape = event_shape if event_shape != torch.Size([1]) else torch.Size([])
        if param_shape is not None:
            self.p_shape = param_shape if param_shape != torch.Size([1]) else torch.Size([])

        # Check whether (only) necessary arguments are provided
        if self.type is MessageType.Tabular:
            assert self.b_shape is not None and self.e_shape is not None
            # Need to provide Categorical distribution and nothing else
            assert isinstance(self.dist, Categorical)
        if self.type is MessageType.Distribution:
            assert self.b_shape is not None and self.e_shape is not None
            # Need to provide some distribution. Particle list is optional
            assert self.dist is not None
        if self.type is MessageType.Particles:
            assert self.s_shape is not None and self.b_shape is not None and self.e_shape is not None
            # Need to provide a particle list, but no distribution
            assert self.particles is not None
            assert self.weights is not None
            assert self.log_density is not None
        if self.type is MessageType.Parameter:
            assert self.b_shape is not None and self.p_shape is not None
            assert self.parameters is not None

        # Check shape
        if self.dist is not None:
            assert self.b_shape == self.dist.batch_shape
            assert self.e_shape == self.dist.event_shape
        if self.particles is not None:
            assert self.s_shape + self.b_shape + self.e_shape == self.particles.shape
        if isinstance(self.weights, torch.Tensor):
            assert self.s_shape + self.b_shape == self.weights.shape
        if isinstance(self.log_density, torch.Tensor):
            assert self.s_shape + self.b_shape == self.log_density.shape
        if self.parameters is not None:
            assert self.b_shape + self.p_shape == self.parameters.shape

    def clone(self):
        """
            Return a cloned message from self. Guarantees that every tensor that constitutes self is cloned
        """
        dist = None
        particles = None
        weights = None
        log_density = None
        parameters = None

        if self.dist is not None:
            params = DistributionServer.dist2param(self.dist)
            params = params.clone()
            dist = DistributionServer.param2dist(type(self.dist), params, self.b_shape, self.e_shape)
        if self.particles is not None:
            particles = self.particles.clone()
        if self.weights is not None:
            weights = self.weights.clone()
        if self.log_density is not None:
            log_density = self.log_density.clone()
        if self.parameters is not None:
            parameters = self.parameters.clone()

        new_msg = Message(self.type,
                          sample_shape=self.s_shape, batch_shape=self.b_shape, event_shape=self.e_shape,
                          param_shape=self.p_shape,
                          dist=dist, particles=particles, weights=weights, log_density=log_density,
                          parameters=parameters)
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
            The behavior of the permutation is similar to torch.Tensor.permute()

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
        # Permute order for sample, batch, and event dimensions altogether
        s_b_e_dims = s_b_dims + list(i + len(s_b_dims) for i in range(len(self.e_shape)))
        # Permute order for batch and parameter dimensions together
        b_p_dims = pos_target_dims + list(i + len(pos_target_dims) for i in range(len(self.p_shape)))

        new_dist = self.dist
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density
        new_parameters = self.parameters

        if self.dist is not None:
            # dist param has shape (b_shape + e_shape)
            dist_param = DistributionServer.dist2param(self.dist)
            new_dist_param = dist_param.permute(list(pos_target_dims[i] if i < len(pos_target_dims) else i
                                                     for i in range(len(dist_param.shape))))
            new_dist = DistributionServer.param2dist(type(self.dist), new_dist_param, new_b_shape, self.e_shape)
        if self.particles is not None:
            # particles has shape (s_shape + b_shape + e_shape)
            new_particles = self.particles.permute(s_b_e_dims)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = self.weights.permute(s_b_dims)
        if isinstance(self.log_density, torch.Tensor):
            # log_density has shape (s_shape + b_shape)
            new_log_density = self.log_density.permute(s_b_dims)
        if self.parameters is not None:
            # parameters has shape (b_shape + p_shape)
            new_parameters = self.parameters.permute(b_p_dims)

        new_msg = Message(self.type,
                          sample_shape=self.s_shape, batch_shape=new_b_shape, event_shape=self.e_shape,
                          param_shape=self.p_shape,
                          dist=new_dist, particles=new_particles, weights=new_weights, log_density=new_log_density,
                          parameters=new_parameters)

        return new_msg

    def batch_unsqueeze(self, dim):
        """
            Returns a new message with a dimension of size one inserted at the specified batch dimension, similar to
                torch.unsqueeze().
            A 'dim' value within the range [-len(batch_shape) - 1, len(batch_shape) + 1] can be used. Note that 'dim'
                is relative to the batch dimension only.

            :param dim:     an int. The place where the new dimension of size one will be inserted at.
        """
        assert isinstance(dim, int) and -len(self.b_shape) - 1 <= dim <= len(self.b_shape) + 1

        # Translate dim to positive value if it is negative
        if dim < 0:
            dim = len(self.b_shape) + dim
        # For message contents who has a sample dimension at front, add 1 to dim
        s_dim = dim + 1
        # Get new batch shape
        new_b_shape = self.b_shape[:dim] + torch.Size([1]) + self.b_shape[dim:]

        new_dist = self.dist
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density
        new_parameters = self.parameters

        if self.dist is not None:
            # dist param has shape (b_shape + e_shape)
            dist_param = DistributionServer.dist2param(self.dist)
            new_dist_param = torch.unsqueeze(dist_param, dim)
            new_dist = DistributionServer.param2dist(type(self.dist), new_dist_param, new_b_shape, self.e_shape)
        if self.particles is not None:
            # particles has shape (s_shape + b_shape + e_shape)
            new_particles = torch.unsqueeze(self.particles, s_dim)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = torch.unsqueeze(self.weights, s_dim)
        if isinstance(self.log_density, torch.Tensor):
            # log_density has shape (s_shape + b_shape)
            new_log_density = torch.unsqueeze(self.log_density, s_dim)
        if self.parameters is not None:
            # parameters has shape (b_shape + p_shape)
            new_parameters = torch.unsqueeze(self.parameters, dim)

        new_msg = Message(self.type,
                          sample_shape=self.s_shape, batch_shape=new_b_shape, event_shape=self.e_shape,
                          param_shape=self.p_shape,
                          dist=new_dist, particles=new_particles, weights=new_weights, log_density=new_log_density,
                          parameters=new_parameters)

        return new_msg

    def batch_index_select(self, dim, index):
        """
            Returns a new Message which indexes the input message along batch dimension dim using the entries in index
                which is a LongTensor.
            A 'dim' value within the range [-len(batch_shape), len(batch_shape) - 1] can be used. Note that 'dim'
                is relative to the batch dimension only.

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

        new_dist = self.dist
        new_particles = self.particles
        new_weights = self.weights
        new_log_density = self.log_density
        new_parameters = self.parameters

        if self.dist is not None:
            # dist param has shape (b_shape + e_shape)
            dist_param = DistributionServer.dist2param(self.dist)
            new_dist_param = torch.index_select(dist_param, dim, index)
            new_dist = DistributionServer.param2dist(type(self.dist), new_dist_param, new_b_shape, self.e_shape)
        if self.particles is not None:
            # particles has shape (s_shape + b_shape + e_shape)
            new_particles = torch.index_select(self.particles, s_dim, index)
        if isinstance(self.weights, torch.Tensor):
            # weights has shape (s_shape + b_shape)
            new_weights = torch.index_select(self.weights, s_dim, index)
        if isinstance(self.log_density, torch.Tensor):
            # log_density has shape (s_shape + b_shape)
            new_log_density = torch.index_select(self.log_density, s_dim, index)
        if self.parameters is not None:
            # parameters has shape (b_shape + p_shape)
            new_parameters = torch.index_select(self.parameters, dim, index)

        new_msg = Message(self.type,
                          sample_shape=self.s_shape, batch_shape=new_b_shape, event_shape=self.e_shape,
                          param_shape=self.p_shape,
                          dist=new_dist, particles=new_particles, weights=new_weights, log_density=new_log_density,
                          parameters=new_parameters)
        return new_msg


# TODO: Enum class of all the inference method
class InferenceMethod(Enum):
    BP = 0
    PARTICLE_BP = 1
    VMP = 2
    PARTICLE_VMP = 3
    EP = 4
    PARTICLE_EP = 5
