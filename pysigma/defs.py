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
        The "Type" of a variable, as defined in a predicate, is not considered here. The only relevant information about
            its "Type" is captured in the 'size' field. This is to support matching across different types of variables
             in a principled way.
    """
    def __init__(self, name, metatype, size, value_constraints=None):
        """
            Instantiate a variable. Optionally indicates a set of value constraints if and only if variable is Random
                metatype.
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
              self.size == other.size

        return val

    def __ne__(self, other):
        # override so '!=' operator test variable inequality
        return not self.__eq__(other)

    def __str__(self):
        # override to provide the name as the string representation
        return self.name

    def __hash__(self):
        # override so that hash value of the string representation of the variable is used
        return hash(self.name + str(self.metatype) + str(self.size))


# Generalized message type and message representation
class MessageType(Enum):
    """
        Enum class to represent message types
    """
    Tabular = 0
    Distribution = 1
    Particles = 2


class Message:
    """
        Message structure to support general inference.
        Three basic message type:
            - Tabular factor
            - Parametric distribution
            - Particle list
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
    """
    def __init__(self, msg_type: MessageType, sample_shape: torch.Size, batch_shape: torch.Size, event_shape: torch.Size,
                 dist: Distribution = None, particles: torch.Tensor = None, weights: [torch.Tensor, int] = None,
                 log_density: [torch.Tensor, int] = None):
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
        assert isinstance(sample_shape, torch.Size)
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(event_shape, torch.Size)
        assert dist is None or isinstance(dist, Distribution)
        assert particles is None or isinstance(particles, torch.Tensor)
        assert weights is None or (isinstance(weights, int) and weights == 1) or isinstance(weights, torch.Tensor)
        assert log_density is None or (isinstance(log_density, int) and log_density == 0) or \
               isinstance(log_density, torch.Tensor)

        # Message type, of type MessageType
        self.type = msg_type
        # Distribution
        self.dist = dist
        # Particle list
        self.particles = particles
        self.weights = weights
        self.log_density = log_density
        # Shapes. Collapse the shape if it is a singleton (because PyTorch's distribution will collapse it anyhow)
        self.s_shape = sample_shape if sample_shape != torch.Size([1]) else torch.Size([])
        self.b_shape = batch_shape if batch_shape != torch.Size([1]) else torch.Size([])
        self.e_shape = event_shape if event_shape != torch.Size([1]) else torch.Size([])

        # Check whether (only) necessary arguments are provided
        if self.type is MessageType.Tabular:
            # Need to provide Categorical distribution and nothing else
            assert isinstance(self.dist, Categorical)
            assert self.particles is None
            assert self.weights is None
        if self.type is MessageType.Distribution:
            # Need to provide some distribution. Particle list is optional
            assert self.dist is not None
        if self.type is MessageType.Particles:
            # Need to provide a particle list, but no distribution
            assert self.particles is not None
            assert self.weights is not None
            assert self.log_density is not None
            assert self.dist is None

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

    def clone(self):
        """
            Return a cloned message from self. Guarantees that every tensor that constitutes self is cloned
        """
        msg_type = self.type
        dist = None
        particles = None
        weights = None
        log_density = None

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

        new_msg = Message(msg_type, self.s_shape, self.b_shape, self.e_shape, dist, particles, weights, log_density)
        return new_msg


# TODO: Enum class of all the inference method
class InferenceMethod(Enum):
    BP = 0
    PARTICLE_BP = 1
    VMP = 2
    PARTICLE_VMP = 3
    EP = 4
    PARTICLE_EP = 5
