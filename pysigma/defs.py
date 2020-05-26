"""
    Basic structures in the graphical architecture
"""
import torch
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from enum import Enum

from utils import Dist2Params, Params2Dist


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
    def __init__(self, name: str, metatype: VariableMetatype, size: int):
        """
            Instantiate a variable
        """
        assert isinstance(name, str)
        assert isinstance(metatype, VariableMetatype)
        assert isinstance(size, int)

        # Variable name, its identity. Used for variable matching. Of type str
        self.name = name
        # Variable meta-type, of type VariableMetatype
        self.metatype = metatype
        # Variable size. Size of the dimension that the variable corresponds to. Of type int
        self.size = size

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
    """
    def __init__(self, msg_type: MessageType, sample_shape: torch.Size, batch_shape: torch.Size, event_shape: torch.Size,
                 dist: Distribution = None, particles: torch.Tensor = None, weights: torch.Tensor = None,
                 log_density: torch.Tensor = None):
        """
            Instantiate a message

            :param msg_type: one of MessageType
            :param sample_shape:
            :param batch_shape:
            :param event_shape:
            :param dist: A PyTorch distribution instance. If the message is not Particles type
            :param particles: Particle list. Must present if message is Particles type
            :param weights: A one-dimension PyTorch tensor as particle weights. Size the same as the first dimension of
                            the particle list.
            :param log_density: A one-dimension PyTorch tensor containing the log probability density (log-pdf) of the
                                particles when they were sampled from the original sampling distribution. Size the same
                                as the first dimension of the particle list.
        """
        assert isinstance(msg_type, MessageType)
        # assert isinstance(sample_shape, torch.Size) and len(sample_shape) == 1
        # assert isinstance(batch_shape, torch.Size) and len(batch_shape) >= 1
        # assert isinstance(event_shape, torch.Size) and len(event_shape) >= 1
        assert isinstance(sample_shape, torch.Size)
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(event_shape, torch.Size)
        assert dist is None or isinstance(dist, Distribution)
        assert particles is None or isinstance(particles, torch.Tensor)
        assert weights is None or isinstance(weights, torch.Tensor)

        # Message type, of type MessageType
        self.type = msg_type
        # Distribution
        self.dist = dist
        # Particle list
        self.particles = particles
        self.weights = weights
        self.log_density = log_density
        # Shapes
        self.s_shape = sample_shape
        self.b_shape = batch_shape
        self.e_shape = event_shape

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
        if self.weights is not None:
            assert self.s_shape == self.weights.shape
        if self.log_density is not None:
            assert self.s_shape == self.log_density.shape

    def clone(self):
        """
            Return a cloned message from self. Guarantees that every tensor that constitutes self is cloned
        """
        msg_type = self.type
        dist = None
        particles = None
        weights = None

        if self.dist is not None:
            params = Dist2Params.convert(self.dist)
            params = params.clone()
            dist = Params2Dist.convert(params, type(self.dist))
        if self.particles is not None:
            particles = self.particles.clone()
        if self.weights is not None:
            weights = self.weights.clone()

        new_msg = Message(msg_type, self.s_shape, self.b_shape, self.e_shape, dist, particles, weights)
        return new_msg


# TODO: Enum class of all the inference method
class InferenceMethod(Enum):
    pass
