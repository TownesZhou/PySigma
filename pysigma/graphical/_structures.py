"""
    Basic structures in the graphical architecture
"""
import torch
from torch.distributions.distribution import Distribution
from torch.distributions.categorical import Categorical
from torch.distributions.constraints import Constraint
from enum import Enum


# Variable Metatypes and Variable for general inference
class VariableMetatype(Enum):
    Indexing = 0  # Particle indexing variable, first dimension of a message
    Relational = 1  # Relational variable, second set of dimensions of a message
    Random = 2  # Random variable, last set of dimensions of a message


class Variable:
    """
        Variable as in variable nodes in the graphical architecture. Store information about this variable such as
            its meta-type, dimension size, and value range.
        The equality testing is used for matching variables in Alpha-Beta graphs. Two variables are equal if and only
            if ALL of the fields are equal. For example, if two Random variables are equal, then they must have the
            same constraints.
        The "type" of a variable, as defined in a predicate, is not considered here. This is to support matching across
            different types in a principle way.
    """
    # Variable name, its identity. Used for variable matching
    name = None
    # Variable meta-type, of type VariableMetatype
    metatype = None
    # Variable size. Size of the dimension that the variable corresponds to
    size = None
    # Constraints of the variable's value range. Of type Constraint from PyTorch. Only set when variable type is Random.
    constraint = None

    def __init__(self, name: str, metatype: VariableMetatype, size: int, constraint: Constraint):
        """
            Instantiate a variable
        """
        assert isinstance(name, str)
        assert isinstance(metatype, VariableMetatype)
        assert isinstance(size, int)
        assert isinstance(constraint, Constraint)

        self.name = name
        self.metatype = metatype
        self.size = size
        self.constraint = constraint

    def __eq__(self, other):
        # override so '==' operator test variable equality
        assert isinstance(other, Variable)
        val = self.name == other.name and \
              self.metatype == other.metatype and \
              self.size == other.size and \
              self.constraint == other.constraint

        return val

    def __ne__(self, other):
        # override so '!=' operator test variable equality
        assert isinstance(other, Variable)
        val = self.name != other.name or \
              self.metatype != other.metatype or \
              self.size != other.size or \
              self.constraint != other.constraint

        return val

    def __str__(self):
        # override to provide the name as the string representation
        return self.name

    def __hash__(self):
        # override so that hash value of the string representation of the variable is used
        return hash(self.name + str(self.metatype) + str(self.size) + str(self.constraint))


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
    # Message type, of type MessageType
    type = None
    # Distribution
    dist = None
    # Particle list
    particles = None
    weights = None
    # Shapes
    s_shape = None
    b_shape = None
    e_shape = None

    def __init__(self, msg_type: MessageType, sample_shape: torch.Size, batch_shape: torch.Size,
                 event_shape: torch.Size,
                 dist: Distribution = None, particles: torch.Tensor = None, weights: torch.Tensor = None):
        """
            Instantiate a message
        """
        assert isinstance(msg_type, MessageType)
        assert isinstance(sample_shape, torch.Size)
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(event_shape, torch.Size)
        assert dist is None or isinstance(dist, Distribution)
        assert particles is None or isinstance(particles, torch.Tensor)
        assert weights is None or isinstance(weights, torch.Tensor)

        self.type = msg_type
        self.dist = dist
        self.particles = particles
        self.weights = weights
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
            assert self.dist is None

        # Check shape
        if self.dist is not None:
            assert self.b_shape == self.dist.batch_shape
            assert self.e_shape == self.dist.event_shape
        if self.particles is not None:
            assert self.s_shape + self.b_shape + self.e_shape == self.particles.shape
        if self.weights is not None:
            assert self.s_shape == self.weights.shape


# TODO general-inf: LinkData
class LinkData:
    """
        Identify the *data* of a ***directed*** link between a factor node and a variable node. Stores intermediate
            messages in the **message memory**.
        Note that links are directional, and two of such links should be specified with opposite directions to
            represent a bidirectional link between a factor node and a variable node, typically in the case of condacts.
        During construction of the graph, its instance will be passed to `NetworkX` methods as the edge data to
            instantiate an edge.
    """

    def __init__(self, vn, fn, var_list, to_fn, **kwargs):
        """
        :param vn:      name of the variable node that this link is incident to
        :param var_list:    list of variables of the adjacent variable node
        :param to_fn:   True/False indicating whether this link is pointing toward a factor node
        """
        # Link message memory
        self.memory = None
        # Whether this message is a new one just sent by adjacent node and haven't been read by the other node
        self.new = False

        # Following fields should correspond to the ones in the incident variable node
        self.vn = vn
        self.fn = fn
        self.var_list = var_list

        # Whether this link is pointing toward a factor node
        self.to_fn = to_fn

        # Register additional attributes, if there are any. For example, a flag indicating whether this linkdata
        #   contains negated message in an action conditional pattern.
        self.attri = kwargs

        # Record the dimensions of link message. Use to check potential dimension mismatch
        self._dims = [var.size for var in self.var_list]

        # Pretty log for GUI display
        self._pretty_log = {}

    def __str__(self):
        # Override for pretty debugging and printing
        fn_name = self.fn.name
        vn_name = self.vn.name
        if self.to_fn:
            return vn_name + " --> " + fn_name
        else:
            return fn_name + " --> " + vn_name

    def set(self, new, epsilon):
        """
            Set the link message memory to the new message arriving at this link. Implement the optimization so that
                memory content is not changed if new message differs from existing content by less than epsilon
            Default to check the absolute maximum difference between the old and new messages
        :param new:         new arriving message
        :param epsilon:     epsilon criterion
        """
        # Check message type and size. In message is constant (int or float), CHANGE TO TENSOR
        assert isinstance(new, (int, float, torch.Tensor))
        if isinstance(new, (int, float)):
            new = new * torch.ones(self._dims)
        else:
            size = list(new.shape)
            if size != self._dims:
                raise ValueError(
                    "The new message's dimension '{}' does not match the link memory's preset dimension '{}'. "
                    "Target variable node: '{}', link direction: toward factor node = '{}'"
                        .format(size, self._dims, str(self.vn), self.to_fn))

        # Check epsilon condition: maximal absolute difference < epsilon
        #   TODO: allow other types of epsilon condition
        if self.memory is not None:
            diff = torch.max(torch.abs(self.memory - new))
            if diff < epsilon:
                return

        self.memory = new
        self.new = True

    def read(self):
        """
            Return the current content stored in memory. Set new to False to indicate this link message have been read
                since current cycle
        :return:    message content
        """
        self.new = False
        return self.memory
