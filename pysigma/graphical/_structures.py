"""
    Basic structures in the graphical architecture
"""
import torch
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from torch.distributions.constraints import Constraint
from enum import Enum
from collections.abc import Iterable
from torch.distributions.kl import kl_divergence as kl

from ._nodes import VariableNode, FactorNode
from ..utils import Dist2Params, Params2Dist


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
    def __init__(self, name: str, metatype: VariableMetatype, size: int, constraint: Constraint):
        """
            Instantiate a variable
        """
        assert isinstance(name, str)
        assert isinstance(metatype, VariableMetatype)
        assert isinstance(size, int)
        assert isinstance(constraint, Constraint)

        # Variable name, its identity. Used for variable matching. Of type str
        self.name = name
        # Variable meta-type, of type VariableMetatype
        self.metatype = metatype
        # Variable size. Size of the dimension that the variable corresponds to. Of type int
        self.size = size
        # Constraints of the variable's value range. Of type Constraint from PyTorch. Only set when variable type is
        #   Random.
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
    def __init__(self, msg_type: MessageType, sample_shape: torch.Size, batch_shape: torch.Size, event_shape: torch.Size,
                 dist: Distribution = None, particles: torch.Tensor = None, weights: torch.Tensor = None):
        """
            Instantiate a message
        """
        assert isinstance(msg_type, MessageType)
        assert isinstance(sample_shape, torch.Size) and len(sample_shape) == 1
        assert isinstance(batch_shape, torch.Size) and len(batch_shape) >= 1
        assert isinstance(event_shape, torch.Size) and len(event_shape) >= 1
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
            assert self.dist is None

        # Check shape
        if self.dist is not None:
            assert self.b_shape == self.dist.batch_shape
            assert self.e_shape == self.dist.event_shape
        if self.particles is not None:
            assert self.s_shape + self.b_shape + self.e_shape == self.particles.shape
        if self.weights is not None:
            assert self.s_shape == self.weights.shape

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


# TODO general-inf: LinkData
class LinkData:
    """
        Identify the data of a directed link between a factor node and a variable node. Stores intermediate
            messages in the message memory
        Note that links are directional, and two of such links should be specified with opposite directions to
            represent a bidirectional link between a factor node and a variable node, typically in the case of condacts.
        During construction of the graph, its instance will be passed to NetworkX methods as the edge data to
            instantiate an edge.
    """
    def __init__(self, vn, fn, var_list, to_fn, epsilon, **kwargs):
        """
        :param vn:      name of the variable node that this link is incident to
        :param var_list:    list of variables of the adjacent variable node
        :param to_fn:   True/False indicating whether this link is pointing toward a factor node
        """
        assert isinstance(vn, VariableNode)
        assert isinstance(fn, FactorNode)
        assert isinstance(var_list, Iterable) and all(isinstance(var, Variable) for var in var_list)
        assert isinstance(to_fn, bool)
        assert isinstance(epsilon, float)

        # Message memory, of type Message
        self.memory = None
        # Whether this message is new, and haven't been read by recipient node. Of type bool
        self.new = False
        # Incident nodes and their variable list
        self.vn = vn                # Of type VariableNode
        self.fn = fn                # Of type FactorNode
        self.var_list = var_list    # Iterable of Variable
        # Link direction. Whether pointing toward a factor node. Of type bool
        self.to_fn = to_fn
        # Threshold of KL-divergence metric to measure whether a candidate message is different from the existing one
        self._epsilon = epsilon
        # Reserved field for additional attributes. Arbitrary type
        self.attri = kwargs
        # Dimension of the link message. Determined from var_list. Of type torch.Size
        self._dims = torch.Size([var.size for var in self.var_list])
        # Pretty log
        self._pretty_log = {}

    def __str__(self):
        # Override for pretty debugging and printing
        fn_name = self.fn.name
        vn_name = self.vn.name
        if self.to_fn:
            return vn_name + " --> " + fn_name
        else:
            return fn_name + " --> " + vn_name

    def set(self, new_msg, check_diff=True, clone=False):
        """
            Set the link message memory to the new message arriving at this link.
            If check_diff is True, then will check if the new message is different from the existing one before
                replacing the existing with the new one.
              If the two message are of different types, then they will be deemed as different. Otherwise, KL-divergence
                will be used as the metric and its value to be compared with pre-set epsilon value. For Particles
                messages, the particles will be assumed to be identical, and the weights will be compared by treating
                them as probs to a Categorical distribution (assuming the weights are already properly normalized),
                from which the KL-divergence value is extracted.
              The following KL formula is used:
                    KL(old || new) = E[ log(old(x) / new(x) ],   x ~ old(x)
            If clone is True, then will store a cloned new_msg
        """
        assert isinstance(new_msg, Message)

        # If check_diff is False or no existing message or message types are different, will replace existing message
        if self.memory is None or check_diff is False or new_msg.type != self.memory.type:
            self.memory = new_msg.clone() if clone is True else new_msg
            self.new = True
            return

        # Otherwise, check difference by KL-divergence
        if new_msg.type in [MessageType.Tabular, MessageType.Distribution]:
            # If message type is Tabular or Distribution, compute KL directly
            p = self.memory.dist
            q = new_msg.dist
        else:
            # Otherwise, message type are Particles, so we will temporarily instantiate Categorical distributions
            p_probs = self.memory.weights
            q_probs = new_msg.weights
            p = Categorical(probs=p_probs)
            q = Categorical(probs=q_probs)
        val = kl(p, q)

        # Compare KL value with pre-set epsilon
        if val > self._epsilon:
            self.memory = new_msg.clone() if clone is True else new_msg
            self.new = True

    def read(self):
        """
            Return the current content stored in memory. Set new to False to indicate this link message have been read
                since current cycle
        :return:    memory content
        """
        self.new = False
        return self.memory
