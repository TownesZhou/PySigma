"""
    Basic structures in the graphical architecture
"""
import torch
import torch.distributions as Dist
from enum import Enum


class Variable:
    """
        Variable as in variable nodes in the graphical architecture. Store information about this variable such as
            whether it is unique or universal.
        Note: The equality of variables' identity is determined by the equality of their name, regardless of the values
            of other fields
    """

    def __init__(self, name, size, probabilistic=False, unique=False, normalize=False, sum_op=None):
        """
        :param name:            Variable name
        :param size:            The size, or maximum number of regions, of this variable dimension
        :param probabilistic:   True/False, whether this variable is probabilistic
        :param unique:          True/False, whether this variable is unique or universal
        :param normalize:       True/False, whether this variable is to be normalized
        :param sum_op:          Summarization operation. If none, then default to "max" if var is universal or "sum"
                                    if var is unique.
        """
        self.name = name
        self.size = size
        self.probabilistic = probabilistic
        self.unique = unique
        self.normalize = normalize

        if sum_op is None:
            self.sum_op = "sum" if self.unique else "max"
        else:
            self.sum_op = sum_op

    def __eq__(self, other):
        # override so '==' operator test the 'name' field
        return self.name == other.name

    def __ne__(self, other):
        # override so '!=' operator test the 'name' field
        return self.name != other.name

    def __str__(self):
        # override to provide the name as the string representation
        return self.name

    def __hash__(self):
        # override so that hash value is that of the Variable's name (which is treated as identity)
        return hash(self.name)


# TODO general-inf: Generalized message type and message representation
class MessageType(Enum):
    """
        Enum class to represent message types
    """
    Tabular = 0
    Distribution = 1
    Particles = 2


class Message():
    """
        Message structure to support general inference.
        Three basic message type:
            - Tabular factor
            - Parametric distribution
            - Particle list
        Tabular factors will be represented by Categorical distribution, a special case of parametric distribution
            message.
        The message type does not impose restriction on the underlying message types available for use, and thus
            different representations may coexist. For instance, in a Distribution message, a torch.distribution may
            coexists with a particle list, whereas in a Particles message only particle list exists most of the time.
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

    def __init__(self, msg_type: MessageType, sample_shape: torch.Size, batch_shape: torch.Size, event_shape: torch.Size,
                 dist: Dist = None, particles: torch.Tensor = None, weights: torch.Tensor = None):
        """
            Instantiate a message
        """
        assert isinstance(msg_type, MessageType)
        assert isinstance(sample_shape, torch.Size)
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(event_shape, torch.Size)
        assert dist is None or isinstance(dist, Dist.Distribution)
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
            assert isinstance(self.dist, Dist.Categorical)
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
