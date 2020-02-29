"""
    Basic structures in the graphical architecture
"""
import torch

# For now will use torch.Tensor directly. Not really necessary to make a subclass of Tensor, which may have side effects
#   to the underlying computation graphs if we do so.

# class Message(torch.Tensor):
#     """
#         A subclass of pytorch tensor. Stores message plm as tensors and can be directly manipulated, but keep extra
#             bookkeeping information for message processing mechanism in Sigma.
#     """
#
#     # Override __new__ so we can use Message() to create new tensors the same way as we use torch.Tensor()
#     @staticmethod
#     def __new__(cls, *args, **kwargs):
#         return super().__new__(cls, *args, **kwargs)
#
#     def __init__(self, x, *args, **kwargs):
#         super().__init__()
#


class Variable:
    """
        Variable as in variable nodes in the graphical architecture. Store information about this variable such as
            whether it is unique or universal.
        Note: The equality of variables' identity is determined by the equality of their name, regardless of the values
            of other fields
    """

    def __init__(self, name, size, probabilistic=False, unique=False, normalize=False):
        """
        :param name:            Variable name
        :param size:            The size, or maximum number of regions, of this variable dimension
        :param probabilistic:   True/False, whether this variable is probabilistic
        :param unique:          True/False, whether this variable is unique or universal
        :param normalize:       True/False, whether this variable is to be normalized
        """
        self.name = name
        self.size = size
        self.probabilistic = probabilistic
        self.unique = unique
        self.normalize = normalize

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


class LinkData:
    """
        Identify the *data* of a ***directed*** link between a factor node and a variable node. Stores intermediate
            messages in the **message memory**.
        Note that links are directional, and two of such links should be specified with opposite directions to
            represent a bidirectional link between a factor node and a variable node, typically in the case of condacts.
        During construction of the graph, its instance will be passed to `NetworkX` methods as the edge data to
            instantiate an edge.
    """

    def __init__(self, vn, var_list, to_fn):
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
        self.var_list = var_list

        # Whether this link is pointing toward a factor node
        self.to_fn = to_fn

        # Record the dimensions of link message. Use to check potential dimension mismatch
        self._dims = [var.size for var in self.var_list]

        # Pretty log for GUI display
        self._pretty_log = {}

    def set(self, new, epsilon):
        """
            Set the link message memory to the new message arriving at this link. Implement the optimization so that
                memory content is not changed if new message differs from existing content by less than epsilon
            Default to check the absolute maximum difference between the old and new messages
        :param new:         new arriving message
        :param epsilon:     epsilon criterion
        """
        # Check dimension mismatch
        size = list(new.shape)
        if size != self._dims:
            raise ValueError("The new message's dimension '{}' does not match the link memory's preset dimension '{}'. "
                             "Target variable node: '{}', link direction: toward factor node = '{}'"
                             .format(size, self._dims, str(self.vn), self.to_fn))

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


