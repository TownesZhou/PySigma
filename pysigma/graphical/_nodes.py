"""
    All kinds of nodes in the graphical architecture
"""
import torch
from abc import ABC, abstractmethod
from collections.abc import Iterable
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from defs import Variable, MessageType, Message
from utils import DistributionServer
import warnings


class LinkData:
    """
        Identify the data of a directed link between a factor node and a variable node. Stores intermediate
            messages in the message memory
        Note that links are directional, and two of such links should be specified with opposite directions to
            represent a bidirectional link between a factor node and a variable node, typically in the case of condacts.
        During construction of the graph, its instance will be passed to NetworkX methods as the edge data to
            instantiate an edge.
    """
    def __init__(self, vn, fn, to_fn, msg_shape, epsilon, **kwargs):
        """
        :param vn:      VariableNode instance that this link is incident to
        :param fn:      FactorNode instance that this link is incident to
        :param to_fn:   True/False indicating whether this link is pointing toward a factor node
        :param msg_shape    Fixed message shape that this linkdata will carry. Used for checking dimensions
                            Conceptually, it should be  sample_shape + batch_shape + event_shape
        :param epsilon:     epsilon upper bound for checking message difference using KL divergence
        """
        assert isinstance(vn, VariableNode)
        assert isinstance(fn, FactorNode)
        assert isinstance(msg_shape, torch.Size)
        assert isinstance(to_fn, bool)
        assert isinstance(epsilon, float)

        # Message memory, of type Message
        self.memory = None
        # Whether this message is new, and haven't been read by recipient node. Of type bool
        self.new = False
        # Incident nodes and their variable list
        self.vn = vn                # Of type VariableNode
        self.fn = fn                # Of type FactorNode
        self.msg_shape = msg_shape
        # Link direction. Whether pointing toward a factor node. Of type bool
        self.to_fn = to_fn
        # Threshold of KL-divergence metric to measure whether a candidate message is different from the existing one
        self.epsilon = epsilon
        # Reserved field for additional attributes. Arbitrary type
        self.attr = kwargs
        # Pretty log
        self.pretty_log = {}

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
        # Check new message shape
        assert self.msg_shape == new_msg.s_shape + new_msg.b_shape + new_msg.e_shape
        assert self.memory is None or (self.memory.s_shape == new_msg.s_shape and
                                       self.memory.b_shape == new_msg.b_shape and
                                       self.memory.e_shape == new_msg.e_shape)

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
            # Otherwise, message type are Particles, so we will treats the particle weights as 'probs' to Categorical
            #   distributions and instantiate temporary distribution instance
            # Need to permute the dimension because we are taking the particle indexing dimension as the probs dimension
            n_dims = len(self.memory.weights.shape)
            p_probs = self.memory.weights.permute(list(range(1, n_dims)) + [0])
            q_probs = new_msg.weights.permute(list(range(1, n_dims)) + [0])
            p = DistributionServer.param2dist(Categorical, p_probs, self.memory.b_shape, torch.Size([]))
            q = DistributionServer.param2dist(Categorical, q_probs, new_msg.b_shape, torch.Size([]))
        val = DistributionServer.kl_norm(p, q)

        # Compare KL value with pre-set epsilon
        if val > self.epsilon:
            self.memory = new_msg.clone() if clone is True else new_msg
            self.new = True

    def read(self, clone: bool = False):
        """
            Return the current content stored in memory. Set new to False to indicate this link message have been read
                since current cycle
            if clone is True, return a cloned version of memory content
        """
        self.new = False
        msg = self.memory.clone() if clone is True else self.memory
        return msg


class Node(ABC):
    """
        The super class of `FactorNode` and `VariableNode`. It declares common attributes between `FactorNode` and
            `VariableNode`, for example a flag indicating whether this node has reach quiescence, i.e., whether no new
            message shall be sent via adjacent link in curretn cycle. During construction of the graph, its instance
            will be passed to `NetworkX` methods to instantiate a node.
    """

    def __init__(self, name):
        self.name = name
        # Flag indicating whether quiescence reached in current cycle. If so, no sum-product local processing needed at
        #   this node.
        self.quiescence = False
        # Flag indicating whether this node has been visited (compute() method called) during a decision cycle
        self.visited = False

        # List of LinkData of those link connecting to this node, incoming and outgoing ones respectively, from
        #   which we retrieve messages
        self.in_linkdata = []
        self.out_linkdata = []

        # Global logging info
        self.log = {}
        self.pretty_log = {}  # Log info to display at the GUI

    def __str__(self):
        # override to provide the node's name as its string representation
        return self.name

    def check_quiesce(self):
        """
            Check quiescence of incoming linkdata to determine whether this node has reached quiescence
            Return True if reached quiescence, False otherwise
            Will update self.quiescence accordingly
        """
        quiesced = True
        for in_ld in self.in_linkdata:
            if in_ld.new:
                quiesced = False
                break
        self.quiescence = quiesced
        return quiesced

    @abstractmethod
    def add_link(self, linkdata):
        pass

    @abstractmethod
    def compute(self):
        pass


class FactorNode(Node, ABC):
    """
        Factor node abstract base class. Guarantees that all incident nodes are variable nodes.
    """
    def __init__(self, name):
        super(FactorNode, self).__init__(name)

    def add_link(self, linkdata):
        """
            Add a linkdata connecting to a variable node
        """
        assert isinstance(linkdata, LinkData)
        assert linkdata.fn is self
        if linkdata in self.in_linkdata + self.out_linkdata:
            return

        if linkdata.to_fn:
            self.in_linkdata.append(linkdata)
        else:
            self.out_linkdata.append(linkdata)


class VariableNode(Node, ABC):
    """
        Variable node abstract base class.
    """

    def __init__(self, name, index_var, rel_var_list, ran_var_list):
        """
            Decalre a VariableNode
        :param name:        name of the variable node
        :param index_var:      Particle indexing variable
        :param rel_var_list:   list of Variables representing the relational variables of this variable nodes
        :param ran_var_list:   list of Variables representing the random variables of this variable nodes
        """
        super(VariableNode, self).__init__(name)
        assert isinstance(index_var, Variable)
        assert isinstance(rel_var_list, Iterable) and all(isinstance(v, Variable) for v in rel_var_list)
        assert isinstance(ran_var_list, Iterable) and all(isinstance(v, Variable) for v in ran_var_list)

        self.index_var = index_var
        self.rel_var_list = rel_var_list
        self.ran_var_list = ran_var_list
        self.s_shape = torch.Size([index_var.size])
        self.b_shape = torch.Size(list(rel_var.size for rel_var in rel_var_list))
        self.e_shape = torch.Size(sum(list(ran_var.size for ran_var in ran_var_list)))

    def add_link(self, linkdata):
        """
            Register the LinkData connecting a factor node to this variable node.
            Check that the variable list specified in linkdata agrees with that pre-specified at this variable node
        """
        assert isinstance(linkdata, LinkData)
        assert linkdata.msg_shape == self.s_shape + self.b_shape + self.e_shape
        if linkdata in self.in_linkdata + self.out_linkdata:
            return

        if linkdata.to_fn:
            self.out_linkdata.append(linkdata)
        else:
            self.in_linkdata.append(linkdata)


class DFN(FactorNode):
    """
        Default (Dummy) Factor Node. No special computation. Simply relay message to one or multiple variable nodes
        Requires that incident variable nodes have the same variable list
        Only admit one incoming link but can connect with multiple outgoing links
    """
    def __init__(self, name):
        super(DFN, self).__init__(name)
        self.pretty_log["node type"] = "Default Factor Node"

        # Since all incident nodes should have the same variable list, we can therefore log it here as an attribute
        self.index_var = None
        self.rel_var_list = None
        self.ran_var_list = None

    def add_link(self, linkdata):
        assert isinstance(linkdata, LinkData)
        # Make sure no more than on incoming alink
        assert not linkdata.to_fn or len(self.in_linkdata) == 0
        # Also make sure incident variable nodes' var_list agree with each other
        if self.index_var is None:
            self.index_var = linkdata.vn.index_var
            self.rel_var_list = linkdata.vn.rel_var_list
            self.ran_var_list = linkdata.vn.ran_var_list
        else:
            assert self.index_var == linkdata.vn.index_var and \
                   self.rel_var_list == linkdata.vn.rel_var_list and \
                   self.ran_var_list == linkdata.vn.ran_var_list

        super(DFN, self).add_link(linkdata)

    def compute(self):
        self.visited = True

        in_ld = self.in_linkdata[0]
        msg = in_ld.read()
        for out_ld in self.out_linkdata:
            out_ld.set(msg)


class DVN(VariableNode):
    """
        Default (Dummy) Variable Node. No special computation. Simply relay message to one or multiple factor nodes
        Only admit one incoming link but can connect with multiple outgoing links
    """
    def __init__(self, name, index_var, rel_var_list, ran_var_list):
        super(DVN, self).__init__(name, index_var, rel_var_list, ran_var_list)
        self.pretty_log["node type"] = "Default Variable Node"

    def add_link(self, linkdata):
        assert isinstance(linkdata, LinkData)
        # Make sure no more than on incoming and one outgoing link
        assert linkdata.to_fn or len(self.in_linkdata) == 0

        super(DVN, self).add_link(linkdata)

    def compute(self):
        self.visited = True

        in_ld = self.in_linkdata[0]
        msg = in_ld.read()
        for out_ld in self.out_linkdata:
            out_ld.set(msg)


class LTMFN(FactorNode):
    """
        Long-Term Memory Factor Node. Holds the distribution class of this predicate, and if necessary can draw particle
            events from currently assumed distribution instance.

        Admits incoming link from WMVN that contains combined action message to this predicate by the end of the
            decision cycle, as well as the incoming link from parameter feed and WMFN that contains parameter messages.
            Therefore needs special attribute in the links to identify which one sends "event" messages and which one
            sends "parameter" messages.

        Only admits one incoming and one outgoing event message link, the former should be connected from WMVN_IN
            and the later from WMVN_OUT (can be the same WMVN also). However can admits multiple incoming parameter
            message link. The parameters of the assumed distribution instance will be taken as the SUM over all the
            incoming parameters.

        Will assume the parameters are represented by Particle type message with shape (batch_shape + param_shape)
        Therefore LTMFN take as input parameter messages, and produce as output distribution/particle messages

        Can be toggled regarding whether to draw particles during compute(). If to draw, default to draw particles with
            uniform weights
    """
    def __init__(self, name, distribution_class, num_particles, batch_shape, event_shape, param_shape):
        super(LTMFN, self).__init__(name)
        self.pretty_log["node type"] = "Long-Term Memory Factor Node"

        assert issubclass(distribution_class, Distribution)
        assert isinstance(num_particles, int)
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(event_shape, torch.Size)
        assert isinstance(param_shape, torch.Size)

        self.dist_class = distribution_class
        self.num_particles = num_particles
        self.sample_shape = torch.Size([num_particles])
        self.batch_shape = batch_shape
        self.event_shape = event_shape
        self.param_shape = param_shape

        # Flag that indicates whether to sample particles during compute()
        self.to_sample = False

        # Knowledge
        self.dist = None            # Distribution instance at current decision cycle
        self.particles = None       # Particle list at current decision cycle
        self.weights = None         # Particle weights
        self.log_density = None     # The log-pdf of particles

    def add_link(self, linkdata):
        """
            Only admits one incoming and one outgoing event message link, the former should be connected from WMVN_IN
                and the later from WMVN_OUT (can be the same WMVN also). However can admits multiple incoming parameter
                message link.
        """
        assert isinstance(linkdata, LinkData)

        # Only admit one outgoing link and that must be WMVN. Check variable dimensions
        if not linkdata.to_fn:
            assert len(self.out_linkdata) == 0 and isinstance(linkdata.vn, WMVN)
            assert linkdata.dims == (self.sample_shape + self.batch_shape + self.event_shape)
            self.out_linkdata.append(linkdata)
        # Can admit multiple incoming links. Check variable dimension to be compatible with parameter messages
        else:
            assert linkdata.dims == (self.batch_shape + self.param_shape)
            self.in_linkdata.append(linkdata)

    def toggle_draw(self, to_sample=False):
        # Turn on or off whether this LTMFN will sample particles during compute()
        self.to_sample = to_sample

    def retrive_particles(self):
        # Return the particles and sampling log density
        return self.particles, self.log_density

    def compute(self):
        self.visited = True

        assert len(self.in_linkdata) > 0, "No incoming link to LTMFN '{}'".format(self.name)
        # Obtain parameter from incoming links, generate new distribution instance, and sample particles if necessary
        msg = torch.zeros(self.batch_shape + self.param_shape)
        for in_ld in self.in_linkdata:
            tmp = in_ld.read()
            msg += tmp
        # Instantiate distribution instance
        self.dist = Params2Dist.convert(msg, self.dist_class, self.batch_shape, self.event_shape)
        # Draw uniform sample if necessary
        if self.to_sample:
            self.particles = self.dist.sample(sample_shape=self.sample_shape)
            self.log_density = self.dist.log_prob(self.particles)

        # Generate message and send it to the WMVN
        # Particles message. Containing both distribution instance and particle list
        if self.to_sample:
            out_msg = Message(MessageType.Particles, self.sample_shape, self.batch_shape, self.event_shape, self.dist,
                              self.particles, self.weights, self.log_density)
        # Distribution message. Containing only distribution instance
        else:
            # Tabular message if distribution of Categorical class
            if issubclass(self.dist_class, Categorical):
                out_msg = Message(MessageType.Tabular, self.sample_shape, self.batch_shape, self.event_shape, self.dist)
            # Otherwise Distribution message
            else:
                out_msg = Message(MessageType.Distribution, self.sample_shape, self.batch_shape, self.event_shape,
                                  self.dist)
        out_ld = self.out_linkdata[0]
        out_ld.set(out_msg)


class WMVN(VariableNode):
    """
        Working Memory Variable Node. Gate node connecting predicate structure to conditionals.
        Will attempt to combine incoming messages if there are multiple incoming links. Subsuming the functionality of
            FAN node in Lisp Sigma.
        When combining messages, will exclude message from the link to which the combined message is to send to (if
            such a bidirected link exists). This implements the Sum-Product algorithm's variable node semantics, if
            this WMVN is served as both WMVN_IN and WMVN_OUT, i.e., if the predicate is of memory-less vector type.

        Note that messages combination can be carried out only if ALL the message are of Tabular type, or all of them
            are of Particles type.
        If messages are all Tabular type, combination is performed by summing the probs vector of the Categorical
            distributions across messages.
        If messages are all Particles type, combination is performed by (1) first concatenating the particles, (2) then
            renormalize the weights, (3) and finally either resample or selecting the subset of particles of the highest
            weight, so that the total number of resulting particles matching sample_shape (num_particles)
    """
    def __init__(self, name, var_list, num_particles, to_resample=True):
        super(WMVN, self).__init__(name, var_list)
        self.pretty_log["node type"] = "Working Memory Variable Node"

        # Number of particles to keep when combining particles incoming messages
        self.num_particles = num_particles
        self.s_shape = torch.Size([self.num_particles])
        # Whether to resample, or to select highest rank of particles, when combining particle messages
        self.to_resample = True

    def compute(self):
        self.visited = True
        # Exit if no outgoing link
        if len(self.out_linkdata) == 0:
            return
        # Relay message if only one incoming link
        if len(self.in_linkdata) == 1:
            in_ld = self.in_linkdata[0]
            msg = in_ld.read()
            for out_ld in self.out_linkdata:
                # Throw a warning if the outgoing link is connected to the same factor node that the only incoming
                #   link is connected to, since in such case no message would be sent to that factor node
                if out_ld.fn is in_ld.fn:
                    warnings.warn("WMVN '{}' is connected to factor node '{}', while its only incoming link is also "
                                  "connected from the same factor node. In this case no message would be sent out to "
                                  "the factor node. Please check if the model is properly defined"
                                  .format(self.name, out_ld.fn.name))
                else:
                    out_ld.set(msg)
        # Otherwise, combine messages
        else:
            for out_ld in self.out_linkdata:
                # exclusion principle
                in_lds = list(in_ld for in_ld in self.in_linkdata if in_ld.fn is not out_ld.fn)
                msgs = list(in_ld.read() for in_ld in in_lds)
                # Check uniformity of incoming message type
                assert all(msg.type is MessageType.Tabular for msg in msgs) or \
                       all(msg.type is MessageType.Particles for msg in msgs), \
                    "In WMVN '{}': when attempting to send message via link '{}', failed to combine incoming messages "\
                    "because incoming messages are not all Tabular or all Particles type. Found message type '{}' " \
                    "from links '{}'".format(self.name, out_ld, list(msg.type for msg in msgs), in_lds)

                # Combine tabular messages: summing the probs and generate a new instance
                b_shape, e_shape = msgs[0].b_shape, msgs[0].e_shape
                if all(msg.type is MessageType.Tabular for msg in msgs):
                    probs = 0
                    for msg in msgs:
                        # Obtain probs parameter of each message's Categorical distribution using utility methods
                        tmp = Dist2Params.convert(msg.dist)
                        probs += tmp
                    dist = Params2Dist.convert(probs, Categorical, b_shape, e_shape)
                    out_msg = Message(MessageType.Tabular, self.s_shape, b_shape, e_shape, dist)

                # Combine particle messages
                else:
                    num = len(msgs)
                    # Concatenate particle list. Dimension 0 is the particle indexing dimension
                    particles = torch.cat(list(msg.particles for msg in msgs), dim=0)
                    weights = torch.cat(list(msg.weights for msg in msgs), dim=0)
                    log_densities = torch.cat(list(msg.log_density for msg in msgs), dim=0)
                    # normalize weights. Devide by the number of messages
                    weights *= 1 / num

                    # If to resample, then feed weights into a Categorical distribution and sample num_particles data
                    #   i.e., resample particles based on their weights
                    if self.to_resample:
                        # Create resampling distribution
                        rs_dist = Categorical(probs=weights)
                        # Sample particle indices
                        indices = rs_dist.sample(sample_shape=self.s_shape)
                        # Using particle indices to select particles
                        new_particles = particles.index_select(dim=0, index=indices)
                        new_log_dens = log_densities.index_select(dim=0, index=indices)
                        # Because we are resampling, the weights are uniform
                        new_weights = torch.ones(size=self.num_particles) * (1 / self.num_particles)

                    # Otherwise, sort the weights in descending and select the particles with the highest weights
                    else:
                        # Sort in descending order
                        new_weights, indices = weights.sort(dim=0, descending=True)
                        # Take firs num_particles elements
                        new_weights = new_weights[:self.num_particles]
                        indices = indices[:self.num_particles]
                        # Select particles
                        new_particles = particles.index_select(dim=0, index=indices)
                        new_log_dens = log_densities.index_select(dim=0, index=indices)
                        # Normalize new weights so that it sums to 1
                        new_weights *= (1 / new_weights.sum())
                    # Output message
                    out_msg = Message(MessageType.Particles, self.s_shape, b_shape, e_shape, dist=None,
                                      particles=new_particles, weights=new_weights, log_density=new_log_dens)

                # Send message
                out_ld.set(out_msg)


class PBFN(FactorNode):
    """
        Perception Buffer Factor Node. No special implementation needed. FactorNode original methods should suffice.
            Except for quiescence checking - Need to make sure compute() always proceed regardless of quiescence
    """
    def __init__(self, name, function=None, func_var_list=None):
        super(PBFN, self).__init__(name, function, func_var_list)
        self.pretty_log["node type"] = "Perceptual Buffer Function Node"

        # TODO: What should be the default function value for a PBFN ?

    def get_shape(self):
        # For use when checking perception content shape
        return [var.size for var in self._var_list]

    # Override check_quiesce() so that compute() will should always proceed. However this would not jeopardize
    # computation because message sent to ld is same as old, so ld.new would not be set to true under epsilon condition
    def check_quiesce(self):
        super(PBFN, self).check_quiesce()
        return False  # Always return False so that compute() will still proceed


class WMFN(FactorNode):
    """
        Working Memory Factor Node

        Special implementation for compute(): to ensure that messages coming out from WMFN is the WMFN content it self
            and stays unchanged through the graph solution phase, we do not let messages on incoming link take part in
            the computation of the outgoing message. They are only used during modification phase to modify the WMFN
            content by performing selection (for unique predicates) or directly replace the WMFN content (for universal
            predicates)
    """
    def __init__(self, name, function=None, func_var_list=None):
        super(WMFN, self).__init__(name, function, func_var_list)
        self.pretty_log["node type"] = "Working Memory Function Node"

    # Override so that only allow one incoming link, and that must be WMVN
    def add_link(self, linkdata):
        if linkdata.to_fn:
            assert isinstance(linkdata.vn, WMVN), "WMFN only admits an incoming link from WMVN"
            assert len(self._in_linkdata) == 0, "WMFN only admits one incoming link"
        super(WMFN, self).add_link(linkdata)

    # Override so that WMFN ignores messages on incoming link, and only send its own function outwards
    def compute(self):
        for out_ld in self._out_linkdata:
            out_ld.set(self._function, self._epsilon)

        # Set quiescence state
        self.quiescence = True

    # Also override get_quiesce() so that quiescence state is not determined by incoming links but rather compute()
    def check_quiesce(self):
        return self.quiescence


class ACFN(FactorNode):
    """
        Action-Combination Factor Node.

        Regarding combination operation:
            - If predicate is a vector predicate (all wm vars are vector vars), then operations are vector summation
            without normalization.
            - If predicate is a probabilistic predicate, will adopt probabilistic logical operations:
                1. Calculate a "positive combined action" message by taking the probabilistic OR of messages of all
                    incoming positive actions
                2. Calculate a "negative combined action" message by taking the probabilistic OR of messages of all
                    incoming negative actions. Note that the incoming messages should already been negated by NFN
                        E.g. For all incoming messages M_i, The PA (positive combined action) is calculated by
                            PA = 1 - (1 - M_1)*(1 - M_2)* ... *(1 - M_n)
                             Similarly for NA (negative combined action)
                3. Final message is calculated by taking the probabilistic AND of PA and (1 - NA). Semantically, this
                    may be interpreted as enforcing what these two combined actions propose should agree with each other
                            msg = PA * (1 - NA)

        Regarding message pre-normalization before taking the above computation:
            - For probabilistic message without normalization (not a discrete distribution), simply scale it so that
                individual values fall in range [0, 1]
            - For ones with normalization (representing a discrete distribution), linearly normalize over the
                distribution variable dimension, so that not only do individual values fall in range [0, 1], but values
                sum up to 1 across those variable dimensions

        Note: ACFN admits only one outgoing linkdata, which connects to corresponding predicate group's WMVN_IN
    """

    # TODO: Special implementation of sum-product to implement message disjunction
    def __init__(self, name, ):
        super(ACFN, self).__init__(name, function=None, func_var_list=None)
        self.pretty_log["node type"] = "Action Combination Function Node"

        # Record message from which linkdata is positive action and which is negative action
        self._pos_in_ld = []
        self._neg_in_ld = []
        self.pretty_log["positive actions from"] = []
        self.pretty_log["negative actions from"] = []

    def add_link(self, linkdata):
        # Make sure only add one outgoing link
        assert linkdata.to_fn or len(self._out_linkdata) == 0, "Attempting to add more than one outgoing link"
        # Make sure every linkdata's var list agree with each other
        if len(self._var_list) > 0:
            assert self._var_list == linkdata.var_list, \
                "linkdata var list mismatch in ACFN. Expects: {}, found: {}".format(self._var_list, linkdata.var_list)

        super(ACFN, self).add_link(linkdata)
        if linkdata.to_fn:
            if linkdata.attri['negation']:
                self._neg_in_ld.append(linkdata)
                self.pretty_log["negative actions from"].append(linkdata.vn.name)
            else:
                self._pos_in_ld.append(linkdata)
                self.pretty_log["positive actions from"].append(linkdata.vn.name)

    def compute(self):

        assert len(self._out_linkdata) == 1
        assert len(self._in_linkdata) > 0
        out_ld = self._out_linkdata[0]

        # Delay computation until all messages on incoming links are valid, i.e., not None
        # for ld in self._in_linkdata:
        #     # DO NOT USE read() HERE because we don't want to alter ld.new state
        #     if ld.memory is None:
        #         return

        # Determine whether to take probabilistic operation, whether any variable dimension represents distribution
        probabilistic = any(var.probabilistic for var in self._var_list)
        normal = any(var.normalize for var in self._var_list)
        normal_dim = list(self._var_list.index(var) for var in self._var_list if var.normalize)

        # util function to normalize msg
        def normalize(msg):
            # First clamp the msg tensor so individual values lie in range [0, 1]
            msg = msg.clamp(min=0., max=1.)
            # If the msg represents distribution, normalize over those dimensions
            if normal:
                msg = msg / msg.sum(dim=normal_dim, keepdim=True)   # linear scale
            return msg

        if probabilistic:
            # Calculate positive combined action PA
            pa = 1
            for ld in self._pos_in_ld:
                msg = ld.read()
                # If msg is None, simply skip it
                if msg is not None:
                    msg = normalize(msg)    # pre-normalize
                    pa *= 1 - msg
            pa = 1 - pa

            # Calculate negative combined action NA. To save computation, na here is actually the "negated" NA
            na = 1
            for ld in self._neg_in_ld:
                msg = ld.read()
                # If msg is None, simply skip it
                if msg is not None:
                    msg = normalize(msg)    # pre-normalize
                    # Note that msg for negative action should already be negated by a negation node,
                    #   so don't need to negate it again here
                    na *= msg

            # Combining PA with NA.
            msg = pa * na
            out_ld.set(msg, self._epsilon)

        else:
            # Simply take vector summation
            msg = 0
            for ld in self._in_linkdata:
                msg += ld.read()
            out_ld.set(msg)


class FFN(FactorNode):
    """
        Filter Node
    """

    # TODO
    def __init__(self, name, function=None, func_var_list=None):
        super(FFN, self).__init__(name, function, func_var_list)
        self.pretty_log["node type"] = "Filter Factor Node"


class NFN(FactorNode):
    """
        Negation Factor Node.

        For vector predicate (all variables are vector), simply take arithmetic negation
        For probabilistic predicate (have probabilistic variable), take probabilistic negation: neg = 1 - p
            Note: assume arriving messages are properly normalized (individual value in range [0, 1])

        Note: NFN admits at most two pairs of incoming / outgoing links (one pair in condition / action pattern,
            two pairs in condact pattern)
    """
    def __init__(self, name):
        """
        :param name:        Name of the node
        """
        super(NFN, self).__init__(name, function=None, func_var_list=None)
        self.pretty_log["node type"] = "Negation Factor Node"

    # Override to make sure only admit at most two pairs of incoming / outgoing links
    def add_link(self, linkdata):
        assert (not linkdata.to_fn or len(self._in_linkdata) <= 1), "Attempting to add more than two incoming links"
        assert (linkdata.to_fn or len(self._out_linkdata) <= 1), "Attempting to add more than two outgoing links"
        super(NFN, self).add_link(linkdata)

        self.pretty_log["negation type"] = "probabilistic" if any(var.probabilistic for var in self._var_list) else \
                                           "vector"

    def compute(self):

        # Check that there are equal number of incoming links and outgoing links
        assert len(self._in_linkdata) == len(self._out_linkdata), \
            "The number of of incoming links ({}) do not match the number of outgoing links ({}). " \
                .format(len(self._in_linkdata), len(self._out_linkdata))

        for in_ld in self._in_linkdata:
            out_ld = [ld for ld in self._out_linkdata if ld.vn is not in_ld.vn][0]

            # Determine if to use arithmetic negation or probabilistic negation
            probabilistic = any(var.probabilistic for var in self._var_list)

            msg = in_ld.read()
            if probabilistic:
                assert torch.all(msg > 0) and torch.all(msg < 1), \
                    "Though declared probabilistic, the incoming message at this Negation Factor Node does not lie " \
                    "within range [0, 1]. Max val: {}, Min val: {}".format(torch.max(msg), torch.min(msg))
                msg = 1 - msg       # Take probabilistic negation
            else:
                msg = -msg          # Otherwise, take arithmetic negation

            out_ld.set(msg, self._epsilon)


class NLFN(FactorNode):
    """
        Nonlinearity Factor Node. Element-wise nonlinearity "filters"
    """

    # TODO:
    def __init__(self, name, nonlinear):
        """
        :param name:        Name of the node
        :param nonlinear:  'str' or function objects, specifying the type of element-wise nonlinearity
        """
        super(NLFN, self).__init__(name, function=None, func_var_list=None)
        self.nonlinear = nonlinear
        self.pretty_log["node type"] = "Nonlinearity Factor Node"
        self.pretty_log["nonlinearity"] = str(nonlinear)


class ADFN(FactorNode):
    """
        Affine Delta Factor Node, generally for matching between WM variables and Pattern variables
        Performs delta function masking and variable swapping
        Essential component of the Rete's Alpha network's within pattern variable matching

        Note: ADFN only admits at most two pairs of incoming / outgoing links. (one pair in condition / action pattern,
            two pairs in condact pattern)

        Note: Special implementation needed for resolving potential conflicts between pattern variables and wm variables,
                especially mismatch in sizes.

        pt_var_info schema:
            { wm_var_name :
                  { "name" : pt_var_name
                    "type" : "var" or "const"
                    "vals" : int/str values if type is const or None otherwisea
                    "rel"  : relation, if specified, otherwise None } }
    """
    def __init__(self, name, pt_var_info):
        super(ADFN, self).__init__(name, function=None, func_var_list=None)
        self.pretty_log["node type"] = "Affine Delta Factor Node"
        self._pt_var_info = pt_var_info

    # Override to make sure only admit at most two pairs of incoming / outgoing links
    def add_link(self, linkdata):
        assert (not linkdata.to_fn or len(self._in_linkdata) <= 1), "Attempting to add more than two incoming links"
        assert (linkdata.to_fn or len(self._out_linkdata) <= 1), "Attempting to add more than two outgoing links"
        super(ADFN, self).add_link(linkdata)

    # Override compute() to implement tensor manipulation
    def compute(self):
        """
            Three cases for each variable dimension:
                1. If associated pattern var shared among multiple wm var:
                    - If message goes inward, take diagonal values and shrink dimension
                    - If message goes outward, expand dimensions, put values on diagonals and put 0 everywhere else
                2. If associated pattern var is constant,
                    - If message goes inward, take the slices and concatenate, slices taken in the order specified by 'vals'
                    - If message goes outward, split into slices and put in place. Put 0 everywhere else
                3. If associated pattern var is a variable but is distinct, then simply swap variable.

            In cases 1 and 3, need to account for mismatch in variable dimension sizes when swapping variables:
                - If need to enlarge dimension size, append necessary slices of 0's at the tail of that dimension
                - If need to shrink dimension size, truncate necessary slices at the tail of that dimension

            # TODO: Accomodate for within-pattern variable relations in future iterations
                        E.g.  pred[ (arg1 v) (arg2 (v + 3)) ]
                    Or more generally,
                        E.g.  pred[ (arg1 v) (arg2 f(v)) ]
                    Where f(v) is a custom mapping embodying the relation
        """

        # Check that there are equal number of incoming links and outgoing links
        assert len(self._in_linkdata) == len(self._out_linkdata), \
            "The number of of incoming links ({}) do not match the number of outgoing links ({}). " \
                .format(len(self._in_linkdata), len(self._out_linkdata))

        for in_ld in self._in_linkdata:
            out_ld = [ld for ld in self._out_linkdata if ld.vn is not in_ld.vn][0]

            # Compare var_list against pt_var_info and determin wm vars and pt vars
            # Use set to discard orders for comparison
            in_ld_varnames, out_ld_varnames = set(var.name for var in in_ld.var_list), set(var.name for var in out_ld.var_list)
            wm_varnames, pt_varnames = set(self._pt_var_info.keys()), set(v["name"] for v in self._pt_var_info.values())
            assert (in_ld_varnames == wm_varnames and out_ld_varnames == pt_varnames) or \
                   (in_ld_varnames == pt_varnames and out_ld_varnames == wm_varnames), \
                "wm vars and pattern vars provided by pt_var_info does not match that as specified in the linkdata. " \
                "In pt_var_info: wm vars - '{}', pattern vars - '{}'. " \
                "In linkdata: incoming link vars - '{}', outgoing link vars - '{}'." \
                .format(wm_varnames, pt_varnames, in_ld_varnames, out_ld_varnames)

            # Determine direction of this pair of links. Inward or outward, w.r.t. beta network
            inward = True if in_ld_varnames == wm_varnames else False

            in_varname_list = list(var.name for var in in_ld.var_list)
            out_varname_list = list(var.name for var in out_ld.var_list)
            # in_var2dim = {v.name: k for k, v in enumerate(in_ld.var_list)}
            # out_var2dim = {v.name: k for k, v in enumerate(out_ld.var_list)}
            in_dim = [var.size for var in in_ld.var_list]
            out_dim = [var.size for var in out_ld.var_list]

            msg = in_ld.read()      # read message
            assert isinstance(msg, torch.Tensor)
            trace_varnames = list(var.name for var in in_ld.var_list)  # Trace the change of variable dimensions of msg

            # Build ptvar -> wmvar lookup table to distinguish shared pt vars from distinct ones
            ptvar2wmvar = {}
            for wm_varname, v in self._pt_var_info.items():
                pt_varname = v['name']
                if pt_varname not in ptvar2wmvar.keys():
                    ptvar2wmvar[pt_varname] = [wm_varname]
                else:
                    ptvar2wmvar[pt_varname].append(wm_varname)

            for wm_varname, v in self._pt_var_info.items():
                pt_varname = v['name']
                # First do case 2, constant variables. Throughout this step, variable order should not be altered
                if v['type'] == "const":
                    vals = v['vals']
                    # If inward, simply index select
                    if inward:
                        dim = in_varname_list.index(wm_varname)
                        # If value is None, then simply take the full span
                        if vals is not None:
                            msg = msg.index_select(dim=dim, index=torch.LongTensor(vals))
                        trace_varnames[dim] = pt_varname
                    # If outward, then need to take slice separately, create empty slices, and concatenate
                    else:
                        dim = in_varname_list.index(pt_varname)
                        # If vals is None, then treat it as full span
                        vals = list(range(in_dim[dim])) if vals is None else vals
                        assert in_dim[dim] == len(vals)     # This should always be correct since size of const pt vars are determined by len of vals
                        slice_shape = list(msg.shape)
                        slice_shape[dim] = 1
                        dim_target_size = out_dim[out_varname_list.index(wm_varname)]
                        # Slice messages
                        sliced_msg = msg.split(1, dim=dim)      # Split into tuple of tensors with shape slice_shape
                        assert all(list(t.shape) == slice_shape for t in sliced_msg)
                        # Create target message
                        target_slices = list(torch.zeros(size=slice_shape)
                                             if i not in vals else sliced_msg[vals.index(i)]
                                             for i in range(dim_target_size))
                        # Concatenate to form new message
                        msg = torch.cat(target_slices, dim=dim)
                        trace_varnames[dim] = wm_varname

                # Second do case 3, swap distinct pt vars and account for mismatch in size
                elif len(ptvar2wmvar[pt_varname]) == 1:
                    if inward:      # inward should be enlarging size, if any
                        dim = in_varname_list.index(wm_varname)
                        wm_size = in_dim[dim]
                        pt_size = out_dim[out_varname_list.index(pt_varname)]
                        assert wm_size <= pt_size
                        trace_varnames[dim] = pt_varname
                        # Enlarge dimension size with zero tensor if needed
                        if pt_size > wm_size:
                            extra_shape = list(msg.shape)
                            extra_shape[dim] = pt_size - wm_size
                            msg = torch.cat((msg, torch.zeros(size=extra_shape)), dim=dim)
                    else:           # outward should be shrinking size, if any
                        dim = in_varname_list.index(pt_varname)
                        pt_size = in_dim[dim]
                        wm_size = out_dim[out_varname_list.index(wm_varname)]
                        assert wm_size <= pt_size
                        trace_varnames[dim] = wm_varname
                        # Shrink dimension size by truncating tail is needed
                        if pt_size > wm_size:
                            msg = torch.split(msg, wm_size, dim=dim)[0]     # Only take the first chunk

            # Finally case 1, where variable order may be changed due to taking the diagonal
            for pt_varname, wm_varname_list in ptvar2wmvar.items():
                if len(wm_varname_list) > 1:
                    # If message goes inward, take diagonal values and shrink dimension
                    if inward:
                        # Diagonalize first pair
                        wm_varname1 = wm_varname_list[0]
                        wm_varname2 = wm_varname_list[1]
                        dim1 = trace_varnames.index(wm_varname1)    # use trace_varnames because var order may have
                        dim2 = trace_varnames.index(wm_varname2)    # changed
                        msg = torch.diagonal(msg, dim1=dim1, dim2=dim2)
                        trace_varnames.remove(wm_varname1)
                        trace_varnames.remove(wm_varname2)
                        trace_varnames.append(pt_varname)       # Diagonalized dimension is appended as the last dim

                        # If there are more than two wm vars sharing the same pt vars
                        for i in range(2, len(wm_varname_list)):
                            wm_varname = wm_varname_list[i]
                            dim1 = trace_varnames.index(wm_varname)     # use trace_varnames because var order changed
                            msg = torch.diagonal(msg, dim1=dim1, dim2=-1)
                            trace_varnames.remove(wm_varname)

                        # Accomodate for mismatch in pt var dimension size
                        tar_dim = out_dim[out_varname_list.index(pt_varname)]
                        cur_dim = msg.shape[-1]
                        if tar_dim > cur_dim:
                            extra_shape = list(msg.shape)
                            extra_shape[-1] = tar_dim - cur_dim
                            msg = torch.cat((msg, torch.zeros(size=extra_shape)), dim=-1)

                    # If message goes outward, expand dimensions, put values on diagonals and put 0 everywhere else
                    else:
                        # Will permute the working dimension to the front for easy indexing
                        dim = trace_varnames.index(pt_varname)
                        perm = (dim,) + tuple(i for i in range(msg.dim()) if i is not dim)  # index permutation for msg
                        perm_msg = msg.permute(perm)        # working pt var dimension now put at front
                        wm_dim_sizes = tuple(out_dim[out_varname_list.index(wm_varname)]
                                             for wm_varname in wm_varname_list)
                        tar_shape = wm_dim_sizes + tuple(perm_msg.size())[1:]
                        buf = torch.zeros(tar_shape)        # zero tensor where diag val will be put on first two dim

                        # Account for mismatch in size
                        for i in range(min(wm_dim_sizes)):
                            buf[(i,) * len(wm_dim_sizes)] += perm_msg[i]
                        msg = buf

                        # bookkeeping stuff
                        trace_varnames.remove(pt_varname)
                        trace_varnames = wm_varname_list + trace_varnames

            # Final step: permute variable dimension to align with out_varname_list and check dim against out_dim
            assert all(var in out_varname_list for var in trace_varnames)
            assert all(var in trace_varnames for var in out_varname_list)

            perm = tuple(trace_varnames.index(var) for var in out_varname_list)
            msg = msg.permute(perm)
            assert list(msg.size()) == out_dim

            out_ld.set(msg, self._epsilon)


class ATFN(FactorNode):
    """
        Affine Transformation Factor Node
    """

    # TODO
    def __init__(self, name, affine):
        """
        :param name:    Name of the node
        :param affine:  Affine transformation specification
        """
        super(ATFN, self).__init__(name, function=None, func_var_list=None)
        self.affine = affine
        self.pretty_log["node type"] = "Affine Transformation Factor Node"


class BJFN(FactorNode):
    """
        Beta-Join Factor Node. Serve as structural juncture at Beta subnet in a conditional. No special computation
    """
    def __init__(self, name):
        super(BJFN, self).__init__(name, function=None, func_var_list=None)
        self.pretty_log["node type"] = "Beta-Join Factor Node"


class GFFN(FactorNode):
    """
        Gamma Function Factor Node
    """

    # TODO
    def __init__(self, name, function, func_var_list):
        super(GFFN, self).__init__(name, function=function, func_var_list=func_var_list)
        self.pretty_log["node type"] = "Gamma Function Factor Node"

    # Override check_quiesce() so that compute() will should always proceed. However this would not jeopardize
    # computation because message sent to ld is same as old, so ld.new would not be set to true under epsilon condition
    def check_quiesce(self):
        super(GFFN, self).check_quiesce()
        return False  # Always return False so that compute() will still proceed
