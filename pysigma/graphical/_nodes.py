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


"""
    Basic data structures and abstract node classes
        - LinkData
        - Node
        - FactorNode
        - VariableNode
"""


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
        """
            Compute method to be called to propagate message during decision phase.

            Note that super() must be called within compute() method in any child class, because all abstract node-level
                statistics logging is taken care of herein. 
        """
        # TODO: Other general logging regarding node computation statistics to be added here
        self.visited = True


class FactorNode(Node, ABC):
    """
        Factor node abstract base class.

        Guarantees that all incident nodes are variable nodes.
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

        Guarantees that all incident nodes are factor nodes
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


"""
    Utility structural nodes
        - DFN
        - DVN
"""


class DFN(FactorNode):
    """
        Default (Dummy) Factor Node.

        No special computation. Simply relay message to one or multiple variable nodes
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
        in_ld = self.in_linkdata[0]
        msg = in_ld.read()
        for out_ld in self.out_linkdata:
            out_ld.set(msg)
            
        super(DFN, self).compute()


class DVN(VariableNode):
    """
        Default (Dummy) Variable Node.

        No special computation. Simply relay message to one or multiple factor nodes
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
        in_ld = self.in_linkdata[0]
        msg = in_ld.read()
        for out_ld in self.out_linkdata:
            out_ld.set(msg)
            
        super(DVN, self).compute()


"""
    Nodes relating to Predicate subgraph structures
        - LTMFN
        - WMVN
        - PBFN
        - WMFN
"""


class LTMFN(FactorNode):
    """
        Long-Term Memory Factor Node.

        Holds the distribution class of this predicate, and if necessary can draw particle events from currently
            assumed distribution instance.

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

        Can be toggled regarding whether to draw particles in draw_particles().
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
        self.s_shape = torch.Size([num_particles])
        self.b_shape = batch_shape
        self.e_shape = event_shape
        self.p_shape = param_shape

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

        # Only admit one outgoing link and that must be WMVN. Check dimensions to be compatible with event message
        if not linkdata.to_fn:
            assert len(self.out_linkdata) == 0 and isinstance(linkdata.vn, WMVN)
            assert linkdata.msg_shape == self.s_shape + self.b_shape + self.e_shape
        # Can admit multiple incoming links. Check that link has special attribute declared.
        #   Check dimension for parameter link and event link respectively
        else:
            assert 'type' in linkdata.attr.keys(), "Incoming link to a LTMFN must specify 'type' special attribute"
            assert linkdata.attr['type'] in ['event', 'param'], "Incoming link to a LTMFN must have 'type' special " \
                                                                "attribute with value 'event' or 'param'"
            if linkdata.attr['type'] == 'event':
                assert len(list(ld for ld in self.in_linkdata if ld.attr['type'] == 'event')) == 0
                assert linkdata.msg_shape == self.s_shape + self.b_shape + self.e_shape
            else:
                assert linkdata.msg_shape == self.b_shape + self.p_shape
        
        super(LTMFN, self).add_link(linkdata)

    def toggle_draw(self, to_sample=False):
        # Turn on or off whether this LTMFN will sample particles during compute()
        self.to_sample = to_sample

    def retrive_particles(self):
        # Return the particles and sampling log density
        return self.particles, self.log_density

    def draw_particles(self):
        """
            Draw particles. This method should be called at the start of each decision cycle. Gather parameters from
                incoming 'param' type of linkdata, obtain new distribution instance, and update 'particles', 'weights',
                'log_density' attributes
        """
        if not self.to_sample:
            return

        # Obtain parameters from incoming 'param' link.
        param_lds = list(ld for ld in self.in_linkdata if ld.attr['type'] == 'param')
        assert len(param_lds) > 0
        param = torch.zeros(self.b_shape + self.p_shape)
        for param_ld in param_lds:
            tmp = param_ld.read()
            param += tmp

        # Obtain new distribution instance
        self.dist = DistributionServer.param2dist(self.dist_class, param, self.b_shape, self.e_shape)

        # Update attributes
        self.particles, self.weights, self.log_density = \
            DistributionServer.draw_particles(self.dist, self.num_particles, self.b_shape, self.e_shape)

    def compute(self):
        """
            Generate message from assumed distribution and send toward WMVN_OUT
        """
        # Generate message and send it to the WMVN
        # Particles message. Containing both distribution instance and particle list
        if self.to_sample:
            out_msg = Message(MessageType.Particles, self.s_shape, self.b_shape, self.e_shape, self.dist,
                              self.particles, self.weights, self.log_density)
        # Distribution message. Containing only distribution instance
        else:
            # Tabular message if distribution of Categorical class
            if issubclass(self.dist_class, Categorical):
                out_msg = Message(MessageType.Tabular, self.s_shape, self.b_shape, self.e_shape, self.dist)
            # Otherwise Distribution message
            else:
                out_msg = Message(MessageType.Distribution, self.s_shape, self.b_shape, self.e_shape, self.dist)
        out_ld = self.out_linkdata[0]
        out_ld.set(out_msg)
        
        super(LTMFN, self).compute()


class WMVN(VariableNode):
    """
        Working Memory Variable Node.

        Gate node connecting predicate structure to conditionals.

        Will attempt to combine incoming messages if there are multiple incoming links, subsuming the functionality of
            FAN node in Lisp Sigma. Combination can be carried out if incoming messages are all Tabular, all exponential
            Distribution, all homogeneous Particles (i.e. messages with same particle values), or a mixture of
            homogeneous Particles with any Distribution.
        Following combination procedure is carried out to conform to the standard of all inference methods

        1. If there are Particles incoming messages:
            a. Ensure that all particle lists are homogeneous
            b. If there are other Distributions/Tabular messages, draw a homogeneous particle list from the carried
                distribution with each particle's weight computed by
                        exp(log_pdf - sampling_log_pdf)
                up to a renormalization factor. In other words, the weight is the pdf of the particle evaluated at the
                distribution, inversely weighted by the sampling density of the particle from its original sampling
                distribution.
            c. Combine all particle list by taking the element-wise product of all associated weights as the new weight
                and renormalize.
        2. If messages are all Distributions:
            a. Ensure that they are all the same class of distribution, and that it is an exponential class distribution.
            b. Combine by taking the sum of the distribution's Natural parameters
        3. If messages are all Tabular:
            a. Combine by taking factor product.

        When combining messages, will exclude message from the link to which the combined message is to send to (if
            such a bidirected link exists). This implements the Sum-Product algorithm's variable node semantics, if
            this WMVN is served as both WMVN_IN and WMVN_OUT, i.e., if the predicate is of memory-less vector type.

        Optimization is implemented by caching the combination result for each outgoing link. If two outgoing links
            share the same set of incoming links that provide the messages, previously computed result will be reused
    """
    def __init__(self, name, index_var, rel_var_list, ran_var_list):
        super(WMVN, self).__init__(name, index_var, rel_var_list, ran_var_list)
        self.pretty_log["node type"] = "Working Memory Variable Node"

        # Cache for temporarily saving computation result for combination
        self.cache = {}

    def compute(self):
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
                in_lds = tuple(in_ld for in_ld in self.in_linkdata if in_ld.fn is not out_ld.fn)

                # Check if there's cached data. If yes, use cached result
                if in_lds in self.cache.keys():
                    out_msg = self.cache[in_lds]
                # Otherwise, compute conbimed message
                else:
                    in_msgs = tuple(in_ld.read() for in_ld in in_lds)
                    # 1. Find if there's any Particles message
                    if any(msg.type == MessageType.Particles for msg in in_msgs):
                        # 1.a. Ensure all particle lists are homogeneous
                        particle_msgs = tuple(msg for msg in in_msgs if msg.type == MessageType.Particles)
                        particle_lds = tuple(ld for ld in in_lds if ld.read().type == MessageType.Particles)
                        tmp_msg, tmp_ld = particle_msgs[0], particle_lds[0]
                        for msg, in_ld in zip(particle_msgs, particle_lds):
                            assert msg.particles == tmp_msg.particles, \
                                "At WMVN '{}': Incoming Particle message's particle values from  linkdata '{}' does " \
                                "not agree with that of incoming Particle message from linkdata '{}'"\
                                .format(self.name, in_ld, tmp_ld)
                            assert msg.log_density == tmp_msg.log_density, \
                                "At WMVN '{}': Incoming Particle message's sampling log density from linkdata '{}' " \
                                "does not agree with that of incoming Particle message  from linkdata '{}'" \
                                .format(self.name, in_ld, tmp_ld)

                        # 1.b Find any distribution message, if they exists, draw particle list
                        dist_msgs = tuple(msg for msg in in_msgs if msg.type in [MessageType.Distribution,
                                                                                 MessageType.Tabular])
                        particles = tmp_msg.particles
                        sampling_log_density = tmp_msg.log_density
                        weights_list = list(msg.weights for msg in particle_msgs)
                        for dist_msg in dist_msgs:
                            # 1.b. Get log-pdf of the particles from this distribution
                            log_density = DistributionServer.log_pdf(dist_msg.dist, particles)
                            # 1.b. Compute new weights by inversely weighting log pdf with sampling log pdf
                            weights = torch.exp(log_density - sampling_log_density)
                            # 1.b. first normalize. Sum is taken over the first dimension
                            weight_sum = weights.sum(dim=0, keepdim=True)
                            weights *= (1 / weight_sum)
                            # 1.b. Append this weights
                            weights_list.append(weights)

                        # 1.c. Take element-wise product of all weights
                        new_weights = torch.tensor(1)
                        for weights in weights_list:
                            new_weights *= weights
                        # 1.c. Normalize again
                        weight_sum = new_weights.sum(dim=0, keepdim=True)
                        new_weights *= (1 / weight_sum)

                        # Generate message
                        out_msg = Message(MessageType.Particles, self.s_shape, self.b_shape, self.e_shape, None,
                                          particles, new_weights, sampling_log_density)

                    # 2. Otherwise if all messages are Distribution
                    elif all(msg.type == MessageType.Distribution for msg in in_msgs):
                        tmp_msg, tmp_ld = in_msgs[0], in_lds[0]
                        dist_class = type(tmp_msg.dist)
                        # 2.a. Ensure all are the same exponential class of distribution
                        for msg, in_ld in zip(in_msgs, in_lds):
                            assert type(msg.dist) == dist_class, \
                                "At WMVN '{}': Incoming Distribution message's distribution class from linkdata '{}' " \
                                "is not the same as that from linkdata '{}'".format(self.name, in_ld, tmp_ld)
                            assert isinstance(msg.dist, torch.distributions.ExponentialFamily), \
                                "At WMVN '{}': All incoming messages are Distribution type, so expect them all of " \
                                "exponential class. However found distribution type '{}' in the message from linkdata " \
                                "'{}'".format(self.name, type(msg.dist), in_ld)
                        # 2.b. Combine by taking the sum of the distributions' natural parameters
                        sum_param = torch.tensor(0)
                        for msg in in_msgs:
                            sum_param += DistributionServer.exp_dist2natural(msg.dist)
                        # Generate message
                        new_dist = DistributionServer.natural2exp_dist(dist_class, sum_param, self.b_shape, self.e_shape)
                        out_msg = Message(MessageType.Distribution, self.s_shape, self.b_shape, self.e_shape, new_dist,
                                          None, None, None)

                    # 3. Otherwise, it can only be all Tabular messages
                    else:
                        assert all(msg.type == MessageType.Tabular for msg in in_msgs)
                        # 3.a. Combine by taking factor product
                        factors = list(DistributionServer.dist2param(msg.dist) for msg in in_msgs)
                        prod_factor = torch.tensor(1)
                        for factor in factors:
                            prod_factor *= factor
                        # Generate message. Note that no need to normalize factor because normalization will be taken
                        #   care by Categorical init
                        new_dist = DistributionServer.param2dist(Categorical, prod_factor, self.b_shape, self.e_shape)
                        out_msg = Message(MessageType.Tabular, self.s_shape, self.b_shape, self.e_shape, new_dist,
                                          None, None, None)

                # Cache result
                self.cache[in_lds] = out_msg
                # Send message
                out_ld.set(out_msg)

            # Clear cache
            self.cache = {}

        super(WMVN, self).compute()


class PBFN(FactorNode):
    """
        Perception Buffer Factor Node.

        Receive perception / observation / evidence as particle list and send to WMVN.
            Shape is assumed correct, so shape check as well as value check should be performed at the Cognitive level
            in the caller of set_perception()
        Currently do not support incoming link. Can only have one outgoing link connecting to a WMVN.
        Perception is buffered, and will be latched to next cycle if no new observation is specified.
        Overwrite check_quiesce() so that quiescence is determined by self.visited, i.e., whether compute() has been
            carried out
    """
    def __init__(self, name):
        super(PBFN, self).__init__(name)
        self.pretty_log["node type"] = "Perceptual Buffer Function Node"

        # Perception buffer
        self.buffer = None
        self.weights = None
        self.s_shape = None
        self.b_shape = None
        self.e_shape = None

    def set_perception(self, obs, weights, num_obs, b_shape, e_shape):
        """
            Update the perception buffer with new observation tensor. Should be called by the cognitive architecture
            Need to specify weights corresponding to the observation particles. Sampling log density on the other hand
                will be set to 0 in the outgoing Particle message.

            :param obs:     Observation tensor. torch.Tensor. shape ([num_obs] + b_shape + e_shape)
            :param weights: Weight tensor. torch.Tensor. shape ([num_obs] + b_shape)
            :param num_obs: torch.Size
            :param b_shape: torch.Size
            :param e_shape: torch.Size
        """
        assert isinstance(obs, torch.Tensor)
        assert isinstance(weights, torch.Tensor)
        assert isinstance(num_obs, int)
        assert isinstance(b_shape, torch.Size)
        assert isinstance(e_shape, torch.Size)
        assert obs.shape == torch.Size([num_obs]) + b_shape + e_shape
        assert weights.shape == torch.Size([num_obs]) + b_shape

        self.buffer = obs
        self.weights = weights
        self.s_shape = torch.Size([num_obs])
        self.b_shape = b_shape
        self.e_shape = e_shape

    def add_link(self, linkdata):
        # Ensure that not incoming link and only one outgoing link connecting to a WMVN
        assert isinstance(linkdata, LinkData)
        assert not linkdata.to_fn
        assert len(self.out_linkdata) == 0
        assert isinstance(linkdata.vn, WMVN)
        super(PBFN, self).add_link(linkdata)

    def compute(self):
        # If no perception has been set in the buffer, then do not send
        if self.buffer is None:
            return

        # Otherwise, send either way. Sampling log density set to uniform 0
        assert len(self.out_linkdata) > 0
        out_ld = self.out_linkdata[0]
        out_msg = Message(MessageType.Particles, self.s_shape, self.buffer, self.e_shape, None, self.buffer,
                          self.weights, 0)
        out_ld.set(out_msg)
        
        super(PBFN, self).compute()

    # Override check_quiesce() so that quiescence is equivalent to visited
    def check_quiesce(self):
        self.quiescence = self.visited
        return self.quiescence


class WMFN(FactorNode):
    """
        Working Memory Factor Node.

        Effectively a buffer node that contains a memory buffer, whose content will be sent
            as outgoing message only until the next decision cycle. During modification phase, the memory buffer content
            can either be replaced entirely by, or taken a weighted sum with incoming message.

        WMFN can buffer either parameters memory or event memory. However, each of them entails different handling.
            Therefore, one must specify a type to be either 'param' or 'event' during initialization.

        Can specify a decay rate. The new memory is then derived via
                old_content * (1 - decay_rate) + new_content
            therefore if decay_rate is 1 (default value), the old content will be completely forgotten.

        The weighted sum procedure for different WMFN types and message types:
            - For 'param' type of WMFN:
                - Assume incoming message is Particle type, presenting parameters as particles in shape
                    (b_shape + e_shape)
                - Perform weighted sum on the particle values
            - For 'event' type of WMFN:
                - If message is Particle type, perform the weighted sum on the particle weights
                - If message is Distribution or Tabular, perform the weighted sum on the distribution parameters

        Note that the incoming message will always be cloned before performing weighted sum update. This is to prevent
            any parts of the memory message from in-place change by some parts elsewhere in the graph.

        Can admit only one incoming and one outgoing links.
    """
    def __init__(self, name, content_type, decay_rate=1):
        """
            :param content_type: one of 'param' or 'event'
            :param decay_rate:   The decay rate at which the old memory vanished. Within range [0, 1]
                                 Default to 1, i.e., entirely forgetting old memory and replace with new content
        """
        super(WMFN, self).__init__(name)
        self.pretty_log["node type"] = "Working Memory Function Node"

        assert content_type in ['param', 'event']
        assert isinstance(decay_rate, (float, int)) and 0 <= decay_rate <= 1

        self.content_type = content_type
        self.decay_rate = decay_rate
        # memory buffer.
        self.memory = None

    # Override so that only allow one incoming link and one outgoing link
    def add_link(self, linkdata):
        if linkdata.to_fn:
            assert len(self.in_linkdata) == 0
        else:
            assert len(self.out_linkdata) == 0

        super(WMFN, self).add_link(linkdata)

    def update_memory(self):
        """
            Update the content in memory using message from incoming link.
            Should only be called during modification phase
        """
        assert len(self.in_linkdata) > 0
        in_ld = self.in_linkdata[0]
        # Clone incoming message
        msg = in_ld.read().clone()
        assert self.memory is None or (msg.type == self.memory.type and
                                       msg.s_shape == self.memory.s_shape and
                                       msg.b_shape == self.memory.b_shape and
                                       msg.e_shape == self.memory.e_shape)

        # If memory is None or decay_rate is 1, directly replace memory buffer content
        if self.memory is None or self.decay_rate == 1:
            self.memory = msg

        # Otherwise, perform weighted sum update
        else:
            # For param type of content, perform update on particle values
            if self.content_type == 'param':
                # Ensure message is Particle type
                assert msg.type is MessageType.Particles
                new_val = (1 - self.decay_rate) * self.memory.particles + msg.particles
                self.memory = Message(MessageType.Particles, msg.s_shape, msg.b_shape, msg.e_shape, None, new_val,
                                      msg.weights, msg.log_density)

            # For event type of content
            else:
                # If message is Particle type, perform update on particle weights, and renoramlize
                if msg.type is MessageType.Particles:
                    # Ensure that incoming message's particle values are the same as that of the memory message
                    assert msg.particles == self.memory.particles
                    new_weights = (1 - self.decay_rate) * self.memory.weights + msg.weights
                    # Normalize weights
                    weight_sum = new_weights.sum(dim=0, keepdim=True)
                    new_weights *= (1 / weight_sum)
                    self.memory = Message(MessageType.Particles, msg.s_shape, msg.b_shape, msg.e_shape, None,
                                          msg.particles, new_weights, msg.log_density)

                # Otherwise, perform update directly on the distribution's parameters
                else:
                    old_param = DistributionServer.dist2param(self.memory.dist)
                    in_param = DistributionServer.dist2param(msg.dist)
                    new_param = (1 - self.decay_rate) * old_param + in_param
                    new_dist = DistributionServer.param2dist(type(msg.dist), new_param, msg.b_shape, msg.e_shape)
                    self.memory = Message(msg.type, msg.s_shape, msg.b_shape, msg.e_shape, new_dist, None, None, None)

    def compute(self):
        """
            Sends memory content toward outgoing link (if memory is not None)
        """
        assert len(self.out_linkdata) > 0
        if self.memory is not None:
            self.out_linkdata[0].set(self.memory)
            
        super(WMFN, self).compute()

    # Overrides so that quiescence for WMFN is equivalent to visited
    def check_quiesce(self):
        self.quiescence = self.visited
        return self.quiescence


"""
    Nodes relating to Conditional subgraph structures
        - Alpha subgraph: AlphaFactorNode, RelMapNode, ExpSumNode, RanTransNode
        - Gamma factor node: GFN
"""


class AlphaFactorNode(FactorNode, ABC):
    """
        Abstract base class for any factor node belonging to a alpha subgraph

        The commonality of all alpha subgraph factor nodes is that they all only admit up to two paris of incoming and
            outgoing link. Additionally, links must declare a special attribute 'direction' with value 'inward' or
            'outward' to indicate whether it is pointing toward the conditional gamma factor node or not.

        Such link check is implemented in add_link() to be inherited by concrete alpha factor node class

        Also implemented is pre-computation check of link presence in compute()
    """
    def add_link(self, linkdata):
        assert isinstance(linkdata, LinkData)
        assert 'direction' in linkdata.attr and linkdata.attr['direction'] in ['inward', 'outward']

        if linkdata.to_fn:
            assert len(self.in_linkdata) == 0 or linkdata.attr['direction'] != self.in_linkdata[0].attr['direction']
            assert len(self.in_linkdata) <= 1
        else:
            assert len(self.out_linkdata) == 0 or linkdata.attr['direction'] != self.out_linkdata[0].attr['direction']
            assert len(self.out_linkdata) <= 1
            
        super(AlphaFactorNode, self).add_link(linkdata)

    def compute(self):
        assert len(self.in_linkdata) == len(self.out_linkdata) and len(self.in_linkdata) > 0
        super(AlphaFactorNode, self).compute()


class RelMapNode(AlphaFactorNode):
    """
        Relation Variable Mapping Node

        Convert between predicate arguments and pattern variables. Apply relational variable's VariableMap (if declared)
            by selecting and placing entries among the message batch dimensions. This node can thus carry out
            inner-pattern relational variable matching by itself.

        This node is a component of the alpha conditionial subgraph, so admits up to two pairs of incoming and outgoing
            links. Link must declare special attribute 'direction' with value 'inward' or 'outward' to indicate whether
            it is pointing toward the conditional gamma factor node or not.

        For inward direction, inner-pattern relational variable matching is handled by selecting entries on the
            diagonals from the incoming message. For outward direction, this is handled by placing incoming message onto
            the diagonals of a larger message tensor.
    """
    def __init__(self, name, arg2var, var2arg):
        super(RelMapNode, self).__init__(name)
        self.pretty_log["node type"] = "Relation Variable Mapping Node"
        
        assert isinstance(name, str)
        assert isinstance(arg2var, dict)
        assert isinstance(var2arg, dict)

        self.arg2var = arg2var
        self.var2arg = var2arg


class ExpSumNode(AlphaFactorNode):
    """
        Expansion / Summarization Node

        This node is a component of the alpha conditionial subgraph, so admits up to two pairs of incoming and outgoing
            links. Link must declare special attribute 'direction' with value 'inward' or 'outward' to indicate whether
            it is pointing toward the conditional gamma factor node or not.

        For inward direction, it expands and permutes the incoming message's relational variable dimensions to match the
            full relational variable dimensions determined by the conditional. For outward direction, it summarizes
            over irrelevant relational variables and permute the dimensions to match the relational variable dimensions
            of this pattern.

        Note that the expanded dimensions will be of size 1, so that the expanded tensor is broadcastable along this
            dimension.

        The summarization step can be thought of as a search or optimization problem, where one finds a single
            distribution instance that best "summarizes" the behaviors of an entire space of distribution instances,
            where the dimensions of the space is defined and spanned by the irrelevant relational variables. Depending
            on the user-specified summarization criteria, different semantics can be interpreted for this step.
    """
    pass


class RanTransNode(AlphaFactorNode):
    """
        Random Variable Transformation Node

        Apply a user-specified transformation procedure defined in torch.distributions.transforms on the random
            variables of the incoming message. In other words, this nodes transforms the values of the particle events
            of the message distributions.

        By transforming the RV values, it also induces new value constraints for the generated messages. For the inward
            direction, the new induced value constraints will be used by the gamma factor node to carry out value check
            for the matching RVs.

        Note that if this node belongs to a Condact predicate pattern, the transformation specified by user MUST be
            bijective. The forward transformation will be used for inward message propagation, whereas the backward
            transformation will be used for outward message propagation.

        This node is a component of the alpha conditionial subgraph, so admits up to two pairs of incoming and outgoing
            links. Link must declare special attribute 'direction' with value 'inward' or 'outward' to indicate whether
            it is pointing toward the conditional gamma factor node or not.
    """
    pass


class GFN(AlphaFactorNode):
    """
        Gamma Factor Node

        Carry out general-inference message computation at the PGM factor node.

        Induce a message computation task for each of the outgoing link.
    """
    pass






















