"""
    All kinds of nodes in the graphical architecture
"""
import copy
import torch
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from torch.distributions import Distribution
from torch.distributions.categorical import Categorical
from torch.nn.functional import cosine_similarity
from defs import Variable, MessageType, Message
from utils import DistributionServer
from structures import VariableMap, Summarization


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
                            For Parameter message, should be (batch_shape + param_shape)
                            For particles message, should be (sample_shape + batch_shape + event_shape)
        :param epsilon:     epsilon upper bound for checking message difference using KL divergence
        """
        assert isinstance(vn, VariableNode)
        assert isinstance(fn, FactorNode)
        assert isinstance(msg_shape, tuple) and all(isinstance(s, torch.Size) for s in msg_shape)
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

    def reset_shape(self, msg_shape):
        """
            Reset shape for the Message
            CAUTION: will clear memory buffer and set self.new to False

            :param msg_shape:       A tuple of torch.Size. Represents all four message shapes.
        """
        assert isinstance(msg_shape, tuple) and all(isinstance(s, torch.Size) for s in msg_shape)
        self.msg_shape = msg_shape
        self.memory = None
        self.new = False

    def write(self, new_msg, check_diff=True, clone=False):
        """
            Set the link message memory to the new message arriving at this link.

            If check_diff is True, then will check if the new message is different from the existing one before
                replacing the existing with the new one.

            Messages will be deemed difference in the following cases:
                - if they are of different types,
                - if they are both Parameter type and the batch average L2 distance between the two parameter tensors
                    is larger than epsilon,
                - if they are both Particles type and they possess either different particle values or different
                    sampling log densities,
                - if they are both Particles type, and they possess the same particles values and same sampling log
                    densities, but the batch average cosine similarity distance between the two particle weight tensors
                    is larger than epsilon.

            If want to set a new message of a different message type than the current memory, make sure reset_shape()
                is first called so that shape check works for the new message.

            If clone is True, then will store a cloned new_msg
        """
        assert isinstance(new_msg, Message)
        # Check new message shape
        assert self.msg_shape == new_msg.size()
        assert self.memory is None or self.memory.same_size_as(new_msg)

        # Will replace the memory immediately if any one of the following conditions is met:
        #   - self.memory is None
        #   - check_diff is False
        #   - new message has different type
        #   - message type is Particles and new message has different particle values and/or sampling log densities
        if self.memory is None or check_diff is False or new_msg.type != self.memory.type or \
                (new_msg.type == MessageType.Particles and not (torch.equal(self.memory.particles, new_msg.particles) and
                                                                torch.equal(self.memory.log_density, new_msg.log_density))):
            self.memory = new_msg.clone() if clone is True else new_msg
            self.new = True
            return

        # Otherwise, check difference by KL-divergence
        if self.memory.type == MessageType.Parameter:
            # For Parameter message, compare batch average L2 distance
            # Parameter tensor has shape (batch_shape + param_shape), with param_shape of length 1
            # L2 distance is computed along param_shape dimension, i.e., the -1 dimension
            diff = new_msg.parameters - self.memory.parameters
            val = diff.norm(dim=-1).mean()
        else:
            # For Particles message, compare batch average cosine similarity distance
            # Particle weights has shape (sample_shape + batch_shape), with sample_shape of length 1
            # cosine similarity distance is computed along sample_shape dimension. i.e., the 0 dimension
            val = cosine_similarity(new_msg.weights, self.memory.weights, dim=0).mean()

        # Compare distance value with epsilon
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
        return self.quiescence

    @abstractmethod
    def add_link(self, linkdata):
        pass

    @abstractmethod
    def compute(self):
        """
            Compute method to be called to propagate message during decision phase.

            Note that super() must be called within compute() method in any child class, because all abstract node-level
                statistics logging is taken care of herein.

            The default quiescence behavior for compute() is to return directly if check_quiesce() returns True, without
                logging anything or carrying out further computation. Note that such behavior may or may not be desired
                by child node class.
        """
        # Return directly if quiesced
        if self.check_quiesce():
            return
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

    def __init__(self, name, index_var, rel_var_list, param_var=None, ran_var_list=None):
        """
            Decalre a VariableNode
        :param name:        name of the variable node
        :param index_var:      Particle indexing variable
        :param rel_var_list:   list of Variables representing the relational variables of this variable nodes
        :param ran_var_list:   Optional. list of Variables representing the random variables of this variable nodes
        :param param_var:      Optional. The parameter variable.
        """
        super(VariableNode, self).__init__(name)
        assert isinstance(index_var, Variable)
        assert isinstance(rel_var_list, Iterable) and all(isinstance(v, Variable) for v in rel_var_list)
        assert ran_var_list is not None or (isinstance(ran_var_list, Iterable) and
                                            all(isinstance(v, Variable) for v in ran_var_list))
        assert param_var is not None or isinstance(param_var, Variable)

        self.index_var = index_var
        self.rel_var_list = rel_var_list
        self.ran_var_list = ran_var_list
        self.s_shape = torch.Size([index_var.size])
        self.b_shape = torch.Size(list(rel_var.size for rel_var in rel_var_list))
        self.p_shape = torch.Size([param_var.size]) if param_var is not None else None
        self.e_shape = torch.Size([sum(list(ran_var.size for ran_var in ran_var_list))]) if ran_var_list is not None \
            else None

    def add_link(self, linkdata):
        """
            Register the LinkData connecting a factor node to this variable node.
            Check that the variable list specified in linkdata agrees with that pre-specified at this variable node
        """
        assert isinstance(linkdata, LinkData)
        assert linkdata.vn is self
        assert linkdata.msg_shape == (self.s_shape, self.b_shape, self.p_shape, self.e_shape)
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
        super(DFN, self).compute()
        in_ld = self.in_linkdata[0]
        msg = in_ld.read()
        for out_ld in self.out_linkdata:
            out_ld.write(msg)


class DVN(VariableNode):
    """
        Default (Dummy) Variable Node.

        No special computation. Simply relay message to one or multiple factor nodes
        Only admit one incoming link but can connect with multiple outgoing links
    """
    def __init__(self, name, index_var, rel_var_list, param_var=None, ran_var_list=None):
        super(DVN, self).__init__(name, index_var, rel_var_list, param_var, ran_var_list)
        self.pretty_log["node type"] = "Default Variable Node"

    def add_link(self, linkdata):
        assert isinstance(linkdata, LinkData)
        # Make sure no more than on incoming and one outgoing link
        assert linkdata.to_fn or len(self.in_linkdata) == 0

        super(DVN, self).add_link(linkdata)

    def compute(self):
        super(DVN, self).compute()
        in_ld = self.in_linkdata[0]
        msg = in_ld.read()
        for out_ld in self.out_linkdata:
            out_ld.write(msg)


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

        Can toggle modes between drawing particles and not drawing by toggle_draw():
            - If drawing mode is turned on, then LTMFN will send Particles message to WMVN_OUT during compute(). The
                particles message to be send is retrieved from memory buffer, which is set by calling draw_particles()
                during the modification phase.
            - Otherwise, LTMFN will send Parameter message, containing the parameter from the parameter feed, directly
                to WMVN_OUT. In this mode, calling draw_particles() will have no effect.
    """
    def __init__(self, name, distribution_class, num_particles, batch_shape, event_shape, param_shape):
        super(LTMFN, self).__init__(name)
        self.pretty_log["node type"] = "Long-Term Memory Factor Node"

        assert issubclass(distribution_class, Distribution)
        assert isinstance(num_particles, int) and num_particles >= 1
        assert isinstance(batch_shape, torch.Size)
        assert isinstance(event_shape, torch.Size)
        assert isinstance(param_shape, torch.Size)

        self.dist_class = distribution_class
        self.num_particles = num_particles
        self.s_shape = torch.Size([self.num_particles])
        self.b_shape = batch_shape
        self.e_shape = event_shape
        self.p_shape = param_shape

        # Flag that indicates whether to sample particles during compute()
        self.to_sample = False

        # Knowledge
        self.param = None           # Parameter tensor to the distribution instance
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
        # Can admit multiple incoming links. Check that link has special attribute declared.
        #   Check dimension for parameter link and event link respectively
        else:
            assert 'type' in linkdata.attr.keys(), "Incoming link to a LTMFN must specify 'type' special attribute"
            assert linkdata.attr['type'] in ['event', 'param'], "Incoming link to a LTMFN must have 'type' special " \
                                                                "attribute with value 'event' or 'param'"
            if linkdata.attr['type'] == 'event':
                assert len(list(ld for ld in self.in_linkdata if ld.attr['type'] == 'event')) == 0
        
        super(LTMFN, self).add_link(linkdata)

    def toggle_draw(self, to_sample=False):
        # Turn on or off whether this LTMFN will sample particles during compute()
        self.to_sample = to_sample
        # Also reset shape of the outgoing linkdata
        assert len(self.out_linkdata) > 0
        if self.to_sample:
            self.out_linkdata[0].reset_shape(self.s_shape + self.b_shape + self.e_shape)
        else:
            self.out_linkdata[0].reset_shape(self.b_shape + self.p_shape)

    def retrive_particles(self):
        # Return the particles and sampling log density
        return self.particles, self.log_density

    def draw_particles(self):
        """
            Draw particles. This method should be called during the modification phase, if necessary. Parameters will
                be gathered from incoming "param" type of linkdata, obtain new distribution instance, and update
                'particles', 'weights', 'log_density' attributes
        """
        # Return directly if drawing mode is off
        if not self.to_sample:
            return

        # Obtain parameters from incoming 'param' link.
        param_lds = list(ld for ld in self.in_linkdata if ld.attr['type'] == 'param')
        assert len(param_lds) > 0

        param_msg = param_lds[0].read()
        assert param_msg.type == MessageType.Parameter
        for other_ld in param_lds[1:]:
            param_msg += other_ld.read()

        # Set param buffer
        self.param = param_msg.parameters
        # Obtain new distribution instance
        self.dist = DistributionServer.param2dist(self.dist_class, self.param, self.b_shape, self.e_shape)
        # Update particles buffer
        self.particles, self.weights, self.log_density = \
            DistributionServer.draw_particles(self.dist, self.num_particles, self.b_shape, self.e_shape)

    def compute(self):
        """
            Generate message from assumed distribution and send toward WMVN_OUT
        """
        super(LTMFN, self).compute()
        assert len(self.out_linkdata) > 0
        out_ld = self.out_linkdata[0]

        # If drawing particles mode is on, send Particles message
        if self.to_sample:
            out_msg = Message(MessageType.Particles,
                              sample_shape=self.s_shape, batch_shape=self.b_shape, event_shape=self.e_shape,
                              particles=self.particles, weights=self.weights, log_density=self.log_density)
        # Otherwise, send Parameter message
        else:
            out_msg = Message(MessageType.Parameter,
                              batch_shape=self.b_shape, param_shape=self.p_shape, parameters=self.param)

        out_ld.write(out_msg)


class WMVN(VariableNode):
    """
        Working Memory Variable Node.

        Gate node connecting predicate structure to conditionals.

        Will attempt to combine incoming messages if there are multiple incoming links, subsuming the functionality of
            FAN node in Lisp Sigma. Combination can be carried out if messages are all Parameter type, or if there
            exist Particles type messages but all of them are homogeneous (sharing the same particle values as well as
            sampling log densities).

        Following combination procedure is carried out to conform to the standard of all inference methods

        1. If there are Particles type incoming messages:
            a. Ensure that all particle lists are homogeneous
            b. If there are other Parameter messages, convert the parameter into distribution instance in the
                pre-specified distribution class, query the log pdf of the homogeneous particle lists w.r.t. this
                distribution instance, and compute the weights by
                        exp(log_pdf - sampling_log_pdf)
                up to a renormalization factor. In other words, the weight is the pdf of the particle evaluated at the
                distribution, inversely weighted by the sampling density of the particle from its ORIGINAL sampling
                distribution.
            c. Combine all particle list by taking the element-wise product of all associated weights as the new weight
                and re-normalize, i.e., addition operation defined for the Particles messages.
        2. If messages are all Parameter type:
            a. Sum over the parameters using the addition operation defined for the Parameter messages.

        Since occasionally querying log pdf is to be performed, WMVN thus needs to know the distribution class the
            predicate it belongs to is assuming.

        When combining messages, will exclude message from the link to which the combined message is to send to (if
            such a bidirected link exists). This implements the Sum-Product algorithm's variable node semantics, if
            this WMVN is served as both WMVN_IN and WMVN_OUT, i.e., if the predicate is of memory-less vector type.

        Optimization is implemented by caching the combination result for each outgoing link. If two outgoing links
            share the same set of incoming links that provide the messages, previously computed result will be reused
    """
    def __init__(self, name, dist_class, index_var, rel_var_list, param_var=None, ran_var_list=None):
        assert issubclass(dist_class, Distribution)
        super(WMVN, self).__init__(name, index_var, rel_var_list, param_var, ran_var_list)
        self.pretty_log["node type"] = "Working Memory Variable Node"

        # Distribution class the Predicate self belongs to is assuming
        self.dist_class = dist_class
        # Cache for temporarily saving computation result for combination
        self.cache = {}

    def compute(self):
        super(WMVN, self).compute()
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
                    out_ld.write(msg)
        # Otherwise, combine messages
        else:
            for out_ld in self.out_linkdata:
                # The tuple of all incoming linkdata that are not connected to the factor node the selected outgoing
                #   linkdata is connected to.
                # Use tuple here because tuple is hashable and we will use it as keys to cache dictionary
                in_lds = tuple(in_ld for in_ld in self.in_linkdata if in_ld.fn is not out_ld.fn)

                # Check if there's cached data. If yes, use cached result
                if in_lds in self.cache.keys():
                    out_msg = self.cache[in_lds]
                # Otherwise, compute combined message
                else:
                    in_msgs = tuple(in_ld.read() for in_ld in in_lds)
                    # 1. Find if there's any Particles message
                    if any(msg.type == MessageType.Particles for msg in in_msgs):
                        # 1.a. Ensure all particle lists are homogeneous
                        particle_msgs = tuple(msg for msg in in_msgs if msg.type == MessageType.Particles)
                        particle_lds = tuple(ld for ld in in_lds if ld.read().type == MessageType.Particles)
                        tmp_msg, tmp_ld = particle_msgs[0], particle_lds[0]
                        for msg, in_ld in zip(particle_msgs, particle_lds):
                            assert torch.equal(tmp_msg.particles, msg.particles), \
                                "At WMVN '{}': When attempting to combine incoming messages, found that incoming " \
                                "Particle message's particle values from  linkdata '{}' does not agree with that of " \
                                "incoming Particle message from linkdata '{}'"\
                                .format(self.name, in_ld, tmp_ld)
                            assert msg.log_density == tmp_msg.log_density, \
                                "At WMVN '{}': When attempting to combine incoming messages, found that incoming " \
                                "Particle message's sampling log density from linkdata '{}' does not agree with that of " \
                                "incoming Particle message  from linkdata '{}'" \
                                .format(self.name, in_ld, tmp_ld)

                        # 1.b Find any Parameter type message, if they exist, draw particle list
                        param_msgs = tuple(msg for msg in in_msgs if msg.type is MessageType.Parameter)
                        particles = tmp_msg.particles
                        sampling_log_density = tmp_msg.log_density
                        s_shape = tmp_msg.s_shape
                        b_shape = tmp_msg.b_shape
                        e_shape = tmp_msg.e_shape

                        # 2.b Compute particle weights w.r.t. distributions induced by the Parameter type messages
                        candidate_msgs = list(particle_msgs)
                        for param_msg in param_msgs:
                            # Obtain distribution instance
                            dist = DistributionServer.param2dist(self.dist_class, param_msg.parameters, b_shape, e_shape)
                            # Query log pdf
                            log_density = DistributionServer.log_pdf(dist, particles)
                            # Compute new weights by inversely weighting log pdf with sampling log pdf
                            weights = torch.exp(log_density - sampling_log_density)
                            # Convert to Particles message and append to candidate msg list. Normalization of weights
                            #   is taken care of during Message initialization
                            msg = Message(MessageType.Particles,
                                          sample_shape=s_shape, batch_shape=b_shape, event_shape=e_shape,
                                          particles=particles, weights=weights, log_density=log_density)
                            candidate_msgs.append(msg)

                        # Combine messages
                        out_msg = candidate_msgs[0]
                        for msg in candidate_msgs[1:]:
                            out_msg += msg

                    # 2. Otherwise all messages are Parameter type
                    else:
                        out_msg = in_msgs[0]
                        for msg in in_msgs[1:]:
                            out_msg += msg

                # Cache result
                self.cache[in_lds] = out_msg
                # Send message
                out_ld.write(out_msg)

            # Clear cache
            self.cache = {}


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

            :param obs:     Observations. torch.Tensor. shape ([num_obs] + b_shape + e_shape)
            :param weights: Weights. torch.Tensor or int. shape ([num_obs] + b_shape) if tensor, otherwise an int of 1
                                to indicate uniform weights
            :param num_obs: torch.Size
            :param b_shape: torch.Size
            :param e_shape: torch.Size
        """
        assert isinstance(obs, torch.Tensor)
        assert isinstance(weights, (torch.Tensor, int))
        assert isinstance(num_obs, int)
        assert isinstance(b_shape, torch.Size)
        assert isinstance(e_shape, torch.Size)
        assert obs.shape == torch.Size([num_obs]) + b_shape + e_shape
        assert (isinstance(weights, torch.Tensor) and weights.shape == torch.Size([num_obs]) + b_shape) or \
               (isinstance(weights, int) and weights == 1)

        self.buffer = obs
        self.weights = weights
        self.s_shape = torch.Size([num_obs])
        self.b_shape = b_shape
        self.e_shape = e_shape

    def add_link(self, linkdata):
        # Ensure that no incoming link and only one outgoing link connecting to a WMVN
        assert isinstance(linkdata, LinkData)
        assert not linkdata.to_fn
        assert len(self.out_linkdata) == 0
        assert isinstance(linkdata.vn, WMVN)
        super(PBFN, self).add_link(linkdata)

    def compute(self):
        super(PBFN, self).compute()
        # If no perception has been set in the buffer, then do not send
        if self.buffer is None:
            return

        # Otherwise, send either way. Sampling log density set to uniform 0
        assert len(self.out_linkdata) > 0
        out_ld = self.out_linkdata[0]
        out_msg = Message(MessageType.Particles,
                          sample_shape=self.s_shape, batch_shape=self.b_shape, event_shape=self.e_shape,
                          particles=self.buffer, weights=self.weights, log_density=0)
        out_ld.write(out_msg)

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

        The weighted sum procedure is viewed as a linear combination process in the message space. The linear operator
            (addition, scalar multiplication) definitions are different depending on the content type. Please see
            Message class documentation for more info.

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
        new_msg = in_ld.read().clone()
        assert self.memory is None or (isinstance(self.memory, Message) and new_msg.type == self.memory.type and
                                       new_msg.size() == self.memory.size())

        # If memory is None or decay_rate is 1, directly replace memory buffer content
        if self.memory is None or self.decay_rate == 1:
            self.memory = new_msg
        # Otherwise, perform weighted sum update
        else:
            self.memory = new_msg + self.memory * (1 - self.decay_rate)

    def compute(self):
        """
            Sends memory content toward outgoing link (if memory is not None)
        """
        super(WMFN, self).compute()
        assert len(self.out_linkdata) > 0
        if self.memory is not None:
            self.out_linkdata[0].write(self.memory)

    # Overrides so that quiescence for WMFN is equivalent to visited
    def check_quiesce(self):
        self.quiescence = self.visited
        return self.quiescence


"""
    Nodes relating to Conditional Alpha subgraph structures
        - AlphaFactorNode, RelMapNode, ExpSumNode, RanTransNode
"""


class AlphaFactorNode(FactorNode, ABC):
    """
        Abstract base class for any factor node belonging to a alpha subgraph

        The commonality of all alpha subgraph factor nodes is that they all only admit up to two paris of incoming and
            outgoing link. Additionally, links must declare a special attribute 'direction' with value 'inward' or
            'outward' to indicate whether it is pointing toward the conditional gamma factor node or not.

        Such link check is implemented in add_link() to be inherited by concrete alpha factor node class. Also
            implemented in this method is the registration of labeled pairs of linkdata in self.labeled_ld_pair

        compute() is implemented so that it execute inward_compute() and/or outward_compute() based on the presence of
            linkdata pairs.

        inward_compute() and outward_compute() are now abstract methods that must be implemeted by child classes, but
            compute() should not be override.
    """
    def __init__(self, name):
        super(AlphaFactorNode, self).__init__(name)

        # Pairs of incoming and outgoing linkdata labeled with their directionality w.r.t. the alpha structure
        self.labeled_ld_pair = {}

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

        # If the other ld of this ld pair has not been added, then temporarily register this ld instance directly
        direction = linkdata.attr['direction']
        if direction not in self.labeled_ld_pair.keys():
            self.labeled_ld_pair[direction] = linkdata
        # Otherwise, take out the other ld of this ld pair from the dict and replace entry with a tuple
        #   Make sure that incoming ld is the first element of the tuple and outgoing ld the second element
        else:
            other_ld = self.labeled_ld_pair[direction]
            self.labeled_ld_pair[direction] = (linkdata, other_ld) if linkdata.to_fn else (other_ld, linkdata)

    def compute(self):
        super(AlphaFactorNode, self).compute()
        assert len(self.in_linkdata) == len(self.out_linkdata) and len(self.in_linkdata) > 0

        # Carry out directional computation based on presence of link in self.labeled_ld_pair
        for direction in self.labeled_ld_pair.keys():
            if direction == 'inward':
                self.inward_compute()
            else:
                self.outward_compute()

    @abstractmethod
    def inward_compute(self):
        """
            Inward message computation. To be implemented by child class.
        """
        pass

    @abstractmethod
    def outward_compute(self):
        """
            Outward message computation. To be implemented by child class.
        """
        pass


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
    def __init__(self, name, arg2var, var2arg, arg2var_map):
        """
            Necessary data structure:

            :param  arg2var:    dictionary mapping predicate argument Variable instance to pattern variable Variable
                                    instance
            :param  var2arg:    dictionary mapping pattern variable Variable instance to LIST of predicate argument
                                    Variable instance(s)
            :param  arg2var_map:    dictionary mapping predicate argument Variable instance to VariableMap instance
        """
        super(RelMapNode, self).__init__(name)
        self.pretty_log["node type"] = "Relation Variable Mapping Node"
        
        assert isinstance(name, str)
        assert isinstance(arg2var, dict) and all(isinstance(k, Variable) for k in arg2var.keys()) and \
               all(isinstance(v, Variable) for v in arg2var.values())
        assert isinstance(var2arg, dict) and all(isinstance(k, Variable) for k in var2arg.keys()) and \
               all(isinstance(v, list) and all(isinstance(arg, Variable) for arg in v) for v in var2arg.values())
        assert isinstance(arg2var_map, dict) and all(isinstance(k, Variable) for k in arg2var_map.keys()) and \
               all(isinstance(v, VariableMap) for v in arg2var_map.values())

        self.arg2var = arg2var
        self.var2arg = var2arg
        self.arg2var_map = arg2var_map

        # Obtain mapping dictionary and inverse mapping dictionary
        self.arg2var_map_tuple = {arg: var_map.get_map() for arg, var_map in self.arg2var_map.items()}
        self.arg2var_map_inv_tuple = {arg: var_map.get_inverse_map() for arg, var_map in self.arg2var_map.items()}

    def inward_compute(self):
        """
            Inward computation. Convert predicate relational arguments to pattern relational variables. Apply mappings
                to relational variable's values, if specified.

            For inward direction, we are assuming this is used in condition or condact patterns. Accordingly, the
                inverse mapping should be used to map predicate arguments to pattern variables.

            Will check anyway if domain and image of the inverse map agree with the size range of the predicate argument
                and pattern variable respectively. However to be user friendly this should be checked beforehand by
                compiler.

            Note that domain should be a subset of predicate argument size range, but image should be exactly equal to
                the pattern variable size range

            The computations to be carried out can be summarized in three steps: map/broaden, diagonalize, & permute
        """
        in_ld, out_ld = self.labeled_ld_pair['inward']
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        msg = in_ld.read()
        assert isinstance(msg, Message)
        in_rel_var_list, out_rel_var_list = in_ld.vn.rel_var_list, out_ld.vn.rel_var_list

        # Check that given data structures agree with variable lists of the incident variable node
        assert set(self.arg2var.keys()) == set(in_rel_var_list)
        assert set(self.var2arg.keys()) == set(out_rel_var_list)
        # Check that mapping's domain and image agree with variables' sizes
        #   Note that for inward computation we are using the inverse map
        for arg, var_map_tuple in self.arg2var_map_inv_tuple.items():
            pat_var = self.arg2var[arg]
            _, domain, image = var_map_tuple
            assert domain.issubset(set(range(arg.size)))
            assert image == set(range(pat_var.size))

        # 1. First, translate predicate arguments to pattern variables. This step involves broadening the variable
        #       dimension if predicate argument size is smaller than pattern variable size, or map predicate argument
        #       values to pattern variable values if a VariableMap is specified for the given predicate argument.
        #    For the mapping we should use original forward mapping, because we are selecting "image" to place in
        #       "domain", in the order mandated by domain.
        #    Note that we have guaranteed that forward mapping's domain is equal to pattern variable's size range
        #    A running list of variables is maintained to keep track of variable dimensions
        mapped_var_list = copy.deepcopy(in_rel_var_list)
        for dim, pred_arg in enumerate(mapped_var_list):
            pat_var = self.arg2var[pred_arg]
            # Apply map if VariableMap is specified
            if pred_arg in self.arg2var_map_tuple.keys():
                map_dict, _, _ = self.arg2var_map_tuple[pred_arg]
                indices = torch.tensor(list(map_dict[i] for i in range(pat_var.size)), dtype=torch.long)
                msg = msg.batch_index_select(dim, indices)
            # Broaden the variable dimension size if currently it is smaller than the pattern variable's size
            if msg.b_shape[dim] < pat_var.size:
                msg = msg.batch_broaden(dim, pat_var.size)
            # Change predicate argument to pattern variable Variable instance
            mapped_var_list[dim] = pat_var

        # 2. The step above guarantees that for any dimensions that share the same pattern variable, their axis values
        #       are semantically identical to the pattern variable's value.
        #    Now we should collapse the dimensions that share the same pattern variable, by selecting the diagonal
        #       entries across these dimensions.
        for pt_var in mapped_var_list:
            # Current position of pt_var in the moving mapped_var_list. Using enumerate() iterator will not return
            #   index of value w.r.t. a moving list
            dim1 = mapped_var_list.index(pt_var)
            # Find any repetition in later part of the list
            if pt_var in mapped_var_list[dim1 + 1:]:
                # Current index of the first repetition
                dim2 = mapped_var_list[dim1 + 1:].index(pt_var) + dim1 + 1
                # Remove these two entries from the list
                mapped_var_list.remove(pt_var)
                mapped_var_list.remove(pt_var)
                # Append one to the end
                mapped_var_list.append(pt_var)
                # Selecting diagonal entries in message
                msg = msg.batch_diagonal(dim1, dim2)

        assert set(mapped_var_list) == set(out_rel_var_list)

        # 3. With all predicate argument dimension converted to pattern variable dimensions and all repetitions
        #       diagonalized, we guarantee that all predicate variable appears in mapped_var_list.
        #    The last thing to do is to permute the batch dimensions so that the processed message's dimension match
        #       exactly with out_rel_var_list
        perm_order = list(mapped_var_list.index(pt_var) for pt_var in out_rel_var_list)
        msg = msg.batch_permute(perm_order)
        assert msg.b_shape == torch.Size([v.size for v in out_rel_var_list])

        # Send message
        out_ld.write(msg)

    def outward_compute(self):
        """
            Outward computation. Convert pattern relational variables to predicate relational arguments. Apply mappings
                to relational variable's values, if specified.

            For outward direction, we are assuming this is used in action or condact patterns. Accordingly, the
                original forward mapping should be used to map pattern variables to predicate arguments.

            Will check anyway if domain and image of the forward map agree with the size range of the predicate argument
                and pattern variable respectively. However to be user friendly this should be checked beforehand by
                compiler.

            Note that image of the map should be a subset of predicate argument size range, but its domain should be
                exactly equal to the pattern variable size range.

            The computations to be carried out can be summarized in three steps: un-diagonalize, map/narrow, & permute
        """
        in_ld, out_ld = self.labeled_ld_pair['outward']
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        msg = in_ld.read()
        assert isinstance(msg, Message)
        in_rel_var_list, out_rel_var_list = in_ld.vn.rel_var_list, out_ld.vn.rel_var_list

        # Check that given data structures agree with variable lists of the incident variable node
        assert set(self.var2arg.keys()) == set(in_rel_var_list)
        assert set(self.arg2var.keys()) == set(out_rel_var_list)
        # Check that mapping's domain and image agree with variables' sizes
        #   For outward computation we are using the forward map
        for arg, var_map_tuple in self.arg2var_map_tuple.items():
            pt_var = self.arg2var[arg]
            _, domain, image = var_map_tuple
            assert image.issubset(set(range(arg.size)))
            assert domain == set(range(pt_var.size))

        # 1. First, translate pattern variables to predicate arguments. This step involves unbinding the predicate
        #       variables that are referenced by multiple predicate arguments.
        #    Computationally, this is achieved by un-diagonalize, or embed entries along that predicate variables'
        #       dimension into a 2D plane (higher dimensional space if there are more than 2 such predicate arguments).
        #    A running list of variables is maintained to keep track of variable dimensions
        mapped_var_list = copy.deepcopy(in_rel_var_list)
        for pt_var in mapped_var_list:
            # Look up how many predicate arguments reference this single pattern variable. If only 1, simply change
            #   variable. Otherwise, need to do more
            # Use .index() to get the position of current variable, instead of from enumerate() iterator, because the
            #   the latter does not keep up with a running list.
            dim1 = mapped_var_list.index(pt_var)
            args = self.var2arg[pt_var]
            num_shared = len(args)
            assert num_shared > 0
            # Switch pattern variable to predicate argument in-place anyway. If the pattern variable is associated with
            #   multiple predicate arguments, process further.
            mapped_var_list[dim1] = args[0]
            if num_shared > 1:
                # Iterate over rest of the associated predicate arguments
                for j, arg in enumerate(args):
                    # Only arguments from the second one will be processed. First associated argument stays in-place.
                    if j == 0:
                        continue
                    # Un-diagonalize the message by embedding the entries along the i-th dimension of the original
                    #   message into a 2D plane specified by the i-th and the last dimension of the new message
                    msg = msg.batch_diag_embed(diag_dim=dim1, target_dim1=dim1, target_dim2=-1)
                    # This corresponds to append the other dimension to the end, so we append the currently referenced
                    #   predicate argument Variable instance to the running variable list
                    mapped_var_list.append(arg)

        assert set(mapped_var_list) == set(out_rel_var_list)

        # 2. The step above guarantees a symbolic one-to-one mapping between message dimensions and predicate arguments'
        #       variable dimensions. We now need to narrow the variable dimension if the predicate argument's size is
        #       smaller than the associated pattern variable's size, or to map pattern variable's value to predicate
        #       argument's value if a VariableMap is specified.
        for dim, pred_arg in enumerate(mapped_var_list):
            pat_var = self.arg2var[pred_arg]
            # Apply map if VariableMap is specified
            if pred_arg in self.arg2var_map_tuple.keys():
                map_dict, _, _ = self.arg2var_map_tuple[pred_arg]
                indices = torch.tensor(list(map_dict[i] for i in range(pat_var.size)), dtype=torch.long)
                msg = msg.batch_index_put(dim, indices)
            # Broaden the variable dimension size if currently it is smaller than the pattern variable's size
            if msg.b_shape[dim] > pred_arg.size:
                msg = msg.batch_narrow(dim, pred_arg.size)

        # 3. Finally, with all pattern variables converted to predicate arguments and the values are now with respect to
        #       the predicate arguments, the last thing to do is to permute the batch dimensions so that the processed
        #       message's dimensions match exactly with out-_rel_var_list
        perm_order = list(mapped_var_list.index(pred_arg) for pred_arg in out_rel_var_list)
        msg = msg.batch_permute(perm_order)
        assert msg.b_shape == torch.Size([v.size for v in out_rel_var_list])

        # Send message
        out_ld.write(msg)


class ExpSumNode(AlphaFactorNode):
    """
        Expansion / Summarization Node

        This node is a component of the alpha conditional subgraph, so admits up to two pairs of incoming and outgoing
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

        A sum_op should be specified during initialization to specify special summarization semantics, such as Max
            Product semantics or searching. If not specified, will default to Sum-Product alike summarization. Please
            refer to Message class documentation for more information.
    """
    def __init__(self, name, sum_op=None):
        """
            Necessary data structure:

            :param sum_op:      None or a Summarization instance. Default is None.
        """
        super(ExpSumNode, self).__init__(name)
        self.pretty_log["node type"] = "Expansion Summarization Factor Node"

        assert sum_op is None or isinstance(sum_op, Summarization)
        if sum_op is not None:
            raise NotImplementedError("Summarization operation using Summarization instance is not yet implemented.")

        self.sum_op = sum_op

    def inward_compute(self):
        """
            Expansion operation. Expand and permutes the incoming message's relational variable dimensions to match the
                outgoing relational relational variable dimensions.
        """
        in_ld, out_ld = self.labeled_ld_pair['inward']
        msg = in_ld.read()
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        assert isinstance(msg, Message)
        in_rel_var_list, out_rel_var_list = in_ld.vn.rel_var_list, out_ld.vn.rel_var_list
        # Check that the set of relational variables of incoming message is a subset of that of outgoing message
        assert set(in_rel_var_list).issubset(set(out_rel_var_list))

        # Keep a running list of variables
        mapped_var_list = copy.deepcopy(in_rel_var_list)

        # For every relational variable in out_rel_var_list that is not in in_rel_var_list, unsqueeze a dimension of the
        #   message as the last dimension.
        for pt_var in out_rel_var_list:
            if pt_var not in in_rel_var_list:
                # Unsqueeze message
                msg = msg.batch_unsqueeze(dim=-1)
                # Append the pt_var to the end of running var list
                mapped_var_list.append(pt_var)

        assert set(mapped_var_list) == set(out_rel_var_list)

        # Permute message dimension so that it matches the order given by out_rel_var_list
        perm_order = list(mapped_var_list.index(v) for v in out_rel_var_list)
        msg = msg.batch_permute(perm_order)

        # Expand to full relational variable dimension shape
        target_shape = torch.Size([v.size for v in out_rel_var_list])
        msg = msg.batch_expand(target_shape)

        # Send message
        out_ld.write(msg)

    def outward_compute(self):
        """
            Summarization operation. Summarize over incoming message's relational variable dimensions that are not
                referenced by outgoing message's relational variables.

            The summarization semantic is defined by the sum_op specified during initialization. If sum_op is None,
                uses default summarization semantic defined at the Message level.
        """
        in_ld, out_ld = self.labeled_ld_pair['outward']
        msg = in_ld.read()
        assert isinstance(in_ld, LinkData) and isinstance(out_ld, LinkData)
        assert isinstance(msg, Message)
        in_rel_var_list, out_rel_var_list = in_ld.vn.rel_var_list, out_ld.vn.rel_var_list
        # Check that the set of relational variables of outgoing message is a subset of that of incoming message
        assert set(out_rel_var_list).issubset(set(in_rel_var_list))

        # Keep a running list of variables
        mapped_var_list = copy.deepcopy(in_rel_var_list)

        # Summarize using custom sum_op
        if self.sum_op is not None:
            # Flatten both the group of dimensions to be summarized over and the group of other dimensions. Put the
            #   former as the last dimension and the latter as the first batch dimension
            sum_dims = list(dim for dim, v in enumerate(in_rel_var_list) if v not in out_rel_var_list)
            other_dims = list(dim for dim, v in enumerate(in_rel_var_list) if v in out_rel_var_list)
            if len(sum_dims) > 0:
                # First flatten other_dims, then sum_dims, so that flattened sum_dims will be the last dim
                msg = msg.batch_flatten(other_dims)
                msg = msg.batch_flatten(sum_dims)
                # Process using the sum_op
                msg = self.sum_op.process(msg)
                # Reshape
                msg = msg.batch_reshape(other_dims)

        # Otherwise if sum_op is None, carry out default summarization
        else:
            # Iterate over all relational variables not referenced by out_rel_var_list
            for pt_var in in_rel_var_list:
                if pt_var not in out_rel_var_list:
                    dim = mapped_var_list.index(pt_var)
                    # Summarize over the message batch dimension
                    msg = msg.batch_summarize(dim)
                    # Remove the variable from the running list
                    mapped_var_list.remove(pt_var)

        assert set(mapped_var_list) == set(out_rel_var_list)

        # Permute message dimension so that it matches the order given by out_rel_var_list
        perm_order = list(mapped_var_list.index(v) for v in out_rel_var_list)
        msg = msg.batch_permute(perm_order)

        # Send message
        out_ld.write(msg)


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

        This node is a component of the alpha conditional subgraph, so admits up to two pairs of incoming and outgoing
            links. Link must declare special attribute 'direction' with value 'inward' or 'outward' to indicate whether
            it is pointing toward the conditional gamma factor node or not.
    """
    pass


"""
    Nodes relating to Conditional Beta subgraph Structures
        - BetaNode, EventSplitVarNode, PartCombNode
"""


class BetaNode(Node, ABC):
    """
        Abstract base class for nodes belonging to the beta subgraph of a conditional.

        Captures the commonality of Beta nodes, include link connectivity and inward & outward compute pattern.
    """
    pass


class EventSplitVarNode(BetaNode, VariableNode):
    pass


class PartCombNode(BetaNode, FactorNode):
    pass


class GFN(AlphaFactorNode):
    """
        Gamma Factor Node

        Carry out general-inference message computation at the PGM factor node.

        Induce a message computation task for each of the outgoing link.
    """
    pass





















