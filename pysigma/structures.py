"""
    Common structures and componenets serving the Sigma cognitive language

    Author of this file: Jincheng Zhou, University of Southern California
"""
from collections.abc import Iterable
from itertools import chain
import warnings
import functools
import numpy as np
import torch
from torch import Size
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint
from .utils import intern_name, extern_name, DistributionServer, KnowledgeServer
from .defs import Variable, VariableMetatype, Message, MessageType


class VariableMap:
    """Class type for declaring mappings on relational variables in pattern elements. Wraps around a user-defined
    mapping function and records the one-to-one correspondence between inputs and outputs.

    Since relational variables can be viewed as non-negative finite integer-valued variables, a VariableMap instance
    therefore declares an integer-valued mapping with finite domain.

    The domain and codomain are assumed fixed, so they should be provided during initialization. The mapping can
    be computed either lazily at runtime or pre-computed at initialization. This flexibility is to allow dynamic
    mappings such as neural attention modules.

    Only injective mapping can be accepted. This is because the semantic of mapping two relational variable's values
    to a single value is ill-defined. The injectivity is checked during `set_map()` by comparing the cardinality of the
    image and the cardinality of the domain.

    Parameters
    ----------
    mapping_func : callable
        a user-specified function. Takes a numpy array of integers as input, and returns a numpy array of integers of
        the same size. Each entry in the output array corresponds to the value of `f(x)` of input `x` at the same index
        in the input array.
    domain : set of int
        The domain of the mapping.
    codomain : set of int
        The codomain of the mapping.
    dynamic : bool, optional
        Indicates whether the mapping is dynamic. If ``True``, then `mapping_func` will be called each time a mapping
        dictionary is desired, i.e., when `get_map()` or `get_inverse_map()` is called. Otherwise, `mapping_func` will
        only be called once during initialization of this VariableMap instance, and the result will cached. Defaults
        to ``False``.

    Attributes
    ----------
    mapping_func
    domain
    codomain
    dynamic
    map : dict
        The mapping cache.
    image : set
        Image of the map. Should be a subset of ``self.codomain``.
    """
    def __init__(self, mapping_func, domain, codomain, dynamic=False):
        # Argument validation
        if not callable(mapping_func):
            raise ValueError("The 1st argument 'mapping_func' should be a python callable, accepting a list of "
                             "integers as the batched input to the mapping and produces a list of integers of the same "
                             "size as the batched output")
        if not isinstance(domain, set) or not all(isinstance(i, int) and i >= 0 for i in domain):
            raise ValueError("The 2nd argument 'domain' must be a set of non-negative integers, denoting the domain "
                             "of the specified mapping")
        if not isinstance(codomain, set) or not all(isinstance(i, int) and i >= 0 for i in codomain):
            raise ValueError("The 3rd argument 'codomain' must be a set of non-negative integers, denoting the "
                             "codomain of the specified mapping")
        if not isinstance(dynamic, bool):
            raise ValueError("The 4th argument 'dynamic' must be of type bool")

        self.mapping_func = mapping_func
        self.domain = domain
        self.codomain = codomain
        self.dynamic = dynamic
        # mapping cache. This field will be looked up later as definition of the mapping
        self.map = None
        # The image of the map, i.e., the set of values who has a corresponding input. It should be a subset of codomain
        self.image = None

        # Set the map cache if not dynamic
        if not dynamic:
            self.set_map()

    def set_map(self):
        """Set ``self.map`` by obtaining the mapping dictionary from ``self.mapping_func``.

        Raises
        ------
        ValueError
            If `self.mapping_func` does not return a numpy array or returns one with wrong data type.
        ValueError
            If ``self.mapping_func`` does not return a numpy array with the same size as the input array.
        ValueError
            If ``self.mapping_func`` contains values not in the codomain.
        ValueError
            If ``self.mapping_func`` is found to be not injective.
        """
        # Input list
        input = np.array(list(self.domain))
        output = self.mapping_func(input)
        # Check output format and if its entries are in codomain range
        if not isinstance(output, np.ndarray) or not all(isinstance(i, np.int64) and i >= 0 for i in output):
            raise ValueError("The provided mapping python callable should return a numpy array of non-negative "
                             "np.int64. Instead, found: '{}'".format(output))
        if input.size != output.size:
            raise ValueError("The output list from the provided mapping python callable should be of the same size as "
                             "the input list. Expecting size '{}', instead found size '{}'"
                             .format(len(input), len(output)))
        for i in output:
            if i not in self.codomain:
                raise ValueError("The output from the provided mapping python callable should be within the specified "
                                 "codomain: '{}'. Instead, found entry in the output list: '{}'"
                                 .format(self.codomain, i))
        # Set mapping
        self.map = dict(zip(input, output))
        # Set mapping image
        self.image = set(self.map.values())
        # Check injectivity
        if len(self.image) != len(self.domain):
            raise ValueError("The specified mapping should be injective. However, found that the cardinality of the "
                             "mapping's image is '{}', whereas the cardinality of the specified domain is '{}'"
                             .format(len(self.image), len(self.domain)))

    def get_map(self):
        """Returns the mapping dictionary, the map's domain, and the map's image.

        If dynamic, then calls `set_map()` to re-compute the dict first, otherwise returns the cached one.

        Returns
        -------
        tuple
            A 3-tuple containing the data ``(map_dict, domain, image)``.
        """
        if self.dynamic:
            self.set_map()
        return self.map, self.domain, self.image

    def get_inverse_map(self):
        """Returns the inverse map's mapping dictionary, the inverse map's domain (original map's image), and the
        inverse map's image (should be the same as the original map's domain)

        Note that because injectivity is guaranteed, computing an inverse map is possible.

        If dynamic, then calls set_map() to re-compute the dict first, otherwise returns the cached one.

        Returns
        -------
        tuple
            A 3-tuple containing the data ``(inv_map_dict, inv_image, inv_domain)``.
        """
        if self.dynamic:
            self.set_map()

        inverse_map = dict(zip(self.map.values(), self.map.keys()))
        return inverse_map, self.image, self.domain


class FactorFunction:
    """
        Class type for factor node function

        A FactorFunction instance can be thought of as a function object that takes in a group of batched tensors, each
            corresponding to a batched value assignments to a random variable, and produces a single batched tensor that
            represents the joint probability density, i.e.,
                    val = Prob(X_1, X_2, ..., X_n)
            where val, X_1, X_2, ..., X_n are all batched over the first dimension.

        The first dimension of all tensors indexes the group of value assignments to the random variables. Therefore, it
            should be ignored by the factor function procedure and simply treated as the "batched" dimension.

        Defines different types of factor functions:
            a. Tabular factor function
                Returns a full-dimensional tensor at once representing the entire factor function table, each dimension
                    corresponds to a random variable's support. Suitable when all random variables X_1, ..., X_n have
                    finite discrete domains and the size is manageable.
                For this type of factor function, no inputs regarding the R.V. values are necessary, because the
                    returned factor table should cover all combinations of inputs.
                This is the same as the legacy Lisp Sigma's factor node function

            b. General form generative joint-probability density function
                Given a batched group of RV value assignments as inputs, returns a batched tensor array representing the
                    probability density. To enforce the probabilistic semantics, i.e.,
                        val = Prob(X_1, X_2, ..., X_n)
                    The entries in the returned tensor should be within the range [0, 1].
                Compatible with all particle-based inference methods

            c. Exponential form joint-probability density function
                Explicitly defines an exponential distribution. Encodes an exponential distribution class as the
                    factor node conditional distribution. Returns the PARAMETERS to the distribution.
                Must declare this type if want architecture to recognize conjugate-exponential model structure and
                    carries out closed-form message updates.

            d. Deterministic factor function
                Returns one or multiple batched VALUE tensors corresponding to one or multiple R.V.s given a batched
                    value assignments to other variables. Enforce the semantics of
                        Y_1, Y_2, ..., Y_m = Func(X_1, X_2, ..., X_n)
                Conceptually identical to (b) type factor function with a delta distribution, but due to concerns of
                    sample efficiency, this type should be used in practice if the factor function is deterministic.
                Note that once defined, the directionality of the corresponding Conditional is also assumed and fixed.
                In other words, X_1, ..., X_n should only appear in condition patterns, and Y_1, ..., Y_m only in action
                patterns.
    """
    pass


class Summarization:
    """The decorator class for the custom summarization callback procedures which will be called during the outward
    summarization step in the Expansion / Summarization Factor Node (ESFN).

    Provides the decorator ``@Summarization``.

    .. note:: The custom callback function should have the following signature:

    Parameters
    ----------
    content_type : {'distribution', 'particles', 'both'}
        The type of the content to process for the current execution.
    rv_info : OrderedDict
        Information / metadata regarding the random variables of the message. Each key-value pair in the dictionary is
        key-ed by the random variable's name ``var_name : str`` and contains a 2-tuple
        ``(var_size : int, constraints : set of torch.distributions.Constraint)`` as the value. Note that
        `rv_info` is ordered, and this order of the random variables will be respected by entries in `events` and
        `log_densities`.
    batch_shape : torch.Size
        A size-2 torch.Size list representing the batch dimension shapes. **The first dimension is the batch dimension
        to be summarized over and reduced, and the second dimension should be left untouched.**
    dist : torch.distributions.Distribution, optional
        The batched distribution instance representing the message. `dist` 's batch shape is `batch_shape`, and its
        event shape is the sum of the sizes of all random variables. This field will be provided when `content_type` is
        ``distribution`` or ``both``, or will be left ``None`` when `content_type` is `particles`.
    events : tuple of torch.Tensor, optional
        The tuple of event value tensors. Each entry is a 2D tensor corresponding to one of the random variables
        declared in `rv_info` associated with the message. The first dimension is the sample dimension with a size
        equal to the number of event particles. The last dimension is the event dimension with a size equal to the
        size of the corresponding random variable. This field will be provided when `content_type` is ``particles`` or
        ``both``, or will be left ``None`` when `content_type` is ``parameter``.
    log_densities : tuple of torch.Tensor, optional
        The tuple of **marginal** log sampling densities of each of the event values in `events`. Each entry is a 1D
        tensor annotating the event value tensor in `events` with the same tuple index. The dimension size equals to
        the number of event particles. This field will be provided when `content_type` is ``particles`` or ``both``, or
        will be left ``None`` when `content_type` is ``parameter``.
    weight : torch.Tensor, optional
        The event particles weight tensor. Have shape `(batch_shape + sample_shape)`, where `sample_shape` is the
        concatenated list of the sample sizes of all event tensors in `events`. This field will be provided when
        `content_type` is ``particles`` or ``both``, or will be left ``None`` when `content_type` is ``parameter``.

    Returns
    -------
    tuple
        A 2-tuple consisting of `(reduced_dist : torch.distributions.Distribution or None, reduced_weight :
        torch.Tensor or None)`. `reduced_dist` will be ignored if `content_type` is ``particles``, and `reduced_weight`
        will be ignored if `content_type` is ``parameter``. Both of the entries should have the reduced batch shape of
        length 1.
    """
    def __init__(self, sum_func):
        # Update wrapped function's information to enable introspection
        functools.update_wrapper(self, sum_func)
        self.sum_func = sum_func

    def __call__(self, msg, ran_vars):
        """Implements so that the Summarization instance is a callable functor.

        The summarization instance should be called by ESFN internally only.

        Parameters
        ----------
        msg : Message
            The incoming message. Batch shape should be of length 2, with the first batch dimension being the one to be
            summarized over, and the second dimension being the one to be left untouched.
        ran_vars : tuple of Variable
            The tuple of random variables of the target Variable node

        Returns
        -------
        Message
            The reduced message. Will have only one batch dimension which is the one to be left untouched from the
            original message.
        """
        # Sort out input data
        assert isinstance(msg, Message) and len(msg.b_shape) == 2
        assert MessageType.Parameter in msg.type or MessageType.Particles in msg.type
        assert 'dist_info' in msg.attr.keys() and 'dist_class' in msg.attr['dist_info'].keys()
        assert isinstance(ran_vars, tuple) and \
            all(isinstance(v, Variable) and v.metatype is VariableMetatype.Random for v in ran_vars)

        # Return directly if message is an identity
        if msg.isid:
            return msg
        # Decide content_type based on BOTH the message type as well as message components type. Only feed components
        # that are necessary for processing to the wrapped callback.
        if msg.type is MessageType.Both:
            if isinstance(msg.parameter, torch.Tensor) and isinstance(msg.weight, torch.Tensor):
                content_type = 'both'
            elif isinstance(msg.parameter, torch.Tensor):
                content_type = 'parameter'
            else:
                content_type = 'particles'
        elif msg.type is MessageType.Parameter:
            content_type = 'parameter'
        else:
            content_type = 'particles'

        rv_info = {v.name: (v.size, v.constraints) for v in ran_vars}

        # Clone tensor contents
        dist = DistributionServer.param2dist(msg.attr['dist_info']['dist_class'], msg.parameter,
                                             b_shape=msg.b_shape, e_shape=msg.e_shape, dist_info=msg.attr) \
            if content_type in ['parameter', 'both'] else None
        events = tuple(p.clone() for p in msg.particles) \
            if content_type in ['particles', 'both'] else None
        log_densities = tuple(d.clone() for d in msg.log_densities) \
            if content_type in ['particles', 'both'] else None
        weight = msg.weight.clone() \
            if content_type in ['particles', 'both'] else None

        # Obtain and check result
        result = self.sum_func(content_type=content_type, rv_info=rv_info, batch_shape=msg.b_shape,
                               dist=dist, events=events, log_densities=log_densities, weight=weight)
        if not isinstance(result, tuple) and len(result) == 2:
            raise ValueError("The result returned by a Summarization callback should be a 2-tuple. Found result type "
                             "{}".format(type(result)))
        reduced_dist, reduced_weight = result
        if content_type in ['parameter', 'both']:
            if not isinstance(reduced_dist, Distribution):
                raise ValueError("When `content_type` is 'parameter' or 'both', the first result returned by a "
                                 "Summarization callback should be a torch.distributions.Distribution instance. "
                                 "Encountered result type: {}.".format(type(reduced_dist)))
            if type(reduced_dist) != type(dist):
                raise ValueError("When `content_type` is 'parameter' or 'both', the distribution instance returned by "
                                 "the Summarization callback should belong to the same distribution class as the "
                                 "original distribution. Expect class {}, encountered class {}."
                                 .format(type(dist), type(reduced_dist)))
            if len(reduced_dist.batch_shape) != 1 or reduced_dist.batch_shape[0] != msg.b_shape[-1]:
                raise ValueError("When `content_type` is 'parameter' or 'both', the distribution instance returned "
                                 "by the Summarization callback should have a batch shape of length 1 and have a size "
                                 "equal to the second batch size of the original message. Expect size {}, encountered "
                                 "batch shape {}".format(msg.b_shape[-1], reduced_dist.batch_shape))
            if reduced_dist.event_shape != msg.e_shape:
                raise ValueError("When `content_type` is 'parameter' or 'both', the distribution instance returned "
                                 "by the Summarization callback should have the same event shape as does the original "
                                 "message. Expect event shape {}, encountered event shape {}"
                                 .format(msg.e_shape, reduced_dist.event_shape))
        if content_type in ['particles', 'both']:
            if not isinstance(reduced_weight, torch.Tensor):
                raise ValueError("When 'content_type' is 'particles' or 'both', the second result returned by a "
                                 "Summarization callback should be a torch.Tensor.. Encountered result type: {}."
                                 .format(type(reduced_weight)))
            if reduced_weight.shape != msg.b_shape[-1:] + msg.e_shape:
                raise ValueError("When 'content_type' is 'particles' or 'both', expect the weight tensor returned by "
                                 "the Summarization callback to have shape {}. Instead found {}."
                                 .format(msg.b_shape[-1:] + msg.e_shape, reduced_weight.shape))

        # Instantiate return message
        param = 0
        weight = 1
        if content_type in ['parameter', 'both']:
            param = DistributionServer.dist2param(reduced_dist, msg.attr['dist_info'])
        if content_type in ['particles', 'both']:
            weight = reduced_weight

        return_msg = Message(msg.type,
                             batch_shape=msg.b_shape[-1:],
                             param_shape=msg.p_shape, sample_shape=msg.s_shape, event_shape=msg.e_shape,
                             parameter=param, particles=msg.particles, weight=weight, log_densities=msg.log_densities,
                             device=msg.device, **msg.attr)
        return return_msg


class Type:
    def __init__(self, type_name, symbolic=False, size=None, symbol_list=None, value_constraint=None):
        """
            Specify a Type for a predicate's relational/random argument.

            For both relational and random arguments, a Type declares the size of the corresponding tensor dimension of
                the predicate's event particle messages. For a random argument, additionally, a Type also defines its
                valid value range and/or value type (e.g. real or integer values)

            If symbolic, must declare a 'symbol_list', and the 'size' field will be inferred from the length of the
                'symbol_list'. Otherwise must declare the 'size' value.

            value_constraint must be provided if and only if the Type is to be associated with random predicate
                arguments. Otherwise an exception will be thrown

        :param type_name: 'str' type. Name of the Sigma type
        :param symbolic: 'bool' type. Declares if the type comes with a symbol list
        :param size: 'int' type. The size of the argument's corresponding message dimension. Must specify if 'symbolic'
                     is False
        :param symbol_list: An iterable of 'str' representing symbols. Must specify if 'symbolic' is True
        :param value_constraint: A 'torch.distributions.constraints.Constraints' instance. Must specify if and only if
                                 this Type is to be associated with random predicate arguments
        """

        # Argument validation
        if not isinstance(type_name, str):
            raise ValueError("The 1st argument 'type_name' of a Type must be a 'str'")
        if not isinstance(symbolic, bool):
            raise ValueError("The 2nd argument 'symbolic' must be of 'bool' type")
        if symbolic is (symbol_list is None):
            raise ValueError("'symbolic' should be True if and only if 'symbol_list' is provided")
        if symbolic is (size is not None):
            raise ValueError("'symbolic' should be False if and only if 'size' is provided")
        if size is not None and (not isinstance(size, int) or size < 1):
            raise ValueError("'size' must be of type 'int' and value be greater than 0")
        if symbol_list is not None and (not isinstance(symbol_list, Iterable) or
                                        not all(isinstance(s, str) for s in symbol_list)):
            raise ValueError("'symbol_list' must be an iterable of 'str'")
        if value_constraint is not None and not isinstance(value_constraint, Constraint):
            raise ValueError("If specified, 'value_constraint' must be an instance of 'torch.distributions.constraints."
                             "Constraint'. Instead found '{}'".format(type(value_constraint)))

        # Declare attributes
        self.name = intern_name(type_name, "type")  # Prepend substring 'TYPE_' and send name to upper case
        self.symbolic = symbolic
        self.value_list = list(symbol_list) if self.symbolic else list(range(size))
        self.size = len(self.value_list)
        self.constraint = value_constraint

        # Set mapping between type values and actual axis values along message tensor's dimension.
        self.value2axis = dict(zip(self.value_list, range(self.size)))
        self.axis2value = dict(zip(range(self.size), self.value_list))

    def __eq__(self, other):
        # Two types are equal if ALL of their fields are equal
        assert isinstance(other, Type)
        val = self.name == other.name and \
              self.symbolic == other.symbolic and \
              self.size == other.size and \
              self.value_list == other.value_list
        return val

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        # String representation for display
        return extern_name(self.name, "type")

    def __hash__(self):
        # Hash by structure's own name. This assumes its name is unique in the model and therefore can be treated as
        #   the identity
        return hash(self.name)


class Predicate:
    def __init__(self, predicate_name, relational_args, random_args, inference_mode, num_particles=128,
                 distribution_class=None, memorial=False, perceptual=False):
        """
            Specify a Sigma predicate.

        :param predicate_name: 'str' type. The name of the predicate.
        :param relational_args: an Iterable of size-2 tuples. Each tuple specifies one relational argument - The first
                                element of the tuple should be of 'str' type specifying the argument's name, whereas
                                the second element should be of 'Type' type specifying the argument's type. The 'Type'
                                of these arguments can be arbitrary, but the arguments' names should be distinct.
        :param random_args: an Iterable of size-2 tuples, similar to the previous one. Besides those requirements for
                            relational arguments, random arguments' types cannot be symbolic
        :param inference_mode: one of "BP", "VMP", or "EP". Choosing the inference method to be used at this predicate.
                               This will also influence the choice of inference method used at related conditionals.
        :param num_particles: 'int' type. The number of particles to be drawn each cognitive cycle. Also the size of the
                              predicate's particle indexing dimension. If left for None, then this field will be decided
                              by the architecture.
        :param distribution_class:  A subclass of 'torch.distributions.Distribution'. Specify the assumed distribution
                                    of this predicate's knowledge. If specified, the predicate will be a "distribution"
                                    predicate, and a number of "num_particles" particles will be actively drawn at the
                                    start of each cognitive cycle. Otherwise, it will be assumed a "vector" predicate,
                                    and no particles will be actively drawn.
        :param memorial: 'bool' type. If True, the predicate is "memorial", meaning its knowledge will be latched to the
                         next cognitive cycle. Otherwise it is "memory-less", and its knowledge will be immediately
                         replaced with new information at each cycle.
        :param perceptual: 'bool' type. If True, the predicate is "perceptual", and so can perceive observations.
        """

        # Argument validation
        if not isinstance(predicate_name, str):
            raise ValueError("The 1st field 'predicate_name' must be of 'str' type")
        if not isinstance(relational_args, Iterable) or \
                not all((isinstance(arg, tuple) and len(arg) == 2) for arg in relational_args):
            raise ValueError("The 2nd field 'relational_args' must be an Iterable of size-2 tuples")
        if not isinstance(random_args, Iterable) or \
                not all((isinstance(arg, tuple) and len(arg) == 2) for arg in random_args):
            raise ValueError("The 3nd field 'random_args' must be an Iterable of size-2 tuples")
        if not isinstance(inference_mode, str) or inference_mode.upper() not in ["BP", "VMP", "EP"]:
            raise ValueError("Must declare the inference method to be used at the predicate using the 4nd field "
                             "'inference_mode', one of 'BP', 'VMP', or 'EP'")
        if num_particles is not None and not (isinstance(num_particles, int) and num_particles > 0):
            raise ValueError("If specified, the 4th field 'num_particles' must be a positive integer")
        if distribution_class is not None and not issubclass(distribution_class, Distribution):
            raise ValueError("If specified, the 5th field 'distribution' must provide a subclass of 'torch."
                             "distributions.Distribution")
        if not isinstance(memorial, bool):
            raise ValueError("The 6th field 'memorial' must be of 'bool' type")
        if not isinstance(perceptual, bool):
            raise ValueError("The 7th field 'perception' must be of 'bool' type")

        # Declare attributes
        self.name = intern_name(predicate_name, "predicate")

        self.relational_args = []
        self.relarg_name2relarg = {}
        self.relarg_name2type = {}

        self.random_args = []
        self.ranarg_name2ranarg = {}
        self.ranarg_name2type = {}

        self.arg_name2metatype = {}

        self.inf_mode = inference_mode.upper()
        self.index_arg = Variable("INDEX", VariableMetatype.Indexing, num_particles)
        self.dist_class = distribution_class
        self.memorial = memorial
        self.perceptual = perceptual
        self.event_dims = None  # Event dimensions. Parameter dims may be different and dependent on distribution class

        for rel_arg in relational_args:
            arg_name = rel_arg[0]
            arg_type = rel_arg[1]
            if not isinstance(arg_name, str) or not isinstance(arg_type, Type):
                raise ValueError("The first element of an argument tuple must be of 'str' type to declare the "
                                 "argument's name, and the second one be an instance of 'Type'. Instead, found: "
                                 "({}, {})".format(type(arg_name), type(arg_type)))

            # Check that arguments' names are distinct
            if arg_name in self.relarg_name2relarg.keys():
                raise ValueError("Relational arguments' names must be distinct. Instead found repetition: {}"
                                 .format(arg_name))
            # Check if variable's type specifies a value constraint. If specified, throw a user warning notifying that
            #   the value constraint will be ignored
            if arg_type.constraint is not None:
                warnings.warn("A value constraint '{}' is found in the type '{}' associated with relational argument "
                              "'{}'. The value constraint is ignored for now, but please check if the type should be "
                              "used to address a relational predicate argument."
                              .format(arg_type.constraint, arg_type, arg_name))

            rel_var = Variable(arg_name, VariableMetatype.Relational, arg_type.size, None)
            self.relational_args.append(rel_var)
            self.relarg_name2relarg[arg_name] = rel_var
            self.relarg_name2type[arg_name] = arg_type

        for ran_arg in random_args:
            arg_name = ran_arg[0]
            arg_type = ran_arg[1]
            if not isinstance(arg_name, str) or not isinstance(arg_type, Type):
                raise ValueError("The first element of an argument tuple must be of 'str' type to declare the "
                                 "argument's name, and the second one be an instance of 'Type'. Instead, found: "
                                 "({}, {})".format(type(arg_name), type(arg_type)))

            # Check that arguments' names are distinct
            if arg_name in chain(self.relarg_name2relarg.keys(), self.ranarg_name2ranarg.keys()):
                raise ValueError("Random arguments' names must be distinct, including from relational arguments. "
                                 "Instead found repetition: {}".format(arg_name))
            # Check that RV's type is not symbolic
            if arg_type.symbolic:
                raise ValueError("Random argument's type cannot be symbolic")
            # Check that RV's type specifies value constraint
            if arg_type.constraint is None:
                raise ValueError("A value constraint must be specified for a random variable's type. Instead found "
                                 "random variable '{}' with type '{}' where no value constraint is specified."
                                 .format(arg_name, arg_type))

            ran_var = Variable(arg_name, VariableMetatype.Random, arg_type.size, arg_type.constraint)
            self.random_args.append(ran_var)
            self.ranarg_name2ranarg[arg_name] = ran_var
            self.ranarg_name2type[arg_name] = arg_type

        for arg_name in self.relarg_name2relarg.keys():
            self.arg_name2metatype[arg_name] = VariableMetatype.Relational
        for arg_name in self.ranarg_name2ranarg.keys():
            self.arg_name2metatype[arg_name] = VariableMetatype.Random

        # TODO: Check knowledge format. e.g. if RV type compatible with declared predicate's distribution class

        # Dimensions of event tensor: [N, B_1, ..., B_n, (S_1+...+S_m)]
        self.event_dims = Size([self.index_arg.size] +
                         [v.size for v in self.relational_args] +
                         [sum([v.size for v in self.random_args])])

    def __str__(self):
        # String representation for display
        return self.name

    def __hash__(self):
        # Hash by structure's own name. This assumes its name is unique in the model and therefore can be treated as
        #   the identity
        return hash(self.name)


class Conditional:
    def __init__(self, conditional_name, conditions=None, condacts=None, actions=None,
                 function=None, function_var_names=None):
        """
            Specify a Sigma conditional

            Each of conditions, condacts, and actions field consists of an Iterable of "predicate patterns", where each
                "predicate pattern" is represented by a size-2 tuple. The first element of the tuple is a Predicate
                instance, representing a predicate. The second element is an Iterbale of "pattern elements", where each
                "pattern element" is a size-2 or size-3 tuple. The first element of such tuple is of 'str' type,
                corresponds to one of the predicate's argument. The second element is either a 'str' (representing a
                pattern variable), an Iterable of 'int' (list of constant integer values), or an Iterable of 'str'
                (also constant but matches symbolic values). Finally, the third element is OPTIONAL, of
                'PatternTransformation' type, representing a transformation on the predicate variable's values.

            In short, a predicate pattern should look like:
                    (pred, [(arg_1, var_1, trans_1), (arg_2, var_2, trans_2), ..., (arg_n, var_n, trans_n)])

            The purpose of declaring pattern variables is to match predicate arguments within or across predicates,
                i.e., variable binding, or (only for random arguments) to be picked up and recognized by the factor
                function. Therefore, if matching is not necessary for a predicate argument, or if it is ignored by the
                factor function, its corresponding pattern element can be left empty, and the architecture will declare
                a default pattern variable that is uniquely associated this predicate argument.
            Thus, it is acceptable to declare the following pattern:
                    (pred, None)
                in which case none of the arguments will be matched with others, and the factor function should ignore
                all of the random arguments.

            Note that for condact and action predicate patterns, it is expected that no transformation is declared on
                any of the variables, All element tuples are size-2 tuples.

            The 'function' field specifies a factor function that semantically operates on the events represented by
                random variables from the incoming messages. Therefore, only random variables are visible to the factor
                function, and the dimensions corresponding to relational variables should be treated as the "batch
                dimensions" of the incoming messages.

        :param conditional_name: 'str' type. The name of this conditional
        :param conditions: an Iterable of size-2 tuples.
        :param condacts: same as 'conditions'
        :param actions: same as 'conditions'
        :param function: 'FactorFunction' type. Declares a factor function at this conditional. If left as None, will
                         default to a constant factor function
        :param function_var_names: an Iterable of 'str'. The list of random variables that the factor function concerns.
                                   The order of the random variables will be respected in accordance to the order given
                                   by the supplied iterable. The list of messages will be given to the factor function
                                   w.r.t. this order at inference time. If left for None, will default to all random
                                   variables
        """
        # TODO: Implement new Conditional

        # Argument validation #
        if not isinstance(conditional_name, str):
            raise ValueError("1st argument 'conditional_name' must be of 'str' type")
        # cannot have only conditions or only actions
        if condacts is None and (conditions is None or actions is None):
            raise ValueError("A conditional cannot have only condition patterns or only action patterns")
        # function must be specified if function_var_names is specified
        if function is None and function_var_names is not None:
            raise ValueError("5th argument 'function' must be specified if 'function_var_names' is specified")
        if function is not None and not isinstance(function, FactorFunction):
            raise ValueError("If specified, 5th argument 'function' must be of 'FactorFunction' type")
        if function_var_names is not None and \
                not (isinstance(function_var_names, Iterable) and all(isinstance(v, str) for v in function_var_names)):
            raise ValueError("If specified, 6th argument 'function_var_names' must be an Iterable of 'str', "
                             "representing the list of pattern random variables to be picked up and recognized by the "
                             "factor function.")

        # Check predicate pattern formats
        if conditions is not None and \
                not (isinstance(conditions, Iterable) and all(isinstance(p, tuple) for p in conditions)):
            raise ValueError("When specified, 2nd argument 'conditions' must be an Iterable of tuples")
        if condacts is not None and \
                not (isinstance(condacts, Iterable) and all(isinstance(p, tuple) for p in condacts)):
            raise ValueError("When specified, 3rd argument 'condacts' must be an Iterable of tuples")
        if actions is not None and \
                not (isinstance(actions, Iterable) and all(isinstance(p, tuple) for p in actions)):
            raise ValueError("When specified, 4th argument 'actions' must be an Iterable of tuples")

        conditions = [] if conditions is None else list(conditions)
        condacts = [] if condacts is None else list(condacts)
        actions = [] if actions is None else list(actions)

        for pat_type, pat_group in zip([0, 1, 2], [conditions, condacts, actions]):
            for pat in pat_group:
                if not len(pat) == 2:
                    raise ValueError("Expect each predicate pattern to be a size-2 tuple, instead found {}".format(pat))
                pred, pat_elements = pat
                if not isinstance(pred, Predicate):
                    raise ValueError("Expect the first element of each predicate pattern tuple to be an instance of "
                                     "'Predicate' class. Instead found {} in pattern {}" .format(pred, pat))
                if pat_elements is not None and not (isinstance(pat_elements, Iterable) and all(isinstance(e, tuple)
                                                                                              for e in pat_elements)):
                    raise ValueError("If specified, expect the second element of each predicate pattern tuple to be an "
                                     "Iterable of  tuples, each tuple represent a single pattern element. Instead "
                                     "found {}".format(pat_elements))
                if pat_elements is not None:
                    for ele in pat_elements:
                        if len(ele) != 2 and len(ele) != 3:
                            raise ValueError("Expect each pattern element to be either a size-2 or size-3 tuple. "
                                             "Instead found {}".format(ele))
                        if pat_type in [1, 2] and len(ele) == 3:
                            raise ValueError("Expect only size-2 element tuple in condact and/or action predicate "
                                             "patterns. Instead found {} in {}"
                                             .format(ele, "condact" if pat_type == 1 else "action"))
                        if len(ele) == 2:
                            arg_name, pat_var = ele
                        else:
                            arg_name, pat_var, trans = ele
                            if not isinstance(trans, VariableMap):
                                raise ValueError("If provided, the third entry of a pattern element should be of "
                                                 "'VariableMap' type, representing a transformation on the "
                                                 "corresponding pattern variable. Instead found {} in element tuple {}"
                                                 .format(type(trans), ele))
                        if not isinstance(arg_name, str):
                            raise ValueError("The first entry of a pattern element should be of 'str' type, "
                                             "representing an argument of the corresponding predicate. Instead found "
                                             "{} in element tuple {}".format(type(arg_name), ele))
                        if not isinstance(pat_var, str) and \
                                not (isinstance(pat_var, Iterable) and (all(isinstance(val, int) for val in pat_var)
                                                                    or all(isinstance(val, str) for val in pat_var))):
                            raise ValueError("The second entry of a pattern element must either be of 'str' type, "
                                             "representing a pattern variable, or an Iterable of 'int' or 'str', the "
                                             "former representing a list of constant discrete variable values while "
                                             "the latter representing a list of constant symbolic variable values. "
                                             "Instead found {} in element tuple {}".format(pat_var, ele))

        # Declare attributes
        self.name = intern_name(conditional_name, "conditional")
        self.conditions = conditions
        self.condacts = condacts
        self.actions = actions
        self.function_var_names = list(function_var_names) if function_var_names is not None else None
        self.function = function

        # Name the predicate patterns for indexing various lookup tables.
        # Map from pattern name to pattern tuple
        self.pat_name2pattern = {"pattern_" + str(i): pat for i, pat in enumerate(conditions + condacts + actions)}

        self.condition_name2pattern = {"pattern_" + str(i): pat for i, pat in enumerate(conditions)}
        self.condact_name2pattern = {"pattern_" + str(i + len(conditions)): pat
                                     for i, pat in enumerate(condacts)}
        self.action_name2pattern = {"pattern_" + str(i + len(conditions + condacts)): pat
                                    for i, pat in enumerate(actions)}

        # Map from pattern name to the corresponding predicate instance
        self.pat_name2pred = {pat_name: pat[0] for pat_name, pat in self.pat_name2pattern.items()}
        self.condition_name2pred = {pat_name: pat[0] for pat_name, pat in self.condition_name2pattern.items()}
        self.condact_name2pred = {pat_name: pat[0] for pat_name, pat in self.condact_name2pattern.items()}
        self.action_name2pred = {pat_name: pat[0] for pat_name, pat in self.action_name2pattern.items()}

        # List of pattern names
        self.pt_names = list(self.pat_name2pattern.keys())
        self.condition_names = list(self.condition_name2pattern.keys())
        self.condact_names = list(self.condact_name2pattern.keys())
        self.action_names = list(self.action_name2pattern.keys())

        # set up LOOKUP TABLES:
        # TODO: several things to be checked and set up at this stage:
        #       1. For each pattern:
        #               a. For each element:
        #                   - Check that element's arg agree with the predicate's argument.
        #                   - Prepend prefix to element's variable name. "VAR_" for pattern variable. "CONST_" for
        #                     constants.
        #                   - Check that trans are not declared on constant values
        #                !! - Check that element's trans agree with the argument's metatype. !! Left for future work
        #                   - Fill in 'pattern_arg2var'
        #               b. For the entire pattern:
        #                   - For undeclared predicate arguments, declare pattern variable with name "DEFAULT_{}"
        #                   - Fill in 'pattern_var2arg'
        #                   - Check that all arguments has been covered
        #                   - Check that each pattern variable is associated with arguments of the same metatype.
        #                   - Check that each Random variable is associated with arguments with the same type size
        #       2. Across the patterns:
        #               a. Gather pattern variables:
        #                   - Check that each pattern variable is associated with arguments of the same metatype.
        #                   - For relational pattern variables, determine its size by taking the max over the sizes of
        #                     all its associated predicate arguments.
        #                   - For random pattern variables, check that associated arguments' type sizes are the same
        #                   - For random pattern variables, gather all value constraints
        #                   - Generate variable instance and fill in 'rel_var_list', and 'ran_var_list'
        #               b. Check that 'function_var_names' agrees with 'ran_var_list'
        #               c. Check that predicate types are compatible with pattern type / pattern directionality
        #                   - At most one vector predicate among all condition and condact patterns that have a matching
        #                     random variable
        #                !! - If Factor function is deterministic, check pattern directionality
        #               d. Determine the global relational pattern variable order and corresponding joint relational
        #                  dimensions
        #
        # Map from predicate argument name to pattern variable info, for each predicate pattern
        #       pattern_arg2var = { pattern_name :
        #                               { pred_arg_name :
        #                                   { "name" : pt_var_name
        #                                     "const" : True/False
        #                                     "vals" : int/str values if const is True or None otherwise
        #                                     "map"  : VariableMap instance, if specified, otherwise None} } }
        # Map from pattern variable name back to predicate argument names, for each predicate pattern
        #       pattern_var2arg = { pattern_name :
        #                               { pt_var_name : list of corresponding predicate argument names } }
        # constant pattern is assigned a unique constant variable name
        self.pattern_arg2var = {}
        self.pattern_var2arg = {}
        self.rel_var_list, self.ran_var_list = [], []
        self.var_list = []      # Determines global pattern variable order
        self.dims = []          # Determines global joint relational dimension (i.e., joint dimensions of all relational
                                #   pattern variables in the order given by 'var_list'

        const_count = 0
        default_count = 0
        for pat_name, pat in self.pat_name2pattern.items():
            pred, elements = pat
            assert isinstance(pred, Predicate)
            self.pattern_arg2var[pat_name] = {}
            self.pattern_var2arg[pat_name] = {}
            if elements is not None:
                for ele in elements:
                    arg_name, pat_var = ele[0], ele[1]
                    var_map = ele[2] if len(ele) == 3 else None

                    # 1.a Check element's arg agree with predicate's arguments
                    if arg_name not in chain(pred.relarg_name2relarg.keys(), pred.ranarg_name2ranarg.keys()):
                        raise ValueError("Unknown argument '{}' declared in the predicate pattern '{}'. The list of "
                                         "declared arguments for predicate '{}' is {}"
                                         .format(arg_name, pat, pred.name, list(chain(pred.relarg_name2relarg.keys(),
                                                                                      pred.ranarg_name2ranarg.keys()))))

                    if isinstance(pat_var, str):        # A pattern variable
                        # 1.a. Prepend prefix to pattern variable's name and fill in look-up tables
                        pat_var = "VAR_" + pat_var
                        self.pattern_arg2var[pat_name][arg_name] = {
                            'name': pat_var,
                            'const': False,
                            'vals': None,
                            'map': var_map
                        }
                        # 1.b Fill in 'pattern_var2arg'
                        if pat_var not in self.pattern_var2arg[pat_name].keys():
                            self.pattern_var2arg[pat_name][pat_var] = [arg_name]
                        else:
                            self.pattern_var2arg[pat_name][pat_var].append(arg_name)
                    else:                               # Constant
                        pat_var_name = "CONST_" + str(const_count)
                        const_count += 1

                        if var_map is not None:
                            raise ValueError("Transformation (VariableMap) should not be declared on constant pattern "
                                             "variable. Found {} in element {}".format(var_map, ele))

                        self.pattern_arg2var[pat_name][arg_name] = {
                            'name': pat_var_name,
                            'const': True,
                            'vals': pat_var,
                            'map': None
                        }
                        self.pattern_var2arg[pat_name][pat_var_name] = [arg_name]
                    # TODO: 1.a. Check that element's trans agree with argument's metatype.

            # 1.b Across the entire pattern, find undeclared arguments, and declare with default name
            for arg_name in chain(pred.relarg_name2relarg.keys(), pred.ranarg_name2ranarg.keys()):
                if arg_name not in self.pattern_arg2var[pat_name].keys():
                    pat_var = "DEFAULT_" + str(default_count)
                    default_count += 1
                    self.pattern_arg2var[pat_name][arg_name] = {
                        'name': pat_var,
                        'const': False,
                        'vals': None,
                        'map': None
                    }
                    # 1.b Fill in 'pattern_var2arg'
                    self.pattern_var2arg[pat_name][pat_var] = [arg_name]

            # 1.b Check that all arguments has been covered
            assert set(self.pattern_arg2var[pat_name].keys()) == \
                   set(chain(pred.relarg_name2relarg.keys(), pred.ranarg_name2ranarg.keys()))

            for pat_var, args in self.pattern_var2arg[pat_name].items():
                # 1.b Check that each pattern variable is associated with arguments of the same metatype
                if not all(arg_name in pred.relarg_name2relarg.keys() for arg_name in args) and \
                        not all(arg_name in pred.ranarg_name2ranarg.keys() for arg_name in args):
                    raise ValueError("Expect all predicate arguments associated with a pattern variable to be of the "
                                     "same metatype. However, for the pattern variable '{}' in pattern '{}', its "
                                     "associated arguments {} has metatypes {}"
                                     .format(pat_var, pat, args, list(pred.arg_name2metatype[arg] for arg in args)))

                # 1.b Check that each Random variable is associated with arguments with the same type size
                if pred.arg_name2metatype[args[0]] == VariableMetatype.Random:
                    sizes = list(pred.ranarg_name2type[arg].size for arg in args)
                    if min(sizes) != max(sizes):
                        raise ValueError("Expect all predicate arguments associated with a Random metatype pattern "
                                         "variable to have the same type size. Instead, found type sizes '{}' "
                                         "associated with arguments '{}'"
                                         .format(sizes, args))

        # 2.a Gather pattern variables globally
        all_pat_var_names = set()
        for entry in self.pattern_var2arg.values():
            all_pat_var_names = all_pat_var_names.union(set(entry.keys()))
        for pat_var_name in all_pat_var_names:
            metatype = None
            size = 0         # find max if metatype is Relational, make sure same if metatype is Random
            constraint_set = set()    # to be filled if metatype is Random
            for pat_name, entry in self.pattern_var2arg.items():
                if pat_var_name not in entry.keys():
                    continue
                pat, pred = self.pat_name2pattern[pat_name], self.pat_name2pred[pat_name]
                args = entry[pat_var_name]

                # 2.b Check each pattern variable is associated with arguments of the same metatype across patterns
                if metatype is None:
                    metatype = pred.arg_name2metatype[args[0]]
                elif metatype is not pred.arg_name2metatype[args[0]]:
                    raise ValueError("A pattern variable should be associated with predicate arguments of the same "
                                         "metatype across patterns. Found pattern variable '{}' associated with "
                                         "predicate argument '{}' with metatype '{}', when the metatype inferred from "
                                         "other arguments is '{}"
                                         .format(pat_var_name, args[0], pred.arg_name2metatype[args[0]], metatype))

                # 2.b For relational variables, find maximal type size
                if metatype is VariableMetatype.Relational:
                    tmp = max(pred.relarg_name2type[arg].size for arg in args)
                    size = max(tmp, size)

                # 2.b For random variables, make sure the type of associated arguments is the same
                if metatype is VariableMetatype.Random:
                    if size == 0:
                        size = pred.ranarg_name2type[args[0]].size
                    elif size != pred.ranarg_name2type[args[0]].size:
                        raise ValueError("A Random metatype pattern variable should be associated with predicate "
                                         "arguments of the same type size across patterns. Found pattern variable '{}' "
                                         "associated with predicate argument '{}' of type size '{}', while the correct "
                                         "type size inferred from other arguments is '{}'"
                                         .format(pat_var_name, args[0], pred.ranarg_name2type[args[0]].size, size))
                    constraint_set = constraint_set.union(set(pred.ranarg_name2type[arg].constraint for arg in args))

            # Generate Variable instance and add to lists
            pat_var = Variable(pat_var_name, metatype, size,
                               constraint_set if metatype is VariableMetatype.Random else None)
            if metatype is VariableMetatype.Relational:
                self.rel_var_list.append(pat_var)
            else:
                self.ran_var_list.append(pat_var)

        # 2.b Check that 'function_var_names' agrees with 'ran_var_list'
        if function_var_names is not None:
            for ran_var in function_var_names:
                if ran_var not in self.ran_var_list:
                    raise ValueError("Unknown random pattern variable '{}' in 'function_var_names'. Existing declared "
                                     "random pattern variables are: {}".format(ran_var, self.ran_var_list))

        # 2.c Check that predicate types are compatible with pattern type / pattern directionality
        for ran_var in self.ran_var_list:
            # 2.c. At most one vector predicate among all condition and condact patterns that have a matching random
            #      variable
            vec_pred = None
            for pat_name in self.condition_names + self.condact_names:
                if ran_var.name in self.pattern_var2arg[pat_name].keys() and \
                        self.pat_name2pred[pat_name].dist_class == None:
                    if vec_pred is not None:
                        raise ValueError("Expect at most one vector predicate among all condition and condact patterns "
                                         "that have a matching random variable. Instead found at least two vector "
                                         "predicates: '{}' and '{}' that matches on the random pattern variable '{}' "
                                         .format(vec_pred, self.pat_name2pred[pat_name], ran_var))
                    vec_pred = self.pat_name2pred[pat_name]

        # TODO: 2.c Check pattern directionality for deterministic factor function

        # 2.d Determine the global relational pattern variable order and corresponding joint relational dimensions
        self.rel_dims = Size(pat_var.size for pat_var in self.rel_var_list)

    def __str__(self):
        # String representation for display
        return self.name

    def __hash__(self):
        # Hash by structure's own name. This assumes its name is unique in the model and therefore can be treated as
        #   the identity
        return hash(self.name)

