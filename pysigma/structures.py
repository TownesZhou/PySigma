"""
    Common structures and componenets serving the Sigma cognitive language

    Author of this file: Jincheng Zhou, University of Southern California
"""

from torch import Size
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint
from collections.abc import Iterable
from itertools import chain
from .utils import *
from defs import Variable, VariableMetatype


class VariableMap:
    """
        Class type for transformation on predicate's variables in pattern elements
    """
    pass


class FactorFunction:
    """
        Class type for factor node function

        A FactorFunction instance can be thought of as a function object that takes in a set of batched tensors, each
            corresponding to a batched value assignments to a random variable, and produces a single batched tensor that
            represents the joint probability density, i.e.,
                    val = Prob(X_1, X_2, ..., X_n)
            where val, X_1, X_2, ..., X_n are all batched over the first dimension.

        The first dimension of all tensors is the result of the particle indexing dimension and all relational
            variables' dimensions being reshaped into one. Therefore, it should be ignored by the factor function
            procedure and simply treated as the "batched" dimension.

        Defines different types of factor functions:
            a. Tabular factor function
                Returns a full-dimension tensor at once representing the entire factor function table. Suitable when all
                random variables X_1, ..., X_n have finite discrete domains.
                Corresponds to legacy Lisp Sigma's factor node function

            b. General form generative joint-probability density function
                Returns a tensor representing the probability density each time given a set of value assignments to all
                random variables. Enforce the semantics of
                    val = Prob(X_1, X_2, ..., X_n)
                Compatible with all particle-based inference methods

            c. Exponential form joint-probability density function
                Explicitly defines an exponential distribution. Encodes a exponential distribution class as the prior
                and returns the PARAMETERS
                Must declare this type if want architecture to recognize conjugate-exponential model structure and
                carries out closed-form message updates

            d. Deterministic factor function
                Returns one or multiple VALUE tensors corresponding to one or multiple R.V.s given a value assignments
                to other variables. Enforce the semantics of
                    Y_1, Y_2, ..., Y_m = Func(X_1, X_2, ..., X_n)
                Conceptually identical to (b) type factor function with a delta distribution, but due to concerns of
                sample efficiency, this type should be used in practice
                Note that once defined, the directionality of the corresponding Conditional is also assumed and fixed.
                In other words, X_1, ..., X_n should only appear in condition patterns, and Y_1, ..., Y_m only in action
                patterns
    """
    pass


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

            rel_var = Variable(arg_name, VariableMetatype.Relational, arg_type.size)
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

            ran_var = Variable(arg_name, VariableMetatype.Random, arg_type.size)
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
                         sum([v.size for v in self.random_args]))

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
        #                                     "map"  : transformation, if specified, otherwise None} } }
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

        # 2.a Gather patern variables globally
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

                # 2.b Check each pattern varaible is associated with arguments of the same metatype across patterns
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

