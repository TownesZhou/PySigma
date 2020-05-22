"""
    Common structures and componenets serving the Sigma cognitive language

    Author of this file: Jincheng Zhou, University of Southern California
"""

from torch.distributions import Distribution
from torch import Size
from collections import namedtuple
from collections.abc import Iterable
from itertools import chain
from .utils import *
from .graphical._defs import Variable, VariableMetatype


# class PredPattern:
#     """
#         Predicate Pattern in a Conditional
#     """
#
#     def __init__(self, predicate_name, elements):
#         """
#             Declare a Predicate Pattern in a Conditional
#         :param predicate_name:      'str'. name of the predicate in question
#         :param elements:            Iterable of 'PatternElement'.
#         """
#         if not isinstance(predicate_name, str):
#             raise ValueError("1st argument 'predicate_name' of a PredicatePattern must be a 'str'")
#         if not isinstance(elements, Iterable) or not all(isinstance(e, PatternElement) for e in elements):
#             raise ValueError("2nd argument 'elements' of a PredicatePattern must be an iterable of 'PatternElement's")
#         if not isinstance(negation, bool):
#             raise ValueError("the argument 'negation' of a PredicatePattern must be a 'bool'")
#         if nonlinearity is not None and nonlinearity not in ['negation', '-', 'exponential', '^', 'sigmoid', 's',
#                                                              'relu', 'r', 'tanh', 't', 'exp', 'e']:
#             raise ValueError("Unknown nonlinearity: '{}'".format(nonlinearity))
#         # TODO: allow customized nonlinearity declaration in future iterations
#
#         self.predicate_name = predicate_name
#         self.elements = list(elements)
#         self.negation = negation
#         self.nonlinearity = nonlinearity
#
#
# class PatternElement:
#     """
#         Element in a PredicatePattern
#           Each element is of the f
#                   - A Constant Region, which is:      # TODO: to implement in v1
#                       - An `int` if the variable is discrete, or an 'str' (a symbol) in this variable's symbol list if the
#                               variable is symbolic.
#                       - A iterable of `int` (if discrete) or 'str' (if symbolic) within the value range of this variable. This
#                               would yield a list of intervals at the specified values in the given list.
#                       - None: This would yield the entire scope of the dimension
#                   - A `Filter` or a `list` of `Filter`s. # TODO: to implement in future iterations
#                   - A `PatternVariable`.              # TODO: to implement in v1
#     """
#
#     def __init__(self, argument_name, value=None):
#         if not isinstance(argument_name, str):
#             raise ValueError("1st argument 'argument_name' of a PatternElement must be a 'str'")
#         if value is not None and not isinstance(value, (int, str, Iterable, Filter, PatternVariable)):
#             raise ValueError("If not None, 2nd argument 'value' of a PatternElement must be an 'int', 'str', "
#                              "an iterable of 'int' or 'str', a 'Filter', or a 'PatternVariable'")
#         if isinstance(value, Iterable) and \
#                 not (all(isinstance(v, int) for v in value) or
#                      all(isinstance(v, str) for v in value) or
#                      all(isinstance(v, Filter) for v in value)):
#             raise ValueError("If an iterable, the 2nd argument 'value' of a PatternElement must be an iterable "
#                              "of 'int', 'str', or 'Filter's")
#
#         self.argument_name = argument_name
#         if isinstance(value, PatternVariable):
#             self.value = value
#         else:
#             self.value = list(value) if isinstance(value, Iterable) else [value]    # turn into list anyway
#
#
# class PatternVariable:
#     """
#         Pattern Variable to be defined in a predicate pattern
#           Each pattern variable is of the form `(variable_name, relation)`
#               - `variable_name`: `str` type. The name of the pattern variable. This is to be distinguished from the variable's
#                       working memory variable, or `argument_name`. These are the actual variables used to determine various
#                       forms of variable bindings in a conditional. Bindings are assumed when same pattern variable is referred
#                       in multiple places in a conditional. It's like the distinction between the actual argument and the
#                       formal argument in a procedural programming language.
#               - `relation`: an optional parameter
#                   - `None`: default value
#                   - an `int`: declaring offset
#                   - an `Affine`: declaring an affine transformation
#                   - a `Filter` or an iterable of `Filter`s.    # TODO: to implement in future iterations
#                   TODO: The `not-equal test` and "highly experimental" `:explicit` are not included here.
#     """
#
#     def __init__(self, variable_name, relation=None):
#         if not isinstance(variable_name, str):
#             raise ValueError("The 1st argument 'variable_name' of a PatternVariable must be a 'str'")
#         if variable_name[0] == '_':
#             raise ValueError("The 1st argument 'variable_name' of a PatternVariable cannot start with '_'")
#         if relation is not None and not isinstance(relation, (int, Affine, Filter, Iterable)):
#             raise ValueError("If not None, the 2nd argument 'relation' of a PatternVariable must be an 'int', "
#                              "an 'Affine', a 'Filter', or an iterable of 'Filter's")
#         if isinstance(relation, Iterable) and not all(isinstance(r, Filter) for r in relation):
#             raise ValueError("If an iterable, the 2bd argument 'relation' of a PatternVariable must be an iterable "
#                              "of Filters")
#
#         self.variable_name = variable_name
#         self.relation = list(relation) if isinstance(relation, Iterable) else relation
#
#
# class Affine:
#     """
#         Affine transformation to be defined alongside a pattern variable
#           Each affine transformation is of the form `(from, offset, coefficient, pad)`
#               - `from`: default to the current variable.
#               - `offset`: default to 0
#               - `coefficient`: default to 1
#               - `pad`: 0 for closed-world predicates and 1 for open-world predicates
#         TODO: Define a more generalized notion of 'mapping' in place of Affine in future iterations
#     """
#
#     def __init__(self, from_var=None, offset=0, coefficient=1, pad=0):
#         if from_var is not None and not isinstance(from_var, str):
#             raise ValueError("If not None, The 1st argument 'from_var' of an Affine must be a 'str'")
#         if not isinstance(offset, int):
#             raise ValueError("The 2nd argument 'offset' of an Affine must be an 'int'")
#         if not isinstance(coefficient, (int, float)):
#             raise ValueError("The 3rd argument 'coefficient' of an Affine must be an 'int' or a 'float'")
#         if not isinstance(pad, int):
#             raise ValueError("The 4th argument 'pad' of an Affine must be an 'int'")
#
#         self.from_var = from_var
#         self.offset = offset
#         self.coefficient = coefficient
#         self.pad = pad
#
#
# # TODO: Filters NOT TO IMPLEMENT IN SHORT TERM
# # Filter to be defined in a PredicatePattern
# #   Each filter is of the form `(constant_region, constant, coefficient)`
# Filter = namedtuple('Filter', ['constant_region', 'constant', 'coefficient'])


class VariableMap:
    """
        Class type for transformation on predicate's variables in pattern elements
    """
    pass


class FactorFunction:
    """
        Class type for factor node function
    """
    pass


class Type:
    def __init__(self, type_name, symbolic=False, size=None, symbol_list=None):
        """
            Specify a Type for a predicate's relational/random argument. Can be symbolic or discrete. Currently, Type
                structure is only applicable to relational and random arguments, and specifications of particle indexing
                arguments are declared at the Predicate's level.

            For both relational and random arguments, a Type mainly declares the size of the corresponding tensor
                dimension of the predicate's particle messages.

            If symbolic, must declare a 'symbol_list', and the 'size' field will be inferred from the length of the
                'symbol_list'. Otherwise must declare the 'size' value.

        :param type_name: 'str' type. Name of the Sigma type
        :param symbolic: 'bool' type. Declares if the type comes with a symbol list
        :param size: 'int' type. The size of the argument's corresponding message dimension. Must specify if 'symbolic'
                     is False
        :param symbol_list: An iterable of 'str' representing symbols. Must specify if 'symbolic' is True
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
        if not isinstance(size, int) or size < 1:
            raise ValueError("'size' must be of type 'int' and value be greater than 0")
        if not isinstance(symbol_list, Iterable) or not all(isinstance(s, str) for s in symbol_list):
            raise ValueError("'symbol_list' must be an iterable of 'str'")

        self.name = intern_name(type_name, "type")  # Prepend substring 'TYPE_' and send name to upper case
        self.symbolic = symbolic

        self.value_list = list(symbol_list) if self.symbolic else list(range(size))
        self.size = len(self.value_list)

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


class Predicate:
    def __init__(self, predicate_name, relational_args, random_args, num_particles=128, distribution_class=None,
                 memorial=False, perceptual=False):
        """
            Specify a Sigma predicate.

        :param predicate_name: 'str' type. The name of the predicate.
        :param relational_args: an Iterable of size-2 tuples. Each tuple specifies one relational argument - The first
                                element of the tuple should be of 'str' type specifying the argument's name, whereas
                                the second element should be of 'Type' type specifying the argument's type. The 'Type'
                                of these arguments can be arbitrary, but the arguments' names should be distinct.
        :param random_args: an Iterable of size-2 tuples, similar to the previous one. However for random arguments,
                            besides the arguments' names having to be distinct (including being distinct from relational
                            arguments' names), all of the arguments must be of the same 'Type' and the 'Type' must NOT
                            be symbolic.
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
        if num_particles is not None and not (isinstance(num_particles, int) and num_particles > 0):
            raise ValueError("If specified, the 4th field 'num_particles' must be a positive integer")
        if distribution_class is not None and not issubclass(distribution_class, Distribution):
            raise ValueError("If specified, the 5th field 'distribution' must provide a subclass of 'torch."
                             "distributions.Distribution")
        if not isinstance(memorial, bool):
            raise ValueError("The 6th field 'memorial' must be of 'bool' type")
        if not isinstance(perceptual, bool):
            raise ValueError("The 7th field 'perception' must be of 'bool' type")

        self.predicate_name = intern_name(predicate_name, "predicate")

        self.relational_vars = []
        self.relvar_name2relvar = {}
        self.relvar_name2type = {}
        for rel_arg in relational_args:
            var_name = rel_arg[0]
            var_type = rel_arg[1]
            if not isinstance(var_name, str) or not isinstance(var_type, Type):
                raise ValueError("The first element of an argument tuple must be of 'str' type to declare the "
                                 "argument's name, and the second one be an instance of 'Type'. Instead, found: "
                                 "({}, {})".format(type(var_name), type(var_type)))
            if var_name in self.relvar_name2relvar.keys():
                raise ValueError("Relational arguments' names must be distinct. Instead found repetition: {}"
                                 .format(var_name))
            rel_var = Variable(var_name, VariableMetatype.Relational, var_type.size)
            self.relational_vars.append(rel_var)
            self.relvar_name2relvar[var_name] = rel_var
            self.relvar_name2type[var_name] = var_type

        self.random_vars = []
        self.ranvar_name2ranvar = {}
        self.ranvar_type = None
        for ran_arg in random_args:
            var_name = ran_arg[0]
            var_type = ran_arg[1]
            if not isinstance(var_name, str) or not isinstance(var_type, Type):
                raise ValueError("The first element of an argument tuple must be of 'str' type to declare the "
                                 "argument's name, and the second one be an instance of 'Type'. Instead, found: "
                                 "({}, {})".format(type(var_name), type(var_type)))
            if var_name in chain(self.relvar_name2relvar.keys(), self.ranvar_name2ranvar.keys()):
                raise ValueError("Random arguments' names must be distinct. Instead found repetition: {}"
                                 .format(var_name))
            if var_type.symbolic:
                raise ValueError("Random argument's type cannot be symbolic")
            if self.ranvar_type is None:
                self.ranvar_type = var_type
            if var_type != self.ranvar_type:
                raise ValueError("Random arguments must all be of the same Type")
            ran_var = Variable(var_name, VariableMetatype.Random, var_type.size)
            self.random_vars.append(ran_var)
            self.ranvar_name2ranvar[var_name] = ran_var

        self.index_var = Variable("INDEX", VariableMetatype.Indexing, num_particles)
        self.dist_class = distribution_class
        self.memorial = memorial
        self.perceptual = perceptual


class Conditional:
    def __init__(self, conditional_name, conditions=None, condacts=None, actions=None,
                 function=None, function_var_names=None):
        """
            Specify a Sigma conditional

            Each of conditions, condacts, and actions field consists of an Iterable of "predicate patterns", where each
                "predicate pattern" is represented by a size-2 tuple. The first element of the tuple is of 'str' type,
                representing the predicate's name. The second element is an Iterbale of "pattern elements", where each
                "pattern element" is a size-2 or size-3 tuple. The first element of such tuple is of 'str' type,
                corresponds to one of the predicate's argument. The second element is either a 'str' (representing a
                pattern variable), an Iterable of 'int' (list of constant integer values), or an Iterable of 'str'
                (also constant but matches symbolic values). Finally, the third element is OPTIONAL, of
                'PatternTransformation' type, representing a transformation on the predicate variable's values.

            In short, a predicate pattern should look like:
                    (pred_name, [(arg_1, var_1, trans_1), (arg_2, var_2, trans_2), ..., (arg_n, var_n, trans_n)])

            The purpose of declaring pattern variables is to match predicate arguments within or across predicates,
                i.e., variable binding, or (only for random arguments) to be picked up and recognized by the factor
                function. Therefore, if matching is not necessary for a predicate argument, or if it is ignored by the
                factor function, its corresponding pattern element can be left empty, and the architecture will declare
                a default pattern variable that is uniquely associated this predicate argument.
            Thus, it is acceptable to declare the following pattern:
                    (pred_name, None)
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
                pred_name, pat_elements = pat
                if not isinstance(pred_name, str):
                    raise ValueError("Expect the first element of each predicate pattern tuple to be of 'str' type, "
                                     "representing the name of a predicate. Instead found {} in pattern {}"
                                     .format(pred_name, pat))
                if pat_elements is not None and not isinstance(pat_elements, Iterable) or all(isinstance(e, tuple)
                                                                                              for e in pat_elements):
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

        # Name the predicate patterns for indexing various lookup tables.
        # Map from pattern name to (internalized) predicate name
        self.pat_name2pattern = {"pattern_" + str(i): pat for i, pat in enumerate(conditions + condacts + actions)}

        self.condition_name2pattern = {"pattern_" + str(i): pat for i, pat in enumerate(conditions)}
        self.condact_name2pattern = {"pattern_" + str(i + len(conditions)): pat
                                     for i, pat in enumerate(condacts)}
        self.action_name2pattern = {"pattern_" + str(i + len(conditions + condacts)): pat
                                    for i, pat in enumerate(actions)}

        self.pat_name2pred_name = {pat_name: intern_name(pat[0], "predicate")
                                   for pat_name, pat in self.pat_name2pattern.items()}
        self.condition_name2pred_name = {pat_name: intern_name(pat[0], "predicate")
                                         for pat_name, pat in self.condition_name2pattern.items()}
        self.condact_name2pred_name = {pat_name: intern_name(pat[0], "predicate")
                                       for pat_name, pat in self.condact_name2pattern.items()}
        self.action_name2pred_name = {pat_name: intern_name(pat[0], "predicate")
                                      for pat_name, pat in self.action_name2pattern.items()}

        self.pt_names = list(self.pat_name2pattern.keys())
        self.condition_names = list(self.condition_name2pattern.keys())
        self.condact_names = list(self.condact_name2pattern.keys())
        self.action_names = list(self.action_name2pattern.keys())

        # set up LOOKUP TABLES:
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
        # Global dictionary of pattern variable info. To be registered once passed into a Sigma program (because need to
        #   look up various Type sizes
        #   For the size field, if the corresponding predicate arguments are relational, then take the max over the size
        #       of all corresponding predicate arguments. Otherwise if random, make sure all corresponding predicate
        #       arguments have the same size. If the pattern variable is a constant, then take as its size the number
        #       of constant values.
        #       global_var_info = { pt_var_name :
        #                               { "metatype" : VariableMetatype field
        #                                 "const" : True/False
        #                                 "size" : max size over associated wm vars
        #                               } }
        # constant pattern is assigned a unique constant variable name
        self.pattern_arg2var = {}
        self.pattern_var2arg = {}
        self.global_var_info = {}       # Fill in when registered in a Sigma model
        self.rel_var_list, self.ran_var_list = [], []   # Fill in when registered in a Sigma model

        const_count = 0
        for pat_name, pat in self.pat_name2pattern.items():
            self.pattern_arg2var[pat_name] = {}
            self.pattern_var2arg[pat_name] = {}
            if pat[1] is not None:
                for ele in pat[1]:
                    arg_name, pat_var = ele[0], ele[1]
                    var_map = ele[2] if len(ele) == 3 else None
                    if isinstance(pat_var, str):        # A pattern variable
                        self.pattern_arg2var[pat_name][arg_name] = {
                            'name': pat_var,
                            'const': False,
                            'vals': None,
                            'map': var_map
                        }
                        if pat_var not in self.pattern_var2arg[pat_name].keys():
                            self.pattern_var2arg[pat_name][pat_var] = [arg_name]
                        else:
                            self.pattern_var2arg[pat_name][pat_var].append(arg_name)
                    else:                               # Constant
                        self.pattern_arg2var[pat_name][arg_name] = {
                            'name': "_CONST_" + str(const_count),
                            'const': True,
                            'vals': pat_var,
                            'map': var_map
                        }
                        const_count += 1

        self.name = intern_name(conditional_name, "conditional")
        self.conditions = conditions
        self.condacts = condacts
        self.actions = actions
        self.function_var_names = list(function_var_names) if function_var_names is not None else None
        self.function = function
