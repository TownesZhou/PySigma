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
                "pattern element" is a size-3 tuple. The first element of such tuple is of 'str' type, corresponds to
                one of the predicate's argument. The second element is either a 'str' (representing a predicate
                variable), an Iterable of 'int' (list of constant integer values), or an Iterable of 'str' (also
                constant but matches symbolic values). Finally, the third element is of 'PatternTransformation' type,
                representing a transformation on the predicate variable's values.

            In short, a predicate pattern should look like:
                (pred_name, [(arg_1, var_1, trans_1), (arg_2, var_2, trans_2), ..., (arg_n, var_n, trans_n)])

            Note that for condact and action predicate patterns, it is expected that no transformation is declared on
                any of the variables, i.e., the third element of the above "pattern element" tuple is left as None.

            The 'function' field specifies a factor function that semantically operates on the events represented by
                random variables from the incoming messages. Therefore, only random variables are visible to the factor
                function, and the dimensions corresponding to relational variables should be treated as the "batch
                dimensions" of the incoming messages.

        :param conditional_name: 'str' type. The name of this conditional
        :param conditions: an Iterable of size-2 tuples.
        :param condacts: same as 'conditions'
        :param actions: same as 'conditions'
        :param function: 'FactorFunction' type. Declares a factor function at this conditional.
        :param function_var_names: an Iterable of 'str'. The list of random variables that the factor function concerns.
                                   The order of the random variables will be respected in accordance to the order given
                                   by the supplied iterable. The list of messages will be given to the factor function
                                   w.r.t. this order at inference time.
        """
        # TODO: Implement new Conditional

        # Check conditions, actions, & condacts
        if not isinstance(conditional_name, str):
            raise ValueError("1st argument 'conditional_name' of a Conditional must be a 'str'")
        if conditions is not None and \
                (not isinstance(conditions, Iterable) or not all(isinstance(p, PredicatePattern) for p in conditions)):
            raise ValueError("If not None, 2nd argument 'conditions' of a Conditional must be an iterable of "
                             "'PredicatePattern's")
        if condacts is not None and \
                (not isinstance(condacts, Iterable) or not all(isinstance(p, PredicatePattern) for p in condacts)):
            raise ValueError("If not None, 3rd argument 'condacts' of a Conditional must be an iterable of "
                             "'PredicatePattern's")
        if actions is not None and \
                (not isinstance(actions, Iterable) or not all(isinstance(p, PredicatePattern) for p in actions)):
            raise ValueError("If not None, 4th argument 'actions' of a Conditional must be an iterable of "
                             "'PredicatePattern's")

        if [conditions, actions, condacts] == [None, None, None]:
            raise ValueError("Cannot specify an empty conditional")

        if conditions is not None:
            if [actions, condacts] == [None, None]:
                raise ValueError("Cannot specify a conditional that consists of only conditions")

        # Check rest of the arguments
        if function_var_names is not None and \
                (not isinstance(function_var_names, Iterable) or not all(
                    isinstance(v, str) for v in function_var_names)):
            raise ValueError(
                "If not None, argument 'function_var_names' of a Conditional must be an iterable of 'str's")

        if function is not None and not isinstance(function, (int, float, torch.Tensor, str)):
            raise ValueError("If not None, argument 'function' of a Conditional must be 'int', 'float', 'torch.Tensor',"
                             " or 'str'")
        if normal is not None and \
                (not isinstance(normal, Iterable) or not all(isinstance(v, str) for v in normal)):
            raise ValueError("If not None, argument 'normal' of a Conditional must be an iterable of 'str's")

        conditions = [] if conditions is None else list(conditions)
        condacts = [] if condacts is None else list(condacts)
        actions = [] if actions is None else list(actions)

        # Change pattern's predicate names into internal names
        #   Note that since Python's namedtuple is immutable, need to recreate namedtuple here. Maybe better way to do this?
        for pt in conditions + condacts + actions:
            pt.predicate_name = intern_name(pt.predicate_name, "predicate")

        # Name the predicate patterns for indexing various lookup tables
        self.name2pattern = {"pattern_" + str(i): pattern for i, pattern in enumerate(conditions + condacts + actions)}
        self.name2condition_pattern = {"pattern_" + str(i): pattern for i, pattern in enumerate(conditions)}
        self.name2condact_pattern = {"pattern_" + str(i + len(conditions)): pattern for i, pattern in
                                     enumerate(condacts)}
        self.name2action_pattern = {"pattern_" + str(i + len(conditions) + len(condacts)): pattern for i, pattern in
                                    enumerate(actions)}
        self.pt_names = list(self.name2pattern.keys())
        self.condition_pt_names = list(self.name2condition_pattern.keys())
        self.condact_pt_names = list(self.name2condact_pattern.keys())
        self.action_pt_names = list(self.name2action_pattern.keys())

        # Set up pattern var list for future lookup
        # Set up internal WM var -- pattern var map per predicate pattern for future lookup.
        #       pattern_pt_vals = { pattern_name :
        #                            { wm_var_name :
        #                               { "name" : pt_var_name
        #                                 "type" : "var" or "const"
        #                                 "vals" : int/str values if type is const or None otherwise
        #                                 "rel"  : relation, if specified, otherwise None} } }
        # Map from pattern variable to wm variables that is associated with it
        #       ptv2wmv = { pattern_name :
        #                    { pt_var_name : list of corresponding wm vars } }
        # Global dictionary of pattern variable info. To be registered once passed into a Sigma program (because need to
        #   look up various Type sizes
        #   For the size field, take the max over the size of all wm vars associated with it if it is of 'var' type.
        #       Else if 'const' type, take the number of elements / values in 'vals' field.
        #       global_pt_vals = { pt_var_name :
        #                           { "type" : "var" or "const",
        #                             "size" : max size over associated wm vars
        #                             "sum_op": Which summarization operation to take in beta branching subnet
        #                                       currently support "sum" and "max"
        #                             # TODO: extend this to support more summarization operations
        #                             } }
        # constant pattern is assigned a unique constant variable name
        self.ptvar_list = []
        self.pattern_pt_vals = {}
        self.global_pt_vals = {}  # To be filled after passed into a Sigma program
        self.ptv2wmv = {}
        const_count = 0

        for pt_name, pattern in self.name2pattern.items():
            self.pattern_pt_vals[pt_name] = {}
            self.ptv2wmv[pt_name] = {}
            for element in pattern.elements:
                pt_var_info = {}
                if type(element.value) is PatternVariable:
                    pt_var_info["name"] = element.value.variable_name
                    pt_var_info["type"] = "var"
                    pt_var_info["vals"] = None
                    pt_var_info["rel"] = element.value.relation
                else:
                    # Assign a unique internal pattern variable name to a pattern w/ constant values
                    pt_var_info["name"] = "_CONST_" + str(const_count)
                    pt_var_info["type"] = "const"
                    pt_var_info["vals"] = element.value
                    pt_var_info["rel"] = None
                    const_count += 1

                if pt_var_info["name"] not in self.ptvar_list:
                    self.ptvar_list.append(pt_var_info["name"])
                self.pattern_pt_vals[pt_name][element.argument_name] = pt_var_info
                if pt_var_info["name"] not in self.ptv2wmv.keys():
                    self.ptv2wmv[pt_name][pt_var_info["name"]] = [element.argument_name]
                else:
                    self.ptv2wmv[pt_name][pt_var_info["name"]].append(element.argument_name)

        # Check that function_var_names agree with pattern variables declared in conditions, actions, & condacts
        if function_var_names is not None:
            for func_var in function_var_names:
                if func_var not in self.ptvar_list:
                    raise ValueError("The function variable '{}' is not one of the pattern variable declared in any "
                                     "predicate pattern".format(func_var))

        self.name = intern_name(conditional_name, "conditional")
        self.conditions = conditions
        self.condacts = condacts
        self.actions = actions
        self.function_var_names = [var for var in function_var_names] if function_var_names is not None else None
        self.normal = normal
        self.function = function
