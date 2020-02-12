"""
    Common structures and componenets serving the Sigma cognitive language

    Author of this file: Jincheng Zhou, University of Southern California
"""

import torch
from collections import namedtuple


# Define (sub)structures using namedtuple or customized class

# Argument in a Predicate
#   Each Argument namedtuple is a tuple of two or three elements of the form `(argument_name, type)` or
#    `(argument_name, type, unique_symbol)`
#       - `argument_name`:  `str` type. the name of the working memory variable
#       - `type`:  `str` or `Type` type. The `Type` of the working memory variable
#       - `unique_symbol`:  `None` or `str` type. Default to None
#           - `'!'`: Select best
#           - `'%'`: Maintain distribution
#           - `'$'`: Select expected value
#           - `'^'`: Maintain exponential transform of distribution
#           - `'='`: Select by probability matching
#           **NOTE**: `unique_symbol` must be specified if to declare this variable to be unique. Otherwise will
#                 be treated as a universal variable. You can use `%` to specify that the distribution should not be
#                 changed during selection phase.
PredicateArgument = namedtuple('PredicateArgument', ['argument_name', 'type', 'unique_symbol'], defaults=[None])

# Predicate Pattern in a Conditional
#   Each predicate pattern is of the form `(predicate_name, nonlinearity, 'elements')
#       - `predicate_name`:  name of the predicate in question
#       - `nonlinearity`: `None` or one of the following: `'negation'` or `'-'`, `'exponential'` or `'^'`, `'sigmoid'`
#               or `'s'`, `'relu'` or `'r'`, `'tanh'` or `'t'`, `'exp'` or `'e'` (true exponential),
#               or customized torch nonlinearity functions
#       - `elements`: `list` type. A list of `PatternElement` namedtuples.
# TODO: Finish construction of following substructure and their docs
PredicatePattern = namedtuple('PredicatePattern', ['predicate_name', 'nonlinearity', 'elements'])

# Element in a PredicatePattern
#   Each element is of the f
#           - A Constant Region, which is:      # TODO: to implement in v1
#               - An `int` if the variable is discrete, or an object (a symbol) in this variable's symbol list if the
#                       variable is symbolic.
#               - A `list` of `int` (if discrete) or symbols (if symbolic) within the value range of this variable. This
#                       would yield a list of intervals at the specified values in the given list.
#               - `*`. This would yield the entire scope of the dimension
#           - A `Filter` or a `list` of `Filter`s. # TODO: to implement in future iterations
#           - A `PatternVariable`.              # TODO: to implement in v1
PatternElement = namedtuple('PatternElement', ['argument_name', 'value'])

# Pattern Variable to be defined in a predicate pattern
#   Each pattern variable is of the form `(variable_name, relation)`
#       - `variable_name`: `str` type. The name of the pattern variable. This is to be distinguished from the variable's
#               working memory variable, or `argument_name`. These are the actual variables used to determine various
#               forms of variable bindings in a conditional. Bindings are assumed when same pattern variable is referred
#               in multiple places in a conditional. It's like the distinction between the actual argument and the
#               formal argument in a procedural programming language.
#       - `relation`: an optional parameter
#           - `None`: default value
#           - an `int`: declaring offset
#           - an `Affine`: declaring an affine transformation
#           - a `Filter` or a list of `Filter`s.    # TODO: to implement in future iterations
#           # TODO: Discuss with Volkan about possible redefinition of Affine and Filter semantics.
#               E.g. should only allow affine declaration in action predicate patterns to better conform to Horn clause semantics,
#                   e.g. pred(v) -> pred(v + 5)
#                   It doesn't make sense to declare an affine transformation to say something like
#                        pred(v + 5) -> pred(v)
#
#           TODO: The `not-equal test` and "highly experimental" `:explicit` are not included here.
PatternVariable = namedtuple('PatternVariable', ['variable_name', 'relation'], defaults=[None])

# Affine transformation to be defined alongside a pattern variable
#   Each affine transformation is of the form `(from, offset, coefficient, pad)`
#       - `from`: default to the current variable.
#       - `offset`: default to 0
#       - `coefficient`: default to 1
#       - `pad`: 0 for closed-world predicates and 1 for open-world predicates
Affine = namedtuple('Affine', ['from_var', 'offset', 'coefficient', 'pad'], defaults=[None, None, None, None])

# TODO: Filters NOT TO IMPLEMENT IN SHORT TERM
# Filter to be defined in a PredicatePattern
#   Each filter is of the form `(constant_region, constant, coefficient)`
Filter = namedtuple('Filter', ['constant_region', 'constant', 'coefficient'])


class Type:
    def __init__(self, type_name, value_type, min=None, max=None, symbol_list=None):
        """
            Specify a Sigma type. Can be symbolic or discrete (will not support continuous type at this point). If
            symbolic, must provide a list of symbols. If discrete, must specify the range of values through `min` and
            `max`. Note that `min` is inclusive while `max` is exclusive, i.e., the actual range would be $[min, max)$
        :param type_name: `str` type. Name of the Sigma type
        :param value_type: `str` type. `"symbolic"` or `"discrete"`.
        :param min: `int` type. The lowest value of a discrete type. Must specify if type is `"discrete"`.
        :param max: `int` type. The highest value + 1 of a discrete type. Must specify if type is `"discrete"`.
        :param symbol_list: `list` type. A list of symbols. Must specify if type is `"symbol"`.
        """
        # Check validity of input
        assert type(type_name) is str, "type_name must be of type str"
        assert value_type in ['symbolic', 'discrete'], "value_type must be either 'symbolic' or 'discrete'"

        if value_type == 'symbolic' and symbol_list is None:
            raise ValueError("A symbol list must be specified via symbol_list when value_type is 'symbolic'")
        if value_type == 'discrete' and (min is None or max is None):
            raise ValueError("Both min and max must be specified when value_type is 'discrete'")
        if value_type == 'discrete' and min >= max:
            raise ValueError("min value must be less than max value")

        self.name = 'TYPE_' + type_name.upper()  # Prepend substring 'TYPE_' and send name to upper case
        self.value_type = value_type
        self.min = min
        self.max = max
        self.value_list = symbol_list if self.value_type == 'symbolic' else [i for i in range(min, max)]
        self.size = len(self.value_list)

        # Set mapping between type values and actual axis values along message tensor's dimension.
        self.value2axis = dict(zip(self.value_list, range(len(self.value_list))))
        self.axis2value = dict(zip(range(len(self.value_list)), self.value_list))


class Predicate:
    def __init__(self, predicate_name, arguments, world='open', exponential=False, normalize=False, perception=False,
                 function=None, *args, **kwargs):
        """
            Speficy a Sigma predicate.
        :param predicate_name:  `str` type
        :param arguments:  `list` type. List of **PredicateArgument namedtuples**. This is to specify the **working
            memory variables** (formal arguments) corresponding to this predicate.
        :param world:  `open` or `closed`. Default to `open`.
        :param exponential:  `bool` type or a `list` of predicate argument (variable) names.
            Whether to exponentiate outgoing messages from Working Memory Variable Node (WMVN).
            If `True`, exponentiate message across all variables. If a `list`, exponentiate only those variable dimensions.
            Default to `False`.
        :param normalize:  `bool` type or a `list` of predicate argument (variable) names.
            Whether to normalize outgoing messages from WMVN.
            If `Ture`, normalize message across all variables. If a `list`, normalize only those variable dimensions.
            Default to `False`.

            NOTE: normalization priorities over exponentiation, i.e., a message will be first exponentiated and then
                normalized, if both operation are requested
            #TODO: discuss with Volkan whether the option of normalizing only some of the predicate argument makes sense
            #TODO: to discuss with Volkan, when normalization and/or exponentiation are specified, the messages sending
                to which factor node are to be normalized and/or normalized.
        :param perception:  `bool` type. Whether this predicate can receive perceptual information.
        :param function:
                - `None`: no function specified at this predicate
                - `int` or `float`: a constant function
                - `torch.tensor`: Use the given tensor as the predicate function. Note that the shape of the tensor must
                    agree with dimensions of the variables and in the order as they are specified.
                - `str`: Name of another predicate. In this case they shares the same function.
        """
        # Check validity of input
        assert type(predicate_name) is str, "predicate_name must be of type str"
        assert type(arguments) is list, "arguments must be of type list"
        assert world in ['open', 'closed'], "world must be either 'open' or 'closed'"
        assert type(exponential) in [bool, list], "exponential must be of type bool or list"
        assert type(normalize) in [bool, list], "no_normalize must be of type bool or list"
        assert type(perception) is bool, "perception must be of type bool"
        assert function is None or type(function) in [int, float, torch.Tensor, str], \
            "function must be one of 'None', 'int', 'float', 'torch.Tensor', or 'str'"

        self.name = 'PRED_[' + predicate_name.upper() + ']'  # Prepend name with substring 'PRED_' and send to upper case

        self.arguments = arguments
        self.wm_var_names, self.wm_var_types, self.wm_var_unique = [], [], []
        self.var_list = []      # List of Variables, will only be set after this structure passed into a Sigma program
        self.var_name2var = {}  # Map from variable name to actual Variable instance. set after structure passed into a Sigma program

        # check selection
        self.selection = False  # Whether this predicate performs selection
        for argument in arguments:
            assert type(argument) is PredicateArgument, "arguments must be a list of 'PredicateArgument' namedtuples"
            assert argument.argument_name not in self.wm_var_names, \
                "argument name cannot duplicate. Duplicate name: {}".format(argument['argument_name'])
            assert type(argument.type) is Type, \
                "The 'type' field in the predicate argument '{}' must be of type Type".format(argument.type, argument)
            assert argument.unique_symbol in [None, '!', '%', '$', '^', '='], \
                "Unknown unique symbol '{}' in the predicate argument '{}'".format(argument.unique_symbol, argument)
            self.wm_var_names.append(argument.argument_name)
            self.wm_var_types.append(argument.type)
            self.wm_var_unique.append(argument.unique_symbol)

            if argument.unique_symbol is not None:
                self.selection = True

        if self.selection:
            assert world == 'closed', \
                "When any of the predicate's variables involves selection, the predicate must be closed-world."

        # if exponential / normalize specified as a list of variable names, check accordance
        if type(exponential) is list:
            for var in exponential:
                if var not in self.wm_var_names:
                    raise ValueError("The variable name {} provided in the argument 'exponential' was not specified as "
                                     "one of the variable of this predicate.".format(var))
        if type(normalize) is list:
            for var in normalize:
                if var not in self.wm_var_names:
                    raise ValueError("The variable name {} provided in the argument 'normalize' was not specified as "
                                     "one of the variable of this predicate.".format(var))

        self.world = world
        self.exponential = exponential
        self.normalize = normalize
        self.perception = perception
        self.function = function


class Conditional:
    def __init__(self, conditional_name, conditions=None, condacts=None, actions=None,
                 function_var_names=None, function=None, normal=None, *args, **kwargs):
        """
            Specify a Sigma conditional
        :param conditions:  a list of instances of PredicatePatterns
        :param condacts:  an list of instances of PredicatePatterns
        :param actions:  a list of instances of PredicatePatterns
        :param function_var_names:  a list of `str`, specifying the variable names in the function. The dimensions
                    of the function will be ordered in agreement with the given order of the variable names.
                Note: function_var_names must consist of pattern variables already declared in the conditions, actions,
                    & condacts.
        :param function:  an `int` or `float` or a `torch.tensor` or 'str', specifying the conditional function defined
                    over function_var_names.
                If None, default to 1 everywhere.
                If 'str', should be the name of another conditional, linking that conditional's function
        :param normal:  an list of `str`, specifying which variables to normalize over in gradient descent learning
            #TODO: leave to implement in future iterations
        """
        # Check conditions, actions, & condacts
        assert type(conditional_name) is str, "Argument 'conditional_name' must be of type str"
        if conditions is not None:
            assert type(conditions) is list, "Argument 'conditions' must be a list"
            assert all(type(p) is PredicatePattern for p in conditions), \
                "Elements in the argument 'conditions' must all be of type PredicatePattern"
        if condacts is not None:
            assert type(condacts) is list, "Argument 'condacts' must be a list"
            assert all(type(p) is PredicatePattern for p in condacts), \
                "Elements in the argument 'condacts' must all be of type PredicatePattern"
        if actions is not None:
            assert type(actions) is list, "Argument 'actions' must be a list"
            assert all(type(p) is PredicatePattern for p in actions), \
                "Elements in the argument 'actions' must all be of type PredicatePattern"

        assert [conditions, actions, condacts] != [None, None, None], "Cannot specify an empty conditional"
        if conditions is None:
            assert [actions, condacts] != [None, None], "Cannot specify a conditional that consists of only conditions"

        if function_var_names is not None:
            assert type(function_var_names) is list, "Argument 'function_var_names' must be a list"
            assert all(type(s) is str for s in function_var_names), \
                "Elements in the argument 'function_var_names' must all be of type str"
        if function is None:
            function = 1
        assert type(function) in [int, float, torch.Tensor, str], \
            "Argument 'function' must be of type int, float, torch.Tensor, or str"
        if normal is not None:
            assert type(normal) is list, "Argument 'normal' must be a list"
            assert all(type(v) is str for v in normal), \
                "Elements in the argument 'normal' must all be of type str"

        # Check Predicate Pattern validity
        conditions = [] if conditions is None else conditions
        condacts = [] if condacts is None else condacts
        actions = [] if actions is None else actions
        for pattern in conditions + condacts + actions:
            assert type(pattern.predicate_name) is str, "The 'predicate_name' field in the predicate pattern '{}' must be of type str".format(pattern)
            # check pattern nonlinearity
            if pattern.nonlinearity is not None:
                # TODO: Allow specifying custom functional object as nonlinearity in future iterations
                assert pattern.nonlinearity in ['negation', '-', 'exponential', '^', 'sigmoid', 's', 'relu', 'r', 'tanh',
                                                't', 'exp', 'e'], \
                    "The specified nonlinearity '{}' in predicate pattern {} cannot be recognized".\
                        format(pattern.nonlinearity, pattern)
            # check pattern elements
            assert type(pattern.elements) is list, "The 'elements' field in the predicate pattern '{}' is not of type list".format(pattern)
            assert all(type(e) is PatternElement for e in pattern.elements), "The 'elements' field in the predicate pattern '{}' should be a list of PatternElement".format(pattern)
            # check pattern elements' value field
            for element in pattern.elements:
                assert type(element.argument_name) is str, "The 'argument_name' field in the pattern element '{}' of predicate pattern '{}' must be of type str".format(element, pattern)
                # TODO: to allow Filter and Affine transformation declaration in future iterations
                assert type(element.value) in [int, str, list, PatternVariable], \
                    "The type of the 'value' field in pattern element '{}' of pattern '{}' is incorrect.".format(element, pattern)
                if type(element.value) is list:
                    assert all(type(v) is int for v in pattern.value) or all(type(v) is str for v in pattern.value), \
                        "When providing a list to the 'value' field in the pattern element '{}' of pattern '{}', it must be a list of all 'int', 'str', or 'Filter'".format(element, pattern)
                # If value is a PtternVariable, check its validity
                if type(element.value) is PatternVariable:
                    assert type(element.value.variable_name) is str, \
                        "The 'variable_name' field must be of type str in the pattern variable '{}' of pattern element '{}' of the predicate pattern '{}'".format(element.value, element, pattern)
                    # Variable name should not start with '_', because '_' is reserved for internal naming of unnamed pattern variables
                    assert element.value.variable_name[0] != '_', \
                        "The 'variable_name' field cannot start with '_' in the pattern variable '{}' of pattern element '{}' of the predicate pattern '{}'".format(element.value, element, pattern)
                    if element.value.relation is not None:
                        assert type(element.value.relation) in [int, Affine, Filter], \
                            "The 'relation' field must be of type 'int', 'Affine', or 'Filter' in the pattern variable '{}' of pattern element '{}' of the predicate pattern '{}'".format(element.value, element, pattern)
                    # TODO: Leave checking validity of Affine and Filter in future iterations

        # Name the predicate patterns for indexing various lookup tables
        self.name2pattern = {"pattern_"+str(i): pattern for i, pattern in enumerate(conditions + condacts + actions)}
        self.name2condition_pattern = {"pattern_"+str(i): pattern for i, pattern in enumerate(conditions)}
        self.name2condact_pattern = {"pattern_"+str(i + len(conditions)): pattern for i, pattern in enumerate(condacts)}
        self.name2action_pattern = {"pattern_"+str(i + len(conditions) + len(condacts)): pattern for i, pattern in enumerate(actions)}

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
        #                             "size" : max size over associated wm vars,
        #                             "unique" : True or False } }
        # constant pattern is assigned a unique constant variable name
        self.ptvar_list = []
        self.pattern_pt_vals = {}
        self.global_pt_vals = {}        # To be filled after passed into a Sigma program
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

                self.ptvar_list.append(pt_var_info["name"])
                self.pattern_pt_vals[pt_name][element.argument_name] = pt_var_info
                if pt_var_info["name"] not in self.ptv2wmv.keys():
                    self.ptv2wmv[pt_var_info["name"]] = [element.argument_name]
                else:
                    self.ptv2wmv[pt_var_info["name"]].append(element.argument_name)

        # Check that function_var_names agree with pattern variables declared in conditions, actions, & condacts
        for func_var in function_var_names:
            assert func_var in self.ptvar_list, \
                "The function variable '{}' is not one of the pattern variable declared in any predicate pattern".format(func_var)

        self.name = "COND_[" + conditional_name + "]"
        self.conditions = conditions
        self.condacts = condacts
        self.actions = actions
        self.function_var_names = [var for var in function_var_names]
        self.normal = normal
        self.function = function
