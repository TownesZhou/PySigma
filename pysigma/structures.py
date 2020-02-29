"""
    Common structures and componenets serving the Sigma cognitive language

    Author of this file: Jincheng Zhou, University of Southern California
"""

import torch
from collections import namedtuple
from collections.abc import Iterable
from typing import Union
from .utils import *
from .graphical._structures import Variable


class PredicateArgument:
    """
        Argument in a Predicate
    """

    def __init__(self, argument_name, argument_type, probabilistic=False, unique_symbol=None, normalize=False, **kwargs):
        """

        :param argument_name:   'str'. Name of the predicate argument (working memory variable)
        :param argument_type:   'Type'. The "type" of the working memory variable
        :param probabilistic:   True or False. Whether this working memory variable is probabilistic, i.e., semantically
                                    its individual values represent probabilistic scores within the interval [0, 1].
                                If False, this variable will be treated as a vector variable.
                                Note that if normalize is True, then probabilistic must be True
        :param unique_symbol:   'str' or None. If None, this variable will be treated as universal.
                                  - `'!'`: Select best
                                  - `'$'`: Select expected value
                                  - `'^'`: Maintain exponential transform of distribution
                                  - `'='`: Select by probability matching
        :param normalize:       True or False. Whether this variable represents a discrete distribution and to normalize
                                    messages over this variable dimension.
        """
        if not isinstance(argument_name, str):
            raise ValueError("1st argument 'argument_name' of a PredicateArgument must be a 'str'")
        if not isinstance(argument_type, Type):
            raise ValueError("2nd argument 'argument_type' of a PredicateArgument must be a 'Type'")
        if not isinstance(probabilistic, bool):
            raise ValueError("argument 'probabilistic' of a PredicateArgument bust be a 'bool'")
        if unique_symbol is not None and not isinstance(unique_symbol, str):
            raise ValueError("argument 'unique_symbol' of a PredicateArgument must be a 'str'")
        if unique_symbol is not None and unique_symbol not in ['!', '$', '^', '=']:
            raise ValueError("Unknown unique symbol: '{}'".format(unique_symbol))
        if not isinstance(normalize, bool):
            raise ValueError("argument 'normalize' or a PredicateArgument must be a 'bool'")
        # Check that if normalize is True, then probabilistic is also true
        if normalize and not probabilistic:
            raise ValueError("If 'normalize' is True for a PredicateArgument, then it must also be probabilistic.")

        self.argument_name = argument_name
        self.type = argument_type
        self.probabilistic = probabilistic
        self.unique_symbol = unique_symbol
        self.normalize = normalize

        # Create working memory variable
        self.wmvar = Variable(argument_name, argument_type.size, probabilistic, unique_symbol is not None, normalize)


class PredicatePattern:
    """
        Predicate Pattern in a Conditional
          Each predicate pattern is of the form `(predicate_name, nonlinearity, 'elements')
              - `predicate_name`:  name of the predicate in question
              - `elements`: A python iterable of `PatternElement` namedtuples.
              - `nonlinearity`: `None` or one of the following: `'negation'` or `'-'`, `'exponential'` or `'^'`, `'sigmoid'`
                      or `'s'`, `'relu'` or `'r'`, `'tanh'` or `'t'`, `'exp'` or `'e'` (true exponential),
                      or customized torch nonlinearity functions
    """

    def __init__(self, predicate_name, elements, nonlinearity=None):
        if not isinstance(predicate_name, str):
            raise ValueError("1st argument 'predicate_name' of a PredicatePattern must be a 'str'")
        if not isinstance(elements, Iterable) or not all(isinstance(e, PatternElement) for e in elements):
            raise ValueError("2nd argument 'elements' of a PredicatePattern must be an iterable of 'PatternElement's")
        if nonlinearity is not None and nonlinearity not in ['negation', '-', 'exponential', '^', 'sigmoid', 's',
                                                             'relu', 'r', 'tanh', 't', 'exp', 'e']:
            raise ValueError("Unknown nonlinearity: '{}'".format(nonlinearity))
        # TODO: allow customized nonlinearity declaration in future iterations

        self.predicate_name = predicate_name
        self.elements = list(elements)
        self.nonlinearity = nonlinearity


class PatternElement:
    """
        Element in a PredicatePattern
          Each element is of the f
                  - A Constant Region, which is:      # TODO: to implement in v1
                      - An `int` if the variable is discrete, or an 'str' (a symbol) in this variable's symbol list if the
                              variable is symbolic.
                      - A iterable of `int` (if discrete) or 'str' (if symbolic) within the value range of this variable. This
                              would yield a list of intervals at the specified values in the given list.
                      - None: This would yield the entire scope of the dimension
                  - A `Filter` or a `list` of `Filter`s. # TODO: to implement in future iterations
                  - A `PatternVariable`.              # TODO: to implement in v1
    """

    def __init__(self, argument_name, value=None):
        if not isinstance(argument_name, str):
            raise ValueError("1st argument 'argument_name' of a PatternElement must be a 'str'")
        if value is not None and not isinstance(value, (int, str, Iterable, Filter, PatternVariable)):
            raise ValueError("If not None, 2nd argument 'value' of a PatternElement must be an 'int', 'str', "
                             "an iterable of 'int' or 'str', a 'Filter', or a 'PatternVariable'")
        if isinstance(value, Iterable) and \
                not (all(isinstance(v, int) for v in value) or
                     all(isinstance(v, str) for v in value) or
                     all(isinstance(v, Filter) for v in value)):
            raise ValueError("If an iterable, the 2nd argument 'value' of a PatternElement must be an iterable "
                             "of 'int', 'str', or 'Filter's")

        self.argument_name = argument_name
        if isinstance(value, PatternVariable):
            self.value = value
        else:
            self.value = list(value) if isinstance(value, Iterable) else [value]    # turn into list anyway


class PatternVariable:
    """
        Pattern Variable to be defined in a predicate pattern
          Each pattern variable is of the form `(variable_name, relation)`
              - `variable_name`: `str` type. The name of the pattern variable. This is to be distinguished from the variable's
                      working memory variable, or `argument_name`. These are the actual variables used to determine various
                      forms of variable bindings in a conditional. Bindings are assumed when same pattern variable is referred
                      in multiple places in a conditional. It's like the distinction between the actual argument and the
                      formal argument in a procedural programming language.
              - `relation`: an optional parameter
                  - `None`: default value
                  - an `int`: declaring offset
                  - an `Affine`: declaring an affine transformation
                  - a `Filter` or an iterable of `Filter`s.    # TODO: to implement in future iterations
                  TODO: The `not-equal test` and "highly experimental" `:explicit` are not included here.
    """

    def __init__(self, variable_name, relation=None):
        if not isinstance(variable_name, str):
            raise ValueError("The 1st argument 'variable_name' of a PatternVariable must be a 'str'")
        if variable_name[0] == '_':
            raise ValueError("The 1st argument 'variable_name' of a PatternVariable cannot start with '_'")
        if relation is not None and not isinstance(relation, (int, Affine, Filter, Iterable)):
            raise ValueError("If not None, the 2nd argument 'relation' of a PatternVariable must be an 'int', "
                             "an 'Affine', a 'Filter', or an iterable of 'Filter's")
        if isinstance(relation, Iterable) and not all(isinstance(r, Filter) for r in relation):
            raise ValueError("If an iterable, the 2bd argument 'relation' of a PatternVariable must be an iterable "
                             "of Filters")

        self.variable_name = variable_name
        self.relation = list(relation) if isinstance(relation, Iterable) else relation


class Affine:
    """
        Affine transformation to be defined alongside a pattern variable
          Each affine transformation is of the form `(from, offset, coefficient, pad)`
              - `from`: default to the current variable.
              - `offset`: default to 0
              - `coefficient`: default to 1
              - `pad`: 0 for closed-world predicates and 1 for open-world predicates
        TODO: Define a more generalized notion of 'mapping' in place of Affine in future iterations
    """

    def __init__(self, from_var=None, offset=0, coefficient=1, pad=0):
        if from_var is not None and not isinstance(from_var, str):
            raise ValueError("If not None, The 1st argument 'from_var' of an Affine must be a 'str'")
        if not isinstance(offset, int):
            raise ValueError("The 2nd argument 'offset' of an Affine must be an 'int'")
        if not isinstance(coefficient, (int, float)):
            raise ValueError("The 3rd argument 'coefficient' of an Affine must be an 'int' or a 'float'")
        if not isinstance(pad, int):
            raise ValueError("The 4th argument 'pad' of an Affine must be an 'int'")

        self.from_var = from_var
        self.offset = offset
        self.coefficient = coefficient
        self.pad = pad


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
        :param symbol_list: An iterable of 'str' representing symbols. Must specify if type is `"symbol"`.
        """
        # Argument validation
        if not isinstance(type_name, str):
            raise ValueError("The 1st argument 'type_name' of a Type must be a 'str'")
        if value_type not in ['symbolic', 'discrete']:
            raise ValueError("The 2nd argument 'value_type' of a Type must be either 'symbolic' or 'discrete'")
        if min is not None and (not isinstance(min, int) or min < 0):
            raise ValueError("The argument 'min' of a Type must be a nonnegative integer")
        if max is not None and (not isinstance(max, int) or max < 0):
            raise ValueError("The argument 'max' of a Type must be a nonnegative integer")
        if symbol_list is not None and \
                (not isinstance(symbol_list, Iterable) or not all(isinstance(s, str) for s in symbol_list)):
            raise ValueError("If not None, the argument 'symbol_list' of a Type must be an iterable of 'str's")
        if value_type == 'symbolic' and symbol_list is None:
            raise ValueError("A symbol list must be specified via symbol_list when value_type is 'symbolic'")
        if value_type == 'discrete' and (min is None or max is None):
            raise ValueError("Both min and max must be specified when value_type is 'discrete'")
        if value_type == 'discrete' and min >= max:
            raise ValueError("min value must be less than max value")

        self.name = intern_name(type_name, "type")  # Prepend substring 'TYPE_' and send name to upper case
        self.value_type = value_type
        self.min = min
        self.max = max
        self.value_list = symbol_list if self.value_type == 'symbolic' else [i for i in range(min, max)]
        self.size = len(self.value_list)

        # Set mapping between type values and actual axis values along message tensor's dimension.
        self.value2axis = dict(zip(self.value_list, range(len(self.value_list))))
        self.axis2value = dict(zip(range(len(self.value_list)), self.value_list))


class Predicate:
    def __init__(self, predicate_name, arguments, world='open', perception=False, function=None, *args, **kwargs):
        """
            Speficy a Sigma predicate.
        :param predicate_name:  `str` type
        :param arguments:  An python iterable of 'PredicateArgument's. This is to specify the **working
            memory variables** (formal arguments) corresponding to this predicate.
        :param world:  `open` or `closed`. Default to `open`.
        :param perception:  `bool` type. Whether this predicate can receive perceptual information.
        :param function:
                - `None`: no function specified at this predicate
                - `int` or `float`: a constant function
                - `torch.tensor`: Use the given tensor as the predicate function. Note that the shape of the tensor must
                    agree with dimensions of the variables and in the order as they are specified.
                - `str`: Name of another predicate. In this case they shares the same function.
        """
        # Argument validation
        if not isinstance(predicate_name, str):
            raise ValueError("The 1st argument 'predicate_name' of a Predicate must be a 'str'")
        if not isinstance(arguments, Iterable) or not all(isinstance(arg, PredicateArgument) for arg in arguments):
            raise ValueError("The 2nd argument 'arguments' of a Predicate must be an iterable of 'PredicateArgument's")
        if world not in ['open', 'closed']:
            raise ValueError("The 3rd argument 'world' of a Predicate must be either 'open' or 'closed'")
        if not isinstance(perception, bool):
            raise ValueError("The 6th argument 'perception' must be either 'True' or 'False'")
        if function is not None and not isinstance(function, (int, float, torch.Tensor, str)):
            raise ValueError("If not None, the 7th argument 'function' of a predicate must be 'int', 'float', "
                             "'torch.Tensor', or 'str'")

        self.var_list = []  # List of Variables
        self.var_name2var = {}  # Map from variable name to actual Variable instance
        self.wm_var_types = []

        # check selection
        self.selection = False  # Whether this predicate performs selection
        for argument in arguments:
            assert isinstance(argument, PredicateArgument)
            # Check duplicate argument names
            if argument.wmvar in self.var_list:
                raise ValueError("arguments in a Predicate cannot duplicate. Duplicate name: {}"
                                 .format(argument.argument_name))

            self.var_list.append(argument.wmvar)
            self.var_name2var[argument.argument_name] = argument.wmvar
            self.wm_var_types.append(argument.type)

            if argument.unique_symbol is not None:
                self.selection = True

        if self.selection:
            if world is not 'closed':
                raise ValueError("When any of the predicate's variables involves selection, the predicate must be "
                                 "closed-world.")

        self.name = intern_name(predicate_name,
                                "predicate")  # Prepend name with substring 'PRED_' and send to upper case
        self.arguments = list(arguments)
        self.world = world
        self.perception = perception
        self.function = function


class Conditional:
    def __init__(self, conditional_name, conditions=None, condacts=None, actions=None,
                 function_var_names=None, function=None, normal=None, *args, **kwargs):
        """
            Specify a Sigma conditional
        :param conditions:  an iterable of instances of PredicatePatterns
        :param condacts:  an iterable of instances of PredicatePatterns
        :param actions:  an iterable of instances of PredicatePatterns
        :param function_var_names:  an iterable of `str`, specifying the variable names in the function. The dimensions
                    of the function will be ordered in agreement with the given order of the variable names.
                Note: function_var_names must consist of pattern variables already declared in the conditions, actions,
                    & condacts.
        :param function:  an `int` or `float` or a `torch.tensor` or 'str', specifying the conditional function defined
                    over function_var_names.
                If None, default to 1 everywhere.
                If 'str', should be the name of another conditional, linking that conditional's function
        :param normal:  an iterable of `str`, specifying which variables to normalize over in gradient descent learning
            #TODO: leave to implement in future iterations

            #TODO: Allow user to define customized summarization step for pattern variables
        """
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
