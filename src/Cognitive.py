"""
    Cognitive interface to Sigma cognitive architecture, which is the top part of the cognitive hourglass that specifies
        the cognitive language.

    In this Python implementation of Sigma that was originally (and currently) written in Lisp, we endeavor to
        1. take advantage of PyTorch’s autograd functionality, parallelizable features, and GPU support to speed up
           inference and learning.
        2. Enable flexible, generalizable, and customizable declarations of (pattern) variable bindings supported by
           highly flexible tensor operations.
                E.g. arbitrary arithmetic equations. Or even more, such as control structures.
        3. provide easy-to-use integration with neural models, and therefore connect to research in deep learning
           community.
        4. take advantage of Python’s language features to support further functional elegance of the architecture,
           and retain, or even improve, the flexibility of the architecture, so as to hopefully speed up iterations of
           further research, such as threads on structure learning and support for continuous distributions.
        5. (optional) experiment with probabilistic programming libraries, such as Pyro, to see if integration of such
           probabilistic programming paradigm to Sigma would enable efficient inference and learning with continuous
           parameterized distributions.


    Author of this file: Jincheng Zhou, University of Southern California
    Reference: Paul S. Rosenbloom, Abram Demski & Volkan Ustun, 2016, The Sigma Cognitive Architecture and System:
        Toward Functionally Elegant Grand Unification
"""

from Graphical import *
from collections import namedtuple
import torch

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
#       - `elements`: `list` type. A list of `PatternElement` namedtuples.
# TODO: Finish construction of following substructure and their docs
PredicatePattern = namedtuple('PredicatePattern', ['predicate_name', 'nonlinearity', 'elements'])

# Element in a PredicatePattern
#   Each element is of the form `(argument_name, `value`)
#       - `argument_name`: `str` type. The name of the formal argument, i.e., working memory variable name, of this
#               predicate
#       - `value`:
#           - A Constant Region, which is:      # TODO: to implement in v1
#               - An `int` if the variable is discrete, or an object (a symbol) in this variable's symbol list if the
#                       variable is symbolic.
#               - A `list` of `int` (if discrete) or symbols (if symbolic) within the value range of this variable. This
#                       would yield a list of intervals at the specified values in the given list.
#               - `*`. This would yield the entire scope of the dimension
#           - A `Filter` or a `list` of `Filter`s.
#           - A `PatternVariable`.              # TODO: to implement in v1
PatternElement = namedtuple('PatternElement', ['argument_name', 'value'])

# TODO: Filters NOT TO IMPLEMENT IN SHORT TERM
# Filter to be defined in a PredicatePattern
#   Each filter is of the form `(constant_region, constant, coefficient)`
Filter = namedtuple('Filter', ['constant_region', 'constant', 'coefficient'])

# Pattern Variable to be defined in a predicate pattern
#   Each pattern variable is of the form `(variable_name, relation)`
#       - `variable_name`: `str` type. The name of the pattern variable. This is to be distinguished from the variable's
#               working memory variable, or `argument_name`. These are the actual variables used to determine various
#               forms of variable bindings in a conditional. Bindings are assumed when same pattern variable is referred
#               in multiple places in a conditional. It's like the distinction between the actual argument and the
#               formal argument in a procedural programming language.
#       - `relation`: an optional parameter
#           - `None`: default value                             # TODO: to implement in v1
#           - an `int`: declaring offset
#           - an `Affine`: declaring an affine transformation
#           - a `Filter` or a list of `Filter`s.
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


class Sigma:
    """
        Sigma's cognitive interface. User should instantiate this class in order to create a Sigma model (or program).

        Graph compilation is eager, i.e., as soon as a predicate or a conditional is added, the corresponding subgraph
            will be compiled and added to the graphical architecture. This is to support structure learning in Sigma in
            the future.
    """

    def __init__(self):

        ### Public bookkeeping data structures ###
        ## Cognitive level bookkeeping data structure
        # list of types, predicates, conditionals
        self.type_list = []
        self.predicate_list = []
        self.conditional_list = []

        # mappings from name to structure
        self.name2type, self.name2predicate, self.name2conditional = {}, {}, {}

        ## Graphical level bookkeeping data structure
        # mappings from predicate, conditional to node group
        self.predicate2group, self.conditional2group = {}, {}

        # The graphical architecture
        self.G = Graph()

        # Sigma program global parameters
        # TODO: initialize Sigma

    def add(self, structure):
        """
            Add a Sigma structure, one of Type, Predicate, or Conditional
        """
        # Check validity of argument
        if type(structure) not in [Type, Predicate, Conditional]:
            raise ValueError("structure must be one of Sigma's Type, Predicate, or Conditional")

        if isinstance(structure, Type):
            self.type_list.append(structure)
            self.name2type[structure.name] = structure

        if isinstance(structure, Predicate):
            # Check if the types in the predicate are already defined, and change str to Type
            for i, argument_type in enumerate(structure.wm_var_types):
                if argument_type not in self.name2type.keys() and argument_type not in self.type_list:
                    raise ValueError("Predicate {} try to include type {} that has not yet been defined in this program"
                                     .format(structure.name, argument_type))
                # If this type specified in that Predicate is a string, than change it to corresponding Type
                if type(argument_type) is str:
                    structure.wm_var_types[i] = self.name2type[argument_type]

            # Register structure
            self.predicate_list.append(structure)
            self.name2predicate[structure.name] = structure

            # Compile this predicate
            self._compile_predicate(structure)

        if isinstance(structure, Conditional):
            # TODO: check entries in the Conditional is valid, such as no undefined predicates or mismatch of argument names

            # Register structure
            self.conditional_list.append(structure)
            self.name2conditional[structure.name] = structure

            # Compile this conditional
            self._compile_conditional(structure)

    def add_type(self, *args, **kwargs):
        """
            Add a Sigma type to this Sigma program directly.
        """
        sigma_type = Type(*args, **kwargs)
        self.add(sigma_type)

    def add_predicate(self, *args, **kwargs):
        """
            Add a Sigma predicate to this Sigma program directly.
        """
        predicate = Predicate(*args, **kwargs)
        self.add(predicate)

    def add_conditional(self, *args, **kwargs):
        """
            Add a Sigma conditional to this Sigma program directly.
        """
        conditional = Conditional(*args, **kwargs)
        self.add(conditional)

    def run(self, num_cycles):
        """
            Run the Sigma program for the given number of cycles.
        """
        # TODO: run Sigma program
        pass

    def _compile_predicate(self, predicate):
        """
            Compile the given predicate to the graphical architecture.
        """
        # TODO: compile predicate. CHECK WITH DR.VOLKAN ABOUT CASES OF PREDICATE NODE GROUP STRUCTURES

        # Create Variables
        var_list = []
        selection = False       # Whether there is any variable that involves selection
        for (var_name, var_type, var_unique) in \
                zip(predicate.wm_var_list, predicate.wm_var_types, predicate.wm_var_unique):
            var_list.append(Variable(var_name, var_type.size, var_unique is not None, var_unique))
            if var_unique is not None:
                selection = True

        # Create new nodes and register bookkeeping info
        nodegroup = {}
        # basic WMVN, or the incoming WMVN in the case of a predicate with variables that involve selection
        if selection:
            wmvn = self.G.new_node(WMVN, predicate.name + "_WMVN_IN", var_list)
            nodegroup['WMVN_IN'] = wmvn
        else:
            wmvn = self.G.new_node(WMVN, predicate.name + "_WMVN", var_list)
            nodegroup['WMVN'] = wmvn

        # PBFN if this predicate is perceptual
        if predicate.perception:
            pbfn = self.G.new_node(PBFN, predicate.name + "_PBFN")
            nodegroup['PBFN'] = pbfn

            # set up unidirectional link from PBFN to WMVN
            self.G.add_unilink(pbfn, wmvn)

        # LTMFN if this predicate is memorial, i.e., if the predicate function is specified
        if predicate.function is not None:
            func = predicate.function

            # If function is str, then check if such predicate exists and if it has a ltmfn set up. Set up link
            if type(func) is str:
                assert func in self.name2predicate.keys(), "Target predicate {} does not exist".format(func)
                assert "LTMFN" in self.predicate2group[self.name2predicate[func]].keys(), \
                    "Target predicate {} exists but it does not have a Long-Term Memory Function (LTMFN)".format(func)

                target_ltmfn = self.predicate2group[self.name2predicate[func]]['LTMFN']
                # Set up bidirectional link between target LTMFN and own WMVN
                self.G.add_bilink(target_ltmfn, wmvn)

            # Else, set up own ltmfn
            else:
                ltmfn = self.G.new_node(LTMFN, predicate.name + "_LTMFN")
                ltmfn.set_function(func, var_list)      # for a pred function, var list is simply own var list
                nodegroup['LTMFN'] = ltmfn

                # set up bidirectional link between LTMFN and WMVN
                self.G.add_bilink(ltmfn, wmvn)

        # WMFN if this predicate is closed-world
        # TODO: QUESTION: is WMVN - WMFN unidirectional for any closed-world predicate, i.e., isn't for a closed-world
        #   predicate, its entire distribution is not replaced each cycle?
        # TODO: here assume it is always such a unidirectional cycle
        if predicate.world is 'closed':
            # Create a unidirectional cycle: WMVN -> WMFN -> WMFN_VN -> WMFN_ACFN -> WMVN
            wmfn = self.G.new_node(WMFN, predicate.name + "_WMFN")
            wmfn_vn = self.G.new_node(WMVN, predicate.name + "_WMFN_VN", var_list)
            wmfn_acfn = self.G.new_node(ACFN, predicate.name + "_WMFN_ACFN")
            nodegroup['WMFN'] = wmfn
            nodegroup['WMFN_VN'] = wmfn_vn
            nodegroup['WMFN_ACFN'] = wmfn_acfn

            # Set up cycle unidirectional link
            self.G.add_unilink(wmvn, wmfn)
            self.G.add_unilink(wmfn, wmfn_vn)
            self.G.add_unilink(wmfn_vn, wmfn_acfn)
            self.G.add_unilink(wmfn_acfn, wmvn)

        # WMVN_OUT if this predicate involves selection
        # TODO: QUESTION: In the case of a open-world with selection predicate, how is the WMVN_OUT node connected? Does
        #  it connect from a WMFN even the predicate is open-world?
        # if selection:
        #     wmvn_out = self.G.new_node(WMVN, predicate.name + "_WMVN_OUT", var_list)
        #     nodegroup['WMVN_OUT'] = wmvn_out
        #
        #     # Set up unidirectional link from WMFN to WMVN_OUT. Create a WMFN if not already created
        #     if 'WMFN' in nodegroup.keys():
        #         wmfn = nodegroup['WMFN']
        #     else:
        #         wmfn = self.G.new_node(WMFN, predicate.name + "_WMFN")
        #         nodegroup['WMFN'] = wmfn
        #     self.G.add_unilink(wmfn, wmvn_out)
        #
        # TODO: For now assume that whenever a selection is involved, the predicate MUST BE CLOSED_WORLD
            if selection:
                wmvn_out = self.G.new_node(WMVN, predicate.name + "_WMVN_OUT", var_list)
                nodegroup['WMVN_OUT'] = wmvn_out

                # Add a unidirectional link from wmfn to wmvn_out
                self.G.add_unilink(wmfn, wmvn_out)

        # Register node group
        self.predicate2group[predicate] = nodegroup

    def _compile_conditional(self, conditional):
        """
            Compile the given conditional to the graphical architecture.
        """
        # TODO: compile conditional
        pass


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
        if type(type_name) is not str:
            raise ValueError("type_name must be of type str")
        if value_type not in ['symbolic', 'discrete']:
            raise ValueError("value_type must be either 'symbolic' or 'discrete'")
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
    def __init__(self, predicate_name, arguments, world='open', exponential=False, no_normalize=True, perception=False,
                 function=None, *args, **kwargs):
        """
            Speficy a Sigma predicate.
        :param predicate_name:  `str` type
        :param arguments:  `list` type. List of **PredicateArgument namedtuples**. This is to specify the **working
            memory variables** (formal arguments) corresponding to this predicate.
        :param world:  `open` or `closed`. Default to `open`.
        :param exponential:  `bool` type. Whether to exponentiate outgoing messages from Working Memory Variable Node
                (WMVN). Default to `False`.
        :param no_normalize:  `bool` type. Whether not to normalize outgoing messages from WMVN. Default to `True`.
        :param perception:  `bool` type. Whether this predicate can receive perceptual information.
        :param function:
                - `None`: no function specified at this predicate
                - `int` or `float`: a constant function
                - `torch.tensor`: Use the given tensor as the predicate function. Note that the shape of the tensor must
                    agree with dimensions of the variables and in the order as they are specified.
                - `str`: Name of another predicate. In this case they shares the same function.
        """
        # Check validity of input
        if type(predicate_name) is not str:
            raise ValueError("predicate_name must be of type str")
        if type(arguments) is not list:
            raise ValueError("arguments must be of type list")
        if world != 'open' and world != 'closed':
            raise ValueError("world must be either 'open' or 'closed'")
        if type(exponential) is not bool:
            raise ValueError("exponential must be of type bool")
        if type(no_normalize) is not bool:
            raise ValueError("no_normalize must be of type bool")
        if type(perception) is not bool:
            raise ValueError("perception must be of type bool")
        if function is not None and type(function) not in [int, float, torch.Tensor, str]:
            raise ValueError("function must be one of 'None', 'int', 'float', 'torch.Tensor', or 'str'")

        self.name = 'PRED_[' + predicate_name.upper() + ']'  # Prepend name with substring 'PRED_' and send to upper case

        self.arguments = arguments
        self.wm_var_list, self.wm_var_types, self.wm_var_unique = [], [], []

        # check selection
        selection = False
        for argument in arguments:
            if type(argument) is not PredicateArgument:
                raise ValueError("arguments must be a list of 'PredicateArgument' namedtuples")
            if argument.argument_name in self.wm_var_list:
                raise ValueError("argument name cannot duplicate. Duplicate name: {}".format(argument['argument_name']))
            self.wm_var_list.append(argument.argument_name)
            self.wm_var_types.append(argument.type)
            self.wm_var_unique.append(argument.unique_symbol)

            if argument.unique_symbol is not None:
                selection = True

        if selection:
            assert world == 'closed', \
                "When any of the predicate's variables involves selection, the predicate must be closed-world."

        self.world = world
        self.exponential = exponential
        self.no_normalize = no_normalize
        self.perception = perception
        self.function = function


class Conditional:
    def __init__(self, conditional_name, conditions=None, condacts=None, actions=None, function_default=None,
                 function_var_names=None, normal=None, function=None, *args, **kwargs):
        """
            Specify a Sigma conditional
        :param conditions:  an iterable of instances of PredicatePatterns
        :param condacts:  an iterable of instances of PredicatePatterns
        :param actions:  an iterable of instances of PredicatePatterns
        :param function_default:  `int` or `float`, the default value for unspecified regions of function
        :param function_var_names:  an iterable of `str`, specifying the variable names in the function. The dimensions
                    of the function will be ordered in agreement with the given order of the variable names.
        :param normal:  an iterable of `str`, specifying which variables to normalize over in gradient descent learning
        :param function:  an `int` or `float` or a `torch.tensor`, specifying the conditional function defined over
                    function_var_names
        """
        # TODO: check validity of given arguments

        self.name = conditional_name
        self.conditions = conditions
        self.condacts = condacts
        self.actions = actions
        self.function_default = function_default
        self.function_var_names = [var for var in function_var_names]
        self.normal = normal
        self.function = function
