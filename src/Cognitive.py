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
#               or customized torch nonlinearity functions
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

    def add(self, structure):
        """
            Add a Sigma structure, one of Type, Predicate, or Conditional
        """
        # Check validity of argument
        if type(structure) not in [Type, Predicate, Conditional]:
            raise ValueError("structure must be one of Sigma's Type, Predicate, or Conditional")

        # Register new Type
        if isinstance(structure, Type):
            self.type_list.append(structure)
            self.name2type[structure.name] = structure

        # Register new Predicate
        if isinstance(structure, Predicate):
            # Check if the types in the predicate are already defined, and change str to Type
            for i, argument_type in enumerate(structure.wm_var_types):
                if argument_type not in self.name2type.keys() and argument_type not in self.type_list:
                    raise ValueError("Predicate {} try to include type {} that has not yet been defined in this program"
                                     .format(structure.name, argument_type))
                # If this type specified in that Predicate is a string, than change it to corresponding Type
                if type(argument_type) is str:
                    structure.wm_var_types[i] = self.name2type[argument_type]

            # Create and register Variables
            for (var_name, var_type, var_unique) in \
                    zip(structure.wm_var_names, structure.wm_var_types, structure.wm_var_unique):
                var = Variable(var_name, var_type.size, var_unique is not None, var_unique)
                structure.var_list.append(var)
                structure.var_name2var[var_name] = var

            # Register structure
            self.predicate_list.append(structure)
            self.name2predicate[structure.name] = structure

            # Compile this predicate
            self._compile_predicate(structure)

        # Register new Conditional
        if isinstance(structure, Conditional):
            # Check if predicate patterns align with already registered predicates
            for pattern in structure.conditions + structure.condacts + structure.actions:
                # Check if predicate exists
                if pattern.predicate_name not in self.name2predicate.keys():
                    raise ValueError("The predicate pattern '{}' includes an unknown predicate '{}'".format(pattern, pattern.predicate_name))
                # Check if the number of predicate elements is no greater than the number of WM vars of that predicate
                pred = self.name2predicate[pattern.predicate_name]
                if len(pattern.elements) > len(pred.wm_var_names):
                    raise ValueError("The number of predicate elements declared in the predicate pattern '{}', "
                                     "currently {}, exceeds the number of working memory variable in the predicate "
                                     "'{}', currently {}".format(pattern, len(pattern.elements), pattern.predicate_name,
                                                                 len(pred.wm_var_names)))
                # Check if the 'argument_name' in the 'elements' agree with that predicate's WM vars
                for element in pattern.elements:
                    if element.argument_name not in pred.wm_var_names:
                        raise ValueError("The 'argument_name' '{}' declared in the pattern element '{}' of the "
                                         "predicate pattern '{}' is not one of the working memory variables of the "
                                         "predicate '{}'".format(element.argument_name, element, pattern,
                                                                 pattern.predicate_name))

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

        # Create new nodes and register bookkeeping info
        nodegroup = {}
        # basic WMVN, or the incoming WMVN in the case of a predicate with variables that involve selection
        if predicate.selection:
            wmvn = self.G.new_node(WMVN, predicate.name + "_WMVN_IN", predicate.var_list)
            nodegroup['WMVN_IN'] = wmvn
        else:
            wmvn = self.G.new_node(WMVN, predicate.name + "_WMVN", predicate.var_list)
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
                ltmfn.set_function(func, predicate.var_list)      # for a pred function, var list is simply own var list
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
            wmfn_vn = self.G.new_node(WMVN, predicate.name + "_WMFN_VN", predicate.var_list)
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
            if predicate.selection:
                wmvn_out = self.G.new_node(WMVN, predicate.name + "_WMVN_OUT", predicate.var_list)
                nodegroup['WMVN_OUT'] = wmvn_out

                # Add a unidirectional link from wmfn to wmvn_out
                self.G.add_unilink(wmfn, wmvn_out)

        # Register node group
        self.predicate2group[predicate] = nodegroup

    def _compile_conditional(self, conditional):
        """
            Compile the given conditional to the graphical architecture.
            A Sigma's implementation of Rete algorithm
        """
        # Step 1: Compile Alpha subnetworks per predicate pattern
        #   Link directions for conditions goes inward toward Beta network
        #   Link directions for actions goes outward toward Working Memory VNs
        #   Link directions for condacts are bidirectional up until Beta join factor node
        #   The order of the different parts in a Alpha subnet, with direction from WM to Beta join, is:
        #       [Filter, Nonlinearity, Affine Delta, Affine]
        # TODO: Discuss with Volkan about this order. E.g. whether Affine should be put BEFORE Affine Delta

        # util inner functions for adding nodes depending on different pattern type. Return the new alpha terminal node
        def grow_alpha(term, fn, vn, ptype):
            # grow a subnet: terminal -- fn -- vn, and set vn as the new terminal. Link direction depends on ptype
            if ptype is "condition":
                self.G.add_unilink(term, fn)
                self.G.add_unilink(fn, vn)
            elif ptype is "action":
                self.G.add_unilink(fn, term)
                self.G.add_unilink(vn, fn)
            else:
                self.G.add_bilink(term, fn)
                self.G.add_bilink(fn, vn)
            return vn

        alpha_terminals = {}    # map from predicate to the terminal variable node of that predicate's alpha subnet
        for ptype, pattern_group in zip(["condition", "condact", "action"],
                                        [conditional.conditions, conditional.condacts, conditional.actions]):
            for pattern in pattern_group:
                name_prefix = conditional.name + "_ALPHA_"
                # Initialize the alpha subnet terminal node to be the predicate's WMVN
                pred = self.name2predicate[pattern.predicate_name]
                nodegroup = self.predicate2group[pred]
                terminal = nodegroup['WMVN'] if 'WMVN' in nodegroup.keys() else nodegroup['WMVN_OUT']

                # Set up Filter nodes
                # TODO: Detailed implementation left for future iterations

                # Set up Nonlinearity nodes
                if pattern.nonlinearity is not None:
                    # Create nonlinearity factor nodes and set up links
                    nlfn = self.G.new_node(NLFN, name_prefix + "NLFN", pattern.nonlinearity)
                    nlfn_vn = self.G.new_node(WMVN, name_prefix + "NLFN_VN", pred.var_list)
                    terminal = grow_alpha(terminal, nlfn, nlfn_vn, ptype)

                # Set up Affine Delta nodes
                # First create Variables for pattern vars. If one pattern vars is associated with multiple wm vars,
                #   then the size of that pattern var is the max size of all associated wm vars (to leave enough region)
                # TODO: discuss with Volkan whether the uniqueness of pattern variable should follow that of its associated
                #       wm variable. If so, what happens if a ptv is associated with multiple wmvs with different uniqueness
                #       Currently assuming pattern varaibles are all unique, i.e., if needed, perform sum reduce.
                pt_var_list = []
                for ptv, wmvs in conditional.ptv2wmv[pattern].items():
                    if len(wmvs) == 1:
                        pt_var_list.append(Variable(ptv, wmvs[0].size, unique=True, selection=False))
                    else:
                        # one pattern variable associated with multiple wm variables
                        size = max(var.size for var in wmvs)
                        pt_var_list.append(Variable(ptv, size, unique=True, selection=False))

                adfn = self.G.new_node(ADFN, name_prefix + "ADFN", pattern.pt_vals)
                adfn_vb = self.G.new_node(WMVN, name_prefix + "ADFN_VN", pt_var_list)
                terminal = grow_alpha(terminal, adfn, adfn_vb, ptype)

                # Set up Affine Transformation nodes
                # TODO: Detailed implementation left for future iterations

                # Alpha subnet construction finished, register terminal node
                alpha_terminals[pattern.predicate_name] = terminal

        # Step 2: Compile Beta network


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
            assert argument.unique_symbol in ['!', '%', '$', '^', '='], \
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
        :param function:  an `int` or `float` or a `torch.tensor`, specifying the conditional function defined over
                    function_var_names. If None, default to 1 everywhere
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
        assert type(function) in [int, float, torch.Tensor], \
            "Argument 'function' must be of type int, float, or torch.Tensor"
        if normal is not None:
            assert type(normal) is list, "Argument 'normal' must be a list"
            assert all(type(v) is str for v in normal), \
                "Elements in the argument 'normal' must all be of type str"

        # Check Predicate Pattern validity
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

        # Set up pattern var list for future lookup
        # Set up internal WM var -- pattern var map per predicate pattern for future lookup.
        #       pt_vals = { pattern :
        #                    { wm_var_name :
        #                       { "name" : pt_var_name
        #                         "type" : "var" or "const"
        #                         "vals" : int/str values if type is const or None otherwise
        #                         "rel"  : relation, if specified, otherwise None} } }
        #       ptv2wmv = { pattern :
        #                    { pt_var_name : list of corresponding wm vars } }
        #   constant pattern is assigned a unique constant variable name
        self.ptvar_list = []
        self.pt_vals = {}
        self.ptv2wmv = {}
        const_count = 0
        for pattern in conditions + condacts + actions:
            self.pt_vals[pattern] = {}
            self.ptv2wmv[pattern] = {}
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
                self.pt_vals[pattern][element.argument_name] = pt_var_info
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
