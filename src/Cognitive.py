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

from Structures import *
from Graphical import *
from collections import namedtuple
import torch


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


