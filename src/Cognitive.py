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
from copy import deepcopy


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
            self._register_type(structure)

        # Register new Predicate
        if isinstance(structure, Predicate):
            self._register_predicate(structure)
            self._compile_predicate(structure)

        # Register new Conditional
        if isinstance(structure, Conditional):
            self._register_conditional(structure)
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

    def _register_type(self, sigma_type):
        """
            Register a new type in this sigma program
        """
        self.type_list.append(sigma_type)
        self.name2type[sigma_type.name] = sigma_type

    def _register_predicate(self, predicate):
        """
            Register a new predicate in this sigma program. Check against existing type entries and fill up relevant
                lookup tables
        """
        # Check if the types in the predicate are already defined, and change str to Type
        for i, argument_type in enumerate(predicate.wm_var_types):
            if argument_type not in self.name2type.keys() and argument_type not in self.type_list:
                raise ValueError("Predicate {} try to include type {} that has not yet been defined in this program"
                                 .format(predicate.name, argument_type))
            # If this type specified in that Predicate is a string, than change it to corresponding Type
            if type(argument_type) is str:
                predicate.wm_var_types[i] = self.name2type[argument_type]

        # Create and register Variables
        for (var_name, var_type, var_unique) in \
                zip(predicate.wm_var_names, predicate.wm_var_types, predicate.wm_var_unique):
            var = Variable(var_name, var_type.size, var_unique is not None, var_unique)
            predicate.var_list.append(var)
            predicate.var_name2var[var_name] = var

        # Register predicate
        self.predicate_list.append(predicate)
        self.name2predicate[predicate.name] = predicate

    def _register_conditional(self, conditional):
        """
            Register a new conditional in this sigma program. Check against existing type and predicate entries.
                Fill up relevant lookup tables.
        """
        # Check if predicate patterns align with already registered predicates
        for pattern in conditional.conditions + conditional.condacts + conditional.actions:
            # Check if predicate exists
            if pattern.predicate_name not in self.name2predicate.keys():
                raise ValueError("The predicate pattern '{}' includes an unknown predicate '{}'".format(pattern,
                                                                                                        pattern.predicate_name))
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

        # Set up lookup tables
        # TODO: Also add a check to make sure that user does not declare a pattern variable associated with multiple
        #       working memory variables with different uniqueness.
        for pt_name, ptv_vals in conditional.pattern_pt_vals.items():
            for wmv, ptv_val in ptv_vals.items():
                ptv_name = ptv_val["name"]
                # Get corresponding wmv's uniqueness
                pred = self.name2predicate[conditional.name2pattern[pt_name].predicate_name]
                unique = pred.var_name2var[wmv].unique
                if ptv_val["type"] is "const":
                    conditional.global_pt_vals[ptv_name] = {
                        "type": "const",
                        "size": len(ptv_val["vals"]) if type(ptv_val["vals"]) is list else 1,
                        "unique": unique
                    }
                else:
                    ptv_size = pred.var_name2var[ptv_name].size
                    # If variable not already exists, insert entry
                    if ptv_name not in conditional.global_pt_vals.keys():
                        conditional.global_pt_vals[ptv_name] = {
                            "type": "var",
                            "size": ptv_size,
                            "unique": unique
                        }
                    # Otherwise if already present, check whether uniqueness agree, and if so, take max size
                    else:
                        assert conditional.global_pt_vals[ptv_name]["unique"] == unique, \
                            "Found conflict of uniqueness among working memory variables that is associated with the " \
                            "pattern variable '{}'".format(ptv_name)
                        # Take max size
                        conditional.global_pt_vals[ptv_name]["size"] = \
                            max(conditional.global_pt_vals[ptv_name]["size"], ptv_size)

        # If function specified as a str, check whether there already exists a conditional with that name.
        # If so, check if there variable dimensions match
        if type(conditional.function) is str:
            if conditional.function not in self.name2conditional.keys():
                raise ValueError(
                    "Unknown conditional '{}' specified in the function field in the conditional '{}'"
                    .format(conditional.function, conditional.name))
            else:
                alien_cond = self.name2conditional[conditional.function]
                for func_var_name in alien_cond.function_var_names:
                    this_attri = conditional.global_pt_vals[func_var_name]
                    alien_attri = alien_cond.global_pt_vals[func_var_name]
                    if this_attri != alien_attri:
                        raise ValueError("The attributes of the function variable '{}' does not match when attempting"
                                         "to share function between conditional '{}' and '{}'.\n The attributes "
                                         "determined in conditional '{}' is: '{}'.\n The attributes determined in "
                                         "conditional '{}' is: '{}'.\n"
                                         .format(func_var_name, conditional.name, alien_cond.name,
                                                 conditional.name, this_attri, alien_cond.name, alien_attri))

        # Register conditional
        self.conditional_list.append(conditional)
        self.name2conditional[conditional.name] = conditional

    def _compile_predicate(self, predicate):
        """
            Compile the given predicate to the graphical architecture.
        """

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
        # For now assume that whenever a selection is involved, the predicate MUST BE CLOSED_WORLD
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
        # Conditional nodegroup schema:
        #       { conditional :
        #           { "alpha“ ：
        #               { predicate_name :
        #                   { "FFN" : filter factor node,
        #                     "FFN_VN" : wmvn after FFN,
        #                     "NLFN" : nonlinearity factor node,
        #                     "NLFN_VN" : wmvn after NLFN,
        #                     "ADFN" : affine delta factor node,
        #                     "ADFN_VN" : wmvn after ADFN,
        #                     "ATFN" : affine transformation factor node,
        #                     "ATFN_VN" : wmvn after ATFN,
        #                     "terminal" : the terminal variable node at the end of the alpha subnet
        #                   }
        #               },
        #             "beta" :
        #               { predicate_name :
        #                   { "BJFN" : beta join factor node,
        #                     "BJFN_VN" : wmvn after BJFN
        #                   }
        #               },
        #             "gamma" :
        #               { "GFFN" : gamma function factor node,
        #                 "GFFN_VN" : wmvn after GFFN,
        #                 "BJFN" : the ultimate beta-join node that joins GFFN_VN. The peak of the alpha-beta subgraph,
        #                 "BJFN_VN" : the wmvn after BJFN
        #               }
        #           }
        #       }
        nodegroup = dict(alpha={}, beta={}, gamma={})

        name_prefix = conditional.name + "_ALPHA_"
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
                                        [conditional.name2condition_pattern,
                                         conditional.name2condact_pattern,
                                         conditional.name2action_pattern]):
            for pt_name, pattern in pattern_group.items():
                # Initialize the alpha subnet terminal node to be the predicate's WMVN
                pred = self.name2predicate[pattern.predicate_name]
                pred_nodegroup = self.predicate2group[pred]
                terminal = pred_nodegroup['WMVN'] if 'WMVN' in pred_nodegroup.keys() else pred_nodegroup['WMVN_OUT']

                # alpha sub nodegroup for this predicate
                alpha_sub_ng = {}

                # Set up Filter nodes
                # TODO: Detailed implementation left for future iterations

                # Set up Nonlinearity nodes
                if pattern.nonlinearity is not None:
                    # Create nonlinearity factor nodes and set up links
                    nlfn = self.G.new_node(NLFN, name_prefix + "NLFN", pattern.nonlinearity)
                    nlfn_vn = self.G.new_node(WMVN, name_prefix + "NLFN_VN", pred.var_list)
                    terminal = grow_alpha(terminal, nlfn, nlfn_vn, ptype)

                    alpha_sub_ng["NLFN"] = nlfn
                    alpha_sub_ng["NLFM_VN"] = nlfn_vn

                # Set up Affine Delta nodes
                # First create Variables for pattern vars. If one pattern vars is associated with multiple wm vars,
                #   then the size of that pattern var is the max size of all associated wm vars (to leave enough region)
                # TODO: Assume now that the uniqueness of a pattern variable follows that of the wm variables associated
                #       with the pattern variable. Assume already check that there is no conflict of uniquesness among
                #       these wm variables.
                pt_var_list = []
                for ptv_name in conditional.ptv2wmv[pt_name].keys():
                    ptv_type = conditional.global_pt_vals[ptv_name]["type"]
                    ptv_size = conditional.global_pt_vals[ptv_name]["size"]
                    ptv_unique = conditional.global_pt_vals[ptv_name]["unique"]
                    pt_var_list.append(Variable(ptv_name, ptv_size, ptv_unique, selection=False))

                adfn = self.G.new_node(ADFN, name_prefix + "ADFN", pattern.pattern_pt_vals)
                adfn_vn = self.G.new_node(WMVN, name_prefix + "ADFN_VN", pt_var_list)
                terminal = grow_alpha(terminal, adfn, adfn_vn, ptype)

                alpha_sub_ng["ADFN"] = adfn
                alpha_sub_ng["ADFN_VN"] = adfn_vn

                # Set up Affine Transformation nodes
                # TODO: Detailed implementation left for future iterations

                # Alpha subnet construction finished, register terminal node
                alpha_terminals[pattern.predicate_name] = terminal

                alpha_sub_ng["terminal"] = terminal

                # Register this alpha sub nodegroup in the global nodegroup
                nodegroup["alpha"][pred.name] = alpha_sub_ng

        # Step 2: Compile Beta network

        name_prefix = conditional.name + "_BETA_"

        # util inner function for taking the union of pattern variable lists. Define such a custom union operation of
        #   lists because we may want to maintain variable orders
        def union_vars(vars_1, vars_2):
            # Assume both arguments are list of Variables
            var_list = [var for var in vars_1]     # Make a replicate (deepcopy) of vars_1 list but not the element of it
            for var in vars_2:
                if var not in vars_1:
                    var_list.append(var)
            return var_list

        # Beta nodegroup
        beta_ng = {}

        # Part 1: Linear growth of conditions Beta subnets
        terminal = None
        if len(conditional.conditions) >= 1:
            terminal = alpha_terminals[conditional.conditions[0]]
        if len(conditional.conditions) > 1:
            for i in range(1, len(conditional.conditions)):
                extra = alpha_terminals[conditional.conditions[i]]

                var_union = union_vars(terminal.var_list, extra.var_list)
                bjfn = self.G.new_node(BJFN, name_prefix + "BJFN")
                bjfn_vn = self.G.new_node(WMVN, name_prefix + "BJFN_VN", var_union)

                # Condition Beta subnets link direction going inward
                self.G.add_unilink(terminal, bjfn)
                self.G.add_unilink(extra, bjfn)
                self.G.add_unilink(bjfn, bjfn_vn)

                # Register nodes in nodegroup. Note that the first two joining pattern in the linear join beta network
                #   has same BJFN and BJFN_VN
                if i == 1:
                    # First pattern's nodes
                    beta_ng[conditional.conditions[0].predicate_name] = \
                        dict(BJFN=bjfn, BJFN_VN=bjfn_vn)
                # All other patterns' nodes
                beta_ng[conditional.conditions[i].predicate_name] = \
                    dict(BJFN=bjfn, BJFN_VN=bjfn_vn)

                # Update terminal node
                terminal = bjfn_vn

        # Part 2: Linear growth of condacts Beta subnets
        if len(conditional.condacts) >= 1:
            if terminal is None:
                terminal = alpha_terminals[conditional.condacts[0]]
            else:
                # When a condition terminal is joined with a condact terminal, the link between the condition terminal
                #   and the BJFN is unidirectional
                extra = alpha_terminals[conditional.condacts[0]]

                var_union = union_vars(terminal.var_list, extra.var_list)
                bjfn = self.G.new_node(BJFN, name_prefix + "BJFN")
                bjfn_vn = self.G.new_node(WMVN, name_prefix + "BJFN_VN", var_union)

                self.G.add_unilink(terminal, bjfn)      # uni-link from condition terminal
                self.G.add_bilink(extra, bjfn)          # bi-link from condact terminal
                self.G.add_bilink(bjfn, bjfn_vn)        # bi-link from now on

                # In this case there are conditions, so only care about this first condact's nodegroup
                beta_ng[conditional.condacts[0].predicate_name] = dict(BJFN=bjfn, BJFN_VN=bjfn_vn)

                terminal = bjfn_vn
        if len(conditional.condacts) > 1:
            for i in range(1, len(conditional.condacts)):
                extra = alpha_terminals[conditional.condacts[i]]

                var_union = union_vars(terminal.var_list, extra.var_list)
                bjfn = self.G.new_node(BJFN, name_prefix + "BJFN")
                bjfn_vn = self.G.new_node(WMVN, name_prefix + "BJFN_VN", var_union)

                # Bidirectional links in condact Beta subnets
                self.G.add_bilink(terminal, bjfn)
                self.G.add_bilink(extra, bjfn)
                self.G.add_bilink(bjfn, bjfn_vn)

                # if there are no conditions, and now we have two condacts, then both the first two condacts have same
                #   BJFN and BJFN_VN
                if i == 1 and len(conditional.conditions) == 0:
                    beta_ng[conditional.condacts[0].predicate_name] = \
                        dict(BJFN=bjfn, BJFN_VN=bjfn_vn)
                # All other condact (including the second one) have normal nodegroup
                beta_ng[conditional.condacts[i].predicate_name] = dict(BJFN=bjfn, BJFN_VN=bjfn_vn)

                terminal = bjfn_vn

        # Register Beta subnet nodegroup
        nodegroup["beta"] = beta_ng

        # Part 3: Joining of Gamma function nodes
        # TODO: Consider whether it is necessary to build a separate gamma factor/variable, i.e., function may be
        #   included in the ultimate Beta join node
        gffn, gffn_vn = None, None
        # If function is specified as a str, find that conditional's gamma function nodes
        if type(conditional.function) is str:
            alien_cond = self.name2conditional[conditional.function]
            gffn = self.conditional2group[alien_cond]["gamma"]["GFFN"]
            _, func_var_list = gffn.get_function()
            gffn_vn = self.G.new_node(WMVN, name_prefix + "GFFN_VN", func_var_list)
        # Otherwise, set gamma function normally
        else:
            # Create function variable list
            func_var_list = []
            for var_name in conditional.function_var_names:
                var_size = conditional.global_pt_vals["size"]
                var_unique = conditional.global_pt_vals["unique"]
                func_var_list.append(Variable(var_name, var_size, var_unique, selection=False))

            gffn = self.G.new_node(GFFN, name_prefix + "GFFN", func_var_list)
            gffn_vn = self.G.new_node(WMVN, name_prefix + "GFFN_VN", func_var_list)

        # Gamma Beta join nodes
        gbjn = self.G.new_node(BJFN, name_prefix + "GammaBetaJoin")
        gbjn_vn = self.G.new_node(WMVN, name_prefix + "GammaBetaJoin_VN", terminal.var_list)

        # Connect
        self.G.add_bilink(gffn, gffn_vn)
        self.G.add_bilink(gffn_vn, gbjn)
        self.G.add_bilink(terminal, gbjn)
        self.G.add_bilink(gbjn, gbjn_vn)

        # Register gamma subnet nodegroup
        nodegroup["gamma"] = dict(GFFN=gffn, GFFN_VN=gffn_vn, BJFN=gbjn, BJFN_VN=gbjn_vn)

        # Part 4: Final part, connect action predicate's alpha terminals to gamma subnet BJFN_VN by creating branching
        #   factor nodes
        for pattern in conditional.actions:
            term = alpha_terminals[pattern.predicate_name]

            # branching factor node. For actions we connect unidirectionally
            bjfn = self.G.new_node(BJFN, name_prefix + "BetaBranch")
            self.G.add_unilink(gbjn_vn, bjfn)
            self.G.add_unilink(bjfn, term)

            # Register. Note: we register the ultimate Beta join variable node as actions' beta subnet's BJFN_VN here
            nodegroup[pattern.predicate_name] = dict(BJFN=bjfn, BJFN_VN=gbjn_vn)

