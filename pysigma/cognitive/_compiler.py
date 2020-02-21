"""
    Class methods for the Sigma main class

    Compiler implementation - compile predicates and conditionals into underlying factor subgraph
"""
from ..graphical._nodes import *
from ..graphical._structures import *


def _compile_predicate(self, predicate):
    """
        Compile the given predicate to the graphical architecture.

        Predicate nodegroup schema:
            { predicate_name:
                { "WMVN": working memory variable node (if no selection),
                  "WMVN_IN": working memory variable node that receives incoming messages (if selection is on),
                  "WMVN_OUT": working memory variable node that receives outgoing messages (if selection is on),
                  "PBFN": perceptual buffer function node (if perception is on),
                  "LTMFN": long-term memory factor node (if memorial),
                  "WMFN": working memory function node (if closed world),
                  "WMFN_VN": vn in the loop that precedes the ACFN (if closed world),
                  "ACFN": action combination factor node (if closed world)
                }
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

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
            assert "LTMFN" in self.predicate2group[func].keys(), \
                "Target predicate {} exists but it does not have a Long-Term Memory Function (LTMFN)".format(func)

            target_ltmfn = self.predicate2group[func]['LTMFN']
            # Set up bidirectional link between target LTMFN and own WMVN
            self.G.add_bilink(target_ltmfn, wmvn)

        # Else, set up own ltmfn
        else:
            ltmfn = self.G.new_node(LTMFN, predicate.name + "_LTMFN")
            ltmfn.set_function(func, predicate.var_list)  # for a pred function, var list is simply own var list
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

    # Register node group. Use predicate name to index
    self.predicate2group[predicate.name] = nodegroup


def _compile_conditional(self, conditional):
    """
        Compile the given conditional to the graphical architecture.
        A Sigma's implementation of Rete algorithm

        Conditional nodegroup schema:
          { conditional_name :
              { "alpha“ ：
                  { pattern_name :
                      { "FFN" : filter factor node,
                        "FFN_VN" : wmvn after FFN,
                        "NLFN" : nonlinearity factor node,
                        "NLFN_VN" : wmvn after NLFN,
                        "ADFN" : affine delta factor node,
                        "ADFN_VN" : wmvn after ADFN,
                        "ATFN" : affine transformation factor node,
                        "ATFN_VN" : wmvn after ATFN,
                        "terminal" : the terminal variable node at the end of the alpha subnet
                      }
                  },
                "beta" :
                  { pattern_name :
                      { "BJFN" : beta join factor node,
                        "BJFN_VN" : wmvn after BJFN
                      }
                  },
                "gamma" :
                  { "GFFN" : gamma function factor node,
                    "GFFN_VN" : wmvn after GFFN,
                    "BJFN" : the ultimate beta-join node that joins GFFN_VN. The peak of the alpha-beta subgraph,
                    "BJFN_VN" : the wmvn after BJFN
                  }
              }
          }
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    nodegroup = dict(alpha={}, beta={}, gamma={})

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

    alpha_terminals = {}  # map from predicate to the terminal variable node of that predicate's alpha subnet
    for ptype, pattern_group in zip(["condition", "condact", "action"],
                                    [conditional.name2condition_pattern,
                                     conditional.name2condact_pattern,
                                     conditional.name2action_pattern]):
        for pt_name, pattern in pattern_group.items():
            # name prefix for each predicate pattern's alpha subnet
            name_prefix = conditional.name + "_ALPHA_{" + pt_name + "}_"

            # Initialize the alpha subnet terminal node to be the predicate's WMVN
            pred = self.name2predicate[pattern.predicate_name]
            pred_nodegroup = self.predicate2group[pattern.predicate_name]
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
                ptv_size = conditional.global_pt_vals[ptv_name]["size"]
                ptv_unique = conditional.global_pt_vals[ptv_name]["unique"]
                pt_var_list.append(Variable(ptv_name, ptv_size, ptv_unique, selection=False))

            adfn = self.G.new_node(ADFN, name_prefix + "ADFN", conditional.pattern_pt_vals[pt_name])
            adfn_vn = self.G.new_node(WMVN, name_prefix + "ADFN_VN", pt_var_list)
            terminal = grow_alpha(terminal, adfn, adfn_vn, ptype)

            alpha_sub_ng["ADFN"] = adfn
            alpha_sub_ng["ADFN_VN"] = adfn_vn

            # Set up Affine Transformation nodes
            # TODO: Detailed implementation left for future iterations

            # Alpha subnet construction finished, register terminal node
            alpha_terminals[pt_name] = terminal

            alpha_sub_ng["terminal"] = terminal

            # Register this alpha sub nodegroup in the global nodegroup
            nodegroup["alpha"][pt_name] = alpha_sub_ng

    # Step 2: Compile Beta network
    # util inner function for taking the union of pattern variable lists. Define such a custom union operation of
    #   lists because we may want to maintain variable orders
    def union_vars(vars_1, vars_2):
        # Assume both arguments are list of Variables
        var_list = [var for var in vars_1]  # Make a replicate (deepcopy) of vars_1 list but not the element of it
        for var in vars_2:
            if var not in vars_1:
                var_list.append(var)
        return var_list

    # Beta nodegroup
    beta_ng = {}
    name_prefix = conditional.name + "_BETA_"

    # Part 1: Linear growth of conditions Beta subnets
    terminal = None
    if len(conditional.conditions) >= 1:
        terminal = alpha_terminals[conditional.condition_pt_names[0]]
    if len(conditional.conditions) > 1:
        for i in range(1, len(conditional.conditions)):
            # name prefix for the join node
            # name_prefix = conditional.name + "_BETA_{" \
            #               + conditional.condition_pt_names[i - 1] + "}*{" \
            #               + conditional.condition_pt_names[i] + "}_"

            extra = alpha_terminals[conditional.condition_pt_names[i]]

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
                beta_ng[conditional.condition_pt_names[0]] = \
                    dict(BJFN=bjfn, BJFN_VN=bjfn_vn)
            # All other patterns' nodes
            beta_ng[conditional.condition_pt_names[i]] = \
                dict(BJFN=bjfn, BJFN_VN=bjfn_vn)

            # Update terminal node
            terminal = bjfn_vn

    # Part 2: Linear growth of condacts Beta subnets
    if len(conditional.condacts) >= 1:
        if terminal is None:
            terminal = alpha_terminals[conditional.condact_pt_names[0]]
        else:
            # name prefix for the join node
            # name_prefix = conditional.name + "_BETA_{" \
            #               + conditional.condition_pt_names[i - 1] + "}*{" \
            #               + conditional.condition_pt_names[i] + "}_"

            # When a condition terminal is joined with a condact terminal, the link between the condition terminal
            #   and the BJFN is unidirectional
            extra = alpha_terminals[conditional.condact_pt_names[0]]

            var_union = union_vars(terminal.var_list, extra.var_list)
            bjfn = self.G.new_node(BJFN, name_prefix + "BJFN")
            bjfn_vn = self.G.new_node(WMVN, name_prefix + "BJFN_VN", var_union)

            self.G.add_unilink(terminal, bjfn)  # uni-link from condition terminal
            self.G.add_bilink(extra, bjfn)  # bi-link from condact terminal
            self.G.add_bilink(bjfn, bjfn_vn)  # bi-link from now on

            # In this case there are conditions, so only care about this first condact's nodegroup
            beta_ng[conditional.condact_pt_names[0]] = dict(BJFN=bjfn, BJFN_VN=bjfn_vn)

            terminal = bjfn_vn
    if len(conditional.condacts) > 1:
        for i in range(1, len(conditional.condacts)):
            # name prefix for the join node
            # name_prefix = conditional.name + "_BETA_{" \
            #               + conditional.condition_pt_names[i - 1] + "}*{" \
            #               + conditional.condition_pt_names[i] + "}_"

            extra = alpha_terminals[conditional.condact_pt_names[i]]

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
                beta_ng[conditional.condact_pt_names[0]] = \
                    dict(BJFN=bjfn, BJFN_VN=bjfn_vn)
            # All other condact (including the second one) have normal nodegroup
            beta_ng[conditional.condact_pt_names[i]] = dict(BJFN=bjfn, BJFN_VN=bjfn_vn)

            terminal = bjfn_vn

    # Register Beta subnet nodegroup
    nodegroup["beta"] = beta_ng

    # Part 3: Joining of Gamma function nodes, if conditional function is specified
    # TODO: Consider whether it is necessary to build a separate gamma factor/variable, i.e., function may be
    #   included in the ultimate Beta join node
    # If function is specified as a str, find that conditional's gamma function nodes
    if conditional.function is not None:
        # name prefix for gamma join nodes
        name_prefix = conditional.name + "_GAMMA_"

        if type(conditional.function) is str:
            gffn = self.conditional2group[conditional.function]["gamma"]["GFFN"]
            _, func_var_list = gffn.get_function()
            gffn_vn = self.G.new_node(WMVN, name_prefix + "GFFN_VN", func_var_list)
        # Otherwise, set gamma function normally
        else:
            # Create function variable list
            func_var_list = []
            for var_name in conditional.function_var_names:
                var_size = conditional.global_pt_vals[var_name]["size"]
                var_unique = conditional.global_pt_vals[var_name]["unique"]
                func_var_list.append(Variable(var_name, var_size, var_unique, selection=False))

            gffn = self.G.new_node(GFFN, name_prefix + "GFFN", conditional.function, func_var_list)
            gffn_vn = self.G.new_node(WMVN, name_prefix + "GFFN_VN", func_var_list)

        # Gamma Beta join nodes
        gbjn = self.G.new_node(BJFN, name_prefix + "GammaBetaJoin")
        gbjn_vn = self.G.new_node(WMVN, name_prefix + "GammaBetaJoin_VN", terminal.var_list)

        # Connect
        self.G.add_bilink(gffn, gffn_vn)
        self.G.add_bilink(gffn_vn, gbjn)
        self.G.add_bilink(terminal, gbjn)
        self.G.add_bilink(gbjn, gbjn_vn)

        # Set GBJN variable node as the new terminal node
        terminal = gbjn_vn

        # Register gamma subnet nodegroup
        nodegroup["gamma"] = dict(GFFN=gffn, GFFN_VN=gffn_vn, BJFN=gbjn, BJFN_VN=gbjn_vn)

    # Part 4: Final part, connect action predicate's alpha terminals to gamma subnet BJFN_VN by creating branching
    #   factor nodes
    for pt_name, pattern in conditional.name2action_pattern.items():
        # name prefix for branch node
        name_prefix = conditional.name + "_BETA_{" + pt_name + "}_"

        term = alpha_terminals[pt_name]

        # branching factor node. For actions we connect unidirectionally
        bjfn = self.G.new_node(BJFN, name_prefix + "BetaBranch")
        self.G.add_unilink(terminal, bjfn)
        self.G.add_unilink(bjfn, term)

        # Register. Note: we register the ultimate Beta join variable node as actions' beta subnet's BJFN_VN here
        nodegroup["beta"][pt_name] = dict(BJFN=bjfn, BJFN_VN=terminal)

    # Regiser nodegroup. Use conditional's name to index
    self.conditional2group[conditional.name] = nodegroup

    # Debug
    pass