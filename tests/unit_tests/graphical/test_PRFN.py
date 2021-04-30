"""
    Unit tests for PRFN class
"""
import pytest
from unittest.mock import MagicMock
from typing import List, Type

import torch
import torch.distributions.constraints as C
import torch.distributions as D
import torch.distributions.transforms as T
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, DVN, VariableNode
from pysigma.graphical.beta_nodes import PRFN
from pysigma.utils import KnowledgeServer as KS
from pysigma.pattern_structures.event_transform import EventTransform
from ...utils import random_message, assert_equal_within_error, assert_proportional_within_error, \
    generate_positive_definite


# Utility methods
def gen_vn(ran_vars, rel_vars):
    # Generate mocked variable node
    mock_vn = MagicMock(spec=VariableNode)
    mock_vn.ran_vars, mock_vn.rel_vars = ran_vars, rel_vars
    return mock_vn


def gen_ld(fn, vn, incoming: bool, inward: bool, alien: bool = False, name=''):
    # Generate mocked linkdata
    mock_ld = MagicMock(spec=LinkData)
    mock_ld.fn, mock_ld.vn = fn, vn
    mock_ld.to_fn = incoming
    mock_ld.attr = {'direction': 'inward' if inward else 'outward'}
    # Add alien attribute if incoming
    if incoming:
        mock_ld.attr['alien'] = alien
    # repr
    mock_ld.__repr__ = MagicMock(return_value=name)
    return mock_ld


def gen_ks(dist_class: Type[D.Distribution], pred_args: List[Variable]):
    # Generate a simple KnowledgeServer based on given predicate arguments
    #   - Use argument's first constraint
    #   - No num_particles
    #   - dist_info defaults to indicate using Regular type distribution parameter
    return KS(dist_class, [arg.size for arg in pred_args], [arg.constraints[0] for arg in pred_args],
              dist_info={'param_type': 'regular'})


def gen_multi_normal_param(b_shape: torch.Size, e_size: int):
    # Generate a random MultivariateNormal distribution parameter
    loc = torch.randn(list(b_shape) + [e_size])
    cov = generate_positive_definite(b_shape, e_size)
    flattened_cov = cov.view(list(b_shape) + [e_size ** 2])
    param = torch.cat([loc, flattened_cov], dim=-1)
    return param


# Utility constants
REL_VARS = [
    Variable('test_rel_var_1', VariableMetatype.Relational, 6),
    Variable('test_rel_var_2', VariableMetatype.Relational, 7),
    Variable('test_rel_var_3', VariableMetatype.Relational, 8),
]


class TestPRFN_Init:

    def test_init_1(self):
        # Test init with
        #   - No concatenated arguments
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])

        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        assert prfn.event_transforms == (pattern_1, pattern_2, pattern_3)
        assert prfn.knowledge_server is ks
        assert prfn.et_pat_vars == (pat_var_1, pat_var_2, pat_var_3)
        assert prfn.et_flattened_pred_args == [pred_arg_1, pred_arg_2, pred_arg_3]
        assert prfn.et_surrogate_pred_args == [pred_arg_1, pred_arg_2, pred_arg_3]

    def test_init_2(self):
        # Test init with
        #   - With concatenated arguments
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 7, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform([pred_arg_1, pred_arg_2], pat_var_1)
        pattern_2 = EventTransform(pred_arg_3, pat_var_2)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])

        prfn = PRFN('test_prfn', [pattern_1, pattern_2,], ks)

        assert prfn.event_transforms == (pattern_1, pattern_2)
        assert prfn.knowledge_server is ks
        assert prfn.et_pat_vars == (pat_var_1, pat_var_2)
        assert prfn.et_flattened_pred_args == [pred_arg_1, pred_arg_2, pred_arg_3]
        assert prfn.et_surrogate_pred_args == [pattern_1.surrogate_pred_arg, pred_arg_3]


class TestPRFN_AddLink:

    def test_error_outward_linkdata(self):
        # Test AssertionError is raised if attempt to register outward directional linkdata
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        inward_vn_1 = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        # Test Linkdata
        # Test outward linkdata
        outward_ld = gen_ld(prfn, inward_vn_1, True, False)

        with pytest.raises(AssertionError) as excinfo:
            prfn.add_link(outward_ld)
        assert str(excinfo.value) == "In test_prfn: PRFN only accept inward-directional linkdata."

    def test_error_multiple_outgoing_linkdata(self):
        # Test AssertionError is raised if attempt to register more than one outgoing linkdata
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        inward_vn_1 = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        # Test Linkdata
        out_ld_1, out_ld_2 = gen_ld(prfn, inward_vn_1, False, True), gen_ld(prfn, inward_vn_1, False, True)

        # Should be fine to add the first one
        prfn.add_link(out_ld_1)
        # Problems when adding second one
        with pytest.raises(AssertionError) as excinfo:
            prfn.add_link(out_ld_2)
        assert str(excinfo.value) == "In test_prfn: PRFN only accept one inward outgoing linkdata."

    def test_error_outgoing_linkdata_conflicting_pred_args(self):
        # Test AssertionError is raised if predicate arguments inferred from the outgoing linkdata does not match those
        #   inferred from the patterns
        # Test with argument concatenation
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 7, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform([pred_arg_1, pred_arg_2], pat_var_1)
        pattern_2 = EventTransform(pred_arg_3, pat_var_2)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2], ks)

        # Mock Variable node
        inward_vn_1 = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)    # Should be the surrogate pred args
        # Test Linkdata
        out_ld_1 = gen_ld(prfn, inward_vn_1, False, True, name='out_ld_1')

        with pytest.raises(AssertionError) as excinfo:
            prfn.add_link(out_ld_1)
        assert str(excinfo.value) == "In test_prfn: The predicate arguments inferred from the outgoing linkdata must " \
                                     "be a superset of the set of surrogate predicate arguments inferred from the " \
                                     "event transforms. Expect a superset of arguments {}, but instead found " \
                                     "arguments [pred_arg_1, pred_arg_2, pred_arg_3] in linkdata 'out_ld_1'." \
                                     .format([pattern_1.surrogate_pred_arg, pred_arg_3])

    def test_error_special_attribute_not_found(self):
        # Test AssertionError is raised if the 'alien' special attribute is not found in incoming linkdata
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        inward_vn_1 = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        # Test Linkdata
        in_ld = gen_ld(prfn, inward_vn_1, True, True)
        in_ld.attr.pop('alien', None)   # Delete 'alien' attribute from the special attribute dict

        with pytest.raises(AssertionError) as excinfo:
            prfn.add_link(in_ld)
        assert str(excinfo.value) == "In test_prfn: Incoming linkdata to a PRFN must be tagged with `alien` special " \
                                     "attribute, whose value must be boolean."

    def test_error_special_attribute_wrong_value(self):
        # Test AssertionError is raised if the 'alien' special attribute does not have boolean value in incoming ld
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        inward_vn_1 = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        # Test Linkdata
        in_ld = gen_ld(prfn, inward_vn_1, True, True)
        in_ld.attr['alien'] = 'something random'    # Change attribute value to something random

        with pytest.raises(AssertionError) as excinfo:
            prfn.add_link(in_ld)
        assert str(excinfo.value) == "In test_prfn: Incoming linkdata to a PRFN must be tagged with `alien` special " \
                                     "attribute, whose value must be boolean."

    def test_error_multiple_home_linkdata(self):
        # Test AssertionError is raised if attempt to register more than one HOME linkdata
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        inward_vn_1 = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        # Test Linkdata
        home_ld_1, home_ld_2 = gen_ld(prfn, inward_vn_1, True, True, alien=False, name='home_ld_1'), \
                               gen_ld(prfn, inward_vn_1, True, True, alien=False, name='home_ld_2')

        # Ok to register the first one
        prfn.add_link(home_ld_1)
        # Problems when adding second one
        with pytest.raises(AssertionError) as excinfo:
            prfn.add_link(home_ld_2)
        assert str(excinfo.value) == "In test_prfn: PRFN can only accept one HOME branch incoming linkdata. Linkdata " \
                                     "'home_ld_1' is already registered as the HOME incoming linkdata."

    def test_error_home_linkdata_conflicting_pred_arg(self):
        # Test AssertionError is raised predicate arguments inferred from the HOME linkdata is not a superset of the
        #   predicate arguments inferred from the patterns
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        inward_vn_1 = gen_vn([pred_arg_1, pred_arg_3], REL_VARS)    # One less predicate argument
        # Test Linkdata
        home_ld = gen_ld(prfn, inward_vn_1, True, True, alien=False, name='home_ld_1')

        with pytest.raises(AssertionError) as excinfo:
            prfn.add_link(home_ld)
        assert str(excinfo.value) == "In test_prfn: The set of predicate arguments inferred from the HOME branch " \
                                     "incoming linkdata must be a superset of the set of predicate arguments inferred "\
                                     "from the event transforms given during initialization. Expect a superset of the "\
                                     "subset {}, however found {}.".format({pred_arg_1, pred_arg_2, pred_arg_3},
                                                                           {pred_arg_1, pred_arg_3})

    def test_error_alien_linkdata_multivariate(self):
        # Test AssertionError is raised if the alien linkdata does not connect to a univariate node
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        aux_vn_1 = gen_vn([pat_var_1, pat_var_2], REL_VARS)     # Multivariate pat var node
        # Test Linkdata
        alien_ld = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld')

        with pytest.raises(AssertionError) as excinfo:
            prfn.add_link(alien_ld)
        assert str(excinfo.value) == "In test_prfn: Every ALIEN incoming linkdata to a PRFN must connect from a " \
                                     "univariate variable node. Found linkdata 'alien_ld' is connected from a node " \
                                     "with random variables [pat_var_1, pat_var_2]."

    def test_error_alien_linkdata_duplicate_node(self):
        # Test AssertionError is raised if two alien linkdata connects to nodes that have the same random variables
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        aux_vn_1, aux_vn_2 = gen_vn([pat_var_1], REL_VARS), gen_vn([pat_var_1], REL_VARS)  # Duplicate nodes
        # Test Linkdata
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')
        alien_ld_2 = gen_ld(prfn, aux_vn_2, True, True, alien=True, name='alien_ld_2')

        prfn.add_link(alien_ld_1)
        with pytest.raises(AssertionError) as excinfo:
            prfn.add_link(alien_ld_2)
        assert str(excinfo.value) == "In test_prfn: Every ALIEN incoming linkdata to a PRFN must connect from a " \
                                     "distinct variable node. Found linkdata 'alien_ld_2' is connected from a node " \
                                     "with random variable 'pat_var_1' that is already registered."

    def test_error_alien_linkdata_unknown_pat_var(self):
        # Test AssertionError is raised if the alien linkdata connects to a node with unknown random variable
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)       # pat_var_3 is not referenced in any pattern
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2], ks)

        # Mock Variable node
        aux_vn_1 = gen_vn([pat_var_3], REL_VARS)        # Node with unknown pattern variable pat_var_3
        # Test Linkdata
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')

        with pytest.raises(AssertionError) as excinfo:
            prfn.add_link(alien_ld_1)
        assert str(excinfo.value) == "In test_prfn: Every ALIEN incoming linkdata to a PRFN must connect from a " \
                                     "variable node whose random variable must be referenced as the pattern variable " \
                                     "by at least one event transforms declared during initialization. Found linkdata "\
                                     "'alien_ld_1' with unknown pattern variable 'pat_var_3'."

    def test_no_concat_no_inner_binding_full_patterns(self):
        # Test init with
        #   - No predicate argument concatenation
        #   - No inner pattern binding
        #   - Patterns covers all pred arg and pat vars
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1, aux_vn_2, aux_vn_3 = gen_vn([pat_var_1], REL_VARS), gen_vn([pat_var_2], REL_VARS), \
                                       gen_vn([pat_var_3], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')
        alien_ld_2 = gen_ld(prfn, aux_vn_2, True, True, alien=True, name='alien_ld_2')
        alien_ld_3 = gen_ld(prfn, aux_vn_3, True, True, alien=True, name='alien_ld_3')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)
        prfn.add_link(alien_ld_2)
        prfn.add_link(alien_ld_3)

        # Check internal data structures
        assert prfn.home_ld_pred_args == ([pred_arg_1, pred_arg_2, pred_arg_3])
        assert prfn.out_ld_pred_args == ([pred_arg_1, pred_arg_2, pred_arg_3])
        assert prfn.et2alien_ld == {
            pattern_1: alien_ld_1,
            pattern_2: alien_ld_2,
            pattern_3: alien_ld_3
        }
        assert prfn.alien_ld_pat_vars == [pat_var_1, pat_var_2, pat_var_3]
        assert prfn.home_ld is home_ld
        assert prfn.alien_lds == [alien_ld_1, alien_ld_2, alien_ld_3]

    def test_no_concat_no_inner_binding_partial_patterns(self):
        # Test init with
        #   - No predicate argument concatenation
        #   - No inner pattern binding
        #   - Patterns do not cover all pred arg and pat vars
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1, aux_vn_2 = gen_vn([pat_var_1], REL_VARS), gen_vn([pat_var_2], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')
        alien_ld_2 = gen_ld(prfn, aux_vn_2, True, True, alien=True, name='alien_ld_2')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)
        prfn.add_link(alien_ld_2)

        # Check internal data structures
        assert prfn.home_ld_pred_args == ([pred_arg_1, pred_arg_2, pred_arg_3])
        assert prfn.out_ld_pred_args == ([pred_arg_1, pred_arg_2, pred_arg_3])
        assert prfn.et2alien_ld == {
            pattern_1: alien_ld_1,
            pattern_2: alien_ld_2
        }
        assert prfn.alien_ld_pat_vars == [pat_var_1, pat_var_2]
        assert prfn.home_ld is home_ld
        assert prfn.alien_lds == [alien_ld_1, alien_ld_2]

    def test_no_concat_with_inner_binding_partial_patterns(self):
        # Test init with
        #   - No predicate argument concatenation
        #   - With inner pattern binding
        #   - Patterns do not cover all pred arg and pat vars
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 3, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_1)   # Binds on pat_var_1
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1 = gen_vn([pat_var_1], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)

        # Check internal data structures
        assert prfn.home_ld_pred_args == ([pred_arg_1, pred_arg_2, pred_arg_3])
        assert prfn.out_ld_pred_args == ([pred_arg_1, pred_arg_2, pred_arg_3])
        assert prfn.et2alien_ld == {
            pattern_1: alien_ld_1,
            pattern_2: alien_ld_1
        }
        assert prfn.alien_ld_pat_vars == [pat_var_1]
        assert prfn.home_ld is home_ld
        assert prfn.alien_lds == [alien_ld_1]

    def test_with_concat_no_inner_binding_full_patterns(self):
        # Test init with
        #   - With predicate argument concatenation
        #   - No inner pattern binding
        #   - Patterns cover all pred arg and pat vars
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 7, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform([pred_arg_1, pred_arg_2], pat_var_1)     # Pred arg concat
        pattern_2 = EventTransform(pred_arg_3, pat_var_2)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2], ks)

        # Mock Variable node
        out_vn = gen_vn([pattern_1.surrogate_pred_arg, pattern_2.surrogate_pred_arg], REL_VARS) # Surrogate pred args
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1, aux_vn_2 = gen_vn([pat_var_1], REL_VARS), gen_vn([pat_var_2], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')
        alien_ld_2 = gen_ld(prfn, aux_vn_2, True, True, alien=True, name='alien_ld_2')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)
        prfn.add_link(alien_ld_2)

        # Check internal data structures
        assert prfn.home_ld_pred_args == ([pred_arg_1, pred_arg_2, pred_arg_3])
        assert prfn.out_ld_pred_args == ([pattern_1.surrogate_pred_arg, pattern_2.surrogate_pred_arg])
        assert prfn.et2alien_ld == {
            pattern_1: alien_ld_1,
            pattern_2: alien_ld_2
        }
        assert prfn.alien_ld_pat_vars == [pat_var_1, pat_var_2]
        assert prfn.home_ld is home_ld
        assert prfn.alien_lds == [alien_ld_1, alien_ld_2]


class TestPRFN_PreComputeCheck:

    def test_error_missing_home_linkdata(self):
        # Test that AssertionError is raised if HOME linkdata is registered
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1, aux_vn_2, aux_vn_3 = gen_vn([pat_var_1], REL_VARS), gen_vn([pat_var_2], REL_VARS), \
                                       gen_vn([pat_var_3], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')
        alien_ld_2 = gen_ld(prfn, aux_vn_2, True, True, alien=True, name='alien_ld_2')
        alien_ld_3 = gen_ld(prfn, aux_vn_3, True, True, alien=True, name='alien_ld_3')

        # Missing HOME linkdata
        prfn.add_link(out_ld)
        prfn.add_link(alien_ld_1)
        prfn.add_link(alien_ld_2)
        prfn.add_link(alien_ld_3)

        with pytest.raises(AssertionError) as excinfo:
            prfn.compute()
        assert str(excinfo.value) == "In test_prfn: One HOME incoming linkdata must be registered to start computation."

    def test_error_missing_alien_linkdata(self):
        # Test that AssertionError is raised if ALIEN linkdata is registered
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1, aux_vn_2, aux_vn_3 = gen_vn([pat_var_1], REL_VARS), gen_vn([pat_var_2], REL_VARS), \
                                       gen_vn([pat_var_3], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')
        alien_ld_2 = gen_ld(prfn, aux_vn_2, True, True, alien=True, name='alien_ld_2')
        alien_ld_3 = gen_ld(prfn, aux_vn_3, True, True, alien=True, name='alien_ld_3')

        # Missing ALIEN linkdata
        prfn.add_link(out_ld)
        prfn.add_link(home_ld)

        with pytest.raises(AssertionError) as excinfo:
            prfn.compute()
        assert str(excinfo.value) == "In test_prfn: At least one ALIEN incoming linkdata must be registered to start " \
                                     "computation."

    def test_error_incomplete_pattern_variable(self):
        # Test that AssertionError is raised if some pattern variable referenced by patterns does not correspond to
        #   any ALIEN linkdata
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        pattern_2 = EventTransform(pred_arg_2, pat_var_2)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2, pattern_3], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1, aux_vn_2, aux_vn_3 = gen_vn([pat_var_1], REL_VARS), gen_vn([pat_var_2], REL_VARS), \
                                       gen_vn([pat_var_3], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')
        alien_ld_2 = gen_ld(prfn, aux_vn_2, True, True, alien=True, name='alien_ld_2')
        alien_ld_3 = gen_ld(prfn, aux_vn_3, True, True, alien=True, name='alien_ld_3')

        # pat_var_3 missing
        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)
        prfn.add_link(alien_ld_2)

        with pytest.raises(AssertionError) as excinfo:
            prfn.compute()
        assert str(excinfo.value) == "In test_prfn: All pattern variables that are referenced by the event transforms " \
                                     "must also be referenced by the ALIEN incoming linkdata. Inferred pattern " \
                                     "variables (pat_var_1, pat_var_2, pat_var_3) from the event transforms, while " \
                                     "inferred [pat_var_1, pat_var_2] from the ALIEN linkdata."

    def test_error_conflicting_outgoing_pred_args(self):
        # Test that AssertionError is raised if the outgoing predicate arguments do not match
        #   - Test with argument concatenation
        #   - Partial pattern variables
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 7, (C.real,))
        # Patterns
        pattern_1 = EventTransform([pred_arg_1, pred_arg_2], pat_var_1)
        # KnowledgeServer
        ks = gen_ks(D.Normal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1], ks)

        # Mock Variable node
        out_vn = gen_vn([pattern_1.surrogate_pred_arg], REL_VARS)       # Should also have pred_arg_3
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1 = gen_vn([pat_var_1], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')

        # pat_var_3 missing
        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)

        with pytest.raises(AssertionError) as excinfo:
            prfn.compute()
        # Due to set order invariant
        assert str(excinfo.value) == "In test_prfn: Unexpected set of outgoing predicate arguments found in the " \
                                     "outgoing linkdata. Expect set of predicate arguments {}, instead found {}" \
                                     .format([pattern_1.surrogate_pred_arg, pred_arg_3], [pattern_1.surrogate_pred_arg]) \
            or str(excinfo.value) == "In test_prfn: Unexpected set of outgoing predicate arguments found in the " \
                                     "outgoing linkdata. Expect set of predicate arguments {}, instead found {}" \
                                     .format([pred_arg_3, pattern_1.surrogate_pred_arg], [pattern_1.surrogate_pred_arg])


class TestPRFN_InwardCompute:

    def test_error_home_msg_wrong_type(self):
        # Test that a ValueError is raised if the HOME message is not Dual type
        pass

    def test_error_message_particles_fail_constraints(self):
        # Test that a ValueError is raised if the ALIEN particles failed the constraints
        pass

    def test_no_concat_no_inner_binding_single_pattern_no_trans_same_ptcl_num_1(self):
        # Test forward_compute with
        #   - No predicate argument concatenation
        #   - No inner pattern binding
        #   - Single pattern declared
        #   - No transformation
        #   - Same number of ALIEN particles and the original particles
        #   - Same in and out pred arg order
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 1, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 2, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 3, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        # KnowledgeServer
        ks = gen_ks(D.MultivariateNormal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1 = gen_vn([pat_var_1], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)

        # Mock message
        home_msg_shape = (Size([6, 7, 8]), Size([42]), Size([10, 15, 20]), Size([1, 2, 3]))
        aux_msg_1_shape = (Size([6, 7, 8]), Size([]), Size([10]), Size([1]))
        # Generate a home message with valid parameter and particles weight that conforms to the parameter
        home_msg = random_message(MessageType.Dual, *home_msg_shape)
        home_msg.parameter = gen_multi_normal_param(Size([6, 7, 8]), 6)     # Replace with valid parameter
        home_log_prob = ks.surrogate_log_prob(home_msg.parameter, home_msg.particles, index_map={0: 0, 1: 1, 2: 2})
        home_msg = home_msg.event_reweight(home_log_prob)   # Reweight to get the real weight
        # Generate random alien message
        alien_msg_1 = random_message(MessageType.Particles, *aux_msg_1_shape)
        # Mock linkdata read and new
        home_ld.read.return_value = home_msg
        alien_ld_1.read.return_value = alien_msg_1
        home_ld.new, alien_ld_1.new = True, True

        # Compute
        prfn.compute()

        # Check out ld and message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]
        # Check message type and shape
        assert MessageType.Particles in out_msg.type
        expected_out_msg_shape = (Size([6, 7, 8]), Size([]), Size([10, 15, 20]), Size([1, 2, 3]))
        assert out_msg.shape == expected_out_msg_shape
        # Check particles and log densities
        assert_equal_within_error(out_msg.particles[0], alien_msg_1.particles[0])
        assert_equal_within_error(out_msg.particles[1], home_msg.particles[1])
        assert_equal_within_error(out_msg.particles[2], home_msg.particles[2])
        assert_equal_within_error(out_msg.log_densities[0], alien_msg_1.log_densities[0])
        assert_equal_within_error(out_msg.log_densities[1], home_msg.log_densities[1])
        assert_equal_within_error(out_msg.log_densities[2], home_msg.log_densities[2])
        # Check particle weight
        # The marginal message w.r.t. unchanged dimension should equal to the original HOME message's marginal
        # Here event dimension 1 and 2 were not tempered with alien particles
        # assert_equal_within_error(out_msg.event_marginalize_over(0).weight, home_msg.event_marginalize_over(0).weight,
        #                           1e-5)

    def test_no_concat_no_inner_binding_single_pattern_no_trans_same_ptcl_num_2(self):
        # Test forward_compute with
        #   - No predicate argument concatenation
        #   - No inner pattern binding
        #   - Single pattern declared
        #   - No transformation
        #   - Same number of ALIEN particles and the original particles
        #   - Different in and out pred arg order
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 1, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 2, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 3, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        # KnowledgeServer
        ks = gen_ks(D.MultivariateNormal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_2, pred_arg_3, pred_arg_1], REL_VARS)     # Different pred arg order
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1 = gen_vn([pat_var_1], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)

        # Mock message
        home_msg_shape = (Size([6, 7, 8]), Size([42]), Size([10, 15, 20]), Size([1, 2, 3]))
        aux_msg_1_shape = (Size([6, 7, 8]), Size([]), Size([10]), Size([1]))
        # Generate a home message with valid parameter and particles weight that conforms to the parameter
        home_msg = random_message(MessageType.Dual, *home_msg_shape)
        home_msg.parameter = gen_multi_normal_param(Size([6, 7, 8]), 6)     # Replace with valid parameter
        home_log_prob = ks.surrogate_log_prob(home_msg.parameter, home_msg.particles, index_map={0: 0, 1: 1, 2: 2})
        home_msg = home_msg.event_reweight(home_log_prob)   # Reweight to get the real weight
        # Generate random alien message
        alien_msg_1 = random_message(MessageType.Particles, *aux_msg_1_shape)
        # Mock linkdata read and new
        home_ld.read.return_value = home_msg
        alien_ld_1.read.return_value = alien_msg_1
        home_ld.new, alien_ld_1.new = True, True

        # Compute
        prfn.compute()

        # Check out ld and message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]
        # Check message type and shape
        assert MessageType.Particles in out_msg.type
        expected_out_msg_shape = (Size([6, 7, 8]), Size([]), Size([15, 20, 10]), Size([2, 3, 1]))
        assert out_msg.shape == expected_out_msg_shape
        # Check particles and log densities
        assert_equal_within_error(out_msg.particles[2], alien_msg_1.particles[0])
        assert_equal_within_error(out_msg.particles[0], home_msg.particles[1])
        assert_equal_within_error(out_msg.particles[1], home_msg.particles[2])
        assert_equal_within_error(out_msg.log_densities[2], alien_msg_1.log_densities[0])
        assert_equal_within_error(out_msg.log_densities[0], home_msg.log_densities[1])
        assert_equal_within_error(out_msg.log_densities[1], home_msg.log_densities[2])

    def test_no_concat_no_inner_binding_single_pattern_no_trans_diff_ptcl_num(self):
        # Test forward_compute with
        #   - No predicate argument concatenation
        #   - No inner pattern binding
        #   - Single pattern declared
        #   - No transformation
        #   - Different number of ALIEN particles and the original particles
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 1, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 2, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 3, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1)
        # KnowledgeServer
        ks = gen_ks(D.MultivariateNormal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1 = gen_vn([pat_var_1], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)

        # Mock message
        home_msg_shape = (Size([6, 7, 8]), Size([42]), Size([10, 15, 20]), Size([1, 2, 3]))
        aux_msg_1_shape = (Size([6, 7, 8]), Size([]), Size([30]), Size([1]))   # Different number of particles
        # Generate a home message with valid parameter and particles weight that conforms to the parameter
        home_msg = random_message(MessageType.Dual, *home_msg_shape)
        home_msg.parameter = gen_multi_normal_param(Size([6, 7, 8]), 6)     # Replace with valid parameter
        home_log_prob = ks.surrogate_log_prob(home_msg.parameter, home_msg.particles, index_map={0: 0, 1: 1, 2: 2})
        home_msg = home_msg.event_reweight(home_log_prob)   # Reweight to get the real weight
        # Generate random alien message
        alien_msg_1 = random_message(MessageType.Particles, *aux_msg_1_shape)
        # Mock linkdata read and new
        home_ld.read.return_value = home_msg
        alien_ld_1.read.return_value = alien_msg_1
        home_ld.new, alien_ld_1.new = True, True

        # Compute
        prfn.compute()

        # Check out ld and message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]
        # Check message type and shape
        assert MessageType.Particles in out_msg.type
        expected_out_msg_shape = (Size([6, 7, 8]), Size([]), Size([30, 15, 20]), Size([1, 2, 3]))
        assert out_msg.shape == expected_out_msg_shape
        # Check particles and log densities
        assert_equal_within_error(out_msg.particles[0], alien_msg_1.particles[0])
        assert_equal_within_error(out_msg.particles[1], home_msg.particles[1])
        assert_equal_within_error(out_msg.particles[2], home_msg.particles[2])
        assert_equal_within_error(out_msg.log_densities[0], alien_msg_1.log_densities[0])
        assert_equal_within_error(out_msg.log_densities[1], home_msg.log_densities[1])
        assert_equal_within_error(out_msg.log_densities[2], home_msg.log_densities[2])

    def test_no_concat_no_inner_binding_single_pattern_with_trans_diff_ptcl_num(self):
        # Test forward_compute with
        #   - No predicate argument concatenation
        #   - No inner pattern binding
        #   - Single pattern declared
        #   - With transformation
        #   - Different number of ALIEN particles and the original particles
        # Transformation
        trans_1 = T.AffineTransform(2.4, 6.7)
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 1, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 2, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 3, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1, trans_1)
        # KnowledgeServer
        ks = gen_ks(D.MultivariateNormal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1 = gen_vn([pat_var_1], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)

        # Mock message
        home_msg_shape = (Size([6, 7, 8]), Size([42]), Size([10, 15, 20]), Size([1, 2, 3]))
        aux_msg_1_shape = (Size([6, 7, 8]), Size([]), Size([30]), Size([1]))   # Different number of particles
        # Generate a home message with valid parameter and particles weight that conforms to the parameter
        home_msg = random_message(MessageType.Dual, *home_msg_shape)
        home_msg.parameter = gen_multi_normal_param(Size([6, 7, 8]), 6)     # Replace with valid parameter
        home_log_prob = ks.surrogate_log_prob(home_msg.parameter, home_msg.particles, index_map={0: 0, 1: 1, 2: 2})
        home_msg = home_msg.event_reweight(home_log_prob)   # Reweight to get the real weight
        # Generate random alien message
        alien_msg_1 = random_message(MessageType.Particles, *aux_msg_1_shape)
        # Mock linkdata read and new
        home_ld.read.return_value = home_msg
        alien_ld_1.read.return_value = alien_msg_1
        home_ld.new, alien_ld_1.new = True, True

        # Compute
        prfn.compute()

        # Check out ld and message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]
        # Check message type and shape
        assert MessageType.Particles in out_msg.type
        expected_out_msg_shape = (Size([6, 7, 8]), Size([]), Size([30, 15, 20]), Size([1, 2, 3]))
        assert out_msg.shape == expected_out_msg_shape
        # Check particles and log densities
        # For alien particles, the backward transformation should have been applied
        assert_equal_within_error(out_msg.particles[0], alien_msg_1.event_transform(trans_1.inv).particles[0])
        assert_equal_within_error(out_msg.particles[1], home_msg.particles[1])
        assert_equal_within_error(out_msg.particles[2], home_msg.particles[2])
        assert_equal_within_error(out_msg.log_densities[0], alien_msg_1.event_transform(trans_1.inv).log_densities[0])
        assert_equal_within_error(out_msg.log_densities[1], home_msg.log_densities[1])
        assert_equal_within_error(out_msg.log_densities[2], home_msg.log_densities[2])

    def test_no_concat_no_inner_binding_multiple_pattern_with_trans_diff_ptcl_num(self):
        # Test forward_compute with
        #   - No predicate argument concatenation
        #   - No inner pattern binding
        #   - Multiple pattern declared
        #   - With transformation
        #   - Different number of ALIEN particles and the original particles
        # Transformations
        trans_1 = T.AffineTransform(2.3, 6.7)
        trans_3 = T.ExpTransform()
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 1, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 2, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 3, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1, trans_1)
        pattern_3 = EventTransform(pred_arg_3, pat_var_3, trans_3, forward=False)
        # KnowledgeServer
        ks = gen_ks(D.MultivariateNormal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_3], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1, aux_vn_3 = gen_vn([pat_var_1], REL_VARS), gen_vn([pat_var_3], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1, alien_ld_3 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1'), \
                                 gen_ld(prfn, aux_vn_3, True, True, alien=True, name='alien_ld_3')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)
        prfn.add_link(alien_ld_3)

        # Mock message
        home_msg_shape = (Size([6, 7, 8]), Size([42]), Size([10, 15, 20]), Size([1, 2, 3]))
        aux_msg_1_shape = (Size([6, 7, 8]), Size([]), Size([30]), Size([1]))   # Different number of particles
        aux_msg_3_shape = (Size([6, 7, 8]), Size([]), Size([45]), Size([3]))  # Different number of particles
        # Generate a home message with valid parameter and particles weight that conforms to the parameter
        home_msg = random_message(MessageType.Dual, *home_msg_shape)
        home_msg.parameter = gen_multi_normal_param(Size([6, 7, 8]), 6)     # Replace with valid parameter
        home_log_prob = ks.surrogate_log_prob(home_msg.parameter, home_msg.particles, index_map={0: 0, 1: 1, 2: 2})
        home_msg = home_msg.event_reweight(home_log_prob)   # Reweight to get the real weight
        # Generate random alien message
        alien_msg_1 = random_message(MessageType.Particles, *aux_msg_1_shape)
        alien_msg_3 = random_message(MessageType.Particles, *aux_msg_3_shape)
        # Mock linkdata read and new
        home_ld.read.return_value = home_msg
        home_ld.new = True
        alien_ld_1.read.return_value, alien_ld_3.read.return_value = alien_msg_1, alien_msg_3
        alien_ld_1.new, alien_ld_3.new = True, True

        # Compute
        prfn.compute()

        # Check out ld and message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]
        # Check message type and shape
        assert MessageType.Particles in out_msg.type
        expected_out_msg_shape = (Size([6, 7, 8]), Size([]), Size([30, 15, 45]), Size([1, 2, 3]))
        assert out_msg.shape == expected_out_msg_shape
        # Check particles and log densities
        assert_equal_within_error(out_msg.particles[0], alien_msg_1.event_transform(trans_1.inv).particles[0])
        assert_equal_within_error(out_msg.particles[1], home_msg.particles[1])
        assert_equal_within_error(out_msg.particles[2], alien_msg_3.event_transform(trans_3).particles[0])
        assert_equal_within_error(out_msg.log_densities[0], alien_msg_1.event_transform(trans_1.inv).log_densities[0])
        assert_equal_within_error(out_msg.log_densities[1], home_msg.log_densities[1])
        assert_equal_within_error(out_msg.log_densities[2], alien_msg_3.event_transform(trans_3).log_densities[0])

    def test_no_concat_with_inner_binding_multiple_pattern_with_trans_diff_ptcl_num(self):
        # Test forward_compute with
        #   - No predicate argument concatenation
        #   - With inner pattern binding
        #   - Multiple pattern declared
        #   - With transformation
        #   - Different number of ALIEN particles and the original particles
        # Transformations
        trans_1 = T.AffineTransform(2.3, 6.7)
        trans_3 = T.ExpTransform()
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 2, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 3, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 2, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 2, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 3, (C.real,))
        # Patterns
        pattern_1 = EventTransform(pred_arg_1, pat_var_1, trans_1)
        pattern_2 = EventTransform(pred_arg_3, pat_var_1, trans_3, forward=False)   # Binds on pat_var_1
        # KnowledgeServer
        ks = gen_ks(D.MultivariateNormal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2], ks)

        # Mock Variable node
        out_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1 = gen_vn([pat_var_1], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)

        # Mock message
        home_msg_shape = (Size([6, 7, 8]), Size([56]), Size([10, 15, 20]), Size([2, 3, 2]))
        aux_msg_1_shape = (Size([6, 7, 8]), Size([]), Size([30]), Size([2]))   # Different number of particles
        # Generate a home message with valid parameter and particles weight that conforms to the parameter
        home_msg = random_message(MessageType.Dual, *home_msg_shape)
        home_msg.parameter = gen_multi_normal_param(Size([6, 7, 8]), 7)     # Replace with valid parameter
        home_log_prob = ks.surrogate_log_prob(home_msg.parameter, home_msg.particles, index_map={0: 0, 1: 1, 2: 2})
        home_msg = home_msg.event_reweight(home_log_prob)   # Reweight to get the real weight
        # Generate random alien message
        alien_msg_1 = random_message(MessageType.Particles, *aux_msg_1_shape)
        # Mock linkdata read and new
        home_ld.read.return_value = home_msg
        home_ld.new = True
        alien_ld_1.read.return_value = alien_msg_1
        alien_ld_1.new = True

        # Compute
        prfn.compute()

        # Check out ld and message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]
        # Check message type and shape
        assert MessageType.Particles in out_msg.type
        expected_out_msg_shape = (Size([6, 7, 8]), Size([]), Size([30, 15, 30]), Size([2, 3, 2]))
        assert out_msg.shape == expected_out_msg_shape
        # Check particles and log densities
        assert_equal_within_error(out_msg.particles[0], alien_msg_1.event_transform(trans_1.inv).particles[0])
        assert_equal_within_error(out_msg.particles[1], home_msg.particles[1])
        assert_equal_within_error(out_msg.particles[2], alien_msg_1.event_transform(trans_3).particles[0])
        assert_equal_within_error(out_msg.log_densities[0], alien_msg_1.event_transform(trans_1.inv).log_densities[0])
        assert_equal_within_error(out_msg.log_densities[1], home_msg.log_densities[1])
        assert_equal_within_error(out_msg.log_densities[2], alien_msg_1.event_transform(trans_3).log_densities[0])

    def test_with_concat_with_inner_binding_multiple_pattern_with_trans_diff_ptcl_num(self):
        # Test forward_compute with
        #   - With predicate argument concatenation
        #   - With inner pattern binding
        #   - Multiple pattern declared
        #   - With transformation
        #   - Different number of ALIEN particles and the original particles
        # Transformations
        trans_1 = T.AffineTransform(2.3, 6.7)
        trans_3 = T.ExpTransform()
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        # Patterns
        pattern_1 = EventTransform([pred_arg_1, pred_arg_2], pat_var_1, trans_1)
        pattern_2 = EventTransform(pred_arg_3, pat_var_1, trans_3, forward=False)   # Binds on pat_var_1
        # KnowledgeServer
        ks = gen_ks(D.MultivariateNormal, [pred_arg_1, pred_arg_2, pred_arg_3])
        # Node
        prfn = PRFN('test_prfn', [pattern_1, pattern_2], ks)

        # Mock Variable node
        out_vn = gen_vn([pattern_1.surrogate_pred_arg, pattern_2.surrogate_pred_arg], REL_VARS)    # Surrogate pred args
        home_vn = gen_vn([pred_arg_1, pred_arg_2, pred_arg_3], REL_VARS)
        aux_vn_1 = gen_vn([pat_var_1], REL_VARS)
        # Test Linkdata
        out_ld = gen_ld(prfn, out_vn, False, True, name='out_ld')
        home_ld = gen_ld(prfn, home_vn, True, True, alien=False, name='home_ld')
        alien_ld_1 = gen_ld(prfn, aux_vn_1, True, True, alien=True, name='alien_ld_1')

        prfn.add_link(out_ld)
        prfn.add_link(home_ld)
        prfn.add_link(alien_ld_1)

        # Mock message
        home_msg_shape = (Size([6, 7, 8]), Size([42]), Size([10, 15, 20]), Size([1, 2, 3]))
        aux_msg_1_shape = (Size([6, 7, 8]), Size([]), Size([30]), Size([3]))   # Different number of particles
        # Generate a home message with valid parameter and particles weight that conforms to the parameter
        home_msg = random_message(MessageType.Dual, *home_msg_shape)
        home_msg.parameter = gen_multi_normal_param(Size([6, 7, 8]), 6)     # Replace with valid parameter
        home_log_prob = ks.surrogate_log_prob(home_msg.parameter, home_msg.particles, index_map={0: 0, 1: 1, 2: 2})
        home_msg = home_msg.event_reweight(home_log_prob)   # Reweight to get the real weight
        # Generate random alien message
        alien_msg_1 = random_message(MessageType.Particles, *aux_msg_1_shape)
        # Mock linkdata read and new
        home_ld.read.return_value = home_msg
        home_ld.new = True
        alien_ld_1.read.return_value = alien_msg_1
        alien_ld_1.new = True

        # Compute
        prfn.compute()

        # Check out ld and message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]
        # Check message type and shape
        assert MessageType.Particles in out_msg.type
        expected_out_msg_shape = (Size([6, 7, 8]), Size([]), Size([30, 30]), Size([3, 3]))
        assert out_msg.shape == expected_out_msg_shape
        # Check particles and log densities
        assert_equal_within_error(out_msg.particles[0], alien_msg_1.event_transform(trans_1.inv).particles[0])
        assert_equal_within_error(out_msg.particles[1], alien_msg_1.event_transform(trans_3).particles[0])
        assert_equal_within_error(out_msg.log_densities[0], alien_msg_1.event_transform(trans_1.inv).log_densities[0])
        assert_equal_within_error(out_msg.log_densities[1], alien_msg_1.event_transform(trans_3).log_densities[0])
