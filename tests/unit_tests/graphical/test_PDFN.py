"""
    Unit tests for PDFN class
"""
import pytest
from unittest.mock import MagicMock

import torch
import torch.distributions as D
import torch.distributions.constraints as C
import torch.distributions.transforms as T
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, DVN, VariableNode
from pysigma.graphical.beta_nodes import PDFN
from pysigma.pattern_structures.event_transform import EventTransform
from ...utils import random_message, assert_equal_within_error, assert_proportional_within_error


# Utility methods
def gen_vn(ran_vars, rel_vars):
    # Generate mocked variable node
    mock_vn = MagicMock(spec=VariableNode)
    mock_vn.ran_vars, mock_vn.rel_vars = ran_vars, rel_vars
    return mock_vn


def gen_ld(fn, vn, incoming: bool, inward: bool):
    # Generate mocked linkdata
    mock_ld = MagicMock(spec=LinkData)
    mock_ld.fn, mock_ld.vn = fn, vn
    mock_ld.to_fn = incoming
    mock_ld.attr = {'direction': 'inward' if inward else 'outward'}
    return mock_ld


class TestPDFN_Init:

    def test_one_to_one_no_binding(self):
        # Test init with
        #   - all one-to-one event transforms and
        #   - no pattern variable bindings
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)

        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Check attributes
        assert pdfn.pat_var2event_transforms == {
            pat_var_1: [et_1],
            pat_var_2: [et_2],
            pat_var_3: [et_3]
        }
        assert pdfn.et_pred_args_flattened == {pred_arg_1, pred_arg_2, pred_arg_3}

    def test_one_to_one_with_binding(self):
        # Test init with
        #   - all one-to-one event transforms and
        #   - with pattern variable binding
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 4, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_2)

        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Check attributes
        assert pdfn.pat_var2event_transforms == {
            pat_var_1: [et_1],
            pat_var_2: [et_2, et_3]
        }
        assert pdfn.et_pred_args_flattened == {pred_arg_1, pred_arg_2, pred_arg_3}

    def test_concat_no_binding(self):
        # Test init with
        #   - some transforms have argument concatenation and
        #   - no pattern variable bindings
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pred_arg_4 = Variable('pred_arg_4', VariableMetatype.Random, 6, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 11, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform([pred_arg_3, pred_arg_4], pat_var_3)

        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Check attributes
        assert pdfn.pat_var2event_transforms == {
            pat_var_1: [et_1],
            pat_var_2: [et_2],
            pat_var_3: [et_3]
        }
        assert pdfn.et_pred_args_flattened == {pred_arg_1, pred_arg_2, pred_arg_3, pred_arg_4}

    def test_concat_with_binding(self):
        # Test init with
        #   - some transforms have argument concatenation and
        #   - with pattern variable bindings
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 11, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pred_arg_4 = Variable('pred_arg_4', VariableMetatype.Random, 6, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 11, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform([pred_arg_3, pred_arg_4], pat_var_2)

        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Check attributes
        assert pdfn.pat_var2event_transforms == {
            pat_var_1: [et_1],
            pat_var_2: [et_2, et_3]
        }
        assert pdfn.et_pred_args_flattened == {pred_arg_1, pred_arg_2, pred_arg_3, pred_arg_4}


class TestPDFN_AddLink:

    def test_multiple_outward_side_linkdata_1(self):
        # Test add link with
        #   - Multiple same directional outward-side linkdata
        #   - Incoming-direction
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes
        outward_vn = MagicMock(spec=VariableNode)
        outward_vn.ran_vars = (pred_arg_1, pred_arg_2, pred_arg_3)
        outward_vn.rel_vars = rel_vars
        # Mock linkdata
        outward_ld_1, outward_ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        outward_ld_1.vn, outward_ld_2.vn = outward_vn, outward_vn
        outward_ld_1.fn, outward_ld_2.fn = pdfn, pdfn
        outward_ld_1.to_fn, outward_ld_2.to_fn = True, True     # Both incoming
        outward_ld_1.attr, outward_ld_2.attr = {'direction': 'inward'}, {'direction': 'inward'}

        # Add first
        pdfn.add_link(outward_ld_1)
        # Attempt to add second
        with pytest.raises(AssertionError) as excinfo:
            pdfn.add_link(outward_ld_2)

        assert str(excinfo.value) == "In test_pdfn: PDFN can accept at most one inward incoming linkdata and one " \
                                     "outward outgoing linkdata."

    def test_multiple_outward_side_linkdata_2(self):
        # Test add link with
        #   - Multiple same directional outward-side linkdata
        #   - Outgoing-direction
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes
        outward_vn = MagicMock(spec=VariableNode)
        outward_vn.ran_vars = (pred_arg_1, pred_arg_2, pred_arg_3)
        outward_vn.rel_vars = rel_vars
        # Mock linkdata
        outward_ld_1, outward_ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        outward_ld_1.vn, outward_ld_2.vn = outward_vn, outward_vn
        outward_ld_1.fn, outward_ld_2.fn = pdfn, pdfn
        outward_ld_1.to_fn, outward_ld_2.to_fn = False, False     # Both outgoing
        outward_ld_1.attr, outward_ld_2.attr = {'direction': 'outward'}, {'direction': 'outward'}

        # Add first
        pdfn.add_link(outward_ld_1)
        # Attempt to add second
        with pytest.raises(AssertionError) as excinfo:
            pdfn.add_link(outward_ld_2)

        assert str(excinfo.value) == "In test_pdfn: PDFN can accept at most one inward incoming linkdata and one " \
                                     "outward outgoing linkdata."

    def test_outward_side_linkdata_conflicting_pred_args_1(self):
        # Test adding an outward-side linkdata whose
        #   - pred args is not a superset of the pred args inferred from the event transforms
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pred_arg_4 = Variable('pred_arg_4', VariableMetatype.Random, 6, (C.real,))      # Reserved for linkdata
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes
        outward_vn = MagicMock(spec=VariableNode)
        outward_vn.ran_vars = (pred_arg_1, pred_arg_2, pred_arg_4)  # Conflicting predicate arguments
        outward_vn.rel_vars = rel_vars
        # Mock linkdata
        outward_ld_1, outward_ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        outward_ld_1.vn, outward_ld_2.vn = outward_vn, outward_vn
        outward_ld_1.fn, outward_ld_2.fn = pdfn, pdfn
        outward_ld_1.to_fn, outward_ld_2.to_fn = True, False  # One incoming and one outgoing
        outward_ld_1.attr, outward_ld_2.attr = {'direction': 'inward'}, {'direction': 'outward'}
        outward_ld_1.__repr__, outward_ld_2.__repr__ = MagicMock(return_value='outward_ld_1'), \
                                                       MagicMock(return_value='outward_ld_2')

        # Test add the inward-incoming linkdata
        with pytest.raises(AssertionError) as excinfo:
            pdfn.add_link(outward_ld_1)
        assert str(excinfo.value) == "In test_pdfn: The predicate arguments (pred_arg_1, pred_arg_2, pred_arg_4) " \
                                     "inferred from linkdata 'outward_ld_1' is not a super set of those predicate " \
                                     "arguments {} inferred from the pattern elements given during initialization." \
                                     .format(tuple(pdfn.et_pred_args_flattened))

    def test_outward_side_linkdata_conflicting_pred_args_2(self):
        # Test adding an outward-side linkdata whose
        #   - pred args is different from the other outward-side linkdata that was previous added
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes
        outward_vn_1, outward_vn_2 = MagicMock(spec=VariableNode), MagicMock(spec=VariableNode)
        outward_vn_1.ran_vars, outward_vn_2.ran_vars = (pred_arg_1, pred_arg_2, pred_arg_3), (pred_arg_1, pred_arg_2)
        outward_vn_1.rel_vars, outward_vn_2.rel_vars = rel_vars, rel_vars
        # Mock linkdata
        outward_ld_1, outward_ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        outward_ld_1.vn, outward_ld_2.vn = outward_vn_1, outward_vn_2
        outward_ld_1.fn, outward_ld_2.fn = pdfn, pdfn
        outward_ld_1.to_fn, outward_ld_2.to_fn = True, False  # One incoming and one outgoing
        outward_ld_1.attr, outward_ld_2.attr = {'direction': 'inward'}, {'direction': 'outward'}
        outward_ld_1.__repr__, outward_ld_2.__repr__ = MagicMock(return_value='outward_ld_1'), \
                                                       MagicMock(return_value='outward_ld_2')

        # Add the first one
        pdfn.add_link(outward_ld_1)
        # Test add the problematic second one
        with pytest.raises(AssertionError) as excinfo:
            pdfn.add_link(outward_ld_2)
        assert str(excinfo.value) == "In test_pdfn: attempts to register an outward-side linkdata with conflicting " \
                                     "predicate arguments. Inferred predicate arguments " \
                                     "(pred_arg_1, pred_arg_2, pred_arg_3) from other linkdata, but found " \
                                     "(pred_arg_1, pred_arg_2) in linkdata 'outward_ld_2'."

    def test_inward_side_linkdata_multivariate(self):
        # Test addling inward-side linkdata that
        #   - connects to a multivariate Variable Node
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes
        inward_vn_1, inward_vn_2 = MagicMock(spec=VariableNode), MagicMock(spec=VariableNode)
        inward_vn_1.ran_vars, inward_vn_2.ran_vars = (pat_var_1, pat_var_2), (pat_var_2, pat_var_3)
        inward_vn_1.rel_vars, inward_vn_2.rel_vars = rel_vars, rel_vars
        # Mock linkdata
        inward_ld_1, inward_ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        inward_ld_1.vn, inward_ld_2.vn = inward_vn_1, inward_vn_2
        inward_ld_1.fn, inward_ld_2.fn = pdfn, pdfn
        inward_ld_1.to_fn, inward_ld_2.to_fn = False, True  # One incoming and one outgoing
        inward_ld_1.attr, inward_ld_2.attr = {'direction': 'inward'}, {'direction': 'outward'}
        inward_ld_1.__repr__, inward_ld_2.__repr__ = MagicMock(return_value='inward_ld_1'), \
                                                     MagicMock(return_value='inward_ld_2')

        # Test adding the invalid inward outgoing linkdata
        with pytest.raises(AssertionError) as excinfo:
            pdfn.add_link(inward_ld_1)
        assert str(excinfo.value) == "In test_pdfn: Every inward outgoing or outward incoming linkdata must connect " \
                                     "to a univariate variable node. Found linkdata 'inward_ld_1' connecting to a " \
                                     "variable node with random variables (pat_var_1, pat_var_2)."
        # Test adding the invalid inward incoming linkdata
        with pytest.raises(AssertionError) as excinfo:
            pdfn.add_link(inward_ld_2)
        assert str(excinfo.value) == "In test_pdfn: Every inward outgoing or outward incoming linkdata must connect " \
                                     "to a univariate variable node. Found linkdata 'inward_ld_2' connecting to a " \
                                     "variable node with random variables (pat_var_2, pat_var_3)."

    def test_inward_side_linkdata_unknown_variable(self):
        # Test add an inward-side linkdata that
        #   - has an unknown variable that is not one of the previously declared pattern variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_4 = Variable('pat_var_4', VariableMetatype.Random, 7, (C.real,))    # Some unknown variable
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes
        inward_vn_1 = MagicMock(spec=VariableNode)
        inward_vn_1.ran_vars = (pat_var_4,)     # Unknown variable
        inward_vn_1.rel_vars = rel_vars
        # Mock linkdata
        inward_ld_1, inward_ld_2 = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        inward_ld_1.vn, inward_ld_2.vn = inward_vn_1, inward_vn_1
        inward_ld_1.fn, inward_ld_2.fn = pdfn, pdfn
        inward_ld_1.to_fn, inward_ld_2.to_fn = False, True  # One incoming and one outgoing
        inward_ld_1.attr, inward_ld_2.attr = {'direction': 'inward'}, {'direction': 'outward'}
        inward_ld_1.__repr__, inward_ld_2.__repr__ = MagicMock(return_value='inward_ld_1'), \
                                                     MagicMock(return_value='inward_ld_2')

        # Test adding the invalid inward outgoing linkdata
        with pytest.raises(AssertionError) as excinfo:
            pdfn.add_link(inward_ld_1)
        assert str(excinfo.value) == "In test_pdfn: The variable node that any inward outgoing or outward incoming " \
                                     "linkdata connects to must have the random variable that is declared as one of " \
                                     "the pattern variables by the pattern elements. Expect one of {}, but found " \
                                     "pat_var_4 in linkdata 'inward_ld_1'." \
                                     .format(pdfn.et_pat_vars)
        # Test adding the invalid inward incoming linkdata
        with pytest.raises(AssertionError) as excinfo:
            pdfn.add_link(inward_ld_2)
        assert str(excinfo.value) == "In test_pdfn: The variable node that any inward outgoing or outward incoming " \
                                     "linkdata connects to must have the random variable that is declared as one of " \
                                     "the pattern variables by the pattern elements. Expect one of {}, but found " \
                                     "pat_var_4 in linkdata 'inward_ld_2'." \
                                     .format(pdfn.et_pat_vars)

    def test_inward_side_linkdata_not_matched_1(self):
        # Test adding an inward-side linkdata that
        #   - cannot match with one event transform because there are more linkdata than the event transform declared
        #   - only one event transform pairs with teh target pattern variable
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes
        inward_vn_1 = MagicMock(spec=VariableNode)
        inward_vn_1.ran_vars = (pat_var_1,)  # target pattern variable
        inward_vn_1.rel_vars = rel_vars
        # Mock linkdata
        inward_ld_1, inward_ld_2, inward_ld_3 = MagicMock(spec=LinkData), MagicMock(spec=LinkData), \
                                                MagicMock(spec=LinkData)
        inward_ld_1.vn, inward_ld_2.vn, inward_ld_3.vn = inward_vn_1, inward_vn_1, inward_vn_1
        inward_ld_1.fn, inward_ld_2.fn, inward_ld_3.fn = pdfn, pdfn, pdfn
        inward_ld_1.to_fn, inward_ld_2.to_fn, inward_ld_3.to_fn = False, True, False  # One incoming and one outgoing
        inward_ld_1.attr, inward_ld_2.attr, inward_ld_3.attr = {'direction': 'inward'}, {'direction': 'outward'}, \
                                                               {'direction': 'inward'}
        inward_ld_1.__repr__, inward_ld_2.__repr__, inward_ld_3.__repr__ = MagicMock(return_value='inward_ld_1'), \
                                                                           MagicMock(return_value='inward_ld_2'), \
                                                                           MagicMock(return_value='inward_ld_3')

        # Add the first ld
        pdfn.add_link(inward_ld_1)
        # Add the second ld. Since it is another direction, this should not cause any problem
        pdfn.add_link(inward_ld_2)
        # Attempt to add the second ld
        with pytest.raises(AssertionError) as excinfo:
            pdfn.add_link(inward_ld_3)
        assert str(excinfo.value) == "In test_pdfn: Cannot associate a pattern element to the inward-side linkdata " \
                                     "'inward_ld_3'."

    def test_inward_side_linkdata_not_matched_2(self):
        # Test adding an inward-side linkdata that
        #   - cannot match with one event transform because there are more linkdata than the event transform declared
        #   - Multiple event transforms pairs with teh target pattern variable
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 4, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_2)    # Both et_2 and et_3 pair with pat_var_2
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes
        inward_vn_1 = MagicMock(spec=VariableNode)
        inward_vn_1.ran_vars = (pat_var_2,)  # target pattern variable
        inward_vn_1.rel_vars = rel_vars
        # Mock linkdata
        inward_ld_1, inward_ld_2, inward_ld_3 = MagicMock(spec=LinkData), MagicMock(spec=LinkData), \
                                                MagicMock(spec=LinkData)
        inward_ld_1.vn, inward_ld_2.vn, inward_ld_3.vn = inward_vn_1, inward_vn_1, inward_vn_1
        inward_ld_1.fn, inward_ld_2.fn, inward_ld_3.fn = pdfn, pdfn, pdfn
        inward_ld_1.to_fn, inward_ld_2.to_fn, inward_ld_3.to_fn = False, False, False  # One incoming and one outgoing
        inward_ld_1.attr, inward_ld_2.attr, inward_ld_3.attr = {'direction': 'inward'}, {'direction': 'inward'}, \
                                                               {'direction': 'inward'}
        inward_ld_1.__repr__, inward_ld_2.__repr__, inward_ld_3.__repr__ = MagicMock(return_value='inward_ld_1'), \
                                                                           MagicMock(return_value='inward_ld_2'), \
                                                                           MagicMock(return_value='inward_ld_3')

        # Add the first ld
        pdfn.add_link(inward_ld_1)
        # Add the second ld.
        pdfn.add_link(inward_ld_2)
        # Attempt to add the second ld
        with pytest.raises(AssertionError) as excinfo:
            pdfn.add_link(inward_ld_3)
        assert str(excinfo.value) == "In test_pdfn: Cannot associate a pattern element to the inward-side linkdata " \
                                     "'inward_ld_3'."

    def test_correct_full_var_inward_only(self):
        # Test internal data structure after correctly adding inward-only linkdata
        # Case 1:
        #   - Full pattern variable references
        #   - Inward only

        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        outward_ld, inward_ld_1, inward_ld_2, inward_ld_3 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True), \
            gen_ld(pdfn, inward_vn_2, False, True), \
            gen_ld(pdfn, inward_vn_3, False, True)

        pdfn.add_link(outward_ld)
        pdfn.add_link(inward_ld_1)
        pdfn.add_link(inward_ld_2)
        pdfn.add_link(inward_ld_3)

        # Check internal data structures
        assert pdfn.ld_pred_args == (pred_arg_1, pred_arg_2, pred_arg_3)
        assert pdfn.event_transform2ld == {
            et_1: {False: inward_ld_1},
            et_2: {False: inward_ld_2},
            et_3: {False: inward_ld_3},
        }

    def test_correct_full_var_outward_only(self):
        # Test internal data structure after correctly adding inward-only linkdata
        # Case 1:
        #   - Full pattern variable references
        #   - Outward only

        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        outward_ld, inward_ld_1, inward_ld_2, inward_ld_3 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False), \
            gen_ld(pdfn, inward_vn_2, True, False), \
            gen_ld(pdfn, inward_vn_3, True, False)

        pdfn.add_link(outward_ld)
        pdfn.add_link(inward_ld_1)
        pdfn.add_link(inward_ld_2)
        pdfn.add_link(inward_ld_3)

        # Check internal data structures
        assert pdfn.ld_pred_args == (pred_arg_1, pred_arg_2, pred_arg_3)
        assert pdfn.event_transform2ld == {
            et_1: {True: inward_ld_1},
            et_2: {True: inward_ld_2},
            et_3: {True: inward_ld_3},
        }

    def test_correct_full_var_both_direction(self):
        # Test internal data structure after correctly adding inward-only linkdata
        # Case 1:
        #   - Full pattern variable references
        #   - Both inward and outward directions

        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        outward_in_ld, inward_out_ld_1, inward_out_ld_2, inward_out_ld_3 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True), \
            gen_ld(pdfn, inward_vn_2, False, True), \
            gen_ld(pdfn, inward_vn_3, False, True)
        outward_out_ld, inward_in_ld_1, inward_in_ld_2, inward_in_ld_3 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False), \
            gen_ld(pdfn, inward_vn_2, True, False), \
            gen_ld(pdfn, inward_vn_3, True, False)

        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)
        pdfn.add_link(inward_out_ld_2)
        pdfn.add_link(inward_out_ld_3)
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)
        pdfn.add_link(inward_in_ld_2)
        pdfn.add_link(inward_in_ld_3)

        # Check internal data structures
        assert pdfn.ld_pred_args == (pred_arg_1, pred_arg_2, pred_arg_3)
        assert pdfn.event_transform2ld == {
            et_1: {True: inward_in_ld_1, False: inward_out_ld_1},
            et_2: {True: inward_in_ld_2, False: inward_out_ld_2},
            et_3: {True: inward_in_ld_3, False: inward_out_ld_3},
        }


class TestPDFN_PrecomputeCheck:

    def test_missing_linkdata(self):
        # Test that precompute_check() would raise AssertionError if
        #   - some of the pattern variable is not matched with any inward-side linkdata.
        #   - Test with inward-only linkdata
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        outward_ld, inward_ld_1, inward_ld_2, inward_ld_3 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True), \
            gen_ld(pdfn, inward_vn_2, False, True), \
            gen_ld(pdfn, inward_vn_3, False, True)

        # Only add two linkdata, leaving the third pattern variable unmatched
        pdfn.add_link(outward_ld)
        pdfn.add_link(inward_ld_1)
        pdfn.add_link(inward_ld_2)

        with pytest.raises(AssertionError) as excinfo:
            pdfn.precompute_check()

        assert str(excinfo.value) == "In test_pdfn: Not all pattern elements have been matched with a inward-side " \
                                     "linkdata."

    def test_unpaired_pattern(self):
        # Test that precompute_check() would raise AssertionError if
        #   - some of the pattern elements are not matched with a pair of inward and outward linkdata
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1, inward_out_ld_2, inward_out_ld_3 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True), \
            gen_ld(pdfn, inward_vn_2, False, True), \
            gen_ld(pdfn, inward_vn_3, False, True)
        # Outward linkdata
        outward_out_ld, inward_in_ld_1, inward_in_ld_2, inward_in_ld_3 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False), \
            gen_ld(pdfn, inward_vn_2, True, False), \
            gen_ld(pdfn, inward_vn_3, True, False)

        # Miss one linkdata so that one pattern element is not paired with a pair of linkdata
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)
        pdfn.add_link(inward_out_ld_2)
        pdfn.add_link(inward_out_ld_3)
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)
        pdfn.add_link(inward_in_ld_2)

        with pytest.raises(AssertionError) as excinfo:
            pdfn.precompute_check()

        assert str(excinfo.value) == "In test_pdfn: All pattern elements must be matched with a pair of " \
                                     "inward-outgoing and outward-incoming linkdata if both inward and outward " \
                                     "directional linkdata are registered."


class TestPDFN_InwardCompute:

    def test_single_arg_no_transform(self):
        # Test inward_compute() with
        #   - Single predicate argument
        #   - No event transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1,), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10]), Size([3]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        out_msg = inward_out_ld_1.write.call_args[0][0]
        # In this test, the out message should be exactly the same as the in message
        assert out_msg == in_msg

    def test_single_arg_with_transform_1(self):
        # Test inward_compute() with
        #   - Single predicate argument
        #   - With event transform: ExpTransform
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        # Transformation
        trans = T.ExpTransform()
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1, trans)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1,), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10]), Size([3]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        out_msg = inward_out_ld_1.write.call_args[0][0]
        # In this test, the out message should equal the transformed in message
        assert out_msg == in_msg.event_transform(trans)

    def test_multiple_arg_single_var_concat_full_args_no_transform_1(self):
        # Test inward_compute() with
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - All predicate arguments are involved
        #   - no transformations
        #   - Same variable order in concatenation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 12, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_1, pred_arg_2, pred_arg_3], pat_var_1)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        out_msg = inward_out_ld_1.write.call_args[0][0]
        # Check message shape
        expected_shape = (Size([3, 4, 5]), Size([]), Size([10 * 12 * 14]), Size([12]))
        assert out_msg.shape == expected_shape
        # Check message content
        expected_msg = in_msg.event_concatenate([0, 1, 2])
        assert out_msg == expected_msg

    def test_multiple_arg_single_var_concat_full_args_no_transform_2(self):
        # Test inward_compute() with
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - All predicate arguments are involved
        #   - no transformations
        #   - Different variable order in the concatenation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 12, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_2, pred_arg_3, pred_arg_1], pat_var_1)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        out_msg = inward_out_ld_1.write.call_args[0][0]
        # Check message shape
        expected_shape = (Size([3, 4, 5]), Size([]), Size([10 * 12 * 14]), Size([12]))
        assert out_msg.shape == expected_shape
        # Check message content
        expected_msg = in_msg.event_concatenate([1, 2, 0])
        assert out_msg == expected_msg

    def test_multiple_arg_single_var_concat_full_args_with_transformation(self):
        # Test inward_compute() with
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - All predicate arguments are involved
        #   - With transformation: AffineTransform
        # Transformation
        trans = T.AffineTransform(2.3, 0.5)
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 12, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_1, pred_arg_2, pred_arg_3], pat_var_1, trans, forward=False)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        out_msg = inward_out_ld_1.write.call_args[0][0]
        # Check message shape
        expected_shape = (Size([3, 4, 5]), Size([]), Size([10 * 12 * 14]), Size([12]))
        assert out_msg.shape == expected_shape
        # Check message content
        expected_msg = in_msg.event_concatenate([0, 1, 2]).event_transform(trans.inv)
        assert out_msg == expected_msg

    def test_multiple_arg_single_var_concat_partial_args_with_transformation_1(self):
        # Test inward_compute() with
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - Only some of the predicate arguments are involved
        #   - With transformation: AffineTransform
        #   - Predicate argument ordering 1
        # Transformation
        trans = T.AffineTransform(2.3, 0.5)
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 8, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_1, pred_arg_3], pat_var_1, trans)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        out_msg = inward_out_ld_1.write.call_args[0][0]
        # Check message shape
        expected_shape = (Size([3, 4, 5]), Size([]), Size([10  * 14]), Size([8]))
        assert out_msg.shape == expected_shape
        # Check message content
        expected_msg = in_msg.event_concatenate([0, 2]).event_marginalize(-1).event_transform(trans)
        assert out_msg == expected_msg

    def test_multiple_arg_single_var_concat_partial_args_with_transformation_2(self):
        # Test inward_compute() with
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - Only some of the predicate arguments are involved
        #   - With transformation: AffineTransform
        #   - Predicate argument ordering 2
        # Transformation
        trans = T.AffineTransform(2.3, 0.5)
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 9, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_3, pred_arg_2], pat_var_1, trans)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        out_msg = inward_out_ld_1.write.call_args[0][0]
        # Check message shape
        expected_shape = (Size([3, 4, 5]), Size([]), Size([14 * 12]), Size([9]))
        assert out_msg.shape == expected_shape
        # Check message content
        expected_msg = in_msg.event_concatenate([2, 1]).event_marginalize(-1).event_transform(trans)
        assert out_msg == expected_msg

    def test_multiple_arg_multiple_var_no_duplicate_var_no_concat_full_args_no_transformation(self):
        # Test inward_compute() with
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - No reuse of same pattern variable (i.e. no inner-pattern binding)
        #   - No predicate argument concatenation
        #   - All predicate arguments are involved
        #   - No transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1, inward_out_ld_2, inward_out_ld_3 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True), \
            gen_ld(pdfn, inward_vn_2, False, True), \
            gen_ld(pdfn, inward_vn_3, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)
        pdfn.add_link(inward_out_ld_2)
        pdfn.add_link(inward_out_ld_3)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        inward_out_ld_2.write.assert_called_once()
        inward_out_ld_3.write.assert_called_once()
        out_msg_1 = inward_out_ld_1.write.call_args[0][0]
        out_msg_2 = inward_out_ld_2.write.call_args[0][0]
        out_msg_3 = inward_out_ld_3.write.call_args[0][0]
        # Check message shape
        expected_shape_1 = (Size([3, 4, 5]), Size([]), Size([10]), Size([3]))
        expected_shape_2 = (Size([3, 4, 5]), Size([]), Size([12]), Size([4]))
        expected_shape_3 = (Size([3, 4, 5]), Size([]), Size([14]), Size([5]))
        assert out_msg_1.shape == expected_shape_1
        assert out_msg_2.shape == expected_shape_2
        assert out_msg_3.shape == expected_shape_3
        # Check message content
        expected_msg_1 = in_msg.event_marginalize(0)
        expected_msg_2 = in_msg.event_marginalize(1)
        expected_msg_3 = in_msg.event_marginalize(2)
        assert out_msg_1 == expected_msg_1
        assert out_msg_2 == expected_msg_2
        assert out_msg_3 == expected_msg_3

    def test_multiple_arg_multiple_var_no_duplicate_var_no_concat_full_args_with_transformation(self):
        # Test inward_compute() with
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - No reuse of same pattern variable (i.e. no inner-pattern binding)
        #   - No predicate argument concatenation
        #   - All predicate arguments are involved
        #   - With transformation: AffineTransform, ExpTransform
        # Transformations
        trans1 = T.AffineTransform(2.5, 3)
        trans3 = T.ExpTransform()
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1, trans1, forward=False)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3, trans3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1, inward_out_ld_2, inward_out_ld_3 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True), \
            gen_ld(pdfn, inward_vn_2, False, True), \
            gen_ld(pdfn, inward_vn_3, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)
        pdfn.add_link(inward_out_ld_2)
        pdfn.add_link(inward_out_ld_3)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        inward_out_ld_2.write.assert_called_once()
        inward_out_ld_3.write.assert_called_once()
        out_msg_1 = inward_out_ld_1.write.call_args[0][0]
        out_msg_2 = inward_out_ld_2.write.call_args[0][0]
        out_msg_3 = inward_out_ld_3.write.call_args[0][0]
        # Check message shape
        expected_shape_1 = (Size([3, 4, 5]), Size([]), Size([10]), Size([3]))
        expected_shape_2 = (Size([3, 4, 5]), Size([]), Size([12]), Size([4]))
        expected_shape_3 = (Size([3, 4, 5]), Size([]), Size([14]), Size([5]))
        assert out_msg_1.shape == expected_shape_1
        assert out_msg_2.shape == expected_shape_2
        assert out_msg_3.shape == expected_shape_3
        # Check message content
        expected_msg_1 = in_msg.event_marginalize(0).event_transform(trans1.inv)
        expected_msg_2 = in_msg.event_marginalize(1)
        expected_msg_3 = in_msg.event_marginalize(2).event_transform(trans3)
        assert out_msg_1 == expected_msg_1
        assert out_msg_2 == expected_msg_2
        assert out_msg_3 == expected_msg_3

    def test_multiple_arg_multiple_var_no_duplicate_var_no_concat_partial_args_no_transformation(self):
        # Test inward_compute() with
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - No reuse of same pattern variable (i.e. no inner-pattern binding)
        #   - No predicate argument concatenation
        #   - Only part of the predicate arguments are involved
        #   - No transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2])  # et_3 is not declared

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1, inward_out_ld_2, inward_out_ld_3 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True), \
            gen_ld(pdfn, inward_vn_2, False, True), \
            gen_ld(pdfn, inward_vn_3, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)
        pdfn.add_link(inward_out_ld_2)      # inward_out_ld_3 is not registered

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        inward_out_ld_2.write.assert_called_once()
        inward_out_ld_3.write.assert_not_called()
        out_msg_1 = inward_out_ld_1.write.call_args[0][0]
        out_msg_2 = inward_out_ld_2.write.call_args[0][0]
        # Check message shape
        expected_shape_1 = (Size([3, 4, 5]), Size([]), Size([10]), Size([3]))
        expected_shape_2 = (Size([3, 4, 5]), Size([]), Size([12]), Size([4]))
        assert out_msg_1.shape == expected_shape_1
        assert out_msg_2.shape == expected_shape_2
        # Check message content
        expected_msg_1 = in_msg.event_marginalize(0)
        expected_msg_2 = in_msg.event_marginalize(1)
        assert out_msg_1 == expected_msg_1
        assert out_msg_2 == expected_msg_2

    def test_multiple_arg_multiple_var_no_duplicate_var_concat_full_args_no_transformation(self):
        # Test inward_compute() with
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - No reuse of same pattern variable (i.e. no inner-pattern binding)
        #   - With predicate argument concatenation
        #   - All predicate arguments are involved
        #   - No transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 7, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_1, pred_arg_2], pat_var_1)
        et_2 = EventTransform(pred_arg_3, pat_var_2)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1, inward_out_ld_2 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True), \
            gen_ld(pdfn, inward_vn_2, False, True)
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)
        pdfn.add_link(inward_out_ld_2)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        inward_out_ld_2.write.assert_called_once()
        out_msg_1 = inward_out_ld_1.write.call_args[0][0]
        out_msg_2 = inward_out_ld_2.write.call_args[0][0]
        # Check message shape
        expected_shape_1 = (Size([3, 4, 5]), Size([]), Size([10 * 12]), Size([7]))
        expected_shape_2 = (Size([3, 4, 5]), Size([]), Size([14]), Size([5]))
        assert out_msg_1.shape == expected_shape_1
        assert out_msg_2.shape == expected_shape_2
        # Check message content
        expected_msg_1 = in_msg.event_concatenate([0, 1], target_event_dim=0).event_marginalize(0)
        expected_msg_2 = in_msg.event_concatenate([0, 1], target_event_dim=0).event_marginalize(1)
        assert out_msg_1 == expected_msg_1
        assert out_msg_2 == expected_msg_2

    def test_multiple_arg_multiple_var_duplicate_var_no_concat_full_args_no_transformation(self):
        # Test inward_compute() with
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - With reuse of same pattern variable (i.e. no inner-pattern binding)
        #   - No predicate argument concatenation
        #   - All predicate arguments are involved
        #   - No transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 5, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_2)    # pat_var_2 is reused
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars)
        # Inward linkdata
        outward_in_ld, inward_out_ld_1, inward_out_ld_2, inward_out_ld_3 = \
            gen_ld(pdfn, outward_vn, True, True), \
            gen_ld(pdfn, inward_vn_1, False, True), \
            gen_ld(pdfn, inward_vn_2, False, True), \
            gen_ld(pdfn, inward_vn_2, False, True)      # Duplicate linkdata connection to pat_var_2 node
        # Add link
        pdfn.add_link(outward_in_ld)
        pdfn.add_link(inward_out_ld_1)
        pdfn.add_link(inward_out_ld_2)
        pdfn.add_link(inward_out_ld_3)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 5, 5]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        outward_in_ld.read.return_value = in_msg
        outward_in_ld.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        inward_out_ld_1.write.assert_called_once()
        inward_out_ld_2.write.assert_called_once()
        inward_out_ld_3.write.assert_called_once()
        out_msg_1 = inward_out_ld_1.write.call_args[0][0]
        out_msg_2 = inward_out_ld_2.write.call_args[0][0]
        out_msg_3 = inward_out_ld_3.write.call_args[0][0]
        # Check message shape
        expected_shape_1 = (Size([3, 4, 5]), Size([]), Size([10]), Size([3]))
        expected_shape_2 = (Size([3, 4, 5]), Size([]), Size([12]), Size([5]))
        expected_shape_3 = (Size([3, 4, 5]), Size([]), Size([14]), Size([5]))
        assert out_msg_1.shape == expected_shape_1
        # Use set comparison here because order is not definite
        assert {out_msg_2.shape, out_msg_3.shape} == {expected_shape_2, expected_shape_3}
        # Check message content
        expected_msg_1 = in_msg.event_marginalize(0)
        expected_msg_2 = in_msg.event_marginalize(1)
        expected_msg_3 = in_msg.event_marginalize(2)
        assert out_msg_1 == expected_msg_1
        # Order-invariant
        assert (out_msg_2 == expected_msg_2 and out_msg_3 == expected_msg_3) or \
               (out_msg_2 == expected_msg_3 and out_msg_3 == expected_msg_2)


class TestPDFN_OutwardCompute:

    def test_single_arg_no_transform(self):
        # Test outward_compute() with
        #   - Single predicate argument
        #   - No event transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1,), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10]), Size([3]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        inward_in_ld_1.read.return_value = in_msg
        inward_in_ld_1.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg = outward_out_ld.write.call_args[0][0]
        # In this test, the out message should be exactly the same as the in message
        assert out_msg == in_msg

    def test_single_arg_with_transform_1(self):
        # Test outward_compute() with
        #   - Single predicate argument
        #   - With event transformation: AffineTransform
        # Transformation
        trans = T.AffineTransform(4.3, 2.5)
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1, trans, forward=True)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1,), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10]), Size([3]))
        in_msg = random_message(MessageType.Particles, *msg_shape)
        inward_in_ld_1.read.return_value = in_msg
        inward_in_ld_1.new = True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg = outward_out_ld.write.call_args[0][0]
        # Check message content
        expected_msg = in_msg.event_transform(trans.inv)
        assert out_msg == expected_msg

    def test_value_error_missing_cache_for_deconcatenation(self):
        # The computation for the following task requires particle cache in order to de-concatenate the particles
        # Test raising ValueError if no cache found
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - All predicate arguments are involved
        #   - no transformations
        #   - Same variable order in concatenation
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 12, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_1, pred_arg_2, pred_arg_3], pat_var_1)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg = orig_msg.event_concatenate([0, 1, 2])
        inward_in_ld_1.read.return_value = in_msg
        inward_in_ld_1.new = True

        # Compute
        with pytest.raises(ValueError) as excinfo:
            pdfn.compute()

        assert str(excinfo.value) == "In test_pdfn: Outward computation needs to access particle cache to " \
                                     "de-concatenate joint particles, but the cache is None."

    def test_value_error_missing_cache_for_cross_product(self):
        # The computation for the following task requires particle cache in order to compute the cross product
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - No reuse of same pattern variable (i.e. no inner-pattern binding)
        #   - No predicate argument concatenation
        #   - Only part of the predicate arguments are involved
        #   - No transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_2 = EventTransform(pred_arg_2, pat_var_2)  # pred_arg_1 is omitted
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_2, inward_in_ld_3 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_2, True, False), \
            gen_ld(pdfn, inward_vn_3, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_2)
        pdfn.add_link(inward_in_ld_3)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg_2 = orig_msg.event_marginalize(1)
        in_msg_3 = orig_msg.event_marginalize(2)
        inward_in_ld_2.read.return_value = in_msg_2
        inward_in_ld_3.read.return_value = in_msg_3
        inward_in_ld_2.new, inward_in_ld_3.new = True, True

        # Compute without having the particle cache.
        # Since pred_arg_1's particles are unknown from incoming messages, this should raise a ValueError
        with pytest.raises(ValueError) as excinfo:
            pdfn.compute()
        assert str(excinfo.value) == "In test_pdfn: Outward computation needs to access particle cache to take the " \
                                     "cross product of marginal particles, but the cache is None."

    def test_multiple_arg_single_var_concat_full_args_no_transform_1(self):
        # Test outward_compute() with
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - All predicate arguments are involved
        #   - no transformations
        #   - Same variable order in concatenation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 12, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_1, pred_arg_2, pred_arg_3], pat_var_1)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg = orig_msg.event_concatenate([0, 1, 2])
        inward_in_ld_1.read.return_value = in_msg
        inward_in_ld_1.new = True

        # Set particles cache
        pdfn.ptcl_cache, pdfn.dens_cache = orig_msg.particles, orig_msg.log_densities
        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg = outward_out_ld.write.call_args[0][0]
        # Check message content
        assert out_msg == orig_msg

    def test_multiple_arg_single_var_concat_full_args_no_transform_2(self):
        # Test outward_compute() with
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - All predicate arguments are involved
        #   - no transformations
        #   - Different variable order in concatenation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 12, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_2, pred_arg_1, pred_arg_3], pat_var_1)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([12, 10, 14]), Size([4, 3, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg = orig_msg.event_concatenate([1, 0, 2])
        inward_in_ld_1.read.return_value = in_msg
        inward_in_ld_1.new = True

        # Set particles cache
        pdfn.ptcl_cache, pdfn.dens_cache = orig_msg.particles, orig_msg.log_densities
        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg = outward_out_ld.write.call_args[0][0]
        # Check message content
        assert out_msg == orig_msg

    def test_multiple_arg_single_var_concat_full_args_with_transformation(self):
        # Test outward_compute() with
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - All predicate arguments are involved
        #   - with transformations
        # Transformation
        trans = T.AffineTransform(2.4, 5)
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 12, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_1, pred_arg_2, pred_arg_3], pat_var_1, trans)
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg = orig_msg.event_concatenate([0, 1, 2]).event_transform(trans)
        inward_in_ld_1.read.return_value = in_msg
        inward_in_ld_1.new = True

        # Set particles cache
        pdfn.ptcl_cache, pdfn.dens_cache = orig_msg.particles, orig_msg.log_densities
        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg = outward_out_ld.write.call_args[0][0]
        # Check message content
        assert out_msg == orig_msg

    def test_multiple_arg_single_var_concat_partial_args_with_transformation_1(self):
        # Test outward_compute() with
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - Only some of the predicate arguments are involved
        #   - With transformation: AffineTransform
        #   - Predicate argument ordering 1
        # Transformation
        trans = T.AffineTransform(2.4, 5)
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 8, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_1, pred_arg_3], pat_var_1, trans)   # pred_arg_2 is omitted
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg = orig_msg.event_concatenate([0, 2], target_event_dim=-1).event_marginalize(-1).event_transform(trans)
        inward_in_ld_1.read.return_value = in_msg
        inward_in_ld_1.new = True

        # Set particles cache
        pdfn.ptcl_cache, pdfn.dens_cache = orig_msg.particles, orig_msg.log_densities
        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg: Message = outward_out_ld.write.call_args[0][0]
        # Check message content
        # Check particle values and densities
        for p1, p2 in zip(out_msg.particles, orig_msg.particles):
            assert_equal_within_error(p1, p2)
        for d1, d2 in zip(out_msg.log_densities, orig_msg.log_densities):
            assert_equal_within_error(d1, d2)
        # Check particle weight, by checking the marginal message of the returning message
        # For pred_arg_2 marginal, the particle weight should be uniform
        pred_arg_2_marg = out_msg.event_marginalize(1)
        assert_proportional_within_error(pred_arg_2_marg.weight, torch.tensor([1.]), dims=[-1, -2, -3])
        # For pred_arg_1 and pred_arg_3, they should equal to the concatenated original message
        pred_arg_1_3_marg = out_msg.event_marginalize_over(1).event_concatenate([0, 1])
        expected_marg = in_msg      # Use in_msg to compare directly because event transformation shouldn't touch weight
        assert_proportional_within_error(pred_arg_1_3_marg.weight, expected_marg.weight, dims=[-1, -2, -3])

    def test_multiple_arg_single_var_concat_partial_args_with_transformation_2(self):
        # Test outward_compute() with
        #   - Multiple predicate arguments
        #   - Single pattern variable
        #   - With predicate argument concatenation
        #   - Only some of the predicate arguments are involved
        #   - With transformation: AffineTransform
        #   - Predicate argument ordering 2
        # Transformation
        trans = T.AffineTransform(2.4, 5)
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 9, (C.real,))
        # Event transforms
        et_1 = EventTransform([pred_arg_3, pred_arg_2], pat_var_1, trans)  # pred_arg_2 is omitted
        # Node
        pdfn = PDFN('test_pdfn', [et_1])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg = orig_msg.event_concatenate([2, 1], target_event_dim=-1).event_marginalize(-1).event_transform(trans)
        inward_in_ld_1.read.return_value = in_msg
        inward_in_ld_1.new = True

        # Set particles cache
        pdfn.ptcl_cache, pdfn.dens_cache = orig_msg.particles, orig_msg.log_densities
        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg: Message = outward_out_ld.write.call_args[0][0]
        # Check message content
        # Check particle values and densities
        for p1, p2 in zip(out_msg.particles, orig_msg.particles):
            assert_equal_within_error(p1, p2)
        for d1, d2 in zip(out_msg.log_densities, orig_msg.log_densities):
            assert_equal_within_error(d1, d2)
        # Check particle weight, by checking the marginal message of the returning message
        # For pred_arg_1 marginal, the particle weight should be uniform
        pred_arg_1_marg = out_msg.event_marginalize(0)
        assert_proportional_within_error(pred_arg_1_marg.weight, torch.tensor([1.]), dims=[-1, -2, -3])
        # For pred_arg_3 and pred_arg_2, they should equal to the concatenated original message
        pred_arg_2_3_marg = out_msg.event_marginalize_over(0).event_concatenate([1, 0])
        expected_marg = in_msg  # Use in_msg to compare directly because event transformation shouldn't touch weight
        assert_proportional_within_error(pred_arg_2_3_marg.weight, expected_marg.weight, dims=[-1, -2, -3])

    def test_multiple_arg_multiple_var_no_duplicate_var_no_concat_full_args_no_transformation(self):
        # Test outward_compute() with
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - No reuse of same pattern variable (i.e. no inner-pattern binding)
        #   - No predicate argument concatenation
        #   - All predicate arguments are involved
        #   - No transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1, inward_in_ld_2, inward_in_ld_3 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False), \
            gen_ld(pdfn, inward_vn_2, True, False), \
            gen_ld(pdfn, inward_vn_3, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)
        pdfn.add_link(inward_in_ld_2)
        pdfn.add_link(inward_in_ld_3)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg_1 = orig_msg.event_marginalize(0)
        in_msg_2 = orig_msg.event_marginalize(1)
        in_msg_3 = orig_msg.event_marginalize(2)
        inward_in_ld_1.read.return_value = in_msg_1
        inward_in_ld_2.read.return_value = in_msg_2
        inward_in_ld_3.read.return_value = in_msg_3
        inward_in_ld_1.new, inward_in_ld_2.new, inward_in_ld_3.new = True, True, True

        # Set particles cache
        pdfn.ptcl_cache, pdfn.dens_cache = orig_msg.particles, orig_msg.log_densities
        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg: Message = outward_out_ld.write.call_args[0][0]
        # Check message content
        # Compare particles and densities with that in the origianl message
        for p1, p2 in zip(out_msg.particles, orig_msg.particles):
            assert_equal_within_error(p1, p2)
        for d1, d2 in zip(out_msg.log_densities, orig_msg.log_densities):
            assert_equal_within_error(d1, d2)
        # Check particle weight by comparing the marginal weight
        assert_equal_within_error(out_msg.event_marginalize(0).weight, in_msg_1.weight)
        assert_equal_within_error(out_msg.event_marginalize(1).weight, in_msg_2.weight)
        assert_equal_within_error(out_msg.event_marginalize(2).weight, in_msg_3.weight)

    def test_multiple_arg_multiple_var_no_duplicate_var_no_concat_full_args_with_transformation(self):
        # Test outward_compute() with
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - No reuse of same pattern variable (i.e. no inner-pattern binding)
        #   - No predicate argument concatenation
        #   - All predicate arguments are involved
        #   - With transformation: AffineTransform, ExpTransform
        # Transformation
        trans1, trans2 = T.AffineTransform(3.1, 2.45), T.ExpTransform()
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1, trans1, forward=False)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_3, trans2)
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1, inward_in_ld_2, inward_in_ld_3 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False), \
            gen_ld(pdfn, inward_vn_2, True, False), \
            gen_ld(pdfn, inward_vn_3, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)
        pdfn.add_link(inward_in_ld_2)
        pdfn.add_link(inward_in_ld_3)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg_1 = orig_msg.event_marginalize(0).event_transform(trans1.inv)
        in_msg_2 = orig_msg.event_marginalize(1)
        in_msg_3 = orig_msg.event_marginalize(2).event_transform(trans2)
        inward_in_ld_1.read.return_value = in_msg_1
        inward_in_ld_2.read.return_value = in_msg_2
        inward_in_ld_3.read.return_value = in_msg_3
        inward_in_ld_1.new, inward_in_ld_2.new, inward_in_ld_3.new = True, True, True

        # Set particles cache
        pdfn.ptcl_cache, pdfn.dens_cache = orig_msg.particles, orig_msg.log_densities
        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg: Message = outward_out_ld.write.call_args[0][0]
        # Check message content
        # Compare particles and densities with that in the origianl message
        for p1, p2 in zip(out_msg.particles, orig_msg.particles):
            assert_equal_within_error(p1, p2)
        for d1, d2 in zip(out_msg.log_densities, orig_msg.log_densities):
            assert_equal_within_error(d1, d2)
        # Check particle weight by comparing the marginal weight
        assert_equal_within_error(out_msg.event_marginalize(0).weight, in_msg_1.weight)
        assert_equal_within_error(out_msg.event_marginalize(1).weight, in_msg_2.weight)
        assert_equal_within_error(out_msg.event_marginalize(2).weight, in_msg_3.weight)

    def test_multiple_arg_multiple_var_no_duplicate_var_no_concat_partial_args_no_transformation(self):
        # Test outward_compute() with
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - No reuse of same pattern variable (i.e. no inner-pattern binding)
        #   - No predicate argument concatenation
        #   - Only part of the predicate arguments are involved
        #   - No transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_2 = EventTransform(pred_arg_2, pat_var_2)    # pred_arg_1 is omitted
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_2, inward_in_ld_3 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_2, True, False), \
            gen_ld(pdfn, inward_vn_3, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_2)
        pdfn.add_link(inward_in_ld_3)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg_2 = orig_msg.event_marginalize(1)
        in_msg_3 = orig_msg.event_marginalize(2)
        inward_in_ld_2.read.return_value = in_msg_2
        inward_in_ld_3.read.return_value = in_msg_3
        inward_in_ld_2.new, inward_in_ld_3.new = True, True

        # Set particles cache
        pdfn.ptcl_cache, pdfn.dens_cache = orig_msg.particles, orig_msg.log_densities
        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg: Message = outward_out_ld.write.call_args[0][0]
        # Check message content
        # Compare particles and densities with that in the origianl message
        for p1, p2 in zip(out_msg.particles, orig_msg.particles):
            assert_equal_within_error(p1, p2)
        for d1, d2 in zip(out_msg.log_densities, orig_msg.log_densities):
            assert_equal_within_error(d1, d2)
        # Check particle weight by comparing the marginal weight
        # For pred_arg_2 and pred_arg_3, their marginal weight should equal to the marginal in_msg
        # For pred_arg_1, the marginal weight should be uniform
        assert_proportional_within_error(out_msg.event_marginalize(0).weight, torch.tensor(1.), dims=[-1, -2, -3])
        assert_equal_within_error(out_msg.event_marginalize(1).weight, in_msg_2.weight)
        assert_equal_within_error(out_msg.event_marginalize(2).weight, in_msg_3.weight)

    def test_multiple_arg_multiple_var_no_duplicate_var_concat_full_args_no_transformation(self):
        # Test outward_compute() with
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - No reuse of same pattern variable (i.e. no inner-pattern binding)
        #   - With predicate argument concatenation
        #   - All predicate arguments are involved
        #   - No transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 7, (C.real,))
        pat_var_3 = Variable('pat_var_3', VariableMetatype.Random, 5, (C.real,))
        # Event transforms
        et_2 = EventTransform([pred_arg_1, pred_arg_2], pat_var_2)  # concatenation
        et_3 = EventTransform(pred_arg_3, pat_var_3)
        # Node
        pdfn = PDFN('test_pdfn', [et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_2, inward_vn_3 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_2,), rel_vars), \
            gen_vn((pat_var_3,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_2, inward_in_ld_3 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_2, True, False), \
            gen_ld(pdfn, inward_vn_3, True, False)
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_2)
        pdfn.add_link(inward_in_ld_3)

        # Mock message
        msg_shape = (Size([3, 4, 5]), Size([]), Size([10, 12, 14]), Size([3, 4, 5]))
        orig_msg = random_message(MessageType.Particles, *msg_shape)
        in_msg_2 = orig_msg.event_concatenate([0, 1], target_event_dim=0).event_marginalize(0)
        in_msg_3 = orig_msg.event_marginalize(2)
        inward_in_ld_2.read.return_value = in_msg_2
        inward_in_ld_3.read.return_value = in_msg_3
        inward_in_ld_2.new, inward_in_ld_3.new = True, True

        # Set particles cache
        pdfn.ptcl_cache, pdfn.dens_cache = orig_msg.particles, orig_msg.log_densities
        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg: Message = outward_out_ld.write.call_args[0][0]
        # Check message content
        # Compare particles and densities with that in the original message
        for p1, p2 in zip(out_msg.particles, orig_msg.particles):
            assert_equal_within_error(p1, p2)
        for d1, d2 in zip(out_msg.log_densities, orig_msg.log_densities):
            assert_equal_within_error(d1, d2)
        # Check particle weight by comparing the marginal weight
        # For pred_arg_1 and pred_arg_2, their concatenated message weight should equal to the in_msg's weight
        # For pred_arg_3, its marginal message's weight should equal to the in_msg's weight
        assert_equal_within_error(out_msg.event_concatenate([0, 1], target_event_dim=0).event_marginalize(0).weight,
                                  in_msg_2.weight)
        assert_equal_within_error(out_msg.event_marginalize(2).weight, in_msg_3.weight)

    def test_multiple_arg_multiple_var_duplicate_var_no_concat_full_args_no_transformation(self):
        # Test outward_compute() with
        #   - Multiple predicate arguments
        #   - Multiple pattern variables
        #   - With reuse of same pattern variable (i.e. with inner-pattern binding)
        #   - No predicate argument concatenation
        #   - All predicate arguments are involved
        #   - No transformation
        # Variables
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 3, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 4, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 4, (C.real,))
        pat_var_1 = Variable('pat_var_1', VariableMetatype.Random, 3, (C.real,))
        pat_var_2 = Variable('pat_var_2', VariableMetatype.Random, 4, (C.real,))
        # Event transforms
        et_1 = EventTransform(pred_arg_1, pat_var_1)
        et_2 = EventTransform(pred_arg_2, pat_var_2)
        et_3 = EventTransform(pred_arg_3, pat_var_2)    # Inner-pattern binding on pat_var_2
        # Node
        pdfn = PDFN('test_pdfn', [et_1, et_2, et_3])

        # Relational variables
        rel_vars = [
            Variable('rel_var_1', VariableMetatype.Relational, 3),
            Variable('rel_var_2', VariableMetatype.Relational, 4),
            Variable('rel_var_3', VariableMetatype.Relational, 5),
        ]
        # Mock variable nodes and linkdata
        outward_vn, inward_vn_1, inward_vn_2 = \
            gen_vn((pred_arg_1, pred_arg_2, pred_arg_3), rel_vars), \
            gen_vn((pat_var_1,), rel_vars), \
            gen_vn((pat_var_2,), rel_vars)
        # Inward linkdata
        outward_out_ld, inward_in_ld_1, inward_in_ld_2, inward_in_ld_3 = \
            gen_ld(pdfn, outward_vn, False, False), \
            gen_ld(pdfn, inward_vn_1, True, False), \
            gen_ld(pdfn, inward_vn_2, True, False), \
            gen_ld(pdfn, inward_vn_2, True, False)      # Multiple linkdata connecting to inward_vn_2 node
        # Add link
        pdfn.add_link(outward_out_ld)
        pdfn.add_link(inward_in_ld_1)
        pdfn.add_link(inward_in_ld_3)
        pdfn.add_link(inward_in_ld_2)

        # Mock message
        msg_shape_1 = (Size([3, 4, 5]), Size([]), Size([10]), Size([3]))
        msg_shape_2 = (Size([3, 4, 5]), Size([]), Size([12]), Size([4]))
        in_msg_1 = random_message(MessageType.Particles, *msg_shape_1)
        in_msg_2 = random_message(MessageType.Particles, *msg_shape_2)

        inward_in_ld_1.read.return_value = in_msg_1
        inward_in_ld_2.read.return_value = in_msg_2
        inward_in_ld_3.read.return_value = in_msg_2
        inward_in_ld_1.new, inward_in_ld_2.new, inward_in_ld_3.new = True, True, True

        # Compute
        pdfn.compute()

        # Check outgoing linkdata called and received the correct message
        outward_out_ld.write.assert_called_once()
        out_msg: Message = outward_out_ld.write.call_args[0][0]
        # Check message content
        # For pred_arg_1, its marginal message shall equal in_msg_1
        # For both pred_arg_2 and pred_arg_3, both their marginal message shall equal in_msg_2
        assert out_msg.event_marginalize(0) == in_msg_1
        assert out_msg.event_marginalize(1) == in_msg_2
        assert out_msg.event_marginalize(2) == in_msg_2