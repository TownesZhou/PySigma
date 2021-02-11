"""
    Unit tests for ESFN class
"""
import pytest
from unittest.mock import MagicMock
import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, VariableNode, NodeConfigurationError, DVN
from pysigma.graphical.alpha_beta_nodes import ESFN
from pysigma.utils import KnowledgeServer as KS
from pysigma.utils import DistributionServer as DS
from pysigma.pattern_structures.summarization import SummarizationClass
from pysigma.pattern_structures.builtin_sum_op import sum_op_combination
from ...utils import random_message, assert_equal_within_error, assert_proportional_within_error

from ...utils import generate_positive_definite


class TestESFN:

    def test_init(self):
        # Test with a mocked Summarization op
        mock_sum_op = MagicMock(spec_set=SummarizationClass)
        esfn = ESFN('test_esfn', mock_sum_op)
        assert esfn.sum_op is mock_sum_op

    def test_precompute_check_incompatible_relational_vars_inward(self):
        # Test assertion error is raised if inward DVS's relational variables are not a subset of the outward DVN's
        #   relational variables
        # Test inward message pathway
        ran_vars = (Variable("test_ran_var_1", VariableMetatype.Random, 1, [C.real]),)
        inward_rel_vars_1 = (Variable("test_rel_var_1", VariableMetatype.Relational, 10),)
        outward_rel_vars_2 = (Variable("test_rel_var_2", VariableMetatype.Relational, 10),)
        # Mock DVN
        mock_in_dvn, mock_out_dvn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        mock_in_dvn.name = 'test_inward_dvn'
        mock_out_dvn.name = 'test_outward_dvn'
        # For the inward message pathway, incoming ld is connected from the Outward-side VN, and vice versa
        mock_in_dvn.rel_vars, mock_out_dvn.rel_vars = outward_rel_vars_2, inward_rel_vars_1
        mock_in_dvn.ran_vars, mock_out_dvn.ran_vars = ran_vars, ran_vars
        # ESFN node and linkdata
        mock_sum_op = MagicMock(spec_set=SummarizationClass)
        esfn = ESFN('test_esfn', mock_sum_op)
        in_ld, out_ld = LinkData(mock_in_dvn, esfn, True), LinkData(mock_out_dvn, esfn, False)
        in_ld.attr, out_ld.attr = {'direction': 'inward'}, {'direction': 'inward'}

        esfn.add_link(in_ld)
        esfn.add_link(out_ld)

        with pytest.raises(AssertionError) as excinfo:
            esfn.compute()

        assert str(excinfo.value) == "In test_esfn: The outward DVN's relational variable set should be an subset of " \
                                     "the inward DVN's relational variable set. Instead, found inward DVN random " \
                                     "variables: [test_rel_var_1], and outward DVN random variables: [test_rel_var_2]"

    def test_precompute_check_incompatible_relational_vars_outward(self):
        # Test assertion error is raised if inward DVS's relational variables are not a subset of the outward DVN's
        #   relational variables
        # Test outward message pathway
        ran_vars = (Variable("test_ran_var_1", VariableMetatype.Random, 1, [C.real]),)
        inward_rel_vars_1 = (Variable("test_rel_var_1", VariableMetatype.Relational, 10),)
        outward_rel_vars_2 = (Variable("test_rel_var_2", VariableMetatype.Relational, 10),)
        # Mock DVN
        mock_in_dvn, mock_out_dvn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        mock_in_dvn.name = 'test_inward_dvn'
        mock_out_dvn.name = 'test_outward_dvn'
        # For the outward message pathway, incoming ld is connected from the Inward-side VN, and vice versa
        mock_in_dvn.rel_vars, mock_out_dvn.rel_vars = inward_rel_vars_1, outward_rel_vars_2
        mock_in_dvn.ran_vars, mock_out_dvn.ran_vars = ran_vars, ran_vars
        # ESFN node and linkdata
        mock_sum_op = MagicMock(spec_set=SummarizationClass)
        esfn = ESFN('test_esfn', mock_sum_op)
        in_ld, out_ld = LinkData(mock_in_dvn, esfn, True), LinkData(mock_out_dvn, esfn, False)
        in_ld.attr, out_ld.attr = {'direction': 'outward'}, {'direction': 'outward'}

        esfn.add_link(in_ld)
        esfn.add_link(out_ld)

        with pytest.raises(AssertionError) as excinfo:
            esfn.compute()

        assert str(excinfo.value) == "In test_esfn: The outward DVN's relational variable set should be an subset of " \
                                     "the inward DVN's relational variable set. Instead, found inward DVN random " \
                                     "variables: [test_rel_var_1], and outward DVN random variables: [test_rel_var_2]"

    def test_inward_compute_identical_relational_vars_same_order(self):
        # Test that the message passed is not changed if inward-side and outward-side DVN have identical relational
        #   variables
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([10]), Size([20, 30]), Size([1, 2])
        ran_vars = (Variable("test_ran_var_1", VariableMetatype.Random, 1, [C.real]),)
        rel_vars = (Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                    Variable("test_rel_var_2", VariableMetatype.Relational, 4),
                    Variable("test_rel_var_3", VariableMetatype.Relational, 5))
        # Mock DVN
        mock_in_dvn, mock_out_dvn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        mock_in_dvn.name = 'test_inward_dvn'
        mock_out_dvn.name = 'test_outward_dvn'
        mock_in_dvn.rel_vars, mock_out_dvn.rel_vars = rel_vars, rel_vars
        mock_in_dvn.ran_vars, mock_out_dvn.ran_vars = ran_vars, ran_vars
        # ESFN node
        mock_sum_op = MagicMock(spec_set=SummarizationClass)
        esfn = ESFN('test_esfn', mock_sum_op)
        # Mock linkdata
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        mock_in_ld.vn, mock_out_ld.vn = mock_in_dvn, mock_out_dvn
        mock_in_ld.fn, mock_out_ld.fn = esfn, esfn
        mock_in_ld.to_fn, mock_out_ld.to_fn = True, False
        mock_in_ld.attr, mock_out_ld.attr = {'direction': 'inward'}, {'direction': 'inward'}
        esfn.add_link(mock_in_ld)
        esfn.add_link(mock_out_ld)
        # Message
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        # Mock in_ld return value
        mock_in_ld.new = True
        mock_in_ld.read.return_value = test_msg

        esfn.compute()

        mock_out_ld.write.assert_called_once()
        received_msg = mock_out_ld.write.call_args[0][0]
        assert test_msg == received_msg

    def test_inward_compute_identical_relational_vars_diff_order(self):
        # Test that dimension permutation
        #   variables
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([10]), Size([20, 30]), Size([1, 2])
        ran_vars = (Variable("test_ran_var_1", VariableMetatype.Random, 1, [C.real]),)
        in_rel_vars = (Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                       Variable("test_rel_var_2", VariableMetatype.Relational, 4),
                       Variable("test_rel_var_3", VariableMetatype.Relational, 5))
        out_rel_vars = (Variable("test_rel_var_3", VariableMetatype.Relational, 5),
                        Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                        Variable("test_rel_var_2", VariableMetatype.Relational, 4))
        # Mock DVN
        mock_in_dvn, mock_out_dvn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        mock_in_dvn.name = 'test_inward_dvn'
        mock_out_dvn.name = 'test_outward_dvn'
        mock_in_dvn.rel_vars, mock_out_dvn.rel_vars = in_rel_vars, out_rel_vars
        mock_in_dvn.ran_vars, mock_out_dvn.ran_vars = ran_vars, ran_vars
        # ESFN node
        mock_sum_op = MagicMock(spec_set=SummarizationClass)
        esfn = ESFN('test_esfn', mock_sum_op)
        # Mock linkdata
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        mock_in_ld.vn, mock_out_ld.vn = mock_in_dvn, mock_out_dvn
        mock_in_ld.fn, mock_out_ld.fn = esfn, esfn
        mock_in_ld.to_fn, mock_out_ld.to_fn = True, False
        mock_in_ld.attr, mock_out_ld.attr = {'direction': 'inward'}, {'direction': 'inward'}
        esfn.add_link(mock_in_ld)
        esfn.add_link(mock_out_ld)
        # Message
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        # Mock in_ld return value
        mock_in_ld.new = True
        mock_in_ld.read.return_value = test_msg

        esfn.compute()

        mock_out_ld.write.assert_called_once()
        received_msg: Message = mock_out_ld.write.call_args[0][0]
        expected_msg = test_msg.batch_permute([2, 0, 1])
        assert received_msg == expected_msg

    def test_inward_compute_expand_same_order(self):
        # Test that dimension expansion
        #   variables
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20, 30]), Size([1, 2])
        ran_vars = (Variable("test_ran_var_1", VariableMetatype.Random, 1, [C.real]),)
        outward_rel_vars = (Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                       Variable("test_rel_var_2", VariableMetatype.Relational, 4))
        inward_rel_vars = (Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                        Variable("test_rel_var_2", VariableMetatype.Relational, 4),
                        Variable("test_rel_var_3", VariableMetatype.Relational, 5),
                        Variable("test_rel_var_4", VariableMetatype.Relational, 6))
        # Mock DVN
        mock_in_dvn, mock_out_dvn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        mock_in_dvn.name = 'test_inward_dvn'
        mock_out_dvn.name = 'test_outward_dvn'
        mock_in_dvn.rel_vars, mock_out_dvn.rel_vars = outward_rel_vars, inward_rel_vars
        mock_in_dvn.ran_vars, mock_out_dvn.ran_vars = ran_vars, ran_vars
        # ESFN node
        mock_sum_op = MagicMock(spec_set=SummarizationClass)
        esfn = ESFN('test_esfn', mock_sum_op)
        # Mock linkdata
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        mock_in_ld.vn, mock_out_ld.vn = mock_in_dvn, mock_out_dvn
        mock_in_ld.fn, mock_out_ld.fn = esfn, esfn
        mock_in_ld.to_fn, mock_out_ld.to_fn = True, False
        mock_in_ld.attr, mock_out_ld.attr = {'direction': 'inward'}, {'direction': 'inward'}
        esfn.add_link(mock_in_ld)
        esfn.add_link(mock_out_ld)
        # Message
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        # Mock in_ld return value
        mock_in_ld.new = True
        mock_in_ld.read.return_value = test_msg

        esfn.compute()

        mock_out_ld.write.assert_called_once()
        received_msg: Message = mock_out_ld.write.call_args[0][0]
        expected_msg = test_msg.batch_unsqueeze(2).batch_unsqueeze(3).batch_expand([-1, -1, 5, 6])
        assert received_msg == expected_msg

    def test_inward_compute_expand_diff_order(self):
        # Test that dimension expansion and permutation
        #   variables
        b_shape, p_shape, s_shape, e_shape = Size([3, 4]), Size([10]), Size([20, 30]), Size([1, 2])
        ran_vars = (Variable("test_ran_var_1", VariableMetatype.Random, 1, [C.real]),)
        outward_rel_vars = (Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                       Variable("test_rel_var_2", VariableMetatype.Relational, 4))
        inward_rel_vars = (Variable("test_rel_var_3", VariableMetatype.Relational, 5),
                           Variable("test_rel_var_2", VariableMetatype.Relational, 4),
                           Variable("test_rel_var_4", VariableMetatype.Relational, 6),
                           Variable("test_rel_var_1", VariableMetatype.Relational, 3),)
        # Mock DVN
        mock_in_dvn, mock_out_dvn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        mock_in_dvn.name = 'test_inward_dvn'
        mock_out_dvn.name = 'test_outward_dvn'
        mock_in_dvn.rel_vars, mock_out_dvn.rel_vars = outward_rel_vars, inward_rel_vars
        mock_in_dvn.ran_vars, mock_out_dvn.ran_vars = ran_vars, ran_vars
        # ESFN node
        mock_sum_op = MagicMock(spec_set=SummarizationClass)
        esfn = ESFN('test_esfn', mock_sum_op)
        # Mock linkdata
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        mock_in_ld.vn, mock_out_ld.vn = mock_in_dvn, mock_out_dvn
        mock_in_ld.fn, mock_out_ld.fn = esfn, esfn
        mock_in_ld.to_fn, mock_out_ld.to_fn = True, False
        mock_in_ld.attr, mock_out_ld.attr = {'direction': 'inward'}, {'direction': 'inward'}
        esfn.add_link(mock_in_ld)
        esfn.add_link(mock_out_ld)
        # Message
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        # Mock in_ld return value
        mock_in_ld.new = True
        mock_in_ld.read.return_value = test_msg

        esfn.compute()

        mock_out_ld.write.assert_called_once()
        received_msg: Message = mock_out_ld.write.call_args[0][0]
        expected_msg = test_msg.batch_unsqueeze(2).batch_unsqueeze(3).batch_expand([-1, -1, 5, 6]).batch_permute([2, 1, 3, 0])
        assert received_msg == expected_msg

    def test_outward_compute_identical_relational_vars_same_order(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([10]), Size([20, 30]), Size([1, 2])
        ran_vars = (Variable("test_ran_var_1", VariableMetatype.Random, 1, [C.real]),)
        rel_vars = (Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                    Variable("test_rel_var_2", VariableMetatype.Relational, 4),
                    Variable("test_rel_var_3", VariableMetatype.Relational, 5))
        # Mock DVN
        mock_in_dvn, mock_out_dvn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        mock_in_dvn.name = 'test_inward_dvn'
        mock_out_dvn.name = 'test_outward_dvn'
        mock_in_dvn.rel_vars, mock_out_dvn.rel_vars = rel_vars, rel_vars
        mock_in_dvn.ran_vars, mock_out_dvn.ran_vars = ran_vars, ran_vars
        # ESFN node
        mock_sum_op = MagicMock(spec_set=SummarizationClass)
        esfn = ESFN('test_esfn', mock_sum_op)       # Using a mocked sum op here
        # Mock linkdata
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        mock_in_ld.vn, mock_out_ld.vn = mock_in_dvn, mock_out_dvn
        mock_in_ld.fn, mock_out_ld.fn = esfn, esfn
        mock_in_ld.to_fn, mock_out_ld.to_fn = True, False
        mock_in_ld.attr, mock_out_ld.attr = {'direction': 'outward'}, {'direction': 'outward'}
        esfn.add_link(mock_in_ld)
        esfn.add_link(mock_out_ld)
        # Message
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        # Mock in_ld return value
        mock_in_ld.new = True
        mock_in_ld.read.return_value = test_msg

        esfn.compute()

        mock_out_ld.write.assert_called_once()
        received_msg = mock_out_ld.write.call_args[0][0]
        assert test_msg == received_msg

        # Also check that the sum op was not called
        mock_sum_op.assert_not_called()

    def test_outward_compute_identical_relational_vars_diff_order(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5]), Size([10]), Size([20, 30]), Size([1, 2])
        ran_vars = (Variable("test_ran_var_1", VariableMetatype.Random, 1, [C.real]),)
        rel_vars_1 = (Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                      Variable("test_rel_var_2", VariableMetatype.Relational, 4),
                      Variable("test_rel_var_3", VariableMetatype.Relational, 5))
        rel_vars_2 = (Variable("test_rel_var_2", VariableMetatype.Relational, 4),
                      Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                      Variable("test_rel_var_3", VariableMetatype.Relational, 5))
        # Mock DVN
        mock_in_dvn, mock_out_dvn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        mock_in_dvn.name = 'test_inward_dvn'
        mock_out_dvn.name = 'test_outward_dvn'
        mock_in_dvn.rel_vars, mock_out_dvn.rel_vars = rel_vars_1, rel_vars_2
        mock_in_dvn.ran_vars, mock_out_dvn.ran_vars = ran_vars, ran_vars
        # ESFN node
        mock_sum_op = MagicMock(spec_set=SummarizationClass)
        esfn = ESFN('test_esfn', mock_sum_op)       # Using a mocked sum op here
        # Mock linkdata
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        mock_in_ld.vn, mock_out_ld.vn = mock_in_dvn, mock_out_dvn
        mock_in_ld.fn, mock_out_ld.fn = esfn, esfn
        mock_in_ld.to_fn, mock_out_ld.to_fn = True, False
        mock_in_ld.attr, mock_out_ld.attr = {'direction': 'outward'}, {'direction': 'outward'}
        esfn.add_link(mock_in_ld)
        esfn.add_link(mock_out_ld)
        # Message
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        # Mock in_ld return value
        mock_in_ld.new = True
        mock_in_ld.read.return_value = test_msg

        esfn.compute()

        mock_out_ld.write.assert_called_once()
        expected_msg = test_msg.batch_permute([1, 0, 2])
        received_msg = mock_out_ld.write.call_args[0][0]
        assert received_msg == expected_msg

        # Also check that the sum op was not called
        mock_sum_op.assert_not_called()

    def test_outward_compute_summarization_diff_order(self):
        b_shape, p_shape, s_shape, e_shape = Size([3, 4, 5, 6, 7]), Size([10]), Size([20, 30]), Size([1, 2])
        ran_vars = (Variable("test_ran_var_1", VariableMetatype.Random, 1, [C.real]),)
        inward_rel_vars = (Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                           Variable("test_rel_var_2", VariableMetatype.Relational, 4),
                           Variable("test_rel_var_3", VariableMetatype.Relational, 5),
                           Variable("test_rel_var_4", VariableMetatype.Relational, 6),
                           Variable("test_rel_var_5", VariableMetatype.Relational, 7))
        outward_rel_vars = (Variable("test_rel_var_2", VariableMetatype.Relational, 4),
                            Variable("test_rel_var_1", VariableMetatype.Relational, 3),
                            Variable("test_rel_var_3", VariableMetatype.Relational, 5))
        # Mock DVN
        mock_in_dvn, mock_out_dvn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        mock_in_dvn.name = 'test_inward_dvn'
        mock_out_dvn.name = 'test_outward_dvn'
        mock_in_dvn.rel_vars, mock_out_dvn.rel_vars = inward_rel_vars, outward_rel_vars
        mock_in_dvn.ran_vars, mock_out_dvn.ran_vars = ran_vars, ran_vars
        # ESFN node
        esfn = ESFN('test_esfn', sum_op_combination)       # Use default sum op
        # Mock linkdata
        mock_in_ld, mock_out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        mock_in_ld.vn, mock_out_ld.vn = mock_in_dvn, mock_out_dvn
        mock_in_ld.fn, mock_out_ld.fn = esfn, esfn
        mock_in_ld.to_fn, mock_out_ld.to_fn = True, False
        mock_in_ld.attr, mock_out_ld.attr = {'direction': 'outward'}, {'direction': 'outward'}
        esfn.add_link(mock_in_ld)
        esfn.add_link(mock_out_ld)
        # Message
        test_msg = random_message(MessageType.Dual, b_shape, p_shape, s_shape, e_shape)
        test_msg.attr = {'dist_class': D, 'dist_info': {}}
        # Mock in_ld return value
        mock_in_ld.new = True
        mock_in_ld.read.return_value = test_msg

        esfn.compute()

        mock_out_ld.write.assert_called_once()
        received_msg: Message = mock_out_ld.write.call_args[0][0]

        # Check message shape
        assert received_msg.b_shape == Size([4, 3, 5])

        # Check message content
        expected_param = test_msg.parameter.sum(dim=[3, 4]).permute([1, 0, 2, 3])
        assert_equal_within_error(received_msg.parameter, expected_param)
        # expected_weight = test_msg.weight.log().sum(dim=[3, 4]).permute([1, 0, 2, 3, 4]).exp()
        expected_weight = test_msg.weight.log().sum(dim=[3, 4])
        expected_weight -= expected_weight.max()
        expected_weight = expected_weight.exp().permute([1, 0, 2, 3, 4])
        expected_weight /= expected_weight.sum(dim=[-1, -2], keepdim=True)
        assert_equal_within_error(received_msg.weight, expected_weight, 1e-5)


