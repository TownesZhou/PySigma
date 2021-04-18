"""
    Unit tests for RMFN class
"""
import pytest
from unittest.mock import MagicMock

import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, DVN
from pysigma.graphical.alpha_nodes import RMFN
from pysigma.pattern_structures.variable_map import VariableMapCallable
from pysigma.pattern_structures.builtin_sum_op import sum_op_combination
from ...utils import random_message, assert_equal_within_error, assert_proportional_within_error


class TestRMFN:

    def test_init(self):
        # Test specifically that the dictionaries are properly instantiated
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 20)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 30)
        pred_arg_4 = Variable("pred_arg_4", VariableMetatype.Relational, 40)
        pred_arg_5 = Variable("pred_arg_5", VariableMetatype.Relational, 50)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 10)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 40)
        pat_var_c = Variable("pat_var_c", VariableMetatype.Relational, 50)
        # Mappings
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_a, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_a, False, False)
        varmap_4 = VariableMapCallable(lambda x: x, pred_arg_4, pat_var_b, False, False)
        varmap_5 = VariableMapCallable(lambda x: x, pred_arg_5, pat_var_c, False, False)

        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3, varmap_4, varmap_5])

        expected_vm_pred_args = {pred_arg_1, pred_arg_2, pred_arg_3, pred_arg_4, pred_arg_5}
        expected_vm_pat_vars = {pat_var_a, pat_var_b, pat_var_c}
        expected_pred_arg2var_map = {
            pred_arg_1: varmap_1,
            pred_arg_2: varmap_2,
            pred_arg_3: varmap_3,
            pred_arg_4: varmap_4,
            pred_arg_5: varmap_5
        }
        expected_pat_var2pred_args = {
            pat_var_a: [pred_arg_1, pred_arg_2, pred_arg_3],
            pat_var_b: [pred_arg_4],
            pat_var_c: [pred_arg_5]
        }

        assert rmfn.vm_pred_args == expected_vm_pred_args
        assert rmfn.vm_pat_vars == expected_vm_pat_vars
        assert rmfn.pred_arg2var_map == expected_pred_arg2var_map
        assert rmfn.pat_var2pred_args == expected_pat_var2pred_args

    def test_add_link_correct_rel_vars(self):
        # Test that linkdata of all directions with correct relational variables can be properly registered
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 20)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 30)
        pred_arg_4 = Variable("pred_arg_4", VariableMetatype.Relational, 40)
        pred_arg_5 = Variable("pred_arg_5", VariableMetatype.Relational, 50)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 10)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 40)
        pat_var_c = Variable("pat_var_c", VariableMetatype.Relational, 50)
        # Mappings
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_a, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_a, False, False)
        varmap_4 = VariableMapCallable(lambda x: x, pred_arg_4, pat_var_b, False, False)
        varmap_5 = VariableMapCallable(lambda x: x, pred_arg_5, pat_var_c, False, False)

        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3, varmap_4, varmap_5])

        # Mock random variable
        ran_var = Variable("ran_var", VariableMetatype.Random, 3, (C.real,))
        # Mock VN and
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.name = "outward_vn"
        inward_vn.name = "inward_vn"
        outward_vn.rel_vars = (pred_arg_5, pred_arg_4, pred_arg_3, pred_arg_2, pred_arg_1)
        outward_vn.ran_vars = (ran_var,)
        inward_vn.rel_vars = (pat_var_c, pat_var_b, pat_var_a)
        inward_vn.ran_vars = (ran_var,)
        # Test linkdata of all directions
        inward_in_ld = LinkData(outward_vn, rmfn, True, (Size([]),) * 4, direction='inward')
        inward_out_ld = LinkData(inward_vn, rmfn, False, (Size([]),) * 4, direction='inward')
        outward_in_ld = LinkData(inward_vn, rmfn, True, (Size([]),) * 4, direction='outward')
        outward_out_ld = LinkData(outward_vn, rmfn, False, (Size([]),) * 4, direction='outward')

        # Test add link
        rmfn.add_link(inward_in_ld)
        rmfn.add_link(inward_out_ld)
        rmfn.add_link(outward_in_ld)
        rmfn.add_link(outward_out_ld)

        assert rmfn.labeled_ld_pair == {
            'inward': (inward_in_ld, inward_out_ld),
            'outward': (outward_in_ld, outward_out_ld)
        }

    def test_add_link_incorrect_rel_vars(self):
        # Test assertion error is raised if the VN linkdata connects to has incorrect set of relational variables
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 20)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 30)
        pred_arg_4 = Variable("pred_arg_4", VariableMetatype.Relational, 40)
        pred_arg_5 = Variable("pred_arg_5", VariableMetatype.Relational, 50)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 10)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 40)
        pat_var_c = Variable("pat_var_c", VariableMetatype.Relational, 50)
        pat_var_d = Variable("pat_var_d", VariableMetatype.Relational, 60)
        # Mappings
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_a, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_a, False, False)
        varmap_4 = VariableMapCallable(lambda x: x, pred_arg_4, pat_var_b, False, False)
        varmap_5 = VariableMapCallable(lambda x: x, pred_arg_5, pat_var_c, False, False)

        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3, varmap_4, varmap_5])

        # Mock random variable
        ran_var = Variable("ran_var", VariableMetatype.Random, 3, (C.real,))
        # Mock VN and
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.name = "outward_vn"
        inward_vn.name = "inward_vn"
        # Incorrect set of relational variables
        outward_vn.rel_vars = (pred_arg_5, pred_arg_4, pred_arg_3, pred_arg_2)  # Missing one
        inward_vn.rel_vars = (pat_var_c, pat_var_b, pat_var_a, pat_var_d)  # Has one more
        outward_vn.ran_vars = (ran_var,)
        inward_vn.ran_vars = (ran_var,)
        # The correct set of relational variables
        expected_pred_args = {pred_arg_1, pred_arg_2, pred_arg_3, pred_arg_4, pred_arg_5}
        expected_pat_vars = {pat_var_a, pat_var_b, pat_var_c}
        # Test linkdata of all directions
        inward_in_ld = LinkData(outward_vn, rmfn, True, (Size([]),) * 4, direction='inward')
        inward_out_ld = LinkData(inward_vn, rmfn, False, (Size([]),) * 4, direction='inward')
        outward_in_ld = LinkData(inward_vn, rmfn, True, (Size([]),) * 4, direction='outward')
        outward_out_ld = LinkData(outward_vn, rmfn, False, (Size([]),) * 4, direction='outward')

        # Add inward incoming linkdata. The predicate arguments should be compared
        with pytest.raises(AssertionError) as excinfo:
            rmfn.add_link(inward_in_ld)
        assert str(excinfo.value) == "In {}: the linkdata {} to add has incompatible predicate arguments. Expect set " \
                                     "of predicate arguments {} (inferred from the given variable mappings), but " \
                                     "found {} in the linkdata." \
            .format(rmfn.name, inward_in_ld, expected_pred_args, set(outward_vn.rel_vars))

        # Add outward outgoing linkdata. The predicate arguments should be compared
        with pytest.raises(AssertionError) as excinfo:
            rmfn.add_link(outward_out_ld)
        assert str(excinfo.value) == "In {}: the linkdata {} to add has incompatible predicate arguments. Expect set " \
                                     "of predicate arguments {} (inferred from the given variable mappings), but " \
                                     "found {} in the linkdata." \
            .format(rmfn.name, outward_out_ld, expected_pred_args, set(outward_vn.rel_vars))

        # Add inward outgoing linkdata. The pattern variables should be compared
        with pytest.raises(AssertionError) as excinfo:
            rmfn.add_link(inward_out_ld)
        assert str(excinfo.value) == "In {}: the linkdata {} to add has incompatible pattern variables. Expect set " \
                                     "of pattern variables {} (inferred from the given variable mappings), but " \
                                     "found {} in the linkdata." \
            .format(rmfn.name, inward_out_ld, expected_pat_vars, set(inward_vn.rel_vars))

        # Add outward incoming linkdata. The pattern variables should be compared
        with pytest.raises(AssertionError) as excinfo:
            rmfn.add_link(outward_in_ld)
        assert str(excinfo.value) == "In {}: the linkdata {} to add has incompatible pattern variables. Expect set " \
                                     "of pattern variables {} (inferred from the given variable mappings), but " \
                                     "found {} in the linkdata." \
            .format(rmfn.name, outward_in_ld, expected_pat_vars, set(inward_vn.rel_vars))

    def test_inward_compute_no_binding_id_map_same_order(self):
        # Test return message correctness without pattern variable binding, with identity variable map, and need not
        #   to permute batch dimensions of the message.
        # Incoming Message shape
        in_msg_shape = (Size([5, 10, 25]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 5)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 10)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 25)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 10)
        pat_var_c = Variable("pat_var_c", VariableMetatype.Relational, 25)
        # Mappings
        # All pairs use identity value map
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_b, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_c, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3)
        inward_vn.rel_vars = (pat_var_a, pat_var_b, pat_var_c)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg = out_ld.write.call_args[0][0]

        # In this test, the out_msg should exactly equal to the in_msg
        assert out_msg == in_msg

    def test_inward_compute_no_binding_id_map_diff_order(self):
        # Test return message correctness without pattern variable binding, with identity variable map, and need to
        #   permute batch dimensions of the message because target vn has different pattern variable ordering
        # Incoming Message shape
        in_msg_shape = (Size([5, 10, 25]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 5)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 10)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 25)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 10)
        pat_var_c = Variable("pat_var_c", VariableMetatype.Relational, 25)
        # Mappings
        # All pairs use identity value map
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_b, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_c, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3)
        # Target variable node has altered pattern variable ordering
        inward_vn.rel_vars = (pat_var_c, pat_var_a, pat_var_b)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg = out_ld.write.call_args[0][0]

        # In this test, the out_msg be the batch-permuted in_msg
        expected_perm_order = [2, 0, 1]
        assert out_msg == in_msg.batch_permute(expected_perm_order)

    def test_inward_compute_no_binding_non_id_map_single_1(self):
        # Test return message correctness without pattern variable binding, and with a single variable map that is a
        #   restrictive variable maps that select message slices from a subset of predicate argument's domain
        # Type 1: restrictive map with same message slice order as input message
        # Incoming Message shape.
        in_msg_shape = (Size([10]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)

        # A restrictive mapping: only values in range [3, 7] of the predicate argument is selected
        def val_map(x):
            return x + 3  # [0, 1, 2, 3, 4] becomes [3, 4, 5, 6, 7]

        # All pairs use identity value map
        varmap_1 = VariableMapCallable(val_map, pred_arg_1, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1,)
        # Target variable node has altered pattern variable ordering
        inward_vn.rel_vars = (pat_var_a,)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # In this test, the out_msg be sub-slices of the in_msg
        # Check shape
        expected_msg_shape = (Size([5]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check content by checking message parameter and weight
        slice_ids = torch.tensor([3, 4, 5, 6, 7])
        expt_out_msg_param = in_msg.parameter.index_select(0, slice_ids)
        expt_out_msg_weight = in_msg.weight.index_select(0, slice_ids)

        assert_equal_within_error(out_msg.parameter, expt_out_msg_param)
        assert_equal_within_error(out_msg.weight, expt_out_msg_weight)

    def test_inward_compute_no_binding_non_id_map_single_2(self):
        # Test return message correctness without pattern variable binding, and with a single variable map that is a
        #   restrictive variable maps that select message slices from a subset of predicate argument's domain
        # Type 2: restrictive map with arbitrary index order
        # Incoming Message shape.
        in_msg_shape = (Size([10]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)

        # A restrictive mapping with arbitrary index order
        target_pred_arg_ids = torch.tensor([7, 5, 3, 4, 6])

        def val_map(x):
            return target_pred_arg_ids  # [0, 1, 2, 3, 4] becomes [7, 5, 3, 4, 6]

        # All pairs use identity value map
        varmap_1 = VariableMapCallable(val_map, pred_arg_1, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1,)
        # Target variable node has altered pattern variable ordering
        inward_vn.rel_vars = (pat_var_a,)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # In this test, the out_msg be sub-slices of the in_msg
        # Check shape
        expected_msg_shape = (Size([5]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check content by checking message parameter and weight
        # Check slice by slice
        for i in range(target_pred_arg_ids.shape[0]):
            expt_out_msg_param_slice = in_msg.parameter[target_pred_arg_ids[i]]
            expt_out_msg_weight_slice = in_msg.weight[target_pred_arg_ids[i]]

            assert_equal_within_error(out_msg.parameter[i], expt_out_msg_param_slice)
            assert_equal_within_error(out_msg.weight[i], expt_out_msg_weight_slice)

    def test_inward_compute_no_binding_non_id_map_single_3(self):
        # Test return message correctness without pattern variable binding, and with a single variable map that is a
        #   expansive variable maps that select message slices from a subset of predicate argument's domain
        # Type 3: expansive map with arbitrary index order
        # Incoming Message shape.
        in_msg_shape = (Size([10]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 6)

        # An expansive mapping with arbitrary index order
        target_pred_arg_ids = torch.tensor([-1, 2, 4, -1, 5, -1])

        def val_map(x):
            return target_pred_arg_ids  # [0, 1, 2, 3, 4, 5] becomes [-1, 2, 4, -1, 5, -1]

        # All pairs use identity value map
        varmap_1 = VariableMapCallable(val_map, pred_arg_1, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1,)
        # Target variable node has altered pattern variable ordering
        inward_vn.rel_vars = (pat_var_a,)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # In this test, the out_msg be sub-slices of the in_msg
        # Check shape
        expected_msg_shape = (Size([6]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check content by checking message parameter and weight

        # Check that all slice whose index is in target_pred_arg_ids equals the corresponding slice from in_msg
        #   and that all other slice whose index not mentioned in target_pred_arg_ids defaults to identity message slice
        for i in range(target_pred_arg_ids.shape[0]):
            if 0 <= target_pred_arg_ids[i].item() < pred_arg_1.size:
                expt_out_msg_param_slice = in_msg.parameter[target_pred_arg_ids[i]]
                expt_out_msg_weight_slice = in_msg.weight[target_pred_arg_ids[i]]

                assert_equal_within_error(out_msg.parameter[i], expt_out_msg_param_slice)
                assert_equal_within_error(out_msg.weight[i], expt_out_msg_weight_slice)
            else:
                assert_equal_within_error(out_msg.parameter[i], torch.tensor([0.]))
                assert_proportional_within_error(out_msg.weight[i], torch.tensor([1.]), dims=[-1])  # Only 1 ran var

    def test_inward_compute_no_binding_non_id_map_multiple(self):
        # Test return message correctness
        #   - without binding
        #   - arbitrary mapping with arbitrary index ordering
        #   - multiple variables
        # Incoming Message shape.
        in_msg_shape = (Size([10, 15, 20]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 15)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 20)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 6)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 6)
        pat_var_c = Variable("pat_var_c", VariableMetatype.Relational, 6)

        # An expansive mapping with arbitrary index order
        target_pred_arg_1_ids = torch.tensor([-1, 2, 4, -1, 5, -1])
        target_pred_arg_2_ids = torch.tensor([1, 3, 4, -1, -1, 2])
        target_pred_arg_3_ids = torch.tensor([-1, 1, -1, 2, -1, 3])

        # Variable maps
        varmap_1 = VariableMapCallable(lambda x: target_pred_arg_1_ids, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: target_pred_arg_2_ids, pred_arg_2, pat_var_b, False, False)
        varmap_3 = VariableMapCallable(lambda x: target_pred_arg_3_ids, pred_arg_3, pat_var_c, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3)
        # Target variable node has altered pattern variable ordering
        inward_vn.rel_vars = (pat_var_a, pat_var_b, pat_var_c)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # In this test, the out_msg be sub-slices of the in_msg
        # Check shape
        expected_msg_shape = (Size([6, 6, 6]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check content by checking message parameter and weight

        # For multi-dimensional batch, if a slice's any one dimensional index is not in target_pred_args_ids, then this
        #   slice should be an identity slice. Only those slices whose all indices are in target_pred_arg_ids shall be
        #   slices taken from the incoming message.
        for pat_var_i, pred_arg_i in enumerate(target_pred_arg_1_ids.tolist()):
            for pat_var_j, pred_arg_j in enumerate(target_pred_arg_2_ids.tolist()):
                for pat_var_k, pred_arg_k in enumerate(target_pred_arg_3_ids.tolist()):
                    actual_out_msg_param_slice = out_msg.parameter[pat_var_i, pat_var_j, pat_var_k]
                    actual_out_msg_weight_slice = out_msg.weight[pat_var_i, pat_var_j, pat_var_k]

                    if 0 <= pred_arg_i < pred_arg_1.size and 0 <= pred_arg_j < pred_arg_2.size and 0 <= pred_arg_k < pred_arg_3.size:
                        expt_out_msg_param_slice = in_msg.parameter[pred_arg_i, pred_arg_j, pred_arg_k]
                        expt_out_msg_weight_slice = in_msg.weight[pred_arg_i, pred_arg_j, pred_arg_k]

                        assert_equal_within_error(actual_out_msg_param_slice, expt_out_msg_param_slice)
                        assert_equal_within_error(actual_out_msg_weight_slice, expt_out_msg_weight_slice)

                    else:
                        assert_equal_within_error(actual_out_msg_param_slice, torch.tensor([0.]))
                        assert_proportional_within_error(actual_out_msg_weight_slice, torch.tensor([1.]), dims=[-1])

    def test_inward_compute_single_binding_two_args_id_map(self):
        # Test return message correctness
        #   - A single binding that binds two predicate arguments
        #   - identity mapping
        # Incoming Message shape.
        in_msg_shape = (Size([10, 15, 20]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 15)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 20)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 8)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 6)

        # Variable maps: pred_arg_1 and pred_arg_2 binds on pat_var_a
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_a, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_b, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        # Outward side predicate arguments
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3)
        # Inward side pattern variables
        inward_vn.rel_vars = (pat_var_a, pat_var_b)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # Check shape
        expected_msg_shape = (Size([8, 6]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check content by checking message parameter and weight

        # The outgoing message slices should be the diagonals of the incoming messages if there are bound predicate
        #   arguments.
        for i in range(pat_var_a.size):
            expt_out_msg_param_slice = in_msg.parameter[i, i, :pat_var_b.size]
            expt_out_msg_weight_slice = in_msg.weight[i, i, :pat_var_b.size]
            actual_out_msg_param_slice = out_msg.parameter[i, :]
            actual_out_msg_weight_slice = out_msg.weight[i, :]

            assert_equal_within_error(actual_out_msg_param_slice, expt_out_msg_param_slice)
            assert_equal_within_error(actual_out_msg_weight_slice, expt_out_msg_weight_slice)

    def test_inward_compute_single_binding_two_args_non_id_map(self):
        # Test return message correctness
        #   - A single binding that binds two predicate arguments
        #   - Arbitrary mapping
        # Incoming Message shape.
        in_msg_shape = (Size([10, 15, 20]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 15)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 20)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 8)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 6)

        # Declare a mapping for pred_arg_1. Others are left with the identity mapping.
        target_pred_arg_1_ids = torch.tensor([8, 2, -1, 5, 14, 3, 0, 9])
        varmap_1 = VariableMapCallable(lambda x: target_pred_arg_1_ids, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_a, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_b, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        # Outward side predicate arguments
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3)
        # Inward side pattern variables
        inward_vn.rel_vars = (pat_var_a, pat_var_b)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # Check shape
        expected_msg_shape = (Size([8, 6]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check content by checking message parameter and weight

        # The outgoing message slices should be the diagonals of the incoming messages if there are bound predicate
        #   arguments.
        # If the outgoing message slice happens to have some dimensional index that are not mapped to any of its
        #   corresponding predicate argument's index, then this slice should be an identity slice.
        for i in range(pat_var_a.size):
            pred_arg_1_val = target_pred_arg_1_ids.tolist()[i]
            actual_out_msg_param_slice = out_msg.parameter[i, :]
            actual_out_msg_weight_slice = out_msg.weight[i, :]

            if 0 <= pred_arg_1_val < pred_arg_1.size:
                expt_out_msg_param_slice = in_msg.parameter[pred_arg_1_val, i, :pat_var_b.size]
                expt_out_msg_weight_slice = in_msg.weight[pred_arg_1_val, i, :pat_var_b.size]

                assert_equal_within_error(actual_out_msg_param_slice, expt_out_msg_param_slice)
                assert_equal_within_error(actual_out_msg_weight_slice, expt_out_msg_weight_slice)

            else:
                assert_equal_within_error(actual_out_msg_param_slice, torch.tensor([0.]))
                assert_proportional_within_error(actual_out_msg_weight_slice, torch.tensor([1.]), dims=[-1])

    def test_inward_compute_single_binding_multiple_args_id_map(self):
        # Test return message correctness
        #   - A single binding that binds more than two predicate arguments
        #   - Identity mapping
        # Incoming Message shape.
        in_msg_shape = (Size([10, 15, 20, 25]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 15)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 20)
        pred_arg_4 = Variable("pred_arg_4", VariableMetatype.Relational, 25)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 16)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 10)

        # Identity mapping for every predicate argument
        # Binds pred_arg_1, pred_arg_2, and pred_arg_3 together on pat_var_a
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_a, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_a, False, False)
        varmap_4 = VariableMapCallable(lambda x: x, pred_arg_4, pat_var_b, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        # Outward side predicate arguments
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3, pred_arg_4)
        # Inward side pattern variables
        inward_vn.rel_vars = (pat_var_a, pat_var_b)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3, varmap_4])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # Check shape
        expected_msg_shape = (Size([16, 10]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check content by checking message parameter and weight

        # The outgoing message slices should be the diagonals of the incoming messages if there are bound predicate
        #   arguments.
        # If the outgoing message slice happens to have some dimensional index that are not mapped to any of its
        #   corresponding predicate argument's index, then this slice should be an identity slice.
        for i in range(pat_var_a.size):
            actual_out_msg_param_slice = out_msg.parameter[i, :]
            actual_out_msg_weight_slice = out_msg.weight[i, :]

            if 0 <= i < pred_arg_1.size and 0 <= i < pred_arg_2.size and 0 <= i < pred_arg_3.size:
                expt_out_msg_param_slice = in_msg.parameter[i, i, i, :pat_var_b.size]
                expt_out_msg_weight_slice = in_msg.weight[i, i, i, :pat_var_b.size]

                assert_equal_within_error(actual_out_msg_param_slice, expt_out_msg_param_slice)
                assert_equal_within_error(actual_out_msg_weight_slice, expt_out_msg_weight_slice)

            else:
                assert_equal_within_error(actual_out_msg_param_slice, torch.tensor([0.]))
                assert_proportional_within_error(actual_out_msg_weight_slice, torch.tensor([1.]), dims=[-1])

    def test_inward_compute_multiple_binding_multiple_args_id_map_1(self):
        # Test return message correctness
        #   - Multiple bindings that bind more than two predicate arguments
        #   - Identity mapping
        # Incoming Message shape.
        in_msg_shape = (Size([10, 15, 20, 25, 30]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 15)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 20)
        pred_arg_4 = Variable("pred_arg_4", VariableMetatype.Relational, 25)
        pred_arg_5 = Variable("pred_arg_5", VariableMetatype.Relational, 30)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 16)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 28)

        # Identity mapping for every predicate argument
        # Binds pred_arg_1, pred_arg_2, with pred_arg_3 together on pat_var_a, and pred_arg_4 with pred_arg_5 on pat_var_b
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_a, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_a, False, False)
        varmap_4 = VariableMapCallable(lambda x: x, pred_arg_4, pat_var_b, False, False)
        varmap_5 = VariableMapCallable(lambda x: x, pred_arg_5, pat_var_b, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        # Outward side predicate arguments
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3, pred_arg_4, pred_arg_5)
        # Inward side pattern variables
        inward_vn.rel_vars = (pat_var_a, pat_var_b)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3, varmap_4, varmap_5])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # Check shape
        expected_msg_shape = (Size([16, 28]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check content by checking message parameter and weight

        # The outgoing message slices should be the diagonals of the incoming messages if there are bound predicate
        #   arguments.
        # If the outgoing message slice happens to have some dimensional index that are not mapped to any of its
        #   corresponding predicate argument's index, then this slice should be an identity slice.
        for i in range(pat_var_a.size):
            for j in range(pat_var_b.size):
                actual_out_msg_param_slice = out_msg.parameter[i, j]
                actual_out_msg_weight_slice = out_msg.weight[i, j]

                if 0 <= i < pred_arg_1.size and 0 <= i < pred_arg_2.size and 0 <= i < pred_arg_3.size and \
                        0 <= j < pred_arg_4.size and 0 <= j < pred_arg_5.size:
                    expt_out_msg_param_slice = in_msg.parameter[i, i, i, j, j]
                    expt_out_msg_weight_slice = in_msg.weight[i, i, i, j, j]

                    assert_equal_within_error(actual_out_msg_param_slice, expt_out_msg_param_slice)
                    assert_equal_within_error(actual_out_msg_weight_slice, expt_out_msg_weight_slice)

                else:
                    assert_equal_within_error(actual_out_msg_param_slice, torch.tensor([0.]))
                    assert_proportional_within_error(actual_out_msg_weight_slice, torch.tensor([1.]), dims=[-1])

    def test_inward_compute_multiple_binding_multiple_args_id_map_2(self):
        # Test return message correctness
        #   - Multiple bindings that bind more than two predicate arguments
        #   - Identity mapping
        #   - An alternative order of outgoing message's corresponding pattern variables
        # Incoming Message shape.
        in_msg_shape = (Size([10, 15, 20, 25, 30]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 15)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 20)
        pred_arg_4 = Variable("pred_arg_4", VariableMetatype.Relational, 25)
        pred_arg_5 = Variable("pred_arg_5", VariableMetatype.Relational, 30)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 16)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 28)

        # Identity mapping for every predicate argument
        # Binds pred_arg_2, pred_arg_3, with pred_arg_5 together on pat_var_a, and pred_arg_1 with pred_arg_4 on pat_var_b
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_b, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_a, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_a, False, False)
        varmap_4 = VariableMapCallable(lambda x: x, pred_arg_4, pat_var_b, False, False)
        varmap_5 = VariableMapCallable(lambda x: x, pred_arg_5, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        # Outward side predicate arguments
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3, pred_arg_4, pred_arg_5)
        # Inward side pattern variables - Change its order so that it is different from the varmap order.
        inward_vn.rel_vars = (pat_var_b, pat_var_a)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = outward_vn, inward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3, varmap_4, varmap_5])

        # Compute
        rmfn.inward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # Check shape
        expected_msg_shape = (Size([28, 16]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check content by checking message parameter and weight

        # The outgoing message slices should be the diagonals of the incoming messages if there are bound predicate
        #   arguments.
        # If the outgoing message slice happens to have some dimensional index that are not mapped to any of its
        #   corresponding predicate argument's index, then this slice should be an identity slice.
        for i in range(pat_var_a.size):
            for j in range(pat_var_b.size):
                actual_out_msg_param_slice = out_msg.parameter[j, i]  # The pattern variables order are inverted
                actual_out_msg_weight_slice = out_msg.weight[j, i]

                if 0 <= j < pred_arg_1.size and \
                        0 <= i < pred_arg_2.size and \
                        0 <= i < pred_arg_3.size and \
                        0 <= j < pred_arg_4.size and \
                        0 <= i < pred_arg_5.size:
                    expt_out_msg_param_slice = in_msg.parameter[j, i, i, j, i]
                    expt_out_msg_weight_slice = in_msg.weight[j, i, i, j, i]

                    assert_equal_within_error(actual_out_msg_param_slice, expt_out_msg_param_slice)
                    assert_equal_within_error(actual_out_msg_weight_slice, expt_out_msg_weight_slice)

                else:
                    assert_equal_within_error(actual_out_msg_param_slice, torch.tensor([0.]))
                    assert_proportional_within_error(actual_out_msg_weight_slice, torch.tensor([1.]), dims=[-1])

    def test_outward_compute_no_binding_id_map_same_order(self):
        # Test return message correctness without pattern variable binding, with identity variable map, and need not
        #   to permute batch dimensions of the message.
        # Incoming Message shape
        in_msg_shape = (Size([5, 10, 25]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)

        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 10)
        pat_var_c = Variable("pat_var_c", VariableMetatype.Relational, 25)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 5)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 10)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 25)

        # Mappings
        # All pairs use identity value map
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_b, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_c, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3)
        inward_vn.rel_vars = (pat_var_a, pat_var_b, pat_var_c)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg = out_ld.write.call_args[0][0]

        # In this test, the out_msg should exactly equal to the in_msg
        assert out_msg == in_msg

    def test_outward_compute_no_binding_id_map_diff_order(self):
        # Test return message correctness without pattern variable binding, with identity variable map, and need not
        #   to permute batch dimensions of the message.
        # Incoming Message shape
        # Permuted batch size due to pattern variables having a different order.
        in_msg_shape = (Size([5, 25, 10]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)

        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 10)
        pat_var_c = Variable("pat_var_c", VariableMetatype.Relational, 25)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 5)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 10)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 25)

        # Mappings
        # All pairs use identity value map
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_b, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_c, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3)
        # Now the inward VN's pattern variables are not in the same order as the outward VN's predicate arguments
        inward_vn.rel_vars = (pat_var_a, pat_var_c, pat_var_b)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg = out_ld.write.call_args[0][0]

        # Check that the in_msg content are the same as the out_msg content when permuted to the right order
        assert out_msg == in_msg.batch_permute([0, 2, 1])

    def test_outward_compute_no_binding_non_id_map_single_1(self):
        # Test return message correctness without pattern variable binding, with identity variable map, and need not
        #   to permute batch dimensions of the message.
        # Type 1: restrictive map with same message slice order as input message
        # Incoming Message shape
        in_msg_shape = (Size([5]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)

        # A restrictive mapping: only values in range [3, 7] of the predicate argument is selected
        def val_map(x):
            return x + 3  # [0, 1, 2, 3, 4] becomes [3, 4, 5, 6, 7]

        # variable map
        varmap_1 = VariableMapCallable(val_map, pred_arg_1, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1,)
        inward_vn.rel_vars = (pat_var_a,)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # In this test, a specific chunk of out_msg at the correct index shall equal to in_msg, while slices of out_msg
        #   not in the chunk shall equal to the identity slice.
        expected_msg_shape = (Size([10]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check content by checking message parameter and weight
        slice_ids = torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)
        # Check chunk
        assert_equal_within_error(out_msg.parameter.index_select(0, slice_ids), in_msg.parameter)
        assert_equal_within_error(out_msg.weight.index_select(0, slice_ids), in_msg.weight)
        # Check other slices not in chunk
        for i in range(pred_arg_1.size):
            if i not in slice_ids.tolist():
                assert_equal_within_error(out_msg.parameter[i], torch.tensor([0.]))
                assert_proportional_within_error(out_msg.weight[i], torch.tensor([1.]), dims=[-1])

    def test_outward_compute_no_binding_non_id_map_single_2(self):
        # Test return message correctness without pattern variable binding, with identity variable map, and need not
        #   to permute batch dimensions of the message.
        # Type 2: restrictive map with arbitrary index order
        # Incoming Message shape
        in_msg_shape = (Size([5]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)

        # A restrictive mapping: only values in range [3, 7] of the predicate argument is selected
        slice_ids = torch.tensor([2, 5, 4, 0, 3], dtype=torch.long)

        def val_map(x):
            return slice_ids

        # variable map
        varmap_1 = VariableMapCallable(val_map, pred_arg_1, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1,)
        inward_vn.rel_vars = (pat_var_a,)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # In this test, slices of out_msg at the correct index shall equal to the slice of in_msg at the corresponding
        #   pre-map index, while other slices of out_msg shall equal to the identity slice.
        expected_msg_shape = (Size([10]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check slices
        for pat_var_id, pred_arg_id in enumerate(slice_ids.tolist()):
            assert_equal_within_error(out_msg.parameter[pred_arg_id], in_msg.parameter[pat_var_id])
            assert_equal_within_error(out_msg.weight[pred_arg_id], in_msg.weight[pat_var_id])
        # Check other slices not in chunk
        for i in range(pred_arg_1.size):
            if i not in slice_ids.tolist():
                assert_equal_within_error(out_msg.parameter[i], torch.tensor([0.]))
                assert_proportional_within_error(out_msg.weight[i], torch.tensor([1.]), dims=[-1])

    def test_outward_compute_no_binding_non_id_map_single_3(self):
        # Test return message correctness without pattern variable binding, with identity variable map, and need not
        #   to permute batch dimensions of the message.
        # Type 2: expansive map with arbitrary index order
        # Incoming Message shape
        in_msg_shape = (Size([5]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)

        # A restrictive mapping: only values in range [3, 7] of the predicate argument is selected
        slice_ids = torch.tensor([-1, 2, 11, 4, 3], dtype=torch.long)

        def val_map(x):
            return slice_ids

        # variable map
        varmap_1 = VariableMapCallable(val_map, pred_arg_1, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1,)
        inward_vn.rel_vars = (pat_var_a,)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # In this test, slices of out_msg at the correct index shall equal to the slice of in_msg at the corresponding
        #   pre-map index when the pre-map index is not out of bound, while other slices of out_msg shall equal to the
        #   identity slice.
        expected_msg_shape = (Size([10]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check slices
        for pred_arg_id in range(pred_arg_1.size):
            if pred_arg_id in slice_ids.tolist():
                pat_var_id = slice_ids.tolist().index(pred_arg_id)
                assert_equal_within_error(out_msg.parameter[pred_arg_id], in_msg.parameter[pat_var_id])
                assert_equal_within_error(out_msg.weight[pred_arg_id], in_msg.weight[pat_var_id])
            else:
                assert_equal_within_error(out_msg.parameter[pred_arg_id], torch.tensor([0.]))
                assert_proportional_within_error(out_msg.weight[pred_arg_id], torch.tensor([1.]), dims=[-1])

    def test_outward_compute_no_binding_non_id_map_multiple(self):
        # Test return message correctness
        #   - without binding
        #   - arbitrary mapping with arbitrary index ordering
        #   - multiple variables
        # Incoming Message shape
        in_msg_shape = (Size([5, 10, 15]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 10)
        pat_var_c = Variable("pat_var_c", VariableMetatype.Relational, 15)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 10)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 10)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 10)

        # A restrictive mapping: only values in range [3, 7] of the predicate argument is selected
        slice_ids_1 = torch.tensor([-1, 2, 11, 4, 3], dtype=torch.long)

        def val_map_1(x):
            return slice_ids_1

        def val_map_2(x):
            return x + 3

        slice_ids_2 = val_map_2(torch.arange(10))

        def val_map_3(x):
            return x - 5

        slice_ids_3 = val_map_3(torch.arange(15))

        # variable map
        varmap_1 = VariableMapCallable(val_map_1, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(val_map_2, pred_arg_2, pat_var_b, False, False)
        varmap_3 = VariableMapCallable(val_map_3, pred_arg_3, pat_var_c, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3)
        inward_vn.rel_vars = (pat_var_a, pat_var_b, pat_var_c)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # In this test, Only slices in out_msg whose indices are all mapped and within bounds come from slices from
        #   in_msg. Other slices shall be the identity slices.
        expected_msg_shape = (Size([10, 10, 10]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check slices
        for pred_arg_1_id in range(pred_arg_1.size):
            for pred_arg_2_id in range(pred_arg_2.size):
                for pred_arg_3_id in range(pred_arg_3.size):
                    if pred_arg_1_id in slice_ids_1.tolist() and \
                            pred_arg_2_id in slice_ids_2.tolist() and \
                            pred_arg_3_id in slice_ids_3.tolist():
                        pat_var_a_id, pat_var_b_id, pat_var_c_id = slice_ids_1.tolist().index(pred_arg_1_id), \
                                                                   slice_ids_2.tolist().index(pred_arg_2_id), \
                                                                   slice_ids_3.tolist().index(pred_arg_3_id)
                        assert_equal_within_error(out_msg.parameter[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id],
                                                  in_msg.parameter[pat_var_a_id, pat_var_b_id, pat_var_c_id])
                        assert_equal_within_error(out_msg.weight[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id],
                                                  in_msg.weight[pat_var_a_id, pat_var_b_id, pat_var_c_id])
                    else:
                        assert_equal_within_error(out_msg.parameter[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id],
                                                  torch.tensor([0.]))
                        assert_proportional_within_error(out_msg.weight[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id],
                                                         torch.tensor(1.), dims=[-1])

    def test_outward_compute_single_binding_two_args_id_map(self):
        # Test return message correctness
        #   - A single binding that binds two predicate arguments
        #   - identity mapping
        # Incoming Message shape
        in_msg_shape = (Size([5]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 6)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 8)

        # variable map with identity value maps
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2)
        inward_vn.rel_vars = (pat_var_a,)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # Check shape
        expected_msg_shape = (Size([6, 8]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check slices
        # In this test, the in_msg slices shall be put onto the diagonal of the out_msg and without exceeding bounds.
        #   All other slices of out_msg shall be the identity slice
        for pred_arg_a_id in range(pred_arg_1.size):
            for pred_arg_b_id in range(pred_arg_2.size):
                if pred_arg_a_id == pred_arg_b_id and 0 <= pred_arg_a_id < pat_var_a.size:
                    assert_equal_within_error(out_msg.parameter[pred_arg_a_id, pred_arg_b_id],
                                              in_msg.parameter[pred_arg_a_id])
                    assert_equal_within_error(out_msg.weight[pred_arg_a_id, pred_arg_b_id],
                                              in_msg.weight[pred_arg_a_id])
                else:
                    assert_equal_within_error(out_msg.parameter[pred_arg_a_id, pred_arg_b_id], torch.tensor([0.]))
                    assert_proportional_within_error(out_msg.weight[pred_arg_a_id, pred_arg_b_id], torch.tensor([1.]),
                                                     dims=[-1])

    def test_outward_compute_single_binding_two_args_non_id_map(self):
        # Test return message correctness
        #   - A single binding that binds two predicate arguments
        #   - Arbitrary mapping
        # Incoming Message shape
        in_msg_shape = (Size([5]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 6)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 8)

        # Arbitrary maps
        slice_ids_1 = torch.tensor([-1, 2, 6, 3, 0], dtype=torch.long)
        slice_ids_2 = torch.tensor([6, 5, 4, 3, 2], dtype=torch.long)

        def valmap_1(x):
            return slice_ids_1

        def valmap_2(x):
            return slice_ids_2

        # variable map
        varmap_1 = VariableMapCallable(valmap_1, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(valmap_2, pred_arg_2, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2)
        inward_vn.rel_vars = (pat_var_a,)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # Check shape
        expected_msg_shape = (Size([6, 8]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check slices
        # In this test, the in_msg slices shall be put onto the corresponding index of out_msg if and only if the index
        #   is valid for out_msg
        paired_mapped_ids = list(zip(slice_ids_1.tolist(), slice_ids_2.tolist()))
        for pred_arg_a_id in range(pred_arg_1.size):
            for pred_arg_b_id in range(pred_arg_2.size):
                if (pred_arg_a_id, pred_arg_b_id) in paired_mapped_ids:
                    pat_var_id = paired_mapped_ids.index((pred_arg_a_id, pred_arg_b_id))
                    assert_equal_within_error(out_msg.parameter[pred_arg_a_id, pred_arg_b_id],
                                              in_msg.parameter[pat_var_id])
                    assert_equal_within_error(out_msg.weight[pred_arg_a_id, pred_arg_b_id],
                                              in_msg.weight[pat_var_id])
                else:
                    assert_equal_within_error(out_msg.parameter[pred_arg_a_id, pred_arg_b_id], torch.tensor([0.]))
                    assert_proportional_within_error(out_msg.weight[pred_arg_a_id, pred_arg_b_id], torch.tensor([1.]),
                                                     dims=[-1])

    def test_outward_compute_single_binding_multiple_args_id_map(self):
        # Test return message correctness
        #   - A single binding that binds more than two predicate arguments
        #   - Identity mapping
        # Incoming Message shape
        in_msg_shape = (Size([5, 10]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 10)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 6)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 8)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 10)
        pred_arg_4 = Variable("pred_arg_4", VariableMetatype.Relational, 12)

        # variable map with identity value maps
        # Bind pred_arg_1, pred_arg_3, pred_arg_4 to pat_var_a, and pred_arg_2 to pat_var_b
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_b, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_a, False, False)
        varmap_4 = VariableMapCallable(lambda x: x, pred_arg_4, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3, pred_arg_4)
        inward_vn.rel_vars = (pat_var_a, pat_var_b)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3, varmap_4])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # Check shape
        expected_msg_shape = (Size([6, 8, 10, 12]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check slices
        for pred_arg_1_id in range(pred_arg_1.size):
            for pred_arg_2_id in range(pred_arg_2.size):
                for pred_arg_3_id in range(pred_arg_3.size):
                    for pred_arg_4_id in range(pred_arg_4.size):
                        if pred_arg_1_id == pred_arg_3_id == pred_arg_4_id and 0 <= pred_arg_1_id < pat_var_a.size and \
                                0 <= pred_arg_2_id < pat_var_b.size:
                            assert_equal_within_error(out_msg.parameter[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                        pred_arg_4_id],
                                                      in_msg.parameter[pred_arg_1_id, pred_arg_2_id])
                            assert_equal_within_error(out_msg.weight[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                     pred_arg_4_id],
                                                      in_msg.weight[pred_arg_1_id, pred_arg_2_id])
                        else:
                            assert_equal_within_error(out_msg.parameter[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                        pred_arg_4_id],
                                                      torch.tensor([0.]))
                            assert_proportional_within_error(out_msg.weight[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                            pred_arg_4_id],
                                                             torch.tensor([1.]), dims=[-1])

    def test_outward_compute_multiple_binding_multiple_args_id_map_1(self):
        # Test return message correctness
        #   - Multiple bindings that bind more than two predicate arguments
        #   - Identity mapping
        # Incoming Message shape
        in_msg_shape = (Size([5, 10]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 10)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 6)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 8)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 10)
        pred_arg_4 = Variable("pred_arg_4", VariableMetatype.Relational, 12)
        pred_arg_5 = Variable("pred_arg_5", VariableMetatype.Relational, 14)

        # variable map with identity value maps
        # Bind pred_arg_1, pred_arg_3, pred_arg_4 to pat_var_a, and pred_arg_2 abd pred_arg_5 to pat_var_b
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_a, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_b, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_a, False, False)
        varmap_4 = VariableMapCallable(lambda x: x, pred_arg_4, pat_var_a, False, False)
        varmap_5 = VariableMapCallable(lambda x: x, pred_arg_5, pat_var_b, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3, pred_arg_4, pred_arg_5)
        inward_vn.rel_vars = (pat_var_a, pat_var_b)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3, varmap_4, varmap_5])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # Check shape
        expected_msg_shape = (Size([6, 8, 10, 12, 14]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check slices
        for pred_arg_1_id in range(pred_arg_1.size):
            for pred_arg_2_id in range(pred_arg_2.size):
                for pred_arg_3_id in range(pred_arg_3.size):
                    for pred_arg_4_id in range(pred_arg_4.size):
                        for pred_arg_5_id in range(pred_arg_5.size):
                            if pred_arg_1_id == pred_arg_3_id == pred_arg_4_id and 0 <= pred_arg_1_id < pat_var_a.size \
                                    and pred_arg_2_id == pred_arg_5_id and 0 <= pred_arg_2_id < pat_var_b.size:
                                assert_equal_within_error(out_msg.parameter[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                            pred_arg_4_id, pred_arg_5_id],
                                                          in_msg.parameter[pred_arg_1_id, pred_arg_2_id])
                                assert_equal_within_error(out_msg.weight[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                         pred_arg_4_id, pred_arg_5_id],
                                                          in_msg.weight[pred_arg_1_id, pred_arg_2_id])
                            else:
                                assert_equal_within_error(out_msg.parameter[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                            pred_arg_4_id, pred_arg_5_id],
                                                          torch.tensor([0.]))
                                assert_proportional_within_error(out_msg.weight[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                                pred_arg_4_id, pred_arg_5_id],
                                                                 torch.tensor([1.]), dims=[-1])

    def test_outward_compute_multiple_binding_multiple_args_id_map_2(self):
        # Test return message correctness
        #   - Multiple bindings that bind more than two predicate arguments
        #   - Identity mapping
        #   - An alternative pattern variable and predicate argument mapping order
        # Incoming Message shape
        in_msg_shape = (Size([5, 10]), Size([3]), Size([10]), Size([2]))
        # Random incoming message
        in_msg = random_message(MessageType.Dual, *in_msg_shape)
        # Test pattern variable
        pat_var_a = Variable("pat_var_a", VariableMetatype.Relational, 5)
        pat_var_b = Variable("pat_var_b", VariableMetatype.Relational, 10)
        # Test predicate arguments
        pred_arg_1 = Variable("pred_arg_1", VariableMetatype.Relational, 6)
        pred_arg_2 = Variable("pred_arg_2", VariableMetatype.Relational, 8)
        pred_arg_3 = Variable("pred_arg_3", VariableMetatype.Relational, 10)
        pred_arg_4 = Variable("pred_arg_4", VariableMetatype.Relational, 12)
        pred_arg_5 = Variable("pred_arg_5", VariableMetatype.Relational, 14)

        # variable map with identity value maps
        # Bind pred_arg_1, pred_arg_3, pred_arg_4 to pat_var_b, and pred_arg_2 abd pred_arg_5 to pat_var_a
        varmap_1 = VariableMapCallable(lambda x: x, pred_arg_1, pat_var_b, False, False)
        varmap_2 = VariableMapCallable(lambda x: x, pred_arg_2, pat_var_a, False, False)
        varmap_3 = VariableMapCallable(lambda x: x, pred_arg_3, pat_var_b, False, False)
        varmap_4 = VariableMapCallable(lambda x: x, pred_arg_4, pat_var_b, False, False)
        varmap_5 = VariableMapCallable(lambda x: x, pred_arg_5, pat_var_a, False, False)

        # Mock linkdata and variable node
        outward_vn, inward_vn = MagicMock(spec=DVN), MagicMock(spec=DVN)
        outward_vn.rel_vars = (pred_arg_1, pred_arg_2, pred_arg_3, pred_arg_4, pred_arg_5)
        inward_vn.rel_vars = (pat_var_a, pat_var_b)
        in_ld, out_ld = MagicMock(spec=LinkData), MagicMock(spec=LinkData)
        in_ld.vn, out_ld.vn = inward_vn, outward_vn
        in_ld.read.return_value = in_msg  # Stub read() method and its return value

        # RMFN
        rmfn = RMFN("test_rmfn", [varmap_1, varmap_2, varmap_3, varmap_4, varmap_5])

        # Compute
        rmfn.outward_compute(in_ld, out_ld)

        # Check that out_ld.write() has been called and then check the return message
        out_ld.write.assert_called_once()
        out_msg: Message = out_ld.write.call_args[0][0]

        # Check shape
        expected_msg_shape = (Size([6, 8, 10, 12, 14]), Size([3]), Size([10]), Size([2]))
        assert out_msg.shape == expected_msg_shape
        # Check slices
        for pred_arg_1_id in range(pred_arg_1.size):
            for pred_arg_2_id in range(pred_arg_2.size):
                for pred_arg_3_id in range(pred_arg_3.size):
                    for pred_arg_4_id in range(pred_arg_4.size):
                        for pred_arg_5_id in range(pred_arg_5.size):
                            if pred_arg_1_id == pred_arg_3_id == pred_arg_4_id and 0 <= pred_arg_1_id < pat_var_b.size \
                                    and pred_arg_2_id == pred_arg_5_id and 0 <= pred_arg_2_id < pat_var_a.size:
                                assert_equal_within_error(out_msg.parameter[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                            pred_arg_4_id, pred_arg_5_id],
                                                          in_msg.parameter[pred_arg_2_id, pred_arg_1_id])
                                assert_equal_within_error(out_msg.weight[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                         pred_arg_4_id, pred_arg_5_id],
                                                          in_msg.weight[pred_arg_2_id, pred_arg_1_id])
                            else:
                                assert_equal_within_error(out_msg.parameter[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                            pred_arg_4_id, pred_arg_5_id],
                                                          torch.tensor([0.]))
                                assert_proportional_within_error(out_msg.weight[pred_arg_1_id, pred_arg_2_id, pred_arg_3_id,
                                                                                pred_arg_4_id, pred_arg_5_id],
                                                                 torch.tensor([1.]), dims=[-1])
