"""
    Unit tests for EventTransform structure
"""
import pytest
import torch
from torch.distributions.transforms import ExpTransform
import torch.distributions.constraints as C

from pysigma.defs import Variable, VariableMetatype
from pysigma.pattern_structures.event_transform import EventTransform


class TestEventTransform:

    def test_init_1(self):
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))

        trans = EventTransform(pred_arg, pat_var)

        assert trans.pred_arg is pred_arg
        assert trans.pat_var is pat_var
        assert trans.transform is None
        assert trans.forward is True
        assert trans.master is False

    def test_init_2(self):
        pred_arg = [
            Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,)),
            Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,)),
            Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))
        transform = torch.distributions.transforms.ExpTransform()

        trans = EventTransform(pred_arg, pat_var, transform, False, True)

        for arg1, arg2 in zip(trans.pred_arg, pred_arg):
            assert arg1 is arg2
        assert trans.pat_var is pat_var
        assert trans.transform is transform
        assert trans.forward is False
        assert trans.master is True

    def test_finalized_true_1(self):
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))

        trans = EventTransform(pred_arg, pat_var)

        assert trans.finalized

    def test_finalized_true_2(self):
        pred_arg = [
            Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,)),
            Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,)),
            Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))

        trans = EventTransform(pred_arg, pat_var)

        assert trans.finalized

    def test_finalized_false_1(self):
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = 'pat_var_1'

        trans = EventTransform(pred_arg, pat_var)

        assert not trans.finalized

    def test_finalized_false_2(self):
        pred_arg = [
            Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,)),
            'pred_arg_2',
            Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))

        trans = EventTransform(pred_arg, pat_var)

        assert not trans.finalized

    def test_forward_trans_backward_trans_property_1(self):
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))
        t = ExpTransform()

        trans = EventTransform(pred_arg, pat_var, t, forward=True)

        assert trans.forward_trans == t
        assert trans.backward_trans == t.inv

    def test_forward_trans_backward_trans_property_2(self):
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))
        t = ExpTransform()

        trans = EventTransform(pred_arg, pat_var, t, forward=False)

        assert trans.forward_trans == t.inv
        assert trans.backward_trans == t

    def test_repr_1(self):
        # Test string representation
        #   - single predicate argument
        #   - no transformation
        #   - not master
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))

        trans = EventTransform(pred_arg, pat_var)

        assert repr(trans) == "pred_arg_1 --> pat_var_1, identity"

    def test_repr_2(self):
        # Test string representation
        #   - single predicate argument
        #   - no transformation
        #   - master
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))

        trans = EventTransform(pred_arg, pat_var, master=True)

        assert repr(trans) == "pred_arg_1 --> pat_var_1, identity, master"

    def test_repr_3(self):
        # Test string representation
        #   - single predicate argument
        #   - with forward transformation
        #   - master
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))
        t = ExpTransform()

        trans = EventTransform(pred_arg, pat_var, t, master=True)

        assert repr(trans) == "pred_arg_1 --> pat_var_1, forward ExpTransform(), master"

    def test_repr_4(self):
        # Test string representation
        #   - single predicate argument
        #   - with backward transformation
        #   - master
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))
        t = ExpTransform()

        trans = EventTransform(pred_arg, pat_var, t, forward=False, master=True)

        assert repr(trans) == "pred_arg_1 --> pat_var_1, backward ExpTransform(), master"

    def test_repr_5(self):
        # Test string representation
        #   - single predicate argument
        #   - with backward transformation
        #   - master
        pred_arg_1 = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pred_arg_2 = Variable('pred_arg_2', VariableMetatype.Random, 5, (C.real,))
        pred_arg_3 = Variable('pred_arg_3', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))
        t = ExpTransform()

        trans = EventTransform([pred_arg_1, pred_arg_2, pred_arg_3], pat_var, t, forward=False, master=True)

        assert repr(trans) == "(pred_arg_1, pred_arg_2, pred_arg_3) --> pat_var_1, backward ExpTransform(), master"
