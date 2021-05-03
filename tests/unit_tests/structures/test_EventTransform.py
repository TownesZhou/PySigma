"""
    Unit tests for EventTransform structure
"""
import pytest
import torch
from torch.distributions.transforms import ExpTransform, AffineTransform
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
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 6, (C.real,))
        transform = torch.distributions.transforms.ExpTransform()

        trans = EventTransform(pred_arg, pat_var, transform, False, True)

        for arg1, arg2 in zip(trans.pred_arg, pred_arg):
            assert arg1 is arg2
        assert trans.pat_var is pat_var
        assert trans.transform is transform
        assert trans.forward is False
        assert trans.master is True

    def test_init_not_finalized_1(self):
        # Test init with not finalized arguments
        #   - Single predicate argument
        pred_arg = 'a'
        pat_var = 'b'

        pattern = EventTransform(pred_arg, pat_var)

        assert pattern.pred_arg == pred_arg
        assert pattern.pat_var == pat_var
        assert not pattern.finalized

    def test_init_not_finalized_2(self):
        # Test init with not finalized arguments
        #   - Multiple predicate argument
        pred_arg = ('a', 'b', 'c')
        pat_var = 'x'

        pattern = EventTransform(pred_arg, pat_var)

        assert pattern.pred_arg == pred_arg
        assert pattern.pat_var == pat_var
        assert not pattern.finalized

    def test_init_wrong_var_type_single_pred_arg(self):
        # Test AssertionError is raised if init with variable with wrong type.
        #   - Test with single pred arg
        pred_arg = Variable('pred_arg_1', VariableMetatype.Relational, 5)
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))

        with pytest.raises(AssertionError) as excinfo:
            pattern = EventTransform(pred_arg, pat_var)

        assert str(excinfo.value) == "EventTransform structure only accepts random variables. Found `pred_arg` " \
                                     "variable metatype: VariableMetatype.Relational"

    def test_init_wrong_var_type_multiple_pred_arg(self):
        # Test AssertionError is raised if init with variable with wrong type.
        #   - Test with multiple pred arg
        #   - One of the pred arg has wrong type
        pred_arg = [
            'pred_arg_1',
            Variable('pred_arg_2', VariableMetatype.Relational, 2),
            Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))

        with pytest.raises(AssertionError) as excinfo:
            pattern = EventTransform(pred_arg, pat_var)

        assert str(excinfo.value) == "EventTransform structure only accepts random variables. Found one of the " \
                                     "variables in `pred_arg` has variable metatype: VariableMetatype.Relational"

    def test_init_wrong_var_type_pat_var(self):
        # Test AssertionError is raised if init with variable with wrong type.
        #   - Test with pat_var
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Relational, 5)

        with pytest.raises(AssertionError) as excinfo:
            pattern = EventTransform(pred_arg, pat_var)

        assert str(excinfo.value) == "EventTransform structure only accepts random variables. Found `pat_var` " \
                                     "variable metatype: VariableMetatype.Relational"

    def test_init_wrong_size_single_pred_arg(self):
        # Test AssertionError is raised if init with finalized arguments but variable sizes do not match
        #   - Test with single pred arg
        pred_arg = Variable('pred_arg_1', VariableMetatype.Random, 5, (C.real,))
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 6, (C.real,))

        with pytest.raises(AssertionError) as excinfo:
            pattern = EventTransform(pred_arg, pat_var)

        assert str(excinfo.value) == "The variable size of `pred_arg` and `pat_var` must be the same. Found " \
                                     "`pred_arg` has size 5 but `pat_var` has size 6."

    def test_init_wrong_size_multiple_pred_arg(self):
        # Test AssertionError is raised if init with finalized arguments but variable sizes do not match
        #   - Test with multiple pred arg
        pred_arg = [
            Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,)),
            Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,)),
            Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 7, (C.real,))

        with pytest.raises(AssertionError) as excinfo:
            pattern = EventTransform(pred_arg, pat_var)

        assert str(excinfo.value) == "The variable size of `pred_arg` and `pat_var` must be the same. Found " \
                                     "multiple predicate arguments in `pred_arg` have a total size of 6, but " \
                                     "`pat_var` has size 7."

    def test_init_diff_argument_constraints_1(self):
        # Test AssertionError is raised if multiple predicate arguments are declared but they do not have the same
        # value constraints
        pred_arg = [
            Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,)),
            Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real, C.interval(2, 3))),
            Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real, C.interval(2, 3))),
        ]
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 6, (C.real,))

        with pytest.raises(AssertionError) as excinfo:
            pattern = EventTransform(pred_arg, pat_var)

        assert str(excinfo.value) == "If multiple predicate arguments declared in a pattern, all of them must have " \
                                     "the same value constraints. Found argument 'pred_arg_1' has constraints {}, but "\
                                     "argument 'pred_arg_2' has constraints {}."\
                                     .format((C.real,), (C.real, C.interval(2, 3)))

    def test_init_diff_argument_constraints_2(self):
        # Test AssertionError is raised if multiple predicate arguments are declared but they do not have the same
        # value constraints
        pred_arg = [
            Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,)),
            Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,)),
            Variable('pred_arg_3', VariableMetatype.Random, 3, (C.interval(2, 3),)),
        ]
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 6, (C.real,))

        with pytest.raises(AssertionError) as excinfo:
            pattern = EventTransform(pred_arg, pat_var)

        assert str(excinfo.value) == "If multiple predicate arguments declared in a pattern, all of them must have " \
                                     "the same value constraints. Found argument 'pred_arg_1' has constraints {}, but "\
                                     "argument 'pred_arg_3' has constraints {}."\
                                     .format((C.real,), (C.interval(2, 3),))

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
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 6, (C.real,))

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

    def test_surrogate_pred_arg_not_finalized(self):
        # Test that AssertionError is raised if the pattern is not yet finalized
        pattern = EventTransform('a', 'b')

        assert not pattern.finalized

        with pytest.raises(AssertionError) as excinfo:
            srg_pred_arg = pattern.surrogate_pred_arg

        assert str(excinfo.value) == "The surrogate predicate argument of a EventTransform can only be generated " \
                                     "when the EventTransform is finalized."

    def test_surrogate_pred_arg_single_pred_arg(self):
        # Test surrogate_pred_arg value
        #   - single predicate argument
        pred_arg = Variable('test_pred_arg', VariableMetatype.Random, 3, (C.real,))
        pat_var = Variable('test_pat_var', VariableMetatype.Random, 3, (C.real,))
        trans = AffineTransform(5.6, 2)
        pattern = EventTransform(pred_arg, pat_var, trans)

        # In this case, the surrogate pred arg should just be the original predicate argument
        assert pattern.surrogate_pred_arg is pred_arg

    def test_surrogate_pred_arg_multiple_pred_arg(self):
        # Test surrogate_pred_arg value
        #   - multiple predicate arguments
        pred_arg_1 = Variable('test_pred_arg_1', VariableMetatype.Random, 3, (C.interval(0, 10),))
        pred_arg_2 = Variable('test_pred_arg_2', VariableMetatype.Random, 4, (C.interval(0, 10),))
        pred_arg_3 = Variable('test_pred_arg_3', VariableMetatype.Random, 5, (C.interval(0, 10),))
        pat_var = Variable('test_pat_var', VariableMetatype.Random, 12, (C.real,))
        trans = AffineTransform(5.6, 2)
        pattern = EventTransform([pred_arg_1, pred_arg_2, pred_arg_3], pat_var, trans)

        # Check surrogate pred arg attributes
        assert isinstance(pattern.surrogate_pred_arg, Variable)
        assert pattern.surrogate_pred_arg.name == 'test_pred_arg_1+test_pred_arg_2+test_pred_arg_3'
        assert pattern.surrogate_pred_arg.metatype == VariableMetatype.Random
        assert pattern.surrogate_pred_arg.size == 12
        assert pattern.surrogate_pred_arg.constraints == pred_arg_1.constraints

    def test_surrogate_pattern_single_pred_arg(self):
        # Test surrogate pattern value
        #   - Single predicate argument
        pred_arg = Variable('test_pred_arg', VariableMetatype.Random, 3, (C.real,))
        pat_var = Variable('test_pat_var', VariableMetatype.Random, 3, (C.real,))
        trans = AffineTransform(5.6, 2)
        pattern = EventTransform(pred_arg, pat_var, trans)

        srg_pattern = pattern.surrogate_pattern()

        assert srg_pattern.pred_arg == pattern.pred_arg
        assert srg_pattern.pat_var == pattern.pat_var
        assert srg_pattern.transform == pattern.transform
        assert srg_pattern.forward == pattern.forward
        assert srg_pattern.master == pattern.master

    def test_surrogate_pattern_multiple_pred_arg(self):
        # Test surrogate pattern value
        #   - Multiple predicate argument
        pred_arg_1 = Variable('test_pred_arg_1', VariableMetatype.Random, 3, (C.interval(0, 10),))
        pred_arg_2 = Variable('test_pred_arg_2', VariableMetatype.Random, 4, (C.interval(0, 10),))
        pred_arg_3 = Variable('test_pred_arg_3', VariableMetatype.Random, 5, (C.interval(0, 10),))
        pat_var = Variable('test_pat_var', VariableMetatype.Random, 12, (C.real,))
        trans = AffineTransform(5.6, 2)
        pattern = EventTransform([pred_arg_1, pred_arg_2, pred_arg_3], pat_var, trans, forward=False, master=True)

        srg_pattern = pattern.surrogate_pattern()

        assert srg_pattern.pred_arg == pattern.surrogate_pred_arg   # Here's the difference
        assert srg_pattern.pat_var == pattern.pat_var
        assert srg_pattern.transform == pattern.transform
        assert srg_pattern.forward == pattern.forward
        assert srg_pattern.master == pattern.master

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
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 15, (C.real,))
        t = ExpTransform()

        trans = EventTransform([pred_arg_1, pred_arg_2, pred_arg_3], pat_var, t, forward=False, master=True)

        assert repr(trans) == "(pred_arg_1, pred_arg_2, pred_arg_3) --> pat_var_1, backward ExpTransform(), master"
