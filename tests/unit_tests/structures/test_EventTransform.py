"""
    Unit tests for EventTransform structure
"""
import pytest
from typing import Union, Optional, Tuple, Callable
import torch
from torch.distributions import Distribution as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import MessageType, Message, Variable, VariableMetatype
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

    def test_init_2(self):
        pred_arg = [
            Variable('pred_arg_1', VariableMetatype.Random, 1, (C.real,)),
            Variable('pred_arg_2', VariableMetatype.Random, 2, (C.real,)),
            Variable('pred_arg_3', VariableMetatype.Random, 3, (C.real,)),
        ]
        pat_var = Variable('pat_var_1', VariableMetatype.Random, 5, (C.real,))

        trans = EventTransform(pred_arg, pat_var)

        for arg1, arg2 in zip(trans.pred_arg, pred_arg):
            assert arg1 is arg2
        assert trans.pat_var is pat_var
        assert trans.transform is None
        assert trans.forward is True

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
