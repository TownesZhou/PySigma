import pytest
from pysigma.structures import *
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint


class TestTypeDataStructure:
    """
        Testing Data inside Types
    """
    c = Constraint()
    type_1 = Type("type1", False, 1)  # non-symbolic without constraint
    type_2 = Type("type2", False, 10, value_constraint=c)  # non-symbolic with constraint
    type_3 = Type("type3", True, symbol_list=["a", "b", "c"])  # symbolic without constraint
    type_4 = Type("type4", True, symbol_list=["1", "2", "3"], value_constraint=c)  # symbolic with constraint

    def test_name(self):
        assert self.type_1.name == "TYPE_[TYPE1]"
        assert self.type_3.name == "TYPE_[TYPE3]"

    def test_symbolic(self):
        assert self.type_1.symbolic == False
        assert self.type_3.symbolic == True

    def test_value_list(self):
        assert self.type_1.value_list == [0]
        assert self.type_2.value_list == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert self.type_3.value_list == ["a", "b", "c"]

    def test_size(self):
        assert self.type_1.size == 1
        assert self.type_2.size == 10
        assert self.type_3.size == 3

    def test_constraint(self):
        assert self.type_1.constraint is None
        assert self.type_2.constraint == self.c

    def test_value2axis(self):
        assert self.type_1.value2axis == {0: 0}
        assert self.type_3.value2axis == {"a": 0, "b": 1, "c": 2}

    def test_axis2value(self):
        assert self.type_1.axis2value == {0: 0}
        assert self.type_3.axis2value == {0: "a", 1: "b", 2: "c"}


class TestPredicateDataStructure:
    """
        Testing Data inside Types
    """
    c = Constraint()
    type_1 = Type("type1", False, 1)  # non-symbolic without constraint
    type_2 = Type("type2", False, 10, value_constraint=c)  # non-symbolic with constraint
    type_3 = Type("type3", True, symbol_list=["a", "b", "c"])  # symbolic without constraint
    type_4 = Type("type4", True, symbol_list=["1", "2", "3"], value_constraint=c)  # symbolic with constraint


    pred_1 = Predicate("pred1", [("arg_1", type_1)], [("arg_2", type_2)], "BP")
    pred_2 = Predicate("pred2", [("arg_1", type_1), ("arg_3", type_3)], [("arg_2", type_2)], "VMP")
    pred_3 = Predicate("pred3", [("arg_4", type_4), ("arg_3", type_3)], [("arg_1", type_2), ("arg_2", type_2)], "EP")

    def test_name(self):
        assert self.pred_1.name == "PRED_[PRED1]"
        assert self.pred_2.name == "PRED_[PRED2]"
        assert self.pred_3.name == "PRED_[PRED3]"

    def test_inf_mode(self):
        assert self.pred_1.inf_mode == "BP"
        assert self.pred_2.inf_mode == "VPM"
        assert self.pred_3.inf_mode == "EP"

    def test_event_dims(self):
        assert self.pred_1.event_dims == torch.Size([])

