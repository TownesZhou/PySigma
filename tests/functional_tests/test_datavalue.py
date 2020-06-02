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

    def test_type_name(self):
        assert self.type_1.name == "TYPE_[TYPE1]"


