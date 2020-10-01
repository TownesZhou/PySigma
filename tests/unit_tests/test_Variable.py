"""
    Unit tests for Variable class
"""

import pytest
from torch.distributions.constraints import integer_interval, interval
from defs import Variable, VariableMetatype


class TestVariable:

    def test_init(self):
        # Test example normal init
        Variable("test", VariableMetatype.Relational, 10)
        Variable("test", VariableMetatype.Random, 10, [integer_interval(0, 3)])
        Variable("test", VariableMetatype.Random, 10, [integer_interval(0, 3), interval(0., 5.)])

        # Test typical wrong data type
        with pytest.raises(AssertionError):
            Variable("test", VariableMetatype.Relational, 10, integer_interval(0, 3))

        # Test conflict arguments
        with pytest.raises(AssertionError):
            Variable("test", VariableMetatype.Relational, 10, [integer_interval(0, 3)])

        with pytest.raises(AssertionError):
            Variable("test", VariableMetatype.Random, 10)

    def test_equal(self):
        c1, c2, c3 = integer_interval(0, 3), interval(0., 5.), interval(0., 10.)

        # Test equal
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        assert test_var1 == test_var2

        test_var1 = Variable("test", VariableMetatype.Relational, 10)
        test_var2 = Variable("test", VariableMetatype.Relational, 10)
        assert test_var1 == test_var2

        # Test unequal
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        test_var2 = Variable("test1", VariableMetatype.Random, 10, [c1, c2])
        assert test_var1 != test_var2

        test_var1 = Variable("test", VariableMetatype.Relational, 10)
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        assert test_var1 != test_var2

        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        test_var2 = Variable("test", VariableMetatype.Random, 20, [c1, c2])
        assert test_var1 != test_var2

        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1])
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        assert test_var1 != test_var2

        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1, c3])
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        assert test_var1 != test_var2

