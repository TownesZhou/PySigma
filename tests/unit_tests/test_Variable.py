"""
    Unit tests for Variable class
"""

import pytest
from torch.distributions.constraints import integer_interval, interval, real
from pysigma.defs import Variable, VariableMetatype


class TestVariableInit:

    def test_init_correct(self):
        # Test example correct init
        Variable("test", VariableMetatype.Relational, 10)
        Variable("test", VariableMetatype.Random, 10, [integer_interval(0, 3)])
        Variable("test", VariableMetatype.Random, 10, [integer_interval(0, 3), interval(0., 5.)])

    def test_init_wrong_data_type(self):
        # Test init with wrong datatype
        # The constraints field should be an iterable of constraints
        with pytest.raises(AssertionError):
            Variable("test", VariableMetatype.Relational, 10, integer_interval(0, 3))

    def test_init_conflicting_arguments_1(self):
        # Test init with conflicting arguments:
        #   - A relational variable but declares value constraint
        with pytest.raises(AssertionError):
            Variable("test", VariableMetatype.Relational, 10, [integer_interval(0, 3)])

    def test_init_conflicting_arguments_2(self):
        # Test init with conflicting arguments:
        #   - A random variable but no value constraints declared
        with pytest.raises(AssertionError):
            Variable("test", VariableMetatype.Random, 10)


class TestVariableEquality:

    def test_equal_without_constraints(self):
        # Test the equality of variables with no constraints declared
        test_var1 = Variable("test", VariableMetatype.Relational, 10)
        test_var2 = Variable("test", VariableMetatype.Relational, 10)
        assert test_var1 == test_var2

    def test_equal_with_constraints_same_instance(self):
        # Test the equality of variables with constraints
        #   - same constraint instances
        c = real
        test_var1 = Variable("test", VariableMetatype.Random, 10, (c,))
        test_var2 = Variable("test", VariableMetatype.Random, 10, (c,))
        assert test_var1 == test_var2

    def test_equal_with_constraints_diff_instance_1(self):
        # Test the equality of variables with constraints
        #   - same constraint instances
        c1, c2 = real, real
        test_var1 = Variable("test", VariableMetatype.Random, 10, (c1,))
        test_var2 = Variable("test", VariableMetatype.Random, 10, (c2,))
        assert test_var1 == test_var2

    def test_equal_with_constraints_diff_instance_2(self):
        # Test the equality of variables with constraints
        #   - same constraint instances
        c1, c2 = interval(0, 10), interval(0, 10)
        test_var1 = Variable("test", VariableMetatype.Random, 10, (c1,))
        test_var2 = Variable("test", VariableMetatype.Random, 10, (c2,))
        assert test_var1 == test_var2

    def test_equal_with_constraints_diff_instance_3(self):
        # Test the equality of variables with constraints
        #   - same constraint instances
        c1, c2 = integer_interval(0, 10), integer_interval(0, 10)
        test_var1 = Variable("test", VariableMetatype.Random, 10, (c1,))
        test_var2 = Variable("test", VariableMetatype.Random, 10, (c2,))
        assert test_var1 == test_var2

    def test_equal_with_constraints_multiple_same_order(self):
        # Test the equality of variables with multiple constraints
        #   - The order of the variables in the given tuple is the same
        c1_1, c2_1 = integer_interval(0, 10), interval(0, 10)
        c1_2, c2_2 = integer_interval(0, 10), interval(0, 10)
        test_var1 = Variable("test", VariableMetatype.Random, 10, (c1_1, c2_1))
        test_var2 = Variable("test", VariableMetatype.Random, 10, (c1_2, c2_2))
        assert test_var1 == test_var2

    def test_unequal_by_name(self):
        # Test that two variables are unequal because of different names
        c1, c2 = integer_interval(0, 3), interval(0., 5.)
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        test_var2 = Variable("test1", VariableMetatype.Random, 10, [c1, c2])
        assert test_var1 != test_var2

    def test_unequal_by_metatype(self):
        # Test that two variables are unequal because of different metatype
        c1, c2 = integer_interval(0, 3), interval(0., 5.)
        test_var1 = Variable("test", VariableMetatype.Relational, 10)
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        assert test_var1 != test_var2

    def test_unequal_by_size(self):
        # Test that two variables are unequal because of different sizes
        c1, c2 = integer_interval(0, 3), interval(0., 5.)
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        test_var2 = Variable("test", VariableMetatype.Random, 15, [c1, c2])
        assert test_var1 != test_var2

    def test_unequal_by_constraints_single(self):
        # Test that two variables are unequal because of different constraints
        #   - Single constraint
        c1, c2 = integer_interval(0, 3), interval(0., 5.)
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1])
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c2])
        assert test_var1 != test_var2

    def test_unequal_by_constraints_diff_num_1(self):
        # Test that two variables are unequal because of different constraints
        #   - Different numbers of constraints
        c1, c2 = integer_interval(0, 3), interval(0., 5.)
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1])
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        assert test_var1 != test_var2

    def test_unequal_by_constraints_multiple_1(self):
        # Test that two variables are unequal because of different constraints
        #   - Multiple constraints
        #   - All constraints same type but different attributes
        c1, c2, c3 = integer_interval(0, 3), integer_interval(0, 4), integer_interval(0, 5)
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1, c3])
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        assert test_var1 != test_var2

    def test_unequal_by_constraints_multiple_2(self):
        # Test that two variables are unequal because of different constraints
        #   - Multiple constraints
        #   - Constraints with different types
        c1, c2, c3 = integer_interval(0, 3), interval(0., 5.), real
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1, c3])
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        assert test_var1 != test_var2

    def test_unequal_by_constraints_multiple_3(self):
        # Test that two variables are unequal because of different constraints
        #   - Multiple constraints
        #   - All same constraints but different ordering
        c1, c2 = integer_interval(0, 3), interval(0., 5.)
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1, c2])
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c2, c1])
        assert test_var1 != test_var2


class TestVariableProperties:

    def test_str(self):
        test_var1 = Variable("test_var", VariableMetatype.Relational, 10)
        assert str(test_var1) == "test_var"

    def test_repr(self):
        test_var1 = Variable("test_var", VariableMetatype.Relational, 10)
        assert repr(test_var1) == "test_var"

    def test_hash_usable(self):
        # Test that a Variable instance can be hashed and used for example as keys to dictionaries
        test_var1 = Variable("test", VariableMetatype.Relational, 10)
        test_var2 = Variable("test", VariableMetatype.Random, 10, [integer_interval(0, 3)])
        test_dict = {test_var1: 1, test_var2: 2}

        assert test_dict[test_var1] == 1 and test_dict[test_var2] == 2

    def test_hash_diff_val_diff_constraint_attributes(self):
        # Test that the hash of two variables are different if they are same except of their constraints
        #   - constraints all have same type but different attributes
        c1, c2, c3 = interval(0., 3.), interval(0., 5.), real
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1, c3])
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c2, c3])
        assert test_var1.__hash__() != test_var2.__hash__()

    def test_hash_same_val_diff_constraint_instances(self):
        # Test that the hash of two variables are same if the constraints values are same but are different instances
        #   - constraints all have same type and attributes but are of different instances
        c1, c2 = interval(0., 3.), interval(0., 3.)
        test_var1 = Variable("test", VariableMetatype.Random, 10, [c1])
        test_var2 = Variable("test", VariableMetatype.Random, 10, [c2])
        assert test_var1.__hash__() == test_var2.__hash__()
