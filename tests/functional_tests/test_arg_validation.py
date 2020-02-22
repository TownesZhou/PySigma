import pytest
from pysigma import Sigma
from pysigma.structures import *


class TestStructureArgValidation:
    """
        Testing Argument Validation of Cognitive Structures
    """

    def test_predicate_argument_1(self):
        """
            Test PredicateArgument with problematic arguments
        """
        with pytest.raises(ValueError):
            p = PredicateArgument(1, 0, 0)

        with pytest.raises(ValueError):
            p = PredicateArgument('test', 0, 0)

        with pytest.raises(ValueError):
            p = PredicateArgument('test', 'test_type', 0)

        with pytest.raises(ValueError):
            p = PredicateArgument('test', 'test_type', '@')


    def test_predicate_argument_2(self):
        """
            Test PredicateArgument with correct arguments
        """
        p = PredicateArgument('test', 'test_type', '!')

        p = PredicateArgument('test', 'test_type', '=')

        test_type = Type('test_type', 'discrete', 0, 3)
        p = PredicateArgument('test', test_type, '%')


    def test_predicate_pattern_1(self):
        """
            Test PredicatePattern with problematic arguments
        """
        with pytest.raises(ValueError):
            p = PredicatePattern(1, 0)

        with pytest.raises(ValueError):
            p = PredicatePattern('test', 0)

        with pytest.raises(ValueError):
            p = PredicatePattern('test', [0, 1, 2])

        with pytest.raises(ValueError):
            test_pattern_element_1 = PatternElement("arg_1")
            test_pattern_element_2 = PatternElement("arg_2")
            p = PredicatePattern('test', [test_pattern_element_1, test_pattern_element_2, "not a pattern element"])

        with pytest.raises(ValueError):
            test_pattern_element_1 = PatternElement("arg_1")
            test_pattern_element_2 = PatternElement("arg_2")
            p = PredicatePattern('test', [test_pattern_element_1, test_pattern_element_2], "invalid non-linearity")


    def test_predicate_pattern_2(self):
        """
            Test PredicatePattern with correct arguments
        """
        test_pattern_element_1 = PatternElement("arg_1")
        test_pattern_element_2 = PatternElement("arg_2")
        p = PredicatePattern('test', [test_pattern_element_1, test_pattern_element_2])

        p = PredicatePattern('test', [test_pattern_element_1, test_pattern_element_2], "sigmoid")

        p = PredicatePattern('test', [test_pattern_element_1, test_pattern_element_2], "^")


    def test_pattern_element_1(self):
        """
            Test PatternElement with problematic arguments
            Skipped Filter test
        """
        with pytest.raises(ValueError):
            p = PatternElement(1)

        with pytest.raises(ValueError):
            p = PatternElement("arg_test", 2.333)

        with pytest.raises(ValueError):
            p = PatternElement("arg_test", [True, True, False])

        with pytest.raises(ValueError):
            p = PatternElement("arg_test", [1, 2, 0.3])

        with pytest.raises(ValueError):
            p = PatternElement("arg_test", [1, 2, 'str'])


    def test_pattern_element_2(self):
        """
            Test PatternElement with correct arguments
            Skipped Filter test
        """
        p = PatternElement("arg_test")

        p = PatternElement("arg_test", 1)

        p = PatternElement("arg_test", "value_str")

        p = PatternElement("arg_test", [0, 1, 2, 3])

        p = PatternElement("arg_test", ["str_1", "str_2", "str_3"])

        test_pattern_variable = PatternVariable("test_pattern_variable")
        p = PatternElement("arg_test", test_pattern_variable)


    def test_pattern_variable_1(self):
        """
            Test PatternVariable with problematic arguments
            Skipped Filter test
        """
        with pytest.raises(ValueError):
            p = PatternVariable(1)

        with pytest.raises(ValueError):
            p = PatternVariable("_variable_name")

        with pytest.raises(ValueError):
            p = PatternVariable("test", "relation")

        with pytest.raises(ValueError):
            p = PatternVariable("test", [1, 2, 3, 4])

        with pytest.raises(ValueError):
            test_affine_1 = Affine()
            test_affine_2 = Affine()
            p = PatternVariable("test", [test_affine_1, test_affine_2])

    def test_pattern_variable_2(self):
        """
            Test PatternVariable with correct arguments
            Skipped Filter test
        """
        p = PatternVariable("test")

        p = PatternVariable("test", 1)

        test_affine = Affine()
        p = PatternVariable("test", test_affine)


    def test_affine_1(self):
        """
            Test Affine with problematic arguments
        """
        with pytest.raises(ValueError):
            a = Affine(1)

        with pytest.raises(ValueError):
            a = Affine("test", [1])

        with pytest.raises(ValueError):
            a = Affine("test", True)

        with pytest.raises(ValueError):
            a = Affine("test", "not int")

        with pytest.raises(ValueError):
            a = Affine("test", 1, "not int or float")

        with pytest.raises(ValueError):
            a = Affine("test", 1, 0.233, "not int")


    def test_affine_2(self):
        """
            Test Affine with correct arguments
        """
        a = Affine()

        a = Affine("test")

        a = Affine("test", 1)

        a = Affine("test", 1, 1)

        a = Affine("test", 1, 0.233)

        a = Affine("test", 1, 1, 1)


    def test_type_1(self):
        """
            Test Type with problematic arguments
        """
        t = Type(1, "discrete", 1, 3)

        t = Type("test", "not symbolic or discrete")

        t = Type("test", "discrete")

        t = Type("test", "symbolic")

        t = Type("test", "discrete", min='a', max='c')

        t = Type("test", "discrete", min=2, max=2)

        t = Type("test", "discrete", min=4, max=2)

        t = Type("test", "discrete", min=1.0, max=3.0)

        t = Type("test", "symbolic", symbol_list="not a list")

        t = Type("test", "symbolic", symbol_list=["check", "symbol", "type"])


    def test_type_2(self):
        """
            Test Type with correct arguments
        """
        t = Type("test", "discrete", 1, 5)

        t = Type("test", "discrete", 1, 2)

        # t = Type("test", "symbolic" [list of symbolic type])





