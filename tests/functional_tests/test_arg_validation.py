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
            p = PredicatePattern('test', list(), "sigmoid")

        with pytest.raises(ValueError):
            p = PredicatePattern('test', "string", "sigmoid")

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
            p = PatternElement("arg_test", list())

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
        with pytest.raises(ValueError):
            t = Type(1, "discrete", 1, 3)

        with pytest.raises(ValueError):
            t = Type("test", "not symbolic or discrete")

        with pytest.raises(ValueError):
            t = Type("test", "discrete")

        with pytest.raises(ValueError):
            t = Type("test", "symbolic")

        with pytest.raises(ValueError):
            t = Type("test", "discrete", min='a', max='c')

        with pytest.raises(ValueError):
            t = Type("test", "discrete", min=2, max=2)

        with pytest.raises(ValueError):
            t = Type("test", "discrete", min=4, max=2)

        with pytest.raises(ValueError):
            t = Type("test", "discrete", min=1.0, max=3.0)

        with pytest.raises(ValueError):
            t = Type("test", "symbolic", symbol_list="not a list")

        with pytest.raises(ValueError):
            t = Type("test", "symbolic", symbol_list=[])

        with pytest.raises(ValueError):
            t = Type("test", "symbolic", symbol_list=["check", "symbol", "type"])



    def test_type_2(self):
        """
            Test Type with correct arguments
        """
        t = Type("test", "discrete", 1, 5)

        t = Type("test", "discrete", 1, 2)

        # t = Type("test", "symbolic" [list of symbolic type])


    def test_predicate_1(self):
        """
            Test Type with problematic arguments
        """
        test_predicate_arg_1 = PredicateArgument('test', 'test_type', '!')
        test_predicate_arg_2 = PredicateArgument('test', 'test_type', '=')
        predicate_list = [test_predicate_arg_1, test_predicate_arg_2]

        with pytest.raises(ValueError):
            p = Predicate(1, [1, 2, 3])

        with pytest.raises(ValueError):
            p = Predicate("test", list())

        with pytest.raises(ValueError):
            p = Predicate("test", [1, 2, 3, 4])

        with pytest.raises(ValueError):
            p = Predicate("test", predicate_list, world="semi-open")

        with pytest.raises(ValueError):
            p = Predicate("test", predicate_list, world="semi-open")

        with pytest.raises(ValueError):
            p = Predicate("test", predicate_list, exponential="string")

        with pytest.raises(ValueError):
            p = Predicate("test", predicate_list, normalize="string")

        with pytest.raises(ValueError):
            p = Predicate("test", predicate_list, function=False)


    def test_predicate_2(self):
        """
            Test Type with correct arguments
        """
        test_predicate_arg_1 = PredicateArgument('test', 'test_type', '!')
        test_predicate_arg_2 = PredicateArgument('test', 'test_type', '=')
        predicate_list = [test_predicate_arg_1, test_predicate_arg_2]

        p = Predicate("test", predicate_list)

        p = Predicate("test", predicate_list, world="closed")

        p = Predicate("test", predicate_list, exponential=True)

        p = Predicate("test", predicate_list, exponential=["str_1", "str_2"])

        p = Predicate("test", predicate_list, normalize=True)

        p = Predicate("test", predicate_list, normalize=["str_1", "str_2"])

        p = Predicate("test", predicate_list, function=1)

        p = Predicate("test", predicate_list, function=2.333)

        p = Predicate("test", predicate_list, function="test_string")

        p = Predicate("test", predicate_list, function=torch.tensor([[1., -1.], [1., -1.]]))

        p = Predicate("test", predicate_list,
                      exponential=["str_1", "str_2"],
                      normalize=["str_1", "str_2"],
                      perception=True,
                      function=torch.tensor([[1., -1.], [1., -1.]]))


    def test_conditional_1(self):
        """
            Test Type with problematic arguments
        """
        test_pattern_element_1 = PatternElement("arg_1")
        test_pattern_element_2 = PatternElement("arg_2")
        pattern_element_list = [test_pattern_element_1, test_pattern_element_2]
        test_predicate_pattern_1 = PredicatePattern('test_1', pattern_element_list)
        test_predicate_pattern_2 = PredicatePattern('test_2', pattern_element_list)
        predicate_pattern_list = [test_predicate_pattern_1, test_predicate_pattern_2]

        with pytest.raises(ValueError):
            c = Conditional(1)

        with pytest.raises(ValueError):
            c = Conditional("test")

        with pytest.raises(ValueError):
            c = Conditional("test", conditions=predicate_pattern_list)

        with pytest.raises(ValueError):
            c = Conditional("test",
                            conditions=predicate_pattern_list,
                            actions=predicate_pattern_list,
                            function_var_names=list())

        with pytest.raises(ValueError):
            c = Conditional("test",
                            conditions=predicate_pattern_list,
                            actions=predicate_pattern_list,
                            function_var_names="test_string")

        with pytest.raises(ValueError):
            c = Conditional("test",
                            conditions=predicate_pattern_list,
                            actions=predicate_pattern_list,
                            normal="test_string")

        with pytest.raises(ValueError):
            c = Conditional("test",
                            conditions=predicate_pattern_list,
                            actions=predicate_pattern_list,
                            function=list())


    def test_conditional_2(self):
        """
            Test Type with correct arguments
        """
        test_pattern_element_1 = PatternElement("arg_1")
        test_pattern_element_2 = PatternElement("arg_2")
        pattern_element_list = [test_pattern_element_1, test_pattern_element_2]
        test_predicate_pattern_1 = PredicatePattern('test_1', pattern_element_list)
        test_predicate_pattern_2 = PredicatePattern('test_2', pattern_element_list)
        predicate_pattern_list = [test_predicate_pattern_1, test_predicate_pattern_2]

        c = Conditional("test",
                        conditions=predicate_pattern_list,
                        actions=predicate_pattern_list)

        c = Conditional("test", actions=predicate_pattern_list)

        c = Conditional("test", conducts=predicate_pattern_list)

        c = Conditional("test",
                        conditions=predicate_pattern_list,
                        actions=predicate_pattern_list,
                        conducts=predicate_pattern_list)

        c = Conditional("test",
                        conditions=predicate_pattern_list,
                        actions=predicate_pattern_list,
                        function_var_names=["str_1", "str_2"])

        c = Conditional("test",
                        conditions=predicate_pattern_list,
                        actions=predicate_pattern_list,
                        normal=["str_1", "str_2"])

        c = Conditional("test",
                        conditions=predicate_pattern_list,
                        actions=predicate_pattern_list,
                        function=1)

        c = Conditional("test",
                        conditions=predicate_pattern_list,
                        actions=predicate_pattern_list,
                        function=2.333)

        c = Conditional("test",
                        conditions=predicate_pattern_list,
                        actions=predicate_pattern_list,
                        function="test_string")

        c = Conditional("test",
                        conditions=predicate_pattern_list,
                        actions=predicate_pattern_list,
                        function=torch.tensor([[1., -1.], [1., -1.]]))

        c = Conditional("test",
                        conditions=predicate_pattern_list,
                        conducts=predicate_pattern_list,
                        actions=predicate_pattern_list,
                        function_var_names=["str_1", "str_2"],
                        function=torch.tensor([[1., -1.], [1., -1.]]),
                        normal=["str_1", "str_2"])











