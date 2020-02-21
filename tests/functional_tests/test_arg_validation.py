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


