import pytest
from pysigma.structures import *


class TestTypeArgumentProblematic:
    """
        Test Type with problematic arguments
    """
    def test_type_problematic_1(self):
        # 1st argument must be str
        with pytest.raises(ValueError):
            t = Type(1, "discrete")

    def test_type_problematic_2(self):
        # 2nd argument must be bool
        with pytest.raises(ValueError):
            t = Type("test", "not symbolic or discrete")

    def test_type_problematic_3(self):
        # 'symbolic' should be True if and only if 'symbol_list' is provided
        with pytest.raises(ValueError):
            t = Type("test", True)

    def test_type_problematic_4(self):
        # 'symbolic' should be False if and only if 'size' is provided
        with pytest.raises(ValueError):
            t = Type("test", False)

    def test_type_problematic_5(self):
        # size must be int
        with pytest.raises(ValueError):
            t = Type("test", False, size=1.1)

    def test_type_problematic_6(self):
        # size must be greater than 0
        with pytest.raises(ValueError):
            t = Type("test", False, size=0)

    def test_type_problematic_7(self):
        # 'symbol_list' must be an iterable of 'str'
        with pytest.raises(ValueError):
            t = Type("test", True, symbol_list=["a", "b", 1])

    def test_type_problematic_8(self):
        # create a symbolic type with both size and symbol_list
        with pytest.raises(ValueError):
            t = Type("test", True, 1, symbol_list=["a", "b", "c"])

    def test_type_problematic_9(self):
        # create a non-symbolic type with both size and symbol_list
        with pytest.raises(ValueError):
            t = Type("test", False, 1, symbol_list=["a", "b", "c"])

class TestTypeArgumentCorrect:
    """
        Test Type with correct arguments
    """

    def test_type_correct_1(self):
        # create a symbolic type with only symbol_list
        t = Type("test", True, symbol_list=["a", "b", "c"])

    def test_type_correct_2(self):
        # create a non-symbolic type with only size
        t = Type("test", False, 1)


# class TestPredicateArgumentCorrect:
#     """
#         Test Predicate with problematic arguments
#     """
#
#     test_type_1 = Type("type1", False, 1)
#     test_type_2 = Type("type2", False, 10)
#     test_type_3 = Type("type3", True, symbol_list=["a", "b", "c"])
#     test_type_4 = Type("type4", True, symbol_list=["1", "2", "3"])
#
#     incorrect_relational_arg = (1, test_type_1)
#     correct_relational_args = ('type1', test_type_3)
#
#     incorrect_random_args = (1, test_type_1)
#     correct_random_args = (test_type_1, test_type_2)
#
#     def test_predicate_problematic_1(self):
#         # name argument must be str
#         with pytest.raises(ValueError):
#             t = Predicate(1, [self.correct_relational_args], [self.correct_random_args], "BP")
#
#     def test_predicate_problematic_2(self):
#         # relational_args must be iterable of size 2 tuple
#         with pytest.raises(ValueError):
#             t = Predicate("test", [(1, 1), (2, 3, 4)])
#
#     def test_predicate_problematic_3(self):
#         # 1st element of tuple should be 'str', 2nd should be 'Type'
#
#         with pytest.raises(ValueError):
#             t1 = Predicate("test", relational_args, random_args, "BP")

    # def test_predicate_problematic_4(self):
    #     # 2nd element type argument name should be distinct
    #     with pytest.raises(ValueError):
    #         t1 = Predicate("test", [("type1", self.test_type_1),
    #                                 ("type2", self.test_type_1)])
    #
    # def test_predicate_problematic_5(self):
    #     # tuple argument name should be distinct among themselves
    #     with pytest.raises(ValueError):
    #         t1 = Predicate("test", [("type1", self.test_type_1),
    #                                 ("type2", self.test_type_1)])
    #
    #
    # def test_predicate_problematic_5(self):
    #
    # def test_predicate_problematic_5(self):
    #
    # def test_predicate_problematic_5(self):
    #
    #
    # def test_predicate_problematic_5(self):
#
#     def test_predicate_correct(self):
#         """
#             Test Predicate with correct arguments
#         """
#
#
#     def test_consitional_problematic(self):
#         """
#             Test Conditional with problematic arguments
#         """
#
#
#     def test_consitional_correct(self):
#         """
#             Test Conditional with correct arguments
#         """
#
#
#
# if __name__ == "__main__":
#     t = TestStructureArgValidation()
#
#     t.test_type_problematic()
#     t.test_type_correct()
#
#     # t.test_predicate_problematic()
#     # t.test_predicate_correct()
#     #
#     # t.test_conditional_problematic()
#     # t.test_conditional_correct()
#
#     # print("Argument test passed successfully without error")