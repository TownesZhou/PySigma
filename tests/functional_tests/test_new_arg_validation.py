import pytest
from pysigma.structures import *
from torch.distributions import Distribution


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


class TestPredicateArgumentProblematic:
    """
        Test Predicate with problematic arguments
    """

    test_type_1 = Type("type1", False, 1)
    test_type_2 = Type("type2", False, 10)
    test_type_3 = Type("type3", True, symbol_list=["a", "b", "c"])
    test_type_4 = Type("type4", True, symbol_list=["1", "2", "3"])

    correct_relational_args = [('type1', test_type_3)]
    correct_random_args = [(test_type_1, test_type_2)]

    def test_predicate_problematic_1(self):
        # name argument must be str
        with pytest.raises(ValueError):
            t = Predicate(1, [self.correct_relational_args], self.correct_random_args, "BP")

    def test_predicate_problematic_2(self):
        # relational_args must be iterable of size 2 tuple
        with pytest.raises(ValueError):
            t = Predicate("test", [(1, 1), (2, 3, 4)], self.correct_random_args, "BP")

    def test_predicate_problematic_3(self):
        # relational_args 1st element of tuple should be 'str', 2nd should be 'Type'
        with pytest.raises(ValueError):
            t1 = Predicate("test", [(1, "str")], self.correct_random_args, "BP")
            t2 = Predicate("test", [(1, self.test_type_1)], self.correct_random_args, "BP")
            t3 = Predicate("test", [(self.test_type_1, self.test_type_2)], self.correct_random_args, "BP")

    def test_predicate_problematic_4(self):
        # relational_args 2nd element type argument name should be distinct
        with pytest.raises(ValueError):
            t1 = Predicate("test", [("arg1", self.test_type_1), ("arg2", self.test_type_2)], self.correct_random_args,
                           "BP")

    def test_predicate_problematic_5(self):
        # random_args tuple argument name should be distinct among themselves
        with pytest.raises(ValueError):
            t1 = Predicate("test", self.correct_relational_args,
                           [("arg1", self.test_type_1), ("arg1", self.test_type_1)], "BP")

    def test_predicate_problematic_6(self):
        # random_args tuple argument name should be distinct from relational args
        with pytest.raises(ValueError):
            t1 = Predicate("test", [("arg1", self.test_type_1)],
                           [("arg1", self.test_type_1), ("arg2", self.test_type_1)], "BP")

    def test_predicate_problematic_7(self):
        # random_args tuple Type must not be symbolic
        with pytest.raises(ValueError):
            t1 = Predicate("test", [("arg1", self.test_type_1)],
                           [("arg1", self.test_type_3), ("arg2", self.test_type_3)], "BP")

    def test_predicate_problematic_8(self):
        # random_args tuple Type must be the same type
        with pytest.raises(ValueError):
            t1 = Predicate("test", [("arg1", self.test_type_1)],
                           [("arg1", self.test_type_1), ("arg2", self.test_type_1)], "BP")

    def test_predicate_problematic_9(self):
        # inference_mode other "BP", "VMP", "EP"
        with pytest.raises(ValueError):
            t1 = Predicate("test1", [("arg_1", self.test_type_1)], [("arg_2", self.test_type_2)], "ELSE")

    def test_predicate_problematic_10(self):
        # num_particles negative size
        with pytest.raises(ValueError):
            t1 = Predicate("test1", [("arg_1", self.test_type_1)], [("arg_2", self.test_type_2)], "BP",
                           num_particles=-1)

    # def test_predicate_problematic_5(self):
    #
    #
    # def test_predicate_problematic_5(self):


class TestPredicateArgumentCorrect:
    """
        Test Predicate with problematic arguments
    """
    test_type_1 = Type("type1", False, 1)
    test_type_2 = Type("type2", False, 10)
    test_type_3 = Type("type3", True, symbol_list=["a", "b", "c"])
    test_type_4 = Type("type4", True, symbol_list=["1", "2", "3"])

    # dist = Distribution(batch_shape=torch.Size([2, 2]), event_shape=torch.Size([2, 2]))

    def test_predicate_correct_1(self):
        # create some correct predicate
        t1 = Predicate("test1", [("arg_1", self.test_type_1)], [("arg_2", self.test_type_2)], "BP")
        t2 = Predicate("test2", [("arg_1", self.test_type_1), ("arg_3", self.test_type_3)],
                       [("arg_2", self.test_type_2)], "VMP")
        t3 = Predicate("test3", [("arg_4", self.test_type_4), ("arg_3", self.test_type_3)],
                       [("arg_1", self.test_type_2), ("arg_2", self.test_type_2)], "EP")

    def test_predicate_correct_2(self):
        # specify number of particles
        t1 = Predicate("test1", [("arg_1", self.test_type_1)], [("arg_2", self.test_type_2)], "BP", num_particles=100)

    # TODO: test the distribution_class argument
    # def test_predicate_correct_3(self):
    #     t1 = Predicate("test1", [("arg_1", self.test_type_1)], [("arg_2", self.test_type_2)], "BP", distribution_class=self.dist)

    def test_predicate_correct_4(self):
        # memorial predicate
        t1 = Predicate("test1", [("arg_1", self.test_type_1)], [("arg_2", self.test_type_2)], "BP", memorial=True)

    def test_predicate_correct_4(self):
        # perceptual predicate
        t1 = Predicate("test1", [("arg_1", self.test_type_1)], [("arg_2", self.test_type_2)], "BP", perceptual=True)

    def test_predicate_correct_4(self):
        # memorial and perceptual predicate
        t1 = Predicate("test1", [("arg_1", self.test_type_1)], [("arg_2", self.test_type_2)], "BP", memorial=True,
                       perceptual=True)


class TestConditionalArgumentProblematic:
    """
        Test Conditional with problematic arguments
    """
    test_type_1 = Type("type1", False, 1)
    test_type_2 = Type("type2", False, 10)
    test_type_3 = Type("type3", True, symbol_list=["a", "b", "c"])
    test_pred_1 = Predicate("test1", [("arg_1", test_type_1)], [("arg_2", test_type_2)], "BP")

    def test_conditional_problematic_1(self):
        # name should be string
        with pytest.raises(ValueError):
            t = Conditional(1, [(self.test_pred_1, None)])



class TestConditionalArgumentCorrect:
    """
        Test Conditional with correct arguments
    """
    test_type_1 = Type("type1", False, 1)
    test_type_2 = Type("type2", False, 10)
    test_type_3 = Type("type3", True, symbol_list=["a", "b", "c"])

    test_pred_1 = Predicate("test1", [("arg_1", test_type_1)], [("arg_2", test_type_2)], "BP")
    test_pred_2 = Predicate("test2", [("arg_1", test_type_1), ("arg_3", test_type_3)],
                       [("arg_2", test_type_2)], "VMP")

    pat_element_1 = ('arg_1', 'var1')
    pat_element_2 = ('arg_2', 'var2')

    pred_patterns_1 = (test_pred_1, [pat_element_1, pat_element_2])
    pred_patterns_2 = (test_pred_2, [pat_element_1, pat_element_2])


    test_conditions = [pred_patterns_1, pred_patterns_2]
    test_condacts = [pred_patterns_2, pred_patterns_1]


    def test_conditional_correct_1(self):
        # create conditional with only name
        t = Conditional("cond_1", self.test_conditions, self.test_condacts)

