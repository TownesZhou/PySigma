import torch
from pysigma.sigmaboard import render
from pysigma import Sigma
from pysigma.structures import *


def run_test(test, *args, **kwargs):
    print("\n\n########## RUNNING TEST: {} ##########\n".format(test.__name__))
    test(*args, **kwargs)
    print("\n####################")


def predicate_compile_test_1(to_render=False):
    """
        open-world predicate without selection
    """
    sigma = Sigma()     # init

    # add two types to Sigma
    type_1 = Type('type_1', 'discrete', min=0, max=5)
    type_2 = Type('type_2', 'symbolic', symbol_list=['a', 'b', 'c'])
    sigma.add(type_1)
    sigma.add(type_2)

    # two predicate arguments, without selection
    arg_1 = PredicateArgument('arg_1', type_1)
    arg_2 = PredicateArgument('arg_2', type_2)
    # Predicate
    test_pred = Predicate('test_pred', [arg_1, arg_2], world='open')
    # Add the predicate to the sigma program
    sigma.add(test_pred)

    # test web-based render
    if to_render:
        render(sigma)


def predicate_compile_test_2(to_render=False):
    """
        closed-world, perceptual, memorial (Long-Term Memory), w/out selection, multiple variables
    """
    # 1 pred, 2 args, without selection
    sigma = Sigma()  # init

    # add two types to Sigma
    type_1 = Type('type_1', 'discrete', min=0, max=5)
    type_2 = Type('type_2', 'symbolic', symbol_list=['a', 'b', 'c'])
    sigma.add(type_1)
    sigma.add(type_2)

    # two predicate arguments without selection
    arg_1 = PredicateArgument('arg_1', type_1)
    arg_2 = PredicateArgument('arg_2', type_2)
    # Predicate
    test_pred = Predicate('test_pred', [arg_1, arg_2], world='closed', perception=True, function=2)
    # Add the predicate to the sigma program
    sigma.add(test_pred)

    # test web-based render
    if to_render:
        render(sigma)


def predicate_compile_test_3(to_render=False):
    """
        closed-world, perceptual, memorial, with selection, multiple variables
    """
    # 1 pred, 2 args, without selection
    sigma = Sigma()  # init

    # add two types to Sigma
    type_1 = Type('type_1', 'discrete', min=0, max=5)
    type_2 = Type('type_2', 'symbolic', symbol_list=['a', 'b', 'c'])
    sigma.add(type_1)
    sigma.add(type_2)

    # two predicate arguments, arg_1 with best-selection, arg_2 without selection
    arg_1 = PredicateArgument('arg_1', type_1, unique_symbol='!')
    arg_2 = PredicateArgument('arg_2', type_2)
    # Predicate
    test_pred = Predicate('test_pred', [arg_1, arg_2], world='closed', perception=True, function=2)
    # Add the predicate to the sigma program
    sigma.add(test_pred)

    # test web-based render
    if to_render:
        render(sigma)


def predicate_compile_test_4(to_render=False):
    """
        closed-world, perceptual, memorial (Long-Term Memory), w/ selection
    :return:
    """
    pass


def conditional_compile_test_1(to_render=True):
    """
        transitivity rule:
            2 conditions, 1 action, without function
    """
    sigma = Sigma()

    # add one type
    type_1 = Type("type_1", 'discrete', min=0, max=3)
    sigma.add(type_1)

    # one arity-2 predicate
    arg_1 = PredicateArgument("arg_1", type_1)
    arg_2 = PredicateArgument("arg_2", type_1)
    pred = Predicate("test_pred", [arg_1, arg_2], world="closed")
    sigma.add(pred)

    # transitivity conditional
    cond_1 = Conditional("cond_1",
                         conditions=[
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("a")),
                                               PatternElement("arg_2", PatternVariable("b"))]),
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("b")),
                                               PatternElement("arg_2", PatternVariable("c"))])

                         ],
                         actions=[
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("a")),
                                               PatternElement("arg_2", PatternVariable("c"))])
                         ]
                         )
    sigma.add(cond_1)

    if to_render:
        render(sigma)


def conditional_compile_test_2(to_render=True):
    """
        transitivity rule:
            2 conditions, 1 action, with function
    """
    sigma = Sigma()

    # add one type
    type_1 = Type("type_1", 'discrete', min=0, max=3)
    sigma.add(type_1)

    # one arity-2 predicate
    arg_1 = PredicateArgument("arg_1", type_1)
    arg_2 = PredicateArgument("arg_2", type_1)
    pred = Predicate("test_pred", [arg_1, arg_2], world="closed")
    sigma.add(pred)

    # transitivity conditional
    cond_1 = Conditional("cond_1",
                         conditions=[
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("a")),
                                               PatternElement("arg_2", PatternVariable("b"))]),
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("b")),
                                               PatternElement("arg_2", PatternVariable("c"))])

                         ],
                         actions=[
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("a")),
                                               PatternElement("arg_2", PatternVariable("c"))])
                         ],
                         function=2, function_var_names=["a", "b", "c"]
                         )
    sigma.add(cond_1)

    if to_render:
        render(sigma)


def conditional_compile_test_3(to_render=True):
    """
        transitivity rule:
            2 conditions, 1 action, with function

        Check if pattern variable's sum_op is correctly detected
    """
    sigma = Sigma()

    # add one type
    type_1 = Type("type_1", 'discrete', min=0, max=3)
    sigma.add(type_1)

    # one arity-2 predicate
    arg_1 = PredicateArgument("arg_1", type_1, probabilistic=True, unique_symbol='!', normalize=True)
    arg_2 = PredicateArgument("arg_2", type_1, probabilistic=True, unique_symbol='!', normalize=True)
    pred = Predicate("test_pred", [arg_1, arg_2], world="closed")
    sigma.add(pred)

    # transitivity conditional
    cond_1 = Conditional("cond_1",
                         conditions=[
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("a")),
                                               PatternElement("arg_2", PatternVariable("b"))]),
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("b")),
                                               PatternElement("arg_2", PatternVariable("c"))])

                         ],
                         actions=[
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("a")),
                                               PatternElement("arg_2", PatternVariable("c"))])
                         ],
                         function=2, function_var_names=["a", "b", "c"]
                         )
    sigma.add(cond_1)

    if to_render:
        render(sigma)


def conditional_compile_test_4(to_render=True):
    """
        transitivity rule:
            2 conditions, 1 action, with function

        Test negation node topology
    """
    sigma = Sigma()

    # add one type
    type_1 = Type("type_1", 'discrete', min=0, max=3)
    sigma.add(type_1)

    # one arity-2 predicate
    arg_1 = PredicateArgument("arg_1", type_1, probabilistic=False, unique_symbol=None, normalize=False)
    arg_2 = PredicateArgument("arg_2", type_1, probabilistic=False, unique_symbol=None, normalize=False)
    pred = Predicate("test_pred", [arg_1, arg_2], world="closed")
    sigma.add(pred)

    # transitivity conditional
    cond_1 = Conditional("cond_1",
                         conditions=[
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("a")),
                                               PatternElement("arg_2", PatternVariable("b"))]),
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("b")),
                                               PatternElement("arg_2", PatternVariable("c"))])

                         ],
                         actions=[
                             PredicatePattern("test_pred",
                                              [PatternElement("arg_1", PatternVariable("a")),
                                               PatternElement("arg_2", PatternVariable("c"))],
                                              negation=True)
                         ],
                         function=2, function_var_names=["a", "b", "c"]
                         )
    sigma.add(cond_1)

    if to_render:
        render(sigma)


if __name__=="__main__":

    run_test(predicate_compile_test_1)
    run_test(predicate_compile_test_2)
    run_test(predicate_compile_test_3, to_render=False)
    run_test(conditional_compile_test_1, to_render=False)
    run_test(conditional_compile_test_2, to_render=False)
    run_test(conditional_compile_test_3, to_render=False)
    run_test(conditional_compile_test_4, to_render=True)

