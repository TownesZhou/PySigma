import torch
from Structures import *
from Cognitive import *
from Graphical import *
from Sigmaboard import *


def run_test(test, *args, **kwargs):
    print("\n\n########## RUNNING TEST: {} ##########\n".format(test.__name__))
    test(*args, **kwargs)
    print("\n####################")


def align(msg, var_list, full_var_list):
    """
        Compute and return the bradcasted message with its dimension properly aligned with the factor node variable
            order.

        Note: MAY BE REPLACED LATER WITH PYTORCH'S BUILT-IN DIMENSION REARRANGEMENT FOR NAMED TENSORS.
    :param msg:  unbroadcasted message from a linkdata
    :param vars: list of variables of the unbroadcasted message
    :return:    broadcasted message with dimension aligned
    """
    # First take a view that append extra dimensions to the message
    view_dim = msg.shape + ((1,) * (len(full_var_list) - len(msg.shape)))
    # Find indices of the variables in _var_list
    var_id = [full_var_list.index(var) for var in var_list]
    # Compute the permutation
    perm = [-1] * len(full_var_list)  # -1 is dummy value
    for i, id in enumerate(var_id):
        perm[id] = i
    rest = len(var_list)
    for i in range(len(full_var_list)):
        if i not in var_id:
            perm[i] = rest
            rest += 1
    # Permute message dimension and return
    aligned_msg = msg.view(view_dim).permute(perm)
    return aligned_msg


def align_test_1():
    full_var_list = ['a', 'b', 'c', 'd', 'e']
    var_list = ['d', 'b', 'e']
    # msg = torch.tensor([[1, 1], [2, 2], [3, 3]])
    msg = torch.ones(2, 3, 6)
    print("original msg dimension: {}".format(msg.shape))
    aligned_msg = align(msg, var_list, full_var_list)
    print("aligned msg dimension: {}".format(aligned_msg.shape))


def align_test_2():
    full_var_list = ['a', 'b', 'c', 'd', 'e']
    var_list = ['d', 'b', 'e', 'a', 'c']
    # msg = torch.tensor([[1, 1], [2, 2], [3, 3]])
    msg = torch.ones(1, 2, 3, 4, 5)
    print("original msg dimension: {}".format(msg.shape))
    aligned_msg = align(msg, var_list, full_var_list)
    print("aligned msg dimension: {}".format(aligned_msg.shape))


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
    arg_1 = PredicateArgument('arg_1', type_1, '!')
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


def conditional_compile_test_1():
    """
        transitivity rule
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
                             PredicatePattern("test_pred", None,
                                              [PatternElement("arg_1", PatternVariable("a")),
                                               PatternElement("arg_2", PatternVariable("b"))]),
                             PredicatePattern("test_pred", None,
                                              [PatternElement("arg_1", PatternVariable("b")),
                                               PatternElement("arg_2", PatternVariable("c"))])

                         ],
                         actions=[
                             PredicatePattern("test_pred", None,
                                              [PatternElement("arg_1", PatternVariable("a")),
                                               PatternElement("arg_2", PatternVariable("c"))])
                         ]
                         )
    sigma.add(cond_1)
    # TODO: Test this case, especially how conditional's lookup tables are properly set up


run_test(align_test_1)
run_test(align_test_2)
# run_test(predicate_compile_test_1)
# run_test(predicate_compile_test_2)
run_test(predicate_compile_test_3, to_render=False)
run_test(conditional_compile_test_1)
