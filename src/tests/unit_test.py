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


if __name__=="__main__":

    run_test(align_test_1)
    run_test(align_test_2)