import torch
from pysigma.sigmaboard import SigmaBoard
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
    test_type_1 = Type("type1", False, 1)
    test_type_2 = Type("type2", False, 10)
    test_type_3 = Type("type3", True, symbol_list=["a", "b", "c"])
    test_type_4 = Type("type4", True, symbol_list=["1", "2", "3"])

    sigma.add(test_type_1)
    sigma.add(test_type_2)
    sigma.add(test_type_3)
    sigma.add(test_type_4)

    test_pred = Predicate("test3", [("arg_4", self.test_type_4), ("arg_3", self.test_type_3)],
                   [("arg_1", self.test_type_2), ("arg_2", self.test_type_2)], "EP")


    # Add the predicate to the sigma program
    sigma.add(test_pred)

    # test web-based render
    if to_render:
        SigmaBoard(sigma).render()

if __name__=="__main__":
    run_test(predicate_compile_test_1)