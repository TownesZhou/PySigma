import torch
from pysigma import Sigma
from pysigma.structures import *
from pysigma.sigmaboard import render


def run_test(test):

    def inner(*args, **kwargs):
        print("\n\n########## RUNNING TEST: {} ##########\n".format(test.__name__))
        returned_val = test(*args, **kwargs)
        print("\n####################")
        return returned_val

    return inner


def build_transitivity_1(init_state, num_objects, to_render):
    sigma = Sigma()  # Empty program

    # One discrete type with span 3
    type1 = Type("type1", 'discrete', min=0, max=num_objects)
    sigma.add(type1)

    # One arity-2 predicate
    arg1 = PredicateArgument('arg1', type1, probabilistic=True, unique_symbol=None, normalize=False)
    arg2 = PredicateArgument('arg2', type1, probabilistic=True, unique_symbol=None, normalize=False)
    pred1 = Predicate('pred1', [arg1, arg2], world='closed', perception=True, function=init_state)
    sigma.add(pred1)

    # Transitivity rule
    cond1 = Conditional('cond1',
                        conditions=[
                            PredicatePattern('pred1', [
                                PatternElement('arg1', PatternVariable('a')),
                                PatternElement('arg2', PatternVariable('b'))
                            ]),
                            PredicatePattern('pred1', [
                                PatternElement('arg1', PatternVariable('b')),
                                PatternElement('arg2', PatternVariable('c'))
                            ])
                        ],
                        actions=[
                            PredicatePattern('pred1', [
                                PatternElement('arg1', PatternVariable('a')),
                                PatternElement('arg2', PatternVariable('c'))
                            ])
                        ])

    sigma.add(cond1)

    # Register print
    # sigma.print_predicate_memory([pred1])
    # sigma.print_combined_action([pred1])

    # Run
    sigma.decide(num_cycles=1, verbose=1)

    # Render (optional, only for debugging)
    if to_render:
        render(sigma)


@run_test
def simple_1(to_render=False):
    """
        Test simple transitivity rule.

        Three objects: {o1, o2, o3}
    """
    init_state = torch.tensor([[1., 1., 0.],
                               [0., 1., 1.],
                               [0., 0., 1.]])
    build_transitivity_1(init_state, 3, to_render)


@run_test
def large_1(N=1000, to_render=False):
    """
        Test large transitivity rule, with large number of objects.
    """
    init_state = torch.zeros((N, N))
    for i in range(N-1):
        init_state[i, i] = 1
        init_state[i, i+1] = 1
    init_state[N-1, N-1] = 1

    build_transitivity_1(init_state, N, to_render)


@run_test
def probabilistic_1(to_render=False):
    init_state = torch.tensor([[1., 0.9, 0., 0.],
                               [0., 1., 0.8, 0],
                               [0., 0., 1., 0.7],
                               [0., 0., 0., 1.]])
    build_transitivity_1(init_state, 4, to_render)


if __name__=="__main__":
    # simple_1(to_render=True)
    large_1(N=200, to_render=True)
    # probabilistic_1(to_render=True)
