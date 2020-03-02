import torch
from pysigma import Sigma
from pysigma.structures import *
from pysigma.sigmaboard import render


def run_test(test, *args, **kwargs):
    print("\n\n########## RUNNING TEST: {} ##########\n".format(test.__name__))
    test(*args, **kwargs)
    print("\n####################")


def simple_1(to_render=False):
    """
        Test simple transitivity rule.

        Three objects: {o1, o2, o3}
    """
    sigma = Sigma()     # Empty program

    # One discrete type with span 3
    type1 = Type("type1", 'discrete', min=0, max=3)
    sigma.add(type1)

    # One arity-2 predicate
    init_state = torch.tensor([[1., 1., 0.],
                               [0., 1., 1.],
                               [0., 0., 1.]])
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
    sigma.print_predicate_memory([pred1])
    sigma.print_combined_action([pred1])

    # Run
    sigma.decide(num_cycles=1, verbose=1)

    # Render (optional, only for debugging)
    if to_render:
        render(sigma)


if __name__=="__main__":
    run_test(simple_1, to_render=False)