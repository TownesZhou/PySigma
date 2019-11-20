"""
    Cognitive interface to Sigma cognitive architecture, which is the top part of the cognitive hourglass that specifies
        the cognitive language.

    In this Python implementation of Sigma that was originally (and currently) wrote in Lisp, we endeavor to
        1. take advantage of PyTorch’s autograd functionality, parallelizable features, and GPU support to speed up
           inference and learning.
        2. Enable flexible, generalizable, and customizable declarations of (pattern) variable bindings supported by
           highly flexible tensor operations.
                E.g. arbitrary arithmetic equations. Or even more, such as control structures.
        3. provide easy-to-use integration with neural models, and therefore connect to research in deep learning
           community.
        4. take advantage of Python’s language features to support further functional elegance of the architecture,
           and retain, or even improve, the flexibility of the architecture, so as to hopefully speed up iterations of
           further research, such as threads on structure learning and support for continuous distributions.
        5. (optional) experiment with probabilistic programming libraries, such as Pyro, to see if integration of such
           probabilistic programming paradigm to Sigma would enable efficient inference and learning with continuous
           parameterized distributions.


    Author: Jincheng Zhou, University of Southern California
    Reference: Paul S. Rosenbloom, Abram Demski & Volkan Ustun, 2016, The Sigma Cognitive Architecture and System:
        Toward Functionally Elegant Grand Unification
"""


class Sigma:
    """
        Sigma's cognitive interface. User should instantiate this class in order to create a Sigma model (or program).
    """
    def __init__(self):
        pass
