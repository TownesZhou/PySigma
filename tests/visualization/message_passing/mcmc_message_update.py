"""
    Visualization for the message update step of Metropolis-Hastings MCMC algorithm, compartmentalized in the WMFN node.

    The goal of this visualization is to show that, given a *stable* target function, the message update step in WMFN
    can incrementally move the particles toward the target function's important region. The target function and outside
    linkdata and node connections are stubbed out for unit testing purpose.
"""
from unittest.mock import MagicMock, patch
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Size
import torch.distributions as D
from pysigma.graphical.predicate_nodes import WMFN


if __name__ == "__main__":
    pass
