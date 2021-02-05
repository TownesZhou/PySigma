"""
    This module defines architecture-defined built-in default Summarization Callback procedures.
"""
from typing import Union, Optional, Tuple, Callable
import inspect
from inspect import Signature, Parameter
import functools
import torch
from torch.distributions import Distribution

from ..utils import DistributionServer
from ..defs import Message, MessageType
from .summarization import Summarization
from .summarization import Summarization, _Summarization
from .summarization import ContentFlagTyping as CFT
from .summarization import DistributionTyping as DT
from .summarization import ParticleTyping as PT
from .summarization import WeightTyping as WT
from .summarization import LogDensityTyping as LDT
from .summarization import ReturnValueTyping as RVT


# Default combination Summarization operation
@Summarization(repr_type='dual', dist_repr_type='parameter')
def sum_op_combination(content_flag: CFT,
                       dist_content: DT,
                       particle: PT,
                       weight: WT,
                       log_density: LDT) -> RVT:
    """The default combination Summarization operation. Semantically, this op "combines" all candidate distribution in
    the batch to derive a single distribution that "best describes all original distributions".

    For distribution contents, it assumes the distribution parameter space is a vector space, and sums the parameter
    tensor across the target (first) batch dimension.

    For particle contents, it combines the particle weight by multiplying the particle weight tensor across the target
    batch dimension.
    """
    return_param = dist_content
    return_weight = weight

    if dist_content is not None:
        # Sum across the first batch dimension
        return_param = dist_content.sum(dim=0)

    if weight is not None:
        # Multiply across the first batch dimension
        log_weight = weight.log()
        sum_log_weight = log_weight.sum(dim=0)
        return_weight = sum_log_weight.exp()

    return return_param, return_weight

