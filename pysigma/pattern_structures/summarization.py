"""
    The Summarization pattern structure. Defines the Summarization decorator class to be used to decorate a
        Summarization callback routine.
"""
from typing import Union, Optional, Tuple, Callable
import inspect
from inspect import Signature, Parameter
from collections.abc import Iterable
from itertools import chain
import functools
import numpy as np
import torch
from torch import Size
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint
from ..utils import intern_name, extern_name, DistributionServer, KnowledgeServer
from ..defs import Variable, VariableMetatype, Message, MessageType


# Typing Annotations for the Summarization Callback routine
ContentFlagTyping = str
DistributionTyping = Union[torch.distributions.Distribution, torch.Tensor, None]
ParticleTyping = Optional[torch.Tensor]
WeightTyping = Optional[torch.Tensor]
LogDensityTyping = Optional[torch.Tensor]
ReturnValueTyping = Tuple[Union[torch.distributions.Distribution, torch.Tensor, None], Optional[torch.Tensor]]

# Typing Annotations for internal use
_Callable_Arguments_Type_List = [ContentFlagTyping, DistributionTyping, ParticleTyping, WeightTyping, LogDensityTyping]
_Callable_Signature = Callable[[ContentFlagTyping,
                                DistributionTyping,
                                ParticleTyping,
                                WeightTyping,
                                LogDensityTyping],
                               ReturnValueTyping]

# Other module-level internal constants
_NUM_CALLBACK_PARAM = 5


# Custom Exception classes
class SummarizationCallbackAnnotationError(Exception):
    pass


class SummarizationContentTypeError(Exception):
    pass


class SummarizationValueError(Exception):
    pass


class _Summarization:
    """The class wrapping the custom summarization callback routine which will be called during the outward
    summarization step in the Expansion / Summarization Factor Node (ESFN).

    This is the underlying wrapper behind the top-level wrapper , the `Summarization` method as defined below which
    provides the ``@Summarization`` decorator.

    In general, a summarization routine "selects" a distribution or "combines" all distributions in a batch to derive
    a single summarized distribution that best semantically "describes" the original batch. To make a summarization
    routine, the decorator ``@Summarization`` must be used to decorate the target callable, before passing the callable
    as an argument to a Conditional's predicate pattern.

    In PySigma, a distribution may be represented in two ways:

        - a PyTorch distribution instance with parameters
        - a particle tuple with three elements: (particle value tensor, weight tensor, log density tensor)

    Therefore, by default, a general summarization routine should expect to receive and process either the distribution
    representation or the particle representation, or both. However, it can be specified via the decorator parameter
    ``repr_type`` whether only the distribution representation is acceptable (``repr_type=distribution``) or the
    particle representation is acceptable (``repr_type=particle``). The default value is ``repr_type=dual``. During
    runtime, if it is specified that only one representation is acceptable to the summarization routine, while the
    incoming message only contains contents in the form of the other representation, a `SummarizationContentTypeError`
    will be raised.

    For the distribution representation, it can be furthermore specified via the decorator parameter
    ``dist_repr_type`` whether only the distribution parameters, in the same form the message uses (e.g. usually the
    natural parameters for exponential family distribution class), should be passed (``dist_repr_type=parameter``), or
    a batched PyTorch distribution instance should be passed (``dist_repr_type=distribution``). The default value is
    ``dist_repr_type=distribution``.

    During each cognitive cycle, the input contents to the Summarization routine will be reshaped and flattened so that
    the batched content tensors will always be 2-dimensional. This includes the distribution parameter tensor from the
    distribution representation and the weight tensor from the particle representation. Note that The particle value
    tensor and the log density tensor from the particle representation does not have batch dimensions, and indeed they
    serves only as references for the computation and should not be altered (for more details, see `Message` class
    documentation). The first dimension of these batched content tensors corresponds to the batch dimensions that should
    be **summarized out**, whereas the second dimension corresponds to other dimensions that should be **left
    untouched**. Shape check will be carried out on the return values, and a `SummarizationValueError` will be raised
    if the shapes are incorrect.

    Note that the particle value tensor and the log density tensor represent joint events, despite in the case of
    multiple random variables it is the marginal events not the joint events that are represented in the
    actual messages. Therefore, the particle value tensor will be 2-dimensional: the first dimension is the sample
    dimension, the second dimension the event dimension of size equal to the sum of all random variables' sizes. The
    log density tensor will be 1-dimensional vector having the same length as the particle value tensor's sample
    dimension.

    A Summarization callback routine should have the following signature. For documentation regarding the decorator
    parameters and runtime exceptions, see the documentation of `__init__` and ``__call__`` methods respectively.

    Note that It is required that **the callback arguments types and return type be annotated using Python type hints**.
    Incomplete or incorrect annotations will result in a ``SummarizationCallbackAnnotationError`` being raised during
    initialization (only the type annotation is checked; it is not required the argument names be followed).

    .. note:: The custom callback routine should have the following signature:

        ``(str, Union[torch.distributions.Distribution, torch.Tensor, None], Optional[torch.Tensor],
        Optional[torch.Tensor], Optional[torch.Tensor])
        -> Tuple[Union[torch.distributions.Distribution, torch.Tensor, None], Optional[torch.Tensor]]``

    Or equivalently (and recommended), use the Typing aliases as defined in this module:

        ``(ContentFlagTyping, DistributionTyping, ParticleTyping, WeightTyping, LogDensityTyping)
        -> ReturnValueTyping``

    Below is the documentation on each input argument as well as the return value.

    Parameters
    ----------
    content_flag : str
        Of value ``"distribution"``, ``"particle"``, or ``"dual"``. The type of the content that is available and passed
        into the callback routine.
    dist_content : torch.distributions.Distribution or torch.Tensor, optional
        The batched distribution contents. If decorator parameter ``dist_repr_type=parameter`` is specified, then
        if available it will be the batched parameter tensor used by the passing messages with shape ``[b_1, b_2, p]``
        where ``p`` is the size of a parameter vector, ``b_1`` the size of the batch dimension to be summarized, and
        ``b_2`` the size of the batch dimension to be left intact. Otherwise, if available it will be a PyTorch
        distribution instance with a batch shape ``[b_1, b_2]``. This field will be ``None`` if there is no distribution
        contents in the incoming message, or if the decorator parameter ``repr_type=particle`` is specified.
    particle : torch.Tensor, optional
        If available, a 2-dimensional particle value tensor with shape ``[s, e]``, where ``s`` is the sample size, and
        ``e`` the event size of a particle vector. This field will be ``None`` if there is no particle contents in the
        incoming message, or if the decorator parameter ``repr_type=distribution`` is specified.
    weight : torch.Tensor, optional
        If available, a batched particle weight tensor with shape ``[b_1, b_2, s]``. Similarly, This field will be
        ``None`` if there is no particle contents in the incoming message, or if the decorator parameter
        ``repr_type=distribution`` is specified.
    log_density : torch.Tensor, optional
        If available, a particle log density tensor with shape ``[s]``. Alternatively, this will will be ``None`` if
        there is no particle contents in the incoming message, or if the decorator parameter ``repr_type=distribution``
        is specified.

    Returns
    -------
    (summed_distribution, summed_weight): (torch.Tensor or torch.distributions.Distribution or None, torch.Tensor or None)
        The return value should be a 2-tuple. If available, the first element should be the summarized distribution
        contents. If the decorator parameter ``dist_repr_type=parameter`` is specified, it should be a summarized
        parameter tensor of shape ``[b_2, p]``, otherwise a summarized PyTorch distribution instance with a batch shape
        ``[b_2]``. If available, the second element should be the summarized particle weight tensor with shape
        ``[b_2, s]``. Either element should not be ``None`` if the corresponding un-summarized contents were provided
        as input, otherwise a ``SummarizationValueError`` will be raised.

    """
    def __init__(self,
                 sum_func: _Callable_Signature,
                 repr_type: str = 'dual',
                 dist_repr_type: str = 'distribution'):
        # Update wrapped function's information to enable introspection
        functools.update_wrapper(self, sum_func)
        self._sum_func = sum_func
        self._repr_type = repr_type
        self._dist_repr_type = dist_repr_type

        assert callable(sum_func)
        assert repr_type in ['distribution', 'particle', 'dual'], \
            "When decorating {}: decorator parameter `repr_type` can only have value 'distribution', 'particle', " \
            "or 'dual'. Encountered '{}'."\
            .format(self._sum_func.__name__, repr_type)
        assert dist_repr_type in ['distribution', 'parameter'], \
            "When decorating {}: decorator parameter `dist_repr_type` can only have value 'distribution' or " \
            "'parameter'. Encountered '{}'."\
            .format(self._sum_func.__name__, dist_repr_type)

        # Check callable signature
        sig = inspect.signature(self._sum_func)
        if len(sig.parameters) != _NUM_CALLBACK_PARAM:
            raise SummarizationCallbackAnnotationError(
                "Wrong number of arguments: A Summarization callback routine should accept exactly {} arguments. "
                "Found {} arguments in the callable {}."
                .format(_NUM_CALLBACK_PARAM, len(sig.parameters), self._sum_func.__name__)
            )
        if any(p.annotation is Parameter.empty for p in sig.parameters.values()):
            raise SummarizationCallbackAnnotationError(
                "Incomplete typing annotations: All arguments and the return value of the summarization callback "
                "routine should be annotated. Found unannotated arguments in the callback {}."
                .format(self._sum_func.__name__)
            )
        if sig.return_annotation is Signature.empty:
            raise SummarizationCallbackAnnotationError(
                "Incomplete typing annotations: All arguments and the return value of the summarization callback "
                "routine should be annotated. The return value is unannotated in the callback {}."
                .format(self._sum_func.__name__)
            )
        if any(pa != t for pa, t in zip(tuple(p.annotation for p in sig.parameters.values()),
                                        _Callable_Arguments_Type_List)):
            raise SummarizationCallbackAnnotationError(
                "Incorrect argument typing annotation: The arguments {} in the callback {} have incorrect annotations."
                .format(list(p.name for p, t in zip(sig.parameters.values(), _Callable_Arguments_Type_List)
                             if p.annotation != t), self._sum_func.__name__)
            )
        if sig.return_annotation != ReturnValueTyping:
            raise SummarizationCallbackAnnotationError(
                "Incorrect return value typing annotation: The return value of the callback {} has incorrect "
                "annotations."
                .format(self._sum_func.__name__)
            )

    def __call__(self, msg: Message) -> Message:
        """Implements so that the Summarization instance is a callable functor.

        The summarization instance should be called by ESFN internally only.

        Parameters
        ----------
        msg : Message
            The incoming message.
            Batch shape should be of length 2, with the first batch dimension being the one to be summarized over, and
            the second dimension being the one to be left untouched.
            All particle contents should already be concatenated so that the particle values represent joint events,
            and that sample shape and event shape should be of length 1.
            The message's special attribute dict should contain `'dist_class'` and `'dist_info'` keyed entries,
            otherwise an AssertionError will be raised.

        Returns
        -------
        Message
            The reduced message. Will have only one batch dimension which is the one to be left untouched from the
            original message.
        """
        # Sort out input data
        assert isinstance(msg, Message)
        assert len(msg.b_shape) == 2 and len(msg.e_shape) == 1
        assert MessageType.Parameter in msg.type or MessageType.Particles in msg.type

        # Target batch shape
        summed_b_shape = msg.b_shape[1:]        # Remove first batch dimension

        # If incoming message is an identity, dispose the first batch dimension and return directly
        if msg.isid:
            return_msg = msg.clone()
            return_msg.b_shape = summed_b_shape
            return return_msg

        # Check compatibility between message type and required content representation type
        if self._repr_type == 'particle' and MessageType.Particles not in msg.type:
            raise SummarizationContentTypeError(
                "In summarization routine '{}': the decorator is specified with argument `repr_type=particle`. "
                "However, the incoming message contains only distribution parameters. Please double check model "
                "configuration."
                .format(self._sum_func.__name__)
            )
        if self._repr_type == 'distribution' and MessageType.Parameter not in msg.type:
            raise SummarizationContentTypeError(
                "In summarization routine '{}': the decorator is specified with argument `repr_type=distribution`. "
                "However, the incoming message contains only particle contents. Please double check model "
                "configuration."
                .format(self._sum_func.__name__)
            )

        # Only check existence of distribution-related special attribute if needed
        if self._repr_type in ['distribution', 'dual'] and MessageType.Parameter in msg.type:
            assert 'dist_info' in msg.attr.keys() and 'dist_class' in msg.attr.keys(), \
                "In summarization routine '{}': incoming message containing distribution parameters does not specify " \
                "both 'dist_info' and 'dist_class' key-ed special attributes. Please double check if the initial " \
                "message was generated with these special attributes, or if the attribute dictionary was corrupted " \
                "during message propagation."\
                .format(self._sum_func.__name__)
        dist_info, dist_class = msg.attr['dist_info'], msg.attr['dist_class']

        # Initialize arguments to pass
        dist_content = None
        particle = None
        weight = None
        log_density = None
        # Decide content to pass
        if self._repr_type in ['distribution', 'dual'] and MessageType.Parameter in msg.type:
            # If requested parameter, give parameters directly
            if self._dist_repr_type == 'parameter':
                dist_content = msg.parameter
            # Otherwise, translate to a PyTorch distribution instance
            else:
                dist_content = DistributionServer.param2dist(dist_class, msg.parameter, dist_info=dist_info)
        if self._repr_type in ['particle', 'dual'] and MessageType.Particles in msg.type:
            particle = msg.particles[0]
            weight = msg.weight
            log_density = msg.log_densities[0]
        if dist_content is not None and particle is not None:
            content_flag = 'dual'
        elif dist_content is not None:
            content_flag = 'distribution'
        else:
            content_flag = 'particle'

        # Calling callback
        return_val = self._sum_func(content_flag, dist_content, particle, weight, log_density)

        # Check return value
        if not isinstance(return_val, tuple) or not len(return_val) == 2:
            raise SummarizationValueError(
                "In summarization routine '{}': the value returned by the summarization callback routine should be a "
                "2-tuple. Found return value type {}."
                .format(self._sum_func.__name__, type(return_val))
            )
        summed_dist_content, summed_weight = return_val
        # Check return value type
        if summed_dist_content is not None and not isinstance(summed_dist_content, Distribution) and \
                not isinstance(summed_dist_content, torch.Tensor):
            raise SummarizationValueError(
                "In summarization routine '{}': the first return value should be either None, a torch distribution, or "
                "a torch tensor. Found type {}."
                .format(self._sum_func.__name__, type(summed_dist_content))
            )
        if isinstance(summed_dist_content, torch.Tensor) and self._dist_repr_type == 'distribution':
            raise SummarizationValueError(
                "In summarization routine '{}': when 'dist_repr_type' is 'distribution', the first return value from "
                "the callable should be a torch.distribution.Distribution. However, found a torch.Tensor."
                .format(self._sum_func.__name__)
            )
        if isinstance(summed_dist_content, Distribution) and self._dist_repr_type == 'parameter':
            raise SummarizationValueError(
                "In summarization routine '{}': when 'dist_repr_type' is 'parameter', the first return value from the "
                "callable should be a torch.Tensor. However, found a torch.distribution.Distribution."
                .format(self._sum_func.__name__)
            )
        if summed_weight is not None and not isinstance(summed_weight, torch.Tensor):
            raise SummarizationValueError(
                "In summarization routine '{}': the second return value should be either None or a torch Tensor. Found "
                "type {}."
                .format(self._sum_func.__name__, type(summed_weight))
            )
        # Check return value is not None when needed
        if content_flag in ['distribution', 'dual'] and summed_dist_content is None:
            raise SummarizationValueError(
                "In summarization routine '{}': found None as the first return value while input contains distribution "
                "contents."
                .format(self._sum_func.__name__)
            )
        if content_flag in ['particle', 'dual'] and summed_weight is None:
            raise SummarizationValueError(
                "In summarization routine '{}': found None as the second return value while input contains particle "
                "contents."
                .format(self._sum_func.__name__)
            )
        # Check returned content shape
        if isinstance(summed_dist_content, Distribution) and summed_dist_content.batch_shape != summed_b_shape:
            raise SummarizationValueError(
                "In summarization routine '{}': expect the returned distribution instance (first return value) to have "
                "batch shape {}. However, found batch shape {}."
                .format(self._sum_func.__name__, summed_b_shape, summed_dist_content.batch_shape)
            )
        if isinstance(summed_dist_content, torch.Tensor) and summed_dist_content.shape != summed_b_shape + msg.p_shape:
            raise SummarizationValueError(
                "In summarization routine '{}': expect the returned distribution parameter (first return value) to "
                "have shape {}. However, found shape {}."
                .format(self._sum_func.__name__, summed_b_shape + msg.p_shape, summed_dist_content.shape)
            )
        if isinstance(summed_weight, torch.Tensor) and summed_weight.shape != summed_b_shape + msg.s_shape:
            raise SummarizationValueError(
                "In summarization routine '{}': expect the returned particle weight (second return value) to have "
                "shape {}. However, found shape {}."
                .format(self._sum_func.__name__, summed_b_shape + msg.s_shape, summed_weight.shape)
            )

        # Translate distribution instance to distribution parameter, if present
        if isinstance(summed_dist_content, Distribution):
            summed_dist_content = DistributionServer.dist2param(summed_dist_content, dist_info)

        if content_flag == 'dual':
            return_msg = Message(MessageType.Dual,
                                 batch_shape=summed_b_shape, param_shape=msg.p_shape,
                                 sample_shape=msg.s_shape, event_shape=msg.e_shape,
                                 parameter=summed_dist_content,
                                 particles=msg.particles, weight=summed_weight, log_densities=msg.log_densities,
                                 device=msg.device,
                                 **msg.attr)
        elif content_flag == 'distribution':
            return_msg = Message(MessageType.Parameter,
                                 batch_shape=summed_b_shape, param_shape=msg.p_shape,
                                 parameter=summed_dist_content,
                                 device=msg.device,
                                 **msg.attr)
        else:
            return_msg = Message(MessageType.Particles,
                                 batch_shape=summed_b_shape, sample_shape=msg.s_shape, event_shape=msg.e_shape,
                                 particles=msg.particles, weight=summed_weight, log_densities=msg.log_densities,
                                 device=msg.device,
                                 **msg.attr)

        return return_msg


def Summarization(sum_func: _Callable_Signature = None,
                  repr_type: str = 'dual',
                  dist_repr_type: str = 'distribution'):
    """The wrapper around _Summarization class to handle decorator arguments.

    If the the decorator ``@Summarization`` is used without arguments, then `sum_func` would be the function object,
    so isinstance the _Summarization wrapper directly. Otherwise, we need to return a callable that will then be
    used as the actual decorator; therefore, we return a wrapper closure that defers the actual calling.

    Note that this means the decorator ``@Summarization`` can only accept keyword arguments.
    """
    if sum_func:
        return _Summarization(sum_func)
    else:
        def wrapper(function):
            return _Summarization(function, repr_type, dist_repr_type)
        return wrapper
