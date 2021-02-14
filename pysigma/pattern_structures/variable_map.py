"""
    The VariableMap pattern structure. Defines the VariableMap decorator class and wrapper to be used to decorate a
        a relational variable mapping callback routine.
"""
from typing import Union, Optional, Tuple, Set, Callable
from typing import Iterable as IterableType
from collections.abc import Iterable
import functools
import numpy as np
import torch
from torch.distributions import Distribution
from ..utils import DistributionServer
from ..defs import Message, MessageType


class VariableMapError(Exception):
    """
        Superclass for all VariableMap-related exceptions.
    """
    pass


class VariableMapInitializationError(VariableMapError):
    """
        Exception related to the initialization of the VariableMap decorator.
    """
    pass


class VariableMapClass:
    """Class type for declaring mappings on relational variables in pattern elements. Wraps around a user-defined
    mapping function and records the one-to-one correspondence between inputs and outputs.

    Since relational variables can be viewed as non-negative finite integer-valued variables, a VariableMap instance
    therefore declares an integer-valued mapping with finite domain.

    The domain and codomain are assumed fixed, so they should be provided during initialization. The mapping can
    be computed either lazily at runtime or pre-computed at initialization. This flexibility is to allow dynamic
    mappings such as neural attention modules.

    Only injective mapping can be accepted. This is because the semantic of mapping two relational variable's values
    to a single value is ill-defined.

    Parameters
    ----------
    mapping_func : callable
        a user-specified function. Takes a numpy array of integers as input, and returns a numpy array of integers of
        the same size. Each entry in the output array corresponds to the value of `f(x)` of input `x` at the same index
        in the input array.
    domain : set of int
        The domain of the mapping.
    codomain : set of int
        The codomain of the mapping.
    dynamic : bool, optional
        Indicates whether the mapping is dynamic. If ``True``, then `mapping_func` will be called each time a mapping
        dictionary is desired, i.e., when `get_map()` or `get_inverse_map()` is called. Otherwise, `mapping_func` will
        only be called once during initialization of this VariableMap instance, and the result will cached. Defaults
        to ``False``.

    Attributes
    ----------
    mapping_func
    domain
    codomain
    dynamic
    map : dict
        The mapping cache.
    image : set
        Image of the map. Should be a subset of ``self.codomain``.
    """
    def __init__(self, mapping_func: Callable, domain: Set[int], codomain: Set[int], dynamic: bool):

        self.mapping_func = mapping_func
        self.domain = domain
        self.codomain = codomain
        self.dynamic = dynamic
        # mapping cache. This field will be looked up later as definition of the mapping
        self.map_cache = None
        # The image of the map, i.e., the set of values who has a corresponding input. It should be a subset of codomain
        self.image_cache = None

        # Set the map cache if not dynamic
        if not dynamic:
            self.set_map()

    def __call__(self, inverse_map=False, get_new_map=True):
        pass

    def _set_map(self):
        """Set ``self.map`` by obtaining the mapping dictionary from ``self.mapping_func``.

        Raises
        ------
        ValueError
            If `self.mapping_func` does not return a numpy array or returns one with wrong data type.
        ValueError
            If ``self.mapping_func`` does not return a numpy array with the same size as the input array.
        ValueError
            If ``self.mapping_func`` contains values not in the codomain.
        ValueError
            If ``self.mapping_func`` is found to be not injective.
        """
        # Input list
        input = np.array(list(self.domain))
        output = self.mapping_func(input)
        # Check output format and if its entries are in codomain range
        if not isinstance(output, np.ndarray) or not all(isinstance(i, np.int64) and i >= 0 for i in output):
            raise ValueError("The provided mapping python callable should return a numpy array of non-negative "
                             "np.int64. Instead, found: '{}'".format(output))
        if input.size != output.size:
            raise ValueError("The output list from the provided mapping python callable should be of the same size as "
                             "the input list. Expecting size '{}', instead found size '{}'"
                             .format(len(input), len(output)))
        for i in output:
            if i not in self.codomain:
                raise ValueError("The output from the provided mapping python callable should be within the specified "
                                 "codomain: '{}'. Instead, found entry in the output list: '{}'"
                                 .format(self.codomain, i))
        # Set mapping
        self.map = dict(zip(input, output))
        # Set mapping image
        self.image = set(self.map.values())
        # Check injectivity
        if len(self.image) != len(self.domain):
            raise ValueError("The specified mapping should be injective. However, found that the cardinality of the "
                             "mapping's image is '{}', whereas the cardinality of the specified domain is '{}'"
                             .format(len(self.image), len(self.domain)))

    def _get_map(self):
        """Returns the mapping dictionary, the map's domain, and the map's image.

        If dynamic, then calls `set_map()` to re-compute the dict first, otherwise returns the cached one.

        Returns
        -------
        tuple
            A 3-tuple containing the data ``(map_dict, domain, image)``.
        """
        if self.dynamic:
            self.set_map()
        return self.map, self.domain, self.image

    def _get_inverse_map(self):
        """Returns the inverse map's mapping dictionary, the inverse map's domain (original map's image), and the
        inverse map's image (should be the same as the original map's domain)

        Note that because injectivity is guaranteed, computing an inverse map is possible.

        If dynamic, then calls set_map() to re-compute the dict first, otherwise returns the cached one.

        Returns
        -------
        tuple
            A 3-tuple containing the data ``(inv_map_dict, inv_image, inv_domain)``.
        """
        if self.dynamic:
            self.set_map()

        inverse_map = dict(zip(self.map.values(), self.map.keys()))
        return inverse_map, self.image, self.domain


def VariableMap(domain: IterableType[int], codomain: IterableType[int], dynamic: bool = False):
    """
        The actual @VariableMap decorator. To use this decorator, the arguments `domain` and `codomain` must be passed
        to parameterize the decorator. This function then returns a wrapper closure that creates a a VariableMapClass
        instance that actually wraps the user-defined callback routine.

        Injectivity will be check by comparing the cardinality of `domain` and `codomain`. If the check fails, a
        VariableMapInitializationError will be raised.
        
        Parameters
        ----------
        domain : Iterable of int
            Specifies the domain of the mapping.
        codomain : Iterable of int
            Specifies the codomain of the mapping.
        dynamic : bool, optional
            Whether the mapping is dynamic or not. If dynamic, the user-defined callback would be called once during 
            each cognitive cycle. Otherwise, it would only be called once during initialization, and the results would
            be cached and reused throughout the model's lifecycle. Defaults to False.
    """
    assert isinstance(domain, Iterable) and all(isinstance(v, int) for v in domain)
    assert isinstance(codomain, Iterable) and all(isinstance(v, int) for v in codomain)
    assert isinstance(dynamic, bool)

    domain, codomain = set(domain), set(codomain)
    # Check validity as well as injectivity
    if len(domain) != len(codomain):
        raise VariableMapInitializationError("The domain and codomain have different cardinality. Please specify a "
                                             "domain and codomain with the same cardinality (length). Found a domain "
                                             "with cardinality {}, and a codomain with cardinality {}."
                                             .format(len(domain), len(codomain)))

    # Return wrapper that instantiate the VariableMapClass instance
    def wrapper(function):
        return VariableMapClass(function, domain, codomain, dynamic)

    return wrapper


