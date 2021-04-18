"""
    Constructs relating to the VariableMap pattern structure. Defines the following:
        - VariableMap decorator:
            The decorator function for user to wrap the callback function. It is the top-level construct of the three.
            Returns a VariableMapPrototype instance that wraps the user callback.
        - VariableMapPrototype class
            The intermediate construct for user to use in a pattern structure. It is a callable that accepts two
            pattern relational variables and returns a VariableMapCallable instance.
        - VariableMapCallable class
            The bottom-level construct to be passed to the pattern structure and eventually to the nodes. It is a
            callable used by the RMFN that actually calls the user callback and checks return value correctness.
"""
from __future__ import annotations      # Postponed evaluation of annotations
from typing import Union, Optional, Tuple, Set, Callable
from typing import Iterable as IterableType
from collections.abc import Iterable
import functools
import numpy as np
import torch
from torch.distributions import Distribution
from ..utils import DistributionServer
from ..defs import Message, MessageType, Variable


# region: Custom Error
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


class VariableMapRuntimeError(VariableMapError):
    """
        Exception related to running the user-defined callback
    """
    pass


class VariableMapValueError(VariableMapError):
    """
        Exception related to the return value of the user-defined callback
    """
    pass
# endregion


# region: module-level global constant
DYNAMIC_DEFAULT_VAL = True
STRICT_DEFAULT_VAL = True


def VariableMap(dynamic: bool = DYNAMIC_DEFAULT_VAL, strict: bool = STRICT_DEFAULT_VAL):
    """
        The top-level decorator used by user to decorate a custom callback function. The returned object, i.e., the
        decorated function, will be a VariableMapPrototype callable instance that accepts two Variable instances and
        returns a VariableMapCallable instance. In other words, once the user-defined function is decorated, the user
        should use it in a conditional's pattern structure to wrap the predicate argument and the pattern variable that
        the user intend to bind. The generated VariableMapCallable instance will be handled by the compiler and passed
        to the underlying graphical level for computation during message decision phase.

        The VariableMap defines two parameter: `dynamic` and `static`. If `dynamic = True`, then the user callback
        will be called exactly once during each cognitive cycle's decision phase for *each* wrapped pattern variable.
        Otherwise, it would only be called once during initialization for *each* wrapped pattern variable, and the
        cached results will be used throughout the model's lifecycle. If `static = True`, then a
        ``VariableMapValueError`` exception will be raised if the returned values from the user callback exceeds the
        corresponding predicate argument's domain of the pattern predicate. Otherwise, this scenario will be
        handled silently by ignoring the exceptional values, and by default the outgoing message at the said exceptional
        slots will be the identity messages.

        Parameters
        ----------
        dynamic : bool, optional
            Decide if the user callback be called only once during initialization, or be dynamically called during
            each cognitive cycle.
        strict : bool, optional
            Decide if domain check of the user callback's return value will be strictly carried out.
    """
    if not (isinstance(dynamic, Callable) or isinstance(dynamic, bool)) or not isinstance(strict, bool):
        raise ValueError("Unrecognizable argument type for the VariableMap decorator: {}"
                         .format([type(dynamic), type(strict)]))

    if isinstance(dynamic, Callable):
        # If the user uses this decorator without parentheses, then `dynamic` will actually be the user-defined
        #   callback.
        # In this case, directly return a VariableMapPrototype instance.
        # Here `dynamic` is the actual callback
        return VariableMapPrototype(dynamic, DYNAMIC_DEFAULT_VAL, STRICT_DEFAULT_VAL)
    elif isinstance(dynamic, bool):
        # Otherwise if decorator is parameterized, return a wrapper instantiate the VariableMapPrototype instance
        def wrapper(function):
            # Here `dynamic` and `strict` are actually the intended decorator parameters
            return VariableMapPrototype(function, dynamic, strict)
        return wrapper


class VariableMapPrototype:
    """
        The intermediate callable construct used by user to bind the target predicate argument and pattern variable in a
        conditional's pattern structure and associate the binding with a mapping. As a callable it returns itself, but
        the `variable_map` property would return a VariableMapCallable instance if both the registered predicate
        argument and pattern variable are Variable instances.

        Note that it is viable for the user to provide a string as the variable representation instead of a Variable
        instance. In this case, the VariableMapCallable instance would not be generated and getting the `variable_map`
        property would raise an assertion error.
    """
    def __init__(self, func: Callable, dynamic: bool, strict: bool):
        self.func = func
        self.dynamic = dynamic
        self.strict = strict
        self._pred_arg = None
        self._pat_var = None
        self._variable_map = None

    @property
    def pred_arg(self) -> Union[Variable, str]:
        """
            Get the registered predicate argument
        """
        return self._pred_arg

    @property
    def pat_var(self) -> Union[Variable, str]:
        """
            Get the registered pattern variable
        """
        return self._pat_var

    @property
    def variable_map(self) -> VariableMapCallable:
        """

        """
        assert self._variable_map is not None, "Predicate argument and/or pattern variable are not finalized."
        return self._variable_map

    def __call__(self, pred_arg: Union[Variable, str], pat_var: Union[Variable, str]) -> VariableMapPrototype:
        """
            Called within a conditional's pattern structure to wrap a pattern variable. The pattern variable will be
            passed to the returning VariableMapCallable instance.
        """
        self._pred_arg = pred_arg
        self._pat_var = pat_var

        # Update and the VariableMapCallable instance if variables are finalized
        if isinstance(self._pred_arg, Variable) and isinstance(self._pat_var, Variable):
            self._variable_map = VariableMapCallable(self.func, self._pred_arg, self._pat_var, self.dynamic, self.strict)
        else:
            self._variable_map = None

        return self


class VariableMapCallable:
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
    def __init__(self, mapping_func: Callable, pred_arg: Variable, pat_var: Variable, dynamic: bool, strict: bool):
        # Update wrapped function's information to enable introspection
        functools.update_wrapper(self, mapping_func)
        self.mapping_func = mapping_func
        self.dynamic = dynamic
        self.strict = strict
        self.pred_arg = pred_arg
        self.pat_var = pat_var
        # Flag indicating if it's a new cycle and the callback shall be called for the first time, or if the callback
        #   has been called previously and the cached result shall be used.
        # `dynamic` field overwrites this field
        self._new_call = True
        # mapping cache. This field will be looked up later
        self.result_cache = None

        # If not dynamic, call the user callback for once during initialization
        if not dynamic:
            self._set_map()

    @property
    def new_call(self):
        """
            Indicate whether the next invocation to this callable would actually call the user-provided callback
            rather than using the cached result
        """
        return self.dynamic and self._new_call

    @new_call.setter
    def new_call(self, value):
        """
            Set `_new_call` value to indicate whether it is currently a new cycle and this callback has not been called.
        """
        self._new_call = value

    def __call__(self) -> (torch.LongTensor, torch.LongTensor):
        """
            Returns the mapping between the pattern variable values and the predicate argument values.

            The first return value is the pattern variable values, the second one the mapped predicate argument values.

            Note that value check of the user callback is carried out herein, so that it is guaranteed that the return
            values are within the valid range. However, this means that the return values may not fully cover the
            pattern variable and/or the predicate argument's domain. Certain values in the domain may be missing in
            this mapping because either the mapping is not surjective or some pattern variable value would be mapped to
            illegal predicate argument value.

            Returns
            -------
            (torch.LongTensor, torch.LongTensor)
                The pattern variable index values and the mapped predicate argument index values:
                `(pat_var_indices, pred_arg_indices)`.
        """
        # Call user callback if new call is needed
        if self.new_call:
            self._set_map()
        # Set the status to tell I have been called during current cognitive cycle
        self.new_call = False
        # Return cache
        return self.result_cache

    def _set_map(self):
        """
            Set ``self._result_cache`` by calling the user-provided callback.
        """
        # Obtain input values from given pattern variable
        input_val = torch.arange(self.pat_var.size)

        # Call callback
        try:
            output_val: torch.LongTensor = self.mapping_func(input_val)
        except Exception as exc:
            # Chain the exception
            raise VariableMapRuntimeError("In VariableMap mapping pattern variable '{}' to predicate argument '{}' "
                                          "with callback '{}': unable to proceed. An exception internal to the "
                                          "callback occurred."
                                          .format(self.pat_var, self.pred_arg, self.mapping_func.__name__)) from exc

        # Check return value type
        if not isinstance(output_val, torch.LongTensor):
            raise VariableMapValueError("In VariableMap mapping pattern variable '{}' to predicate argument '{}' with "
                                        "callback '{}': the return value from the custom callback must be a "
                                        "LongTensor. Found return value type: '{}'"
                                        .format(self.pat_var, self.pred_arg, self.mapping_func.__name__,
                                                type(output_val)))
        # Check return LongTensor shape
        if output_val.shape != input_val.shape:
            raise VariableMapValueError("In VariableMap mapping pattern variable '{}' to predicate argument '{}' with "
                                        "callback '{}': the returned LongTensor must have the same shape as the input "
                                        "LongTensor. Expect shape {}, but found shape {}."
                                        .format(self.pat_var, self.pred_arg, self.mapping_func.__name__,
                                                input_val.shape, output_val.shape))

        # Check if return values are out of range w.r.t. the predicate argument's domain
        # Raise exception if strict, otherwise remove corresponding values from input and output
        valid_mask = torch.bitwise_and(0 <= output_val, output_val < self.pred_arg.size)
        if not torch.all(valid_mask):
            if self.strict:
                raise VariableMapValueError("In VariableMap mapping pattern variable '{}' to predicate argument '{}' "
                                            "with callback '{}': returned LongTensor value out of range. Expect values "
                                            "in range [0, {}], but found maximum value {} and minimum value {}."
                                            .format(self.pat_var, self.pred_arg, self.mapping_func.__name__,
                                                    self.pred_arg.size - 1,
                                                    output_val.max().item(), output_val.min().item()))
            else:
                # Exclude exceptional values from input and output tensor
                input_val = input_val.masked_select(valid_mask)
                output_val = output_val.masked_select(valid_mask)

        # Cache the result
        self.result_cache = (input_val, output_val)
