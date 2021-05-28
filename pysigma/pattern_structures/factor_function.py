"""
    Factor Function structure in a Conditional.
"""
from __future__ import annotations  # Postponed evaluation of annotations
from typing import Union, Callable, Tuple, Optional
from typing import Iterable as IterableType
from collections.abc import Iterable
import torch
from torch.nn import Module
from ..defs import Variable, VariableMetatype
from ..utils import equal_constraints


# Custom Errors
class FactorFunctionError(Exception):
    """
        Superclass for all factor function related errors.
    """
    pass


class FactorFunctionRuntimeError(FactorFunctionError):
    """
        Error related to running callback.
    """
    pass


class FactorFunctionValueError(FactorFunctionError):
    """
        Error related to the returning value of the callback.
    """
    pass


class FactorFunction:
    """
        Factor function in a Conditional.

        A FactorFunction in general is a function that takes as input a tuple of batched tensors, each tensor
        corresponding to a batch of value assignments to a random variable. It outputs a single vector that represents
        the batched joint probability density associated with the input tuple of tensors, i.e.::

            val = Prob(X_1, X_2, ..., X_n)

        where ``X_1, X_2, ..., X_n`` are 2-dimensional tensors with the first dimension being the batch dimension and
        the second dimension the event dimension of their corresponding random variable. And ``val`` is a 1-dimensional
        vector, with a vector size equal to the batch size of ``X_1, X_2, ..., X_n``.

        Above is the general form generative factor function. Alternatively, a factor function can also be specified as
        a list of discriminative functions generating particles for each of the output random variables::

            X_1 = F_1(X_2, X_3, ..., X_n)
            X_2 = F_2(X_1, X_3, ..., X_n)
            ...
            X_n = F_n(X_1, X_2, ..., X_(n-1))

        This discriminative form is useful when deterministic functional relationships is easy to establish between
        the random variables.

        Note that the generative form and the discriminative form are not mutually exclusive to each other when
        defining a factor function. The can be declared together so that for some random variables, the probability
        densities are computed, whereas for others, the particle values are computed. However, it is required that
        if all random variables have corresponding discriminative function (meaning the output of the function is for
        that random variable), then the generative form is not specified, because in this case the generative form
        will be entirely replaced by the list of discriminative functions.

        For both forms of factor functions, the first dimension of all input tensors to the factor function callback
        will index the sample dimension, which should not be modified by the user-defined callback routine. For
        the generative factor function, the callback shall return a 1-dimensional vector with non-negative values
        representing the joint probability densities. This vector shall have a same size as the input tensors' sample
        size. For the discriminative factor function, the callback shall return a 2-dimensional tensor, with the
        first dimension's size equal to the input tensors' sample size and the second dimension's size equal to the
        corresponding output random variable's event size.




        Parameters
        ----------
        ran_vars : Iterable of Variables or str
            The random variables of the factor function. It can consist of Variable instances or string representations
            of the variables.
        prob_func : torch.nn.Module, optional
            The generative joint probability density factor function. Must be a torch nn module. Shall be left as None
            if every random variable has a corresponding discriminative function declared in `vector_func`. Defaults to
            None.
        vector_func : Iterable of torch.nn.Module or None, optional
            The tuple of discriminative factor functions. Must have the same length as the number of random
            variables in `ran_vars`. If a random variable is not to be associated with a vector function, then None
            should be declared in the corresponding position in `vector_func`. Defaults to None.
        device : int, str, or torch.device
            The device on which the factor function is run. The input tensors to the factor function callbacks will
            first be converted to this device. Defaults to ``'cpu'``.
    """

    def __init__(self,
                 ran_vars: IterableType[Union[Variable, str]],
                 prob_func: Module = None,
                 vector_func: Iterable[Optional[Module]] = None,
                 device: Union[int, str, torch.device] = 'cpu'):
        # Validate arguments
        assert isinstance(ran_vars, Iterable) and all(isinstance(v, Variable) or isinstance(v, str) for v in ran_vars),\
            "`ran_vars` must be an iterable of Variable instances or the string representations of Variable instances."
        ran_vars = tuple(ran_vars)
        assert len(ran_vars) > 0, \
            "`ran_vars` must be an iterable with at least one item. "
        assert all(isinstance(v, str) or v.metatype is VariableMetatype.Random for v in ran_vars), \
            "All variables in `ran_vars` must be Random Variables."
        assert prob_func is None or isinstance(prob_func, Module), \
            "If specified, `prob_func` must be a torch nn module."
        assert vector_func is None or (isinstance(vector_func, Iterable) and
                                       all(vf is None or isinstance(vf, Module) for vf in vector_func)), \
            "If specified, `vector_func` must be an iterable consisting of None and/or torch nn module."

        if vector_func is not None:
            assert len(ran_vars) > 1, \
                "If `vector_func` is specified, `ran_vars` must contain at least two Variables."
            assert any(isinstance(vf, Module) for vf in vector_func), \
                "If specified, at least one element in `vector_func` must be a torch nn module."
            vector_func = tuple(vector_func)
            assert len(vector_func) == len(ran_vars), \
                "`vector_func` must have the same length as `ran_vars`. Expect length {}, instead found {}." \
                .format(len(ran_vars), len(vector_func))
        # Check that at least one type function is declared
        assert prob_func is not None or vector_func is not None, \
            "At least one of `prob_func` or `vector_func` must be specified."
        # # Check that if prob_func is not declared, vector_func must be fully specified
        # if prob_func is None:
        #     assert all(vf is not None for vf in vector_func), \
        #         "When `prob_func` is not specified, `vector_func` must be fully specified with torch nn modules."
        # Check that if prob_func is declared, vector_func is not fully specified.
        if prob_func is not None:
            assert vector_func is None or any(vf is None for vf in vector_func), \
                "When `prob_func` is specified, `vector_func` shall not be fully specified."

        self.ran_vars: Tuple[Union[Variable, str]] = ran_vars
        self._prob_func = prob_func
        self._vector_func = vector_func
        self._device = torch.device(device)

        self._directional_vars = None

    @property
    def finalized(self) -> bool:
        """
            Boolean property. True if all variables are Variable instances instead of string representation of
            variables.

            Used by Gamma Factor Node.
        """
        return all(isinstance(v, Variable) for v in self.ran_vars)

    @property
    def generative(self) -> bool:
        """
            Boolean property. True if either prob_func or exp_func is declared. False otherwise.

            Used by Gamma Factor Node.
        """
        return self._prob_func is not None

    @property
    def discriminative(self):
        """
            Boolean property. True if **any** declarative vector function. False otherwise.

            Note: since a prob_func or a exp_func can be declared along with vector_func, a factor function can be
            both generative and discriminative. However, if generative, a factor function can only be partially
            discriminative, meaning that some random variable has no corresponding vector function.

            Used by Gamma Factor Node.
        """
        return self._vector_func is not None and len(self.directional_vars) > 0

    @property
    def directional_vars(self) -> Tuple[Variable]:
        """
            Returns the tuple of random variables for which a vector function is declared. If no vector function is
            declared, returns an empty tuple.

            Used by Gamma Factor Node.
        """
        assert self.finalized
        if self._directional_vars is None:
            self._directional_vars = \
                tuple(rv for i, rv in enumerate(self.ran_vars) if self._vector_func[i] is not None) \
                if self._vector_func is not None else ()
        return self._directional_vars

    def __call__(self, inputs: IterableType[torch.Tensor], target_var: Variable = None, func_type: str = 'generative',
                 node_device: torch.device = 'cpu') -> torch.Tensor:
        """
            Call the wrapped factor function.

            Used by Gamma Factor Node.

            Parameters
            ----------
            inputs : Iterable of torch.Tensor
                The input tensors. Must have the same length as the number of random variables. The first dimension
                (the batch dimension) of each input tensor must have the same size.
            target_var : Variable, optional
                The target/output variable. Used to identify which one of the discriminative functions is to be called.
                Should be specified when `func_type` is ``"generative"``. When `func_type` is ``"discriminative"``, this
                value will be ignored.
            func_type : str, optional
                The function type. ``"generative"`` or ``"discriminative"``. Defaults to ``"generative"``.
            node_device : torch.device, optional
                The calling node's device. Output tensor from factor function callback will be converted to this device.

            Returns
            -------
            torch.Tensor
                If `func_type` is ``"generative"``, then returns a 1-dimensional vector representing the joint
                probability densities. Otherwise if `func_type` is ``"discriminative"``, returns a 2-dimensional tensor
                representing the output particles for the specified `target_var`.
        """
        # Validate arguments
        assert self.finalized, "This factor function construct is not yet finalized."
        assert isinstance(inputs, Iterable) and all(isinstance(x, torch.Tensor) for x in inputs)
        inputs = tuple(inputs)
        assert len(inputs) == len(self.ran_vars)
        b_size = inputs[0].shape[0]
        assert all(b_size == x.shape[0] for x in inputs[1:]), \
            "Input tensors do not have the same batch dimension (first dimension) size."
        assert target_var is None or isinstance(target_var, Variable)
        assert isinstance(func_type, str) and func_type in ['generative', 'discriminative']

        # Convert input tensor to the target device
        inputs = tuple(x.to(device=self._device) for x in inputs)
        # Generative function
        if func_type == 'generative':
            assert self.generative
            # Call the wrapped function
            try:
                prob_dens: torch.Tensor = self._prob_func(*inputs)
            except Exception as exc:
                # Chain the exception
                raise FactorFunctionRuntimeError(
                    "An exception occurred when calling the generative joint probability density factor function {}."
                    .format(self._prob_func)
                ) from exc
            # Validate result
            if not isinstance(prob_dens, torch.Tensor):
                raise FactorFunctionValueError(
                    "The returning value from a factor function callback must be a torch tensor. Found value with "
                    "type {}.".format(type(prob_dens))
                )
            # Result from a generative callback must represent probability density. Thus must be 1-dimensional with the
            #   same size as the batch size
            if not prob_dens.shape == torch.Size([b_size]):
                raise FactorFunctionValueError(
                    "The returning tensor from a factor function must represent batched probability densities. Expect "
                    "shape {}, instead found shape {}.".format(torch.Size([b_size]), prob_dens.shape)
                )
            # Values must be non-negative
            if not torch.all(prob_dens >= 0):
                raise FactorFunctionValueError(
                    "The returning tensor from a factor function must represent batched probability densities. Expect "
                    "all values in the tensor be non-negative, instead found minimum value {}.".format(prob_dens.min())
                )

            # Convert result to node device and return
            prob_dens = prob_dens.to(device=node_device)
            return prob_dens

        # Declarative function
        else:
            assert self.discriminative
            # Check that the discriminative function for the target_var is defined
            assert target_var is not None and target_var in self.directional_vars, \
                "The target_var '{}' is not one of the directional variables {} for which a discriminative function " \
                "is specified." \
                .format(target_var, self.directional_vars)
            target_var_idx = self.ran_vars.index(target_var)
            vector_func = self._vector_func[target_var_idx]
            # Remove the input tensor that corresponds to this random variable
            discriminative_inputs = inputs[:target_var_idx] + inputs[target_var_idx + 1:]
            # Call wrapped function
            try:
                result: torch.Tensor = vector_func(*discriminative_inputs)
            except Exception as exc:
                raise FactorFunctionRuntimeError(
                    "An exception occurred when calling the discriminative factor function {} corresponding to "
                    "the random variable {}.".format(vector_func, target_var)
                ) from exc
            # Validate result
            # Check return value is a torch tensor
            if not isinstance(result, torch.Tensor):
                raise FactorFunctionValueError(
                    "The returning value from a factor function callback must be a torch tensor. Found value "
                    "with type {} in the discriminative factor function {} corresponding to the random "
                    "variable {}."
                    .format(type(result), vector_func, target_var)
                )
            # Resulting tensor from discriminative callback must be 2-dimensional batched particles
            if not result.ndim == 2:
                raise FactorFunctionValueError(
                    "The returning tensor from a discriminative callback must represent batched particles. "
                    "Expect a 2-dimensional tensor. Instead, found a tensor with {} dimensions in the "
                    "discriminative factor function {} corresponding to the random variable {}."
                    .format(result.ndim, vector_func, target_var)
                )
            if not result.shape == torch.Size([b_size, target_var.size]):
                raise FactorFunctionValueError(
                    "The returning tensor from a discriminative callback must represent batched particles. "
                    "Expect a tensor with shape {}. Instead, found a tensor with shape {} in the "
                    "discriminative factor function {} corresponding to the random variable {}."
                    .format(torch.Size([b_size, target_var.size]), result.shape, vector_func, target_var)
                )
            # Convert result to node device and add to returning tuple
            result = result.to(device=node_device)

            # Return
            return result
