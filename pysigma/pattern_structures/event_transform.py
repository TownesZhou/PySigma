"""
    Constructs relating to the Event Transform pattern structure for mapping predicate random arguments to pattern
    random variables.

    Defines the following:
"""
from __future__ import annotations  # Postponed evaluation of annotations
from typing import Union, Callable
from typing import Iterable as IterableType
from collections.abc import Iterable
import torch
from torch.distributions.transforms import Transform
from ..defs import Variable, VariableMetatype


# region: Custom Error

# endregion

class EventTransform:
    """
        The Conditional pattern construct for declaring a mapping between random predicate argument(s) and random
        pattern variable, as well as optionally declaring a transformation on the particle events in a pattern element.

        There are two ways to provide predicate arguments and pattern variables in `pred_arg` and `pat_var`: either
        using `Variable` instances or variable names as string. If using the string representation of a variable, the
        actual`Variable` instance will be determined by the Conditional structure later. If using a `Variable` instance,
        its variable metatype must be `Random`, since this construct associates mapping between random variables.

        `pred_arg` can be specified as a single `Variable` instance or `str`, or an iterable of `Variable` instances
        and `str`. The former means to associate this single predicate argument with `pat_var`, while the latter means
        to associate the **concatenation** of the variables given in `pred_arg` with `pat_var`. In the latter case,
        `pred_arg` as a list will be treated as a single variable, and their corresponding particle events will be
        concatenated in the order given by the list.

        To specify a event transformation, a `torch.distributions.transforms.Transform` instance must be specified in
        the `transform` argument. In addition, `forward` argument can be specified to indicate the direction of the
        transformation. In other words, if ``forward=True``, then it is assumed that ``transform(pred_arg) = pat_var``.
        Otherwise, it is assumed that ``pred_arg = transform(pat_var)``.

        Note that it is always assumed that the transformation is invertible.

        In any group of inner-pattern or inter-pattern bindings, one and only one of the pattern element branch must be
        declared as the "master" branch that sends its particle events to every other pattern element branch in this
        group for them to carry out Importance reweighting. This shall be done by setting `master` argument to True to
        signify that this pattern element is the master branch.

        Parameters
        ----------
        pred_arg : Iterable of Variable or str, or Variable, or str
            The predicate argument(s). If a single variable, it stands for that variable. If an iterable of variables,
            it stands for the concatenation of those variables. If `Variable` instances are provided, type checks and
            size checks will be carried out.
        pat_var : Variable or str
            The pattern variable. Must be a `Variable` instance or a str. If a `Variable` instance, type check and size
            check will be carried out.
        transform : torch.distributions.transforms.Transform, optional
            The optional transformation on the particle events. Defaults to None, meaning identity transformation.
        forward : bool, Optional
            The direction of the transformation. Defaults to True.
        master : bool, Optional
            Indicates whether this branch is the master branch in a variable binding that sends its particle events.
            Only has an effect on the compiled graph if this pattern element indeed binds with other pattern element
            on the pattern variable specified by `pat_var`.

        Raises
        ------
        AssertionError
            If the provided arguments do not meet the requirements.
    """

    def __init__(self,
                 pred_arg: Union[IterableType[Variable], Variable, IterableType[str], str],
                 pat_var: Union[Variable, str],
                 transform: Transform = None,
                 forward: bool = True,
                 master: bool = False):
        # Validate arguments
        assert isinstance(pred_arg, Iterable) or isinstance(pred_arg, (Variable, str))
        if isinstance(pred_arg, Iterable):
            assert all(isinstance(pa, (Variable, str)) for pa in pred_arg)
        assert isinstance(pat_var, (Variable, str))
        assert transform is None or isinstance(transform, Transform)
        assert isinstance(forward, bool)
        assert isinstance(master, bool)

        self.pred_arg = tuple(pred_arg) if isinstance(pred_arg, Iterable) else pred_arg
        self.pat_var = pat_var
        self.transform = transform
        self.forward = forward
        self.master = master

        # Check type for any Variable instance
        if isinstance(self.pred_arg, Variable):
            assert self.pred_arg.metatype is VariableMetatype.Random, \
                "EventTransform structure only accepts random variables. Found `pred_arg` variable metatype: {}" \
                .format(self.pred_arg.metatype)
        elif isinstance(self.pred_arg, Iterable):
            for pa in self.pred_arg:
                assert not isinstance(pa, Variable) or pa.metatype is VariableMetatype.Random, \
                    "EventTransform structure only accepts random variables. Found one of the variables in `pred_arg` "\
                    "has variable metatype: {}".format(pa.metatype)
        if isinstance(self.pat_var, Variable):
            assert self.pat_var.metatype is VariableMetatype.Random, \
                "EventTransform structure only accepts random variables. Found `pat_var` variable metatype: {}" \
                    .format(self.pat_var.metatype)
        # Check variable size if already finalized
        if self.finalized:
            if isinstance(self.pred_arg, Variable):
                assert self.pred_arg.size == self.pat_var.size, \
                    "The variable size of `pred_arg` and `pat_var` must be the same. Found `pred_arg` has size {} and "\
                    "`pat_var` has size {}.".format(self.pred_arg.size, self.pat_var.size)

    @property
    def finalized(self):
        """
            If both `pred_arg` and `pat_var` are `Variable` instances. This means that any string representation of a
            Variable has been replaced with an actual `Variable` instance.
        """
        return (isinstance(self.pred_arg, Variable) or
                (isinstance(self.pred_arg, Iterable) and all(isinstance(pa, Variable) for pa in self.pred_arg))) and \
               isinstance(self.pat_var, Variable)
