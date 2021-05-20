"""
    Constructs relating to the Event Transform pattern structure for mapping predicate random arguments to pattern
    random variables.

    Defines the following:
"""
from __future__ import annotations  # Postponed evaluation of annotations
from typing import Union, Tuple, Optional
from typing import Iterable as IterableType
from collections.abc import Iterable
from torch.distributions.transforms import Transform
from ..defs import Variable, VariableMetatype
from ..utils import equal_constraints

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
        assert isinstance(pred_arg, (str, Variable, Iterable))
        if not isinstance(pred_arg, str) and isinstance(pred_arg, Iterable):
            assert len(tuple(pred_arg)) > 1
            assert all(isinstance(pa, (Variable, str)) for pa in pred_arg)
        assert isinstance(pat_var, (Variable, str))
        assert transform is None or isinstance(transform, Transform)
        assert isinstance(forward, bool)
        assert isinstance(master, bool)

        self.pred_arg: Union[str, Variable, Tuple[Variable]] = tuple(pred_arg) \
            if isinstance(pred_arg, Iterable) and not isinstance(pred_arg, str) \
            else pred_arg
        self.pat_var: Union[str, Variable] = pat_var
        self.transform = transform
        self.forward = forward
        self.master = master

        # Check type for any Variable instance
        if isinstance(self.pred_arg, Variable):
            assert self.pred_arg.metatype is VariableMetatype.Random, \
                "EventTransform structure only accepts random variables. Found `pred_arg` variable metatype: {}" \
                .format(self.pred_arg.metatype)
        elif isinstance(self.pred_arg, tuple):
            for pa in self.pred_arg:
                assert not isinstance(pa, Variable) or pa.metatype is VariableMetatype.Random, \
                    "EventTransform structure only accepts random variables. Found one of the variables in `pred_arg` "\
                    "has variable metatype: {}".format(pa.metatype)
        if isinstance(self.pat_var, Variable):
            assert self.pat_var.metatype is VariableMetatype.Random, \
                "EventTransform structure only accepts random variables. Found `pat_var` variable metatype: {}" \
                .format(self.pat_var.metatype)
        if self.finalized:
            if isinstance(self.pred_arg, tuple):
                # If multiple pred args, check that size sum equal to the pat var size
                pred_arg_sum_size = sum([arg.size for arg in self.pred_arg])
                assert pred_arg_sum_size == self.pat_var.size, \
                    "The variable size of `pred_arg` and `pat_var` must be the same. Found multiple predicate " \
                    "arguments in `pred_arg` have a total size of {}, but `pat_var` has size {}." \
                    .format(pred_arg_sum_size, self.pat_var.size)
                # Check that if multiple predicate arguments, all predicate arguments have the same constraints
                c1_cstr_list = self.pred_arg[0].constraints
                for other_arg in self.pred_arg[1:]:
                    other_cstr_list = other_arg.constraints
                    assert len(c1_cstr_list) == len(other_cstr_list) and all(equal_constraints(c1, c2) for c1, c2 in
                                                                             zip(c1_cstr_list, other_cstr_list)), \
                        "If multiple predicate arguments declared in a pattern, all of them must have the same value " \
                        "constraints. Found argument '{}' has constraints {}, but argument '{}' has constraints {}." \
                        .format(self.pred_arg[0], c1_cstr_list, other_arg, other_cstr_list)
            else:
                # If single pred arg, check that its size equal to pat var size
                assert self.pred_arg.size == self.pat_var.size, \
                    "The variable size of `pred_arg` and `pat_var` must be the same. Found `pred_arg` has size {} but " \
                    "`pat_var` has size {}.".format(self.pred_arg.size, self.pat_var.size)


        # Surrogate predicate argument
        self._surrogate_pred_arg: Optional[Variable] = None

    @property
    def finalized(self) -> bool:
        """
            If both `pred_arg` and `pat_var` are `Variable` instances. This means that any string representation of a
            Variable has been replaced with an actual `Variable` instance.
        """
        return (isinstance(self.pred_arg, Variable) or
                (isinstance(self.pred_arg, tuple) and all(isinstance(pa, Variable) for pa in self.pred_arg))) and \
               isinstance(self.pat_var, Variable)

    @property
    def has_trans(self) -> bool:
        """
            Return true if `self.transform` is not None
        """
        return self.transform is not None

    @property
    def forward_trans(self) -> Transform:
        """
            Return the forward transformation.
        """
        if self.forward:
            return self.transform
        else:
            return self.transform.inv

    @property
    def backward_trans(self) -> Transform:
        """
            Return the backward transformation.
        """
        if self.forward:
            return self.transform.inv
        else:
            return self.transform

    @property
    def surrogate_pred_arg(self) -> Variable:
        """
            Get the surrogate predicate argument.

            If this event transform has a single predicate argument, then the surrogate predicate argument is simply
            the originally specified predicate argument. Otherwise if a list of predicate arguments are specified,
            then the surrogate predicate argument is a single Variable with a size equal to the sum of all the specified
            predicate arguments' sizes.

            The surrogate predicate argument will have the same value constraints as does the predicate argument. Note
            that if there are multiple predicate arguments, it is required that all of them have the same tuple of
            constraints.
        """
        # If not yet generated, then check if finalized then generated
        if self._surrogate_pred_arg is None:
            # complain if not finalized
            assert self.finalized, "The surrogate predicate argument of a EventTransform can only be generated when " \
                                   "the EventTransform is finalized."
            # Generate
            self._surrogate_pred_arg = Variable(
                '+'.join([arg.name for arg in self.pred_arg]),
                VariableMetatype.Random,
                sum(arg.size for arg in self.pred_arg),
                self.pred_arg[0].constraints    # The original predicate argument's value constraint
            ) if isinstance(self.pred_arg, tuple) else self.pred_arg
        return self._surrogate_pred_arg

    def surrogate_pattern(self) -> EventTransform:
        """
            Returns a new EventTransform pattern with all attributes being the same as this pattern except its
            `pred_arg` which is this pattern's surrogate predicate argument.

            Returns
            -------
            EventTransform
                The surrogate EventTransform pattern.
        """
        return EventTransform(self.surrogate_pred_arg, self.pat_var, self.transform, self.forward, self.master)

    def __repr__(self):
        """
            String representation. Example:

                - ``pred_arg_1 --> pat_var_1, identity, master``
                - ``(pred_arg_1, pred_arg_2, pred_arg_3) --> pat_var_1, forward real()``
        """
        return str(self.pred_arg) + ' --> ' + str(self.pat_var) + ', ' + \
               ((('forward ' if self.forward else 'backward ') + str(self.transform))
                if self.transform is not None else 'identity') + \
               (', master' if self.master else '')
