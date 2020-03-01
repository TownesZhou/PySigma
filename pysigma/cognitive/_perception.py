"""
    Class methods for Sigma main class

    Methods regarding setting perception and assumption
"""

import torch
from collections.abc import Iterable
from ..structures import Predicate


def perceive(self, predicates, contents):
    """
        Perceive new messages (set PBFN content) for specified predicates directly. To be called before running for a
            cycle.
    :param predicates:      Iterable of 'str' or Predicates.
    :param contents:        Iterable of torch.Tensor. Message must agree in shape
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    if not isinstance(predicates, Iterable) or not all(isinstance(p, (str, Predicate)) for p in predicates):
        raise ValueError("Argument 'predicates' must be an iterable of 'str's or 'Predicate's.")
    if not isinstance(contents, Iterable) or not all(isinstance(msg, torch.Tensor) for msg in contents):
        raise ValueError("Argument 'contents' must be an iterable of 'torch.Tensor'.")
    if len(list(predicates)) != len(list(contents)):
        raise ValueError("Number of predicates: {}, does not match the number of contents provided: {}."
                         .format(len(list(predicates)), len(list(contents))))

    predicates = list(predicates)
    contents = list(contents)
    for i, pred in enumerate(predicates):
        if isinstance(pred, str):
            if pred not in self.name2predicate.keys():
                raise ValueError("Unknown predicate: {}".format(pred))
            pred = self.name2predicate[pred]
            if 'PBFN' not in self.predicate2group[pred].keys():
                raise ValueError("Predicate '{}' is not perceptual. Cannot perceive on this predicate.")

            # Check content shape
            var_list = pred.var_list
            var_sizes = [var.size for var in var_list]
            if var_sizes != list(contents[i].shape):
                raise ValueError("The shape of {}th content does not agree with the corresponding predicate's variable "
                                 "sizes. Expect: {}, got: {}.".format(i, var_sizes, list(contents[i].shape)))

            # Set PBFN's function
            pbfn = self.predicate2group[pred]["PBFN"]
            pbfn.set_function(contents[i], var_sizes)


def set_perception(self, predicates, perceive_func):
    """
        Set perception callable function. perceive_func will be called every cycle before solution phase

        Callables specified here overrides perceive()
    :param predicates:      Iterable of 'str' or Predicates
    :param perceive_func:   Python callable. Should returns an iterable of torch.Tensor, corresponding to 'predicates'
                            (same order)
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    if not isinstance(predicates, Iterable) or not all(isinstance(p, (str, Predicate)) for p in predicates):
        raise ValueError("Argument 'predicates' must be an iterable of 'str's or 'Predicate's.")
    if not callable(perceive_func):
        raise ValueError("Argument 'perceive_func' must be a Python callable.")

    self._perception_callable = None


def set_assumption(self, predicates, priors):
    """
        Set assumptions (LTMFN contents) for predicates, once and for all
    :param predicates:      Iterable of 'str' or Predicates
    :param priors:          Iterable of torch.Tensor
    :return:
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    if not isinstance(predicates, Iterable) or not all(isinstance(p, (str, Predicate)) for p in predicates):
        raise ValueError("Argument 'predicates' must be an iterable of 'str's or 'Predicate's.")
    if not isinstance(priors, Iterable) or not all(isinstance(p, torch.Tensor) for p in priors):
        raise ValueError("Argument 'priors' must be an iterable of 'torch.Tensor'")
    if len(list(predicates)) != len(list(priors)):
        raise ValueError("Number of predicates: {}, does not match the number of priors provided: {}."
                         .format(len(list(predicates)), len(list(priors))))

    predicates = list(predicates)
    priors = list(priors)
    for i, pred in enumerate(predicates):
        if isinstance(pred, str):
            if pred not in self.name2predicate.keys():
                raise ValueError("Unknown predicate: {}".format(pred))
            pred = self.name2predicate[pred]
            if 'LTMFN' not in self.predicate2group[pred].keys():
                raise ValueError("Predicate '{}' is not memorial. Cannot set assumption on this predicate.")

            # Check content shape
            var_list = pred.var_list
            var_sizes = [var.size for var in var_list]
            if var_sizes != list(priors[i].shape):
                raise ValueError("The shape of {}th priors does not agree with the corresponding predicate's variable "
                                 "sizes. Expect: {}, got: {}.".format(i, var_sizes, list(priors[i].shape)))

            # Set PBFN's function
            ltmfn = self.predicate2group[pred]["LTMFN"]
            ltmfn.set_function(priors[i], var_sizes)
