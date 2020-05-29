"""
    Utility functions
"""
import torch
import torch.distributions
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint, integer_interval
from itertools import Iterable
import math
from typing import List, Type


def intern_name(name: str, struc_type: str):
    """
        Add prefix and brackets to transform user provided structure name to internal name
    :param name:    Structure name
    :param type:    one of "type", "predicate", or "conditional
    :return:        processed name
    """
    assert struc_type in ["type", "predicate", "conditional"], "unknown type for processing structure name"
    assert isinstance(name, str)
    if struc_type is "type":
        return "TYPE_[" + name.upper() + "]"
    elif struc_type is "predicate":
        return "PRED_[" + name.upper() + "]"
    elif struc_type is "conditional":
        return "COND_[" + name.upper() + "]"


def extern_name(name: str, struc_type: str):
    """
        Inverse operation of intern_name
    """
    assert struc_type in ["type", "predicate", "conditional"], "unknown type for processing structure name"
    assert isinstance(name, str)
    if struc_type is "type":
        assert name.find("TYPE_") >= 0
    if struc_type is "predicate":
        assert name.find("PRED_[") >= 0
    if struc_type is "conditional":
        assert name.find("COND_[") >= 0
    return name[6:-1]


# TODO general-inf: Utility methods for instantiate a distribution from the given list of tensors. The given list of
#   tensors is treated as the parameters necessary to define that distribution. The semantics of the individual tensors
#   in the list depends on the distribution.
class Params2Dist:
    # Lookup table for automatic conversion
    type2method = \
        {}

    @classmethod
    def convert(cls, params: torch.Tensor, dist_type: Type, b_shape: torch.Size = None, e_shape: torch.Size = None):
        """
            Automatic conversion. Provide the params and the distribution class, return a distribution instance.
            If b_shape, e_shape not None, then will check if the shape of the resulting distribution matches

            assume params has shape (b_shape + param_shape)
        """
        assert isinstance(params, torch.Tensor)
        assert issubclass(dist_type, torch.distributions.Distribution)
        assert (b_shape is None) == (e_shape is None)
        assert b_shape is None or isinstance(b_shape, torch.Size)

        if dist_type not in cls.type2method.keys():
            raise NotImplementedError("Conversion from parameters to distributions for distribution class {} has not"
                                      "been implemented".format(dist_type))

        dist = cls.type2method[dist_type](params)

        if b_shape is not None:
            if dist.batch_shape != b_shape or dist.event_shape != e_shape:
                raise ValueError("The shape of the generated distribution {} does not match the provided shape. "
                                 "Expect (batch_shape, event_shape) == ({}, {}), but instead got ({}, {})."
                                 .format(str(dist), b_shape, e_shape, dist.batch_shape, dist.event_shape))

        return dist


# TODO general-inf: Inversed version of the above class
class Dist2Params:
    # Lookup table for automatic conversion
    type2method = \
        {}

    @classmethod
    def convert(cls, dist: torch.distributions.Distribution):
        """
            Automatic conversion. Provide the distribution instance, return its params as list of tensors.
        """
        assert isinstance(dist, torch.distributions.Distribution)
        dist_type = type(dist)
        if dist_type not in cls.type2method.keys():
            raise NotImplementedError("Conversion from distributions to parameters for distribution class {} has not"
                                      "been implemented".format(dist_type))

        return cls.type2method[dist_type](dist)


# TODO general-inf: Utility methods for extracting natural parameters from an exponential torch.distributions, or
#  instantiate an exponential torch.distribution from given natural parameters
class Natural2Exp:
    pass


class Exp2Natural:
    pass


# TODO: Knowledge format check.
#       Check if RV size, type, and value constraint are compatible with declared distribution class of predicate
#       distribution class dependent
class FormatCheck:
    pass


# TODO: Particle knowledge translator class
class KnowledgeTraslator:
    """
        knowledge translator class. Translate knowledge tensors between the forms understandable by Predicate and that
            understandable by PyTorch's distribution class. This includes both event particle tensors and parameter
            tensors.

        Different distribution class requires different handling, but in general values for multiple random variables
            (potentially of different sizes) will be concatenated together to form the last dimension of event samples
            that can be interpreted by PyTorch.

        A translator instance should be instantiated and hold by each Predicate.
    """

    # distribution-dependent translation method pointer. Indexed by distribution class
    dict_2torch_event = {

    }
    dict_2pred_event = {

    }
    dict_2torch_param = {

    }
    dict_2pred_param = {

    }

    def __init__(self, dist_class, var_sizes, var_constraints):
        """
            Instantiate a translator

            :param dist_class: A subclass of torch.distributions.Distributions
            :param var_sizes:  A sequence of integers, denoting the sizes of a predicate's random variables. The order
                               will be respected when concatenating event values.
            :param var_constraints: A sequence of torch.distributions.constraints.Constraint object. Each denoting the
                                    value constraint of the corresponding random variable.
        """
        assert issubclass(dist_class, Distribution)
        assert isinstance(var_sizes, Iterable) and all(isinstance(size, int) for size in var_sizes)
        assert isinstance(var_constraints, Iterable) and all(isinstance(c, Constraint) for c in var_constraints)
        assert len(var_sizes) == len(var_constraints)

        self.dist_class = dist_class
        self.var_sizes = var_sizes
        self.var_constraints = var_constraints

        self.num_vars = len(var_sizes)

    def event2torch_event(self, particles):
        """
            Translate event particles from the format understandable by Predicate to the format understandable by
                PyTorch
        """
        assert isinstance(particles, torch.Tensor)
        if self.dist_class not in self.dict_2torch_event.keys():
            raise NotImplementedError("Translation for distribution class '{}' not yet implemented"
                                      .format(self.dist_class))
        return self.dict_2torch_event[self.dist_class](particles)

    def event2pred_event(self, particles):
        """
            Translate event particles from the format understandable by PyTorch to the format understandable by
                Predicate
        """
        assert isinstance(particles, torch.Tensor)
        if self.dist_class not in self.dict_2pred_event.keys():
            raise NotImplementedError("Translation for distribution class '{}' not yet implemented"
                                      .format(self.dist_class))
        return self.dict_2pred_event[self.dist_class](particles)

    def param2torch_param(self, params):
        """
            Translate parameters from the format understandable by Predicate to the format understandable by
                PyTorch
        """
        assert isinstance(params, torch.Tensor)
        if self.dist_class not in self.dict_2torch_param.keys():
            raise NotImplementedError("Translation for distribution class '{}' not yet implemented"
                                      .format(self.dist_class))
        return self.dict_2torch_param[self.dist_class](params)

    def param2pred_event(self, params):
        """
            Translate parameters from the format understandable by PyTorch to the format understandable by
                Predicate
        """
        assert isinstance(params, torch.Tensor)
        if self.dist_class not in self.dict_2pred_param.keys():
            raise NotImplementedError("Translation for distribution class '{}' not yet implemented"
                                      .format(self.dist_class))
        return self.dict_2pred_param[self.dist_class](params)

    """
        Categorical distribution. Assumes all RV have size 1
            - event translation from pred to torch:
                Split last dimension by number of variables. Compute value by taking volume multiplication
                
    """
    def _categorical_var_span(self):
        # Helper function to determine the value range of each rv
        assert all(isinstance(c, integer_interval) for c in self.var_constraints)
        return list(c.upper_bound - c.lower_bound + 1 for c in self.var_constraints)

    def _categorical_2torch_event(self, particles):
        assert isinstance(particles, torch.Tensor) and particles.shape[-1] == self.num_vars
        split_particles = particles.split(split_size=1, dim=-1)
        var_span = self._categorical_var_span()

        # Taking volume product
        volume_prod = 1
        base = 1
        # Going backward through spans to take product
        for val, span in zip(reversed(split_particles), reversed(var_span)):
            # Cumulative summation by the product of i_th variable's value with its base
            volume_prod += val * base
            # Base of i_th var is the product of the spans of all later variables (i.e. from (i+1)th to n_th variable)
            base *= span

        return volume_prod

    def _categorical_2pred_event(self, particles):
        assert isinstance(particles, torch.Tensor)
        # TODO: check particles dtype and shape
        var_span = self._categorical_var_span()

        # Treat values as volume products and take mod w.r.t. variables' spans
        particle_list = []
        residue = particles
        base = math.prod(var_span)
        # Going forward through spans to take modulo
        for span in var_span:
            base /= span
            particle_list.append(residue % base)
            residue = residue // base

        # Concatenate along the last dimension
        result = torch.cat(particle_list, dim=-1)
        return result


