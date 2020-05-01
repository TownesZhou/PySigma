"""
    Utility functions
"""
import torch.distributions


def intern_name(name, struc_type):
    """
        Add prefix and brackets to transform user provided structure name to internal name
    :param name:    Structure name
    :param type:    one of "type", "predicate", or "conditional
    :return:        processed name
    """
    assert struc_type in ["type", "predicate", "conditional"], "unknown type for processing structure name"
    assert type(name) is str, "name must be str type"
    if struc_type is "type":
        return "TYPE_[" + name.upper() + "]"
    elif struc_type is "predicate":
        return "PRED_[" + name.upper() + "]"
    elif struc_type is "conditional":
        return "COND_[" + name.upper() + "]"


# TODO general-inf: Utility methods for instantiate a distribution from the given list of tensors. The given list of
#   tensors is treated as the parameters necessary to define that distribution. The semantics of the individual tensors
#   in the list depends on the distribution.
class Params2Dist:
    # Lookup table for automatic conversion
    type2method = \
        {}

    @classmethod
    def convert(cls, params, dist_type):
        """
            Automatic conversion. Provide the params and the distribution class, return a distribution instance.
        """
        assert isinstance(params, list) and all(isinstance(param, torch.Tensor) for param in params)
        assert issubclass(dist_type, torch.distributions.Distribution)
        if dist_type not in cls.type2method.keys():
            raise NotImplementedError("Conversion from parameters to distributions for distribution class {} has not"
                                      "been implemented".format(dist_type))

        return cls.type2method[dist_type](params)


# TODO general-inf: Inversed version of the above class
class Dist2Params:
    # Lookup table for automatic conversion
    type2method = \
        {}

    @classmethod
    def convert(cls, dist):
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
#  Note that, a new distribution will be instantiated with parameter tensors cloned from the provided ones. This is to
#  protect the distribution instance from changes in the provided tensor. However .clone() retains the cloned tensor in
#  the computation graph, so that gradients can still flow to the provided tensor.
class Natural2Exp:
    pass


class Exp2Natural:
    pass

