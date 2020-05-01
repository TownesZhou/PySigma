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


# TODO general-inf: Utility methods for extracting natural parameters from an exponential torch.distributions, or
#  instantiate an exponential torch.distribution from given natural parameters
#  Note that, a new distribution will be instantiated with parameter tensors cloned from the provided ones. This is to
#  protect the distribution instance from changes in the provided tensor. However .clone() retains the cloned tensor in
#  the computation graph, so that gradients can still flow to the provided tensor.
class Params2Dist:
    pass


class Dist2Params:
    pass
