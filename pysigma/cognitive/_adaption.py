"""
    Methods relating to the Adaption phase / Graph Modification phase, including selection at WMFN, learning at
        assumption nodes, attention, apprisals, etc.

"""
import torch
from ..structures import *


def _best_select(msg, dims):
    """
        Select max values over specified dimensions
    """
    pass


def prob_match_select(msg, dims):
    """
        Sample values over the specified dimensions as if 
    """


def _select(self):
    """
        Selection at closed-world unique predicates' WMFN node
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    for pred in self.unique_preds:
        assert isinstance(pred, Predicate)

        unique_sym = pred.pred_unique_sym
        unique_vars = pred.pred_unique_vars
        wmfn = self.predicate2group[pred.name]['WMFN']
        in_ld = wmfn._in_linkdata[0]
        in_msg = in_ld.memory
        assert isinstance(in_msg, (int, float, torch.Tensor))

        # If predicate unique symbol is '%', simply swap wmfn content
        if unique_sym == '%':
            if isinstance(in_msg, (int, float)):
                wmfn.set_function(in_msg, None)
            else:
                wmfn.set_function(in_msg, in_ld.var_list)
        # Otherwise, perform selection after setting content
        else:
            # If msg is not torch tensor (constant), first broadcast to torch.Tensor
            if not isinstance(in_msg, torch.Tensor):
                in_msg = in_msg * torch.ones(size=in_ld._dims)
            # TODO: Special Selection
            pass

