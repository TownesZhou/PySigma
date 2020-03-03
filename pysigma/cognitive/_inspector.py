"""
    Methods for inspecting, displaying, and debugging critical values in a Sigma program.
"""

from ..structures import *
from ..utils import *


def print_predicate_memory(self, predicates):
    """
        Will print WMFN and LTMFN content (if any) of specified predicates after each decision cycle
    :param predicates:      An iterable of 'str' or 'Predicate's.
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    if not isinstance(predicates, Iterable) or not all(isinstance(p, (str, Predicate)) for p in predicates):
        raise ValueError("Argument 'predicates' must be an iterable of 'str' or 'Predicate's.")

    for pred in predicates:
        if isinstance(pred, str):
            name = intern_name(pred, "predicate")
            if name not in self.name2predicate.keys():
                raise ValueError("Unknown predicate: {}.".format(pred))
            self._print_pred_mem_list.append(name)
        else:
            self._print_pred_mem_list.append(pred.name)


def _print_pred_mem(self):
    """
        Actual memory printing
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    if len(self._print_pred_mem_list) == 0:
        return

    print("\n### Printing Predicate Memories ###\n")
    for i, name in enumerate(self._print_pred_mem_list):
        nodegroup = self.predicate2group[name]
        wmfn = nodegroup["WMFN"]
        ltmfn = nodegroup["LTMFN"] if "LTMFN" in nodegroup.keys() else None

        print("({}) {} : \n".format(i, name))
        print("\tWorking Memory: \n\t\t{}\n".format(wmfn.get_function()[0]))
        if ltmfn is not None:
            print("\tLong-Term Memory: \n\t\t{}\n".format(ltmfn.get_function()[0]))


def print_combined_action(self, predicates):
    """
        Print the combined actions arriving at specified predicates at the end of each decision cycle.

        Combinaed action is retrieved from the linkdata connecting ACFN to predicate‘s WMVN
    :param predicates:      An iterable of 'str' or Predicate's
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    if not isinstance(predicates, Iterable) or not all(isinstance(p, (str, Predicate)) for p in predicates):
        raise ValueError("Argument 'predicates' must be an iterable of 'str' or 'Predicate's.")

    for pred in predicates:
        if isinstance(pred, str):
            name = intern_name(pred, "predicate")
            if name not in self.name2predicate.keys():
                raise ValueError("Unknown predicate: {}.".format(pred))
            self._print_pred_action_list.append(name)
        else:
            self._print_pred_action_list.append(pred.name)


def _print_pred_action(self):
    """
        Actual action printing
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    if len(self._print_pred_action_list) == 0:
        return

    print("\n### Printing Predicate Combined Actions ###\n")
    for i, name in enumerate(self._print_pred_action_list):
        nodegroup = self.predicate2group[name]
        wmvn = nodegroup["WMVN"] if "WMVN" in nodegroup.keys() else nodegroup["WMVN_IN"]
        acfn = nodegroup["ACFN"]
        ld = self.G.get_linkdata(acfn, wmvn)
        msg = ld.memory
        print("({}) {} : \n".format(i, name))
        print("\tActions arriving: \n\t\t{}\n".format(msg))


