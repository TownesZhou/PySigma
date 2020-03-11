"""
    Methods relating to the Adaption phase / Graph Modification phase, including selection at WMFN, learning at
        assumption nodes, attention, apprisals, etc.

"""

def _select(self):
    """
        Selection at closed-world unique predicates' WMFN node
    """
    from .. import Sigma
    assert isinstance(self, Sigma)


