"""
    Class methods for Sigma main class

    Methods associated with running a Sigma program
"""
import networkx as nx
from tqdm import tqdm        # progress bar
from ..cognitive._inspector import *


def _order_nodes(self):
    """
        Calculate the node ordering used for message passing in graph solution phase, based on ascending order of
            the link depths, starting from PBFN, WMFN & LTMFN as source nodes.
        Set the self._node_order field

        Note that the calculation is lazy. If no new structures were added and node ordered has been calculated
            previously, then return directly
        # TODO: Discuss with Volkan about proper node ordering
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    if self._order_set:
        return

    # Add a temporary source node to graph connecting PBFN, WMFN & LTMFN
    source = "source"
    self.G.add_node(source)

    # Collect all PBFN and LTMFNs and connect to source
    first_nodes = []
    for ng in self.predicate2group.values():
        if "PBFN" in ng.keys():
            first_nodes.append(ng["PBFN"])
        if "WMFN" in ng.keys():
            first_nodes.append(ng["WMFN"])
        if "LTMFN" in ng.keys():
            first_nodes.append(ng["LTMFN"])
    for node in first_nodes:
        self.G.add_edge(source, node)

    # Obtain a bfs order
    edges = nx.bfs_edges(self.G, source)
    self._node_order = [v for u, v in edges]

    # Set node's pretty log
    for i, node in enumerate(self._node_order):
        node.pretty_log["node order"] = i

    # Remove temporary source node
    self.G.remove_node(source)

    # Set flag
    self._order_set = True


def _solve(self, verbose):
    """
        One phase of graph solution of message passing until quiescence is reached.

        :param node_order: An iterator of nodes. The node will be traversed in this order
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    # TODO: A LOT of logging may be needed here
    # Set all nodes' quiescence to False before perform message passing
    for node in self.G.nodes:
        node.quiescence = False

    quiesced = False
    iter = 1
    while not quiesced:
        quiesced = True

        # Code for computation on each node
        def compute():
            nonlocal node, quiesced       # Declare those as outer variables
            if not node.quiescence:
                quiesced = False
                node.compute()

        # Seperate main for loop so that we can display statistics such as progress bar
        # TODO: make pretty display
        if verbose == 2:
            iter += 1
            for node in tqdm(self._node_order):
                compute()
        else:
            for node in self._node_order:
                compute()


def _modify(self):
    """
        One phase of graph modification on existing graph memories.
    """
    from .. import Sigma
    assert isinstance(self, Sigma)

    # TODO
    pass


def decide(self, num_cycles, verbose=0):
    """
        Run the Sigma program for the given number of decision cycles.

        "Set up our new world, and let's begin the SIMULATION."

        :param num_cycles:      int. Number of decision cycles
        :param verbose:         int. Verbose level
                                    0 = print nothing
                                    1 = display general performance statistics regarding message passing
                                    2 = verbose 1 and display progress bar
    """
    from .. import Sigma
    assert isinstance(self, Sigma)
    assert isinstance(num_cycles, int)
    assert isinstance(verbose, int)

    # TODO: Logging
    for i in range(num_cycles):
        # Compute node order
        self._order_nodes()
        # Solution Phase
        self._solve(verbose)
        # TODO: Modification phase
        self._modify()

        # Print stuff
        self._print_pred_mem()          # Print predicate memories (WMFN and LTMFN if any)
        self._print_pred_action()       # Print combined actions arriving at certain predicates
