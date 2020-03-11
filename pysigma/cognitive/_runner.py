"""
    Class methods for Sigma main class

    Methods associated with running a Sigma program
"""
import networkx as nx
from tqdm import tqdm        # progress bar
import time
from prettytable import PrettyTable         # For printing performance statistics in tables
from ..cognitive._inspector import *
from ..cognitive._adaption import *
from ..graphical._nodes import PBFN, LTMFN, GFFN


def _order_nodes(self):
    """
        Calculate the node ordering used for message passing in graph solution phase, based on ascending order of
            the link depths, starting from PBFN, WMFN & LTMFN as source nodes.
        Set the self._node_order field

        Note that the calculation is lazy. If no new structures were added and node ordered has been calculated
            previously, then return directly
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

    # TODO: Clean up the code. Seperate logging from core functions.
    # Set all nodes' quiescence to False before perform message passing
    for node in self.G.nodes:
        node.quiescence = False

    # Performance statistics
    num_acess = [0] * len(self._node_order)
    time_access = [[] for i in range(len(self._node_order))]

    quiesced = False
    iter = 0
    while not quiesced:
        iter += 1
        quiesced = True

        inspect_list = []  # Print which node was inspected during this iteration


        # Code for computation on each node
        def compute():
            nonlocal node, quiesced, self, inspect_list, num_acess, time_access, i   # Declare those as outer variables
            order = self._node_order            # Only for debugging display

            # Quiescence of node is defined by quiescence of links. So check links every time
            if not node.check_quiesce():
                # PBFN, LTMFN, and GFFN node should always try sending its messages, so its execution does not count
                #   toward breaking quiescence
                if not isinstance(node, (PBFN, LTMFN, GFFN)):
                    quiesced = False
                inspect_list.append(str(node))

                # Recore compute time
                tic = time.perf_counter()
                node.compute()
                toc = time.perf_counter()
                num_acess[i] += 1
                time_access[i].append(toc - tic)

        # Seperate main for loop so that we can display statistics such as progress bar
        # TODO: make pretty display
        if verbose == 2:
            for i, node in enumerate(tqdm(self._node_order)):
                compute()
        else:
            if verbose == 1:
                print("### Iteration {} ###".format(iter))
            for i, node in enumerate(self._node_order):
                compute()
            if verbose == 1:
                print("Node inspected (not quiesced): {}\n".format(inspect_list))

    # Print performance statistics
    if verbose == 1:
        col_text = ["Node name", "Number of visits", "Total compute time", "Average compute time", "Min compute time", "Max compute time"]
        t = PrettyTable(col_text)
        for i, node in enumerate(self._node_order):
            text_format = "{:.5f}"
            row_text = [str(node),
                        num_acess[i],
                        sum(time_access[i]),
                        sum(time_access[i]) / num_acess[i] if num_acess[i] != 0 else 0,
                        min(time_access[i]),
                        max(time_access[i])]
            row_text = [text_format.format(t) if isinstance(t, float) else t for t in row_text]
            t.add_row(row_text)
        print(t)
        total_time = sum(sum(time_access[i]) for i in range(len(self._node_order)))
        print("Total time: {:.5f}".format(total_time))


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
        # Modification phase
        self._modify()

        # Print stuff
        self._print_pred_mem()          # Print predicate memories (WMFN and LTMFN if any)
        self._print_pred_action()       # Print combined actions arriving at certain predicates
