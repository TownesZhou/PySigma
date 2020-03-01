"""
    Class methods for Sigma main class

    Methods associated with running a Sigma program
"""
import networkx as nx


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


def decide(self, num_cycles):
    """
        Run the Sigma program for the given number of decision cycles.

        "Set up our new world, and let's begin the SIMULATION."
    """
    from .. import Sigma
    assert isinstance(self, Sigma)
    assert isinstance(num_cycles, int)

    # TODO: Logging
    for i in range(num_cycles):
        # Compute node order
        self._order_nodes()
        # Solution Phase
        self.G.solve(self._node_order)
        # TODO: Modification phase

