"""
    Graphical architecture submodule init file. Include main class of the Graphical architecture
"""

import networkx
from .basic_nodes import *
from .predicate_nodes import WMVN, LTMFN, PBFN, WMFN


class Graph(networkx.DiGraph):
    """
        The actual graph instance representing a compiled graphical architecture. Can be manipulated directly as a
            `networkx.DiGraph` but store extra bookkeeping information. use `networkx.DiGraph` as super class to support
            directed links.
    """

    def __init__(self):
        super(Graph, self).__init__()

        # node type name2type dictionary
        self._name2type = \
            {
             DFN.__name__: DVN,
             PBFN.__name__: PBFN,
             LTMFN.__name__: LTMFN,
             WMFN.__name__: WMFN,
             ACFN.__name__: ACFN,
             FFN.__name__: FFN,
             NFN.__name__: NFN,
             NLFN.__name__: NLFN,
             ADFN.__name__: ADFN,
             ATFN.__name__: ATFN,
             BJFN.__name__: BJFN,
             GFFN.__name__: GFFN,
             DVN.__name__: DVN,
             WMVN.__name__: WMVN,
             }

    def new_node(self, node_type, *args, **kwargs):
        """
            A node factory method. Generate new node based on node_type, add it to DiGraph, and return it.
        :param node_type:   str or class. Specifying the node type, such as 'WMVN' or WMVN
        :return:            the newly generated node instance
        """
        assert node_type in self._name2type.keys() or node_type in self._name2type.values(), \
            "first argument {} not a known node type".format(node_type)

        if node_type in self._name2type.keys():
            node_type = self._name2type[node_type]
        node = node_type(*args, **kwargs)

        # Add node to DiGraph
        self.add_node(node)

        return node

    def add_unilink(self, node1, node2, **kwargs):
        """
            Add a unidirectional link FROM node1 TO node2, and create and register corresponding LinkData
            Note that one of the nodes should be a variable node and the other a factor node

            Additional attributes in kwargs will be set in the linkdata
        """
        assert (isinstance(node1, VariableNode) and isinstance(node2, FactorNode)) or \
               (isinstance(node1, FactorNode) and isinstance(node2, VariableNode)), \
            "One of the nodes must be a variable node and the other one a factor node"

        # Create a LinkData and set its attributes
        vn = node1 if isinstance(node1, VariableNode) else node2
        fn = node2 if isinstance(node1, VariableNode) else node1
        var_list = vn.var_list
        to_fn = True if isinstance(node2, FactorNode) else False
        linkdata = LinkData(vn, fn, var_list, to_fn, **kwargs)

        # Create edge in graph. The LinkData exists in the 'data' field in an edge of the NetworkX graph
        self.add_edge(node1, node2, data=linkdata)

        # register LinkData in both nodes
        node1.add_link(linkdata)
        node2.add_link(linkdata)

    def add_bilink(self, node1, node2, **kwargs):
        """
            Add a bidrectional link between node 1 and node2. Note that one of the nodes should be a variable node and
                the other a factor node

            Additional attributes in kwargs will be set in the linkdata
        """
        assert (isinstance(node1, VariableNode) and isinstance(node2, FactorNode)) or \
               (isinstance(node1, FactorNode) and isinstance(node2, VariableNode)), \
            "One of the nodes must be a variable node and the other one a factor node"

        self.add_unilink(node1, node2, **kwargs)
        self.add_unilink(node2, node1, **kwargs)

    def get_linkdata(self, node1, node2):
        """
            Return the linkdata on the edge from node1 to node2
        """
        return self[node1][node2]["data"]
