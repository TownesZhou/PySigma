"""
    PySigma package init file. Include definition of the Sigma class, the top-level interface to Sigma cognitive
        architecture.
"""

import networkx as nx
from copy import deepcopy
from .structures import *
from .graphical import *
from .cognitive import _add_structure, _compiler, _run


class Sigma:
    """
        Sigma's cognitive interface. User should instantiate this class to create a Sigma model (program).

        Implements Sigma compiler that translates cognitive languages into underlying factor graph representations.

        Graph compilation is eager, i.e., as soon as a predicate or a conditional is added, the corresponding subgraph
            will be compiled and added to the graphical architecture. This is in support of future research on structure
            learning.
    """
    # Import class methods
    # from .cognitive._add_structure import add, add_type, add_predicate, add_conditional, \
    #     _register_type, _register_predicate, _register_conditional
    # from .cognitive._compiler import _compile_predicate, _compile_conditional
    # from .cognitive._run import _order_nodes, decide



    def __init__(self, *args, **kwargs):
        """
            Register global parameters. Create an empty Sigma program.

            All lookup tables use structure name to index

            "Lay down your pieces, and let's begin OBJECT CREATION."
        """
        ### Public bookkeeping data structures ###
        ## Cognitive level lookup tables
        # list of types, predicates, conditionals
        self.type_list = []
        self.predicate_list = []
        self.conditional_list = []

        # mappings from name to structure
        self.name2type, self.name2predicate, self.name2conditional = {}, {}, {}

        ## Graphical level bookkeeping data structure
        # mappings from predicate, conditional to node group
        #   use structure name to index
        self.predicate2group, self.conditional2group = {}, {}

        # The graphical architecture
        self.G = Graph()

        ### Private data members ###
        # Field that contains node order and Flag indicating whether node ordering has been calculated
        self._node_order = None
        self._order_set = False

        ### Sigma program global parameters ###


    # Methods for adding Sigma structures #

    def add(self, structure):
        _add_structure.add(self, structure)

    def add_type(self, *args, **kwargs):
        _add_structure.add_type(self, *args, **kwargs)

    def add_predicate(self, *args, **kwargs):
        _add_structure.add_predicate(self, *args, **kwargs)

    def add_conditional(self, *args, **kwargs):
        _add_structure.add_conditional(self, *args, **kwargs)

    def _register_type(self, sigma_type):
        _add_structure._register_type(self, sigma_type)

    def _register_predicate(self, predicate):
        _add_structure._register_predicate(self, predicate)

    def _register_conditional(self, conditional):
        _add_structure._register_conditional(self, conditional)


    # Methods for Sigma structure compilation #

    def _compile_predicate(self, predicate):
        _compiler._compile_predicate(self, predicate)

    def _compile_conditional(self, conditional):
        _compiler._compile_conditional(self, conditional)


    # Methods for running Sigma program #

    def _order_nodes(self):
        _run._order_nodes(self)

    def decide(self, num_cycles):
        _run.decide(self, num_cycles)



