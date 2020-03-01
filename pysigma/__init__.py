"""
    PySigma package init file. Include definition of the Sigma class, the top-level interface to Sigma cognitive
        architecture.
"""
from .structures import *
from .graphical import *
from .cognitive import _register, _compiler, _runner, _perception, _learning


class Sigma:
    """
        Sigma's cognitive interface. User should instantiate this class to create a Sigma model (program).

        Implements Sigma compiler that translates cognitive languages into underlying factor graph representations.

        Graph compilation is eager, i.e., as soon as a predicate or a conditional is added, the corresponding subgraph
            will be compiled and added to the graphical architecture. This is in support of future research on structure
            learning.
    """
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

        # Perception Callables
        self._perception_callable = None

        ### Sigma program global parameters ###


    # Methods for adding Sigma structures #

    def add(self, structure):
        _register.add(self, structure)

    def add_type(self, *args, **kwargs):
        _register.add_type(self, *args, **kwargs)

    def add_predicate(self, *args, **kwargs):
        _register.add_predicate(self, *args, **kwargs)

    def add_conditional(self, *args, **kwargs):
        _register.add_conditional(self, *args, **kwargs)

    def _register_type(self, sigma_type):
        _register._register_type(self, sigma_type)

    def _register_predicate(self, predicate):
        _register._register_predicate(self, predicate)

    def _register_conditional(self, conditional):
        _register._register_conditional(self, conditional)


    # Methods for Sigma structure compilation #

    def _compile_predicate(self, predicate):
        _compiler._compile_predicate(self, predicate)

    def _compile_conditional(self, conditional):
        _compiler._compile_conditional(self, conditional)


    # Methods for Setting Perceptions and Prior Knowledge #

    def perceive(self, predicates, messages):
        _perception.perceive(self, predicates, messages)

    def set_perception(self, predicates, perceive_funcs):
        _perception.set_perception(self, predicates, perceive_funcs)

    def set_assumption(self, predicates, priors):
        _perception.set_assumption(self, predicates, priors)


    # Methods for running Sigma program #

    def _order_nodes(self):
        _runner._order_nodes(self)

    def decide(self, num_cycles):
        _runner.decide(self, num_cycles)



