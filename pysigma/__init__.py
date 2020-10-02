"""
    PySigma package init file. Include definition of the Sigma class, the top-level interface to Sigma cognitive
        architecture.
"""
from .graphical import Graph
from .cognitive import _register, _compiler, _runner, _perception, _adaption, _inspector


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

        self.unique_preds = []      # List of unique predicates, over which selection on WMFN should be performed

        # mappings from name to structure
        self.name2type, self.name2predicate, self.name2conditional = {}, {}, {}

        ## Graphical level bookkeeping data structure
        # mappings from predicate name, conditional name to node group
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

        # Printing lists
        self._print_pred_mem_list = []       # List of predicate's names whose memories (WMFN, LTMFN) to be printed
        self._print_pred_action_list = []    # List of predicate's names. Actions arriving at these to be printed

        ### Sigma program global parameters ###


    # Methods for adding Sigma structures #


    # Methods for Sigma structure compilation #


    # Methods for Setting Perceptions and Prior Knowledge #


    # Methods for adaption phase #


    # Methods for running Sigma program #


    # Methods for printing stuff

