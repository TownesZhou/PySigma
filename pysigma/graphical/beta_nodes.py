"""
    All nodes related to the Conditional Beta subgraph
"""
from __future__ import annotations      # For postponed evaluation of typing annotations
from typing import Optional, List, Tuple, Dict, DefaultDict
from typing import Iterable as IterableType
import copy
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
import torch
from ..defs import Variable
from ..pattern_structures.variable_map import VariableMapCallable
from .basic_nodes import LinkData, FactorNode, NodeConfigurationError
from ..pattern_structures.summarization import SummarizationClass, SummarizationError


class BetaFactorNode(FactorNode, ABC):
    """
        Abstract base class for any factor node belonging to the Conditional Beta subgraph.

        Captures the commonalities of all Beta subgraph factor nodes:

        * **Topology**: a Beta Factor Node accepts one and up to two groups of linkdata, with one group propagating
          messages inward to the Gamma Factor Node, and another group propagating messages outward to the predicate
          subgraph. In each group, there should be at least one incoming and outgoing linkdata, but unlike an Alpha
          Factor Node, there is no restriction on the maximum number of linkdata in each group.
        * **Admissible variable nodes**: a Beta factor node works on the random variables, and does not touch the batch
          dimensions of the messages. Thus, incident variable nodes should have the same tuple of relational variables.
        * **Compute pattern**: a Beta factor node computes outgoing messages for each group of linkdata individually.
          In other words, for instance, the outgoing message to an inward outgoing link is solely dependent on the
          message received from the inward incoming links. Accordingly, the `compute()` method is divided into an
          `inward_compute()` and an `outward_compute()` subroutine.
        * **Quiescence state**: a Beta factor node as a whole reaches quiescence if and only if **all** incoming
          linkdata do not contain new message. However, for the two subroutines `inward_compute()` and
          `outward_compute()`, either one should not be called once its incoming linkdata do not contain new message.

        Attributes
        ----------
        labeled_ld_group : dict
            Dictionary of groups of incoming and outgoing linkdata labeled by their message propagation direction.
            Structure:
                {
                    [ linkdata.attr['direction] ] : {
                            [ linkdata.to_fn ] : linkdata
                        }
                }
        rel_vars: tuple of Variable
            The relational variables of all incident variable node.
    """
    def __init__(self, name: str, **kwargs):
        super(BetaFactorNode, self).__init__(name, **kwargs)

        # Groups of incoming and outgoing linkdata labeled with their directionality w.r.t. the Beta structure
        # A dictionary mapping direction to dictionaries mapping incoming/outgoing indicator to list of Linkdata
        #   instances.
        # Use a default dict
        self.labeled_ld_group: Dict[str, Dict[bool, List[LinkData]]] = defaultdict(
            lambda: defaultdict(
                lambda: []
            )
        )
        # Cache the relational variables when add_link() is called to ensure that incident variable nodes have the
        #   same tuple of relational variables.
        self.rel_vars: Optional[Tuple[Variable]] = None

    def add_link(self, linkdata: LinkData):
        """
            A Beta factor node accepts at least one but no more than two groups of incoming and outgoing linkdata. The
            incident variable nodes should have the same tuple of relational variables. Furthermore, a ``"direction"``
            key-ed attribute should be included in the linkdata's optional attribute map with value ``"inward"`` or
            ``"outward"`` to indicate its message propagation direction. The two groups of linkdata should have
            different message propagation directions.

            Parameters
            ----------
            linkdata : LinkData
                The linkdata to be registered. Must specify a ``"direction"`` attribute in its optional attribute map
                ``linkdata.attr``.
        """
        assert isinstance(linkdata, LinkData)
        assert 'direction' in linkdata.attr.keys() and linkdata.attr['direction'] in ['inward', 'outward'], \
            "In {}: For a Beta Factor Node, the linkdata '{}' should specify a special attribute named 'direction', " \
            "with value either 'inward' or 'outward'.".format(self.name, linkdata)

        # Check relational variables
        # Cache the first linkdata's relational variables every registered
        if self.rel_vars is None:
            self.rel_vars = linkdata.vn.rel_vars
        else:
            assert self.rel_vars == linkdata.vn.rel_vars, \
                "In {}: linkdata '{}' has conflicting relational variables. This Beta Factor Node infers relational " \
                "variables {} from other previously registered linkdata, but found different relational variables {} " \
                "in this linkdata." \
                .format(self.name, linkdata, self.rel_vars, linkdata.vn.rel_vars)
            
        super(BetaFactorNode, self).add_link(linkdata)

        # Register the linkdata in the paired group based on its direction.
        # Because we are using a default dict, we can directly register
        self.labeled_ld_group[linkdata.attr['direction']][linkdata.to_fn].append(linkdata)

    def precompute_check(self):
        """
            Basic computable requirement for a Beta factor node:

                - It has at least one directional group of linkdata, and in each group, there should be at least one
                  incoming and one outgoing linkdata.

            This method shall be extended by a Beta Factor Node subclass if it has more specific requirements.
        """
        if len(self.labeled_ld_group) == 0:
            raise NodeConfigurationError(
                "Wrong configuration for node {}: no registered linkdata.".format(self.name)
            )
        for direction, ld_group in self.labeled_ld_group.items():
            if len(ld_group[True]) == 0:    # No incoming linkdata
                raise NodeConfigurationError(
                    "Wrong configuration for node {}: incomplete group of linkdata. For the group of linkdata with "
                    "message propagation direction '{}', there is no incoming linkdata."
                    .format(self.name, direction)
                )
            if len(ld_group[False]) == 0:   # No outgoing linkdata
                raise NodeConfigurationError(
                    "Wrong configuration for node {}: incomplete group of linkdata. For the group of linkdata with "
                    "message propagation direction '{}', there is no outgoing linkdata."
                    .format(self.name, direction)
                )

    @FactorNode.compute_control
    def compute(self):
        """
            Calls `inward_compute()` and `outward_compute()` individually if any of their corresponding incoming
            linkdata contains new message.
        """
        for direction, ld_group in self.labeled_ld_group.items():
            if any(ld.new for ld in ld_group[True]):
                if direction == 'inward':
                    self.inward_compute(ld_group[True], ld_group[False])
                else:
                    self.outward_compute(ld_group[True], ld_group[False])

    @abstractmethod
    def inward_compute(self, in_lds: List[LinkData], out_lds: List[LinkData]):
        """
            Inward message computation. To be implemented by child class.
        """
        raise NotImplementedError

    @abstractmethod
    def outward_compute(self, in_lds: List[LinkData], out_lds: List[LinkData]):
        """
            Outward message computation. To be implemented by child class.
        """
        raise NotImplementedError


class PDFN(FactorNode):
    """
        Particle Dispatch Factor Node
    """
    def __init__(self, name):
        super(PDFN, self).__init__(name)


class PRFN(FactorNode):
    """
        Particle Reweight Factor Node
    """
    def __init__(self, name):
        super(PRFN, self).__init__(name)


