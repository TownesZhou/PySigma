"""
    The gamma node, the peak of a conditional subgraph.
"""
from __future__ import annotations      # For postponed evaluation of typing annotations
from typing import Optional, List, Tuple, Dict
from typing import Iterable as IterableType
import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
import torch

from ..utils import KnowledgeServer
from ..pattern_structures.event_transform import EventTransform
from ..defs import Variable, Message, MessageType
from .basic_nodes import LinkData, VariableNode, FactorNode, NodeConfigurationError


class GVN(VariableNode):
    """Gamma Variable Node

    This node is the terminal node of the Gamma subgraph in a Conditional subgraph, connecting the PDFN and the GFN.
    For the inward message propagation, it combines the condition messages sent from the PDFNs using message
    arithmetics. For the outward propagation, it dispatches the action message from the GFN to each of its connecting
    PDFN.

    Each GVN in a Conditional represents a distinct random variable. Semantically, it represents the combined
    distributional knowledge from all the condition predicate patterns w.r.t. this random variable.

    Parameters
    ----------
    rel_var_list: Iterable of Variables
        The relational variable list. There is no restriction as to what relational variables a GVN can admit. However,
        all GVN in a same Conditional shall admit the same list of relational variables.
    ran_var_list: Iterable of Variables
        The random variable list. Must be a list of length 1, since a GVN represents a single random variable. Each GVN
        in a same Conditional must represent a distinct random variable.
    """
    def __init__(self, name, rel_var_list: IterableType[Variable], ran_var_list: IterableType[Variable], **kwargs):
        super(GVN, self).__init__(name, rel_var_list=rel_var_list, ran_var_list=ran_var_list, **kwargs)
        self.pretty_log['node type'] = "Gamma Variable Node"

        assert len(self.ran_vars) == 1, \
            "GVN can admit only one random variable. Given random variables: {}".format(self.ran_vars)

        # Groups of incoming and outgoing linkdata labeled with their directionality
        # A dictionary mapping direction to dictionaries mapping incoming/outgoing indicator to list of Linkdata
        #   instances.
        # Use a default dict
        self.labeled_ld_group: Dict[str, Dict[bool, List[LinkData]]] = defaultdict(
            lambda: defaultdict(
                lambda: []
            )
        )

    def add_link(self, linkdata):
        """
            Similar to Beta nodes, GVN propagates messages bidirectionally, both inward to and outward from the GFN.
            Thus, linkdata connecting a GVN must be tagged with special attribute `direction` with value ``inward`` or
            ``outward``.

            GVN can admit multiple inward incoming or outward outgoing linkdata, but can only accept one inward outgoing
            and/or outward incoming linkdata that connects to the GFN.
        """
        super(GVN, self).add_link(linkdata)
        assert 'direction' in linkdata.attr.keys() and linkdata.attr['direction'] in ['inward', 'outward'], \
            "In {}: For a Gamma Variable Node, the linkdata '{}' should specify a special attribute named " \
            "'direction', with value either 'inward' or 'outward'.".format(self.name, linkdata)

        # If inward outgoing or outward incoming, check that no linkdata of same direction already registered.
        if (linkdata.attr['direction'] == 'inward' and linkdata.to_fn) or \
           (linkdata.attr['direction'] == 'outward' and not linkdata.to_fn):
            assert len(self.labeled_ld_group[linkdata.attr['direction']][linkdata.to_fn]) == 0, \
                "In {}: A GVN can admit at most one {} {} linkdata. Linkdata {} with the same direction is already " \
                "registered." \
                .format(self.name, linkdata.attr['direction'], 'outgoing' if linkdata.to_fn else 'incoming',
                        self.labeled_ld_group[linkdata.attr['direction']][linkdata.to_fn][0])

        # Register with direction
        self.labeled_ld_group[linkdata.attr['direction']][linkdata.to_fn].append(linkdata)

    def precompute_check(self):
        """
            Check structural integrity.

                1. At least one group of same directional linkdata
                2. In each group, at least one incoming and one outgoing linkdata.
        """
        # Check requirement 1
        if len(self.labeled_ld_group) == 0:
            raise NodeConfigurationError(
                "In {}: No linkdata is registered.".format(self.name)
            )
        # Check requirement 2
        for direction, ld_group in self.labeled_ld_group.items():
            if len(ld_group) < 2:
                to_fn = list(ld_group.keys())[0]
                raise NodeConfigurationError(
                    "In {}: In the {} directional linkdata group, expect both incoming and outgoing linkdata. However, "
                    "found only {} linkdata are registered. "
                    .format(self.name, direction, 'outgoing' if to_fn else 'incoming')
                )

    @VariableNode.compute_control
    def compute(self):
        """
            For inward direction, combine incoming messages using message arithmetics.

            For outward direction, relay incoming message from GFN to each of the outgoing linkdata.

            Note that although GVN breaks quiescence if any of its incoming linkdata contains new message, message
            computation shall be carried out in the relevant directional linkdata group only.
        """
        for direction, ld_group in self.labeled_ld_group.items():
            # Inward message propagation
            if direction == "inward":
                # Continue only if incoming linkdata contains new message
                if any(in_ld.new for in_ld in ld_group[False]):
                    # Gather messages
                    msgs = [in_ld.read() for in_ld in ld_group[False]]
                    out_msg = sum(msgs, Message.identity())
                    # Send message
                    out_ld = ld_group[True][0]
                    out_ld.write(out_msg)

            # Outward message propagation
            else:
                # Continue only if incoming linkdata contains new message
                in_ld = ld_group[False][0]
                if in_ld.new:
                    # Gather message
                    msg = in_ld.read()
                    # Relay message
                    for out_ld in ld_group[True]:
                        out_ld.write(msg)


class GFN(FactorNode):
    """
        Gamma Factor Node

        Carry out general-inference message computation at the PGM factor node.

        Induce a message computation task for each of the outgoing link.
    """
    pass
