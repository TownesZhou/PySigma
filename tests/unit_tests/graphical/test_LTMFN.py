"""
    Unit tests for LTMFN class
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, FactorNode
from pysigma.graphical.predicate_nodes import LTMFN
from pysigma.utils import KnowledgeServer as KS

from ...utils import generate_positive_definite


class TestLTMFN():

    # Utility methods
    def generate_ltmfn_1(self, b_shape=Size([4, 5]), p_shape=Size([2]), s_shape=Size([10, 15, 20]),
                         e_shape=Size([1, 2, 3])):
        name = "test_ltmfn"

        dist_class = D.Distribution
        rv_sizes = list(e_shape)
        rv_cstr = [C.real, ] * len(e_shape)
        rv_num_ptcl = list(s_shape)
        ks = KS(dist_class, rv_sizes, rv_cstr, rv_num_ptcl)

        rel_var_list = [Variable("rel_" + str(i), VariableMetatype.Relational, b) for i, b in enumerate(b_shape)]
        param_var = Variable("param", VariableMetatype.Parameter, p_shape[0])
        index_var_list = [Variable("index_" + str(i), VariableMetatype.Indexing, s) for i, s in enumerate(s_shape)]
        ran_var_list = [Variable("ran_" + str(i), VariableMetatype.Random, e, [rv_cstr[i]])
                        for i, e in enumerate(e_shape)]

        ltmfn = LTMFN(name, ks, rel_var_list, param_var, index_var_list, ran_var_list)

        return ltmfn

    def test_init(self):
        ltmfn = self.generate_ltmfn_1()

        assert ltmfn.to_draw
        assert ltmfn.msg_cache is None
