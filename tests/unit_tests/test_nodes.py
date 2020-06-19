from pysigma.graphical._nodes import *
import torch
import pytest
from torch.distributions.normal import Normal
from pysigma.defs import Variable
from pysigma.structures import *


# some testing parameters that might be used frequently in multiple place
c = torch.distributions.constraints.Constraint()
var = Variable("test", VariableMetatype.Indexing, 2) # s_shape == 2
rel_var_l = [Variable("rel", VariableMetatype.Relational, 3)] # b_shape == 3
ran_var_l = [Variable("ran", VariableMetatype.Random, 4, {c})] # e_shape == 4
var_node_param = [var, rel_var_l, ran_var_l]

class TestInitialization:



    def test_init_LTMFN(self):
        ltmfn = LTMFN("hello", Normal, 100, torch.Size([10]), torch.Size([3]), torch.Size([0, 1]))

    def test_init_WMVN(self):
        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]
        ran_var_l = [Variable("ran", VariableMetatype.Random, 5, {self.c})]
        wmvn = WMVN("hi", var, rel_var_l, ran_var_l, Normal)


    def test_init_PBFN(self):
        pbfn = PBFN("pbfn")

    def test_init_WMFN(self):
        wmfn = WMFN("emfn", "param")


class TestAddLink:
    """
        LTMFN gets one WMVN_IN and one WMVN_OUT, can be the same WMVN

    """
    def test_WMVN_doublelink_LTMFN(self):
        c = torch.distributions.constraints.Constraint()
        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]
        ran_var_l = [Variable("ran", VariableMetatype.Random, 5, {c})]
        wmvn = WMVN("hi", var, rel_var_l, ran_var_l, Normal)

        ltmfn = LTMFN("hello", Normal, 100, torch.Size([10]), torch.Size([3]), torch.Size([0, 1]))

        wmvn.add_link(LinkData(wmvn, ltmfn, True, torch.Size([5, 5, 5]), 0.0003))
        wmvn.add_link(LinkData(wmvn, ltmfn, False, torch.Size([5, 5, 5]), 0.0003))

    def test_WMVN_IN_linkTo_LTMFN_linkTo_WMVN_OUT(self):
        c = torch.distributions.constraints.Constraint()
        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]
        ran_var_l = [Variable("ran", VariableMetatype.Random, 5, {c})]
        wmvn_in = WMVN("wmvn_in", var, rel_var_l, ran_var_l, Normal)
        wmvn_out = WMVN("wmvn_out", var, rel_var_l, ran_var_l, Normal)

        ltmfn = LTMFN("ltmfn", Normal, 100, torch.Size([10]), torch.Size([3]), torch.Size([0, 1]))

        wmvn_in.add_link(LinkData(wmvn_in, ltmfn, False, torch.Size([5, 5, 5]), 0.0003))
        ltmfn.add_link(LinkData(wmvn_out, ltmfn, False, torch.Size([5, 5, 5]), 0.0003))

class TestFactorNode():
    """
        Testing thr function: FactorNode.add_link()
    """
    def test_init(self):
        # f = FactorNode("factor")
        pass


class TestVariableNode:
    """
        Testing thr function: VariableNode.add_link()
    """
    pass





class TestDFNandDVN:
    """
        Testing add_link() and compute() function for Dummy Factor Node
    """
    def test_init(self):
        dfn = DFN("my_dfn")
        dvn = DVN("my_dvn", *var_node_param)

    def test_add_link_(self):
        # Try to link DFN with DVN and link back
        dfn = DFN("my_dfn")
        dvn = DVN("my_dvn", *var_node_param)

        # data from dvn to dfn
        ld1 = LinkData(dvn, dfn, True, torch.Size([2, 3, 4]), 0.00001)
        dfn.add_link(ld1)
        dvn.add_link(ld1)

        # data from dfn to dvn
        ld2 = LinkData(dvn, dfn, False, torch.Size([2, 3, 4]), 0.00001)
        dfn.add_link(ld2)
        dvn.add_link(ld2)


    def test_add_link_problematic(self):
        dfn1 = DFN("my_dfn1")
        dfn2 = DFN("my_dfn2")

        dvn1 = DVN("my_dvn1", *var_node_param)
        dvn2 = DVN("my_dvn2", *var_node_param)

        # Try to link DFN with another DFN
        with pytest.raises(AssertionError):
            ld1 = LinkData(dfn1, dfn2, True, torch.Size([2, 3, 4]), 0.00001)

        # Try to link DVN with another DVN
        with pytest.raises(AssertionError):
            ld1 = LinkData(dvn1, dvn2, False, torch.Size([2, 3, 4]), 0.00001)

        # Try to link data not self
        with pytest.raises(AssertionError):
            ld2 = LinkData(dvn1, dfn1, True, torch.Size([2, 3, 4]), 0.00001)
            dfn2.add_link(ld2)

        with pytest.raises(AssertionError):
            ld2 = LinkData(dvn1, dfn1, False, torch.Size([2, 3, 4]), 0.00001)
            dvn2.add_link(ld2)

    def test_compute(self):
        # Try to link DFN with DVN and link back
        dfn = DFN("my_dfn")
        dvn = DVN("my_dvn", *var_node_param)

        # data from dvn to dfn
        ld1 = LinkData(dvn, dfn, True, torch.Size([2, 3, 4]), 0.00001)
        dfn.add_link(ld1)
        dvn.add_link(ld1)

        # data from dfn to dvn
        ld2 = LinkData(dvn, dfn, False, torch.Size([2, 3, 4]), 0.00001)
        dfn.add_link(ld2)
        dvn.add_link(ld2)

        dfn.compute()

class TestLTMFN:
    """
    Testing For:
        LTMFN.add_link(),
        LTMFN.toggle_draw() + LTMFN.draw_particles(),
        LTMFN.compute()
    """
    pass

class TestWMVN:
    """
    Testing For:
        WMVN.compute()
    """
    pass

class TestPBFN:
    """
    Testing For:
        PBFN.add_link(),
        PBFN.set_perception() + PBFN.compute()
    """
    pass

class TestWMFN:
    """
    Testing For:
        WMFN.add_link(),
        WMFN.update_memory() + WMFN.compute()
    """
    pass

class TestAlphaFactorNode:
    """
    Testing For:
        AlphaFactorNode.add_link()
    """
    pass

class TestRelMapNode:
    """
    Testing For:
        RelMapNode.inward_compute() and RelMapNode.outward_compute()
    """
    pass

class TestExpSumNode:
    """
    Testing For:
        ExpSumNode.inward_compute() and ExpSumNode.outward_compute()
    """
    pass