from pysigma.graphical._nodes import *
import torch
from torch.distributions.normal import Normal
from pysigma.defs import Variable
from pysigma.structures import *

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

    def test_

class TestCompute:
    pass
