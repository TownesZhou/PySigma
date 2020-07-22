from graphical import LTMFN, PBFN, WMFN
from pysigma.graphical.basic_nodes import *
import pytest
from torch.distributions.normal import Normal
from pysigma.structures import *


# some testing parameters that might be used frequently in multiple place
c = Constraint()
n = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
var = Variable("test", VariableMetatype.Indexing, 2) # s_shape == 2
rel_var_l = [Variable("rel", VariableMetatype.Relational, 3)] # b_shape == 3
ran_var_l = [Variable("ran", VariableMetatype.Random, 4, {c})] # e_shape == 4
var_node_param = [var, rel_var_l]


def tuple_shape(size_list):
    return tuple(torch.Size([s]) if isinstance(s, int) else torch.Size(s) for s in size_list)

def test_tuple_shape():
    a = tuple_shape([1, 2, 3])
    b = tuple_shape([[1, 2], 3, [4, 5]])
    assert a == (torch.Size([1]), torch.Size([2]), torch.Size([3]))
    assert b == (torch.Size([1, 2]), torch.Size([3]), torch.Size([4, 5]))

class TestInitialization:



    def test_init_LTMFN(self):
        ltmfn = LTMFN("hello", Normal, 100, torch.Size([10]), torch.Size([3]), torch.Size([0, 1]))

    def test_init_WMVN(self):
        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]
        ran_var_l = [Variable("ran", VariableMetatype.Random, 5, {c})]
        wmvn = WMVN("hi", Normal, var, rel_var_l)


    def test_init_PBFN(self):
        pbfn = PBFN("pbfn")

    def test_init_WMFN(self):
        wmfn = WMFN("emfn", "param")


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
        ld1 = LinkData(dvn, dfn, True, tuple_shape([2, 3, 4]), 0.00001)
        dfn.add_link(ld1)
        dvn.add_link(ld1)

        # data from dfn to dvn
        ld2 = LinkData(dvn, dfn, False, tuple_shape([2, 3, 4]), 0.00001)
        dfn.add_link(ld2)
        dvn.add_link(ld2)


    def test_add_link_problematic(self):
        dfn1 = DFN("my_dfn1")
        dfn2 = DFN("my_dfn2")

        dvn1 = DVN("my_dvn1", *var_node_param)
        dvn2 = DVN("my_dvn2", *var_node_param)

        # Try to link DFN with another DFN
        with pytest.raises(AssertionError):
            ld1 = LinkData(dfn1, dfn2, True, tuple_shape([2, 3, 4]), 0.00001)

        # Try to link DVN with another DVN
        with pytest.raises(AssertionError):
            ld1 = LinkData(dvn1, dvn2, False, tuple_shape([2, 3, 4]), 0.00001)

        # Try to link data not self
        with pytest.raises(AssertionError):
            ld2 = LinkData(dvn1, dfn1, True, tuple_shape([2, 3, 4]), 0.00001)
            dfn2.add_link(ld2)

        with pytest.raises(AssertionError):
            ld2 = LinkData(dvn1, dfn1, False, tuple_shape([2, 3, 4]), 0.00001)
            dvn2.add_link(ld2)

    def test_compute(self):
        # Try to link DFN with DVN and link back
        dfn = DFN("my_dfn")
        dvn = DVN("my_dvn", *var_node_param)

        # data from dvn to dfn
        ld1 = LinkData(dvn, dfn, True, tuple_shape([[2, 3], 4]), 0.00001)
        rand_msg = Message(MessageType.Parameter,
                           batch_shape=torch.Size([2, 3]), param_shape=torch.Size([4]), parameters=torch.rand(2, 3, 4))
        ld1.write(rand_msg)
        dfn.add_link(ld1)
        dvn.add_link(ld1)

        # data from dfn to dvn
        ld2 = LinkData(dvn, dfn, False, tuple_shape([[2, 3], 4]), 0.00001)
        rand_msg = Message(MessageType.Parameter,
                           batch_shape=torch.Size([2, 3]), param_shape=torch.Size([4]), parameters=torch.rand(2, 3, 4))
        ld2.write(rand_msg)
        dfn.add_link(ld2)
        dvn.add_link(ld2)

        dvn.compute()
        dfn.compute()

class TestLTMFNandWMVN:
    """
    Testing For:
        LTMFN.add_link(),
        LTMFN.toggle_draw() + LTMFN.draw_particles(),
        LTMFN.compute()
        WMVN.compute()
    """
    def test__init(self):
        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]

        wmvn = WMVN("hi", Normal, var, rel_var_l)
        ltmfn = LTMFN("hello", Normal, 100, torch.Size([10]), torch.Size([3]), torch.Size([0, 1]))

    def test_WMVN_doublelink_LTMFN(self):
        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]

        wmvn = WMVN("hi", Normal, var, rel_var_l)
        ltmfn = LTMFN("hello", Normal, 100, torch.Size([10]), torch.Size([3]), torch.Size([4]))

        ld1 = LinkData(wmvn, ltmfn, True, tuple_shape([5, 5, 5]), 0.0003, type="param")
        ld2 = LinkData(wmvn, ltmfn, False, tuple_shape([5, 5, 5]), 0.0003)

        msg_1 = Message(MessageType.Particles,
                     sample_shape=torch.Size([2]),
                     batch_shape=torch.Size([3,4,5]),
                     event_shape=torch.Size([2]),
                     particles=torch.rand(2, 2),
                     weights=1,
                     log_density=0
                     )

        msg_2 = Message(MessageType.Parameter,
                     param_shape=torch.Size([3]),
                     batch_shape=torch.Size([2,3]),
                     parameters=torch.rand(2, 3, 3)
                     )

        ld1.write(msg_1)
        ld2.write(msg_2)

        wmvn.add_link(ld1)
        ltmfn.add_link(ld1)
        wmvn.add_link(ld2)
        ltmfn.add_link(ld2)

        ltmfn.toggle_draw()
        ltmfn.draw_particles()

        wmvn.compute()
        ltmfn.compute()

    def test_WMVN_IN_linkTo_LTMFN_linkTo_WMVN_OUT(self):
        c = Constraint()
        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]
        ran_var_l = [Variable("ran", VariableMetatype.Random, 5, {c})]

        wmvn_in = WMVN("wmvn_in", Normal, var, rel_var_l)
        wmvn_out = WMVN("wmvn_out", Normal, var, rel_var_l)
        ltmfn = LTMFN("ltmfn", Normal, 100, torch.Size([10]), torch.Size([3]), torch.Size([0, 1]))

        ld1 = LinkData(wmvn_in, ltmfn, True, tuple_shape([5, 5, 5]), 0.0003, type="event")
        ld2 = LinkData(wmvn_out, ltmfn, False, tuple_shape([5, 5, 5]), 0.0003)

        msg_1 = Message(MessageType.Particles,
                        sample_shape=torch.Size([2]),
                        batch_shape=torch.Size([3]),
                        event_shape=torch.Size([2]),
                        particles=torch.ones(2, 2),
                        weights=torch.tensor([[0.1, 0.5, 0.7], [0.9, 0.5, 0.3]]),
                        log_density=torch.tensor([1, 2])
                        )

        msg_2 = Message(MessageType.Parameter,
                        batch_shape=torch.Size([2, 3, 4]),
                        param_shape=torch.Size([2]),
                        parameters=torch.ones(2, 3, 4, 2)
                        )

        ld1.write(msg_1)
        ld2.write(msg_2)

        wmvn_in.add_link(ld1)
        ltmfn.add_link(ld1)
        ltmfn.add_link(ld2)
        wmvn_out.add_link(ld2)

class TestPBFN:
    """
    Testing For:
        PBFN.add_link(),
        PBFN.set_perception() + PBFN.compute()
    """
    def test_init(self):
        pbfn = PBFN("pbfn")

    def test_addlink_problematic(self):
        with pytest.raises(AssertionError):
            # add link with DVN
            pbfn = PBFN("pbfn")
            dvn = DVN("my_dvn", *var_node_param)

            ld = LinkData(dvn, pbfn, True, tuple_shape([5, 5, 5]), 0.0003)
            pbfn.add_link(ld)
            dvn.add_link(ld)

    def test_addlink_wmvn(self):
        pbfn = PBFN("pbfn")

        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]

        wmvn = WMVN("hi", Normal, var, rel_var_l)
        ld = LinkData(wmvn, pbfn, False, tuple_shape([5, 5, 5]), 0.0003)

        pbfn.add_link(ld)
        wmvn.add_link(ld)

    def test_set_perception(self):
        pbfn = PBFN("pbfn")

        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]

        wmvn = WMVN("hi", Normal, var, rel_var_l)
        ld = LinkData(wmvn, pbfn, False, tuple_shape([5, 5, 5]), 0.0003)

        pbfn.add_link(ld)
        wmvn.add_link(ld)

        num_obs = 10
        b_shape = torch.Size([5, 5, 5])
        e_shape = torch.Size([3])

        observation = torch.rand(10, 5, 5, 5, 3)

        pbfn.set_perception(observation, 1, num_obs, b_shape, e_shape)
        pbfn.compute()


class TestWMFN:
    """
    Testing For:
        WMFN.add_link(),
        WMFN.update_memory() + WMFN.compute()
    """
    def test_init(self):
        # defualt decay
        wmfn = WMFN("wmfn", "param")
        wmfn = WMFN("wmfn", "event")

        # other decay
        wmfn = WMFN("wmfn", "param", decay_rate=0)
        wmfn = WMFN("wmfn", "event", decay_rate=0)
        wmfn = WMFN("wmfn", "param", decay_rate=0.5)
        wmfn = WMFN("wmfn", "event", decay_rate=0.5)

    def test_addlink(self):
        # wmvn_in --> wmfn --> wmvn_out
        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]

        wmfn = WMFN("wmfn", "param", decay_rate=0.5)

        wmvn_in = WMVN("wmvn_in", Normal, var, rel_var_l)
        wmvn_out = WMVN("wmvn_out", Normal, var, rel_var_l)

        ld1 = LinkData(wmvn_in, wmfn, True, tuple_shape([5, 5, 5]), 0.0003)
        ld2 = LinkData(wmvn_out, wmfn, False, tuple_shape([5, 5, 5]), 0.0003)

        wmvn_in.add_link(ld1)
        wmfn.add_link(ld1)
        wmfn.add_link(ld2)
        wmvn_out.add_link(ld2)

    def test_update_memory_compute(self):
        # wmvn_in --> wmfn --> wmvn_out
        var = Variable("test", VariableMetatype.Indexing, 5)
        rel_var_l = [Variable("rel", VariableMetatype.Relational, 5)]

        wmfn = WMFN("wmfn", "param", decay_rate=0.5)
        wmvn_in = WMVN("wmvn_in", Normal, var, rel_var_l)
        wmvn_out = WMVN("wmvn_out", Normal, var, rel_var_l)

        ld1 = LinkData(wmvn_in, wmfn, True, tuple_shape([5, 5, 5]), 0.0003)
        ld2 = LinkData(wmvn_out, wmfn, False, tuple_shape([5, 5, 5]), 0.0003)

        msg_1 = Message(MessageType.Particles,
                        sample_shape=torch.Size([2]),
                        batch_shape=torch.Size([3]),
                        event_shape=torch.Size([2]),
                        particles=torch.ones(2, 2),
                        weights=torch.tensor([[0.1, 0.5, 0.7], [0.9, 0.5, 0.3]]),
                        log_density=torch.tensor([1, 2])
                        )

        msg_2 = Message(MessageType.Parameter,
                        batch_shape=torch.Size([2, 3, 4]),
                        param_shape=torch.Size([2]),
                        parameters=torch.ones(2, 3, 4, 2)
                        )

        ld1.write(msg_1)
        ld2.write(msg_2)

        wmvn_in.add_link(ld1)
        wmfn.add_link(ld1)
        wmfn.add_link(ld2)
        wmvn_out.add_link(ld2)

        wmfn.update_memory()



        # wmvn_in.compute()
        # wmfn.compute()



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
    # def test_init(self):
    #     rmn = RelMapNode("rmn", )
    pass


class TestExpSumNode:
    """
    Testing For:
        ExpSumNode.inward_compute() and ExpSumNode.outward_compute()
    """
    pass

