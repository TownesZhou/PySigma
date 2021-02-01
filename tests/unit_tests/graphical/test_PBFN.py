"""
    Unit tests for PBFN class
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
import torch.distributions as D
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, FactorNode, VariableNode, Node
from pysigma.graphical.predicate_nodes import PBFN, WMVN
from pysigma.utils import KnowledgeServer as KS
from pysigma.utils import DistributionServer as DS
from ..test_Message import TestMessage

from ...utils import assert_equal_within_error, EPS


class TestPBFN:

    def test_init(self):
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        assert pbfn.b_shape == b_shape and pbfn.e_shape == e_shape
        assert isinstance(pbfn.buffer, Message) and pbfn.buffer.type is MessageType.Dual and pbfn.buffer.isid

    def test_add_link_wrong_incoming(self):
        # Test that AssertionError is raised if linkdata is incoming
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        msg_shape = (b_shape, Size([]), Size([]), e_shape)
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        wmvn = MagicMock(spec=WMVN)
        wmvn.name = "test_wmvn"
        linkdata = LinkData(wmvn, pbfn, to_fn=True, msg_shape=msg_shape)

        with pytest.raises(AssertionError) as excinfo:
            pbfn.add_link(linkdata)

        assert str(excinfo.value) == "In test_pbfn: PBFN can only admit outgoing linkdata that connects to a WMVN. The " \
                                     "given linkdata {} is incoming.".format(linkdata)

    def test_add_link_wrong_multiple(self):
        # Test that AssertionError is raised if attempt to register more than one linkdata
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        msg_shape = (b_shape, Size([]), Size([]), e_shape)
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        wmvn_1, wmvn_2 = MagicMock(spec=WMVN), MagicMock(spec=WMVN)
        wmvn_1.name = "test_wmvn_1"
        wmvn_2.name = "test_wmvn_2"
        linkdata_1 = LinkData(wmvn_1, pbfn, to_fn=False, msg_shape=msg_shape)
        linkdata_2 = LinkData(wmvn_2, pbfn, to_fn=False, msg_shape=msg_shape)

        # Add first
        pbfn.add_link(linkdata_1)

        # Add second
        with pytest.raises(AssertionError) as excinfo:
            pbfn.add_link(linkdata_2)

        assert str(excinfo.value) == "In test_pbfn: PBFN only accepts one linkdata. Linkdata {} is already registered."\
                                     .format(linkdata_1)

    def test_add_link_wrong_node(self):
        # Test that AssertionError is raised if linkdata does not connect to a WMVN
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        msg_shape = (b_shape, Size([]), Size([]), e_shape)
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        # Mock some random node
        node = MagicMock(spec=VariableNode)
        node.name = "test_random_node"
        linkdata = LinkData(node, pbfn, to_fn=False, msg_shape=msg_shape)

        with pytest.raises(AssertionError) as excinfo:
            pbfn.add_link(linkdata)

        assert str(excinfo.value) == "In test_pbfn: PBFN only admits linkdata that connects to a WMVN. The given " \
                                     "linkdata {} connects to a {}.".format(linkdata, type(node))

    def test_add_link_wrong_shape_1(self):
        # Test that AssertionError is raised if linkdata has incompatible batch shape
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        wmvn = MagicMock(spec=WMVN)
        wmvn.name = "test_wmvn"
        linkdata = LinkData(wmvn, pbfn, to_fn=False, msg_shape=(Size([11, 15, 20]), Size([]), Size([]), e_shape))

        with pytest.raises(AssertionError) as excinfo:
            pbfn.add_link(linkdata)

        assert str(excinfo.value) == "In test_pbfn: Attempting to add a linkdata with incompatible message shapes. " \
                                     "Expect batch shape {} and event shape {}, but found {} and {} in the given " \
                                     "linkdata respectively.".format(b_shape, e_shape, Size([11, 15, 20]), e_shape)

    def test_add_link_wrong_shape_2(self):
        # Test that AssertionError is raised if linkdata has incompatible event shape
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        wmvn = MagicMock(spec=WMVN)
        wmvn.name = "test_wmvn"
        linkdata = LinkData(wmvn, pbfn, to_fn=False, msg_shape=(b_shape, Size([]), Size([]), Size([1, 2, 4])))

        with pytest.raises(AssertionError) as excinfo:
            pbfn.add_link(linkdata)

        assert str(excinfo.value) == "In test_pbfn: Attempting to add a linkdata with incompatible message shapes. " \
                                     "Expect batch shape {} and event shape {}, but found {} and {} in the given " \
                                     "linkdata respectively.".format(b_shape, e_shape, b_shape, Size([1, 2, 4]))

    def test_add_link_correct_execution(self):
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        msg_shape = (b_shape, Size([]), Size([]), e_shape)
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        wmvn = MagicMock(spec=WMVN)
        wmvn.name = "test_wmvn"
        linkdata = LinkData(wmvn, pbfn, to_fn=False, msg_shape=msg_shape)

        pbfn.add_link(linkdata)

        assert len(pbfn.out_linkdata) == 1 and pbfn.out_linkdata[0] is linkdata

    def test_perceive_joint_wrong_shape_obs_1(self):
        # Test wrong dimension `obs` in joint mode
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        obs = torch.randn(1, 2, 3)

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, None, mode)

        assert str(excinfo.value) == "In test_pbfn: when perceiving 'joint' observations, if specified, `obs` should be " \
                                     "a 2-dimensional tensor."

    def test_perceive_joint_wrong_shape_obs_2(self):
        # Test correct dimension but wrong shape `obs` in joint mode
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        obs = torch.randn(10, 7)

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, None, mode)

        assert str(excinfo.value) == "In test_pbfn: in 'joint' perception mode, the size of the observation's event " \
                                     "dimension must match the sum of random variable sizes. Expect size 6, but found " \
                                     "size 7."

    def test_perceive_joint_wrong_shape_weight_1(self):
        # Test wrong dimension `weight` in joint mode
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        obs = torch.randn(10, 6)
        weight = torch.rand(3, 4)

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, weight, mode)

        assert str(excinfo.value) == "In test_pbfn: when perceiving 'joint' observations, if specified, `weight` " \
                                     "should be a 1-dimensional tensor."

    def test_perceive_joint_wrong_shape_weight_2(self):
        # Test correct dimension but wrong shape `weight` in joint mode
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        obs = torch.randn(10, 6)
        weight = torch.rand(11)

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, weight, mode)

        assert str(excinfo.value) == "In test_pbfn: in 'joint' perception mode, when specified, the weight tensor must " \
                                     "have same length as the observation tensor's first dimension. Found weight " \
                                     "length 11, and observation tensor's first dimension size 10."

    def test_perceive_joint_wrong_value_weight(self):
        # Test negative value in `weight`
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        obs = torch.randn(10, 6)
        weight = torch.rand(10) - 1

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, weight, mode)

        assert str(excinfo.value) == "In test_pbfn: in 'joint' perception mode, if specified, 'weight' must be a " \
                                     "positive tensor. Found minimum value {}.".format(weight.min())

    def test_perceive_marginal_wrong_shape_obs_1(self):
        # Test wrong dimensions `obs`
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        obs = [torch.randn([10, 1]), torch.randn([15, 2]), torch.randn([20, 3, 4])]
        weight = None

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, weight, mode)

        assert str(excinfo.value) == "In test_pbfn: when perceiving 'marginal' observations, if specified, `obs` should " \
                                     "be an iterable of 2-dimensional tensors."

    def test_perceive_marginal_wrong_shape_obs_2(self):
        # Test wrong number of `obs`
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        obs = [torch.randn([10, 1]), torch.randn([15, 2])]
        weight = None

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, weight, mode)

        assert str(excinfo.value) == "In test_pbfn: in 'marginal' perception mode, the number of observations must " \
                                     "match the number of random variables. Found 2 entries in `obs` but the predicate " \
                                     "has 3 random variables."

    def test_perceive_marginal_wrong_shape_obs_3(self):
        # Test wrong shape `obs`
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        obs = [torch.randn([10, 1]), torch.randn([15, 2]), torch.randn([20, 4])]
        weight = None

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, weight, mode)

        assert str(excinfo.value) == "In test_pbfn: in 'marginal' perception mode, the size of each marginal " \
                                     "observation's event dimension must match the size of the corresponding random " \
                                     "variable. Expect event sizes [1, 2, 3], but found sizes [1, 2, 4]."

    def test_perceive_marginal_wrong_shape_weight_1(self):
        # Test wrong dimension `weight`
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        obs = [torch.randn([10, 1]), torch.randn([15, 2]), torch.randn([20, 3])]
        weight = [torch.rand(10), torch.rand(15), torch.rand(20, 1)]

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, weight, mode)

        assert str(excinfo.value) == "In test_pbfn: when perceiving 'marginal' observations, if specified, `weight` " \
                                     "should be an iterable of 1-dimensional tensors."

    def test_perceive_marginal_wrong_shape_weight_2(self):
        # Test wrong number of `weight`
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        obs = [torch.randn([10, 1]), torch.randn([15, 2]), torch.randn([20, 3])]
        weight = [torch.rand(10), torch.rand(15), torch.rand(20), torch.rand(25)]

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, weight, mode)

        assert str(excinfo.value) == "In test_pbfn: in 'marginal' perception mode, the number of weights must match " \
                                     "the number of random variables. Found 4 entries in `weight` but the predicate " \
                                     "has 3 random variables."

    def test_perceive_marginal_wrong_shape_weight_3(self):
        # Test wrong shape `weight`
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        obs = [torch.randn([10, 1]), torch.randn([15, 2]), torch.randn([20, 3])]
        weight = [torch.rand(10), torch.rand(15), torch.rand(30)]

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, weight, mode)

        assert str(excinfo.value) == "In test_pbfn: the first dimension size of each observation tensor in `obs` " \
                                     "should match the length of the corresponding weight tensor in `weight`. Found " \
                                     "observation first dimension sizes [10, 15, 20], and weight lengths [10, 15, 30]."

    def test_perceive_marginal_wrong_value_weight(self):
        # Test negative value `weight`
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        obs = [torch.randn([10, 1]), torch.randn([15, 2]), torch.randn([20, 3])]
        weight = [torch.rand(10) - 1, torch.rand(15), torch.rand(20)]

        with pytest.raises(AssertionError) as excinfo:
            pbfn.perceive(obs, weight, mode)

        assert str(excinfo.value) == "In test_pbfn: in 'marginal' observation mode, if specified, all entries in the " \
                                     "`weight` must be positive tensors. Found the minimum values for each of the " \
                                     "entry in `weight`: {}.".format([w.min() for w in weight])

    def test_perceive_none_obs_default_buffer_msg(self):
        # Test that when `obs` is None, the instantiated buffer content is a correct identity message
        b_shape, e_shape = Size([10, 15, 20]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        obs = None
        weight = [torch.rand(10), torch.rand(15), torch.rand(20)]

        pbfn.perceive(obs, weight, mode)

        assert pbfn.buffer.isid and pbfn.buffer.type is MessageType.Dual and pbfn.buffer.b_shape == b_shape

    def test_perceive_correct_msg_joint_single_rv_uniform_weight(self):
        # Test the correctness of buffer content with joint mode, single rv, and uniform weight
        b_shape, e_shape = Size([2, 3, 4]), Size([6])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        num_ptcl = 20
        obs = torch.randn([num_ptcl, 6])
        weight = None

        pbfn.perceive(obs, weight, mode)

        assert pbfn.buffer.type is MessageType.Particles
        # Check particles. Compare the sorted tensor by event dimension since element order may be changed
        assert_equal_within_error(pbfn.buffer.particles[0][pbfn.buffer.particles[0][:, 0].argsort()],
                                  obs[obs[:, 0].argsort()])
        # Check weight
        weight_shape = b_shape + Size([num_ptcl])
        assert_equal_within_error(pbfn.buffer.weight, torch.ones(weight_shape) / num_ptcl)
        # Check densities
        assert_equal_within_error(pbfn.buffer.log_densities[0], torch.zeros(num_ptcl) / num_ptcl)

    def test_perceive_correct_msg_joint_single_rv_random_weight(self):
        # Test the correctness of buffer content with joint mode, single rv, and random weight
        b_shape, e_shape = Size([2, 3, 4]), Size([6])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        num_ptcl = 20
        obs = torch.randn([num_ptcl, 6])
        weight = torch.rand(num_ptcl)

        pbfn.perceive(obs, weight, mode)

        assert pbfn.buffer.type is MessageType.Particles
        # Check particles. Compare the sorted tensor by event dimension since element order may be changed
        sort_id = obs[:, 0].argsort()
        assert_equal_within_error(pbfn.buffer.particles[0][pbfn.buffer.particles[0][:, 0].argsort()],
                                  obs[sort_id])
        # Check weight
        expected_weight = weight[sort_id].view(1, 1, 1, 20).expand(2, 3, 4, 20).contiguous()
        expected_weight /= expected_weight.sum(dim=-1, keepdim=True )
        assert_equal_within_error(pbfn.buffer.weight, expected_weight)
        # Check densities
        assert_equal_within_error(pbfn.buffer.log_densities[0], torch.zeros(num_ptcl) / num_ptcl)

    def test_perceive_correct_msg_joint_single_rv_uniform_weight_duplicate_particle(self):
        # Test the correctness of buffer content with joint mode, single rv, and random weight when exists duplicate
        #   particles
        b_shape, e_shape = Size([2, 3, 4]), Size([3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        num_ptcl = 6
        obs = torch.tensor(
            [[0.1, 0.2, 0.3],
             [0.1, 0.2, 0.3],
             [0.4, 0.5, 0.6],
             [0.4, 0.5, 0.6],
             [0.4, 0.5, 0.6]]
        )
        weight = None

        pbfn.perceive(obs, weight, mode)

        assert pbfn.buffer.type is MessageType.Particles
        # Check particles. Compare the sorted tensor by event dimension since element order may be changed
        expected_particles = torch.tensor(
            [[0.1, 0.2, 0.3],
             [0.4, 0.5, 0.6]]
        )
        assert_equal_within_error(pbfn.buffer.particles[0][pbfn.buffer.particles[0][:, 0].argsort()],
                                  expected_particles)
        # Check weight
        expected_weight = torch.tensor([2., 3.])
        expected_weight = expected_weight.view(1, 1, 1, 2).expand(2, 3, 4, 2).contiguous()
        expected_weight /= expected_weight.sum(dim=-1, keepdim=True)
        assert_equal_within_error(pbfn.buffer.weight, expected_weight)
        # Check densities
        assert_equal_within_error(pbfn.buffer.log_densities[0], torch.zeros(2) / 2)

    def test_perceive_correct_msg_joint_multiple_rv_uniform_weight(self):
        # Test the correctness of buffer content with joint mode, multiple rv, and random weight
        b_shape, e_shape = Size([2, 3, 4]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        num_ptcl = 3
        obs = torch.tensor(
            [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
             [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
             [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]]
        )
        weight = None

        pbfn.perceive(obs, weight, mode)

        assert pbfn.buffer.type is MessageType.Particles
        # Check particles. Compare the sorted tensor by event dimension since element order may be changed
        expected_particles = [
            torch.tensor([[1.1],
                          [2.1],
                          [3.1]]),
            torch.tensor([[1.2, 1.3],
                          [2.2, 2.3],
                          [3.2, 3.3]]),
            torch.tensor([[1.4, 1.5, 1.6],
                          [2.4, 2.5, 2.6],
                          [3.4, 3.5, 3.6]])
        ]
        for ptcl, expected_ptcl in zip(pbfn.buffer.particles, expected_particles):
            assert_equal_within_error(ptcl, expected_ptcl)
        # Check weight
        buffer_weight = pbfn.buffer.weight
        buffer_weight_slice = buffer_weight[0, 0, 0, :, :, :]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if not i == j == k:
                        assert_equal_within_error(buffer_weight_slice[i, j, k], torch.tensor(EPS))
                    else:
                        assert_equal_within_error(buffer_weight_slice[i, j, k], torch.tensor(1 / num_ptcl), 1e-5)
        # Check densities
        assert_equal_within_error(pbfn.buffer.log_densities[0], torch.zeros(num_ptcl) / num_ptcl)

    def test_perceive_correct_msg_joint_multiple_rv_random_weight(self):
        # Test the correctness of buffer content with joint mode, multiple rv, and random weight
        b_shape, e_shape = Size([2, 3, 4]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        num_ptcl = 3
        obs = torch.tensor(
            [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
             [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
             [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]]
        )
        weight = torch.rand(3)

        pbfn.perceive(obs, weight, mode)

        assert pbfn.buffer.type is MessageType.Particles
        # Check particles. Compare the sorted tensor by event dimension since element order may be changed
        expected_particles = [
            torch.tensor([[1.1],
                          [2.1],
                          [3.1]]),
            torch.tensor([[1.2, 1.3],
                          [2.2, 2.3],
                          [3.2, 3.3]]),
            torch.tensor([[1.4, 1.5, 1.6],
                          [2.4, 2.5, 2.6],
                          [3.4, 3.5, 3.6]])
        ]
        for ptcl, expected_ptcl in zip(pbfn.buffer.particles, expected_particles):
            assert_equal_within_error(ptcl, expected_ptcl)
        # Check weight
        buffer_weight = pbfn.buffer.weight
        buffer_weight_slice = buffer_weight[0, 0, 0, :, :, :]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if not i == j == k:
                        assert_equal_within_error(buffer_weight_slice[i, j, k], torch.tensor(EPS))
        expected_weight_slice = weight / weight.sum()
        for i in range(3):
            assert_equal_within_error(buffer_weight_slice[i, i, i], expected_weight_slice[i], 1e-5)
        # Check densities
        assert_equal_within_error(pbfn.buffer.log_densities[0], torch.zeros(num_ptcl) / num_ptcl)

    def test_perceive_correct_msg_joint_multiple_rv_random_weight_duplicate_particles(self):
        # Test the correctness of buffer content with joint mode, multiple rv, and random weight with duplicate particles
        b_shape, e_shape = Size([2, 3, 4]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'joint'
        num_ptcl = 6
        obs = torch.tensor(
            [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
             [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
             [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
             [3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
             [3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
             [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]]
        )
        weight = torch.rand(6)

        pbfn.perceive(obs, weight, mode)

        assert pbfn.buffer.type is MessageType.Particles
        # Check particles. Compare the sorted tensor by event dimension since element order may be changed
        expected_particles = [
            torch.tensor([[1.1],
                          [2.1],
                          [3.1]]),
            torch.tensor([[1.2, 1.3],
                          [2.2, 2.3],
                          [3.2, 3.3]]),
            torch.tensor([[1.4, 1.5, 1.6],
                          [2.4, 2.5, 2.6],
                          [3.4, 3.5, 3.6]])
        ]
        for ptcl, expected_ptcl in zip(pbfn.buffer.particles, expected_particles):
            assert_equal_within_error(ptcl, expected_ptcl)
        # Check weight
        buffer_weight = pbfn.buffer.weight
        buffer_weight_slice = buffer_weight[0, 0, 0, :, :, :]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if not i == j == k:
                        assert_equal_within_error(buffer_weight_slice[i, j, k], torch.tensor(EPS))
        expected_weight_slice = torch.tensor([weight[0] + weight[1], weight[2], weight[3] + weight[4] + weight[5]])
        expected_weight_slice /= expected_weight_slice.sum()
        for i in range(3):
            assert_equal_within_error(buffer_weight_slice[i, i, i], expected_weight_slice[i], 1e-5)
        # Check densities
        assert_equal_within_error(pbfn.buffer.log_densities[0], torch.zeros(3) / 3)

    def test_perceive_correct_msg_marginal_single_rv_uniform_weight(self):
        # Test the correctness of buffer content with marginal mode, single rv, and uniform weight
        b_shape, e_shape = Size([2, 3, 4]), Size([6])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        num_ptcl = 20
        obs = [torch.randn([num_ptcl, 6])]
        weight = None

        pbfn.perceive(obs, weight, mode)

        assert pbfn.buffer.type is MessageType.Particles
        # Check particles. Compare the sorted tensor by event dimension since element order may be changed
        expected_ptcl = obs[0]
        assert_equal_within_error(pbfn.buffer.particles[0][pbfn.buffer.particles[0][:, 0].argsort()],
                                  expected_ptcl[expected_ptcl[:, 0].argsort()])
        # Check weight
        weight_shape = b_shape + Size([num_ptcl])
        assert_equal_within_error(pbfn.buffer.weight, torch.ones(weight_shape) / num_ptcl)
        # Check densities
        assert_equal_within_error(pbfn.buffer.log_densities[0], torch.zeros(num_ptcl) / num_ptcl)

    def test_perceive_correct_msg_marginal_single_rv_random_weight(self):
        # Test the correctness of buffer content with marginal mode, single rv, and random weight
        b_shape, e_shape = Size([2, 3, 4]), Size([6])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        num_ptcl = 20
        obs = [torch.randn([num_ptcl, 6])]
        weight = [torch.rand(num_ptcl)]

        pbfn.perceive(obs, weight, mode)

        assert pbfn.buffer.type is MessageType.Particles
        # Check particles. Compare the sorted tensor by event dimension since element order may be changed
        expected_ptcl = obs[0]
        sort_id = expected_ptcl[:, 0].argsort()
        assert_equal_within_error(pbfn.buffer.particles[0][pbfn.buffer.particles[0][:, 0].argsort()],
                                  expected_ptcl[sort_id])
        # Check weight
        expected_weight = weight[0].view(1, 1, 1, 20).expand(2, 3, 4, 20).contiguous()
        expected_weight /= expected_weight.sum(dim=-1, keepdim=True)
        assert_equal_within_error(pbfn.buffer.weight, expected_weight)
        # Check densities
        assert_equal_within_error(pbfn.buffer.log_densities[0], torch.zeros(num_ptcl) / num_ptcl)

    def test_perceive_correct_msg_marginal_multiple_rv_random_weight(self):
        # Test the correctness of buffer content with marginal mode, multiple rvs, and random weight
        b_shape, e_shape = Size([2, 3, 4]), Size([1, 2, 3])
        rel_var_list = [Variable("test_rel_var", VariableMetatype.Relational, b_size) for b_size in b_shape]
        ran_var_list = [Variable("test_ran_var", VariableMetatype.Random, e_size, [C.real]) for e_size in e_shape]
        pbfn = PBFN("test_pbfn", rel_var_list, ran_var_list)

        mode = 'marginal'
        num_ptcl = 5
        obs = [torch.randn([num_ptcl, 1]), torch.randn([num_ptcl, 2]), torch.randn([num_ptcl, 3])]
        weight = [torch.rand(num_ptcl), torch.rand(num_ptcl), torch.rand(num_ptcl)]

        pbfn.perceive(obs, weight, mode)

        assert pbfn.buffer.type is MessageType.Particles
        # Check particles. Compare the sorted tensor by event dimension since element order may be changed
        for ptcl, expected_ptcl in zip(pbfn.buffer.particles, obs):
            assert_equal_within_error(ptcl, expected_ptcl)
        # Check weight
        expanded_log_weight = [weight[0].log().view(-1, 1, 1).expand(num_ptcl, num_ptcl, num_ptcl),
                               weight[1].log().view(1, -1, 1).expand(num_ptcl, num_ptcl, num_ptcl),
                               weight[2].log().view(1, 1, -1).expand(num_ptcl, num_ptcl, num_ptcl)]
        sum_weight = sum(expanded_log_weight).exp()
        expected_weight = sum_weight.view(1, 1, 1, num_ptcl, num_ptcl, num_ptcl)\
            .expand(2, 3, 4, num_ptcl, num_ptcl, num_ptcl).contiguous()
        expected_weight /= expected_weight.sum(dim=[-1, -2, -3], keepdims=True)
        assert_equal_within_error(pbfn.buffer.weight, expected_weight)
        # Check densities
        assert_equal_within_error(pbfn.buffer.log_densities[0], torch.zeros(num_ptcl) / num_ptcl)

