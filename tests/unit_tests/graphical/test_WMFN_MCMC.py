"""
    Unit tests for WMFN class
"""
import pytest
from unittest.mock import MagicMock
import torch
import torch.distributions.constraints as C
from torch import Size

from pysigma.defs import Message, MessageType, Variable, VariableMetatype
from pysigma.graphical.basic_nodes import LinkData, VariableNode, NodeConfigurationError
from pysigma.graphical.predicate_nodes import WMFN_MCMC, WMVN
from ...utils import random_message


class TestWMFN_MCMC:

    def test_init(self):
        s_shape, e_shape = Size([10, 15, 20]), Size([2, 3, 4])
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        assert wmfn.eval_msg_cache.isid and wmfn.post_msg_cache.isid

        # Check distribution parameters' shape
        test_ptcl = [torch.randn(s_size, e_size) for s_size, e_size in zip(s_shape, e_shape)]
        log_pdf = [walk_dist.log_prob(ptcl) for ptcl, walk_dist in zip(test_ptcl, wmfn.walk_dist)]

        assert all(pdf.shape == Size([s_size]) for pdf, s_size in zip(log_pdf, s_shape))

    def test_add_link_not_wmvn(self):
        # Test AssertionError is raised if connected to not a WMVN
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        mock_vn = MagicMock(spec_set=VariableNode)
        ld = LinkData(mock_vn, wmfn, True, msg_shape)

        with pytest.raises(AssertionError, match="In test_wmfn: WMFN_MCMC can only be connected to WMVN."):
            wmfn.add_link(ld)

    def test_add_link_missing_special_attribute(self):
        # Test AssertionError is raised if 'type' keyed special attribute is not specified in the linkdata
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, True, msg_shape)

        with pytest.raises(AssertionError, match="In test_wmfn: Linkdata must specify a special attribute `type` with "
                                                 "value `posterior` or `evaluation`."):
            wmfn.add_link(ld)

    def test_add_link_wrong_special_attribute_value(self):
        # Test AssertionError is raised if `type` keyed special attribute does not have value `posterior` or
        #  `evaluation`
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, True, msg_shape)
        ld.attr['type'] = 'something else'

        with pytest.raises(AssertionError, match="In test_wmfn: Linkdata must specify a special attribute `type` with "
                                                 "value `posterior` or `evaluation`."):
            wmfn.add_link(ld)

    def test_add_link_duplicate_ld_1(self):
        # Test AssertionError is raised if linkdata connects to a duplicate type of WMVN
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, to_fn=True, msg_shape=msg_shape)
        ld.attr['type'] = 'evaluation'

        with pytest.raises(AssertionError, match="In test_wmfn: Duplicate connection to WMVN_IN_EVAL."):
            # Add twice
            wmfn.add_link(ld)
            wmfn.add_link(ld)

    def test_add_link_duplicate_ld_2(self):
        # Test AssertionError is raised if linkdata connects to a duplicate type of WMVN
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, to_fn=True, msg_shape=msg_shape)
        ld.attr['type'] = 'posterior'

        with pytest.raises(AssertionError, match="In test_wmfn: Duplicate connection to WMVN_IN_POST."):
            # Add twice
            wmfn.add_link(ld)
            wmfn.add_link(ld)

    def test_add_link_duplicate_ld_3(self):
        # Test AssertionError is raised if linkdata connects to a duplicate type of WMVN
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, to_fn=False, msg_shape=msg_shape)
        ld.attr['type'] = 'evaluation'

        with pytest.raises(AssertionError, match="In test_wmfn: Duplicate connection to WMVN_OUT_EVAL."):
            # Add twice
            wmfn.add_link(ld)
            wmfn.add_link(ld)

    def test_add_link_duplicate_ld_4(self):
        # Test AssertionError is raised if linkdata connects to a duplicate type of WMVN
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, to_fn=False, msg_shape=msg_shape)
        ld.attr['type'] = 'posterior'

        with pytest.raises(AssertionError, match="In test_wmfn: Duplicate connection to WMVN_OUT_POST."):
            # Add twice
            wmfn.add_link(ld)
            wmfn.add_link(ld)

    def test_add_link_wrong_msg_shape_1(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, to_fn=False, msg_shape=(b_shape, p_shape, Size([10, 15, 25]), Size([2, 3, 4])))
        ld.attr['type'] = 'posterior'

        with pytest.raises(AssertionError) as excinfo:
            wmfn.add_link(ld)

        assert str(excinfo.value) == "In test_wmfn: linkdata should have the same sample shape as what is declared for" \
                                     " this node. Expect {}, but found {}.".format(s_shape, Size([10, 15, 25]))

    def test_add_link_wrong_msg_shape_2(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, to_fn=False, msg_shape=(b_shape, p_shape, Size([10, 15, 20]), Size([2, 3, 5])))
        ld.attr['type'] = 'posterior'

        with pytest.raises(AssertionError) as excinfo:
            wmfn.add_link(ld)

        assert str(excinfo.value) == "In test_wmfn: linkdata should have the same event shape as what is declared " \
                                     "for this node. Expect {}, but found {}.".format(e_shape, Size([2, 3, 5]))

    def test_add_link_correct_1(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, to_fn=True, msg_shape=msg_shape)
        ld.attr['type'] = 'posterior'

        wmfn.add_link(ld)

        assert wmfn.ld_in_post is ld

    def test_add_link_correct_2(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, to_fn=True, msg_shape=msg_shape)
        ld.attr['type'] = 'evaluation'

        wmfn.add_link(ld)

        assert wmfn.ld_in_eval is ld

    def test_add_link_correct_3(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, to_fn=False, msg_shape=msg_shape)
        ld.attr['type'] = 'posterior'

        wmfn.add_link(ld)

        assert wmfn.ld_out_post is ld

    def test_add_link_correct_4(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wmvn = MagicMock(spec_set=WMVN)
        ld = LinkData(wmvn, wmfn, to_fn=False, msg_shape=msg_shape)
        ld.attr['type'] = 'evaluation'

        wmfn.add_link(ld)

        assert wmfn.ld_out_eval is ld

    def test_init_particles_wrong_msg_type(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([1]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        test_msg = random_message(MessageType.Parameter, *msg_shape)

        with pytest.raises(AssertionError):
            wmfn.init_particles(test_msg)

    def test_init_particles_wrong_msg_shape(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        wrong_shape = (b_shape, p_shape, Size([10, 15, 21]), e_shape)
        test_msg = random_message(MessageType.Particles, *wrong_shape)

        with pytest.raises(ValueError) as excinfo:
            wmfn.init_particles(test_msg)

        assert str(excinfo.value) == "In test_wmfn: `init_ptcl_msg`'s sample shape and event shape are incompatible. " \
                                     "Expecting sample shape {} and event shape {}, but found {} and {}."\
                                     .format(s_shape, e_shape, Size([10, 15, 21]), e_shape)

    def test_init_particles_correct_execution(self):
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        test_msg = random_message(MessageType.Particles, *msg_shape)

        wmfn.init_particles(test_msg)

        assert wmfn.post_msg_cache == test_msg
        assert isinstance(wmfn.eval_msg_cache, Message) and MessageType.Particles in wmfn.eval_msg_cache.type

    def test_modify_empty_in_msg(self):
        # Test that no computation is carried out and msg cache remain as is if one of the two incoming messages is
        #   empty
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        # Mock linkdata
        in_ld_eval = MagicMock(spec_set=LinkData)
        in_ld_post = MagicMock(spec_set=LinkData)
        wmfn.ld_in_eval, wmfn.ld_in_post = in_ld_eval, in_ld_post

        in_ld_eval.read.return_value = Message.identity()
        in_ld_post.read.return_value = Message.identity()

        wmfn.modify()

        assert wmfn.post_msg_cache.isid and wmfn.eval_msg_cache.isid

    def test_modify_wrong_msg_type(self):
        # Test AssertionError is raised if incoming message has wrong type
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([5]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        # Mock linkdata
        in_ld_eval = MagicMock(spec_set=LinkData)
        in_ld_post = MagicMock(spec_set=LinkData)
        wmfn.ld_in_eval, wmfn.ld_in_post = in_ld_eval, in_ld_post

        in_ld_eval.read.return_value = random_message(MessageType.Parameter, *msg_shape)
        in_ld_post.read.return_value = random_message(MessageType.Parameter, *msg_shape)

        with pytest.raises(AssertionError, match="In test_wmfn: Expect all incoming messages to be Particles type, but "
                                                 "instead found type {}, {} for in_eval_msg and in_post_msg "
                                                 "respectively.".format(MessageType.Parameter, MessageType.Parameter)):
            wmfn.modify()

    def test_modify_correct_execution(self):
        # Test the method can execute normally
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        # Mock linkdata
        in_ld_eval = MagicMock(spec_set=LinkData)
        in_ld_post = MagicMock(spec_set=LinkData)
        wmfn.ld_in_eval, wmfn.ld_in_post = in_ld_eval, in_ld_post

        in_ld_eval.read.return_value = random_message(MessageType.Particles, *msg_shape)
        in_ld_post.read.return_value = random_message(MessageType.Particles, *msg_shape)

        wmfn.modify()

        assert MessageType.Particles in wmfn.post_msg_cache.type and MessageType.Particles in wmfn.eval_msg_cache.type

    def test_ill_configuration(self):
        # Test NodeConfigurationError is raised if the 4 linkdata are not all registered
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        # Mock only two linkdata
        in_ld_eval = MagicMock(spec_set=LinkData)
        out_ld_post = MagicMock(spec_set=LinkData)
        wmfn.ld_in_eval, wmfn.ld_out_post = in_ld_eval, out_ld_post

        with pytest.raises(NodeConfigurationError) as excinfo:
            wmfn.compute()

        assert str(excinfo.value) == "Wrong configuration for node test_wmfn: All four of the linkdata must be " \
                                     "specified for a WMFN-MCMC node. Only 2 of them are specified."


    def test_quiescence_no_compute(self):
        # Test that no message is sent when quiesced
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        # Mock linkdata
        in_ld_eval, in_ld_post = MagicMock(spec_set=LinkData), MagicMock(spec_set=LinkData)
        out_ld_eval, out_ld_post = MagicMock(spec_set=LinkData), MagicMock(spec_set=LinkData)
        wmfn.ld_in_eval, wmfn.ld_in_post = in_ld_eval, in_ld_post
        wmfn.ld_out_eval, wmfn.ld_out_post = out_ld_eval, out_ld_post

        # Set to visited
        wmfn.visited = True

        wmfn.compute()

        out_ld_eval.write.assert_not_called()
        out_ld_post.write.assert_not_called()

    def test_quiescence_after_compute(self):
        # Test that when a compute is taken place, the node reaches quiescence
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        # Mock linkdata
        in_ld_eval, in_ld_post = MagicMock(spec_set=LinkData), MagicMock(spec_set=LinkData)
        out_ld_eval, out_ld_post = MagicMock(spec_set=LinkData), MagicMock(spec_set=LinkData)
        wmfn.ld_in_eval, wmfn.ld_in_post = in_ld_eval, in_ld_post
        wmfn.ld_out_eval, wmfn.ld_out_post = out_ld_eval, out_ld_post

        wmfn.compute()

        assert wmfn.quiescence

    def test_compute(self):
        # Test invocation with correct messages when calling compute()
        b_shape, p_shape, s_shape, e_shape = Size([1, 2]), Size([]), Size([10, 15, 20]), Size([2, 3, 4])
        msg_shape = (b_shape, p_shape, s_shape, e_shape)
        index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
        ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
        wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

        # Mock linkdata
        in_ld_eval, in_ld_post = MagicMock(spec_set=LinkData), MagicMock(spec_set=LinkData)
        out_ld_eval, out_ld_post = MagicMock(spec_set=LinkData), MagicMock(spec_set=LinkData)
        wmfn.ld_in_eval, wmfn.ld_in_post = in_ld_eval, in_ld_post
        wmfn.ld_out_eval, wmfn.ld_out_post = out_ld_eval, out_ld_post

        test_msg_1, test_msg_2 = random_message(MessageType.Particles, *msg_shape), \
                                 random_message(MessageType.Particles, *msg_shape)
        wmfn.eval_msg_cache, wmfn.post_msg_cache = test_msg_1, test_msg_2

        wmfn.compute()

        out_ld_eval.write.assert_called_once_with(test_msg_1)
        out_ld_post.write.assert_called_once_with(test_msg_2)

