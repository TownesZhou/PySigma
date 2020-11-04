"""
    Unit tests for LinkData class
"""

import pytest
from unittest.mock import MagicMock
import torch
from torch import Size

from pysigma.graphical.basic_nodes import LinkData, VariableNode, FactorNode
from pysigma.defs import Message, MessageType

# Numerical accuracy
EPS = 1e-6


class TestLinkData:

    def test_init_correct(self):
        # Mock variable node and factor node
        mock_vn = MagicMock(spec_set=VariableNode)
        mock_fn = MagicMock(spec_set=FactorNode)
        to_fn = True
        msg_shape = (Size([1]), Size([2]), Size([3, 4]), Size([5, 6]))
        epsilon = 1e-5
        attr = {"a": 1, "b": 2, "c": 3}

        ld = LinkData(mock_vn, mock_fn, to_fn, msg_shape, epsilon, **attr)

        assert ld.vn is mock_vn and ld.fn is mock_fn and ld.to_fn is to_fn and ld.msg_shape is msg_shape and \
            ld.epsilon == epsilon and ld.attr == attr
        assert ld.memory.isid
        assert not ld.new

    def test_str(self):
        # Mock variable node and factor node
        # Use spec instead of spec_set here because mock seems unable to grab attribute "name" from the parent class
        #   "Node" of child class "VariableNode" or "FactorNode"
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([1]), Size([2]), Size([3, 4]), Size([5, 6]))

        # Test 1: direction to factor node
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)
        assert str(ld) == "test_variable_node --> test_factor_node"

        # Test 2: direction to variable node
        ld = LinkData(mock_vn, mock_fn, False, msg_shape)
        assert str(ld) == "test_factor_node --> test_variable_node"

    def test_reset_shape(self):
        mock_vn = MagicMock(spec_set=VariableNode)
        mock_fn = MagicMock(spec_set=FactorNode)
        to_fn = True
        msg_shape_1 = (Size([1]), Size([2]), Size([3, 4]), Size([5, 6]))
        msg_shape_2 = (Size([1]), Size([2]), Size([7, 8]), Size([9, 10]))

        ld = LinkData(mock_vn, mock_fn, to_fn, msg_shape_1)
        ld.memory = Message(MessageType.Parameter,
                            batch_shape=Size([10]), param_shape=Size([5]), parameter=torch.randn([10, 5]))
        ld.new = True

        ld.reset_shape(msg_shape_2)
        assert ld.memory == Message.identity()
        assert not ld.new

    def test_write_incompatible_shape(self):
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([1]), Size([2]), Size([3, 4]), Size([5, 6]))

        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        # Test 1: Parameter type new message - wrong batch shape
        new_msg = Message(MessageType.Parameter,
                          batch_shape=Size([10]), param_shape=Size([2]), parameter=torch.randn([10, 2]))
        with pytest.raises(AssertionError):
            ld.write(new_msg)

        # Test 2: Parameter type new message - wrong param shape
        new_msg = Message(MessageType.Parameter,
                          batch_shape=Size([1]), param_shape=Size([5]), parameter=torch.randn([1, 5]))
        with pytest.raises(AssertionError):
            ld.write(new_msg)

        # Test 3: particles type new message - wrong batch shape
        new_msg = Message(MessageType.Particles,
                          batch_shape=Size([10]), sample_shape=Size([3, 4]), event_shape=Size([5, 6]),
                          weight=torch.rand([10, 3, 4]),
                          particles=[torch.randn([3, 5]), torch.randn([4, 6])],
                          log_densities=[torch.randn([3]), torch.randn([4])])
        with pytest.raises(AssertionError):
            ld.write(new_msg)

        # Test 4: particles type new message - wrong sample shape
        new_msg = Message(MessageType.Particles,
                          batch_shape=Size([1]), sample_shape=Size([3, 10]), event_shape=Size([5, 6]),
                          weight=torch.rand([1, 3, 10]),
                          particles=[torch.randn([3, 5]), torch.randn([10, 6])],
                          log_densities=[torch.randn([3]), torch.randn([10])])
        with pytest.raises(AssertionError):
            ld.write(new_msg)

        # Test 4: particles type new message - wrong event shape
        new_msg = Message(MessageType.Particles,
                          batch_shape=Size([1]), sample_shape=Size([3, 4]), event_shape=Size([5, 10]),
                          weight=torch.rand([1, 3, 4]),
                          particles=[torch.randn([3, 5]), torch.randn([4, 10])],
                          log_densities=[torch.randn([3]), torch.randn([4])])
        with pytest.raises(AssertionError):
            ld.write(new_msg)

    def test_write_check_diff_off(self):
        # Test that if check_diff=False, the memory will be replaced with new content anyway
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([1]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        new_msg = Message.identity()
        old_msg = ld.memory
        # Check that the new message's content is exactly the same as the old (default) memory content
        assert ld.memory == new_msg

        # Write to linkdata and check that the new message is no longer the same instance as the old one
        ld.write(new_msg, check_diff=False)
        assert ld.memory is not old_msg
        assert ld.new

    def test_write_check_diff_on_reject_default(self):
        # Test initial linkdata rejects identity messages of default type when check_diff is True
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([1]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)
        old_msg = ld.memory

        # Test general identity message
        new_msg = Message.identity()
        ld.write(new_msg, check_diff=True)
        assert ld.memory is old_msg
        assert not ld.new

    def test_write_check_diff_on_reject_parameter(self):
        # Test linkdata rejects parameter message when check_diff is True
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        old_param = torch.randn([10, 2])
        old_msg = Message(MessageType.Parameter,
                          batch_shape=Size([10]), param_shape=Size([2]), parameter=old_param)
        # Slight perturbation to old parameter
        new_param = old_param + 1e-8
        new_msg = Message(MessageType.Parameter,
                          batch_shape=Size([10]), param_shape=Size([2]), parameter=new_param)

        # Set linkdata memory to old message
        ld.memory = old_msg

        # Write and assert
        ld.write(new_msg, check_diff=True)
        assert ld.memory is old_msg
        assert not ld.new

    def test_write_check_diff_on_reject_particles(self):
        # Test linkdata rejects parameter message when check_diff is True
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        ptcl = [torch.randn([3, 5]), torch.randn([4, 6])]
        dens = [torch.randn([3]), torch.randn([4])]
        old_weight = torch.rand([10, 3, 4])
        old_msg = Message(MessageType.Particles,
                          Size([10]), Size([2]), Size([3, 4]), Size([5, 6]),
                          particles=ptcl, log_densities=dens, weight=old_weight)
        # Slight perturbation to weight
        new_weight = old_weight + 1e-8
        new_msg = Message(MessageType.Particles,
                          Size([10]), Size([2]), Size([3, 4]), Size([5, 6]),
                          particles=ptcl, log_densities=dens, weight=new_weight)

        # Set linkdata memory to old message
        ld.memory = old_msg

        # Write and assert
        ld.write(new_msg, check_diff=True)
        assert ld.memory is old_msg
        assert not ld.new

    def test_write_check_diff_on_reject_both(self):
        # Test linkdata rejects both type message when check_diff is True
        # based ONLY on the difference between parameters, even if particle weights are totally different
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        ptcl = [torch.randn([3, 5]), torch.randn([4, 6])]
        dens = [torch.randn([3]), torch.randn([4])]
        old_weight = torch.rand([10, 3, 4])
        old_param = torch.randn([10, 2])
        old_msg = Message(MessageType.Both,
                          Size([10]), Size([2]), Size([3, 4]), Size([5, 6]),
                          parameter=old_param,
                          particles=ptcl, log_densities=dens, weight=old_weight)
        # Slight perturbation to parameter, but totally different new weight
        new_weight = torch.rand([10, 3, 4])
        new_param = old_param + 1e-8
        new_msg = Message(MessageType.Both,
                          Size([10]), Size([2]), Size([3, 4]), Size([5, 6]),
                          parameter=new_param,
                          particles=ptcl, log_densities=dens, weight=new_weight)

        # Set linkdata memory to old message
        ld.memory = old_msg

        # Write and assert
        ld.write(new_msg, check_diff=True)
        assert ld.memory is old_msg
        assert not ld.new

    def test_write_check_diff_on_accept_differnt_type(self):
        # Test that new message of different type would be accepted even if its contents are identical
        # Use a typed identity message to test against the initial default type identity message
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        old_msg = ld.memory
        # Test parameter type
        new_msg = Message.identity(MessageType.Parameter)
        ld.write(new_msg, check_diff=True)
        assert ld.memory is new_msg
        assert ld.new

        # Test particles type
        new_msg = Message.identity(MessageType.Particles)
        ld.write(new_msg, check_diff=True)
        assert ld.memory is new_msg
        assert ld.new

    def test_write_check_diff_on_accept_different_content_1(self):
        # Test that new message is accepted if it has different content from the old message
        # Tes 1: test parameter message
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        old_msg = Message(MessageType.Parameter,
                          batch_shape=Size([10]), param_shape=Size([2]), parameter=torch.randn([10, 2]))
        new_msg = Message(MessageType.Parameter,
                          batch_shape=Size([10]), param_shape=Size([2]), parameter=torch.randn([10, 2]))
        ld.memory = old_msg

        ld.write(new_msg, check_diff=True)
        assert ld.memory is new_msg
        assert ld.new

    def test_write_check_diff_on_accept_different_content_2(self):
        # Test 2: test particles message with incompatible particles
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        # Scenario 1: different particle values, same densities and weight
        dens = [torch.randn([3]), torch.randn([4])]
        weight = torch.rand([10, 3, 4])
        old_msg = Message(MessageType.Particles,
                          batch_shape=Size([10]), sample_shape=Size([3, 4]), event_shape=Size([5, 6]),
                          particles=[torch.randn([3, 5]), torch.randn([4, 6])],
                          weight=weight, log_densities=dens)
        new_msg = Message(MessageType.Particles,
                          batch_shape=Size([10]), sample_shape=Size([3, 4]), event_shape=Size([5, 6]),
                          particles=[torch.randn([3, 5]), torch.randn([4, 6])],
                          weight=weight, log_densities=dens)
        ld.memory = old_msg

        ld.write(new_msg, check_diff=True)
        assert ld.memory is new_msg
        assert ld.new

        # Scenario 2: different densities, same particle values and weight
        ptcl = [torch.randn([3, 5]), torch.randn([4, 6])]
        weight = torch.rand([10, 3, 4])
        old_msg = Message(MessageType.Particles,
                          batch_shape=Size([10]), sample_shape=Size([3, 4]), event_shape=Size([5, 6]),
                          particles=ptcl, weight=weight,
                          log_densities=[torch.randn([3]), torch.randn([4])])
        new_msg = Message(MessageType.Particles,
                          batch_shape=Size([10]), sample_shape=Size([3, 4]), event_shape=Size([5, 6]),
                          particles=ptcl, weight=weight,
                          log_densities=[torch.randn([3]), torch.randn([4])])
        ld.memory = old_msg

        ld.write(new_msg, check_diff=True)
        assert ld.memory is new_msg
        assert ld.new

    def test_write_check_diff_on_accept_different_content_3(self):
        # Test 3: test particles message with compatible particles but different weight
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        ptcl = [torch.randn([3, 5]), torch.randn([4, 6])]
        dens = [torch.randn([3]), torch.randn([4])]
        old_msg = Message(MessageType.Particles,
                          batch_shape=Size([10]), sample_shape=Size([3, 4]), event_shape=Size([5, 6]),
                          particles=ptcl,
                          weight=torch.rand([10, 3, 4]),
                          log_densities=dens)
        new_msg = Message(MessageType.Particles,
                          batch_shape=Size([10]), sample_shape=Size([3, 4]), event_shape=Size([5, 6]),
                          particles=ptcl,
                          weight=torch.rand([10, 3, 4]),
                          log_densities=dens)
        ld.memory = old_msg

        ld.write(new_msg, check_diff=True)
        assert ld.memory is new_msg
        assert ld.new

    def test_write_check_diff_on_accept_different_content_3(self):
        # Test 4: test Both type messages
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        # Scenario 1: same particle values and weight, but different params
        ptcl = [torch.randn([3, 5]), torch.randn([4, 6])]
        dens = [torch.randn([3]), torch.randn([4])]
        weight = torch.rand([10, 3, 4])
        old_msg = Message(MessageType.Both,
                          Size([10]), Size([2]), Size([3, 4]), Size([5, 6]),
                          parameter=torch.randn([10, 2]),
                          particles=ptcl, weight=weight, log_densities=dens)
        new_msg = Message(MessageType.Both,
                          Size([10]), Size([2]), Size([3, 4]), Size([5, 6]),
                          parameter=torch.randn([10, 2]),
                          particles=ptcl, weight=weight, log_densities=dens)
        ld.memory = old_msg

        ld.write(new_msg, check_diff=True)
        assert ld.memory is new_msg
        assert ld.new

        # Scenario 2: same particle values, but both parameter and weight different
        old_msg = Message(MessageType.Both,
                          Size([10]), Size([2]), Size([3, 4]), Size([5, 6]),
                          parameter=torch.randn([10, 2]),
                          particles=ptcl,
                          weight=torch.rand([10, 3, 4]),
                          log_densities=dens)
        new_msg = Message(MessageType.Both,
                          Size([10]), Size([2]), Size([3, 4]), Size([5, 6]),
                          parameter=torch.randn([10, 2]),
                          particles=ptcl,
                          weight=torch.rand([10, 3, 4]),
                          log_densities=dens)
        ld.memory = old_msg

        ld.write(new_msg, check_diff=True)
        assert ld.memory is new_msg
        assert ld.new

    def test_write_clone_no_clone_when_same(self):
        # Test 1: test that no cloning happens when new message is not different from old one when check_diff is True
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        # Use parameter message as testing example
        param = torch.randn([10, 2])
        old_msg = Message(MessageType.Parameter,
                          batch_shape=Size([10]), param_shape=Size([2]), parameter=param)
        new_msg = Message(MessageType.Parameter,
                          batch_shape=Size([10]), param_shape=Size([2]), parameter=param)
        ld.memory = old_msg

        ld.write(new_msg, check_diff=True, clone=True)
        assert ld.memory is old_msg
        assert not ld.new

    def test_write_clone_do_clone_when_different(self):
        # Test 2: test that cloning is taken when messages are different
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        # Use parameter message as testing example
        old_msg = Message(MessageType.Parameter,
                          batch_shape=Size([10]), param_shape=Size([2]), parameter=torch.randn([10, 2]))
        new_msg = Message(MessageType.Parameter,
                          batch_shape=Size([10]), param_shape=Size([2]), parameter=torch.randn([10, 2]))
        ld.memory = old_msg

        ld.write(new_msg, check_diff=True, clone=True)
        # Check that memory is a different instance than the new message
        assert ld.memory is not old_msg and ld.memory is not new_msg
        assert ld.memory == new_msg

    def test_read_no_clone(self):
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        # Use parameter message as testing example
        msg = Message(MessageType.Parameter,
                          batch_shape=Size([10]), param_shape=Size([2]), parameter=torch.randn([10, 2]))
        ld.write(msg, check_diff=True)

        read_msg = ld.read()
        assert read_msg is msg
        assert not ld.new

    def test_read_do_clone(self):
        mock_vn = MagicMock(spec=VariableNode)
        mock_vn.name = "test_variable_node"
        mock_fn = MagicMock(spec=FactorNode)
        mock_fn.name = "test_factor_node"
        msg_shape = (Size([10]), Size([2]), Size([3, 4]), Size([5, 6]))
        ld = LinkData(mock_vn, mock_fn, True, msg_shape)

        # Use parameter message as testing example
        msg = Message(MessageType.Parameter,
                      batch_shape=Size([10]), param_shape=Size([2]), parameter=torch.randn([10, 2]))
        ld.write(msg, check_diff=True)

        read_msg = ld.read(clone=True)
        assert read_msg is not msg and read_msg == msg
        assert not ld.new
