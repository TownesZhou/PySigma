from pysigma.graphical import Graph
from pysigma.graphical.basic_nodes import *


def generate_var_list(var_names, var_sizes):
    return [Variable(name, size) for name, size in zip(var_names, var_sizes)]


def generate_pt_var_info(wm_var_names, pt_vals):
    """
        pt_vals is a list of pattern values. In case of pattern variable, simply provide the variable as a 'str'.
            Otherwise, if a const, provide a list of int as the const values

        Relation assumed None
    """
    pt_var_info = {}
    const_count = 0
    for wm_var_name, pt_val in zip(wm_var_names, pt_vals):
        if isinstance(pt_val, str):     # pattern variable
            pt_var_info[wm_var_name] = {
                "name": pt_val,
                "type": "var",
                "vals": None,
                "rel": None
            }
        else:
            assert (pt_val is None) or (isinstance(pt_val, list) and all(isinstance(val, int) for val in pt_val))
            pt_var_info[wm_var_name] = {
                "name": "CONST_" + str(const_count),
                "type": "const",
                "vals": pt_val,
                "rel": None
            }
            const_count += 1
    return pt_var_info


def run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg):
    """
        Set up the graph with inward direction and compute message on ADFN. Return the message set on outgoing linkdata.
    """
    G = Graph()

    wmvn = G.new_node(WMVN, "working memory variable node", wm_var_list)
    ptvn = G.new_node(WMVN, "pattern variable node", pt_var_list)
    adfn = G.new_node(ADFN, "ADFN", pt_var_info)

    # inward direction
    G.add_unilink(wmvn, adfn)
    G.add_unilink(adfn, ptvn)

    in_ld = G.get_linkdata(wmvn, adfn)
    out_ld = G.get_linkdata(adfn, ptvn)

    # Set init message, compute, and read output message
    in_ld.set(init_msg, 10e-5)
    adfn.compute()
    out_msg = out_ld.read()

    return out_msg


def run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg):
    """
        Set up the graph with outward direction and compute message on ADFN. Return the message set on outgoing linkdata.
    """
    G = Graph()

    wmvn = G.new_node(WMVN, "working memory variable node", wm_var_list)
    ptvn = G.new_node(WMVN, "pattern variable node", pt_var_list)
    adfn = G.new_node(ADFN, "ADFN", pt_var_info)

    # outward direction
    G.add_unilink(ptvn, adfn)
    G.add_unilink(adfn, wmvn)

    in_ld = G.get_linkdata(ptvn, adfn)
    out_ld = G.get_linkdata(adfn, wmvn)

    # Set init message, compute, and read output message
    in_ld.set(init_msg, 10e-5)
    adfn.compute()
    out_msg = out_ld.read()

    return out_msg


class TestADFN:
    """
        Testing ADFN message passing
    """

    def test_simple_inward_1(self):
        """
            Simple variable swapping with matching size. Inward direction
                wm vars:  arg1  arg2  arg3
                wm size:   3     4     5
                pt vars:   x     y     z
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 4, 5])
        pt_var_list = generate_var_list(['x', 'y', 'z'], [3, 4, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', 'z'])

        # Set init message
        init_msg = torch.rand(3, 4, 5)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert torch.equal(init_msg, out_msg)

    def test_simple_enlarge_inward_1(self):
        """
            Simple variable swapping with unmatched size. Inward direction
                wm vars:  arg1  arg2  arg3
                wm size:   3     4     5
                pt vars:   x     y     z
                pt size:   10    4    5
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 4, 5])
        pt_var_list = generate_var_list(['x', 'y', 'z'], [10, 4, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', 'z'])

        # Set init message
        init_msg = torch.rand(3, 4, 5)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        extra = torch.zeros(7, 4, 5)
        true_msg = torch.cat([init_msg, extra], dim=0)
        assert torch.equal(out_msg, true_msg)

    def test_simple_enlarge_inward_2(self):
        """
            Simple variable swapping with unmatched size. Inward direction
                wm vars:  arg1  arg2  arg3
                wm size:   3     4     5
                pt vars:   x     y     z
                pt size:   10    10    10
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 4, 5])
        pt_var_list = generate_var_list(['x', 'y', 'z'], [10, 10, 10])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', 'z'])

        # Set init message
        init_msg = torch.rand(3, 4, 5)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        extra1 = torch.zeros(7, 4, 5)
        buf = torch.cat([init_msg, extra1], dim=0)
        extra2 = torch.zeros(10, 6, 5)
        buf = torch.cat([buf, extra2], dim=1)
        extra3 = torch.zeros(10, 10, 5)
        true_msg = torch.cat([buf, extra3], dim=2)
        assert torch.equal(out_msg, true_msg)

    def test_bind_inward_1(self):
        """
            Variable binding with matching size. Inward direction
                wm vars: arg1  arg2  arg3
                wm size:  3     3     5
                pt vars:  x     x     y
                pt_size:  3     -     5
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 3, 5])
        pt_var_list = generate_var_list(['x', 'y'], [3, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'x', 'y'])

        init_msg = torch.rand(3, 3, 5)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        buf = init_msg.diagonal(dim1=0, dim2=1)
        true_msg = buf.permute((1, 0))

        assert torch.equal(out_msg, true_msg)

    def test_bind_inward_2(self):
        """
            Variable binding with matching size. Inward direction
                wm vars: arg1  arg2  arg3  arg4
                wm size:  3     3     5     3
                pt vars:  x     x     y     x
                pt_size:  3     -     5     -
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3', 'arg4'], [3, 3, 5, 3])
        pt_var_list = generate_var_list(['x', 'y'], [3, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3', 'arg4'], ['x', 'x', 'y', 'x'])

        init_msg = torch.rand(3, 3, 5, 3)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # # Check
        assert all(torch.equal(init_msg[i, i, j, i], out_msg[i, j]) for i in range(3) for j in range(5))

    def test_bind_inward_3(self):
        """
            Variable binding with unmatched size. Inward direction
                wm vars: arg1  arg2  arg3  arg4
                wm size:  8     4     5     3
                pt vars:  x     x     y     x
                pt_size:  3     -     5     -
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3', 'arg4'], [8, 4, 5, 3])
        pt_var_list = generate_var_list(['x', 'y'], [3, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3', 'arg4'], ['x', 'x', 'y', 'x'])

        init_msg = torch.rand(8, 4, 5, 3)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # # Check
        assert all(torch.equal(init_msg[i, i, j, i], out_msg[i, j]) for i in range(3) for j in range(5))

    def test_bind_inward_4(self):
        """
            Variable binding with unmatched size. Inward direction
                wm vars: arg1  arg2  arg3  arg4  arg5
                wm size:  8     4     5     3     4
                pt vars:  x     x     y     x     y
                pt_size:  -     -     -     3     4
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3', 'arg4', 'arg5'], [8, 4, 5, 3, 4])
        pt_var_list = generate_var_list(['x', 'y'], [3, 4])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3', 'arg4', 'arg5'], ['x', 'x', 'y', 'x', 'y'])

        init_msg = torch.rand(8, 4, 5, 3, 4)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # # Check
        assert all(torch.equal(init_msg[i, i, j, i, j], out_msg[i, j]) for i in range(3) for j in range(4))

    def test_bind_inward_5(self):
        """
            Variable binding with unmatched size with dimension size enlargement involved. Inward direction
                wm vars: arg1  arg2  arg3  arg4  arg5
                wm size:  8     4     5     3     4
                pt vars:  x     x     y     x     y
                pt_size:  -     -     -     5     5
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3', 'arg4', 'arg5'], [8, 4, 5, 3, 4])
        pt_var_list = generate_var_list(['x', 'y'], [5, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3', 'arg4', 'arg5'], ['x', 'x', 'y', 'x', 'y'])

        init_msg = torch.rand(8, 4, 5, 3, 4)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # # Check
        assert all(torch.equal(init_msg[i, i, j, i, j], out_msg[i, j]) for i in range(3) for j in range(4))
        assert all(torch.equal(torch.zeros(5), out_msg[i, :]) for i in range(3, 5))
        assert all(torch.equal(torch.zeros(5), out_msg[:, i]) for i in range(4, 5))

    def test_const_inward_1(self):
        """
            Constant values. Inward direction
                wm vars: arg1  arg2  arg3
                wm size:  3     4     5
                pt vars:  x     y    [1, 2, 3]
                pt size:  3     4     3
        """
        const_vals = [1, 2, 3]
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 4, 5])
        pt_var_list = generate_var_list(['x', 'y', 'CONST_0'], [3, 4, 3])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', const_vals])

        init_msg = torch.rand(3, 4, 5)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert torch.equal(init_msg[:, :, 1:4], out_msg)

    def test_const_inward_2(self):
        """
            Constant values, with out-of-order values. Inward direction
                wm vars: arg1  arg2  arg3
                wm size:  3     4     5
                pt vars:  x     y    [2, 1, 3]
                pt size:  3     4     3
        """
        const_vals = [2, 1, 3]
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 4, 5])
        pt_var_list = generate_var_list(['x', 'y', 'CONST_0'], [3, 4, 3])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', const_vals])

        init_msg = torch.rand(3, 4, 5)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert all(torch.equal(init_msg[:, :, i], out_msg[:, :, j]) for i, j in zip(const_vals, range(3)))

    def test_const_inward_3(self):
        """
            Constant values, mixing None const with valued const
                wm vars: arg1  arg2  arg3
                wm size:     3     4     5
                pt vars:  [0, 2]  None  None
                pt size:     2     4     5
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 4, 5])
        pt_var_list = generate_var_list(['CONST_0', 'CONST_1', 'CONST_2'], [2, 4, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], [[0, 2], None, None])

        init_msg = torch.rand(3, 4, 5)
        out_msg = run_inward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert all(torch.equal(init_msg[i, :, :], out_msg[j, :, :]) for j, i in enumerate([0, 2]))

    def test_simple_outward_1(self):
        """
            Simple variable swapping with matching size. Outward direction
                pt vars:    x     y     z
                pt size:    3     4     5
                wm vars:  arg1  arg2  arg3
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 4, 5])
        pt_var_list = generate_var_list(['x', 'y', 'z'], [3, 4, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', 'z'])

        init_msg = torch.rand(3, 4, 5)
        out_msg = run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert torch.equal(init_msg, out_msg)

    def test_simple_shrink_outward_1(self):
        """
            Simple variable swapping with unmatched size. Outward direction
                pt vars:    x     y     z
                pt size:    3     4     5
                wm vars:  arg1  arg2  arg3
                wm size:    3     4     3
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 4, 3])
        pt_var_list = generate_var_list(['x', 'y', 'z'], [3, 4, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', 'z'])

        init_msg = torch.rand(3, 4, 5)
        out_msg = run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert torch.equal(init_msg[:, :, :3], out_msg)

    def test_simple_shrink_outward_2(self):
        """
            Simple variable swapping with unmatched size. Outward direction
                pt vars:    x     y     z
                pt size:    3     4     5
                wm vars:  arg1  arg2  arg3
                wm size:    3     3     3
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 3, 3])
        pt_var_list = generate_var_list(['x', 'y', 'z'], [3, 4, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', 'z'])

        init_msg = torch.rand(3, 4, 5)
        out_msg = run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert torch.equal(init_msg[:, :3, :3], out_msg)

    def test_bind_outward_1(self):
        """
            Variable binding with matching size. Outward direction
                pt vars:    x     x     y
                pt size:    3     3     5
                wm vars:  arg1  arg2  arg3
                wm size:    3     3     5
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 3, 5])
        pt_var_list = generate_var_list(['x', 'y'], [3, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'x', 'y'])

        init_msg = torch.rand(3, 5)
        out_msg = run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert all(torch.equal(init_msg[i, :], out_msg[i, i, :]) for i in range(3))
        assert all(torch.equal(torch.zeros(5), out_msg[i, j, :]) for i in range(3) for j in range(3) if i != j)

    def test_bind_outward_2(self):
        """
            Variable binding with unmatched size. Outward direction
                pt vars:    x     x     y
                pt size:    5     5     8
                wm vars:  arg1  arg2  arg3
                wm size:    3     2     8
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 2, 8])
        pt_var_list = generate_var_list(['x', 'y'], [5, 8])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'x', 'y'])

        init_msg = torch.rand(5, 8)
        out_msg = run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert all(torch.equal(init_msg[i, :], out_msg[i, i, :]) for i in range(2))
        assert all(torch.equal(torch.zeros(8), out_msg[i, j, :]) for i in range(3) for j in range(2) if i != j)

    def test_bind_outward_3(self):
        """
            Variable binding with unmatched size. Outward direction
                pt vars:    x     x     y    x
                pt size:    5     5     8    5
                wm vars:  arg1  arg2  arg3  arg4
                wm size:    3     2     8    5
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3', 'arg4'], [3, 2, 8, 5])
        pt_var_list = generate_var_list(['x', 'y'], [5, 8])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3', 'arg4'], ['x', 'x', 'y', 'x'])

        init_msg = torch.rand(5, 8)
        out_msg = run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert all(torch.equal(init_msg[i, :], out_msg[i, i, :, i]) for i in range(2))
        assert all(torch.equal(torch.zeros(8), out_msg[i, j, :, k])
                   for i in range(3) for j in range(2) for k in range(5)
                   if i != j or i != k or j != k)

    def test_bind_outward_4(self):
        """
            Variable binding with unmatched size. Outward direction
                pt vars:    x     x     y     x     y
                pt size:    5     5     8     5     8
                wm vars:  arg1  arg2  arg3  arg4  arg5
                wm size:    3     2     8     5     6
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3', 'arg4', 'arg5'], [3, 2, 8, 5, 6])
        pt_var_list = generate_var_list(['y', 'x'], [8, 5])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3', 'arg4', 'arg5'], ['x', 'x', 'y', 'x', 'y'])

        init_msg = torch.rand(8, 5)
        out_msg = run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert all(torch.equal(init_msg[j, i], out_msg[i, i, j, i, j]) for i in range(2) for j in range(6))

        assert all(torch.equal(torch.tensor(0.), out_msg[a, b, c, d, e])
                   for a in range(3) for b in range(2) for c in range(8) for d in range(5) for e in range(6)
                   if a != b or a != d or b != d or c != e)

    def test_const_outward_1(self):
        """
            Variable binding with unmatched size. Outward direction
                pt vars:    x     y   [1, 2, 3]
                pt size:    5     5       3
                wm vars:  arg1  arg2    arg3
                wm size:    3     2       8
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 2, 8])
        pt_var_list = generate_var_list(['x', 'y', 'CONST_0'], [5, 5, 3])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', [1, 2, 3]])

        init_msg = torch.rand(5, 5, 3)
        out_msg = run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert torch.equal(init_msg[:3, :2, :3], out_msg[:, :, 1:4])
        assert all(torch.equal(torch.zeros(3, 2), out_msg[:, :, i]) for i in range(8) if i not in [1, 2, 3])

    def test_const_outward_2(self):
        """
            Variable binding with unmatched size. Outward direction
                pt vars:    x     y   [3, 5, 2]
                pt size:    5     5       3
                wm vars:  arg1  arg2    arg3
                wm size:    3     2       8
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 2, 8])
        pt_var_list = generate_var_list(['x', 'y', 'CONST_0'], [5, 5, 3])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', [3, 5, 2]])

        init_msg = torch.rand(5, 5, 3)
        out_msg = run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert all(torch.equal(init_msg[:3, :2, i], out_msg[:, :, j]) for i, j in enumerate([3, 5, 2]))
        assert all(torch.equal(torch.zeros(3, 2), out_msg[:, :, i]) for i in range(8) if i not in [3, 5, 2])

    def test_const_outward_3(self):
        """
            Variable binding with unmatched size. Outward direction
                pt vars:    x     y   None
                pt size:    5     5     3
                wm vars:  arg1  arg2  arg3
                wm size:    3     2     8
        """
        wm_var_list = generate_var_list(['arg1', 'arg2', 'arg3'], [3, 2, 8])
        pt_var_list = generate_var_list(['x', 'y', 'CONST_0'], [5, 5, 3])
        pt_var_info = generate_pt_var_info(['arg1', 'arg2', 'arg3'], ['x', 'y', None])

        init_msg = torch.rand(5, 5, 3)
        out_msg = run_outward(wm_var_list, pt_var_list, pt_var_info, init_msg)

        # Check
        assert torch.equal(init_msg[:3, :2, :], out_msg[:, :, :3])
        assert all(torch.equal(torch.zeros(3, 2), out_msg[:, :, i]) for i in range(3, 8))
