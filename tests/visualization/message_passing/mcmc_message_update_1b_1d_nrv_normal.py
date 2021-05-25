"""
    Visualization for the message update step of Metropolis-Hastings MCMC algorithm, compartmentalized in the WMFN node.

    The goal of this visualization is to show that, given a *stable* target function, the message update step in WMFN
    can incrementally move the particles toward the target function's important region. The target function and outside
    linkdata and node connections are stubbed out for unit testing purpose.

    This script demonstrates the process with:
        - 1 batch
        - 1-dimensional for each random variable
        - 2 random variables, so the joint distribution is 2-dimensional
        - multivariate normal distribution
"""
from unittest.mock import MagicMock, patch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import Size
import torch.distributions as D
import torch.distributions.constraints as C
from pysigma.defs import Variable, VariableMetatype, Message, MessageType
from pysigma.graphical.basic_nodes import LinkData
from pysigma.graphical.predicate_nodes import WMFN_MCMC, WMVN
from tests.utils import generate_positive_definite

# Visualization parameters
num_ptcl = 30       # number of particles for EACH random variable
num_iter = 200
x_lim = [-20, 20]  # y_lim same as x_lim
normal_cov_scale = 4

x_plot_num = 100  # y_plot_num same as x_plot_num. There will be in total x_plot_num * y_plot_num plotting particles
histogram_n_bins = 20

if __name__ == "__main__":
    # Initialize WMFN and initial message
    b_shape, p_shape, s_shape, e_shape = Size([1]), Size([]), Size([num_ptcl, num_ptcl]), Size([1, 1])
    msg_shape = (b_shape, p_shape, s_shape, e_shape)
    index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
    ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
    wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

    # init_ptcl = torch.rand(Size([num_ptcl, 2])) * (x_lim[1] - x_lim[0]) + x_lim[0]
    init_ptcl_ls = [torch.rand(num_ptcl, 1) * (x_lim[1] - x_lim[0]) + x_lim[0] for i in range(2)]
    init_ptcl_dens = [torch.zeros(num_ptcl) for i in range(2)]
    init_ptcl_msg = Message(MessageType.Particles,
                            batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                            particles=init_ptcl_ls, weight=1, log_densities=init_ptcl_dens)
    wmfn.init_particles(init_ptcl_msg)

    # Initialize the target function
    x_half_lim = [x / 2 for x in x_lim]
    loc = torch.rand(2) * (x_half_lim[1] - x_half_lim[0]) + x_half_lim[0]
    cov = generate_positive_definite(Size([]), 2) * normal_cov_scale
    target_dist = D.MultivariateNormal(loc, cov)

    # Initialize mocked linkdata
    wmvn_in_post, wmvn_in_eval, wmvn_out_post, wmvn_out_eval = \
        MagicMock(spec_set=WMVN), MagicMock(spec_set=WMVN), MagicMock(spec_set=WMVN), MagicMock(spec_set=WMVN)
    ld_in_post = LinkData(wmvn_in_post, wmfn, to_fn=True, msg_shape=msg_shape, type='posterior')
    ld_in_eval = LinkData(wmvn_in_eval, wmfn, to_fn=True, msg_shape=msg_shape, type='evaluation')
    ld_out_post = LinkData(wmvn_out_post, wmfn, to_fn=False, msg_shape=msg_shape, type='posterior')
    ld_out_eval = LinkData(wmvn_out_eval, wmfn, to_fn=False, msg_shape=msg_shape, type='evaluation')

    wmfn.add_link(ld_in_post)
    wmfn.add_link(ld_in_eval)
    wmfn.add_link(ld_out_post)
    wmfn.add_link(ld_out_eval)

    # Inference loop
    # Debug
    sum_vals = []
    for i in range(num_iter):
        # Cognitive cycle: decision phase
        wmfn.compute()
        # Retrieve messages
        post_msg, eval_msg = ld_out_post.read(), ld_out_eval.read()
        # Evaluate against the target distributions
        # Create mesh grid and stack at the last dimension to restore the joint particles
        post_ptcl_array, eval_ptcl_array = [ptcl.squeeze() for ptcl in post_msg.particles], [ptcl.squeeze() for ptcl in eval_msg.particles]
        post_ptcl_meshgrid, eval_ptcl_meshgrid = torch.meshgrid(post_ptcl_array), torch.meshgrid(eval_ptcl_array)
        post_ptcl_stacked, eval_ptcl_stacked = torch.stack(post_ptcl_meshgrid, dim=-1), torch.stack(eval_ptcl_meshgrid, dim=-1)
        post_ptcl_flattened, eval_ptcl_flattened = post_ptcl_stacked.reshape(-1, 2), eval_ptcl_stacked.reshape(-1, 2)
        # Insert batch shape between sample shape and event shape
        post_log_prob, eval_log_prob = \
            target_dist.log_prob(post_ptcl_flattened), target_dist.log_prob(eval_ptcl_flattened)
        # Massage log density tensor shape: reshape to square matrix and prepend a singleton batch dimension
        post_log_prob, eval_log_prob = post_log_prob.reshape(s_shape).unsqueeze(dim=0), eval_log_prob.reshape(s_shape).unsqueeze(dim=0)
        # Generate re-weighted messages and send back
        post_reweighted_msg, eval_reweighted_msg = \
            post_msg.event_reweight(post_log_prob), eval_msg.event_reweight(eval_log_prob)
        ld_in_post.write(post_reweighted_msg)
        ld_in_eval.write(eval_reweighted_msg)

        # Cognitive cycle: modification phase
        wmfn.modify()
        wmfn.reset_state()

        # Print out sum of posterior particles' log densities
        sum_val = post_log_prob.sum().item()
        print("##### Iteration {} #####".format(i + 1))
        print("Posterior log densities sum: %.10f" % sum_val)

        sum_vals.append(sum_val)

    # Plot the value curves and particles concentration
    # Set seaborn theme
    sns.set()

    # fig, axs = plt.subplots(nrows=2, figsize=(8, 16))
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(8, 8))
    ax_loss.grid()

    # Log density sum value curve
    ax_loss.plot(sum_vals)
    ax_loss.title.set_text("Log Density Sum")
    plt.show()

    # Particles concentration
    x_range_np, y_range_np = np.arange(start=x_lim[0], stop=x_lim[1], step=(x_lim[1] - x_lim[0]) / x_plot_num), \
                             np.arange(start=x_lim[0], stop=x_lim[1], step=(x_lim[1] - x_lim[0]) / x_plot_num)
    x_grid_np, y_grid_np = np.meshgrid(x_range_np, y_range_np)
    x_grid, y_grid = torch.as_tensor(x_grid_np, dtype=torch.float), torch.as_tensor(y_grid_np, dtype=torch.float)

    # x_range, y_range = torch.arange(start=x_lim[0], end=x_lim[1], step=(x_lim[1] - x_lim[0]) / x_plot_num), \
    #                    torch.arange(start=x_lim[0], end=x_lim[1], step=(x_lim[1] - x_lim[0]) / x_plot_num)
    # x_grid, y_grid = torch.meshgrid(x_range, y_range)
    plot_grid = torch.stack([x_grid, y_grid], dim=-1)
    plot_grid_flattened = plot_grid.view(-1, plot_grid.shape[-1]).unsqueeze(dim=1)
    # Distribution curve
    dist_plot_prob_flattened = target_dist.log_prob(plot_grid_flattened).exp()
    # Summarize over batches
    dist_plot_prob_flattened = dist_plot_prob_flattened.sum(dim=-1)
    dist_plot_prob = dist_plot_prob_flattened.view(x_range_np.shape[0], y_range_np.shape[0])
    # x_grid_np, y_grid_np, dist_plot_prob_np = np.asarray(x_grid), np.asarray(y_grid), np.asarray(dist_plot_prob)
    dist_plot_prob_np = np.asarray(dist_plot_prob)

    # Create figure
    fig_ptcl = plt.figure(figsize=(8, 8))
    gs = fig_ptcl.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7), left=0.1, right=0.9, bottom=0.1,
                               top=0.9,
                               wspace=0.05, hspace=0.05)
    ax_ptcl = fig_ptcl.add_subplot(gs[1, 0])
    ax_histx = fig_ptcl.add_subplot(gs[0, 0], sharex=ax_ptcl)
    ax_histy = fig_ptcl.add_subplot(gs[1, 1], sharey=ax_ptcl)
    ax_histx.tick_params(axis='x', labelbottom=False)       # Remove tick labels
    ax_histy.tick_params(axis='y', labelleft=False)

    fig_ptcl.suptitle("Particles Concentration vs. Target Pdf")
    ax_loss.set_xlim(left=x_lim[0], right=x_lim[1])
    ax_loss.set_ylim(top=x_lim[1], bottom=x_lim[0])

    ax_ptcl.pcolormesh(x_range_np, y_range_np, dist_plot_prob_np, shading='gouraud')

    # Particles concentration
    # Scatter plot: scatter on the xy-plane
    # Display the restored joint particles grid
    post_joint_ptcl_x, post_joint_ptcl_y = post_ptcl_flattened.split(1, dim=-1)
    post_joint_ptcl_x_np, post_joint_ptcl_y_np = np.asarray(post_joint_ptcl_x), np.asarray(post_joint_ptcl_y)
    ax_ptcl.scatter(post_joint_ptcl_x_np, post_joint_ptcl_y_np, marker='+')

    # Marginal histogram
    post_ptcl_x, post_ptcl_y = np.asarray(post_ptcl_array[0]), np.asarray(post_ptcl_array[1])
    ax_histx.hist(post_ptcl_x, bins=histogram_n_bins, range=x_lim)
    ax_histy.hist(post_ptcl_y, bins=histogram_n_bins, range=x_lim, orientation='horizontal')

    plt.show()

    # Debug
    pass
