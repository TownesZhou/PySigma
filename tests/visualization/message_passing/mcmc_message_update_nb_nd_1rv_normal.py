"""
    Visualization for the message update step of Metropolis-Hastings MCMC algorithm, compartmentalized in the WMFN node.

    The goal of this visualization is to show that, given a *stable* target function, the message update step in WMFN
    can incrementally move the particles toward the target function's important region. The target function and outside
    linkdata and node connections are stubbed out for unit testing purpose.

    This script demonstrates the process with:
        - Multiple batch
        - 2-dimensional
        - 1 random variable
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
n_batches = 3
num_ptcl = 200
num_iter = 200
x_lim = [-20, 20]  # y_lim same as x_lim
normal_cov_scale = 4

x_plot_num = 100  # y_plot_num same as x_plot_num. There will be in total x_plot_num * y_plot_num plotting particles
histogram_n_bins = 50

if __name__ == "__main__":
    # Initialize WMFN and initial message
    b_shape, p_shape, s_shape, e_shape = Size([n_batches]), Size([]), Size([num_ptcl]), Size([2])
    msg_shape = (b_shape, p_shape, s_shape, e_shape)
    index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
    ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
    wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

    init_ptcl = torch.rand(Size([num_ptcl, 2])) * (x_lim[1] - x_lim[0]) + x_lim[0]
    init_ptcl_msg = Message(MessageType.Particles,
                            batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                            particles=[init_ptcl], weight=1, log_densities=[torch.zeros(s_shape)])
    wmfn.init_particles(init_ptcl_msg)

    # Initialize the target function
    x_half_lim = [x / 2 for x in x_lim]
    loc = torch.rand(n_batches, 2) * (x_half_lim[1] - x_half_lim[0]) + x_half_lim[0]
    cov = generate_positive_definite(b_shape, 2) * normal_cov_scale
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
        post_ptcl, eval_ptcl = post_msg.particles[0], eval_msg.particles[0]
        # Insert batch shape between sample shape and event shape
        post_ptcl, eval_ptcl = post_ptcl.unsqueeze(dim=1), eval_ptcl.unsqueeze(dim=1)
        post_log_prob, eval_log_prob = \
            target_dist.log_prob(post_ptcl), target_dist.log_prob(eval_ptcl)
        # Massage log density tensor shape: swap batch and sample dimension
        post_log_prob, eval_log_prob = post_log_prob.permute((1, 0)).contiguous(), eval_log_prob.permute((1, 0)).contiguous()
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
    fig, axs = plt.subplots(nrows=2, figsize=(8, 16))
    for ax in axs:
        ax.grid()

    # Set seaborn theme
    sns.set(style='darkgrid')

    # Log density sum value curve
    axs[0].plot(sum_vals)
    axs[0].title.set_text("Log Density Sum")

    # Particles concentration
    axs[1].title.set_text("Particles Concentration vs. Target Pdf")
    axs[1].set_xlim(left=x_lim[0], right=x_lim[1])
    axs[1].set_ylim(top=x_lim[1], bottom=x_lim[0])

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

    # axs[1, 0].remove()
    # axs[1, 0] = fig.add_subplot(2, 1, (1, 2), projection='3d')
    # axs[1, 0].plot_surface(x_grid_np, y_grid_np, dist_plot_prob_np)

    # axs[1, 0].contour([x_range_np, y_range_np], dist_plot_prob_np)

    axs[1].pcolormesh(x_range_np, y_range_np, dist_plot_prob_np, shading='gouraud')

    # Particles concentration
    # Scatter plot: scatter on the xy-plane
    post_ptcl_x, post_ptcl_y = post_ptcl.split(1, dim=-1)
    post_ptcl_x, post_ptcl_y = post_ptcl_x.squeeze(), post_ptcl_y.squeeze()
    post_ptcl_x_np, post_ptcl_y_np = np.asarray(post_ptcl_x), np.asarray(post_ptcl_y)
    axs[1].scatter(post_ptcl_x_np, post_ptcl_y_np)

    # 2D particles concentration 2D histogram
    # axs[1, 1].title.set_text("Particles Concentration in 2D histogram")
    # axs[1, 1].set_xlim(left=x_lim[0], right=x_lim[1])
    # axs[1, 1].set_ylim(top=x_lim[1], bottom=x_lim[0])
    # axs[1, 1].hist2d(post_ptcl_x_np, post_ptcl_y_np, range=[x_lim, x_lim], bins=histogram_n_bins)

    plt.show()

    # Debug
    pass
