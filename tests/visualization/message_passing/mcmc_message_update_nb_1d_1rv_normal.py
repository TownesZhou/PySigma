"""
    Visualization for the message update step of Metropolis-Hastings MCMC algorithm, compartmentalized in the WMFN node.

    The goal of this visualization is to show that, given a *stable* target function, the message update step in WMFN
    can incrementally move the particles toward the target function's important region. The target function and outside
    linkdata and node connections are stubbed out for unit testing purpose.

    This script demonstrates the process with:
        - Multiple batches
        - 1-dimensional
        - 1 random variable
        - univariate normal distribution
"""
from unittest.mock import MagicMock, patch
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Size
import torch.distributions as D
import torch.distributions.constraints as C
from pysigma.defs import Variable, VariableMetatype, Message, MessageType
from pysigma.graphical.basic_nodes import LinkData
from pysigma.graphical.predicate_nodes import WMFN_MCMC, WMVN

# Visualization parameters
n_batches = 3
num_ptcl = 100
num_iter = 100
x_lim = [-20, 20]
normal_std = 1.

x_plot_num = 1000
histogram_n_bins = 20


if __name__ == "__main__":
    # Initialize WMFN and initial message
    b_shape, p_shape, s_shape, e_shape = Size([n_batches]), Size([]), Size([num_ptcl]), Size([1])
    msg_shape = (b_shape, p_shape, s_shape, e_shape)
    index_var_list = [Variable("test_index_var", VariableMetatype.Indexing, s_size, None) for s_size in s_shape]
    ran_var_list = [Variable("test_random_var", VariableMetatype.Random, e_size, (C.real,)) for e_size in e_shape]
    wmfn = WMFN_MCMC("test_wmfn", index_var_list, ran_var_list)

    init_ptcl = torch.rand(Size([num_ptcl, 1])) * (x_lim[1] - x_lim[0]) + x_lim[0]
    init_ptcl_msg = Message(MessageType.Particles,
                            batch_shape=b_shape, sample_shape=s_shape, event_shape=e_shape,
                            particles=[init_ptcl], weight=1, log_densities=[torch.zeros(s_shape)])
    wmfn.init_particles(init_ptcl_msg)

    # Initialize the target function
    x_half_lim = [x/2 for x in x_lim]
    batch_loc_interval = (x_half_lim[1] - x_half_lim[0]) / n_batches
    batch_loc_start = [x_half_lim[0] + batch_loc_interval * i for i in range(n_batches)]
    loc = torch.rand(n_batches) * batch_loc_interval + torch.tensor(batch_loc_start, dtype=torch.float)
    scale = torch.ones(n_batches) * normal_std
    target_dist = D.Normal(loc=loc, scale=scale)

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
        post_log_prob, eval_log_prob = \
            target_dist.log_prob(post_ptcl), target_dist.log_prob(eval_ptcl)
        # Massage log density tensor shape: Swap sample and batch dimension so that batch dim is at front
        post_log_prob, eval_log_prob = post_log_prob.permute([1, 0]).contiguous(), eval_log_prob.permute([1, 0]).contiguous()
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
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 24))
    for ax in axs:
        ax.grid()

    # Log density sum value curve
    axs[0].plot(sum_vals)
    axs[0].title.set_text("Log Density Sum")

    # Particles concentration
    axs[1].title.set_text("Particles Concentration vs. Target Pdf")
    axs[1].set_xlim(left=x_lim[0], right=x_lim[1])
    x_plot = torch.arange(start=x_lim[0], end=x_lim[1], step=(x_lim[1] - x_lim[0]) / x_plot_num)
    x_plot_np = np.asarray(x_plot)
    # Distribution curve
    dist_plot_log_prob = target_dist.log_prob(x_plot.unsqueeze(dim=-1))
    dist_plot_probs = dist_plot_log_prob.exp().split(1, dim=1)
    for prob in dist_plot_probs:
        prob_np = np.asarray(prob)
        axs[1].plot(x_plot_np, prob_np, 'b')
    # Particles concentration
    # Scatter plot: scatter on the x axis
    post_ptcl_np = np.asarray(post_ptcl.squeeze())
    post_ptcl_y_val = np.zeros(num_ptcl)
    axs[1].scatter(post_ptcl_np, post_ptcl_y_val)
    # Batch-wise weight curve
    # post_weight_ls = post_msg.weight.split(1, dim=0)    # Split across batch dimension
    post_weight_ls = ld_in_post.read().weight.split(1, dim=0)
    for weight in post_weight_ls:
        post_ptcl_sorted, sort_indices = torch.sort(post_ptcl.squeeze())
        weight_sorted = weight.squeeze()[sort_indices]
        post_ptcl_sorted_np = np.asarray(post_ptcl_sorted)
        weight_sorted_np = np.asarray(weight_sorted)
        axs[1].plot(post_ptcl_sorted_np, weight_sorted_np, linestyle='--')

    # Histogram
    axs[2].title.set_text("Particles Concentration in histogram")
    axs[2].set_xlim(left=x_lim[0], right=x_lim[1])
    axs[2].hist(post_ptcl_np, range=x_lim, bins=histogram_n_bins)

    plt.show()

    # Debug
    pass
