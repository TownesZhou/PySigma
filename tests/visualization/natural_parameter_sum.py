"""
    Visualize the combined distribution obtained from summing the natural parameters of two distribution instances, and
    compare to particles scatter plot.

    Use univariate normal distribution as example.
"""
import torch
from torch import Size
from torch.distributions import Normal
import torch.distributions.constraints as C
import numpy as np
from matplotlib import pyplot as plt
from pysigma.utils import KnowledgeServer as KS
from pysigma.utils import DistributionServer as DS


if __name__ == "__main__":
    num_eval_ptcl = 20      # evaluating particles for scatter plot
    num_plot_ptcl = 200     # plotting particles for line plot of pdf curves
    dist_class = Normal

    # Separate the mean so that the two original distribution does not overlay with each other
    loc_ls = [
        torch.randn([1, 1]) + 1,
        torch.randn([1, 1]) - 1,
    ]
    scale_ls = [
        torch.rand([1, 1]) / 2 + 0.5, torch.rand([1, 1]) / 2 + 0.5
    ]
    org_reg_param_ls = [torch.stack([loc, scale], dim=-1) for loc, scale in zip(loc_ls, scale_ls)]

    org_dist_info = {'param_type': 'regular'}
    org_dist_ls = [DS.param2dist(dist_class, org_param, dist_info=org_dist_info) for org_param in org_reg_param_ls]

    # Get combined distribution
    org_nat_param_ls = [DS.dist2param(org_dist) for org_dist in org_dist_ls]
    comb_nat_param = sum(org_nat_param_ls)
    comb_dist = DS.param2dist(dist_class, comb_nat_param)

    # Visualization
    x_lim, y_lim = [-5, 5], [-5, 5]
    axis_text = ['X', 'Y']
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7, 14))
    for ax in axs:
        ax.grid(True)

    # Plotting particles and distributions' pdf's y-value
    x_plot_ptcl = torch.arange(x_lim[0], x_lim[1], step=(x_lim[1] - x_lim[0])/num_plot_ptcl)
    x_plot_ptcl_exp = x_plot_ptcl.unsqueeze(dim=0)      # expand a batch dimension
    y_org_val_exp_ls = [org_dist.log_prob(x_plot_ptcl_exp).exp() for org_dist in org_dist_ls]
    y_org_val_ls = [y_org_val_exp.squeeze(dim=0) for y_org_val_exp in y_org_val_exp_ls]
    y_comb_val_exp = comb_dist.log_prob(x_plot_ptcl_exp).exp()
    y_comb_val = y_comb_val_exp.squeeze(dim=0)

    # plot pdf curve
    x_plot_ptcl_np = np.asarray(x_plot_ptcl)
    for i, y_org_val in enumerate(y_org_val_ls):
        y_org_val_np = np.asarray(y_org_val)
        axs[0].plot(x_plot_ptcl_np, y_org_val_np, label='original distribution {}'.format(1), linestyle=':')
    y_comb_val_np = np.asarray(y_comb_val)
    axs[0].plot(x_plot_ptcl_np, y_comb_val_np, label='combined distribution')
    y_mul_val = y_org_val_ls[0] * y_org_val_ls[1]
    y_mul_val_np = np.asarray(y_mul_val)
    axs[0].plot(x_plot_ptcl_np, y_mul_val_np, label='multiplication of original pdf')
    axs[0].legend()
    axs[0].title.set_text("pdf")

    # plot log pdf curve
    for i, y_org_val in enumerate(y_org_val_ls):
        y_org_val_log_np = np.asarray(y_org_val.log())
        axs[1].plot(x_plot_ptcl_np, y_org_val_log_np, label='original_{}'.format(1), linestyle=':')
    y_comb_val_np = np.asarray(y_comb_val.log())
    axs[1].plot(x_plot_ptcl_np, y_comb_val_np, label='combined')
    y_sum_log = y_org_val_ls[0].log() + y_org_val_ls[1].log()
    y_sum_log_np = np.asarray(y_sum_log)
    axs[1].plot(x_plot_ptcl_np, y_sum_log_np, label='summation of original log pdf')
    axs[1].legend()
    axs[1].title.set_text("log pdf")

    plt.show()

