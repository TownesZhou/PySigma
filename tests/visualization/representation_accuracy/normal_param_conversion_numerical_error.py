"""
    Visualize the numerical error when converting Normal distribution's natural parameters (loc, scale) to regular
    parameters (loc, scale) then back to natural parameters again.
"""
import torch
from torch.distributions import Normal
import numpy as np
from matplotlib import pyplot as plt
from pysigma.utils import DistributionServer as DS


if __name__ == "__main__":
    num_plot_ptcl = 100
    dist_class = Normal

    loc_x_lim = [-1, 1]
    scale_x_lim = [0, 1]

    loc_x = torch.arange(loc_x_lim[0], loc_x_lim[1], step=(loc_x_lim[1] - loc_x_lim[0]) / num_plot_ptcl)
    scale_x = torch.arange(scale_x_lim[0], scale_x_lim[1], step=(scale_x_lim[1] - scale_x_lim[0]) / num_plot_ptcl)

    # Convert to natural param first
    p1 = loc_x / scale_x ** 2
    p2 = -1 / (2 * scale_x ** 2)
    nat_param = torch.stack([p1, p2], dim=-1)

    # Conversion
    val_1 = DS.param2dist(dist_class, nat_param)
    val_2 = DS.dist2param(val_1)

    # Numerical error
    err = torch.abs(val_2 - nat_param)
    p1_err, p2_err = err[:, 0].squeeze(dim=-1), err[:, 1].squeeze(dim=-1)

    # Plot
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(16, 16))

    p1_np, p1_err_np = np.asarray(p1), np.asarray(p1_err)
    axs[0, 0].scatter(p1_np, p1_err_np)
    axs[0, 0].title.set_text("Error of p1 w.r.t. natural parameters")

    p2_np, p2_err_np = np.asarray(p2), np.asarray(p2_err)
    axs[1, 0].scatter(p2_np, p2_err_np)
    axs[1, 0].title.set_text("Error of p2 w.r.t. natural parameters")

    loc_x_np = np.asarray(loc_x)
    axs[0, 1].scatter(loc_x_np, p1_err_np)
    axs[0, 1].title.set_text("Error of p1 w.r.t. original regular parameters loc")

    scale_x_np = np.asarray(scale_x)
    axs[1, 1].scatter(scale_x_np, p2_err_np)
    axs[1, 1].title.set_text("Error of p2 w.r.t. original regular parameters scale")

    plt.show()


