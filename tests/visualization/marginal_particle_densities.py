"""
    Visualize the computed sampling densities of the marginal particles drawn using KnowledgeServer::draw_particles(),
    and compare to the actual marginal probability density function.

    Using MultivariateNormal distribution as example.

    The purpose of this script is to show that the computed sampling densities for the marginal particles drawn match
    the actual marginal density function, up to a constant factor. Therefore, the marginal particles drawn are valid
    particles, in that they can be used for importance sampling further down the line.
"""
from unittest.mock import patch
import torch
from torch import Size
from torch.distributions import MultivariateNormal
import torch.distributions.constraints as C
import numpy as np
from matplotlib import pyplot as plt
from pysigma.utils import KnowledgeServer as KS
from pysigma.utils import DistributionServer as DS

from tests.utils import generate_positive_definite


if __name__ == "__main__":
    # Using a 2-dimensional multivariate normal distribution, with random batched mean vector and
    #   random positive definite matrix as the covariance matrix
    # 2 random variables, each of size 1
    num_ptcl = 30
    b_shape, s_shape, e_shape = Size([1]), Size([num_ptcl, num_ptcl]), Size([1, 1])

    # Use multivariate normal distribution
    dist_class = MultivariateNormal
    dist_e_shape = Size([sum(e_shape)])
    loc = torch.randn(b_shape + dist_e_shape)

    # Type 1: random positive definite matrix
    cov = generate_positive_definite(b_shape, sum(e_shape))

    # Type 2: identity matrix
    # cov = torch.eye(dist_e_shape[0])
    # for i in range(len(b_shape)):
    #     cov = cov.unsqueeze(dim=0)
    # repeat_times = list(b_shape) + [1, 1]
    # cov = cov.repeat(repeat_times)

    dist = MultivariateNormal(loc, cov)

    # Joint particles: Note that these particles are NOT the same as the marginal particles drawn below
    joint_ptcl = DS.draw_particles(dist, num_ptcl)
    # # Using mock patch to wrap DS.draw_particles() so that we can capture method call arguments
    # with patch("pysigma.utils.DistributionServer.draw_particles", wraps=DS.draw_particles) as mock_draw_particles:
    #     # Callable that captures the return value of the actual method and store in joint_ptcl
    #     def capture_return_val(*args, **kwargs):
    #         global joint_ptcl
    #         return_val = DS.draw_particles(*args, **kwargs)
    #         joint_ptcl = return_val
    #         return return_val
    #
    #     mock_draw_particles.side_effect = capture_return_val

    # Get marginal particles and approximated densities
    rv_cstr = [C.real, ] * len(e_shape)
    ks = KS(dist_class, rv_sizes=list(e_shape), rv_constraints=rv_cstr, rv_num_particles=list(s_shape))
    return_ptcl, return_log_dens = ks._default_draw(dist)
    return_dens = [torch.exp(d) for d in return_log_dens]
    return_dens_normalized = [d / d.sum() for d in return_dens]

    # The actual marginal distribution for each RV
    marg_dist_list = [
        MultivariateNormal(loc[0, :1], cov[0, :1, :1]),
        MultivariateNormal(loc[0, 1:], cov[0, 1:, 1:]),
    ]
    actual_dens = [marg_dist.log_prob(p).exp() for marg_dist, p in zip(marg_dist_list, return_ptcl)]
    actual_dens_normalized = [d / d.sum() for d in actual_dens]

    num_plots = 4
    x_lim, y_lim = [-8, 8], [-8, 8]
    axis_text = ['X', 'Y']
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    fig, axs = plt.subplots(nrows=num_plots, ncols=2, figsize=(10, 20))
    # Add gridlines to every subplot
    for row in axs:
        for ax in row:
            ax.grid()

    # Part 1: visualize joint distribution and particles
    axs[0, 0].title.set_text("Joint distribution and particles")
    # Visualize drawn joint particles via a scatter plot
    x, y = np.array(joint_ptcl[:, 0]), np.array(joint_ptcl[:, 1])
    axs[0, 0].scatter(x, y)
    # Visualize joint density function via a counter plot
    x_points, y_points = np.linspace(-8, 8, 100), np.linspace(-8, 8, 100)
    x_grid, y_grid = np.meshgrid(x_points, y_points)
    grid = np.array((x_grid.ravel(), y_grid.ravel())).T.reshape(100, 100, 2)
    grid_tensor = torch.tensor(grid, dtype=torch.float)
    grid_tensor = grid_tensor.unsqueeze(dim=-2)     # Insert a singleton batch dimension
    log_prob = dist.log_prob(grid_tensor).squeeze(dim=-1)   # Squeeze out singleton batch dimension
    density = torch.exp(log_prob)
    z = np.array(density)
    axs[0, 0].contour(x_grid, y_grid, z, 10)

    # Part 2: Visualize marginal particle densities compared to actual density function, unscaled
    grid_log_prob_list, grid_density_list = [], []
    for i in range(2):
        axs[1, i].title.set_text("%s marginal densities (original value)" % axis_text[i])
        # Visualize drawn marginal particles via scatter plot
        x, y = np.array(return_ptcl[i][:, 0]), np.array(return_dens[i])
        axs[1, i].scatter(x, y)
        # Visualize actual density function via line plot
        x_grid = np.linspace(-8, 8, 100)
        grid_tensor = torch.tensor(x_grid, dtype=torch.float).unsqueeze(dim=-1).unsqueeze(dim=-1)
        log_prob = marg_dist_list[i].log_prob(grid_tensor).squeeze()
        density = torch.exp(log_prob)

        grid_log_prob_list.append(log_prob)
        grid_density_list.append(density)

        y_val = np.array(density)
        axs[1, i].plot(x_grid, y_val)

    # Part 3: Similar to part 2, but scaled comparison
    for i in range(2):
        # Visualize drawn marginal particles via scatter plot
        x, y = np.array(return_ptcl[i][:, 0]), np.array(return_dens[i])
        axs[2, i].scatter(x, y)

        # Pick the particle with larges computed densities as the one for scaling calibration
        ref_dens, index = return_dens[i].max(dim=0)
        ref_ptcl = return_ptcl[i][index]

        # Visualize actual density function via line plot
        density = grid_density_list[i]

        # Calculate scale
        actual_val = marg_dist_list[i].log_prob(ref_ptcl.unsqueeze(dim=-1).unsqueeze(dim=-1)).exp().squeeze()
        scale = (ref_dens / actual_val).item()
        density *= scale

        y_val = np.array(density)
        axs[2, i].plot(x_grid, y_val)

        # Calculate absolute maximum difference between computed densities and actual densities
        diff = (return_dens_normalized[i] - actual_dens_normalized[i]).abs().max().item()

        axs[2, i].title.set_text("%s marginal densities (scaled) \n "
                                     "Maximum absolute normalized error: %.6f" % (axis_text[i], diff))

    # Part 4: Similar to 2, but this time on logarithmic scale
    for i in range(2):
        axs[3, i].title.set_text("%s marginal log densities (original value)" % axis_text[i])
        # Visualize drawn marginal particles via scatter plot
        x, y = np.array(return_ptcl[i][:, 0]), np.array(return_log_dens[i])
        axs[3, i].scatter(x, y)
        # Visualize actual density function via line plot
        log_prob = grid_log_prob_list[i]

        y_val = np.array(log_prob)
        x_grid = np.linspace(-8, 8, 100)
        axs[3, i].plot(x_grid, y_val)

    plt.show()
