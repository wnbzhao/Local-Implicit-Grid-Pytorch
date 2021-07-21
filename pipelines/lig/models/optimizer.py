import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class LIGOptimizer(object):
    """Utility class for optimizing the input latent code at inference phase.

    Attributes:
        model (nn.Module): `GridInterpolationLayer` module which inputs xyz and latent code
        and outputs sdf values.
        latent_size (int): latent code length.
        grid_shape (tuple or torch.Size): grid shape of latent code grid.
        alpha_lat (float): loss weight of latent code norm loss during optimization process.
        num_optim_samples (int): number of points sampled at each step of optimization.
        init_std (float): standard deviation for initializing random latent codes.
        learning_rate (float): learning rate of the optimizer.
        optim_steps (int): total steps for optimizing the latent codes.
        print_every_n_steps (int): frequency of printing the loss information.
    """
    def __init__(self,
                 model,
                 latent_size=32,
                 alpha_lat=1e-2,
                 num_optim_samples=2048,
                 init_std=1e-2,
                 learning_rate=1e-3,
                 optim_steps=10000,
                 print_every_n_steps=1000,
                 indep_pt_loss=True,
                 device=None):
        super(LIGOptimizer, self).__init__()
        self.model = model.to(device)
        self.latent_size = latent_size
        self.alpha_lat = alpha_lat
        self.num_optim_samples = num_optim_samples
        self.init_std = init_std
        self.lr = learning_rate
        self.optim_steps = optim_steps
        self.print_every_n_steps = print_every_n_steps
        self.indep_pt_loss = indep_pt_loss
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def optimize_latent_code(self, points, point_values, occ_idxs, grid_shape):
        """Optimizes the latent code for each part of the grid.

        Args:
            points (tensor): bs*npoints*3, point samples near the mesh surface.
            point_values (tensor): bs*npoints*1, occupancy labels of the corresponding points. # (-1 / +1)
            occ_idxs (tensor): bs*noccupied*3, indices of the occupied grid,
            i.e. indices of grids to be optimized.
            grid_shape (list or tuple): [3], latent grid shape.
        Returns:
            latent_grid (tensor): bs*d*h*w*c, optimized latent grid.
        """
        device = self.device

        # Get latent code grid for optimization process
        bs, npoints, _ = points.shape
        noccupied = occ_idxs.shape[1]
        si, sj, sk = grid_shape
        occ_idxs_flatten = occ_idxs[:, :, 0] * (sj * sk) + occ_idxs[:, :, 1] * sk + occ_idxs[:, :, 2]  # bs*npoints
        random_latents = torch.randn(bs, noccupied, self.latent_size).type(torch.cuda.FloatTensor) * self.init_std
        latent_grid = torch.zeros(bs, (si * sj * sk), self.latent_size).type(torch.cuda.FloatTensor)
        occ_idxs_flatten_expanded = occ_idxs_flatten.unsqueeze(2).expand(bs, noccupied, self.latent_size).type(torch.cuda.LongTensor)
        latent_grid.scatter_(dim=1, index=occ_idxs_flatten_expanded, src=random_latents)
        latent_grid = latent_grid.reshape(bs, si, sj, sk, self.latent_size)

        latent_grid.requires_grad = True
        optimizer = optim.Adam([latent_grid], lr=self.lr)

        # Randomly shuffle the points before optimizing
        shuffled_idxs = np.random.permutation(npoints)
        points = points[:, shuffled_idxs, :]
        point_values = point_values[:, shuffled_idxs, :]

        self.model.train()  # ????
        for s in range(self.optim_steps):
            loss, acc = self.optimize_step(optimizer, latent_grid, points, point_values)
            if s % self.print_every_n_steps == 0:
                print('Step [{:6d}] Acc: {:5.4f} Loss: {:5.4f}'.format(s, acc.item(), loss.item()))

        return latent_grid

    def optimize_step(self, optimizer, latent_grid, points, point_values):
        """Performs an optimize step.

        In-place version of optimizing the input latent grid.

        Args:
            optimizer (torch.optim): py-torch optimizer
            latent_grid (tensor): 1*h*w*d*c, input random latent grid.
            points (tensor): 1*npoints*3, input query points.
            point_values (tensor): 1*npoints*1, sign of point sdf values
        Returns:
            loss (tensor): [1] loss in this optimization step.
            acc (tensor): [1] predicted sdf values' sign accuracy.
        """
        optimizer.zero_grad()
        point_samples, point_val_samples = self.random_point_sample(points, point_values)
        if self.indep_pt_loss:
            # 1*npoints*nneighbors*1 1*npoints*nneighbors
            pred, weights = self.model.decode(latent_grid, point_samples)
            pred_interp = (pred * weights.unsqueeze(3)).sum(dim=2, keepdim=True)
            pred = torch.cat([pred, pred_interp], dim=2)  # 1*npoints*9*1
            point_val_samples = point_val_samples.unsqueeze(2).expand(*pred.size())  # 1*npoints*9*1
        else:
            pred = self.model.decode(latent_grid, point_samples)

        binary_labels = (point_val_samples + 1) / 2  # 0 / 1
        pred_flatten = pred.reshape(-1, 1)
        binary_labels = binary_labels.reshape(-1, 1)
        loss = self.loss_fn(pred_flatten, binary_labels).mean()
        all_norm = torch.norm(latent_grid, dim=4).reshape(-1)
        loss_lat = all_norm[torch.abs(all_norm) > 1e-7].mean() * self.alpha_lat
        loss = loss + loss_lat

        if self.indep_pt_loss:
            both_pos = (pred[:, :, -1, :].sign() > 0) & (point_val_samples[:, :, -1, :].sign() > 0)
            both_neg = (pred[:, :, -1, :].sign() < 0) & (point_val_samples[:, :, -1, :].sign() < 0)
        else:
            both_pos = (pred.sign() > 0) & (point_val_samples.sign() > 0)
            both_neg = (pred.sign() < 0) & (point_val_samples.sign() < 0)
        correct = (both_pos | both_neg).sum().float()

        bs, nsamples = point_val_samples.shape[0], point_val_samples.shape[1]
        acc = correct / (bs * nsamples)

        loss.backward()
        optimizer.step()
        return loss, acc

    def random_point_sample(self, points, point_vals):
        """Samples point-occ pairs randomly.

        Args:
            points (tensor): bs*npoints*3
            point_vals (tensor): bs*npoints*1
        Returns:
            points_samples (tensor): bs*self.num_optim_samples*3
            point_val_samples (tensor): bs*self.num_optim_samples*1
        """
        self.num_optim_samples = min(self.num_optim_samples, points.shape[1])
        start_idx = np.random.choice(points.shape[1] - self.num_optim_samples + 1)
        end_idx = start_idx + self.num_optim_samples
        point_samples = points[:, start_idx:end_idx, :]
        point_val_samples = point_vals[:, start_idx:end_idx, :]
        return point_samples, point_val_samples
