import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
from skimage import measure
import warnings
import time
from pipelines.utils.point_utils import sample_points_from_ray, np_get_occupied_idx, occupancy_sparse_to_dense
from pipelines.utils.postprocess_utils import remove_backface


class Generator3D(object):
    '''  Generator class for Local implicit grid Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Local implicit grid model
        optimizer (object): optimization utility class for optimizing latent grid
        part_size (float): size of a part
        num_optim_samples (int): number of points to sample at each optimization step
        res_per_part (int): how many parts we split a grid into
        overlap (bool): whether we use overlapping grids
        device (device): pytorch device
        points_batch (int): number of points we evaluate sdf values each time
        conservative (bool): whether we evaluate a grid when all of its 8 neighbors contain points
        postprocess (bool): whether to use post process to remove back faces
    '''
    def __init__(self,
                 model,
                 optimizer,
                 part_size=0.25,
                 num_optim_samples=2048,
                 res_per_part=0,
                 overlap=True,
                 device=None,
                 points_batch=20000,
                 conservative=False,
                 postprocess=True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.part_size = part_size
        self.num_optim_samples = num_optim_samples
        if res_per_part == 0:
            self.res_per_part = int(64 * self.part_size)
        else:
            self.res_per_part = res_per_part
        self.overlap = overlap
        self.device = device
        self.points_batch = points_batch
        self.conservative = conservative
        self.postprocess = postprocess

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh from inputs loaded from dataset.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        stats_dict = {}

        v = data.get('inputs', torch.empty(1, 0)).squeeze(0).cpu().numpy()
        n = data.get('inputs.normals', torch.empty(1, 0)).squeeze(0).cpu().numpy()
        mesh = self.generate_single_obj_mesh(v, n)
        return mesh

    def generate_single_obj_mesh(self, v, n):
        ''' Generates the output mesh of user specified single object.

        Args:
            v (numpy array): [#v, 3], input point cloud.
            n (numpy array): [#v, 3], normals of the input point cloud.
        Returns:
            mesh (trimesh.Trimesh obj): output mesh object.
        '''
        device = self.device

        surface_points = np.concatenate([v, n], axis=1)

        xmin = np.min(v, axis=0)
        xmax = np.max(v, axis=0)

        # check if part size is too large
        min_bb = np.min(xmax - xmin)
        if self.part_size > 0.25 * min_bb:
            warnings.warn(
                'WARNING: part_size seems too large. Recommend using a part_size < '
                '{:.2f} for this shape.'.format(0.25 * min_bb), UserWarning)

        # add some extra slack to xmin and xmax
        xmin -= self.part_size
        xmax += self.part_size

        #########################################################################
        # generate sdf samples from pc
        point_samples, sdf_values = sample_points_from_ray(v, n, sample_factor=10, std=0.01)

        # shuffle
        shuffle_index = np.random.permutation(point_samples.shape[0])
        point_samples = point_samples[shuffle_index]
        sdf_values = sdf_values[shuffle_index]

        #########################################################################
        ################### only evaluated at sparse grid location ##############
        #########################################################################
        # get valid girds (we only evaluate on sparse locations)
        # _.shape==(total_ncrops, ntarget, v.shape[1])     points within voxel
        # occ_idx.shape==(total_ncrops, 3)                 index of each voxel
        # grid_shape == (rr[0], rr[1], rr[2])
        _, occ_idx, grid_shape = np_get_occupied_idx(
            point_samples[:100000, :3],
            # point_samples[:, :3],
            xmin=xmin - 0.5 * self.part_size,
            xmax=xmax + 0.5 * self.part_size,
            crop_size=self.part_size,
            ntarget=1,  # we do not require `point_crops` (i.e. `_` in returns), so we set it to 1
            overlap=self.overlap,
            normalize_crops=False,
            return_shape=True)

        print('LIG shape: {}'.format(grid_shape))

        #########################################################################
        # treat as one batch
        point_samples = torch.from_numpy(point_samples).to(device)
        sdf_values = torch.from_numpy(sdf_values).to(device)
        occ_idx_tensor = torch.from_numpy(occ_idx).to(device)
        point_samples = point_samples.unsqueeze(0)  # shape==(1, npoints, 3)
        sdf_values = sdf_values.unsqueeze(0)  # shape==(1, npoints, 1)
        occ_idx_tensor = occ_idx_tensor.unsqueeze(0)  # shape==(1, total_ncrops, 3)

        # set range for computation
        true_shape = ((np.array(grid_shape) - 1) / (2.0 if self.overlap else 1.0)).astype(np.int32)
        self.model.set_xrange(xmin=xmin, xmax=xmin + true_shape * self.part_size)

        # Clip the point position
        xmin_ = self.model.grid_interp_layer.xmin
        xmax_ = self.model.grid_interp_layer.xmax
        x = point_samples[:, :, 0].clamp(xmin_[0], xmax_[0])
        y = point_samples[:, :, 1].clamp(xmin_[1], xmax_[1])
        z = point_samples[:, :, 2].clamp(xmin_[2], xmax_[2])
        point_samples = torch.stack([x, y, z], dim=2)

        # get label (inside==-1, outside==+1)
        point_values = torch.sign(sdf_values)

        #########################################################################
        ###################### Build/Optimize latent grid #######################
        #########################################################################
        # optimize latent grids, shape==(1, *grid_shape, code_len)
        print('Optimizing latent codes in LIG...')
        latent_grid = self.optimizer.optimize_latent_code(point_samples, point_values, occ_idx_tensor, grid_shape)

        #########################################################################
        ##################### Evaluation (Marching Cubes) #######################
        #########################################################################
        # sparse occ index to dense occ grids
        # (total_ncrops, 3) --> (*grid_shape, )  bool
        occ_mask = occupancy_sparse_to_dense(occ_idx, grid_shape)

        # points shape to be evaluated
        output_grid_shape = list(self.res_per_part * true_shape)
        # output_grid is ones, shape==(?, )
        # xyz is points to be evaluated (dense, shape==(?, 3))
        output_grid, xyz = self.get_eval_grid(xmin=xmin,
                                              xmax=xmin + true_shape * self.part_size,
                                              output_grid_shape=output_grid_shape)

        # we only evaluate eval_points
        # out_mask is for xyz, i.e. eval_points = xyz[occ_mask]
        eval_points, out_mask = self.get_eval_inputs(xyz, xmin, occ_mask)
        eval_points = torch.from_numpy(eval_points).to(device)

        # evaluate dense grid for marching cubes (on sparse grids)
        output_grid = self.generate_occ_grid(latent_grid, eval_points, output_grid, out_mask)
        output_grid = output_grid.reshape(*output_grid_shape)

        v, f, _, _ = measure.marching_cubes_lewiner(output_grid, 0)  # logits==0
        v *= (self.part_size / float(self.res_per_part) * (np.array(output_grid.shape, dtype=np.float32) /
                                                           (np.array(output_grid.shape, dtype=np.float32) - 1)))
        v += xmin

        # Create mesh
        mesh = trimesh.Trimesh(v, f)

        # Post-process the generated mesh to prevent artifacts
        if self.postprocess:
            print('Postprocessing generated mesh...')
            mesh = remove_backface(mesh, surface_points)

        return mesh

    def get_eval_grid(self, xmin, xmax, output_grid_shape):
        """Initialize the eval output grid and its corresponding grid points.

        Args:
            xmin (numpy array): [3], minimum xyz values of the entire space.
            xmax (numpy array): [3], maximum xyz values of the entire space.
            output_grid_shape (list): [3], latent grid shape.
        Returns:
             output_grid (numpy array): [d*h*w] output grid sdf values.
             xyz (numpy array): [d*h*w, 3] grid point xyz coordinates.
        """
        # setup grid
        eps = 1e-6
        l = [np.linspace(xmin[i] + eps, xmax[i] - eps, output_grid_shape[i]) for i in range(3)]
        xyz = np.stack(np.meshgrid(l[0], l[1], l[2], indexing='ij'), axis=-1).astype(np.float32)

        output_grid = np.ones(output_grid_shape, dtype=np.float32)
        xyz = xyz.reshape(-1, 3)
        output_grid = output_grid.reshape(-1)

        return output_grid, xyz

    def get_eval_inputs(self, xyz, xmin, occ_mask):
        """Gathers the points within the grids that any/all of its 8 neighbors
        contains points.

        If self.conservative is True, gathers the points within the grids that any of its 8 neighbors
        contains points.
        If self.conservative is False, gathers the points within the grids that all of its 8 neighbors
        contains points.
        Returns the points need to be evaluate and the mask of the points and the output grid.

        Args:
            xyz (numpy array): [h*w*d, 3]
            xmin (numpy array): [3] minimum value of the entire space.
            occ_mask (numpy array): latent grid occupancy mask.
        Returns:
            eval_points (numpy array): [neval, 3], points to be evaluated.
            out_mask (numpy array): [h*w*d], 0 1 value eval mask of the final sdf grid.
        """
        mask = occ_mask.astype(np.bool)
        if self.overlap:
            mask = np.stack([
                mask[:-1, :-1, :-1], mask[:-1, :-1, 1:], mask[:-1, 1:, :-1], mask[:-1, 1:, 1:], mask[1:, :-1, :-1],
                mask[1:, :-1, 1:], mask[1:, 1:, :-1], mask[1:, 1:, 1:]
            ],
                            axis=-1)
            if self.conservative:
                mask = np.any(mask, axis=-1)
            else:
                mask = np.all(mask, axis=-1)

        g = np.stack(np.meshgrid(np.arange(mask.shape[0]),
                                 np.arange(mask.shape[1]),
                                 np.arange(mask.shape[2]),
                                 indexing='ij'),
                     axis=-1).reshape(-1, 3)
        g = g[:, 0] * (mask.shape[1] * mask.shape[2]) + g[:, 1] * mask.shape[2] + g[:, 2]
        g_valid = g[mask.ravel()]  # valid grid index

        if self.overlap:
            ijk = np.floor((xyz - xmin) / self.part_size * 2).astype(np.int32)
        else:
            ijk = np.floor((xyz - xmin + 0.5 * self.part_size) / self.part_size).astype(np.int32)
        ijk_idx = (ijk[:, 0] * (mask.shape[1] * mask.shape[2]) + ijk[:, 1] * mask.shape[2] + ijk[:, 2])
        out_mask = np.isin(ijk_idx, g_valid)
        eval_points = xyz[out_mask]
        return eval_points, out_mask

    def generate_occ_grid(self, latent_grid, eval_points, output_grid, out_mask):
        """Gets the final output occ grid.

        Args:
            latent_grid (tensor): [1, *grid_shape, latent_size], optimized latent grid.
            eval_points (tensor): [neval, 3], points to be evaluated.
            output_grid (numpy array): [d*h*w], final output occ grid.
            out_mask (numpy array): [d*h*w], mask indicating the grids evaluated.
        Returns:
            output_grid (numpy array): [d*h*w], final output occ grid flattened.
        """
        interp_old = self.model.interp
        self.model.interp = True

        split = int(np.ceil(eval_points.shape[0] / self.points_batch))
        occ_val_list = []
        self.model.eval()
        with torch.no_grad():
            for s in range(split):
                sid = s * self.points_batch
                eid = min((s + 1) * self.points_batch, eval_points.shape[0])
                eval_points_slice = eval_points[sid:eid, :]
                occ_vals = self.model.decode(latent_grid, eval_points_slice.unsqueeze(0))
                occ_vals = occ_vals.squeeze(0).squeeze(1).cpu().numpy()
                occ_val_list.append(occ_vals)
        occ_vals = np.concatenate(occ_val_list, axis=0)
        output_grid[out_mask] = occ_vals

        self.model.interp = interp_old
        return output_grid
