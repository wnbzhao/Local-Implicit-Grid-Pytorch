import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class ResBlock3D(nn.Module):
    """3D convolutional Residual Block Layer.

    Maintains same resolution.
    """
    def __init__(self, dim, neck_channels, out_channels, final_relu=True):
        """Initialization.

        Args:
            dim (int): input feature dim
            neck_channels (int): number of channels in bottleneck layer.
            out_channels (int): number of output channels.
            final_relu (bool): whether to add relu to the last layer.
        """
        super(ResBlock3D, self).__init__()
        self.neck_channels = neck_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv3D(dim, neck_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3D(neck_channels, neck_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3D(neck_channels, out_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm3d(neck_channels)
        self.bn2 = nn.BatchNorm3d(neck_channels)
        self.bn3 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Conv3D(dim, out_channels, kernel_size=1, stride=1)
        self.final_relu = final_relu

    def forward(self, x):
        # x.shape == (N, C, D, W, H)

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x += self.shortcut(identity)
        if self.final_relu:
            x = F.relu(x)

        return x


class GridInterpolationLayer(nn.Module):
    def __init__(self, xmin=(0, 0, 0), xmax=(1, 1, 1)):
        """
        Args:
            xmin (tuple): the min vertex of bbox of the scene
            xmax (tuple): the max vertex of bbox of the scene
        """
        super(GridInterpolationLayer, self).__init__()
        self.xmin = torch.cuda.FloatTensor(xmin)
        self.xmax = torch.cuda.FloatTensor(xmax)

    def forward(self, grid, pts):
        """ Forward pass of grid interpolation layer.
            Returning trilinear interpolation neighbor latent codes, weights, and relative coordinates

        Args:
            grid (tensor): latent grid | shape==(bs, d, h, w, code_len) | `bs` is scenes batch num (not grids batch num)
            pts (tensor): query point, should be xmin<=pts<=xmax | shape==(bs, npoints, 3)
        Returns:
            lat (tensor): neighbors' latent codes | shape==(bs, npoints, 8, code_len)
            weight (tensor): trilinear interpolation weight | shape==(bs, npoints, 8)
            xloc (tensor): relative coordinate in local grid, it is normalized into (-1, 1) | shape==(bs, npoints, 8, 3)
        """
        # get dimensions
        bs, npoints, _ = pts.shape
        xmin = self.xmin.reshape([1, 1, -1])
        xmax = self.xmax.reshape([1, 1, -1])
        size = torch.cuda.FloatTensor(list(grid.shape[1:-1]))
        cube_size = 1 / (size - 1)

        # normalize coords for interpolation
        pts = (pts - xmin) / (xmax - xmin)  # normalize to 0 ~ 1
        pts = pts.clamp(min=1e-6, max=1 - 1e-6)
        ind0 = (pts / cube_size.reshape([1, 1, -1])).floor()  # grid index (bs, npoints, 3)

        # get 8 neighbors
        offset = torch.Tensor([0, 1])
        grid_x, grid_y, grid_z = torch.meshgrid(*tuple([offset] * 3))
        neighbor_offsets = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        neighbor_offsets = neighbor_offsets.reshape(-1, 3)  # 8*3
        nneighbors = neighbor_offsets.shape[0]
        neighbor_offsets = neighbor_offsets.type(torch.cuda.FloatTensor)  # shape==(8, 3)

        # get neighbor 8 latent codes
        neighbor_indices = ind0.unsqueeze(2) + neighbor_offsets[None, None, :, :]  # (bs, npoints, 8, 3)
        neighbor_indices = neighbor_indices.type(torch.cuda.LongTensor)
        neighbor_indices = neighbor_indices.reshape(bs, -1, 3)  # (bs, npoints*8, 3)
        d, h, w = neighbor_indices[:, :, 0], neighbor_indices[:, :, 1], neighbor_indices[:, :, 2]  # (bs, npoints*8)
        batch_idxs = torch.arange(bs).type(torch.cuda.LongTensor)
        batch_idxs = batch_idxs.unsqueeze(1).expand(bs, npoints * nneighbors)  # bs, 8*npoints
        lat = grid[batch_idxs, d, h, w, :]  # bs, (npoints*8), c
        lat = lat.reshape(bs, npoints, nneighbors, -1)

        # get the tri-linear interpolation weights for each point
        xyz0 = ind0 * cube_size.reshape([1, 1, -1])  # (bs, npoints, 3)
        xyz0_expand = xyz0.unsqueeze(2).expand(bs, npoints, nneighbors, 3)  # (bs, npoints, nneighbors, 3)
        xyz_neighbors = xyz0_expand + neighbor_offsets[None, None, :, :] * cube_size

        neighbor_offsets_oppo = 1 - neighbor_offsets
        xyz_neighbors_oppo = xyz0.unsqueeze(2) + neighbor_offsets_oppo[None,
                                                                       None, :, :] * cube_size  # bs, npoints, 8, 3
        dxyz = (pts.unsqueeze(2) - xyz_neighbors_oppo).abs() / cube_size
        weight = dxyz[:, :, :, 0] * dxyz[:, :, :, 1] * dxyz[:, :, :, 2]

        # relative coordinates inside the grid (-1 ~ 1, e.g. [0~1,0~1,0~1] for min vertex, [-1~0,-1~0,-1~0] for max vertex)
        xloc = (pts.unsqueeze(2) - xyz_neighbors) / cube_size[None, None, None, :]

        return lat, weight, xloc


if __name__ == '__main__':
    grid_interp_layer = GridInterpolationLayer()
    grid = torch.randn(1, 5, 5, 5, 3).type(torch.cuda.FloatTensor)
    # cube_size 0.25
    pts = torch.cuda.FloatTensor([
        [0.125, 0.125, 0.125],  # [0.5, 0.5, 0.5]
        [0.1875, 0.125, 0.125],  # [0.75, 0.5, 0.5]
        [0.05, 0.2, 0.05],  # [0.2, 0.8, 0.2]
        [0.3, 0.575, 0.925],  # [1.2, 2.3, 3.7]
        [0.2, 0.8, 0.5],  # [0.8, 3.2, 2.0]
        [0.975, 0.275, 0.65]  # [3.9, 1.1, 2.6]
    ])
    pts = pts.unsqueeze(0)
    ipdb.set_trace()
    lat, weight, loc = grid_interp_layer(grid, pts)
