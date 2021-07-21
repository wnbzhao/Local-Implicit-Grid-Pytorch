import math
import torch
import torch.nn as nn
from pipelines.lig.models.layers import ResBlock3D


class UNet3D(nn.Module):
    """UNet that inputs even dimension grid and outputs even dimension grid."""

    def __init__(self,
                 dim=3,
                 in_grid_res=32,
                 out_grid_res=16,
                 num_filters=16,
                 max_filters=512,
                 out_features=32):
        """Initialization.

        Args:
          in_grid_res: int, input grid resolution, must be powers of 2.
          out_grid_res: int, output grid resolution, must be powers of 2.
          num_filters: int, number of feature layers at smallest grid resolution.
          max_filters: int, max number of feature layers at any resolution.
          out_features: int, number of output feature channels.

        Raises:
          ValueError: if in_grid_res or out_grid_res is not powers of 2.
        """
        super(UNet3D, self).__init__()
        self.in_grid_res = in_grid_res
        self.out_grid_res = out_grid_res
        self.num_filters = num_filters
        self.max_filters = max_filters
        self.out_features = out_features

        # assert dimensions acceptable
        if math.log(out_grid_res, 2) % 1 != 0 or math.log(in_grid_res, 2) % 1 != 0:
            raise ValueError('in_grid_res and out_grid_res must be 2**n.')

        self.num_in_level = math.log(self.in_grid_res, 2)
        self.num_out_level = math.log(self.out_grid_res, 2)
        self.num_in_level = int(self.num_in_level)  # number of input levels
        self.num_out_level = int(self.num_out_level)  # number of output levels

        self._create_layers()

    def _create_layers(self):
        num_filter_down = [
            self.num_filters * (2 ** (i + 1)) for i in range(self.num_in_level)
        ]
        # num. features in downward path
        num_filter_down = [
            n if n <= self.max_filters else self.max_filters
            for n in num_filter_down
        ]
        num_filter_up = num_filter_down[::-1][:self.num_out_level]
        self.num_filter_down = num_filter_down
        self.num_filter_up = num_filter_up
        self.conv_in = ResBlock3D(self.num_filters, self.num_filters)
        self.conv_out = ResBlock3D(
            self.out_features, self.out_features, final_relu=False)
        self.down_modules = [ResBlock3D(int(n / 2), n) for n in num_filter_down]
        self.up_modules = [ResBlock3D(n, n) for n in num_filter_up]
        self.dnpool = nn.MaxPool3d(2, stride=2)
        self.upsamp = nn.Upsample(scale_factor=2, mode='trilinear')
        self.up_final = nn.Upsample(scale_factor=2, mode='trilinear')

    def forward(self, x):
        """Forward method.

        Args:
          x: `[batch, in_grid_res, in_grid_res, in_grid_res, in_features]` tensor,
            input voxel grid.
          training: bool, flag indicating whether model is in training mode.

        Returns:
          `[batch, out_grid_res, out_grid_res, out_grid_res, out_features]` tensor,
          output voxel grid.
        """
        x = self.conv_in(x)
        x_dns = [x]
        for mod in self.down_modules:
            x_ = self.dnpool(mod(x_dns[-1]))
            x_dns.append(x_)

        x_ups = [x_dns.pop(-1)]
        for mod in self.up_modules:
            x_ = torch.cat([self.upsamp(x_ups[-1]), x_dns.pop(-1)], dim=-1)
            x_ = mod(x_)
            x_ups.append(x_)

        x = self.conv_out(x_ups[-1])
        return x
