import torch
import torch.nn as nn

class NearestNeighbor:
    """Nearest neighbor method to get the final sdf prediction.

    Attributes:
        decoder (nn.Module): Decoder Module which inputs xyz coordinate and latent code and outputs sdf value.
    """
    def __init__(self, decoder, interp=True):
        super(NearestNeighbor, self).__init__()
        self.decoder = decoder

    def forward(self, lat, weights, xloc):
        """Forward pass process of Nearest Neighbor Module.

        Args:
            lat (tensor): neighbors' latent codes | shape==(bs, npoints, 8, code_len)
            weights (tensor): trilinear interpolation weight | shape==(bs, npoints, 8)
            xloc (tensor): relative coordinate in local grid, it is normalized into (-1, 1) | shape==(bs, npoints, 8, 3)
        Returns:
            values (tensor): interpolated value | shape==(bs, npoints, 1)
        """
        bs, npoints, nneighbors, c = lat.size()
        nearest_neighbor_idxs = weights.max(dim=2, keepdim=True)[1]
        lat = torch.gather(lat, dim=2, index=nearest_neighbor_idxs.unsqueeze(3).expand(bs, npoints, 1,
                                                                                       c))  # bs*npoints*1*c
        lat = lat.squeeze(2)  # bs*npoints*c
        xloc = torch.gather(xloc, dim=2, index=nearest_neighbor_idxs.unsqueeze(3).expand(bs, npoints, 1, 3))
        xloc = xloc.squeeze(2)  # bs*npoints*3
        input_features = torch.cat([xloc, lat], dim=2)
        values = self.decoder(input_features)
        return values


class Linear(nn.Module):
    """Linear weighted sum method to get the final sdf prediction.

    Attributes:
        decoder (nn.Module): Decoder Module which inputs xyz coordinate and latent code and outputs sdf value.
    """
    def __init__(self, decoder, interp=True):
        super(Linear, self).__init__()
        self.decoder = decoder
        self.interp = interp

    def forward(self, lat, weights, xloc):
        """Forward pass process of Nearest Neighbor Module.

        Args:
            lat (tensor):  neighbors' latent codes | shape==(bs, npoints, 8, code_len)
            weights (tensor): trilinear interpolation weight | shape==(bs, npoints, 8)
            xloc (tensor): relative coordinate in local grid, it is normalized into (-1, 1) | shape==(bs, npoints, 8, 3)
        Returns:
            values (tensor): interpolated value | shape==(bs, npoints, 1)
        """
        input_features = torch.cat([xloc, lat], dim=3)  # shape==(bs, npoints, 8, 3+code_len)
        values = self.decoder(input_features)
        if self.interp:
            values = (values * weights.unsqueeze(3)).sum(dim=2)  # bs*npoints*1
            return values
        else:
            return (values, weights)
