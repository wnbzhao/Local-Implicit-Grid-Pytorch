import torch
import torch.nn as nn
import torch.nn.functional as F
from pipelines.lig.models import encoder, decoder

encoder_dict = {
    'unet3d': encoder.UNet3D,
}

decoder_dict = {
    'imnet': decoder.IMNet,
}


class LocalImplicitGrid(nn.Module):
    def __init__(
        self,
        encoder,  # 
        decoder,  # e.g. imnet
        grid_interp_layer,
        method,
        x_location_max,
        interp,
        device,
    ):
        super(LocalImplicitGrid, self).__init__()
        if encoder is not None:
            self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.grid_interp_layer = grid_interp_layer
        self.method = method
        self.x_location_max = x_location_max
        self.interp = interp
        self.device = device
        # Print warning if x_location_max and method do not match
        if not ((x_location_max == 1 and method == "linear") or (x_location_max == 2 and method == "nn")):
            raise ValueError("Bad combination of x_location_max and method.")

    def forward(self, inputs, pts):
        grid = self.encoder(inputs)
        values = self.decode(grid, pts)
        return values

    def set_xrange(self, xmin, xmax):
        """Sets the xyz range during inference.

        Args:
            xmin (numpy array): minimum xyz values of input points.
            xmax (numpy array): maximum xyz values of input points.
        """
        setattr(self.grid_interp_layer, 'xmin', torch.from_numpy(xmin).type(torch.cuda.FloatTensor))
        setattr(self.grid_interp_layer, 'xmax', torch.from_numpy(xmax).type(torch.cuda.FloatTensor))

    def decode(self, grid, pts):
        lat, weights, xloc = self.grid_interp_layer(grid, pts)
        xloc = xloc * self.x_location_max
        if self.method == "linear":
            input_features = torch.cat([xloc, lat], dim=3)  # bs*npoints*nneighbors*c
            values = self.decoder(input_features)
            if self.interp:
                values = (values * weights.unsqueeze(3)).sum(dim=2)  # bs*npoints*1
            else:
                values = (values, weights)
        else:
            # nearest neighbor
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
