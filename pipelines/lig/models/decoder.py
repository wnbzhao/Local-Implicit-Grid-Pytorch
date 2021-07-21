import torch
import torch.nn as nn


class IMNet(nn.Module):
    """ImNet layer py-torch implementation."""

    def __init__(self, dim=3, in_features=128, out_features=1, num_filters=128, activation=nn.LeakyReLU(0.2)):
        """Initialization.

        Args:
          dim: int, dimension of input points.
          in_features: int, length of input features (i.e., latent code).
          out_features: number of output features.
          num_filters: int, width of the second to last layer.
          activation: activation function.
        """
        super(IMNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.num_filters = num_filters
        self.activ = activation
        self.fc0 = nn.Linear(self.dimz, num_filters * 16)
        self.fc1 = nn.Linear(self.dimz + num_filters * 16, num_filters * 8)
        self.fc2 = nn.Linear(self.dimz + num_filters * 8, num_filters * 4)
        self.fc3 = nn.Linear(self.dimz + num_filters * 4, num_filters * 2)
        self.fc4 = nn.Linear(self.dimz + num_filters * 2, num_filters * 1)
        self.fc5 = nn.Linear(num_filters * 1, out_features)
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]

    def forward(self, x):
        """Forward method.

        Args:
          x: `[batch_size, dim+in_features]` tensor, inputs to decode.
        Returns:
          x_: output through this layer.
        """
        x_ = x
        for dense in self.fc[:4]:
            x_ = self.activ(dense(x_))
            x_ = torch.cat([x_, x], dim=-1)
        x_ = self.activ(self.fc4(x_))
        x_ = self.fc5(x_)
        return x_
