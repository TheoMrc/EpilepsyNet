"""EpilepsyNet implementation"""

import torch
import torch.nn.functional as F
from torch import nn


def get_device():
    """
    Get the device to use for the model.
    :return: device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")
    return device


DEVICE = get_device()

FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
DILATIONS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]


class EpilepsyNet(nn.Module):
    """Model for prediction of epileptic and moving state from time series behavioral features"""

    def __init__(
        self,
        features,
        hidden_channels,
        num_layers,
        kernel_size,
        n_output_states,
        dropout_rate,
    ):
        """Initiation of model layers"""
        super().__init__()
        layers = []

        self.first_conv = nn.Sequential(
            nn.Conv1d(
                len(features),
                hidden_channels,
                kernel_size=1,
                padding="same",
                padding_mode="replicate",
            ),
            nn.ReLU(inplace=True),
        )

        for i in range(num_layers - 1):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        dilation=i * 2 + 1,
                        padding="same",
                        padding_mode="replicate",
                    ),
                    nn.Dropout(dropout_rate),
                )
            )

        self.conv = nn.ModuleList(layers)
        self.out_conv = nn.Conv1d(hidden_channels, n_output_states, kernel_size=1)

    def forward(self, x):
        """Forward pass of the model"""

        x = self.first_conv(x)

        for n, conv in enumerate(self.conv):
            res = x
            x = F.relu(conv(x), inplace=True)
            if n % 2 == 0:
                x = x + res

        x = self.out_conv(x)
        return x
