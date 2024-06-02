import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=2000,
        layers=[512, 256],
        latent_dim=50,
    ):
        super().__init__()

        hidden_dim = layers[-1]

        hidden_layers = []
        for i in range(len(layers) - 1):
            hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
            hidden_layers.append(nn.Tanh())
            # hidden_layers.append(nn.Sigmoid())

        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, layers[0]),  # input_layer
            *hidden_layers  # Unpack the list of layers
        )

        self.var_enc = nn.Sequential(nn.Linear(hidden_dim, latent_dim))
        self.mu_enc = nn.Sequential(nn.Linear(hidden_dim, latent_dim))

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x):
        # Pass input through encoder layers
        x = self.encoder_layers(x)

        # Compute mean and variance
        mu = self.mu_enc(x)
        # make sure var>0
        var = torch.clamp(torch.exp(self.var_enc(x)), min=1e-20)
        z = self.reparameterize(mu, var)

        return z, mu, var


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim=2000,
        latent_dim=50,
        layers=[256, 512],
        is_norm_init=True,
    ):
        super().__init__()

        hidden_layers = []
        for i in range(len(layers) - 1):
            hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
            hidden_layers.append(nn.Tanh())
            # hidden_layers.append(nn.Sigmoid())

        self.decoder_layers = nn.Sequential(
            nn.Linear(latent_dim, layers[0]),  # input_layer
            *hidden_layers  # Unpack the list of layers
        )

        self.out_layer = nn.Sequential(nn.Linear(layers[-1], output_dim), nn.Sigmoid())

        if is_norm_init:
            self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x):
        x = self.decoder_layers(x)
        x = self.out_layer(x)

        return x


class VAE(nn.Module):
    def __init__(
        self,
        input_dim=2000,
        latent_dim=50,
        enc_layers=[512, 256],
        dec_layers=[256, 512],
        is_initialize=True,
        dec_norm_init=True,
    ):
        super().__init__()
        # for parameter record
        self.latent_dim = latent_dim

        # 输出：z, mu, var
        self.encoder = Encoder(
            input_dim=input_dim,
            layers=enc_layers,
            latent_dim=latent_dim,
        )

        self.decoder = Decoder(
            output_dim=2000,
            latent_dim=50,
            layers=[256, 512],
            is_norm_init=dec_norm_init,
        )

        if is_initialize:
            self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
