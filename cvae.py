import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from utils import one_hot_labels


class ConditionalVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 img_size: int = 224,
                 device: str = 'cpu') -> None:
        super(ConditionalVAE, self).__init__()
        self.device = device

        input_channels = in_channels

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes

        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        in_channels += 1  # To account for the extra label channel
        # build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 7 * 7, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 7 * 7, latent_dim)

        # build decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 7 * 7)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=input_channels,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode an image and compute latent distribution parameters

        :param input: Input image
        :return: Parameters of the latent distribution
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # compute parameters of the latent distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate image given a sample from the latent distribution

        :param z: Sample from the latent distribution
        :return: Generated image
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 7, 7)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Perform reparametrization trick

        :param mu:  Predicted mean vector of the latent distribution
        :param logvar: Predicted log variance vector of the latent distribution
        :return: Sample from the latent distribution
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, labels) -> List[torch.Tensor]:
        embedded_class = self.embed_class(labels)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)

        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, labels], dim=1)
        return [self.decode(z), mu, log_var]

    def loss_function(self,
                      reconstructed_image,
                      input_image,
                      mu,
                      log_var) -> torch.Tensor:
        """
        Compute loss function value

        :param reconstructed_image: Reconstructed image
        :param input_image: Original image
        :param mu: Predicted mean vector of the latent distribution
        :param log_var: Predicted log variance vector of the latent distribution
        :return: Loss function value
        """
        kld_weight = 0.005
        recons_loss = F.mse_loss(reconstructed_image, input_image)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return loss

    def generate(self, label: int, device) -> torch.Tensor:
        """
        Generate an average image given class label

        :param label: Class label
        :return: Generated image as tensor
        """
        label = one_hot_labels([label], num_classes=self.num_classes)

        z = torch.zeros(1, self.latent_dim)
        z = z.to(device)

        z = torch.cat([z, label], dim=1)
        generated_image = self.decode(z)
        return generated_image
