# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [
    "Discriminator", "Generator", "discriminator",
    "imagenet"
]

model_urls = {
    "imagenet": "https://github.com/Lornatang/ACGAN-PyTorch/releases/download/0.1.0/ACGAN_lsun-ada40795.pth"
}


class Discriminator(nn.Module):
    r""" An Discriminator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1610.09585>`_ paper.
    """

    def __init__(self, image_size: int = 64, num_classes: int = 100):
        """
        Args:
            image_size (int): The size of the image. (Default: 64).
            num_classes (int): Number of classes for dataset. (default: 10).
        """
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25),

            nn.Conv2d(256, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(0.25)
        )

        self.image_size = image_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(
            nn.Linear(64 * 512 * self.image_size * self.image_size, 1),
            nn.Sigmoid()
        )
        self.aux_layer = nn.Sequential(
            nn.Linear(64 * 512 * self.image_size * self.image_size, num_classes),
            nn.Softmax()
        )

    def forward(self, input: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        r""" Defines the computation performed at every call.

        Args:
            input (tensor): input tensor into the calculation.

        Returns:
            A four-dimensional vector (NCHW).
        """
        out = self.conv(input)
        out = torch.flatten(out)
        prediction = self.adv_layer(out)
        target = self.aux_layer(out)
        return prediction, target


class Generator(nn.Module):
    r""" An Generator model.

    `Generative Adversarial Networks model architecture from the One weird trick...
    <https://arxiv.org/abs/1610.09585>`_ paper.
    """

    def __init__(self, image_size: int = 64, num_classes: int = 100):
        """
        Args:
            image_size (int): The size of the image. (Default: 64).
            num_classes (int): Number of classes for dataset. (default: 10).
        """
        super(Generator, self).__init__()
        self.image_size = int(image_size // 2 ** 4)

        self.label_embedding = nn.Embedding(num_classes, 100)
        self.linear = nn.Linear(100, 64 * 512 * self.image_size * self.image_size)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: list = None) -> torch.Tensor:
        r"""Defines the computation performed at every call.

        Args:
            noise (tensor): input random tensor into the calculation.
            labels (list):  input tensor label.

        Returns:
            A four-dimensional vector (NCHW).
        """
        generated_images = torch.mul(self.label_embedding(labels), noise)
        out = self.linear(generated_images)
        out = out.view(out.shape[0], 512, self.image_size, self.image_size)
        out = self.conv(out)
        return out


def _gan(arch, pretrained, progress):
    model = Generator()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    return model


def discriminator() -> Discriminator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1610.09585>`_ paper.
    """
    model = Discriminator()
    return model


def imagenet(pretrained: bool = False, progress: bool = True) -> Generator:
    r"""GAN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1610.09585>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _gan("imagenet", pretrained, progress)
