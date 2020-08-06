import json
import numpy as np
from PIL import Image
import torch


class Waifu2x:
    def __init__(self):
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            torch.nn.LeakyReLU(0.1),
            torch.nn.ConvTranspose2d(in_channels=256, out_channels=3, kernel_size=4, stride=2, padding=3),
        )
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.network.cuda()

        self.network.load_state_dict(torch.load("noise1_scale2.0x_model.pth"))

    def process(self, x):
        if isinstance(x, str):
            im = Image.open(x)
            im = np.asarray(im)
        elif isinstance(x, Image.Image):
            im = np.asarray(x)
        elif isinstance(x, np.ndarray):
            im = np.asarray(x)
        else:
            raise RuntimeError
        im = im.transpose((2, 0, 1)).astype(np.float32)
        im = im[:3]
        im = np.pad(im, ((0, 0), (7, 7), (7, 7)), mode='edge')

        with torch.no_grad():
            if self.use_cuda:
                x = torch.tensor(im[None, ...], device=next(self.network.parameters()).get_device())
            else:
                x = torch.tensor(im[None, ...])
            x = x / 255

            x = self.network(x)
            im = x.cpu()

            im = im[0].mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im = Image.fromarray(im)
        return im
