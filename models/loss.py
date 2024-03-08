import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        filterx = torch.tensor([[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]])
        self.fx = filterx.expand(1, 3, 3, 3).cuda()

        filtery = torch.tensor([[-3.0, -10, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]])
        self.fy = filtery.expand(1, 3, 3, 3).cuda()

    def forward(self, x, y):
        schxx = F.conv2d(x, self.fx, stride=1, padding=1)
        schxy = F.conv2d(x, self.fy, stride=1, padding=1)
        gradx = torch.sqrt(torch.pow(schxx, 2) + torch.pow(schxy, 2) + 1e-6)

        schyx = F.conv2d(y, self.fx, stride=1, padding=1)
        schyy = F.conv2d(y, self.fy, stride=1, padding=1)
        grady = torch.sqrt(torch.pow(schyx, 2) + torch.pow(schyy, 2) + 1e-6)

        loss = F.l1_loss(gradx, grady)
        return loss


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(
            3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False
        )
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class FilterLow(nn.Module):
    def __init__(
        self,
        recursions=1,
        kernel_size=5,
        stride=1,
        padding=True,
        include_pad=True,
        gaussian=False,
    ):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(
                kernel_size=kernel_size, stride=stride, padding=pad
            )
        else:
            self.filter = nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                count_include_pad=include_pad,
            )
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img


class FilterHigh(nn.Module):
    def __init__(
        self,
        recursions=1,
        kernel_size=5,
        stride=1,
        include_pad=True,
        normalize=True,
        gaussian=False,
    ):
        super(FilterHigh, self).__init__()
        self.filter_low = FilterLow(
            recursions=1,
            kernel_size=kernel_size,
            stride=stride,
            include_pad=include_pad,
            gaussian=gaussian,
        )
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                img = self.filter_low(img)
        img = img - self.filter_low(img)
        if self.normalize:
            return 0.5 + img * 0.5
        else:
            return img


class FSLoss(nn.Module):
    def __init__(self, recursions=1, stride=1, kernel_size=5, gaussian=False):
        super(FSLoss, self).__init__()
        self.filter = FilterHigh(
            recursions=recursions,
            stride=stride,
            kernel_size=kernel_size,
            include_pad=False,
            gaussian=gaussian,
        )

    def forward(self, x, y):
        x_ = self.filter(x)
        y_ = self.filter(y)
        loss = F.l1_loss(x_, y_)
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-4):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == "gan" or self.gan_type == "ragan":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan-gp":

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError(
                "GAN type [{:s}] is not found".format(self.gan_type)
            )

    def get_target_label(self, input, target_is_real):
        if self.gan_type == "wgan-gp":
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer("grad_outputs", torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(
            outputs=interp_crit,
            inputs=interp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction="mean", toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == "mean"
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0

            pred, target = pred / 255.0, target / 255.0
            pass
        assert len(pred.size()) == 4

        return (
            self.loss_weight
            * self.scale
            * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        )


class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        # diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss


class fftLoss_old_version(nn.Module):
    def __init__(self):
        super(fftLoss_old_version, self).__init__()

    def forward(self, x, y):
        # diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        diff = torch.rfft(
            x, signal_ndim=2, normalized=False, onesided=False
        ) - torch.rfft(y, signal_ndim=2, normalized=False, onesided=False)
        # torch.fft.fft2(x) - torch.fft.fft2(y)
        loss = torch.mean(abs(diff))
        return loss
