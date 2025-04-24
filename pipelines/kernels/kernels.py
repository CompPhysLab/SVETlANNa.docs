import torch
import math
import torch.nn.functional as F


# ============================
# Gaussian Kernel
# ============================
def gaussian_kernel(size=9, sigma=1.0):
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel


# ============================
# Laplacian of Gaussian
# ============================
def laplacian_of_gaussian(size=9, sigma=1.0):
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    norm = (xx**2 + yy**2 - 2 * sigma**2) / (sigma**4)
    kernel = norm * torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel - kernel.mean()
    return kernel


# ============================
# Gabor Filter
# ============================
def gabor_kernel(size=9, sigma=2.0, theta=0, Lambda=4.0, psi=0, gamma=0.5):
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    x_theta = xx * math.cos(theta) + yy * math.sin(theta)
    y_theta = -xx * math.sin(theta) + yy * math.cos(theta)
    gb = torch.exp(-0.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2) * \
         torch.cos(2 * math.pi * x_theta / Lambda + psi)
    return gb


# ============================
# Upscale 3x3 Kernel to 9x9
# ============================
def upscale_kernel(kernel_3x3, size=9):
    return F.interpolate(kernel_3x3.unsqueeze(0).unsqueeze(0), size=(size, size), mode='bilinear', align_corners=True).squeeze()


# ============================
# Predefined 3x3 Kernels
# ============================
def predefined_3x3_kernels():
    sobel_x = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ])

    sobel_y = torch.tensor([
        [-1., -2., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  2.,  1.]
    ])

    prewitt_x = torch.tensor([
        [-1., 0., 1.],
        [-1., 0., 1.],
        [-1., 0., 1.]
    ])

    prewitt_y = torch.tensor([
        [-1., -1., -1.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.]
    ])

    emboss = torch.tensor([
        [-2., -1.,  0.],
        [-1.,  1.,  1.],
        [ 0.,  1.,  2.]
    ])

    return {
        'sobel_x': sobel_x,
        'sobel_y': sobel_y,
        'prewitt_x': prewitt_x,
        'prewitt_y': prewitt_y,
        'emboss': emboss
    }


# ============================
# Identity and Edge Templates
# ============================
def identity_kernel(size=9):
    kernel = torch.zeros((size, size))
    kernel[size // 2, size // 2] = 1.0
    return kernel


def center_surround_edge_kernel(size=9):
    kernel = -1 * torch.ones((size, size))
    kernel[size // 2, size // 2] = size * size - 1
    kernel /= torch.sum(torch.abs(kernel))
    return kernel