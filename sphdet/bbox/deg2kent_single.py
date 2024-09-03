import torch
import sys
import pdb
from sphdet.bbox.kent_formator_torch import kent_me_matrix_torch, get_me_matrix_torch

def project_equirectangular_to_sphere(u, w, h):
    """
    Projects equirectangular coordinates to spherical coordinates.

    Args:
        u (torch.Tensor): Input tensor with equirectangular coordinates.
        w (int): Width of the equirectangular image.
        h (int): Height of the equirectangular image.

    Returns:
        torch.Tensor: Tensor with spherical coordinates.
    """
    alpha = u[:, 1] * (torch.pi / float(h))
    eta = u[:, 0] * (2. * torch.pi / float(w))
    return torch.vstack([torch.cos(alpha), torch.sin(alpha) * torch.cos(eta), torch.sin(alpha) * torch.sin(eta)]).T

def sample_from_annotation_deg(annotation, shape):
    """
    Samples points from an annotation in degrees.

    Args:
        annotation (torch.Tensor): Tensor with annotation data.
        shape (tuple): Shape of the image (height, width).

    Returns:
        torch.Tensor: Sampled points in spherical coordinates.
    """
    h, w = shape
    device = annotation.device
    eta_deg, alpha_deg, fov_h, fov_v = annotation

    eta00 = torch.deg2rad(eta_deg - 180)
    alpha00 = torch.deg2rad(alpha_deg - 90)

    a_lat = torch.deg2rad(fov_v)
    a_long = torch.deg2rad(fov_h)

    r = 11
    epsilon = 1e-6  # Increased epsilon for better numerical stability
    d_lat = r / (2 * torch.tan(a_lat / 2 + epsilon))
    d_long = r / (2 * torch.tan(a_long / 2 + epsilon))

    i, j = torch.meshgrid(torch.arange(-(r - 1) // 2, (r + 1) // 2, device=device), 
                          torch.arange(-(r - 1) // 2, (r + 1) // 2, device=device), indexing='ij')
    
    p = torch.stack([i * d_lat / d_long, j, d_lat.expand_as(i)], dim=-1).reshape(-1, 3)

    sin_eta00, cos_eta00 = torch.sin(eta00), torch.cos(eta00)
    sin_alpha00, cos_alpha00 = torch.sin(alpha00), torch.cos(alpha00)

    Ry = torch.stack([
            cos_eta00, torch.zeros_like(eta00), sin_eta00,
            torch.zeros_like(eta00), torch.ones_like(eta00), torch.zeros_like(eta00),
            -sin_eta00, torch.zeros_like(eta00), cos_eta00
        ]).reshape(3, 3)
    
    Rx = torch.stack([
            torch.ones_like(alpha00), torch.zeros_like(alpha00), torch.zeros_like(alpha00),
            torch.zeros_like(alpha00), cos_alpha00, -sin_alpha00,
            torch.zeros_like(alpha00), sin_alpha00, cos_alpha00
        ]).reshape(3, 3)

    R = torch.matmul(Ry, Rx)

    if torch.isnan(R).any():
        raise ValueError("NaNs detected in rotation matrix R")

    p = torch.matmul(p, R.T)

    if torch.isnan(p).any():
        raise ValueError("NaNs detected in p after matrix multiplication")

    norms = torch.norm(p, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=epsilon)
    p = p / norms

    eta = torch.atan2(p[:, 0], p[:, 2])
    alpha = torch.clamp(p[:, 1], -1 + epsilon, 1 - epsilon)
    alpha = torch.asin(alpha)

    u = (eta / (2 * torch.pi) + 1. / 2.) * w
    v = h - (-alpha / torch.pi + 1. / 2.) * h
    return project_equirectangular_to_sphere(torch.vstack((u, v)).T, w, h)

def deg2kent_single(annotations, h=960, w=1920):
    """
    Converts annotations in degrees to Kent distribution parameters.

    Args:
        annotations (torch.Tensor): Tensor with annotation data.
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        torch.Tensor: Kent distribution parameters.
    """
    Xs = sample_from_annotation_deg(annotations, (h, w))
    S_torch, xbar_torch = get_me_matrix_torch(Xs)
    k_torch = kent_me_matrix_torch(S_torch, xbar_torch)
    
    assert S_torch.requires_grad, "S_torch does not support backpropagation"
    assert xbar_torch.requires_grad, "xbar_torch does not support backpropagation"
    assert k_torch.requires_grad, "k_torch does not support backpropagation"
    
    return k_torch

if __name__ == '__main__':
    annotations = torch.tensor([350.0, 0.0, 23.0, 20.0], dtype=torch.float32, requires_grad=True)

    kent = deg2kent_single(annotations, 480, 960)
    print("Kent:", kent)
    
    if not kent.requires_grad:
        kent = kent.detach().requires_grad_(True)
    
    loss = kent.sum()
    print("Loss:", loss)
    
    if loss.requires_grad:
        loss.retain_grad()
        loss.backward()
    else:
        print("Loss does not require gradients")
    
    print("Loss Grad:", loss.grad)
    print("Annotations Grad:", annotations.grad)