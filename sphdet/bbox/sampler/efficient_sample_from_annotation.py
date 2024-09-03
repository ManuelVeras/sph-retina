import torch
import numpy as np
from numpy.linalg import norm
from numpy import *

class Rotation:
    @staticmethod
    def Rx(alpha):
        return torch.stack([
            torch.tensor([1, 0, 0], device=alpha.device),
            torch.tensor([0, torch.cos(alpha), -torch.sin(alpha)], device=alpha.device),
            torch.tensor([0, torch.sin(alpha), torch.cos(alpha)], device=alpha.device)
        ])

    @staticmethod
    def Ry(beta):
        return torch.stack([
            torch.tensor([torch.cos(beta), 0, torch.sin(beta)], device=beta.device),
            torch.tensor([0, 1, 0], device=beta.device),
            torch.tensor([-torch.sin(beta), 0, torch.cos(beta)], device=beta.device)
        ])

    @staticmethod
    def Rz(gamma):
        return torch.stack([
            torch.tensor([torch.cos(gamma), -torch.sin(gamma), 0], device=gamma.device),
            torch.tensor([torch.sin(gamma), torch.cos(gamma), 0], device=gamma.device),
            torch.tensor([0, 0, 1], device=gamma.device)
        ])

def projectEquirectangular2Sphere(u, w, h):
    # Ensure the function is differentiable
    alpha = u[:, 1] * (torch.pi / float(h))
    eta = u[:, 0] * (2. * torch.pi / float(w))
    return torch.vstack([torch.cos(alpha), torch.sin(alpha) * torch.cos(eta), torch.sin(alpha) * torch.sin(eta)]).T

def sampleFromAnnotation_deg(annotation, shape):
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

    # Use torch.sin and torch.cos directly on tensors
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

    # Debugging: Check for NaNs in R
    if torch.isnan(R).any():
        raise ValueError("NaNs detected in rotation matrix R")

    p = torch.matmul(p, R.T)

    # Debugging: Check for NaNs in p after matrix multiplication
    if torch.isnan(p).any():
        raise ValueError("NaNs detected in p after matrix multiplication")

    # Add epsilon to the norm to avoid division by zero
    norms = torch.norm(p, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=epsilon)
    p = p / norms

    eta = torch.atan2(p[:, 0], p[:, 2])
    alpha = torch.clamp(p[:, 1], -1 + epsilon, 1 - epsilon)
    alpha = torch.asin(alpha)

    u = (eta / (2 * torch.pi) + 1. / 2.) * w
    v = h - (-alpha / torch.pi + 1. / 2.) * h
    #return torch.vstack((u, v)).T
    return projectEquirectangular2Sphere(torch.vstack((u, v)).T, w, h)

def norm_stale(x, axis=None):
  if isinstance(x, list) or isinstance(x, tuple):
    x = array(x)
  return sqrt(sum(x*x, axis=axis))  
