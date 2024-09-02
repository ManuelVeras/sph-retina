import torch
import numpy as np
from numpy.linalg import norm
from numpy import *

class Rotation:
    @staticmethod
    def Rx(alpha):
        return torch.tensor([[1, 0, 0], [0, torch.cos(alpha), -torch.sin(alpha)], [0, torch.sin(alpha), torch.cos(alpha)]], device=alpha.device)

    @staticmethod
    def Ry(beta):
        return torch.tensor([[torch.cos(beta), 0, torch.sin(beta)], [0, 1, 0], [-torch.sin(beta), 0, torch.cos(beta)]], device=beta.device)

    @staticmethod
    def Rz(gamma):
        return torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0], [torch.sin(gamma), torch.cos(gamma), 0], [0, 0, 1]], device=gamma.device)

def projectEquirectangular2Sphere(u, w, h):
    # Ensure the function is differentiable
    alpha = u[:, 1] * (torch.pi / float(h))
    eta = u[:, 0] * (2. * torch.pi / float(w))
    return torch.vstack([torch.cos(alpha), torch.sin(alpha) * torch.cos(eta), torch.sin(alpha) * torch.sin(eta)]).T

def sampleFromAnnotation_deg(annotation, shape):
    h, w = shape
    device = annotation.device  # Get the device of the input tensor
    eta_deg, alpha_deg, fov_h, fov_v = annotation

    eta00 = torch.deg2rad(eta_deg - 180)
    alpha00 = torch.deg2rad(alpha_deg - 90)

    a_lat = torch.deg2rad(fov_v)
    a_long = torch.deg2rad(fov_h)
    r = 11

    epsilon = 1e-10
    d_lat = r / (2 * torch.tan(a_lat / 2 + epsilon))
    d_long = r / (2 * torch.tan(a_long / 2 + epsilon))

    i, j = torch.meshgrid(torch.arange(-(r - 1) // 2, (r + 1) // 2, device=device), 
                          torch.arange(-(r - 1) // 2, (r + 1) // 2, device=device), indexing='ij')
    p = torch.stack([i * d_lat / d_long, j, d_lat.expand_as(i)], dim=-1).reshape(-1, 3)

    R = torch.matmul(Rotation.Ry(eta00), Rotation.Rx(alpha00))
    p = torch.matmul(p, R.T)

    # Add epsilon to the norm to avoid division by zero
    norms = torch.norm(p, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.tensor(epsilon, device=device), norms)
    p /= norms

    eta = torch.atan2(p[:, 0], p[:, 2])
    alpha = torch.asin(p[:, 1])
    u = (eta / (2 * torch.pi) + 1. / 2.) * w
    v = h - (-alpha / torch.pi + 1. / 2.) * h
    return projectEquirectangular2Sphere(torch.vstack((u, v)).T, w, h)

def norm_stale(x, axis=None):
  if isinstance(x, list) or isinstance(x, tuple):
    x = array(x)
  return sqrt(sum(x*x, axis=axis))  

def sampleFromAnnotation_deg_stale(annotation, shape):
    h, w = shape
    annotation = annotation.cpu().numpy()
    eta_deg, alpha_deg, fov_h, fov_v = annotation

    eta00 = deg2rad(eta_deg - 180)
    alpha00 = deg2rad(alpha_deg - 90)

    a_lat = deg2rad(fov_v)
    a_long = deg2rad(fov_h)
    r = 11

    epsilon = 1e-10
    d_lat = r / (2 * tan(a_lat / 2 + epsilon))
    d_long = r / (2 * tan(a_long / 2 + epsilon))

    p = []
    for i in range(-(r - 1) // 2, (r + 1) // 2):
        for j in range(-(r - 1) // 2, (r + 1) // 2):
            p += [asarray([i * d_lat / d_long, j, d_lat])]

    p = np.array(p)
    # Sort points to ensure consistent order
    p = p[np.lexsort((p[:, 2], p[:, 1], p[:, 0]))]

    R = dot(Rotation.Ry(torch.tensor(eta00)), Rotation.Rx(torch.tensor(alpha00) ))
    p = asarray([dot(R, (p[ij] / norm_stale(p[ij]))) for ij in range(r * r)])

    print("p (stale):", p)

    eta = asarray([arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
    alpha = asarray([arcsin(p[ij][1]) for ij in range(r * r)])
    u = torch.tensor((eta / (2 * pi) + 1. / 2.) * w)
    v = torch.tensor(h - (-alpha / pi + 1. / 2.) * h)
    return projectEquirectangular2Sphere(torch.vstack((u, v)).T, w, h)

def compare_sample_functions(annotation, shape):
    result1 = sampleFromAnnotation_deg(annotation, shape)
    result2 = sampleFromAnnotation_deg_stale(annotation, shape)
    
    if np.allclose(result1, result2):
        print("The results are the same.")
    else:
        print("The results are different.")
        print("Result from sampleFromAnnotation_deg:")
        print((result1 - result2))
        print("Result from sampleFromAnnotation_deg_stale:")
        print((result2))
        #print(result2)

#compare_sample_functions(torch.tensor([10,20, 20,20]), (480, 960))
