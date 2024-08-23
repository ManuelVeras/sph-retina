import torch
import numpy as np
from numpy.linalg import norm
from numpy import *

class Rotation:
    @staticmethod
    def Rx(alpha):
        return asarray([[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]])

    @staticmethod
    def Ry(beta):
        return asarray([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])

    @staticmethod
    def Rz(gamma):
        return asarray([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])
    
def projectEquirectangular2Sphere(u, w, h):
   #NOTE: eta and alpha differ from usual definition
   alpha = u[:,1] * (pi/float(h))
   eta = u[:,0] * (2.*pi/float(w))
   return vstack([cos(alpha), sin(alpha)*cos(eta), sin(alpha)*sin(eta)]).T

def sampleFromAnnotation_deg(annotation, shape):
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

    i, j = np.meshgrid(np.arange(-(r - 1) // 2, (r + 1) // 2), np.arange(-(r - 1) // 2, (r + 1) // 2))
    p = np.stack([i * d_lat / d_long, j, np.full_like(i, d_lat)], axis=-1).reshape(-1, 3)

    R = np.dot(Rotation.Ry(eta00), Rotation.Rx(alpha00))
    p = np.dot(p, R.T)

    # Add epsilon to the norm to avoid division by zero
    norms = np.linalg.norm(p, axis=1, keepdims=True)
    norms[norms == 0] = epsilon
    p /= norms

    #print("p (vectorized):", p)

    eta = np.arctan2(p[:, 0], p[:, 2])
    alpha = np.arcsin(p[:, 1])
    u = (eta / (2 * np.pi) + 1. / 2.) * w
    v = h - (-alpha / np.pi + 1. / 2.) * h
    return projectEquirectangular2Sphere(np.vstack((u, v)).T, w, h)


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
    r = 1100

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

    R = dot(Rotation.Ry(eta00), Rotation.Rx(alpha00))
    p = asarray([dot(R, (p[ij] / norm_stale(p[ij]))) for ij in range(r * r)])

    print("p (stale):", p)

    eta = asarray([arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
    alpha = asarray([arcsin(p[ij][1]) for ij in range(r * r)])
    u = (eta / (2 * pi) + 1. / 2.) * w
    v = h - (-alpha / pi + 1. / 2.) * h
    return projectEquirectangular2Sphere(vstack((u, v)).T, w, h)

def compare_sample_functions(annotation, shape):
    result1 = sampleFromAnnotation_deg(annotation, shape)
    result2 = sampleFromAnnotation_deg_stale(annotation, shape)
    
    if np.allclose(result1, result2):
        print("The results are the same.")
    else:
        print("The results are different.")
        print("Result from sampleFromAnnotation_deg:")
        print(max(result1 - result2))
        print("Result from sampleFromAnnotation_deg_stale:")
        #print(result2)