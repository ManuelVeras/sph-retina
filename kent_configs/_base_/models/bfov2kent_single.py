import torch
from skimage.io import imread
from numpy import *
from math import pi
import json
from numpy.random import seed, uniform, randint	
import pdb

class Rotation:
    @staticmethod
    def Rx(alpha):
        return torch.tensor([[1, 0, 0], 
                             [0, torch.cos(alpha), -torch.sin(alpha)], 
                             [0, torch.sin(alpha), torch.cos(alpha)]])

    @staticmethod
    def Ry(beta):
        return torch.tensor([[torch.cos(beta), 0, torch.sin(beta)], 
                             [0, 1, 0], 
                             [-torch.sin(beta), 0, torch.cos(beta)]])

    @staticmethod
    def Rz(gamma):
        return torch.tensor([[torch.cos(gamma), -torch.sin(gamma), 0], 
                             [torch.sin(gamma), torch.cos(gamma), 0], 
                             [0, 0, 1]])

def projectEquirectangular2Sphere(u, w, h):
    """Projects equirectangular coordinates to spherical coordinates."""
    # NOTE: phi and theta differ from usual definition
    theta = u[:, 1] * (pi / float(h))
    phi = u[:, 0] * (2.0 * pi / float(w))

    # Compute spherical coordinates
    x = torch.cos(theta)
    y = torch.sin(theta) * torch.cos(phi)
    z = torch.sin(theta) * torch.sin(phi)

    # Stack and transpose the results
    return torch.vstack([x, y, z]).T

def createSphere(I):
    """Converts an equirectangular image to spherical coordinates."""
    h, w = I.shape  # 960, 1920
    v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    X = projectEquirectangular2Sphere(torch.vstack((u.reshape(-1), v.reshape(-1))).T, w, h)
    return X, I.reshape(-1)

def sampleFromAnnotation(annotation, shape):
    h, w = shape
    data_x, data_y, _, _, data_fov_h, data_fov_v, label = annotation
    phi00 = torch.tensor((data_x - w / 2.0) * ((2.0 * pi) / w), dtype=torch.float32)
    theta00 = torch.tensor((data_y - h / 2.0) * (pi / h), dtype=torch.float32)
    a_lat = torch.tensor(deg2rad(data_fov_v), dtype=torch.float32)
    a_long = torch.tensor(deg2rad(data_fov_h), dtype=torch.float32)
    r = 11
    d_lat = r / (2 * tan(a_lat / 2))
    d_long = r / (2 * tan(a_long / 2))
    per = []
    for i in range(-(r - 1) // 2, (r + 1) // 2):
        for j in range(-(r - 1) // 2, (r + 1) // 2):
            per.append(torch.tensor([i * d_lat / d_long, j, d_lat], dtype=torch.float32))
    
    R = torch.matmul(Rotation.Ry(phi00), Rotation.Rx(theta00))
    
    per = torch.stack([torch.matmul(R, (per[ij] / torch.norm(per[ij]))) for ij in range(r * r)])

    phi = torch.tensor([arctan2(per[ij][0], per[ij][2]) for ij in range(r * r)], dtype=torch.float32)
    theta = torch.tensor([arcsin(per[ij][1]) for ij in range(r * r)], dtype=torch.float32)
    u = (phi / (2 * pi) + 0.5) * w
    v = h - (-theta / pi + 0.5) * h

    return u, v, label

def gamma1_to_spherical_coordinates(gamma1):
    gamma1 = torch.tensor(gamma1)  # Ensure gamma1 is a tensor
    theta = torch.acos(gamma1[0])
    phi = torch.atan2(gamma1[2], gamma1[1])
    return theta, phi

def create_matrix_H(theta, phi):
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0.0],
        [torch.sin(theta) * torch.cos(phi), torch.cos(theta) * torch.cos(phi), -torch.sin(phi)],
        [torch.sin(theta) * torch.sin(phi), torch.cos(theta) * torch.sin(phi), torch.cos(phi)]
    ])

def create_matrix_Ht(theta, phi):
    H = create_matrix_H(theta, phi)
    return H.t()

def kent_me(xs, xbar=None):
    """Generates and returns a KentDistribution based on a moment estimation."""
    pdb.set_trace()
    lenxs = len(xs[0])

    if xbar is None:
        xbar = torch.mean(xs, dim=0)  # average direction of samples from origin

    S = torch.mean(xs.unsqueeze(2) * xs.unsqueeze(1), dim=0)  # dispersion (or covariance) matrix around origin
    gamma1 = xbar / torch.norm(xbar)  # has unit length and is in the same direction and parallel to xbar
    theta, phi = gamma1_to_spherical_coordinates(gamma1)

    H = create_matrix_H(theta, phi)
    Ht = create_matrix_Ht(theta, phi)
    B = Ht @ S @ H

    eigvals, eigvects = torch.linalg.eigh(B[1:, 1:])
    eigvals = eigvals.real
    if eigvals[0] < eigvals[1]:
        eigvals = eigvals.flip(0)
        eigvects = eigvects.flip(1)
    
    K = torch.eye(3)
    K[1:, 1:] = eigvects

    G = H @ K
    Gt = G.t()
    T = Gt @ S @ G

    r1 = torch.norm(xbar)
    t22, t33 = T[1, 1], T[2, 2]
    r2 = t22 - t33

    min_kappa = 1E-6
    kappa = max(min_kappa, 1.0 / (2.0 - 2.0 * r1 - r2) + 1.0 / (2.0 - 2.0 * r1 + r2))
    beta = 0.5 * (1.0 / (2.0 - 2.0 * r1 - r2) - 1.0 / (2.0 - 2.0 * r1 + r2))
    return G, kappa, beta

def selectAnnotation(annotations, class_name=None):
	idx = 0
	rnd = randint(0,len(annotations['boxes']))
	for ann in annotations['boxes']:
		if class_name and ann[6] == class_name: return ann
		elif class_name is None:
			if idx == rnd: return ann
			else: idx+=1 

if __name__ == '__main__':
    I = imread('7fB0x.jpg', as_gray=True)
    X, C = createSphere(I)
    with open('~/masters_sph_det/360-indoor/annotations/7fB0x.json') as file: A = json.load(file)
    annotation = (selectAnnotation(A, 'chair'))
    #data_x, data_y, _, _, data_fov_h, data_fov_v, label = annotation
	
	#phi, theta = 2*pi*data_x/I.shape[1], pi*data_y/I.shape[0]
	#beta = deg2rad(data_fov_v)/deg2rad(data_fov_h)
	
	#beta = deg2rad(tan(data_fov_v / 2))/deg2rad(tan(data_fov_h / 2))
	#psi = 0
	#xbar = asarray([cos(theta), sin(theta)*cos(phi), sin(theta)*sin(phi)])
    Xs = sampleFromAnnotation(annotation, I.shape)	
    k = kent_me(Xs) #WARNING: 
	#k = tlts_kent_me(Xs, xbar)
	#k = kent_mle(Xs, warning=sys.stdout)
	#P = k.pdf(X, normalize=False)
    print(k) # theta, phi, psi, kappa, beta