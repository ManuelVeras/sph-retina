#!/usr/bin/env python
"""
The algorithms here are partially based on methods described in:
[The Fisher-Bingham Distribution on the Sphere, John T. Kent
Journal of the Royal Statistical Society. Series B (Methodological)
Vol. 44, No. 1 (1982), pp. 71-80 Published by: Wiley
Article Stable URL: http://www.jstor.org/stable/2984712]

The example code in kent_example.py serves not only as an example
but also as a test. It performs some higher level tests but it also 
generates example plots if called directly from the shell.
"""

import torch
from torch import tensor, sqrt, sum, cos, sin, arccos, arctan2, exp, log, pi, vstack, squeeze, clip, diag, real, transpose, reshape, cat
from torch.linalg import eig
from torch.distributions import Uniform, Normal
import sys
import warnings
import pdb
from efficient_sample_from_annotation import sampleFromAnnotation_deg

torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

#helper function
def MMul(A, B):
  return torch.inner(A, torch.transpose(B, 0, 1))

#helper function to compute the L2 norm. torch.linalg.norm is not used because this function does not allow to choose an axis
def norm(x, axis=None):
  if axis is None:
    axis = 0  # or set to a default value that makes sense for your use case
  return sqrt(sum(x*x, dim=axis))  

def __generate_arbitrary_orthogonal_unit_vector(x):
  v1 = torch.cross(x, torch.tensor([1.0, 0.0, 0.0]))
  v2 = torch.cross(x, torch.tensor([0.0, 1.0, 0.0]))
  v3 = torch.cross(x, torch.tensor([0.0, 0.0, 1.0]))
  v1n = norm(v1)
  v2n = norm(v2)
  v3n = norm(v3)
  v = [v1, v2, v3][torch.argmax(torch.tensor([v1n, v2n, v3n]))]
  return v/norm(v)

def kent(alpha, eta, psi, kappa, beta):
  """
  Generates the Kent distribution based on the spherical coordinates alpha, eta, psi
  with the concentration parameter kappa and the ovalness beta
  """
  gamma1, gamma2, gamma3 = KentDistribution.spherical_coordinates_to_gammas(alpha, eta, psi)
  k = KentDistribution(gamma1, gamma2, gamma3, kappa, beta)
  return k

def kent2(gamma1, gamma2, gamma3, kappa, beta):
  """
  Generates the Kent distribution using the orthonormal vectors gamma1, 
  gamma2 and gamma3, with the concentration parameter kappa and the ovalness beta
  """
  return KentDistribution(gamma1, gamma2, gamma3, kappa, beta)
  
def kent4(Gamma, kappa, beta):
  """
  Generates the kent distribution
  """
  gamma1 = Gamma[:,0]
  gamma2 = Gamma[:,1]
  gamma3 = Gamma[:,2]
  return kent2(gamma1, gamma2, gamma3, kappa, beta)
  
class KentDistribution(object):
  minimum_value_for_kappa = 1E-6
  @staticmethod
  def create_matrix_H(alpha, eta):
    return torch.stack([
        torch.stack([cos(alpha), -sin(alpha), torch.tensor(0.0, dtype=alpha.dtype)]),
        torch.stack([sin(alpha) * cos(eta), cos(alpha) * cos(eta), -sin(eta)]),
        torch.stack([sin(alpha) * sin(eta), cos(alpha) * sin(eta), cos(eta)])
    ])

  @staticmethod
  def create_matrix_Ht(alpha, eta):
    return torch.transpose(KentDistribution.create_matrix_H(alpha, eta), 0, 1)

  @staticmethod
  def create_matrix_K(psi):
    return torch.tensor([
      [1.0, 0.0,      0.0      ],
      [0.0, cos(psi), -sin(psi)],
      [0.0, sin(psi), cos(psi) ]
    ])  
  
  @staticmethod
  def create_matrix_Kt(psi):
    return torch.transpose(KentDistribution.create_matrix_K(psi), 0, 1) 
  
  @staticmethod
  def create_matrix_Gamma(alpha, eta, psi):
    H = KentDistribution.create_matrix_H(alpha, eta)
    K = KentDistribution.create_matrix_K(psi)
    return MMul(H, K)
  
  @staticmethod
  def create_matrix_Gammat(alpha, eta, psi):
    return torch.transpose(KentDistribution.create_matrix_Gamma(alpha, eta, psi), 0, 1)
  
  @staticmethod
  def spherical_coordinates_to_gammas(alpha, eta, psi):
    Gamma = KentDistribution.create_matrix_Gamma(alpha, eta, psi)
    gamma1 = Gamma[:,0]
    gamma2 = Gamma[:,1]
    gamma3 = Gamma[:,2]    
    return gamma1, gamma2, gamma3

  @staticmethod
  def gamma1_to_spherical_coordinates(gamma1):
    alpha = arccos(gamma1[0])
    eta = arctan2(gamma1[2], gamma1[1])
    return alpha, eta

  @staticmethod
  def gammas_to_spherical_coordinates(gamma1, gamma2):
    alpha, eta = KentDistribution.gamma1_to_spherical_coordinates(gamma1)
    Ht = KentDistribution.create_matrix_Ht(alpha, eta)
    u = MMul(Ht, reshape(gamma2, (3, 1)))
    psi = arctan2(u[2][0], u[1][0])
    return alpha, eta, psi

  
  def __init__(self, gamma1, gamma2, gamma3, kappa, beta):
    self.gamma1 = torch.tensor(gamma1, dtype=torch.float64)
    self.gamma2 = torch.tensor(gamma2, dtype=torch.float64)
    self.gamma3 = torch.tensor(gamma3, dtype=torch.float64)
    self.kappa = float(kappa)
    self.beta = float(beta)

    self.alpha, self.eta, self.psi = KentDistribution.gammas_to_spherical_coordinates(self.gamma1, self.gamma2)
    
    for gamma in gamma1, gamma2, gamma3:
      assert len(gamma) == 3

    # Replace the direct shape assignment with a new empty tensor of the correct shape
    self._cached_rvs = torch.empty((0, 3), dtype=torch.float64)
  
  @property
  def Gamma(self):
    return self.create_matrix_Gamma(self.alpha, self.eta, self.psi)
  
  def normalize(self, cache=dict(), return_num_iterations=False, approximate=True):
    """
    Returns the normalization constant of the Kent distribution.
    The proportional error may be expected not to be greater than 1E-11.
    
    >>> gamma1 = torch.tensor([1.0, 0.0, 0.0])
    >>> gamma2 = torch.tensor([0.0, 1.0, 0.0])
    >>> gamma3 = torch.tensor([0.0, 0.0, 1.0])
    >>> tiny = KentDistribution.minimum_value_for_kappa
    >>> abs(kent2(gamma1, gamma2, gamma3, tiny, 0.0).normalize() - 4*pi) < 4*pi*1E-12
    True
    >>> for kappa in [0.01, 0.1, 0.2, 0.5, 2, 4, 8, 16]:
    ...     print abs(kent2(gamma1, gamma2, gamma3, kappa, 0.0).normalize() - 4*pi*torch.sinh(kappa)/kappa) < 1E-15*4*pi*torch.sinh(kappa)/kappa,
    ... 
    True True True True True True True True
    """
    k, b = self.kappa, self.beta
    if not (k, b) in cache:
      if approximate and (2*b)/k < 1:
        print('Aproximei')
        result = exp(k) * ((k-2*b)*(k+2*b))**(-.5)
      else:
        G = torch.special.gamma
        I = torch.special.i0
        result = 0.0
        j = 0
        if torch.isclose(torch.tensor(b), torch.tensor(0.0)):
          result = ((0.5*k)**(-2*j-0.5)) * I(2*j+0.5, k)
          result /= G(j+1)
          result *= G(j+0.5)
        else:
          while True:
            a = exp(log(b)*2*j + log(0.5*k)*(-2*j-0.5)) * I(2*j+0.5, k)
            a /= G(j+1)
            a *= G(j+0.5)
            result += a
            j += 1
            if abs(a) < abs(result)*1E-12 and j > 5:
              break
      cache[k, b] = 2*pi*result
    if return_num_iterations:
      return cache[k, b], j
    else:
      return cache[k, b]

  def log_normalize(self, return_num_iterations=False):
    """
    Returns the logarithm of the normalization constant.
    """
    if return_num_iterations:
      normalize, num_iter = self.normalize(return_num_iterations=True)
      return log(normalize), num_iter
    else:
      return log(self.normalize())
      
  def pdf_max(self, normalize=True):
    return exp(self.log_pdf_max(normalize))

  def log_pdf_max(self, normalize=True):
    """
    Returns the maximum value of the log(pdf)
    """
    if self.beta == 0.0:
      x = 1
    else:
      x = self.kappa * 1.0 / (2 * self.beta)
    if x > 1.0:
      x = 1
    fmax = self.kappa * x + self.beta * (1 - x**2)
    if normalize:
      return fmax - self.log_normalize()
    else:
      return fmax
    
  def pdf(self, xs, normalize=True):
    """
    Returns the pdf of the kent distribution for 3D vectors that
    are stored in xs which must be an array of N x 3 or N x M x 3
    N x M x P x 3 etc.
    
    The code below shows how points in the pdf can be evaluated. An integral is
    calculated using random points on the sphere to determine wether the pdf is
    properly normalized.
    
    >>> from torch import seed
    >>> from torch.distributions import Normal
    >>> seed(666)
    >>> num_samples = 400000
    >>> xs = Normal(0, 1).sample((num_samples, 3))
    >>> xs = xs / reshape(norm(xs, 1), (num_samples, 1))
    >>> assert abs(4*pi*torch.mean(kent(1.0, 1.0, 1.0, 4.0,  2.0).pdf(xs)) - 1.0) < 0.01
    >>> assert abs(4*pi*torch.mean(kent(1.0, 2.0, 3.0, 4.0,  2.0).pdf(xs)) - 1.0) < 0.01
    >>> assert abs(4*pi*torch.mean(kent(1.0, 2.0, 3.0, 4.0,  8.0).pdf(xs)) - 1.0) < 0.01
    >>> assert abs(4*pi*torch.mean(kent(1.0, 2.0, 3.0, 16.0, 8.0).pdf(xs)) - 1.0) < 0.01
    """
    return exp(self.log_pdf(xs, normalize))
  
  
  def log_pdf(self, xs, normalize=True):
    """
    Returns the log(pdf) of the kent distribution.
    """
    axis = len(xs.shape)-1
    g1x = sum(self.gamma1*xs, axis)
    g2x = sum(self.gamma2*xs, axis)
    g3x = sum(self.gamma3*xs, axis)
    k, b = self.kappa, self.beta

    f = k*g1x + b*(g2x**2 - g3x**2)
    if normalize:
      return f - self.log_normalize()
    else:
      return f
      
  def pdf_prime(self, xs, normalize=True):
    """
    Returns the derivative of the pdf with respect to kappa and beta. 
    """
    return self.pdf(xs, normalize)*self.log_pdf_prime(xs, normalize)
    
  def log_pdf_prime(self, xs, normalize=True):
    """
    Returns the derivative of the log(pdf) with respect to kappa and beta.
    """
    axis = len(xs.shape)-1
    g1x = sum(self.gamma1*xs, axis)
    g2x = sum(self.gamma2*xs, axis)
    g3x = sum(self.gamma3*xs, axis)
    k, b = self.kappa, self.beta

    dfdk = g1x
    dfdb = g2x**2 - g3x**2
    df = torch.tensor([dfdk, dfdb])
    if normalize:
      return torch.transpose(torch.transpose(df, 0, 1) - self.log_normalize_prime(), 0, 1)
    else:
      return df
          
  def normalize_prime(self, cache=dict(), return_num_iterations=False):
    """
    Returns the derivative of the normalization factor with respect to kappa and beta.
    """
    k, b = self.kappa, self.beta
    if not (k, b) in cache:
      G = torch.special.gamma
      I = torch.special.i0
      dIdk = lambda v, z : torch.special.i0e(v, z)
      dcdk, dcdb = 0.0, 0.0
      j = 0
      if b == 0:
        dcdk = (
          ( G(j+0.5)/G(j+1) )*
          ( (-0.5*j-0.125)*(k)**(-2*j-1.5) )*
          ( I(2*j+0.5, k) )
        )
        dcdk += (
          ( G(j+0.5)/G(j+1) )*
          ( (0.5*k)**(-2*j-0.5) )*
          ( dIdk(2*j+0.5, k) )
        )   

        dcdb = 0.0 
      else:
        while True:
          dk =  ((-1*j-0.25)*exp(
              log(b)*2*j + 
              log(0.5*k)*(-2*j-1.5)
            )*I(2*j+0.5, k))
          
          dk += (exp(
              log(b)*2*j +
              log(0.5*k)*(-2*j-0.5)
            )*dIdk(2*j+0.5, k))
                  
          dk /= G(j+1)
          dk *= G(j+0.5)                        

          db = (2*j*exp(
              log(b)*(2*j-1) +
              log(0.5*k)*(-2*j-0.5)
            ) * I(2*j+0.5, k))
          
          db /= G(j+1)
          db *= G(j+0.5)                     
          dcdk += dk
          dcdb += db
        
          j += 1
          if abs(dk) < abs(dcdk)*1E-12 and abs(db) < abs(dcdb)*1E-12  and j > 5:
            break
      
      cache[k, b] = 2*pi*torch.tensor([dcdk, dcdb])
    if return_num_iterations:
      return cache[k, b], j
    else:
      return cache[k, b]
    
  def log_normalize_prime(self, return_num_iterations=False):
    """
    Returns the derivative of the logarithm of the normalization factor.
    """
    if return_num_iterations:
      normalize_prime, num_iter = self.normalize_prime(return_num_iterations=True)
      return normalize_prime/self.normalize(), num_iter
    else:
      return self.normalize_prime()/self.normalize()

  def log_likelihood(self, xs):
    """
    Returns the log likelihood for xs.
    """
    retval = self.log_pdf(xs)
    return sum(retval, len(retval.shape) - 1)
    
  def log_likelihood_prime(self, xs):
    """
    Returns the derivative with respect to kappa and beta of the log likelihood for xs.
    """
    retval = self.log_pdf_prime(xs)
    if len(retval.shape) == 1:
      return retval
    else:
      return sum(retval, len(retval.shape) - 1)
    
  def _rvs_helper(self):
    num_samples = 10000
    xs = Normal(0, 1).sample((num_samples, 3))
    xs = xs / reshape(norm(xs, 1), (num_samples, 1))
    pvalues = self.pdf(xs, normalize=False)
    fmax = self.pdf_max(normalize=False)
    return xs[Uniform(0, fmax).sample((num_samples,)) < pvalues]
  
  def rvs(self, n_samples=None):
    """
    Returns random samples from the Kent distribution by rejection sampling. 
    May become inefficient for large kappas.

    The returned random samples are 3D unit vectors.
    If n_samples == None then a single sample x is returned with shape (3,)
    If n_samples is an integer value N then N samples are returned in an array with shape (N, 3)
    """
    num_samples = 1 if n_samples is None else n_samples
    rvs = self._cached_rvs
    while len(rvs) < num_samples:
      new_rvs = self._rvs_helper()
      rvs = cat([rvs, new_rvs])
    if n_samples is None:
      self._cached_rvs = rvs[1:]
      return rvs[0]
    else:
      self._cached_rvs = rvs[num_samples:]
      retval = rvs[:num_samples]
      return retval
      
  def __repr__(self):
    return "kent(%s, %s, %s, %s, %s)" % (self.alpha, self.eta, self.psi, self.kappa, self.beta)

class Rotation:
    @staticmethod
    def Rx(alpha):
        return torch.tensor([[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]])

    @staticmethod
    def Ry(beta):
        return torch.tensor([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])

    @staticmethod
    def Rz(gamma):
        return torch.tensor([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])

def projectEquirectangular2Sphere(u, w, h):
   #NOTE: eta and alpha differ from usual definition
   alpha = u[:,1] * (pi/float(h))
   eta = u[:,0] * (2.*pi/float(w))
   return torch.vstack([cos(alpha), sin(alpha)*cos(eta), sin(alpha)*sin(eta)]).T

def projectSphere2Equirectangular(x, w, h):
   #NOTE: eta and alpha differ from usual definition
   alpha = squeeze(arccos(clip(x[:,0],-1,1)))
   eta = squeeze(arctan2(x[:,2],x[:,1]))
   eta[eta < 0] += 2*pi 
   return torch.vstack([eta * float(w)/(2*pi), alpha * float(h)/(pi)])

def deg2kent_approx(annotations, h=960, w=1920):
  #operacões com gamma (3 primeiras coordaneadas)
  pass
  
def get_me_matrix_torch(xs):
  lenxs = len(xs)
  xbar = torch.mean(xs, 0) # average direction of samples from origin
  S = torch.mean(xs.reshape((lenxs, 3, 1))*xs.reshape((lenxs, 1, 3)), 0) # dispersion (or covariance) matrix around origin
  return S, xbar

def kent_me_matrix_torch(S_torch, xbar_torch):
    # Ensure consistent data types
    S_torch = S_torch.double()
    xbar_torch = xbar_torch.double()

    gamma1 = xbar_torch / norm(xbar_torch, axis=0)  # Ensure axis is specified
    alpha, eta = KentDistribution.gamma1_to_spherical_coordinates(gamma1)
    
    H = KentDistribution.create_matrix_H(alpha, eta)
    Ht = KentDistribution.create_matrix_Ht(alpha, eta)
    B = MMul(Ht, MMul(S_torch, H)) 
    
    alpha_hat = 0.5 * torch.atan2(2 * B[1, 2], B[1, 1] - B[2, 2])

    # Use operations that maintain requires_grad
    K = torch.stack([
        torch.tensor([1, 0, 0], dtype=torch.float64),
        torch.stack([torch.tensor(0, dtype=torch.float64), torch.cos(alpha_hat), -torch.sin(alpha_hat)]),
        torch.stack([torch.tensor(0, dtype=torch.float64), torch.sin(alpha_hat), torch.cos(alpha_hat)])
    ])

    G = MMul(H, K)
    Gt = torch.transpose(G, 0, 1)
    T = MMul(Gt, MMul(S_torch, G))
    
    r1 = norm(xbar_torch)
    t22, t33 = T[1, 1], T[2, 2]
    r2 = t22 - t33
    
    min_kappa = 1E-6
    kappa = torch.max(torch.tensor(min_kappa), 1.0/(2.0-2.0*r1-r2) + 1.0/(2.0-2.0*r1+r2))
    beta  = 0.5*(1.0/(2.0-2.0*r1-r2) - 1.0/(2.0-2.0*r1+r2))
    
    '''print("S_torch.requires_grad:", S_torch.requires_grad)
    print("xbar_torch.requires_grad:", xbar_torch.requires_grad)
    print("gamma1.requires_grad:", gamma1.requires_grad)
    print("alpha.requires_grad:", alpha.requires_grad)
    print("eta.requires_grad:", eta.requires_grad)
    print("H.requires_grad:", H.requires_grad)
    print("Ht.requires_grad:", Ht.requires_grad)
    print("B.requires_grad:", B.requires_grad)
    print("alpha_hat.requires_grad:", alpha_hat.requires_grad)
    print("K.requires_grad:", K.requires_grad)
    print("G.requires_grad:", G.requires_grad)
    print("Gt.requires_grad:", Gt.requires_grad)
    print("T.requires_grad:", T.requires_grad)
    print("r1.requires_grad:", r1.requires_grad)
    print("t22.requires_grad:", t22.requires_grad)
    print("t33.requires_grad:", t33.requires_grad)
    print("r2.requires_grad:", r2.requires_grad)
    #print("min_kappa.requires_grad:", min_kappa.requires_grad)
    print("kappa.requires_grad:", kappa.requires_grad)
    print("beta.requires_grad:", beta.requires_grad)'''
    
    gamma1 = G[:,0]
    gamma2 = G[:,1]
    gamma3 = G[:,2]
  
    psi, alpha, eta = KentDistribution.gammas_to_spherical_coordinates(gamma1, gamma2)

    return torch.stack([psi, alpha, eta, kappa, beta])
  
  
def deg2kent_torch(annotations, h=960, w=1920):
    results = torch.empty((0, 5), device=annotations.device)  # Initialize an empty tensor
    for annotation in annotations:
        with torch.autograd.detect_anomaly():
            Xs = sampleFromAnnotation_deg(annotation, (h, w))
            Xs.backward(torch.ones_like(Xs))  # Provide gradient argument
        S, xbar = get_me_matrix_torch(Xs)
        k = kent_me_matrix_torch(S, xbar)
        results = torch.cat((results, k.unsqueeze(0)), dim=0)  # Concatenate each k tensor
    return results

def gradient_check():
    # Create mock inputs with requires_grad=True
    S_torch = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    xbar_torch = torch.randn(3, dtype=torch.float64, requires_grad=True)

    # Call the function
    result = kent_me_matrix_torch(S_torch, xbar_torch)

    # Create a dummy loss
    loss = result.sum()

    # Backpropagate
    loss.backward(retain_graph=True)

    # Check gradients
    print("Gradients for S_torch:", S_torch.grad)
    print("Gradients for xbar_torch:", xbar_torch.grad)

    # Gradient magnitude check
    assert S_torch.grad is not None, "Gradient for S_torch is None"
    assert xbar_torch.grad is not None, "Gradient for xbar_torch is None"
    assert torch.all(S_torch.grad != 0), "Gradient for S_torch is zero"
    assert torch.all(xbar_torch.grad != 0), "Gradient for xbar_torch is zero"

    # Gradient consistency check
    S_torch.grad.zero_()
    xbar_torch.grad.zero_()
    loss.backward(retain_graph=True)
    assert torch.allclose(S_torch.grad, S_torch.grad), "Inconsistent gradients for S_torch"
    assert torch.allclose(xbar_torch.grad, xbar_torch.grad), "Inconsistent gradients for xbar_torch"

    # Finite differences gradient check
    epsilon = 1e-5
    S_torch_fd = S_torch.clone().detach().requires_grad_(True)
    xbar_torch_fd = xbar_torch.clone().detach().requires_grad_(True)
    result_fd = kent_me_matrix_torch(S_torch_fd, xbar_torch_fd)
    loss_fd = result_fd.sum()
    loss_fd.backward(retain_graph=True)

    numerical_grad_S = torch.zeros_like(S_torch)
    numerical_grad_xbar = torch.zeros_like(xbar_torch)

    for i in range(S_torch.numel()):
        S_torch_flat = S_torch.view(-1).clone().detach().requires_grad_(True)
        S_torch_perturbed_pos = S_torch_flat.clone()
        S_torch_perturbed_pos[i] += epsilon
        S_torch_perturbed_pos = S_torch_perturbed_pos.view(S_torch.size())
        result_pos = kent_me_matrix_torch(S_torch_perturbed_pos, xbar_torch).sum()
        
        S_torch_perturbed_neg = S_torch_flat.clone()
        S_torch_perturbed_neg[i] -= epsilon
        S_torch_perturbed_neg = S_torch_perturbed_neg.view(S_torch.size())
        result_neg = kent_me_matrix_torch(S_torch_perturbed_neg, xbar_torch).sum()
        
        numerical_grad = (result_pos - result_neg) / (2 * epsilon)
        numerical_grad_S.view(-1)[i] = numerical_grad

    for i in range(xbar_torch.numel()):
        xbar_torch_flat = xbar_torch.view(-1).clone().detach().requires_grad_(True)
        xbar_torch_perturbed_pos = xbar_torch_flat.clone()
        xbar_torch_perturbed_pos[i] += epsilon
        xbar_torch_perturbed_pos = xbar_torch_perturbed_pos.view(xbar_torch.size())
        result_pos = kent_me_matrix_torch(S_torch, xbar_torch_perturbed_pos).sum()
        
        xbar_torch_perturbed_neg = xbar_torch_flat.clone()
        xbar_torch_perturbed_neg[i] -= epsilon
        xbar_torch_perturbed_neg = xbar_torch_perturbed_neg.view(xbar_torch.size())
        result_neg = kent_me_matrix_torch(S_torch, xbar_torch_perturbed_neg).sum()
        
        numerical_grad = (result_pos - result_neg) / (2 * epsilon)
        numerical_grad_xbar.view(-1)[i] = numerical_grad

    # Compare numerical gradients with analytical gradients
    assert torch.allclose(S_torch.grad, numerical_grad_S, atol=1e-4), "Gradient check failed for S_torch"
    assert torch.allclose(xbar_torch.grad, numerical_grad_xbar, atol=1e-4), "Gradient check failed for xbar_torch"

    print("Numerical gradients match analytical gradients.")

if __name__ == "__main__":
    gradient_check()