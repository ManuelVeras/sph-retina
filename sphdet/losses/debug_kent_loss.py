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

import torch.nn as nn
import torch.optim as optim
import torch
from torch import tensor, sqrt, sum, cos, sin, arccos, arctan2, exp, log, pi, vstack, squeeze, clip, diag, real, transpose, reshape, cat
from torch.linalg import eig
from torch.distributions import Uniform, Normal
import sys
import warnings
import pdb
#from efficient_sample_from_annotation import sampleFromAnnotation_deg

def hook_fn(grad):
    #print("Gradient in backward pass:")
    #print(grad)
    #print("Contains NaN:", torch.isnan(grad).any())
    #print("Contains Inf:", torch.isinf(grad).any())

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
    device = alpha.device
    dtype = alpha.dtype
    return torch.stack([
        torch.stack([torch.cos(alpha), -torch.sin(alpha), torch.tensor(0.0, dtype=dtype, device=device)]),
        torch.stack([torch.sin(alpha) * torch.cos(eta), torch.cos(alpha) * torch.cos(eta), -torch.sin(eta)]),
        torch.stack([torch.sin(alpha) * torch.sin(eta), torch.cos(alpha) * torch.sin(eta), torch.cos(eta)])
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
  
def get_me_matrix_torch(xs):
  lenxs = len(xs)
  xbar = torch.mean(xs, 0) # average direction of samples from origin
  S = torch.mean(xs.reshape((lenxs, 3, 1))*xs.reshape((lenxs, 1, 3)), 0) # dispersion (or covariance) matrix around origin
  return S, xbar

def MMul(a, b):
    #print("MMul input a requires_grad:", a.requires_grad)
    #print("MMul input b requires_grad:", b.requires_grad)
    result = torch.matmul(a, b)
    #print("MMul result requires_grad:", result.requires_grad)
    if result.requires_grad:
        result.register_hook(hook_fn)
    return result


import torch
import torch.nn as nn
import torch.optim as optim
from only_kent_loss import OnlyKentLoss

import torch
import torch.nn as nn

def verbose_log(tensor, name):
    print(f"{name} shape: {tensor.shape}")
    print(f"{name} min/max: {tensor.min().item():.4e}/{tensor.max().item():.4e}")
    print(f"{name} mean/std: {tensor.mean().item():.4e}/{tensor.std().item():.4e}")
    print(f"{name} has NaN: {torch.isnan(tensor).any().item()}")
    print(f"{name} has Inf: {torch.isinf(tensor).any().item()}")

def safe_div(a, b, eps=1e-8):
    return a / (b + eps)

class SafeMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = torch.matmul(grad_output, b.t())
        if ctx.needs_input_grad[1]:
            grad_b = torch.matmul(a.t(), grad_output)
        
        # Clip gradients to prevent inf values
        max_grad = 1e15
        grad_a = torch.clamp(grad_a, -max_grad, max_grad) if grad_a is not None else None
        grad_b = torch.clamp(grad_b, -max_grad, max_grad) if grad_b is not None else None
        
        return grad_a, grad_b

def safe_matmul(a, b):
    return SafeMatmul.apply(a, b)

class GradientClippingHook:
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def __call__(self, grad):
        return torch.nn.utils.clip_grad_norm_(grad, self.max_norm)

def kent_me_matrix_torch(S_torch, xbar_torch):
    # Convert to double precision
    S_torch = S_torch.double().requires_grad_(True)
    xbar_torch = xbar_torch.double().requires_grad_(True)
    
    verbose_log(S_torch, "S_torch")
    verbose_log(xbar_torch, "xbar_torch")
    
    gamma1 = safe_div(xbar_torch, torch.norm(xbar_torch, dim=-1, keepdim=True))
    verbose_log(gamma1, "gamma1")
    
    alpha, eta = KentDistribution.gamma1_to_spherical_coordinates(gamma1)
    verbose_log(alpha, "alpha")
    verbose_log(eta, "eta")
    
    H = KentDistribution.create_matrix_H(alpha, eta)
    Ht = KentDistribution.create_matrix_Ht(alpha, eta)
    verbose_log(H, "H")
    verbose_log(Ht, "Ht")
    
    B = safe_matmul(Ht, safe_matmul(S_torch, H))
    verbose_log(B, "B")
    
    alpha_hat = 0.5 * torch.atan2(2 * B[1, 2], B[1, 1] - B[2, 2])
    verbose_log(alpha_hat, "alpha_hat")

    K = torch.stack([
        torch.tensor([1, 0, 0], dtype=torch.float64, device=S_torch.device),
        torch.stack([torch.tensor(0, dtype=torch.float64, device=S_torch.device), torch.cos(alpha_hat), -torch.sin(alpha_hat)]),
        torch.stack([torch.tensor(0, dtype=torch.float64, device=S_torch.device), torch.sin(alpha_hat), torch.cos(alpha_hat)])
    ])
    verbose_log(K, "K")
    
    G = safe_matmul(H, K)
    verbose_log(G, "G")

    Gt = torch.transpose(G, 0, 1)
    T = safe_matmul(Gt, safe_matmul(S_torch, G))
    verbose_log(T, "T")
    
    r1 = torch.norm(xbar_torch, dim=-1)
    t22, t33 = T[1, 1], T[2, 2]
    r2 = t22 - t33
    
    min_kappa = torch.tensor(1e-6, dtype=torch.float64, device=S_torch.device)
    kappa = torch.max(min_kappa, safe_div(1.0, (2.0-2.0*r1-r2)) + safe_div(1.0, (2.0-2.0*r1+r2)))
    beta  = 0.5 * (safe_div(1.0, (2.0-2.0*r1-r2)) - safe_div(1.0, (2.0-2.0*r1+r2)))
    verbose_log(kappa, "kappa")
    verbose_log(beta, "beta")
    
    gamma1 = G[:,0]
    gamma2 = G[:,1]
    gamma3 = G[:,2]
  
    psi, alpha, eta = KentDistribution.gammas_to_spherical_coordinates(gamma1, gamma2)
    verbose_log(psi, "psi")
    verbose_log(alpha, "final_alpha")
    verbose_log(eta, "final_eta")

    result = torch.stack([psi, alpha, eta, kappa, beta])
    result.register_hook(GradientClippingHook(max_norm=1e15))
    return result

class LossScaler(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, *args, **kwargs):
        loss = self.loss_fn(*args, **kwargs)
        return loss * torch.exp(self.scale)

def train_with_anomaly_detection(model, criterion, optimizer, input_data, target):
    with torch.autograd.detect_anomaly():
        output = model(input_data)
        loss = criterion(output, target).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return loss

def main():
    model = SimpleModel().double()  # Convert model to double precision
    criterion = LossScaler(OnlyKentLoss())  # Wrap your custom loss with LossScaler
    optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=0.001)

    batch_size = 32
    input_data = torch.randn(batch_size, 4, dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 40.0, 40.0]] * batch_size, dtype=torch.float64)

    num_epochs = 100
    for epoch in range(num_epochs):
        try:
            loss = train_with_anomaly_detection(model, criterion, optimizer, input_data, target)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        except RuntimeError as e:
            print(f"Error occurred in epoch {epoch+1}: {str(e)}")
            break


# 2. Gradient Checking
def gradient_check(model, loss_fn, input_data, target):
    eps = 1e-6
    for name, param in model.named_parameters():
        param.requires_grad = False
        original = param.clone()
        for i in range(param.numel()):
            param.flatten()[i] += eps
            output = model(input_data)
            loss_plus = loss_fn(output, target)
            
            param.flatten()[i] -= 2 * eps
            output = model(input_data)
            loss_minus = loss_fn(output, target)
            
            param.flatten()[i] += eps
            numeric_grad = (loss_plus - loss_minus) / (2 * eps)
            
            param.requires_grad = True
            output = model(input_data)
            loss = loss_fn(output, target)
            loss.backward()
            analytic_grad = param.grad.flatten()[i].item()
            param.grad.zero_()
            
            if abs(numeric_grad - analytic_grad) > 1e-5:
                print(f"Gradient mismatch for {name}[{i}]: numeric={numeric_grad:.6f}, analytic={analytic_grad:.6f}")
        param.data = original
        param.requires_grad = True

# 3. Input Data Validation
def validate_input(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

# 4. Layer-by-Layer Analysis
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        verbose_log(x, "Model Input")
        x = self.fc1(x)
        verbose_log(x, "After FC1")
        x = torch.relu(x)
        verbose_log(x, "After ReLU")
        x = self.fc2(x)
        verbose_log(x, "Model Output")
        return x





# 1. Implement Gradient Clipping
def clip_gradient(model, clip_value):
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.clamp_(-clip_value, clip_value)



if __name__ == "__main__":
    main()