import torch
import sys
import pdb
from sphdet.bbox.sampler.efficient_sample_from_annotation import sampleFromAnnotation_deg
from sphdet.bbox.kent_formator_torch_simple import kent_me_matrix_torch, get_me_matrix_torch

def deg2kent_single_torch(annotations, h=960, w=1920):
  pdb.set_trace()
  Xs = sampleFromAnnotation_deg(annotations, (h, w))
  S_torch, xbar_torch = get_me_matrix_torch(Xs)
  k_torch = kent_me_matrix_torch(S_torch, xbar_torch)
  
  # Check if the tensors require gradients
  assert S_torch.requires_grad, "S_torch does not support backpropagation"
  assert xbar_torch.requires_grad, "xbar_torch does not support backpropagation"
  assert k_torch.requires_grad, "k_torch does not support backpropagation"
    
  return k_torch


if __name__=='__main__':
    # Create a leaf tensor with requires_grad=True
    annotations = torch.tensor([350.0, 0.0, 23.0, 20.0], dtype=torch.float32, requires_grad=True)

    # Call the function
    kent = deg2kent_single_torch(annotations, 480, 960)
    print("Kent:", kent)
    
    # Ensure kent requires gradients
    if not kent.requires_grad:
        kent = kent.detach().requires_grad_(True)
    
    # Compute a loss (for example, sum of kent)
    loss = kent.sum()
    print("Loss:", loss)
    
    # Ensure loss requires gradients
    if loss.requires_grad:
        # Perform backward pass
        loss.retain_grad()
        loss.backward()
    else:
        print("Loss does not require gradients")
    
    # Access the gradient of the leaf tensor
    print("Loss Grad:", loss.grad)
    print("Annotations Grad:", annotations.grad)
