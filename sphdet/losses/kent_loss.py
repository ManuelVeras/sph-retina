import torch
import pdb
import torch.nn as nn
from mmdet.models.builder import LOSSES

def check_nan_inf(tensor: torch.Tensor, name: str):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

def radians_to_Q(psi: torch.Tensor, alpha: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    N = alpha.size(0)
    psi = psi.view(N, 1)
    alpha = alpha.view(N, 1)
    eta = eta.view(N, 1)
    
    gamma_1 = torch.cat([
        torch.cos(alpha),
        torch.sin(alpha) * torch.cos(eta),
        torch.sin(alpha) * torch.sin(eta)
    ], dim=1).unsqueeze(2)

    gamma_2 = torch.cat([
        -torch.cos(psi) * torch.sin(alpha),
        torch.cos(psi) * torch.cos(alpha) * torch.cos(eta) - torch.sin(psi) * torch.sin(eta),
        torch.cos(psi) * torch.cos(alpha) * torch.sin(eta) + torch.sin(psi) * torch.cos(eta)
    ], dim=1).unsqueeze(2)

    gamma_3 = torch.cat([
        torch.sin(psi) * torch.sin(alpha),
        -torch.sin(psi) * torch.cos(alpha) * torch.cos(eta) - torch.cos(psi) * torch.sin(eta),
        -torch.sin(psi) * torch.cos(alpha) * torch.sin(eta) + torch.cos(psi) * torch.cos(eta)
    ], dim=1).unsqueeze(2)

    gamma = torch.cat((gamma_1, gamma_2, gamma_3), dim=2)
    check_nan_inf(gamma, "gamma")
    return gamma

def c_approximation(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8  # Small value to avoid division by zero
    #print("kappa:", kappa)
    
    exp_kappa = torch.exp(kappa)
    
    term1 = kappa - 2 * beta
    term2 = kappa + 2 * beta
    
    # Debug prints to inspect intermediate values
    #print("kappa:", kappa)
    #print("beta:", beta)
    #print("term1:", term1)
    #print("term2:", term2)
    
    product = term1 * term2
    #print("product:", product)
    
    #if torch.isnan(product).any() or torch.isinf(product).any():
    #    print("NaN or Inf detected in product")
    
    #if (product < 0).any():
    #    print("Negative values detected in product")
    
    denominator = (product + epsilon)**(-0.5)  # Add epsilon to avoid division by zero
    
    # More debug prints
    #print("denominator:", denominator)
    
    #if torch.isnan(denominator).any() or torch.isinf(denominator).any():
    #    print("NaN or Inf detected in denominator")
    #print('2 * torch.pi * exp_kappa = ', 2 * torch.pi * exp_kappa)
    
    result = 2 * torch.pi * exp_kappa * denominator
    
    #print("kld:", result)
    
    check_nan_inf(result, "c_approximation")
    return result

def del_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8  # Small value to avoid division by zero

    numerator = -2 * torch.pi * (4 * beta**2 + kappa - kappa**2) * torch.exp(kappa)
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2) + epsilon  # Add epsilon to avoid division by zero
    result = numerator / denominator
    check_nan_inf(result, "del_kappa")
    return result

def del_2_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8  # Small value to avoid division by zero

    numerator = 2 * torch.pi * (kappa**4 - 2 * kappa**3 + (2 - 8 * beta**2) * kappa**2 + 8 * beta**2 * kappa + 16 * beta**4 + 4 * beta**2) * torch.exp(kappa)
    denominator = (kappa - 2 * beta)**(5/2) * (kappa + 2 * beta)**(5/2) + epsilon  # Add epsilon to avoid division by zero
    result = numerator / denominator
    check_nan_inf(result, "del_2_kappa")
    return result

def del_beta(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8  # Small value to avoid division by zero

    numerator = 8 * torch.pi * torch.exp(kappa) * beta
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2) + epsilon  # Add epsilon to avoid division by zero
    result = numerator / denominator
    check_nan_inf(result, "del_beta")
    return result

def expected_x(gamma_a1: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8  # Small value to avoid division by zero
    const = (c_k / (c+epsilon)).view(-1, 1)
    result = const * gamma_a1
    check_nan_inf(result, "expected_x")
    return result

def expected_xxT(kappa: torch.Tensor, beta: torch.Tensor, Q_matrix: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    c_kk = del_2_kappa(kappa, beta)
    c_beta = del_beta(kappa, beta)
    epsilon = 1e-8  # Small value to avoid division by zero

    lambda_1 = c_k / c
    lambda_2 = (c - c_kk + c_beta) / (2 * c + epsilon)  # Add epsilon to avoid division by zero
    lambda_3 = (c - c_kk - c_beta) / (2 * c + epsilon)  # Add epsilon to avoid division by zero

    lambdas = torch.stack([lambda_1, lambda_2, lambda_3], dim=-1)  # Shape: [N, 3]
    lambda_matrix = torch.diag_embed(lambdas)  # Shape: [N, 3, 3]

    Q_matrix_T = Q_matrix.transpose(-1, -2)  # Transpose the last two dimensions: [N, 3, 3]
    result = torch.matmul(Q_matrix, torch.matmul(lambda_matrix, Q_matrix_T))  # Shape: [N, 3, 3]
    check_nan_inf(result, "expected_xxT")
    return result

def beta_gamma_exxt_gamma(beta: torch.Tensor, gamma: torch.Tensor, ExxT: torch.Tensor) -> torch.Tensor:
    gamma_unsqueezed = gamma.unsqueeze(1)  # Shape: (N, 1, 3)
    intermediate_result = torch.bmm(gamma_unsqueezed, ExxT)  # Shape: (N, 1, 3)
    gamma_unsqueezed_2 = gamma.unsqueeze(2)  # Shape: (N, 3, 1)
    result = torch.bmm(intermediate_result, gamma_unsqueezed_2).squeeze()  # Shape: (N,)
    result = beta * result  # Shape: (N,)
    check_nan_inf(result, "beta_gamma_exxt_gamma")
    return result

def calculate_log_term(c_b, c_a):
    result = torch.log(c_b.view(-1, 1) / c_a.view(1, -1)).T
    check_nan_inf(result, "calculate_log_term")
    return result

def calculate_kappa_term(kappa_a, gamma_a1, kappa_b, gamma_b1, Ex_a):
    """
    Calculate the kappa term of the KLD matrix.
    """
    kappa_a_gamma_a1 = kappa_a.view(-1, 1) * gamma_a1
    kappa_b_gamma_b1 = kappa_b.view(-1, 1) * gamma_b1
    kappa_a_gamma_a1_expanded = kappa_a_gamma_a1.unsqueeze(1)
    kappa_b_gamma_b1_expanded = kappa_b_gamma_b1.unsqueeze(0)
    diff_kappa_term = kappa_a_gamma_a1_expanded - kappa_b_gamma_b1_expanded
    Ex_a_expanded = Ex_a.unsqueeze(1).expand(-1, diff_kappa_term.size(1), -1)
    result = torch.sum(diff_kappa_term * Ex_a_expanded, dim=-1)
    check_nan_inf(result, "calculate_kappa_term")
    return result

def calculate_beta_term(beta_a: torch.Tensor, gamma_a2: torch.Tensor, beta_b: torch.Tensor, gamma_b2: torch.Tensor, ExxT_a: torch.Tensor) -> torch.Tensor:
    """
    Calculate the beta term of the KLD matrix.
    """
    beta_a_gamma_a2 = beta_a.view(-1, 1) * gamma_a2
    beta_a_gamma_a2_expanded = beta_a_gamma_a2.unsqueeze(1)
    intermediate_result_a2 = torch.bmm(beta_a_gamma_a2_expanded, ExxT_a)
    beta_a_term_1 = torch.bmm(intermediate_result_a2, gamma_a2.unsqueeze(2)).squeeze(-1)
    beta_a_term_1_expanded = beta_a_term_1.expand(-1, beta_b.size(0))

    beta_b_gamma_b2 = beta_b.view(-1, 1) * gamma_b2
    beta_b_gamma_b2_expanded = beta_b_gamma_b2.unsqueeze(0)
    ExxT_a_expanded = ExxT_a.unsqueeze(1)
    product = beta_b_gamma_b2_expanded.unsqueeze(2) * ExxT_a_expanded
    result = product.sum(dim=-1)
    gamma_b2_expanded = gamma_b2.unsqueeze(0)
    beta_b_term_1 = torch.sum(result * gamma_b2_expanded, dim=-1)

    check_nan_inf(beta_a_term_1_expanded, "calculate_beta_term: beta_a_term_1_expanded")
    check_nan_inf(beta_b_term_1, "calculate_beta_term: beta_b_term_1")
    return beta_a_term_1_expanded, beta_b_term_1

def kld_matrix(kappa_a: torch.Tensor, beta_a: torch.Tensor, gamma_a1: torch.Tensor, gamma_a2: torch.Tensor, gamma_a3: torch.Tensor,
               kappa_b: torch.Tensor, beta_b: torch.Tensor, gamma_b1: torch.Tensor, gamma_b2: torch.Tensor, gamma_b3: torch.Tensor,
               Ex_a: torch.Tensor, ExxT_a: torch.Tensor, c_a: torch.Tensor, c_b: torch.Tensor, c_ka: torch.Tensor) -> torch.Tensor:
    
    log_term = calculate_log_term(c_b, c_a)
    ex_a_term = calculate_kappa_term(kappa_a, gamma_a1, kappa_b, gamma_b1, Ex_a)
    beta_a_term_1_expanded, beta_b_term_1 = calculate_beta_term(beta_a, gamma_a2, beta_b, gamma_b2, ExxT_a)
    beta_a_term_2_expanded, beta_b_term_2 = calculate_beta_term(beta_a, gamma_a3, beta_b, gamma_b3, ExxT_a)
    
    kld = log_term + ex_a_term + beta_a_term_1_expanded - beta_b_term_1 - beta_a_term_2_expanded + beta_b_term_2
    check_nan_inf(kld, "kld_matrix")
    return kld

def get_kld(kent_pred: torch.Tensor, kent_target: torch.Tensor) -> torch.Tensor:
    
    #print(kent_pred)
    
    psi_a, alpha_a, eta_a, kappa_a, beta_a = kent_pred[:, 0], kent_pred[:, 1], kent_pred[:, 2], kent_pred[:, 3], kent_pred[:, 4]
    Q_matrix_a = radians_to_Q(psi_a, alpha_a, eta_a)

    psi_b, alpha_b, eta_b, kappa_b, beta_b = kent_target[:, 0], kent_target[:, 1], kent_target[:, 2], kent_target[:, 3], kent_target[:, 4]
    Q_matrix_b = radians_to_Q(psi_b, alpha_b, eta_b)

    gamma_a1, gamma_a2, gamma_a3 = Q_matrix_a[:, :, 0], Q_matrix_a[:, :, 1], Q_matrix_a[:, :, 2]
    gamma_b1, gamma_b2, gamma_b3 = Q_matrix_b[:, :, 0], Q_matrix_b[:, :, 1], Q_matrix_b[:, :, 2]

    c_a = c_approximation(kappa_a, beta_a)
    c_b = c_approximation(kappa_b, beta_b)
    c_ka = del_kappa(kappa_a, beta_a)

    ExxT_a = expected_xxT(kappa_a, beta_a, Q_matrix_a, c_a, c_ka)
    Ex_a = expected_x(gamma_a1, c_a, c_ka)

    kld = kld_matrix(kappa_a, beta_a, gamma_a1, gamma_a2, gamma_a3,
                            kappa_b, beta_b, gamma_b1, gamma_b2, gamma_b3,
                            Ex_a, ExxT_a, c_a, c_b, c_ka)
    check_nan_inf(kld, "get_kld")
    return kld

def kent_loss(kent_pred: torch.Tensor, kent_target: torch.Tensor, const: float = 2.0) -> torch.Tensor:
    # Ensure the first three columns are between -pi and pi,
    # and the last two columns are greater than 0 and less than a constant
    
    invalid_a_first_three = ~(kent_pred[:, :3].ge(-torch.pi) & kent_pred[:, :3].le(torch.pi))
    invalid_a_last_two = ~(kent_pred[:, 3:].gt(0) & kent_pred[:, 3:].lt(200))
    invalid_b_first_three = ~(kent_target[:, :3].ge(-torch.pi) & kent_target[:, :3].le(torch.pi))
    invalid_b_last_two = ~(kent_target[:, 3:].gt(0) & kent_target[:, 3:].lt(200))

    invalid_a_first_three_indices = invalid_a_first_three.nonzero(as_tuple=True)
    invalid_a_last_two_indices = invalid_a_last_two.nonzero(as_tuple=True)
    invalid_b_first_three_indices = invalid_b_first_three.nonzero(as_tuple=True)
    invalid_b_last_two_indices = invalid_b_last_two.nonzero(as_tuple=True)

    #assert not invalid_a_first_three.any(), f"First three columns of kent_pred must be between -pi and pi, but found invalid indices: {invalid_a_first_three_indices}"
    #assert not invalid_a_last_two.any(), f"Last two columns of kent_pred must be > 0 and < 200, but found invalid indices: {invalid_a_last_two_indices}"
    #assert not invalid_b_first_three.any(), f"First three columns of kent_target must be between -pi and pi, but found invalid indices: {invalid_b_first_three_indices}"
    #assert not invalid_b_last_two.any(), f"Last two columns of kent_target must be > 0 and < 200, but found invalid indices: {invalid_b_last_two_indices}"
    #print(kent_pred)
    
    if torch.all(kent_target == 0):
        print("Tensor contains all zeros")
    else:
        print("Tensor does not contain all zeros")
        #GAMBIARRA DO CARALHO
        eps = 1e-7
        kent_target[..., 3].clamp_(min=10, max=50)
        kent_target[..., 4].clamp_(min=eps, max=25)
    
    #print('kent target = ', kent_target)
    #print('kent pred = ', kent_pred)
    
    kld = get_kld(kent_pred, kent_target)
    
    
    #pdb.set_trace()
    result = 1 - 1 / (const + torch.sqrt(kld))
    check_nan_inf(result, "kent_loss")
    return result

@LOSSES.register_module()
class KentLoss(nn.Module):
    def __init__(self):
        super(KentLoss, self).__init__()
    
    def forward(self, pred, target, weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        return kent_loss(pred, target)

def kent_iou_calculator(kent_pred: torch.Tensor, kent_target: torch.Tensor) -> torch.Tensor:
    kld = get_kld(kent_pred, kent_target)
    print('kent target = ', kent_target)
    print('kent pred = ', kent_pred)
    result = 1 / (1 + torch.sqrt(kld))
    check_nan_inf(result, "kent_iou_calculator")
    return result

def generate_random_kent_distributions(num_samples: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)  # For reproducibility

    # Define the ranges
    psi_min, psi_max = 0, torch.pi
    alpha_min, alpha_max = 0, torch.pi
    eta_min, eta_max = 0, 2 * torch.pi
    kappa_min, kappa_max = 3, 10 # Using 10 as a practical upper bound for kappa

    psi = psi_min + (psi_max - psi_min) * torch.rand(num_samples)
    alpha = alpha_min + (alpha_max - alpha_min) * torch.rand(num_samples)
    eta = eta_min + (eta_max - eta_min) * torch.rand(num_samples)
    kappa = kappa_min + (kappa_max - kappa_min) * torch.rand(num_samples)
    beta = torch.rand(num_samples) * (kappa / 2.2)

    kent_distributions = torch.stack([psi, alpha, eta, kappa, beta], dim=1)
    check_nan_inf(kent_distributions, "generate_random_kent_distributions")
    return kent_distributions