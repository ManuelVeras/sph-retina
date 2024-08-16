import torch
import pdb
import torch.nn as nn
from mmdet.models.builder import LOSSES

def radians_to_Q(psi: torch.Tensor, alpha: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    N = alpha.size(0)
    psi = psi.view(N, 1)
    alpha = alpha.view(N, 1)
    eta = eta.view(N, 1)
    
    # Convert input angles from degrees to radians
    psi = torch.deg2rad(psi)
    alpha = torch.deg2rad(alpha)
    eta = torch.deg2rad(eta)
    
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
    return gamma

def c_approximation(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8  # Small value to avoid division by zero
    exp_kappa = torch.exp(kappa)
    
    term1 = kappa - 2 * beta
    term2 = kappa + 2 * beta
    
    denominator = (term1 * term2 + epsilon)**(-0.5)  # Add epsilon to avoid division by zero
    
    result = 2 * torch.pi * exp_kappa * denominator
    return result

def del_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8  # Small value to avoid division by zero

    numerator = -2 * torch.pi * (4 * beta**2 + kappa - kappa**2) * torch.exp(kappa)
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2) + epsilon  # Add epsilon to avoid division by zero
    return numerator / denominator

def del_2_kappa(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8  # Small value to avoid division by zero

    numerator = 2 * torch.pi * (kappa**4 - 2 * kappa**3 + (2 - 8 * beta**2) * kappa**2 + 8 * beta**2 * kappa + 16 * beta**4 + 4 * beta**2) * torch.exp(kappa)
    denominator = (kappa - 2 * beta)**(5/2) * (kappa + 2 * beta)**(5/2) + epsilon  # Add epsilon to avoid division by zero
    return numerator / denominator

def del_beta(kappa: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8  # Small value to avoid division by zero

    numerator = 8 * torch.pi * torch.exp(kappa) * beta
    denominator = (kappa - 2 * beta)**(3/2) * (kappa + 2 * beta)**(3/2) + epsilon  # Add epsilon to avoid division by zero
    return numerator / denominator

def expected_x(gamma_a1: torch.Tensor, c: torch.Tensor, c_k: torch.Tensor) -> torch.Tensor:
    epsilon = 1e-8  # Small value to avoid division by zero
    const = (c_k / (c+epsilon)).view(-1, 1)
    return const * gamma_a1

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

    return result

def beta_gamma_exxt_gamma(beta: torch.Tensor, gamma: torch.Tensor, ExxT: torch.Tensor) -> torch.Tensor:
    gamma_unsqueezed = gamma.unsqueeze(1)  # Shape: (N, 1, 3)
    intermediate_result = torch.bmm(gamma_unsqueezed, ExxT)  # Shape: (N, 1, 3)
    gamma_unsqueezed_2 = gamma.unsqueeze(2)  # Shape: (N, 3, 1)
    result = torch.bmm(intermediate_result, gamma_unsqueezed_2).squeeze()  # Shape: (N,)
    return beta * result  # Shape: (N,)

def calculate_log_term(c_b, c_a):
    """
    Calculate the log term of the KLD matrix.
    """
    return torch.log(c_b.view(-1, 1) / c_a.view(1, -1)).T

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
    return torch.sum(diff_kappa_term * Ex_a_expanded, dim=-1)

def calculate_beta_term(beta_a: torch.Tensor, gamma_a2: torch.Tensor, beta_b: torch.Tensor, gamma_b2: torch.Tensor, ExxT_a: torch.Tensor) -> torch.Tensor:
    """
    Calculate the beta term of the KLD matrix.
    """
    beta_a_gamma_a2 = beta_a.view(-1, 1) * gamma_a2
    beta_a_gamma_a2_expanded = beta_a_gamma_a2.unsqueeze(1)
    intermediate_result_a2 = torch.bmm(beta_a_gamma_a2_expanded, ExxT_a)
    beta_a_term_1 = torch.bmm(intermediate_result_a2, gamma_a2.unsqueeze(2)).squeeze(-1)
    # Ensure beta_a_term_1 is correctly expanded
    beta_a_term_1_expanded = beta_a_term_1.expand(-1, beta_b.size(0))

    beta_b_gamma_b2 = beta_b.view(-1, 1) * gamma_b2
    beta_b_gamma_b2_expanded = beta_b_gamma_b2.unsqueeze(0)
    ExxT_a_expanded = ExxT_a.unsqueeze(1)
    product = beta_b_gamma_b2_expanded.unsqueeze(2) * ExxT_a_expanded
    result = product.sum(dim=-1)
    gamma_b2_expanded = gamma_b2.unsqueeze(0)
    beta_b_term_1 = torch.sum(result * gamma_b2_expanded, dim=-1)

    return beta_a_term_1_expanded, beta_b_term_1

def calculate_kld(log_term, ex_a_term, beta_a_term_1_expanded, beta_b_term_1, beta_a_term_2_expanded, beta_b_term_2):
    """
    Calculate the final KLD matrix.
    """

    return log_term + ex_a_term + beta_a_term_1_expanded - beta_b_term_1 - beta_a_term_2_expanded + beta_b_term_2

def kld_matrix(kappa_a: torch.Tensor, beta_a: torch.Tensor, gamma_a1: torch.Tensor, gamma_a2: torch.Tensor, gamma_a3: torch.Tensor,
               kappa_b: torch.Tensor, beta_b: torch.Tensor, gamma_b1: torch.Tensor, gamma_b2: torch.Tensor, gamma_b3: torch.Tensor,
               Ex_a: torch.Tensor, ExxT_a: torch.Tensor, c_a: torch.Tensor, c_b: torch.Tensor, c_ka: torch.Tensor) -> torch.Tensor:
    

    log_term = calculate_log_term(c_b, c_a)
    ex_a_term = calculate_kappa_term(kappa_a, gamma_a1, kappa_b, gamma_b1, Ex_a)
    beta_a_term_1_expanded, beta_b_term_1 = calculate_beta_term(beta_a, gamma_a2, beta_b, gamma_b2, ExxT_a)
    beta_a_term_2_expanded, beta_b_term_2 = calculate_beta_term(beta_a, gamma_a3, beta_b, gamma_b3, ExxT_a)
    
    kld = calculate_kld(log_term, ex_a_term, beta_a_term_1_expanded, beta_b_term_1, beta_a_term_2_expanded, beta_b_term_2)
    
    if torch.isnan(kld).any():
        print("NaN detected in kld_matrix")
        print("log_term:", log_term)
        print("ex_a_term:", ex_a_term)
        print("beta_a_term_1_expanded:", beta_a_term_1_expanded)
        print("beta_b_term_1:", beta_b_term_1)
        print("beta_a_term_2_expanded:", beta_a_term_2_expanded)
        print("beta_b_term_2:", beta_b_term_2)
        #pdb.set_trace()
    
    return kld

def get_kld(kent_a: torch.Tensor, kent_b: torch.Tensor) -> torch.Tensor:
    psi_a, alpha_a, eta_a, kappa_a, beta_a = kent_a[:, 0], kent_a[:, 1], kent_a[:, 2], kent_a[:, 3], kent_a[:, 4]
    Q_matrix_a = radians_to_Q(psi_a, alpha_a, eta_a)

    psi_b, alpha_b, eta_b, kappa_b, beta_b = kent_b[:, 0], kent_b[:, 1], kent_b[:, 2], kent_b[:, 3], kent_b[:, 4]
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
    return kld

def kent_loss(kent_a: torch.Tensor, kent_b: torch.Tensor, const: float = 2.0) -> torch.Tensor:
    # Ensure the first three columns are between -pi and pi,
    # and the last two columns are greater than 0 and less than a constant
    
    invalid_a_first_three = ~(kent_a[:, :3].ge(-torch.pi) & kent_a[:, :3].le(torch.pi))
    invalid_a_last_two = ~(kent_a[:, 3:].gt(0) & kent_a[:, 3:].lt(200))
    invalid_b_first_three = ~(kent_b[:, :3].ge(-torch.pi) & kent_b[:, :3].le(torch.pi))
    invalid_b_last_two = ~(kent_b[:, 3:].gt(0) & kent_b[:, 3:].lt(200))

    invalid_a_first_three_indices = invalid_a_first_three.nonzero(as_tuple=True)
    invalid_a_last_two_indices = invalid_a_last_two.nonzero(as_tuple=True)
    invalid_b_first_three_indices = invalid_b_first_three.nonzero(as_tuple=True)
    invalid_b_last_two_indices = invalid_b_last_two.nonzero(as_tuple=True)

    #assert not invalid_a_first_three.any(), f"First three columns of kent_a must be between -pi and pi, but found invalid indices: {invalid_a_first_three_indices}"
    #assert not invalid_a_last_two.any(), f"Last two columns of kent_a must be > 0 and < 200, but found invalid indices: {invalid_a_last_two_indices}"
    #assert not invalid_b_first_three.any(), f"First three columns of kent_b must be between -pi and pi, but found invalid indices: {invalid_b_first_three_indices}"
    #assert not invalid_b_last_two.any(), f"Last two columns of kent_b must be > 0 and < 200, but found invalid indices: {invalid_b_last_two_indices}"
    kld = get_kld(kent_a, kent_b)
    return 1 - 1 / (const + torch.sqrt(kld))

@LOSSES.register_module()
class KentLoss(nn.Module):
    def __init__(self):
        super(KentLoss, self).__init__()
    
    def forward(self, pred, target, weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        #pdb.set_trace()
        return kent_loss(pred, target)

def kent_iou_calculator(kent_a: torch.Tensor, kent_b: torch.Tensor) -> torch.Tensor:
    kld = get_kld(kent_a, kent_b)
    return 1 / (1 + torch.sqrt(kld))

def generate_random_kent_distributions(num_samples: int) -> torch.Tensor:
    torch.manual_seed(42)  # For reproducibility

    # Define the ranges
    psi_min, psi_max = 0, torch.pi
    alpha_min, alpha_max = 0, torch.pi
    eta_min, eta_max = 0, 2 * torch.pi
    kappa_min, kappa_max = 0.1, 10  # Using 10 as a practical upper bound for kappa

    # Generate random values within the specified ranges
    psi = psi_min + (psi_max - psi_min) * torch.rand(num_samples)
    alpha = alpha_min + (alpha_max - alpha_min) * torch.rand(num_samples)
    eta = eta_min + (eta_max - eta_min) * torch.rand(num_samples)
    kappa = kappa_min + (kappa_max - kappa_min) * torch.rand(num_samples)
    beta = torch.rand(num_samples) * (kappa / 2)

    # Combine into a single tensor
    kent_distributions = torch.stack([psi, alpha, eta, kappa, beta], dim=1)
    return kent_distributions

def pairwise_l1_distance(kent_a: torch.Tensor, kent_b: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise L1 distance matrix between two sets of Kent distributions.
    Each Kent distribution is represented as a vector [psi, alpha, eta, kappa, beta].
    """
    # Expand dimensions to enable broadcasting
    kent_a_expanded = kent_a.unsqueeze(1)  # Shape: [num_samples_a, 1, 5]
    kent_b_expanded = kent_b.unsqueeze(0)  # Shape: [1, num_samples_b, 5]

    # Compute the pairwise L1 distance
    l1_dist_matrix = torch.sum(torch.abs(kent_a_expanded - kent_b_expanded), dim=2)
    return l1_dist_matrix

if __name__ == "__main__":
    '''
    kent_a1 = [20, 0, 0, 20.2, 4.1] 
    kent_a2 = [10, 0, 0, 9.1, 4.1]
    kent_a3 = [0, 0, 0, 10.1, 4.1]
    kent_a4 = [0, 0, 0, 10.1, 4.1]
    
    kent_a = torch.tensor([kent_a1, kent_a2, kent_a3, kent_a4], dtype=torch.float32, requires_grad=True)

    kent_b1 = [0,0,0, 30.1, 4.1] 
    kent_b2 = [20, 0, 0, 20.1, 4.1]
    kent_b3 = [0, 0, 0, 30.1, 4.1]
    kent_b4 = kent_a1
    kent_b5 = kent_a2

    kent_b = torch.tensor([kent_b1, kent_b2, kent_b3, kent_b4, kent_b5], dtype=torch.float32, requires_grad=True)

    #nkent_loss_result = get_kld(kent_a, kent_b)'''

    num_samples = 10
    kent_a = generate_random_kent_distributions(3)
    kent_b = generate_random_kent_distributions(3)
    #print(kent_distributions)


    #kent_a = torch.tensor([[7.0000, 9.0000, 7.0000, 6.3434, 3.0207]])
    #kent_b = torch.tensor([[3.0000, 3.0000, 0.0000, 2.3307, 1.1098]])
    #kent_b = torch.tensor([[6., 3., 6.0000, 9.0000, 7.0000]])
    #kent_a = torch.tensor([[8.8785, 3.0125, 5.3710, 3.5621, 0.1642], [8.8785, 3.0125, 5.3710, 3.5621, 0.1642]])
    #kent_b = torch.tensor([[7.2833e+00, 8.0882e+00, 2.2422e+00, 1.1072e-01, 5.2726e-02], [7.2833e+00, 8.0882e+00, 2.2422e+00, 1.1072e-01, 5.2726e-02]])
    
    #num cols = kent_b, num rows = kent_a
    print(get_kld(kent_a, kent_b))
    l1_dist_matrix = pairwise_l1_distance(kent_a, kent_b)
    print(l1_dist_matrix)