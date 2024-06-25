import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------- #
#                             Sph2Pob BoxTransfrom                             #
# ---------------------------------------------------------------------------- #
def kent_standard(sph_gt, sph_pred, rbb_angle_version='deg', rbb_edge='arc', rbb_angle='equator'):
    """Transform spherical boxes to planar oriented boxes.
    NOTE: It's a standard implement of Sph2Pob.

    Args:
        sph_gt (torch.Tensor): N x 4(5), deg
        sph_pred (torch.Tensor): N x 4(5), deg
        rbb_angle_version (str, optional): The angle version of output boxes. Defaults to 'deg'.
        rbb_edge (str, optional): Algorithm option. Defaults to 'arc'.
        rbb_angle (str, optional): Algorithm option. Defaults to 'equator'.

    Returns:
        plannar_gt (torch.tensor): N x 5
        plannar_pred (torch.tensor): N x 5
    """
    sph_gt   = torch.deg2rad(sph_gt)
    sph_pred = torch.deg2rad(sph_pred)

    theta_g, phi_g, alpha_g, beta_g = torch.chunk(sph_gt[:, :4], chunks=4, dim=1)   # Nx1
    theta_p, phi_p, alpha_p, beta_p = torch.chunk(sph_pred[:, :4], chunks=4, dim=1) # Nx1
    theta_r, phi_r = (theta_g+theta_p) / 2, (phi_g + phi_p) / 2 

    from torch import cos, sin
    sin_theta, cos_theta = sin(theta_g), cos(theta_g)
    sin_phi, cos_phi = sin(phi_g), cos(phi_g)
    sin_cos_cache = sin_theta, cos_theta, sin_phi, cos_phi
    coor_g = compute_3d_coordinate(theta_g, phi_g, sin_cos_cache) # Nx3x1
    dir_g = compute_tangential_direction_along_longitude(theta_g, phi_g, sin_cos_cache) # Nx3x1

    sin_theta, cos_theta = sin(theta_p), cos(theta_p)
    sin_phi, cos_phi = sin(phi_p), cos(phi_p)
    sin_cos_cache = sin_theta, cos_theta, sin_phi, cos_phi
    coor_p = compute_3d_coordinate(theta_p, phi_p, sin_cos_cache) # Nx3x1
    dir_p = compute_tangential_direction_along_longitude(theta_p, phi_p, sin_cos_cache) # Nx3x1

    sin_theta = cos_theta = sin_phi = cos_phi = sin_cos_cache = None

    R = compute_rotate_matrix_auto(coor_g, coor_p, theta_r, phi_r)

    if sph_gt.size(1) == 5:
        gamma = sph_gt[:, -1].view((-1, 1))
        R_gamma = compute_gamma_matrix(theta_g, phi_g, -gamma)
        dir_g = torch.bmm(R_gamma, dir_g)

        gamma = sph_pred[:, -1].view((-1, 1))
        R_gamma = compute_gamma_matrix(theta_p, phi_p, -gamma)
        dir_p = torch.bmm(R_gamma, dir_p)

        del R_gamma
        #torch.cuda.empty_cache()
    
    coor_g = torch.bmm(R, coor_g) # Nx3x1
    coor_p = torch.bmm(R, coor_p) # Nx3x1
    dir_g  = torch.bmm(R, dir_g)  # Nx3x1
    dir_p  = torch.bmm(R, dir_p)  # Nx3x1

    angle_g_ = compute_internal_angle(dir_g, rbb_angle) # Nx1
    angle_p_ = compute_internal_angle(dir_p, rbb_angle) # Nx1

    theta_g_, phi_g_ = compute_spherical_coordinate(coor_g) # Nx1
    theta_p_, phi_p_ = compute_spherical_coordinate(coor_p) # Nx1

    alpha_g_ = compute_edge_length(alpha_g, rbb_edge)
    beta_g_  = compute_edge_length(beta_g, rbb_edge)
    alpha_p_ = compute_edge_length(alpha_p, rbb_edge)
    beta_p_  = compute_edge_length(beta_p, rbb_edge)

    plannar_gt = torch.concat([theta_g_, phi_g_, alpha_g_, beta_g_, angle_g_], dim=1)
    plannar_pred = torch.concat([theta_p_, phi_p_, alpha_p_, beta_p_, angle_p_], dim=1)

    plannar_gt, plannar_pred = standardize_rotated_box(plannar_gt, plannar_pred, rbb_angle_version)

    return plannar_gt, plannar_pred
#iou: A tensor of shape (N,), representing the IoU for each pair of ground truth and predicted spherical coordinates.