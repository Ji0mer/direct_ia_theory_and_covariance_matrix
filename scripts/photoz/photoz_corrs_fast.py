from __future__ import print_function

import numpy as np
from cosmosis.datablock import option_section
from hankl import FFTLog
from scipy.interpolate import RectBivariateSpline, interp1d


clight = 299792.4580


def gaussian_val(diff, sigma):
    return np.exp(-(diff ** 2) / (2 * sigma ** 2))


def get_nz_on_grid(block, sample_name, z_grid):
    section = "nz_%s" % sample_name
    if not block.has_section(section):
        return np.zeros_like(z_grid)
    z_orig = block[section, "z"]
    nz_orig = block[section, "bin_1"]
    return np.interp(z_grid, z_orig, nz_orig, left=0.0, right=0.0)


def setup(options):
    density_sample = options.get_string(option_section, "density_sample", default="forecast_sample_density")
    shape_sample = options.get_string(option_section, "shape_sample", default="forecast_sample_shape")
    timing = options.get_bool(option_section, "timing", default=True)
    constant_sigmaz = options.get_bool(option_section, "constant_sigmaz", default=True)
    n_pi = options.get_int(option_section, "N_pi", default=200)
    pi_mask_max = options.get_double(option_section, "Pi_mask_max", default=-1.0)
    return density_sample, shape_sample, timing, constant_sigmaz, n_pi, pi_mask_max


def build_common_state(block, constant_sigmaz, n_pi, pi_mask_max):
    nz_low = 200
    z_low = np.linspace(0.01, 4.00, nz_low)
    zf = np.linspace(0.0, 4.0, 400)

    pi = np.linspace(-block["LOS_bin", "Pi_max"], block["LOS_bin", "Pi_max"], n_pi)

    z_distance = block["distances", "z"]
    chi_distance = block["distances", "d_m"] * block["cosmological_parameters", "h0"]
    chi_of_z_spline = interp1d(z_distance, chi_distance, bounds_error=False, fill_value="extrapolate")
    chi = chi_of_z_spline(zf)

    if not constant_sigmaz:
        sigmaz = 0.01
    else:
        sigmaz = block["photoz_errors", "sigmaz"]

    hz_all = 100.0 * np.sqrt(
        block["cosmological_parameters", "omega_m"] * (1 + z_low) ** 3
        + block["cosmological_parameters", "omega_lambda"]
    )
    z1_grid = z_low[:, None]
    z2_grid = z_low[:, None] + (hz_all[:, None] * pi[None, :] / clight)

    zf_b = zf[None, None, :]
    diff1 = zf_b - z1_grid[:, :, None]
    pz1_mat = gaussian_val(diff1, sigmaz)
    norm1 = np.trapz(pz1_mat, x=chi, axis=-1)
    norm1[norm1 == 0] = 1.0
    pz1_mat /= norm1[:, :, None]

    diff2 = zf_b - z2_grid[:, :, None]
    valid_mask = z2_grid >= 0
    pz2_mat = np.zeros_like(diff2)
    pz2_mat[valid_mask, :] = gaussian_val(diff2[valid_mask, :], sigmaz)
    norm2 = np.trapz(pz2_mat, x=chi, axis=-1)
    norm2[norm2 == 0] = 1.0
    pz2_mat /= norm2[:, :, None]

    with np.errstate(divide="ignore", invalid="ignore"):
        inv_chi2 = 1.0 / (chi ** 2)
        inv_chi2[np.isinf(inv_chi2)] = 0.0
        inv_chi2[np.isnan(inv_chi2)] = 0.0

    az_mat = inv_chi2[None, None, :] * pz1_mat * pz2_mat
    pi_max = block["LOS_bin", "Pi_max"] if pi_mask_max < 0.0 else pi_mask_max
    pi_mask = (pi >= -pi_max) & (pi <= pi_max)
    az_projected = np.trapz(az_mat[:, pi_mask, :], x=pi[pi_mask], axis=1)

    dchi = np.diff(chi)
    weights = np.zeros_like(chi)
    weights[1:-1] = 0.5 * (dchi[:-1] + dchi[1:])
    weights[0] = 0.5 * dchi[0]
    weights[-1] = 0.5 * dchi[-1]

    rp = np.logspace(np.log10(0.1), np.log10(300), 300)
    x0_arr = chi_of_z_spline(z_low)
    theta_radians = np.arctan(rp[None, :] / x0_arr[:, None])

    return {
        "z_low": z_low,
        "zf": zf,
        "chi": chi,
        "chi_of_z_spline": chi_of_z_spline,
        "az_projected": az_projected,
        "weights": weights,
        "rp": rp,
        "theta_radians": theta_radians,
    }


def compute_window_weights(block, sample_a, sample_b, zf, z_low, chi):
    dz = zf[1] - zf[0]
    dxdz = np.gradient(chi, dz)
    nz_a = get_nz_on_grid(block, sample_a, zf)
    nz_b = get_nz_on_grid(block, sample_b, zf)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = nz_a * nz_b / (chi ** 2) / dxdz
        kernel[np.isinf(kernel)] = 0.0
        kernel[np.isnan(kernel)] = 0.0
    kernel[0] = 0.0
    norm = np.trapz(kernel, x=zf)
    if norm == 0:
        window = np.zeros_like(kernel)
    else:
        window = kernel / norm
    return np.interp(z_low, zf, window)


def interpolate_power_2d(block, section, ell, chi):
    p_k = block[section, "p_k"]
    k_power = block[section, "k_h"]
    z_power = block[section, "z"]
    chi_power = interp1d(
        block["distances", "z"],
        block["distances", "d_m"] * block["cosmological_parameters", "h0"],
        bounds_error=False,
        fill_value="extrapolate",
    )(z_power)
    spline = RectBivariateSpline(np.log(k_power), chi_power, p_k.T)

    chi_safe = chi.copy()
    chi_safe[chi_safe == 0] = 1.0
    ell_grid = ell[:, None]
    k_eval = (ell_grid + 0.5) / chi_safe[None, :]
    p_2d = spline.ev(np.log(k_eval), np.tile(chi, (ell.shape[0], 1)))
    return p_2d.reshape(ell.shape[0], chi.shape[0])


def compute_projected_cells(az_projected, power_2d, weights):
    return np.dot(az_projected, (power_2d * weights[None, :]).T)


def hankel_wgg(ell, cell_projected, theta_radians):
    xi_projected = np.zeros((cell_projected.shape[0], theta_radians.shape[1]))
    for i in range(cell_projected.shape[0]):
        theta_new, xi_new = FFTLog(ell, cell_projected[i] * ell, 0, 0, lowring=True)
        xi_new = xi_new / theta_new / 2.0 / np.pi
        xi_projected[i] = np.interp(
            theta_radians[i], theta_new, xi_new, left=xi_new[0], right=xi_new[-1]
        )
    return xi_projected


def hankel_wgp(ell, cell_projected, theta_radians):
    xi_projected = np.zeros((cell_projected.shape[0], theta_radians.shape[1]))
    for i in range(cell_projected.shape[0]):
        theta_new, xi_new = FFTLog(ell, cell_projected[i] * ell, 0, 2, lowring=True)
        xi_new = -xi_new / theta_new / 2.0 / np.pi
        xi_projected[i] = np.interp(
            theta_radians[i], theta_new, xi_new, left=xi_new[0], right=xi_new[-1]
        )
    return xi_projected


def hankel_wpp(ell, cell_projected, theta_radians):
    xi_projected = np.zeros((cell_projected.shape[0], theta_radians.shape[1]))
    for i in range(cell_projected.shape[0]):
        theta_new_0, xi_new_0 = FFTLog(ell, cell_projected[i] * ell, 0, 0, lowring=True)
        theta_new_4, xi_new_4 = FFTLog(ell, cell_projected[i] * ell, 0, 4, lowring=True)
        fact_0 = 1.0 / (theta_new_0 * 2.0 * np.pi)
        xi_new_0 *= fact_0
        xi_new_4 *= fact_0
        xi_projected[i] = np.interp(theta_radians[i], theta_new_0, xi_new_0) + np.interp(
            theta_radians[i], theta_new_4, xi_new_4
        )
    return xi_projected


def integrate_over_redshift(xi_projected, window_at_zlow, z_low):
    return np.trapz(xi_projected * window_at_zlow[:, None], x=z_low, axis=0)


def execute(block, config):
    density_sample, shape_sample, timing, constant_sigmaz, n_pi, pi_mask_max = config

    if timing:
        from time import time

        t0 = time()

    state = build_common_state(block, constant_sigmaz, n_pi, pi_mask_max)
    z_low = state["z_low"]
    zf = state["zf"]
    chi = state["chi"]
    az_projected = state["az_projected"]
    weights = state["weights"]
    rp = state["rp"]
    theta_radians = state["theta_radians"]

    window_density_density = compute_window_weights(block, density_sample, density_sample, zf, z_low, chi)
    window_density_shape = compute_window_weights(block, density_sample, shape_sample, zf, z_low, chi)
    window_shape_shape = compute_window_weights(block, shape_sample, shape_sample, zf, z_low, chi)

    if timing:
        t1 = time()
        print("Photo-z common setup done. Starting projected spectra...")

    # wgp
    ell_gp = np.logspace(-6, np.log10(19000), 300)
    p_gi_2d = interpolate_power_2d(block, "galaxy_intrinsic_power", ell_gp, chi)
    cell_gp = compute_projected_cells(az_projected, p_gi_2d, weights)
    xi_gp = hankel_wgp(ell_gp, cell_gp, theta_radians)
    wgp = integrate_over_redshift(xi_gp, window_density_shape, z_low)

    # wpp
    ell_pp = np.logspace(-6, np.log10(19000), 300)
    p_ii_2d = interpolate_power_2d(block, "intrinsic_power", ell_pp, chi)
    cell_pp = compute_projected_cells(az_projected, p_ii_2d, weights)
    xi_pp = hankel_wpp(ell_pp, cell_pp, theta_radians)
    wpp = integrate_over_redshift(xi_pp, window_shape_shape, z_low)

    # wgg
    ell_gg = np.logspace(-6, np.log10(20000), 300)
    p_gg = block["galaxy_power", "p_k"] * block["galaxy_power", "blin_1"] * block["galaxy_power", "blin_2"]
    k_power = block["galaxy_power", "k_h"]
    z_power = block["galaxy_power", "z"]
    chi_power = state["chi_of_z_spline"](z_power)
    p_gg_spline = RectBivariateSpline(np.log(k_power), chi_power, p_gg.T)
    chi_safe = chi.copy()
    chi_safe[chi_safe == 0] = 1.0
    ell_grid = ell_gg[:, None]
    k_eval = (ell_grid + 0.5) / chi_safe[None, :]
    p_gg_2d = p_gg_spline.ev(np.log(k_eval), np.tile(chi, (ell_gg.shape[0], 1))).reshape(
        ell_gg.shape[0], chi.shape[0]
    )
    cell_gg = compute_projected_cells(az_projected, p_gg_2d, weights)
    xi_gg = hankel_wgg(ell_gg, cell_gg, theta_radians)
    wgg = integrate_over_redshift(xi_gg, window_density_density, z_low)

    block["galaxy_intrinsic_w", "w_rp_1_1_%s_%s" % (density_sample, shape_sample)] = wgp
    block["galaxy_intrinsic_w", "r_p"] = rp
    block["intrinsic_w", "w_rp_1_1_%s_%s" % (shape_sample, shape_sample)] = wpp
    block["intrinsic_w", "r_p"] = rp
    block["galaxy_w", "w_rp_1_1_%s_%s" % (density_sample, density_sample)] = wgg
    block["galaxy_w", "r_p"] = rp

    if timing:
        t2 = time()
        print("Photo-z projected correlations done. Total Time:", t2 - t0)

    return 0
