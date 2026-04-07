from __future__ import print_function

import os
import sys

import numpy as np
from cosmosis.datablock import option_section
from hankl import FFTLog
from scipy.interpolate import RectBivariateSpline, interp1d

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from cache_utils import (
    build_cache_dir,
    build_cache_key,
    cache_file,
    ensure_dir,
    is_complete_cache_dir,
)


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
    return {
        "density_sample": options.get_string(
            option_section, "density_sample", default="forecast_sample_density"
        ),
        "shape_sample": options.get_string(
            option_section, "shape_sample", default="forecast_sample_shape"
        ),
        "timing": options.get_bool(option_section, "timing", default=True),
        "constant_sigmaz": options.get_bool(
            option_section, "constant_sigmaz", default=True
        ),
        "n_pi": options.get_int(option_section, "N_pi", default=200),
        "pi_mask_max": options.get_double(
            option_section, "Pi_mask_max", default=-1.0
        ),
        "cache_dir": options.get_string(
            option_section, "cache_dir", default="output/accelerated_cache/photoz"
        ),
        "reuse_loaded_cache": options.get_bool(
            option_section, "reuse_loaded_cache", default=False
        ),
    }


def build_common_state(block, constant_sigmaz, n_pi, pi_mask_max):
    nz_low = 200
    z_low = np.linspace(0.01, 4.00, nz_low)
    zf = np.linspace(0.0, 4.0, 400)

    pi = np.linspace(-block["LOS_bin", "Pi_max"], block["LOS_bin", "Pi_max"], n_pi)

    z_distance = block["distances", "z"]
    chi_distance = block["distances", "d_m"] * block["cosmological_parameters", "h0"]
    chi_of_z_spline = interp1d(
        z_distance, chi_distance, bounds_error=False, fill_value="extrapolate"
    )
    chi = chi_of_z_spline(zf)

    sigmaz = 0.01 if not constant_sigmaz else block["photoz_errors", "sigmaz"]

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
        "z_distance": z_distance,
        "chi_distance": chi_distance,
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
    window = np.zeros_like(kernel) if norm == 0 else kernel / norm
    return np.interp(z_low, zf, window)


def build_hankel_operator(ell, theta_values, corr_type):
    operator = np.empty((ell.size, theta_values.size))
    basis = np.zeros_like(ell)

    for idx in range(ell.size):
        basis.fill(0.0)
        basis[idx] = 1.0
        weighted_basis = basis * ell

        if corr_type == "wgg":
            theta_new, xi_new = FFTLog(ell, weighted_basis, 0, 0, lowring=True)
            xi_new = xi_new / theta_new / 2.0 / np.pi
            operator[idx] = np.interp(
                theta_values, theta_new, xi_new, left=xi_new[0], right=xi_new[-1]
            )
        elif corr_type == "wgp":
            theta_new, xi_new = FFTLog(ell, weighted_basis, 0, 2, lowring=True)
            xi_new = -xi_new / theta_new / 2.0 / np.pi
            operator[idx] = np.interp(
                theta_values, theta_new, xi_new, left=xi_new[0], right=xi_new[-1]
            )
        elif corr_type == "wpp":
            theta_new_0, xi_new_0 = FFTLog(ell, weighted_basis, 0, 0, lowring=True)
            theta_new_4, xi_new_4 = FFTLog(ell, weighted_basis, 0, 4, lowring=True)
            factor = 1.0 / (theta_new_0 * 2.0 * np.pi)
            xi_new_0 *= factor
            xi_new_4 *= factor
            operator[idx] = np.interp(theta_values, theta_new_0, xi_new_0) + np.interp(
                theta_values, theta_new_4, xi_new_4
            )
        else:
            raise ValueError("Unknown corr_type %s" % corr_type)

    return operator


def build_cached_operators(block, config):
    state = build_common_state(
        block,
        config["constant_sigmaz"],
        config["n_pi"],
        config["pi_mask_max"],
    )
    z_low = state["z_low"]
    zf = state["zf"]
    chi = state["chi"]
    theta_radians = state["theta_radians"]

    ell_gp = np.logspace(-6, np.log10(19000), 300)
    ell_pp = np.logspace(-6, np.log10(19000), 300)
    ell_gg = np.logspace(-6, np.log10(20000), 300)

    hankel_wgp = np.stack(
        [build_hankel_operator(ell_gp, theta_radians[i], "wgp") for i in range(len(z_low))],
        axis=0,
    )
    hankel_wpp = np.stack(
        [build_hankel_operator(ell_pp, theta_radians[i], "wpp") for i in range(len(z_low))],
        axis=0,
    )
    hankel_wgg = np.stack(
        [build_hankel_operator(ell_gg, theta_radians[i], "wgg") for i in range(len(z_low))],
        axis=0,
    )

    return {
        "z_low": z_low,
        "zf": zf,
        "chi": chi,
        "z_distance": state["z_distance"],
        "chi_distance": state["chi_distance"],
        "az_projected": state["az_projected"],
        "weights": state["weights"],
        "rp": state["rp"],
        "ell_gp": ell_gp,
        "ell_pp": ell_pp,
        "ell_gg": ell_gg,
        "chi_safe": np.where(chi == 0.0, 1.0, chi),
        "window_density_density": compute_window_weights(
            block,
            config["density_sample"],
            config["density_sample"],
            zf,
            z_low,
            chi,
        ),
        "window_density_shape": compute_window_weights(
            block,
            config["density_sample"],
            config["shape_sample"],
            zf,
            z_low,
            chi,
        ),
        "window_shape_shape": compute_window_weights(
            block,
            config["shape_sample"],
            config["shape_sample"],
            zf,
            z_low,
            chi,
        ),
        "hankel_wgp": hankel_wgp,
        "hankel_wpp": hankel_wpp,
        "hankel_wgg": hankel_wgg,
    }


def cache_dir_from_file(cache_path):
    return os.path.splitext(cache_path)[0]


def save_cache_arrays(cache_root, cache):
    ensure_dir(cache_root)
    for key, value in cache.items():
        with open(os.path.join(cache_root, f"{key}.npy"), "wb") as handle:
            np.save(handle, value, allow_pickle=False)


def load_cache_arrays(cache_root):
    cache = {}
    for filename in os.listdir(cache_root):
        if filename.endswith(".npy"):
            key = filename[:-4]
            cache[key] = np.load(os.path.join(cache_root, filename), mmap_mode="r")
    return cache


def load_or_build_cache(block, config):
    z_distance = block["distances", "z"]
    chi_distance = block["distances", "d_m"] * block["cosmological_parameters", "h0"]
    zf = np.linspace(0.0, 4.0, 400)
    cache_key = build_cache_key(
        [
            "photoz_corrs_exact",
            config["density_sample"],
            config["shape_sample"],
            config["constant_sigmaz"],
            config["n_pi"],
            config["pi_mask_max"],
            block["LOS_bin", "Pi_max"],
            block["photoz_errors", "sigmaz"] if config["constant_sigmaz"] else 0.01,
            z_distance,
            chi_distance,
            get_nz_on_grid(block, config["density_sample"], zf),
            get_nz_on_grid(block, config["shape_sample"], zf),
        ]
    )
    cache_path = cache_file(config["cache_dir"], "photoz_exact", cache_key)
    cache_root = cache_dir_from_file(cache_path)
    required_files = [
        "z_low.npy",
        "zf.npy",
        "chi.npy",
        "z_distance.npy",
        "chi_distance.npy",
        "az_projected.npy",
        "weights.npy",
        "rp.npy",
        "ell_gp.npy",
        "ell_pp.npy",
        "ell_gg.npy",
        "chi_safe.npy",
        "window_density_density.npy",
        "window_density_shape.npy",
        "window_shape_shape.npy",
        "hankel_wgp.npy",
        "hankel_wpp.npy",
        "hankel_wgg.npy",
    ]

    if is_complete_cache_dir(cache_root, required_files):
        return load_cache_arrays(cache_root)

    def write_cache(root):
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=False)
            cache = {key: data[key] for key in data.files}
        else:
            cache = build_cached_operators(block, config)
        save_cache_arrays(root, cache)

    build_cache_dir(
        cache_root,
        write_cache,
        required_files=required_files,
    )
    return load_cache_arrays(cache_root)


def interpolate_power_2d_cached(block, section, ell, chi, chi_safe, z_distance, chi_distance):
    p_k = block[section, "p_k"]
    k_power = block[section, "k_h"]
    z_power = block[section, "z"]
    chi_power = np.interp(z_power, z_distance, chi_distance)
    spline = RectBivariateSpline(np.log(k_power), chi_power, p_k.T)
    ell_grid = ell[:, None]
    k_eval = (ell_grid + 0.5) / chi_safe[None, :]
    chi_eval = np.broadcast_to(chi[None, :], k_eval.shape)
    return spline.ev(np.log(k_eval), chi_eval).reshape(ell.shape[0], chi.shape[0])


def project_cells(az_projected, power_2d, weights):
    return np.dot(az_projected, (power_2d * weights[None, :]).T)


def hankel_project(cell_projected, hankel_operator):
    return np.einsum("ze,zer->zr", cell_projected, hankel_operator, optimize=True)


def integrate_over_redshift(xi_projected, window_at_zlow, z_low):
    return np.trapz(xi_projected * window_at_zlow[:, None], x=z_low, axis=0)


def execute(block, config):
    if config["timing"]:
        from time import time

        t0 = time()

    if config["reuse_loaded_cache"] and "_loaded_cache" in config:
        cache = config["_loaded_cache"]
    else:
        cache = load_or_build_cache(block, config)
        if config["reuse_loaded_cache"]:
            config["_loaded_cache"] = cache
    z_low = cache["z_low"]
    chi = cache["chi"]
    az_projected = cache["az_projected"]
    weights = cache["weights"]
    rp = cache["rp"]
    z_distance = cache["z_distance"]
    chi_distance = cache["chi_distance"]
    chi_safe = cache["chi_safe"]

    if config["timing"]:
        t1 = time()
        print("Photo-z exact cache ready. Starting projected spectra...")

    p_gi_2d = interpolate_power_2d_cached(
        block,
        "galaxy_intrinsic_power",
        cache["ell_gp"],
        chi,
        chi_safe,
        z_distance,
        chi_distance,
    )
    cell_gp = project_cells(az_projected, p_gi_2d, weights)
    xi_gp = hankel_project(cell_gp, cache["hankel_wgp"])
    wgp = integrate_over_redshift(xi_gp, cache["window_density_shape"], z_low)

    p_ii_2d = interpolate_power_2d_cached(
        block,
        "intrinsic_power",
        cache["ell_pp"],
        chi,
        chi_safe,
        z_distance,
        chi_distance,
    )
    cell_pp = project_cells(az_projected, p_ii_2d, weights)
    xi_pp = hankel_project(cell_pp, cache["hankel_wpp"])
    wpp = integrate_over_redshift(xi_pp, cache["window_shape_shape"], z_low)

    p_gg = (
        block["galaxy_power", "p_k"]
        * block["galaxy_power", "blin_1"]
        * block["galaxy_power", "blin_2"]
    )
    k_power = block["galaxy_power", "k_h"]
    z_power = block["galaxy_power", "z"]
    chi_power = np.interp(z_power, z_distance, chi_distance)
    gg_spline = RectBivariateSpline(np.log(k_power), chi_power, p_gg.T)
    ell_grid = cache["ell_gg"][:, None]
    k_eval = (ell_grid + 0.5) / chi_safe[None, :]
    chi_eval = np.broadcast_to(chi[None, :], k_eval.shape)
    p_gg_2d = gg_spline.ev(np.log(k_eval), chi_eval).reshape(
        cache["ell_gg"].shape[0], chi.shape[0]
    )
    cell_gg = project_cells(az_projected, p_gg_2d, weights)
    xi_gg = hankel_project(cell_gg, cache["hankel_wgg"])
    wgg = integrate_over_redshift(xi_gg, cache["window_density_density"], z_low)

    density_sample = config["density_sample"]
    shape_sample = config["shape_sample"]
    block["galaxy_intrinsic_w", "w_rp_1_1_%s_%s" % (density_sample, shape_sample)] = wgp
    block["galaxy_intrinsic_w", "r_p"] = rp
    block["intrinsic_w", "w_rp_1_1_%s_%s" % (shape_sample, shape_sample)] = wpp
    block["intrinsic_w", "r_p"] = rp
    block["galaxy_w", "w_rp_1_1_%s_%s" % (density_sample, density_sample)] = wgg
    block["galaxy_w", "r_p"] = rp

    if config["timing"]:
        t2 = time()
        print("Photo-z exact projected correlations done. Total Time:", t2 - t0)
        print("Cache preparation time:", t1 - t0)

    return 0
