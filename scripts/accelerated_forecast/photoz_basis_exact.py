from __future__ import print_function

import os
import sys

import numpy as np
from cosmosis.datablock import option_section

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
NONLINEAR_BIAS_DIR = os.path.abspath(os.path.join(MODULE_DIR, "..", "nonlinear_bias"))
TATT_DIR = os.path.abspath(os.path.join(MODULE_DIR, "..", "tatt"))
for path in (MODULE_DIR, NONLINEAR_BIAS_DIR, TATT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from cache_utils import (
    build_cache_dir,
    build_cache_key,
    cache_file,
    is_complete_cache_dir,
)
from fastpt_tools import get_bias_params_bin
from photoz_corrs_exact import (
    build_cached_operators,
    cache_dir_from_file,
    load_cache_arrays,
    project_cells,
    hankel_project,
    integrate_over_redshift,
    save_cache_arrays,
)
from tatt_interface import FASTPT_KEYS, PkInterp, compute_amplitudes, grow


GM_TERM_SPECS = (
    ("Pd1d1", "Pnl"),
    ("Pd1d2", "Pd1d2"),
    ("Pd1s2", "Pd1s2"),
    ("sig3nl", "sig3nl"),
    ("k2P", "k2P"),
)

GG_TERM_SPECS = (
    ("Pd1d1", "Pnl"),
    ("Pd1d2", "Pd1d2"),
    ("Pd2d2", "Pd2d2"),
    ("Pd1s2", "Pd1s2"),
    ("Pd2s2", "Pd2s2"),
    ("Ps2s2", "Ps2s2"),
    ("sig3nl", "sig3nl"),
    ("k2P", "k2P"),
)

OPERATOR_CACHE_FILES = [
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


def template_cache_files(config):
    files = [f"wgg_{coeff_name}.npy" for coeff_name, _ in GG_TERM_SPECS]
    files.extend(f"wgp_nla_{coeff_name}.npy" for coeff_name, _ in GM_TERM_SPECS)
    files.append("wpp_nla.npy")
    if config["ia_model"] == "tatt":
        files.extend(f"wgp_ta_{coeff_name}.npy" for coeff_name, _ in GM_TERM_SPECS)
        files.extend(f"wgp_tt_{coeff_name}.npy" for coeff_name, _ in GM_TERM_SPECS)
        files.extend(
            [
                "wpp_ta_ee.npy",
                "wpp_ta_cross.npy",
                "wpp_tt_ee.npy",
                "wpp_mix_ab.npy",
                "wpp_mix_d.npy",
            ]
        )
    return files


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
            option_section,
            "cache_dir",
            default="output/accelerated_cache/photoz_basis_exact",
        ),
        "template_cache_dir": options.get_string(
            option_section,
            "template_cache_dir",
            default="output/accelerated_cache/photoz_basis_templates",
        ),
        "reuse_loaded_cache": options.get_bool(
            option_section, "reuse_loaded_cache", default=False
        ),
        "reuse_loaded_templates": options.get_bool(
            option_section, "reuse_loaded_templates", default=False
        ),
        "ia_model": options.get_string(option_section, "ia_model", default="nla").lower(),
        "sub_lowk": options.get_bool(option_section, "sub_lowk", default=False),
        "do_galaxy_intrinsic": options.get_bool(
            option_section, "do_galaxy_intrinsic", default=True
        ),
        "no_IA_E": options.get_bool(option_section, "no_IA_E", default=False),
        "no_IA_B": options.get_bool(option_section, "no_IA_B", default=False),
        "pt_type": options.get_string(option_section, "pt_type", default="oneloop_eul_bk"),
    }


def load_or_build_operator_cache(block, config):
    z_distance = block["distances", "z"]
    chi_distance = block["distances", "d_m"] * block["cosmological_parameters", "h0"]
    zf = np.linspace(0.0, 4.0, 400)
    density_section = "nz_%s" % config["density_sample"]
    shape_section = "nz_%s" % config["shape_sample"]
    density_nz = np.interp(
        zf,
        block[density_section, "z"],
        block[density_section, "bin_1"],
        left=0.0,
        right=0.0,
    )
    shape_nz = np.interp(
        zf,
        block[shape_section, "z"],
        block[shape_section, "bin_1"],
        left=0.0,
        right=0.0,
    )
    cache_key = build_cache_key(
        [
            "photoz_basis_exact_operator",
            config["density_sample"],
            config["shape_sample"],
            config["constant_sigmaz"],
            config["n_pi"],
            config["pi_mask_max"],
            block["LOS_bin", "Pi_max"],
            block["photoz_errors", "sigmaz"] if config["constant_sigmaz"] else 0.01,
            z_distance,
            chi_distance,
            density_nz,
            shape_nz,
        ]
    )
    cache_path = cache_file(config["cache_dir"], "photoz_basis_operator", cache_key)
    cache_root = cache_dir_from_file(cache_path)
    if is_complete_cache_dir(cache_root, OPERATOR_CACHE_FILES):
        return load_cache_arrays(cache_root)
    build_cache_dir(
        cache_root,
        lambda root: save_cache_arrays(root, build_cached_operators(block, config)),
        required_files=OPERATOR_CACHE_FILES,
    )
    return load_cache_arrays(cache_root)


def load_fastpt_terms_safe(block, k_out, z_out, growth, sub_lowk):
    terms = {}
    for key in FASTPT_KEYS:
        z_fastpt, k_fastpt, power = block.get_grid("fastpt", "z", "k_h", key)

        if sub_lowk and key in {
            "P_tt_EE",
            "P_tt_BB",
            "P_ta_EE",
            "P_ta_BB",
            "P_mix_D_EE",
            "P_mix_D_BB",
        }:
            power = power.copy()
            power -= power[:, 0][:, np.newaxis]
            power[:, 0] = power[:, 1]

        if not np.allclose(z_fastpt, z_out):
            raise ValueError(
                "Expected fastpt z grid to match matter-power z grid for %s" % key
            )

        same_k = k_out.shape == k_fastpt.shape and np.allclose(k_out, k_fastpt)
        if same_k:
            terms[key] = power
            continue

        power_z0 = PkInterp(k_fastpt, power[0])(k_out)
        growth_power = 2 if key == "Plin" else 4
        terms[key] = grow(power_z0, growth, growth_power)

    return terms


def project_power_array(cache, p_k, k_h, z, corr_type):
    if corr_type == "wgp":
        ell = cache["ell_gp"]
        hankel = cache["hankel_wgp"]
        window = cache["window_density_shape"]
    elif corr_type == "wpp":
        ell = cache["ell_pp"]
        hankel = cache["hankel_wpp"]
        window = cache["window_shape_shape"]
    elif corr_type == "wgg":
        ell = cache["ell_gg"]
        hankel = cache["hankel_wgg"]
        window = cache["window_density_density"]
    else:
        raise ValueError("Unknown corr_type %s" % corr_type)

    from scipy.interpolate import RectBivariateSpline

    chi = cache["chi"]
    chi_safe = cache["chi_safe"]
    z_distance = cache["z_distance"]
    chi_distance = cache["chi_distance"]
    chi_power = np.interp(z, z_distance, chi_distance)
    spline = RectBivariateSpline(np.log(k_h), chi_power, p_k.T)
    ell_grid = ell[:, None]
    k_eval = (ell_grid + 0.5) / chi_safe[None, :]
    chi_eval = np.broadcast_to(chi[None, :], k_eval.shape)
    power_2d = spline.ev(np.log(k_eval), chi_eval).reshape(ell.shape[0], chi.shape[0])
    cell = project_cells(cache["az_projected"], power_2d, cache["weights"])
    xi = hankel_project(cell, hankel)
    return integrate_over_redshift(xi, window, cache["z_low"])


def get_growth_and_grids(block):
    z_lin, k_lin, p_lin = block.get_grid("matter_power_lin", "z", "k_h", "p_k")
    z_nl, k_nl, p_nl = block.get_grid("matter_power_nl", "z", "k_h", "p_k")
    if not np.array_equal(z_lin, z_nl):
        raise ValueError("Expected identical z grids in matter_power_lin and matter_power_nl")
    ind = np.where(k_lin > 0.03)[0][0]
    growth = np.sqrt(p_lin[:, ind] / p_lin[0, ind])
    return z_lin, k_nl, p_nl, growth


def get_amplitude_profiles(block, k_nl):
    ia_section = "intrinsic_alignment_parameters"
    a1 = block.get_double(ia_section, "A1", 1.0)
    a2 = block.get_double(ia_section, "A2", 1.0)
    alpha1 = block.get_double(ia_section, "alpha1", 0.0)
    alpha2 = block.get_double(ia_section, "alpha2", 0.0)
    alphadel = block.get_double(ia_section, "alphadel", alpha1)
    z_piv = block.get_double(ia_section, "z_piv", 0.0)
    if (ia_section, "Adel") in block:
        if (ia_section, "bias_ta") in block:
            raise ValueError("bias_ta is not used when Adel is specified.")
        adel = block.get_double(ia_section, "Adel", 1.0)
    else:
        adel = block.get_double(ia_section, "bias_ta", 1.0) * a1

    z_lin, _, _, growth = get_growth_and_grids(block)
    omega_m = block["cosmological_parameters", "omega_m"]
    c1_base, cdel_base, c2_base = compute_amplitudes(
        z_lin,
        growth,
        1.0,
        1.0,
        1.0,
        alpha1,
        alpha2,
        alphadel,
        z_piv,
        omega_m,
        len(k_nl),
    )
    return {
        "a1": a1,
        "a2": a2,
        "adel": adel,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "alphadel": alphadel,
        "z_piv": z_piv,
        "c1_base": c1_base,
        "cdel_base": cdel_base,
        "c2_base": c2_base,
    }


def get_template_cache_key(block, config, profiles):
    z_lin, k_nl, p_nl, growth = get_growth_and_grids(block)
    fastpt_terms = load_fastpt_terms_safe(block, k_nl, z_lin, growth, config["sub_lowk"])
    return build_cache_key(
        [
            "photoz_basis_exact_templates",
            config["ia_model"],
            config["sub_lowk"],
            config["density_sample"],
            config["shape_sample"],
            profiles["alpha1"],
            profiles["alpha2"],
            profiles["alphadel"],
            profiles["z_piv"],
            z_lin,
            k_nl,
            p_nl,
            block["galaxy_power", "Pnl"],
            block["galaxy_power", "Pd1d2"],
            block["galaxy_power", "Pd2d2"],
            block["galaxy_power", "Pd1s2"],
            block["galaxy_power", "Pd2s2"],
            block["galaxy_power", "Ps2s2"],
            block["galaxy_power", "sig3nl"],
            block["galaxy_power", "k2P"],
            fastpt_terms["P_ta_dE1"],
            fastpt_terms["P_ta_dE2"],
            fastpt_terms["P_ta_EE"],
            fastpt_terms["P_tt_EE"],
            fastpt_terms["P_mix_A"],
            fastpt_terms["P_mix_B"],
            fastpt_terms["P_mix_D_EE"],
        ]
    )


def load_templates(cache_root):
    templates = {}
    for filename in os.listdir(cache_root):
        if filename.endswith(".npy"):
            templates[filename[:-4]] = np.load(
                os.path.join(cache_root, filename), mmap_mode="r"
            )
    return templates


def build_templates(block, config, cache, profiles):
    z_lin, k_nl, p_nl, growth = get_growth_and_grids(block)
    fastpt_terms = load_fastpt_terms_safe(block, k_nl, z_lin, growth, config["sub_lowk"])
    c1_base = profiles["c1_base"]
    cdel_base = profiles["cdel_base"]
    c2_base = profiles["c2_base"]

    ta_sum = fastpt_terms["P_ta_dE1"] + fastpt_terms["P_ta_dE2"]
    mix_ab = fastpt_terms["P_mix_A"] + fastpt_terms["P_mix_B"]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_pnl = np.divide(1.0, p_nl, out=np.zeros_like(p_nl), where=p_nl != 0.0)

    templates = {}

    for coeff_name, basis_name in GG_TERM_SPECS:
        templates["wgg_%s" % coeff_name] = project_power_array(
            cache,
            np.asarray(block["galaxy_power", basis_name]),
            k_nl,
            z_lin,
            "wgg",
        )

    gm_fields = {}
    for coeff_name, basis_name in GM_TERM_SPECS:
        gm_fields[coeff_name] = np.asarray(block["galaxy_power", basis_name])
        templates["wgp_nla_%s" % coeff_name] = project_power_array(
            cache,
            c1_base * gm_fields[coeff_name],
            k_nl,
            z_lin,
            "wgp",
        )

    if config["ia_model"] == "tatt":
        for coeff_name, gm_field in gm_fields.items():
            ta_field = cdel_base * (gm_field * inv_pnl) * ta_sum
            tt_field = c2_base * (gm_field * inv_pnl) * mix_ab
            templates["wgp_ta_%s" % coeff_name] = project_power_array(
                cache, ta_field, k_nl, z_lin, "wgp"
            )
            templates["wgp_tt_%s" % coeff_name] = project_power_array(
                cache, tt_field, k_nl, z_lin, "wgp"
            )

    templates["wpp_nla"] = project_power_array(
        cache, c1_base * c1_base * p_nl, k_nl, z_lin, "wpp"
    )

    if config["ia_model"] == "tatt":
        templates["wpp_ta_ee"] = project_power_array(
            cache,
            cdel_base * cdel_base * fastpt_terms["P_ta_EE"],
            k_nl,
            z_lin,
            "wpp",
        )
        templates["wpp_ta_cross"] = project_power_array(
            cache,
            c1_base * cdel_base * (2.0 * ta_sum),
            k_nl,
            z_lin,
            "wpp",
        )
        templates["wpp_tt_ee"] = project_power_array(
            cache,
            c2_base * c2_base * fastpt_terms["P_tt_EE"],
            k_nl,
            z_lin,
            "wpp",
        )
        templates["wpp_mix_ab"] = project_power_array(
            cache,
            2.0 * c1_base * c2_base * mix_ab,
            k_nl,
            z_lin,
            "wpp",
        )
        templates["wpp_mix_d"] = project_power_array(
            cache,
            2.0 * cdel_base * c2_base * fastpt_terms["P_mix_D_EE"],
            k_nl,
            z_lin,
            "wpp",
        )

    return templates


def load_or_build_templates(block, config, cache, profiles):
    cache_key = get_template_cache_key(block, config, profiles)
    cache_path = cache_file(config["template_cache_dir"], "photoz_basis_templates", cache_key)
    cache_root = cache_dir_from_file(cache_path)
    required_files = template_cache_files(config)
    if is_complete_cache_dir(cache_root, required_files):
        return load_templates(cache_root)
    build_cache_dir(
        cache_root,
        lambda root: save_cache_arrays(root, build_templates(block, config, cache, profiles)),
        required_files=required_files,
    )
    return load_templates(cache_root)


def gm_coeffs(block, pt_type, density_sample):
    bias_values = get_bias_params_bin(block, 1, pt_type, "bias_%s" % density_sample)
    if pt_type != "oneloop_eul_bk":
        raise NotImplementedError("photoz_basis_exact.py currently supports oneloop_eul_bk")
    return {
        "Pd1d1": bias_values["b1E"],
        "Pd1d2": 0.5 * bias_values["b2E"],
        "Pd1s2": 0.5 * bias_values["bsE"],
        "sig3nl": 0.5 * bias_values["b3nlE"],
        "k2P": bias_values["bkE"],
    }


def gg_coeffs(block, pt_type, density_sample):
    bias_values = get_bias_params_bin(block, 1, pt_type, "bias_%s" % density_sample)
    if pt_type != "oneloop_eul_bk":
        raise NotImplementedError("photoz_basis_exact.py currently supports oneloop_eul_bk")
    b1 = bias_values["b1E"]
    b2 = bias_values["b2E"]
    bs = bias_values["bsE"]
    b3nl = bias_values["b3nlE"]
    bk = bias_values["bkE"]
    return {
        "Pd1d1": b1 * b1,
        "Pd1d2": b1 * b2,
        "Pd2d2": 0.25 * b2 * b2,
        "Pd1s2": b1 * bs,
        "Pd2s2": 0.5 * b2 * bs,
        "Ps2s2": 0.25 * bs * bs,
        "sig3nl": b1 * b3nl,
        "k2P": 2.0 * b1 * bk,
    }


def execute(block, config):
    if config["timing"]:
        from time import time

        t0 = time()

    if config["reuse_loaded_cache"] and "_loaded_cache" in config:
        cache = config["_loaded_cache"]
    else:
        cache = load_or_build_operator_cache(block, config)
        if config["reuse_loaded_cache"]:
            config["_loaded_cache"] = cache

    profiles = get_amplitude_profiles(block, block["matter_power_nl", "k_h"])
    template_mem_key = (
        config["ia_model"],
        profiles["alpha1"],
        profiles["alpha2"],
        profiles["alphadel"],
        profiles["z_piv"],
        config["sub_lowk"],
    )
    if config["reuse_loaded_templates"] and config.get("_loaded_templates_key") == template_mem_key:
        templates = config["_loaded_templates"]
    else:
        templates = load_or_build_templates(block, config, cache, profiles)
        if config["reuse_loaded_templates"]:
            config["_loaded_templates"] = templates
            config["_loaded_templates_key"] = template_mem_key

    if config["timing"]:
        t1 = time()
        print("Photo-z basis templates ready. Combining exact templates...")

    density_sample = config["density_sample"]
    shape_sample = config["shape_sample"]
    gg = gg_coeffs(block, config["pt_type"], density_sample)
    gm = gm_coeffs(block, config["pt_type"], density_sample)

    wgg = np.zeros_like(np.asarray(templates["wgg_Pd1d1"]))
    for name in gg:
        wgg += gg[name] * np.asarray(templates["wgg_%s" % name])

    if config["no_IA_E"]:
        wgp = np.zeros_like(wgg)
        wpp = np.zeros_like(wgg)
    else:
        a1 = profiles["a1"]
        a2 = profiles["a2"]
        adel_scalar = profiles["adel"]

        wgp = np.zeros_like(wgg)
        for name in gm:
            wgp += gm[name] * a1 * np.asarray(templates["wgp_nla_%s" % name])

        wpp = a1 * a1 * np.asarray(templates["wpp_nla"])

        if config["ia_model"] == "tatt":
            for name in gm:
                wgp += gm[name] * adel_scalar * np.asarray(templates["wgp_ta_%s" % name])
                wgp += gm[name] * a2 * np.asarray(templates["wgp_tt_%s" % name])
            wpp += adel_scalar * adel_scalar * np.asarray(templates["wpp_ta_ee"])
            wpp += a1 * adel_scalar * np.asarray(templates["wpp_ta_cross"])
            wpp += a2 * a2 * np.asarray(templates["wpp_tt_ee"])
            wpp += a1 * a2 * np.asarray(templates["wpp_mix_ab"])
            wpp += adel_scalar * a2 * np.asarray(templates["wpp_mix_d"])

        if not config["do_galaxy_intrinsic"]:
            wgp[:] = 0.0

    rp = np.asarray(cache["rp"])
    block["galaxy_intrinsic_w", "w_rp_1_1_%s_%s" % (density_sample, shape_sample)] = wgp
    block["galaxy_intrinsic_w", "r_p"] = rp
    block["intrinsic_w", "w_rp_1_1_%s_%s" % (shape_sample, shape_sample)] = wpp
    block["intrinsic_w", "r_p"] = rp
    block["galaxy_w", "w_rp_1_1_%s_%s" % (density_sample, density_sample)] = wgg
    block["galaxy_w", "r_p"] = rp

    if config["timing"]:
        t2 = time()
        print("Photo-z basis exact correlations done. Total Time:", t2 - t0)
        print("Template/cache preparation time:", t1 - t0)

    return 0
