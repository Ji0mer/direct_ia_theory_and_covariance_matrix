from __future__ import print_function

import os
import sys

import numpy as np
import scipy.integrate as sint
import scipy.interpolate as spi
from cosmosis.datablock import names, option_section

TRAPEZOID = getattr(sint, "trapezoid", np.trapz)

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
from photoz_corrs_exact import cache_dir_from_file, load_cache_arrays, save_cache_arrays
from tatt_interface import FASTPT_KEYS, PkInterp, compute_amplitudes, grow


GM_TERM_SPECS = (
    ("Pd1d1", "Pnl"),
    ("Pd1d2", "Pd1d2"),
    ("Pd1s2", "Pd1s2"),
    ("sig3nl", "sig3nl"),
    ("k2P", "k2P"),
)


def template_cache_files(block, config):
    density_sample = config["density_sample"]
    shape_sample = config["shape_sample"]
    nbin_a = block["nz_%s" % density_sample, "nbin"]
    nbin_b = block["nz_%s" % shape_sample, "nbin"]

    files = ["k_h.npy"]
    for i in range(1, nbin_a + 1):
        for j in range(1, nbin_b + 1):
            for coeff_name, _ in GM_TERM_SPECS:
                files.append(f"gp_nla_{i}_{j}_{coeff_name}.npy")
                if config["ia_model"] == "tatt":
                    files.append(f"gp_ta_{i}_{j}_{coeff_name}.npy")
                    files.append(f"gp_tt_{i}_{j}_{coeff_name}.npy")
            files.append(f"pp_nla_{j}_{j}.npy")
            if config["ia_model"] == "tatt":
                files.extend(
                    [
                        f"pp_ta_ee_{j}_{j}.npy",
                        f"pp_ta_cross_{j}_{j}.npy",
                        f"pp_tt_ee_{j}_{j}.npy",
                        f"pp_mix_ab_{j}_{j}.npy",
                        f"pp_mix_d_{j}_{j}.npy",
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
        "template_cache_dir": options.get_string(
            option_section,
            "template_cache_dir",
            default="output/accelerated_cache/projected_ia_templates",
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


def get_growth_and_grids(block):
    z_lin, k_lin, p_lin = block.get_grid(names.matter_power_lin, "z", "k_h", "p_k")
    z_nl, k_nl, p_nl = block.get_grid(names.matter_power_nl, "z", "k_h", "p_k")
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


def build_window(block, sample_a, sample_b, bin_i, bin_j):
    z = block["distances", "z"]
    chi = block["distances", "d_m"]
    interp_chi = spi.interp1d(z, chi)
    dz = z[1] - z[0]
    dchi = np.gradient(chi, dz)
    interp_dchi = spi.interp1d(z, dchi)

    za = block["nz_%s" % sample_a, "z"]
    nz_a = block["nz_%s" % sample_a, "bin_%d" % bin_i]
    zb = block["nz_%s" % sample_b, "z"]
    nz_b = block["nz_%s" % sample_b, "bin_%d" % bin_j]
    if len(za) != len(zb):
        nz_b = spi.interp1d(zb, nz_b)(za)

    x = interp_chi(za)
    dxdz = interp_dchi(za)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = nz_a * nz_b / x / x / dxdz
    X[0] = 0.0
    V = np.trapz(X, za)
    if V == 0.0:
        return za, np.zeros_like(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = nz_a * nz_b / x / x / dxdz / V
    W[~np.isfinite(W)] = 1.0e-30
    return za, W


def project_pk_window(field_zk, k_h, z_field, z_window, window):
    log_k = np.log(k_h)
    if len(field_zk[field_zk > 0]) == len(field_zk.ravel()):
        spline = spi.RectBivariateSpline(z_field, log_k, np.log(field_zk), kx=1, ky=1)
        field_interp = np.exp(spline(z_window, log_k))
    else:
        spline = spi.RectBivariateSpline(z_field, log_k, field_zk, kx=1, ky=1)
        field_interp = spline(z_window, log_k)

    W2d, _ = np.meshgrid(window, k_h)
    W2d[np.invert(np.isfinite(W2d))] = 1.0e-30
    integrand = W2d.T * field_interp
    return TRAPEZOID(integrand, z_window, axis=0) / TRAPEZOID(W2d.T, z_window, axis=0)


def get_template_cache_key(block, config, profiles):
    z_lin, k_nl, p_nl, growth = get_growth_and_grids(block)
    fastpt_terms = load_fastpt_terms_safe(block, k_nl, z_lin, growth, config["sub_lowk"])
    return build_cache_key(
        [
            "projected_ia_basis_exact_v3",
            config["ia_model"],
            config["sub_lowk"],
            config["density_sample"],
            config["shape_sample"],
            profiles["alpha1"],
            profiles["alpha2"],
            profiles["alphadel"],
            profiles["z_piv"],
            block["distances", "z"],
            block["distances", "d_m"],
            block["nz_%s" % config["density_sample"], "z"],
            block["nz_%s" % config["density_sample"], "bin_1"],
            block["nz_%s" % config["shape_sample"], "z"],
            block["nz_%s" % config["shape_sample"], "bin_1"],
            z_lin,
            k_nl,
            p_nl,
            block["galaxy_power", "Pnl"],
            block["galaxy_power", "Pd1d2"],
            block["galaxy_power", "Pd1s2"],
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


def build_templates(block, config, profiles):
    z_lin, k_nl, p_nl, growth = get_growth_and_grids(block)
    fastpt_terms = load_fastpt_terms_safe(block, k_nl, z_lin, growth, config["sub_lowk"])
    density_sample = config["density_sample"]
    shape_sample = config["shape_sample"]
    c1_base = profiles["c1_base"]
    cdel_base = profiles["cdel_base"]
    c2_base = profiles["c2_base"]
    ta_sum = fastpt_terms["P_ta_dE1"] + fastpt_terms["P_ta_dE2"]
    mix_ab = fastpt_terms["P_mix_A"] + fastpt_terms["P_mix_B"]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_pnl = np.divide(1.0, p_nl, out=np.zeros_like(p_nl), where=p_nl != 0.0)

    templates = {"k_h": k_nl}
    nbin_a = block["nz_%s" % density_sample, "nbin"]
    nbin_b = block["nz_%s" % shape_sample, "nbin"]

    for i in range(1, nbin_a + 1):
        for j in range(1, nbin_b + 1):
            z_gp, w_gp = build_window(block, density_sample, shape_sample, i, j)
            z_pp, w_pp = build_window(block, shape_sample, shape_sample, j, j)

            for coeff_name, basis_name in GM_TERM_SPECS:
                gm_field = np.asarray(block["galaxy_power", basis_name])
                templates["gp_nla_%d_%d_%s" % (i, j, coeff_name)] = project_pk_window(
                    c1_base * gm_field, k_nl, z_lin, z_gp, w_gp
                )
                if config["ia_model"] == "tatt":
                    templates["gp_ta_%d_%d_%s" % (i, j, coeff_name)] = project_pk_window(
                        cdel_base * (gm_field * inv_pnl) * ta_sum,
                        k_nl,
                        z_lin,
                        z_gp,
                        w_gp,
                    )
                    templates["gp_tt_%d_%d_%s" % (i, j, coeff_name)] = project_pk_window(
                        c2_base * (gm_field * inv_pnl) * mix_ab,
                        k_nl,
                        z_lin,
                        z_gp,
                        w_gp,
                    )

            templates["pp_nla_%d_%d" % (j, j)] = project_pk_window(
                c1_base * c1_base * p_nl, k_nl, z_lin, z_pp, w_pp
            )
            if config["ia_model"] == "tatt":
                templates["pp_ta_ee_%d_%d" % (j, j)] = project_pk_window(
                    cdel_base * cdel_base * fastpt_terms["P_ta_EE"],
                    k_nl,
                    z_lin,
                    z_pp,
                    w_pp,
                )
                templates["pp_ta_cross_%d_%d" % (j, j)] = project_pk_window(
                    c1_base * cdel_base * (2.0 * ta_sum),
                    k_nl,
                    z_lin,
                    z_pp,
                    w_pp,
                )
                templates["pp_tt_ee_%d_%d" % (j, j)] = project_pk_window(
                    c2_base * c2_base * fastpt_terms["P_tt_EE"],
                    k_nl,
                    z_lin,
                    z_pp,
                    w_pp,
                )
                templates["pp_mix_ab_%d_%d" % (j, j)] = project_pk_window(
                    2.0 * c1_base * c2_base * mix_ab,
                    k_nl,
                    z_lin,
                    z_pp,
                    w_pp,
                )
                templates["pp_mix_d_%d_%d" % (j, j)] = project_pk_window(
                    2.0 * cdel_base * c2_base * fastpt_terms["P_mix_D_EE"],
                    k_nl,
                    z_lin,
                    z_pp,
                    w_pp,
                )

    return templates


def load_or_build_templates(block, config, profiles):
    cache_key = get_template_cache_key(block, config, profiles)
    cache_path = cache_file(
        config["template_cache_dir"], "projected_ia_templates", cache_key
    )
    cache_root = cache_dir_from_file(cache_path)
    required_files = template_cache_files(block, config)
    if is_complete_cache_dir(cache_root, required_files):
        return load_templates(cache_root)
    build_cache_dir(
        cache_root,
        lambda root: save_cache_arrays(root, build_templates(block, config, profiles)),
        required_files=required_files,
    )
    return load_templates(cache_root)


def gm_coeffs(block, pt_type, density_sample):
    bias_values = get_bias_params_bin(block, 1, pt_type, "bias_%s" % density_sample)
    if pt_type != "oneloop_eul_bk":
        raise NotImplementedError("projected_ia_basis_exact.py currently supports oneloop_eul_bk")
    return {
        "Pd1d1": bias_values["b1E"],
        "Pd1d2": 0.5 * bias_values["b2E"],
        "Pd1s2": 0.5 * bias_values["bsE"],
        "sig3nl": 0.5 * bias_values["b3nlE"],
        "k2P": bias_values["bkE"],
    }


def execute(block, config):
    if config["timing"]:
        from time import time

        t0 = time()

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
        templates = load_or_build_templates(block, config, profiles)
        if config["reuse_loaded_templates"]:
            config["_loaded_templates"] = templates
            config["_loaded_templates_key"] = template_mem_key

    if config["timing"]:
        t1 = time()
        print("Projected IA basis templates ready. Combining projected spectra...")

    density_sample = config["density_sample"]
    shape_sample = config["shape_sample"]
    gm = gm_coeffs(block, config["pt_type"], density_sample)
    a1 = profiles["a1"]
    a2 = profiles["a2"]
    adel_scalar = profiles["adel"]
    k_h = np.asarray(templates["k_h"])

    nbin_a = block["nz_%s" % density_sample, "nbin"]
    nbin_b = block["nz_%s" % shape_sample, "nbin"]
    block["galaxy_intrinsic_power_projected", "k_h"] = k_h
    block["galaxy_intrinsic_power_projected", "nbin_a"] = nbin_a
    block["galaxy_intrinsic_power_projected", "nbin_b"] = nbin_b
    block["intrinsic_power_projected", "k_h"] = k_h
    block["intrinsic_power_projected", "nbin_a"] = nbin_b
    block["intrinsic_power_projected", "nbin_b"] = nbin_b

    for i in range(1, nbin_a + 1):
        for j in range(1, nbin_b + 1):
            if config["no_IA_E"] or not config["do_galaxy_intrinsic"]:
                p_gp = np.zeros_like(k_h)
            else:
                p_gp = np.zeros_like(k_h)
                for name in gm:
                    p_gp += gm[name] * a1 * np.asarray(
                        templates["gp_nla_%d_%d_%s" % (i, j, name)]
                    )
                if config["ia_model"] == "tatt":
                    for name in gm:
                        p_gp += gm[name] * adel_scalar * np.asarray(
                            templates["gp_ta_%d_%d_%s" % (i, j, name)]
                        )
                        p_gp += gm[name] * a2 * np.asarray(
                            templates["gp_tt_%d_%d_%s" % (i, j, name)]
                        )

            gp_name = "p_k_%d_%d_%s_%s" % (i, j, density_sample, shape_sample)
            block["galaxy_intrinsic_power_projected", gp_name] = p_gp

    for j in range(1, nbin_b + 1):
        if config["no_IA_E"]:
            p_pp = np.zeros_like(k_h)
        else:
            p_pp = a1 * a1 * np.asarray(templates["pp_nla_%d_%d" % (j, j)])
            if config["ia_model"] == "tatt":
                p_pp += adel_scalar * adel_scalar * np.asarray(
                    templates["pp_ta_ee_%d_%d" % (j, j)]
                )
                p_pp += a1 * adel_scalar * np.asarray(
                    templates["pp_ta_cross_%d_%d" % (j, j)]
                )
                p_pp += a2 * a2 * np.asarray(templates["pp_tt_ee_%d_%d" % (j, j)])
                p_pp += a1 * a2 * np.asarray(templates["pp_mix_ab_%d_%d" % (j, j)])
                p_pp += adel_scalar * a2 * np.asarray(
                    templates["pp_mix_d_%d_%d" % (j, j)]
                )

        pp_name = "p_k_%d_%d_%s_%s" % (j, j, shape_sample, shape_sample)
        block["intrinsic_power_projected", pp_name] = p_pp

    if config["timing"]:
        t2 = time()
        print("Projected IA basis spectra done. Total Time:", t2 - t0)
        print("Template/cache preparation time:", t1 - t0)

    return 0
