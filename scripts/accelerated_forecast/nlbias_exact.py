import os
import sys

import numpy as np
from cosmosis.datablock import option_section

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
NONLINEAR_BIAS_DIR = os.path.abspath(os.path.join(MODULE_DIR, "..", "nonlinear_bias"))
for path in (MODULE_DIR, NONLINEAR_BIAS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from cache_utils import build_cache_key, cache_file, ensure_dir
from fastpt_tools import get_PXX, get_PXm, get_Pk_basis_funcs, get_bias_params_bin


TERM_NAMES = {
    "Pnl": "Pk1_Pd1d1.npz",
    "Pd1d2": "Pk2_Pd1d2.npz",
    "Pd2d2": "Pk3_Pd2d2.npz",
    "Pd1s2": "Pk4_Pd1s2.npz",
    "Pd2s2": "Pk5_Pd2s2.npz",
    "Ps2s2": "Pk6_Ps2s2.npz",
    "sig3nl": "Pk7_sig3nl.npz",
    "k2P": "Pk8_k2P.npz",
}


def parse_sample_pairs(option_value):
    pairs = []
    if not option_value:
        return pairs
    for token in option_value.strip().split():
        sample_a, sample_b = token.split("-", 1)
        if ":" in sample_b:
            sample_b = sample_b.split(":", 1)[0]
        pairs.append((sample_a.strip(), sample_b.strip()))
    return pairs


def load_bias(block, sample, pt_type):
    bias_values = {1: get_bias_params_bin(block, 1, pt_type, "bias_%s" % sample)}
    lin_bias_values = {1: bias_values[1]["b1E"]}
    return bias_values, lin_bias_values


def set_linear_bias_aliases(block, samples, lin_bias_prefix):
    for sample in samples:
        block["bias_parameters", "%s_%s" % (lin_bias_prefix, sample)] = block[
            "bias_%s" % sample, "b1E_bin1"
        ]


def save_legacy_pk_terms(pks_folder, basis_funcs):
    ensure_dir(pks_folder)
    for key, filename in TERM_NAMES.items():
        path = os.path.join(pks_folder, filename)
        if not os.path.exists(path):
            np.savez(path, basis_funcs[key])


def cache_root_from_file(cache_path):
    return os.path.splitext(cache_path)[0]


def save_basis_cache(cache_root, basis_funcs, k_nl_bias):
    ensure_dir(cache_root)
    np.save(os.path.join(cache_root, "k_nl_bias.npy"), k_nl_bias)
    for key, value in basis_funcs.items():
        np.save(os.path.join(cache_root, f"{key}.npy"), value)


def load_basis_cache(cache_root):
    basis_funcs = {}
    for key in TERM_NAMES:
        basis_funcs[key] = np.load(os.path.join(cache_root, f"{key}.npy"), mmap_mode="r")
    k_nl_bias = np.load(os.path.join(cache_root, "k_nl_bias.npy"), mmap_mode="r")
    return k_nl_bias, basis_funcs


def load_or_build_basis_cache(block, config):
    z_lin, k_lin, p_lin = block.get_grid("matter_power_lin", "z", "k_h", "p_k")
    z_nl, k_nl, p_nl = block.get_grid("matter_power_nl", "z", "k_h", "p_k")

    cache_key = build_cache_key(
        ["nlbias_exact", config["pt_type"], z_lin, k_lin, p_lin, z_nl, k_nl, p_nl]
    )
    cache_path = cache_file(config["basis_cache_dir"], "nlbias_basis", cache_key)
    cache_root = cache_root_from_file(cache_path)

    if os.path.isdir(cache_root):
        return load_basis_cache(cache_root)

    k_nl_bias, basis_funcs = get_Pk_basis_funcs(
        block, config["pt_type"], output_nl_grid=True
    )
    save_basis_cache(cache_root, basis_funcs, k_nl_bias)
    save_legacy_pk_terms(config["pks_folder"], basis_funcs)
    return load_basis_cache(cache_root)


def build_galaxy_power(block, sample_a, sample_b, pt_type, basis_funcs):
    bias_values_a, lin_bias_values_a = load_bias(block, sample_a, pt_type)
    if sample_a == sample_b:
        bias_values_b = bias_values_a
        lin_bias_values_b = lin_bias_values_a
    else:
        bias_values_b, lin_bias_values_b = load_bias(block, sample_b, pt_type)

    block["galaxy_power", "bias_values_a[bin1]"] = [
        bias_values_a[1]["b1E"],
        bias_values_a[1]["b2E"],
        bias_values_a[1]["bsE"],
        bias_values_a[1]["b3nlE"],
        bias_values_a[1]["bkE"],
    ]
    block["galaxy_power", "bias_values_b[bin2]"] = [
        bias_values_b[1]["b1E"],
        bias_values_b[1]["b2E"],
        bias_values_b[1]["bsE"],
        bias_values_b[1]["b3nlE"],
        bias_values_b[1]["bkE"],
    ]
    for key in TERM_NAMES:
        block["galaxy_power", key] = basis_funcs[key]

    p_gg, _ = get_PXX(bias_values_a[1], bias_values_b[1], basis_funcs, pt_type)
    blin_1 = lin_bias_values_a[1]
    blin_2 = lin_bias_values_b[1]
    p_gg_div_bias = p_gg / blin_1 / blin_2

    z = block["matter_power_nl", "z"]
    k_h = block["matter_power_nl", "k_h"]

    block["galaxy_power", "blin_1"] = blin_1
    block["galaxy_power", "blin_2"] = blin_2
    block["galaxy_power", "k_h"] = k_h
    block["galaxy_power", "z"] = z
    block["galaxy_power", "p_k"] = p_gg_div_bias
    block["galaxy_power", "_cosmosis_order_p_k"] = block[
        "matter_power_nl",
        "_cosmosis_order_p_k",
    ]


def build_matter_galaxy_power(block, sample_a, pt_type, basis_funcs):
    bias_values_a, lin_bias_values_a = load_bias(block, sample_a, pt_type)
    p_gm, _ = get_PXm(bias_values_a[1], basis_funcs, pt_type)

    z = block["matter_power_nl", "z"]
    k_h = block["matter_power_nl", "k_h"]

    block["matter_galaxy_power", "k_h"] = k_h
    block["matter_galaxy_power", "z"] = z
    block["matter_galaxy_power", "p_k"] = p_gm
    block["matter_galaxy_power", "_cosmosis_order_p_k"] = block[
        "matter_power_nl",
        "_cosmosis_order_p_k",
    ]
    block["galaxy_power", "blin_1"] = lin_bias_values_a[1]


def setup(options):
    pks_folder = options.get_string(option_section, "pks_folder")
    basis_cache_dir = options.get_string(
        option_section, "basis_cache_dir", default=os.path.join(pks_folder, "basis_exact")
    )
    ensure_dir(pks_folder)
    ensure_dir(basis_cache_dir)

    return {
        "pks_folder": pks_folder,
        "basis_cache_dir": basis_cache_dir,
        "reuse_loaded_basis": options.get_bool(
            option_section, "reuse_loaded_basis", default=False
        ),
        "lin_bias_prefix": options.get_string(option_section, "lin_bias_prefix", "b"),
        "pt_type": options.get_string(option_section, "pt_type", "oneloop_eul_bk"),
        "nlgal_nlgal_pairs": parse_sample_pairs(
            options.get_string(option_section, "nlgal-nlgal", "")
        ),
        "nlgal_shear_pairs": parse_sample_pairs(
            options.get_string(option_section, "nlgal-shear", "")
        ),
    }


def execute(block, config):
    block["galaxy_power", "pks_folder"] = config["pks_folder"]

    density_samples = set()
    for sample_a, sample_b in config["nlgal_nlgal_pairs"]:
        density_samples.add(sample_a)
        density_samples.add(sample_b)
    for sample_a, _ in config["nlgal_shear_pairs"]:
        density_samples.add(sample_a)
    set_linear_bias_aliases(block, density_samples, config["lin_bias_prefix"])

    if not (config["nlgal_nlgal_pairs"] or config["nlgal_shear_pairs"]):
        return 0

    if config["reuse_loaded_basis"] and "_loaded_basis" in config:
        _, basis_funcs = config["_loaded_basis"]
    else:
        loaded_basis = load_or_build_basis_cache(block, config)
        if config["reuse_loaded_basis"]:
            config["_loaded_basis"] = loaded_basis
        _, basis_funcs = loaded_basis

    for sample_a, sample_b in config["nlgal_nlgal_pairs"]:
        build_galaxy_power(block, sample_a, sample_b, config["pt_type"], basis_funcs)

    for sample_a, _ in config["nlgal_shear_pairs"]:
        build_matter_galaxy_power(block, sample_a, config["pt_type"], basis_funcs)

    return 0
