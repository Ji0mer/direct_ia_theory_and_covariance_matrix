# coding:utf-8
import os

from cosmosis.datablock import option_section

from fastpt_tools import get_PXX, get_PXm, get_Pk_basis_funcs, get_bias_params_bin


def folder_has_files(folder_path):
    return os.path.isdir(folder_path) and any(
        os.path.isfile(os.path.join(folder_path, name)) for name in os.listdir(folder_path)
    )


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


def maybe_save_pk_terms(pks_folder, basis_funcs):
    if folder_has_files(pks_folder):
        return

    os.makedirs(pks_folder, exist_ok=True)
    term_names = {
        "Pk1_Pd1d1.npz": "Pnl",
        "Pk2_Pd1d2.npz": "Pd1d2",
        "Pk3_Pd2d2.npz": "Pd2d2",
        "Pk4_Pd1s2.npz": "Pd1s2",
        "Pk5_Pd2s2.npz": "Pd2s2",
        "Pk6_Ps2s2.npz": "Ps2s2",
        "Pk7_sig3nl.npz": "sig3nl",
        "Pk8_k2P.npz": "k2P",
    }
    for filename, key in term_names.items():
        path = os.path.join(pks_folder, filename)
        if not os.path.exists(path):
            # Match the original cache format used by legendre_interface.py.
            import numpy as np

            np.savez(path, basis_funcs[key])


def set_linear_bias_aliases(block, samples, lin_bias_prefix):
    for sample in samples:
        block["bias_parameters", "%s_%s" % (lin_bias_prefix, sample)] = block[
            "bias_%s" % sample, "b1E_bin1"
        ]


def build_galaxy_power(block, sample_a, sample_b, pt_type, pks_folder, basis_funcs):
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
    block["galaxy_power", "Pnl"] = basis_funcs["Pnl"]
    block["galaxy_power", "Pd1d2"] = basis_funcs["Pd1d2"]
    block["galaxy_power", "Pd2d2"] = basis_funcs["Pd2d2"]
    block["galaxy_power", "Pd1s2"] = basis_funcs["Pd1s2"]
    block["galaxy_power", "Pd2s2"] = basis_funcs["Pd2s2"]
    block["galaxy_power", "Ps2s2"] = basis_funcs["Ps2s2"]
    block["galaxy_power", "sig3nl"] = basis_funcs["sig3nl"]
    block["galaxy_power", "k2P"] = basis_funcs["k2P"]

    maybe_save_pk_terms(pks_folder, basis_funcs)

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

    # Keep the original side-effect keys used downstream.
    block["galaxy_power", "blin_1"] = lin_bias_values_a[1]


def setup(options):
    pks_folder = options.get_string(option_section, "pks_folder")
    os.makedirs(pks_folder, exist_ok=True)

    return {
        "pks_folder": pks_folder,
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

    needs_fastpt = bool(config["nlgal_nlgal_pairs"] or config["nlgal_shear_pairs"])
    if not needs_fastpt:
        return 0

    k_nl_bias, basis_funcs = get_Pk_basis_funcs(
        block, config["pt_type"], output_nl_grid=True
    )

    # Match the original nlbias output k grid.
    if k_nl_bias.shape == block["matter_power_nl", "k_h"].shape:
        pass

    for sample_a, sample_b in config["nlgal_nlgal_pairs"]:
        build_galaxy_power(
            block,
            sample_a,
            sample_b,
            config["pt_type"],
            config["pks_folder"],
            basis_funcs,
        )

    for sample_a, _ in config["nlgal_shear_pairs"]:
        build_matter_galaxy_power(block, sample_a, config["pt_type"], basis_funcs)

    return 0
