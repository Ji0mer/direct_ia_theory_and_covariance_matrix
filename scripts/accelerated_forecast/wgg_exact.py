from __future__ import print_function

import os
import sys

import numpy as np
import scipy.integrate as sint
from cosmosis.datablock import option_section

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTION_DIR = os.path.abspath(
    os.path.join(MODULE_DIR, "..", "projection", "projected_corrs_legendre")
)
for path in (MODULE_DIR, PROJECTION_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from cache_utils import build_cache_key, cache_file, ensure_dir
from legendre_interface import Projected_Corr_RSD, get_redshift_kernel, interp_power


MULTIPOLES = (0, 2, 4)
TERM_FILES = (
    "Pk1_Pd1d1.npz",
    "Pk2_Pd1d2.npz",
    "Pk3_Pd2d2.npz",
    "Pk4_Pd1s2.npz",
    "Pk5_Pd2s2.npz",
    "Pk6_Ps2s2.npz",
    "Pk7_sig3nl.npz",
    "Pk8_k2P.npz",
)


def return_pk_terms(bv1, bv2):
    return np.array(
        [
            bv1[0] * bv2[0],
            0.5 * (bv1[0] * bv2[1] + bv2[0] * bv1[1]),
            0.25 * bv1[1] * bv2[1],
            0.5 * (bv1[0] * bv2[2] + bv2[0] * bv1[2]),
            0.25 * (bv2[1] * bv1[2] + bv1[1] * bv2[2]),
            0.25 * bv1[2] * bv2[2],
            0.5 * (bv1[0] * bv2[3] + bv2[0] * bv1[3]),
            bv1[0] * bv2[4] + bv2[0] * bv1[4],
        ]
    )


def _same_grid(z, znew):
    return z.shape == znew.shape and np.allclose(z, znew)


def _interp_power_same_z(input_k, input_power, knew):
    logk = np.log10(input_k)
    logknew = np.log10(knew)
    mini_power = np.min(input_power)
    modified_power = input_power - mini_power + 10.0
    out = np.empty((input_power.shape[0], knew.shape[0]))
    for idx in range(input_power.shape[0]):
        out[idx] = (
            10.0 ** np.interp(logknew, logk, np.log10(modified_power[idx])) - 10.0 + mini_power
        )
    return out


def interp_power_fast(input_k, input_z, input_power, knew, znew):
    if _same_grid(input_z, znew):
        return _interp_power_same_z(input_k, input_power, knew)
    return interp_power(input_k, input_z, input_power, knew, znew)


def load_pk_terms(pks_folder):
    return [np.load(os.path.join(pks_folder, name))["arr_0"] for name in TERM_FILES]


def cache_root_from_file(cache_path):
    return os.path.splitext(cache_path)[0]


def save_xi_cache(cache_root, xi_terms):
    ensure_dir(cache_root)
    for ell in MULTIPOLES:
        np.save(os.path.join(cache_root, f"xi{ell}_terms.npy"), xi_terms[ell])


def load_xi_cache(cache_root):
    return {
        ell: np.load(os.path.join(cache_root, f"xi{ell}_terms.npy"), mmap_mode="r")
        for ell in MULTIPOLES
    }


def build_xi_cache(X, k, z, pk_terms, knew, z1, fz, ba, bb):
    xi_terms = {ell: [] for ell in MULTIPOLES}
    for term in pk_terms:
        term_new = interp_power_fast(k, z, term, knew, z1)
        xi = X.get_xi(pk=term_new, l=list(MULTIPOLES))
        _, xi = X.xi_wgg(f=fz, bg=ba, bg2=bb, pk=term_new, xi=xi, l=list(MULTIPOLES))
        for ell in MULTIPOLES:
            xi_terms[ell].append(xi[ell])
    for ell in MULTIPOLES:
        xi_terms[ell] = np.stack(xi_terms[ell], axis=0)
    return xi_terms


def setup(options):
    sample_a = options.get_string(option_section, "sample_a", default="lens lens").split()
    sample_b = options.get_string(option_section, "sample_b", default="lens source").split()
    rmin = options.get_double(option_section, "rpmin", default=0.01)
    rmax = options.get_double(option_section, "rpmax", default=500.0)
    nr = options.get_int(option_section, "nr", default=1024)
    nk = options.get_int(option_section, "nk", default=200)
    rp = np.logspace(np.log10(rmin), np.log10(rmax), nr)
    pimax = options.get_double(option_section, "pimax", default=100.0)
    corrs = options.get_string(option_section, "correlations", default="wgg").split()
    do_rsd = options.get_bool(option_section, "include_rsd", default=False)
    do_lensing = options.get_bool(option_section, "include_lensing", default=False)
    do_magnification = options.get_bool(option_section, "include_magnification", default=False)
    pks_folder = options.get_string(option_section, "pks_folder")
    wgg_folder = options.get_string(option_section, "wgg_folder")
    xi_cache_dir = options.get_string(
        option_section, "xi_cache_dir", default=os.path.join(wgg_folder, "exact_xi")
    )
    ensure_dir(wgg_folder)
    ensure_dir(xi_cache_dir)
    return (
        sample_a,
        sample_b,
        rp,
        pimax,
        nk,
        corrs,
        do_rsd,
        do_lensing,
        do_magnification,
        pks_folder,
        xi_cache_dir,
        options.get_bool(option_section, "reuse_loaded_xi", default=False),
    )


def execute(block, config):
    (
        sample_a,
        sample_b,
        rp,
        pimax,
        nk,
        corrs,
        do_rsd,
        do_lensing,
        do_magnification,
        pks_folder,
        xi_cache_dir,
        reuse_loaded_xi,
    ) = config

    if block.has_value("LOS_bin", "Pi_max"):
        pimax = block["LOS_bin", "Pi_max"]

    k = block["galaxy_power", "k_h"]
    knew = np.logspace(np.log10(0.001), np.log10(k.max()), nk)
    X = Projected_Corr_RSD(rp=rp, pi_max=pimax, k=knew, lowring=True)

    if do_rsd:
        z1 = block["growth_parameters", "z"]
        dz = block["growth_parameters", "d_z"] / block["growth_parameters", "d_z"][0]
        lnD = np.log(dz)
        lna = np.log(block["growth_parameters", "a"])
        fz = np.gradient(lnD, lna)
    else:
        z1 = block["growth_parameters", "z"]
        fz = 0.0

    for corr, s1, s2 in zip(corrs, sample_a, sample_b):
        if corr != "wgg":
            raise NotImplementedError("wgg_exact.py supports wgg only")
        if do_lensing or do_magnification:
            raise NotImplementedError("wgg_exact.py keeps the exact fast-path without lensing terms")

        ba = (
            block["bias_parameters", "b_%s" % s1]
            if ("bias_parameters", "b_%s" % s1) in block.keys()
            else 1.0
        )
        bb = ba
        z = block["galaxy_power", "z"]
        bv1 = np.asarray(block["galaxy_power", "bias_values_a[bin1]"])
        bv2 = np.asarray(block["galaxy_power", "bias_values_b[bin2]"])
        coeffs = return_pk_terms(bv1, bv2)
        blin_1 = block["galaxy_power", "blin_1"]
        blin_2 = block["galaxy_power", "blin_2"]

        term_mtimes = []
        for name in TERM_FILES:
            path = os.path.join(pks_folder, name)
            term_mtimes.append(os.path.getmtime(path))
        cache_key = build_cache_key([rp, z1, knew, pimax, term_mtimes])
        cache_path = cache_file(xi_cache_dir, "wgg_exact", cache_key)
        cache_root = cache_root_from_file(cache_path)

        loaded_xi_by_root = getattr(execute, "_loaded_xi_by_root", {})
        if reuse_loaded_xi and cache_root in loaded_xi_by_root:
            xi_terms = loaded_xi_by_root[cache_root]
        else:
            if os.path.isdir(cache_root):
                xi_terms = load_xi_cache(cache_root)
            else:
                xi_terms = build_xi_cache(
                    X, k, z, load_pk_terms(pks_folder), knew, z1, fz, ba, bb
                )
                save_xi_cache(cache_root, xi_terms)
                xi_terms = load_xi_cache(cache_root)
            if reuse_loaded_xi:
                loaded_xi_by_root[cache_root] = xi_terms
                execute._loaded_xi_by_root = loaded_xi_by_root

        xisum = {}
        for ell in MULTIPOLES:
            xisum[ell] = np.tensordot(coeffs, xi_terms[ell], axes=(0, 0)) / blin_1 / blin_2

        beta1 = fz / ba
        beta2 = fz / bb
        W = (
            (xisum[0].T * X.alpha(0, beta1, beta2) * ba * bb).T
            + (xisum[2].T * X.alpha(2, beta1, beta2) * ba * bb).T
            + (xisum[4].T * X.alpha(4, beta1, beta2) * ba * bb).T
        )

        _, W_kernel = get_redshift_kernel(
            block, 0, 0, z1, block["distances", "d_m"], s1, s2
        )
        W_flat = sint.trapz(W * W_kernel[:, np.newaxis], z1, axis=0) / sint.trapz(W_kernel, z1)
        block.put_double_array_1d("galaxy_w", "w_rp_1_1_%s_%s" % (s1, s2), W_flat)
        try:
            block.put_double_array_1d("galaxy_w", "r_p", X.rp)
        except Exception:
            block.replace_double_array_1d("galaxy_w", "r_p", X.rp)

    return 0
