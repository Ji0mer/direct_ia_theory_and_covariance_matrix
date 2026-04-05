from __future__ import print_function

import os

import numpy as np
import scipy.integrate as sint

from cosmosis.datablock import option_section

from legendre_interface import (
    Projected_Corr_RSD,
    add_gg_mag_terms,
    get_redshift_kernel,
    interp_power,
    setup as base_setup,
)


XI_CACHE_FILENAME = "wgg_xi_terms_fast.npz"
MULTIPOLES = (0, 2, 4)


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
    term_names = (
        "Pk1_Pd1d1.npz",
        "Pk2_Pd1d2.npz",
        "Pk3_Pd2d2.npz",
        "Pk4_Pd1s2.npz",
        "Pk5_Pd2s2.npz",
        "Pk6_Ps2s2.npz",
        "Pk7_sig3nl.npz",
        "Pk8_k2P.npz",
    )
    return [np.load(os.path.join(pks_folder, name))["arr_0"] for name in term_names]


def cache_is_valid(cache_path, rp, z1, knew, pimax, pks_folder):
    if not os.path.exists(cache_path):
        return False

    cache_mtime = os.path.getmtime(cache_path)
    term_paths = [
        os.path.join(pks_folder, "Pk%d_%s.npz" % (idx, suffix))
        for idx, suffix in (
            (1, "Pd1d1"),
            (2, "Pd1d2"),
            (3, "Pd2d2"),
            (4, "Pd1s2"),
            (5, "Pd2s2"),
            (6, "Ps2s2"),
            (7, "sig3nl"),
            (8, "k2P"),
        )
    ]
    for path in term_paths:
        if os.path.exists(path) and os.path.getmtime(path) > cache_mtime:
            return False

    data = np.load(cache_path)
    return (
        np.array_equal(data["rp"], rp)
        and np.array_equal(data["z1"], z1)
        and np.array_equal(data["knew"], knew)
        and float(data["pimax"]) == float(pimax)
    )


def load_xi_terms(cache_path):
    data = np.load(cache_path)
    return {
        0: data["xi0_terms"],
        2: data["xi2_terms"],
        4: data["xi4_terms"],
    }


def save_xi_terms(cache_path, rp, z1, knew, pimax, xi_terms):
    np.savez(
        cache_path,
        rp=rp,
        z1=z1,
        knew=knew,
        pimax=np.array(float(pimax)),
        xi0_terms=xi_terms[0],
        xi2_terms=xi_terms[2],
        xi4_terms=xi_terms[4],
    )


def build_xi_terms_from_pk_terms(X, k, z, pk_terms, knew, z1, fz, ba, bb):
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
    return base_setup(options)


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
        cl_dir,
        pks_folder,
        wgg_folder,
    ) = config

    if block.has_value("LOS_bin", "Pi_max"):
        pimax = block["LOS_bin", "Pi_max"]
    elif pimax is None:
        raise ValueError(
            "Projected correlations require Pi_max. Set [LOS_bin] Pi_max in the values file "
            "or provide pimax in the module options."
        )

    k = block["galaxy_power", "k_h"]
    knew = np.logspace(np.log10(0.001), np.log10(k.max()), nk)
    X = Projected_Corr_RSD(rp=rp, pi_max=pimax, k=knew, lowring=True)

    if do_rsd:
        z1 = block["growth_parameters", "z"]
        dz = block["growth_parameters", "d_z"] / block["growth_parameters", "d_z"][0]
        lnD = np.log(dz)
        lna = np.log(block["growth_parameters", "a"])
        fz = np.gradient(lnD, lna)
        beta2 = -1
    else:
        z1 = block["growth_parameters", "z"]
        fz = 0.0
        beta2 = 0.0

    pknames = {"wgg": "galaxy_power", "wgp": "galaxy_intrinsic_power"}

    for corr, s1, s2 in zip(corrs, sample_a, sample_b):
        ba = block["bias_parameters", "b_%s" % s1] if ("bias_parameters", "b_%s" % s1) in block.keys() else 1.0
        bb = block["bias_parameters", "b_%s" % s2] if ("bias_parameters", "b_%s" % s2) in block.keys() else 1.0

        P = block[pknames[corr], "p_k"]
        z = block[pknames[corr], "z"]

        if corr != "wgg":
            raise NotImplementedError("legendre_interface_fast.py currently supports wgg only")

        za, W_kernel = get_redshift_kernel(block, 0, 0, z1, block["distances", "d_m"], s1, s2)
        z0 = np.trapz(za * W_kernel, za)

        if do_magnification:
            raise NotImplementedError(
                "legendre_interface_fast.py does not implement magnification corrections"
            )

        bb = ba
        bv1 = np.asarray(block[pknames[corr], "bias_values_a[bin1]"])
        bv2 = np.asarray(block[pknames[corr], "bias_values_b[bin2]"])
        coeffs = return_pk_terms(bv1, bv2)
        blin_1 = block["galaxy_power", "blin_1"]
        blin_2 = block["galaxy_power", "blin_2"]

        cache_path = os.path.join(wgg_folder, XI_CACHE_FILENAME)
        if cache_is_valid(cache_path, rp, z1, knew, pimax, pks_folder):
            xi_terms = load_xi_terms(cache_path)
        else:
            xi_terms = build_xi_terms_from_pk_terms(
                X, k, z, load_pk_terms(pks_folder), knew, z1, fz, ba, bb
            )
            save_xi_terms(cache_path, rp, z1, knew, pimax, xi_terms)

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

        W_flat = sint.trapz(W * W_kernel[:, np.newaxis], z1, axis=0) / sint.trapz(W_kernel, z1)

        section = pknames[corr].replace("_power", "_w")
        block.put_double_array_1d(section, "w_rp_1_1_%s_%s" % (s1, s2), W_flat)
        try:
            block.put_double_array_1d(section, "r_p", X.rp)
        except Exception:
            block.replace_double_array_1d(section, "r_p", X.rp)

    return 0
