import numpy as np
import scipy.interpolate as interp
from cosmosis.datablock import names, option_section

SUPPORTED_IA_MODELS = {"nla", "tatt"}
FASTPT_KEYS = [
    "P_tt_EE",
    "P_tt_BB",
    "P_ta_dE1",
    "P_ta_dE2",
    "P_ta_EE",
    "P_ta_BB",
    "P_mix_A",
    "P_mix_B",
    "P_mix_D_EE",
    "P_mix_D_BB",
    "Plin",
]


class PkInterp:
    def __init__(self, ks, pks):
        if np.all(pks > 0):
            self.interp_func = interp.interp1d(
                np.log(ks), np.log(pks), bounds_error=False, fill_value=-np.inf
            )
            self.interp_type = "loglog"
        elif np.all(pks < 0):
            self.interp_func = interp.interp1d(
                np.log(ks), np.log(-pks), bounds_error=False, fill_value=-np.inf
            )
            self.interp_type = "minus_loglog"
        else:
            self.interp_func = interp.interp1d(
                np.log(ks), pks, bounds_error=False, fill_value=0.0
            )
            self.interp_type = "log_linear"

    def __call__(self, ks):
        values = self.interp_func(np.log(ks))
        if self.interp_type == "loglog":
            return np.exp(values)
        if self.interp_type == "minus_loglog":
            return -np.exp(values)
        return values


def compute_c1_baseline():
    c1_m_sun = 5e-14
    m_sun = 1.9891e30
    mpc_in_m = 3.0857e22
    c1_si = c1_m_sun / m_sun * (mpc_in_m**3)
    gravitational_constant = 6.67384e-11
    hubble = 100
    hubble_si = hubble * 1000.0 / mpc_in_m
    rho_crit_0 = 3 * hubble_si**2 / (8 * np.pi * gravitational_constant)
    return c1_si * rho_crit_0


def grow(power_spectrum_z0, growth, power):
    power_spectrum = np.zeros((len(growth), len(power_spectrum_z0)))
    for i, growth_i in enumerate(growth):
        power_spectrum[i] = power_spectrum_z0 * growth_i**power
    return power_spectrum


def amp_3d(amplitude, num_z, num_k):
    amplitude = np.atleast_1d(amplitude)
    if amplitude.shape == (1,):
        return amplitude * np.ones((num_z, num_k))
    if amplitude.shape == (num_z,):
        return np.outer(amplitude, np.ones(num_k))
    if amplitude.shape == (num_z, num_k):
        return amplitude
    raise ValueError(f"Unexpected amplitude shape {amplitude.shape}")


def compute_amplitudes(z, dz, a1, a2, adel, alpha1, alpha2, alphadel, z_piv, omega_m, num_k):
    c1_rhocrit = compute_c1_baseline()
    c1 = -a1 * c1_rhocrit * omega_m / dz * ((1.0 + z) / (1.0 + z_piv)) ** alpha1
    cdel = -adel * c1_rhocrit * omega_m / dz * ((1.0 + z) / (1.0 + z_piv)) ** alphadel
    c2 = 5 * a2 * c1_rhocrit * omega_m / dz**2 * ((1.0 + z) / (1.0 + z_piv)) ** alpha2
    return (
        amp_3d(c1, len(z), num_k),
        amp_3d(cdel, len(z), num_k),
        amp_3d(c2, len(z), num_k),
    )


def load_fastpt_terms(block, k_out, z_out, growth, sub_lowk):
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
            raise ValueError(f"Expected fastpt z grid to match matter-power z grid for {key}")

        if np.allclose(k_out, k_fastpt):
            terms[key] = power
            continue

        power_z0 = PkInterp(k_fastpt, power[0])(k_out)
        growth_power = 2 if key == "Plin" else 4
        terms[key] = grow(power_z0, growth, growth_power)

    return terms


def get_ia_terms(
    block,
    k_nl,
    z_out,
    p_nl,
    growth,
    a1,
    a2,
    adel,
    alpha1,
    alpha2,
    alphadel,
    z_piv,
    omega_m,
    sub_lowk=False,
):
    k_use = k_nl
    c1, cdel, c2 = compute_amplitudes(
        z_out, growth, a1, a2, adel, alpha1, alpha2, alphadel, z_piv, omega_m, len(k_use)
    )
    fastpt_terms = load_fastpt_terms(block, k_use, z_out, growth, sub_lowk)

    ta_ii_ee = cdel**2 * fastpt_terms["P_ta_EE"] + c1 * cdel * (
        2 * fastpt_terms["P_ta_dE1"] + 2 * fastpt_terms["P_ta_dE2"]
    )
    ta_ii_bb = cdel**2 * fastpt_terms["P_ta_BB"]
    ta_gi = cdel * (fastpt_terms["P_ta_dE1"] + fastpt_terms["P_ta_dE2"])

    tt_gi = c2 * (fastpt_terms["P_mix_A"] + fastpt_terms["P_mix_B"])
    tt_ii_ee = c2**2 * fastpt_terms["P_tt_EE"]
    tt_ii_bb = c2**2 * fastpt_terms["P_tt_BB"]

    mix_ii_ee = 2.0 * c2 * (
        c1 * fastpt_terms["P_mix_A"]
        + c1 * fastpt_terms["P_mix_B"]
        + cdel * fastpt_terms["P_mix_D_EE"]
    )
    mix_ii_bb = 2.0 * cdel * c2 * fastpt_terms["P_mix_D_BB"]

    return {
        "k_h": k_use,
        "nla_gi": c1 * p_nl,
        "nla_ii_ee": c1 * c1 * p_nl,
        "ta_gi": ta_gi,
        "ta_ii_ee": ta_ii_ee,
        "ta_ii_bb": ta_ii_bb,
        "tt_gi": tt_gi,
        "tt_ii_ee": tt_ii_ee,
        "tt_ii_bb": tt_ii_bb,
        "mix_ii_ee": mix_ii_ee,
        "mix_ii_bb": mix_ii_bb,
    }


def setup(options):
    sub_lowk = options.get_bool(option_section, "sub_lowk", False)
    ia_model = options.get_string(option_section, "ia_model", "nla").lower()
    if ia_model not in SUPPORTED_IA_MODELS:
        supported = ", ".join(sorted(SUPPORTED_IA_MODELS))
        raise ValueError(f"Unsupported ia_model '{ia_model}'. Supported models: {supported}")

    name = options.get_string(option_section, "name", default="").lower()
    do_galaxy_intrinsic = options.get_bool(option_section, "do_galaxy_intrinsic", False)
    no_ia_e = options.get_bool(option_section, "no_IA_E", False)
    no_ia_b = options.get_bool(option_section, "no_IA_B", False)
    suffix = f"_{name}" if name else ""
    return sub_lowk, ia_model, suffix, do_galaxy_intrinsic, no_ia_e, no_ia_b


def execute(block, config):
    sub_lowk, ia_model, suffix, do_galaxy_intrinsic, no_ia_e, no_ia_b = config

    lin = names.matter_power_lin
    nl = names.matter_power_nl
    cosmo = names.cosmological_parameters
    omega_m = block[cosmo, "omega_m"]

    z_lin, k_lin, p_lin = block.get_grid(lin, "z", "k_h", "p_k")
    z_nl, k_nl, p_nl = block.get_grid(nl, "z", "k_h", "p_k")

    if not np.array_equal(z_nl, z_lin):
        raise ValueError("Expected identical z values for matter power NL and Linear in IA code")

    ind = np.where(k_lin > 0.03)[0][0]
    growth = np.sqrt(p_lin[:, ind] / p_lin[0, ind])

    ia_section = "intrinsic_alignment_parameters"
    if (ia_section, "C1") in block or (ia_section, "C2") in block:
        raise ValueError("Deprecated TATT parameters C1/C2 are not supported")

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

    c1, _, _ = compute_amplitudes(
        z_lin, growth, a1, a2, adel, alpha1, alpha2, alphadel, z_piv, omega_m, len(k_nl)
    )
    nla_gi = c1 * p_nl
    nla_ii_ee = c1 * c1 * p_nl

    e_factor = 0 if no_ia_e else 1
    b_factor = 0 if no_ia_b else 1

    if ia_model == "nla":
        ii_ee_total = e_factor * nla_ii_ee
        ii_bb_total = np.zeros_like(ii_ee_total)
        gi_e_total = e_factor * nla_gi
        k_use = k_nl
    else:
        terms = get_ia_terms(
            block,
            k_nl,
            z_lin,
            p_nl,
            growth,
            a1,
            a2,
            adel,
            alpha1,
            alpha2,
            alphadel,
            z_piv,
            omega_m,
            sub_lowk=sub_lowk,
        )
        ii_ee_total = e_factor * (
            terms["nla_ii_ee"]
            + terms["ta_ii_ee"]
            + terms["tt_ii_ee"]
            + terms["mix_ii_ee"]
        )
        ii_bb_total = b_factor * (
            terms["ta_ii_bb"] + terms["tt_ii_bb"] + terms["mix_ii_bb"]
        )
        gi_e_total = e_factor * (
            terms["nla_gi"] + terms["ta_gi"] + terms["tt_gi"]
        )
        k_use = terms["k_h"]
    block.put_grid("intrinsic_power_ee" + suffix, "z", z_lin, "k_h", k_use, "p_k", ii_ee_total)
    block.put_grid("intrinsic_power_bb" + suffix, "z", z_lin, "k_h", k_use, "p_k", ii_bb_total)
    block.put_grid(names.matter_intrinsic_power + suffix, "z", z_lin, "k_h", k_use, "p_k", gi_e_total)
    block.put_grid(names.intrinsic_power + suffix, "z", z_lin, "k_h", k_use, "p_k", ii_ee_total)

    if do_galaxy_intrinsic:
        gm = "matter_galaxy_power" + suffix
        _, _, p_gm = block.get_grid(gm, "z", "k_h", "p_k")
        if p_gm.shape == p_nl.shape and np.allclose(p_gm, p_nl):
            bias = 1
        else:
            print("WARNING: bias has already been applied to P_gm.")
            print("b_temp=P_gm/P_NL is being applied to P_gal_I by tatt_interface.py")
            bias = p_gm / p_nl

        block.put_grid(
            names.galaxy_intrinsic_power + suffix,
            "z",
            z_lin,
            "k_h",
            k_use,
            "p_k",
            bias * gi_e_total,
        )

    return 0
