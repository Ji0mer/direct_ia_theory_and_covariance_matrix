# -*- coding: utf-8 -*-
import sys
import os

# from ia_lib import tatt, tidal_alignment, tidal_torque, del4
from des_ia_lib import del4
from des_ia_lib.common import resample_power
import numpy as np

# We now return you to your module.


from cosmosis.datablock import names, option_section
import scipy.interpolate as interp


def compute_c1_baseline():
    C1_M_sun = 5e-14  # h^-2 M_S^-1 Mpc^3
    M_sun = 1.9891e30  # kg
    Mpc_in_m = 3.0857e22  # meters
    C1_SI = C1_M_sun / M_sun * (Mpc_in_m) ** 3  # h^-2 kg^-1 m^3
    # rho_crit_0 = 3 H^2 / 8 pi G
    G = 6.67384e-11  # m^3 kg^-1 s^-2
    H = 100  #  h km s^-1 Mpc^-1
    H_SI = H * 1000.0 / Mpc_in_m  # h s^-1
    rho_crit_0 = 3 * H_SI ** 2 / (8 * np.pi * G)  #  h^2 kg m^-3
    f = C1_SI * rho_crit_0
    return f

def amp_3d(C, num_z, num_k):
    C = np.atleast_1d(C)

    if C.shape == (1,):
        return C * np.ones((num_z, num_k))
    elif C.shape == (num_z,):
        return np.outer(C, np.ones(num_k))
    else:
        assert C.shape == (num_z, num_k)
        return C
        

def setup(options):
    sub_lowk = options.get_bool(option_section, "sub_lowk", False)
    ia_model = options.get_string(option_section, "ia_model", "nla")
    name = options.get_string(option_section, "name", default="").lower()
    do_galaxy_intrinsic = options.get_bool(option_section, "do_galaxy_intrinsic", False)
    no_IA_E = options.get_bool(option_section, "no_IA_E", False)
    no_IA_B = options.get_bool(option_section, "no_IA_B", False)

    if name:
        suffix = "_" + name
    else:
        suffix = ""
    return (
        sub_lowk,
        ia_model,
        suffix,
        do_galaxy_intrinsic,
        no_IA_E,
        no_IA_B,
    )


def execute(block, config):
    (
        sub_lowk,
        ia_model,
        suffix,
        do_galaxy_intrinsic,
        no_IA_E,
        no_IA_B,
    ) = config

    # Load linear and non-linear matter power spectra
    lin = names.matter_power_lin
    nl = names.matter_power_nl
    cosmo = names.cosmological_parameters
    Omega_m = block[cosmo, "omega_m"]

    # Load the matter power spectra
    z_lin, k_lin, p_lin = block.get_grid(lin, "z", "k_h", "p_k")
    z_nl, k_nl, p_nl = block.get_grid(nl, "z", "k_h", "p_k")

    # use ind to handle mild scale-dependence in growth
    ind = np.where(k_lin > 0.03)[0][0]
    Dz = np.sqrt(p_lin[:, ind] / p_lin[0, ind])

    # Re-sample nonlinear power onto same grid as linear
    assert (
        z_nl == z_lin
    ).all(), "Expected identical z values for matter power NL & Linear in IA code"

    # pre-factors to turn off E and B modes
    E_factor = 0 if no_IA_E else 1
    B_factor = 0 if no_IA_B else 1

    ia_section = "intrinsic_alignment_parameters"

    # check for deprecated parameters
    if (ia_section, "C1") in block:
        raise ValueError("Deprecated TATT parameter specified: " + "C1")

    if (ia_section, "C2") in block:
        raise ValueError("Deprecated TATT parameter specified: " + "C2")

    # Get main parameters - note that all are optional.
    A1 = block.get_double(ia_section, "A1", 1.0)
    A2 = block.get_double(ia_section, "A2", 1.0)
    alpha1 = block.get_double(ia_section, "alpha1", 0.0)
    alpha2 = block.get_double(ia_section, "alpha2", 0.0)
    alphadel = block.get_double(ia_section, "alphadel", alpha1)
    z_piv = block.get_double(ia_section, "z_piv", 0.0)


    if (ia_section, "Adel") in block:
        if (ia_section, "bias_ta") in block:
            raise ValueError("bias_ta is not used when Adel is specified.")
        else:
            Adel = block.get_double(ia_section, "Adel", 1.0)
            bias_ta = bias_tt = 1.0
    else:
        bias_ta = block.get_double(ia_section, "bias_ta", 1.0)
        bias_tt = block.get_double(ia_section, "bias_tt", 1.0)
        Adel = bias_ta * A1

    
    # compute intrinsic alignment parameters
    
    z_out = z_lin
    
    C1_RHOCRIT = compute_c1_baseline()

    C1 = (
        -1.0
        * A1
        * C1_RHOCRIT
        * Omega_m
        / Dz
        * ((1.0 + z_out) / (1.0 + z_piv)) ** alpha1
    )
    Cdel = (
        -1.0
        * Adel
        * C1_RHOCRIT
        * Omega_m
        / Dz
        * ((1.0 + z_out) / (1.0 + z_piv)) ** alphadel
    )
    C2 = (
        5
        * A2
        * C1_RHOCRIT
        * Omega_m
        / Dz ** 2
        * ((1.0 + z_out) / (1.0 + z_piv)) ** alpha2
    )
    
    k_use = k_nl  # this sets which k grid will be used. Changing this may break some things.
    
    # make all the amplitude terms (num_z, num_k).
    # This should be compatible with the z-dependent values passed from above.
    # C1, C2, bias_ta, bias_tt = amp_3d(C1, len(Dz), len(k_out)), amp_3d(C2, len(Dz), len(k_out)), amp_3d(bias_ta, len(Dz), len(k_out)), amp_3d(bias_tt, len(Dz), len(k_out))
    C1 = amp_3d(C1, len(Dz), len(k_use))
    Cdel = amp_3d(Cdel, len(Dz), len(k_use))
    C2 = amp_3d(C2, len(Dz), len(k_use))
    
    if ia_model == "nla":
        
        gi_e_total = E_factor * C1 * p_nl
        ii_ee_total = E_factor * C1 * C1 *p_nl
        
        block.put( "gI_power","ia_model",[0] )
        block.put( "gI_power","z",z_lin )
        block.put( "gI_power","k_h",k_use )
        block.put( "gI_power","c1",C1 )
        block.put( "gI_power","p_k",p_nl )
        
        block.put( "II_power","ia_model",[0] )
        block.put( "II_power","z",z_lin )
        block.put( "II_power","k_h",k_use )
        block.put( "II_power","c1",C1 )
        block.put( "II_power","p_k",p_nl )
        
    elif ia_model == "tatt":
        
        # get from the block
        P_IA_dict = {}
        for key in ["P_tt_EE","P_tt_BB","P_ta_dE1","P_ta_dE2","P_ta_EE","P_ta_BB","P_mix_A","P_mix_B","P_mix_D_EE","P_mix_D_BB","Plin"]:
            z, k_IA, p = block.get_grid("fastpt", "z", "k_h", key)
            """
            if sub_lowk and key in ["P_tt_EE","P_tt_BB","P_ta_EE","P_ta_BB","P_mix_D_EE","P_mix_D_BB"]:
                to_sub = p[:, 0][:, np.newaxis]
                p -= to_sub
                p[:, 0] = p[:, 1]
            """
            
            assert np.allclose(z, z_out)
        
            try:
                assert np.allclose(k_use, k_IA)
                P_IA_dict[key] = p
            except (AssertionError, ValueError):
                # interpolate and re-apply correct growth factor if necessary
                p0_orig = p[0]
                p0_out = Pk_interp(k_IA, p0_orig)(k_use)
                if key == "Plin":
                    P_IA_dict[key] = grow(p0_out, Dz, 2)
                else:
                    P_IA_dict[key] = grow(p0_out, Dz, 4)
        
        # galaxy intrinsic power
        nla_GI = C1 * p_nl
        ta_GI = Cdel * (P_IA_dict["P_ta_dE1"] + P_IA_dict["P_ta_dE2"])
        tt_GI = C2 * (P_IA_dict["P_mix_A"] + P_IA_dict["P_mix_B"])
        
        gi_e_total = E_factor * ( nla_GI + ta_GI + tt_GI )
        
        block.put( "gI_power","ia_model",[1] )
        block.put( "gI_power","z",z_lin )
        block.put( "gI_power","k_h",k_use )
        block.put( "gI_power","c1",C1 )
        block.put( "gI_power","cdel",Cdel )
        block.put( "gI_power","c2",C2 )
        block.put( "gI_power","p_k",p_nl )
        block.put( "gI_power","p_ta_gi",P_IA_dict["P_ta_dE1"] + P_IA_dict["P_ta_dE2"] )
        block.put( "gI_power","p_tt_gi",P_IA_dict["P_mix_A"] + P_IA_dict["P_mix_B"] )
        
        # intrinsic power
        nla_II_EE = C1 * C1 * p_nl
        ta_II_EE = Cdel ** 2 * P_IA_dict["P_ta_EE"] + C1 * Cdel * ( 2 * P_IA_dict["P_ta_dE1"] + 2 * P_IA_dict["P_ta_dE2"] )
        tt_II_EE = C2 * C2 * P_IA_dict["P_tt_EE"]
        mix_II_EE = 2.0 * C2 * ( C1 * P_IA_dict["P_mix_A"] + C1 * P_IA_dict["P_mix_B"] + Cdel * P_IA_dict["P_mix_D_EE"] )
        
        ii_ee_total = E_factor * ( nla_II_EE + ta_II_EE + tt_II_EE + mix_II_EE )
        
        block.put( "II_power","ia_model",[1] )
        block.put( "II_power","z",z_lin )
        block.put( "II_power","k_h",k_use )
        block.put( "II_power","c1",C1 )
        block.put( "II_power","cdel",Cdel )
        block.put( "II_power","c2",C2 )
        block.put( "II_power","p_k",p_nl )
        block.put( "II_power","p_ta_ee",P_IA_dict["P_ta_EE"] )
        block.put( "II_power","2p_ta_de",2 * (P_IA_dict["P_ta_dE1"] + P_IA_dict["P_ta_dE2"]) )
        block.put( "II_power","p_tt_ee",P_IA_dict["P_tt_EE"] )
        block.put( "II_power","p_mix_a",P_IA_dict["P_mix_A"] )
        block.put( "II_power","p_mix_b",P_IA_dict["P_mix_B"] )
        block.put( "II_power","p_mix_d_ee",P_IA_dict["P_mix_D_EE"] )
        
    else:
        print("This model not included your input ia model...")
        
    
    # We also save the EE total to intrinsic power for consistency with other modules
    block.put_grid( names.intrinsic_power + suffix, "z", z_lin, "k_h", k_use, "p_k", ii_ee_total )

    # If we've been told to include galaxy-intrinsic power then we
    # need to check if the galaxy bias has already been applied to
    # it or not. We'd prefer people not do that, apparently, so
    # we print out some stuff if it happens.
    if do_galaxy_intrinsic:
        gm = "matter_galaxy_power" + suffix
        z, k, p_gm = block.get_grid(gm, "z", "k_h", "p_k")

        # Check that the bias has not already been applied
        if p_gm.shape == p_nl.shape and np.allclose(p_gm, p_nl):
            b_temp = 1
        else:
            print("WARNING: bias has already been applied to P_gm.")
            print("b_temp=P_gm/P_NL is being applied to P_gal_I by tatt_interface.py")
            b_temp = p_gm / p_nl


        gal_i_total = b_temp * gi_e_total
        block.put_grid( names.galaxy_intrinsic_power + suffix, "z", z_lin, "k_h", k_use, "p_k",gal_i_total )
        
        block.put( "gI_power","b_temp",b_temp )
    return 0







































































