import os
import sys
import time
import numpy as np
from scipy import integrate
import scipy.integrate as sint
import scipy.interpolate as spi
from scipy.interpolate import interp1d
from cosmosis.datablock import names, option_section
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15

def interp_func(x,y,xnew,axis=0,kind='linear'):
    interp_func = interp1d(x,y,axis=axis,kind=kind,bounds_error=False,fill_value=0)
    y_new = interp_func(xnew)
    return y_new

def get_pz_from_nz(z, nz, area, cosmo=Planck15.clone(H0=69)):
    z = np.asarray(z, dtype=float)
    nz = np.asarray(nz, dtype=float)
    chi = cosmo.comoving_distance(z).value
    dchidz = np.gradient(chi, z)
    shell_counts = area * chi**2 * dchidz * nz
    n2d = np.trapz(shell_counts, z) / area
    if (not np.isfinite(n2d)) or (n2d <= 0.0):
        return np.zeros_like(z, dtype=float)
    pz = nz / n2d * chi**2 * dchidz
    norm = np.trapz(pz, z)
    if (not np.isfinite(norm)) or (norm <= 0.0):
        return np.zeros_like(z, dtype=float)
    return pz / norm

def setup(options):
    sample = options.get_string(option_section,"sample")
    nz_shape = options.get_string(option_section,"nz_shape")
    nz_dens = options.get_string(option_section,"nz_dens")
    #number_bins = options.get_int(option_section,"number_bins",default = 4)
    #bins_index = options.get_int(option_section,"bins_index",default=0)
    return sample, nz_shape, nz_dens


def execute(block, config):
    sample, nz_shape, nz_dens = config
    shape_section = "nz_%s_shape" % sample
    density_section = "nz_%s_density" % sample
    
    z_shape_target = block["growth_parameters","z"]
    z_shape_synthetic = np.load(nz_shape)["arr_0"]
    nz_shape_synthetic = np.load(nz_shape)["arr_1"]
    nz_shape_target = interp_func( z_shape_synthetic,nz_shape_synthetic,z_shape_target )
    pz_shape_target = get_pz_from_nz(z_shape_target,nz_shape_target,1.0)
    block[shape_section,"raw"] = nz_shape_target
    block[shape_section,"z"] = z_shape_target
    pz_bin_shape_target = pz_shape_target
    pz_bin_shape_target /= np.trapz( pz_shape_target,z_shape_target )
    block[shape_section,"bin_1"] = pz_bin_shape_target
    

    z_dens_target = block["growth_parameters","z"]
    z_dens_synthetic = np.load(nz_dens)["arr_0"]
    nz_dens_synthetic = np.load(nz_dens)["arr_1"]
    nz_dens_target = interp_func( z_dens_synthetic,nz_dens_synthetic,z_dens_target )
    pz_dens_target = get_pz_from_nz(z_dens_target,nz_dens_target,1.0)
    block[density_section,"raw"] = nz_dens_target
    block[density_section,"z"] = z_dens_target
    pz_bin_dens_target = pz_dens_target
    pz_bin_dens_target /= np.trapz( pz_dens_target,z_dens_target )
    block[density_section,"bin_1"] = pz_bin_dens_target
    
    
    return 0












































