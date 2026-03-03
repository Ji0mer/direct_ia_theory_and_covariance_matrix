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

def interp_func(x,y,xnew,axis=0,kind='linear'):
    interp_func = interp1d(x,y,axis=axis,kind=kind,bounds_error=False,fill_value=0)
    y_new = interp_func(xnew)
    return y_new

def setup(options):
    sample = options.get_string(option_section,"sample")
    nz_shape = options.get_string(option_section,"nz_shape")
    nz_shape_all = options.get_string(option_section,"nz_shape_all")
    nz_dens = options.get_string(option_section,"nz_dens")
    nz_dens_all = options.get_string(option_section,"nz_dens_all")
    #number_bins = options.get_int(option_section,"number_bins",default = 4)
    #bins_index = options.get_int(option_section,"bins_index",default=0)
    return sample, nz_shape, nz_dens, nz_shape_all, nz_dens_all


def execute(block, config):
    sample, nz_shape, nz_dens, nz_shape_all, nz_dens_all = config
    
    #slope = 0.0
    #nzintersec = 2.0
    
    #z_shape_target = block["nz_forecast_sample_shape","z"]
    z_shape_target = block["growth_parameters","z"]
    z_shape_synthetic = np.load(nz_shape)["arr_0"]
    nz_shape_synthetic = np.load(nz_shape)["arr_1"]
    
    z_shape_all_synthetic = np.load(nz_shape_all)["arr_0"]
    nz_shape_all_synthetic = np.load(nz_shape_all)["arr_1"]
    
    nz_shape_target = interp_func( z_shape_synthetic,nz_shape_synthetic,z_shape_target )
    nz_shape_all_target = interp_func( z_shape_all_synthetic,nz_shape_all_synthetic,z_shape_target )
    
    block["nz_forecast_sample_shape","raw"] = nz_shape_target
    block["nz_forecast_sample_shape","z"] = z_shape_target
    block["nz_forecast_sample_shape","raw_all"] = nz_shape_all_target
    
    nz_bin_shape_target = nz_shape_target
    #block["nz_forecast_sample_shape","raw_bin_index"] = nz_bin_shape_target
    ##### normalize nz
    nz_bin_shape_target /= np.trapz( nz_shape_all_target,z_shape_target )
    block["nz_forecast_sample_shape","bin_1"] = nz_bin_shape_target
    
    ###
    # test diff W(z) effect
    ###
    #indx = np.where( nz_shape_target == 0 )
    #nz_shape_test = slope*z_shape_target + nzintersec
    #nz_shape_test[indx] = 0
    #block["nz_forecast_sample_shape","raw"] = nz_shape_test * 3e-4
    #nz_shape_test /= np.trapz(nz_shape_test,z_shape_target)
    #block["nz_forecast_sample_shape","bin_1"] = nz_shape_test
    ###
    # end of the test
    ###
    
    
    
    #z_dens_target = block["nz_forecast_sample_density","z"]
    z_dens_target = block["growth_parameters","z"]
    z_dens_synthetic = np.load(nz_dens)["arr_0"]
    nz_dens_synthetic = np.load(nz_dens)["arr_1"]
    
    z_dens_all_synthetic = np.load(nz_dens_all)["arr_0"]
    nz_dens_all_synthetic = np.load(nz_dens_all)["arr_1"]
    
    nz_dens_target = interp_func( z_dens_synthetic,nz_dens_synthetic,z_dens_target )
    nz_dens_all_target = interp_func( z_dens_all_synthetic,nz_dens_all_synthetic,z_dens_target )
    
    block["nz_forecast_sample_density","raw"] = nz_dens_target
    block["nz_forecast_sample_density","z"] = z_dens_target
    block["nz_forecast_sample_density","raw_all"] = nz_dens_all_target
    
    nz_bin_dens_target = nz_dens_target
    #block["nz_forecast_sample_density","raw_bin_index"] = nz_bin_dens_target
    ##### normalize nz
    nz_bin_dens_target /= np.trapz( nz_dens_all_target,z_dens_target )
    block["nz_forecast_sample_density","bin_1"] = nz_bin_dens_target
    
    ###
    # test diff W(z) effect
    ###
    #indx = np.where( nz_dens_target == 0 )
    #nz_dens_test = slope*z_dens_target + nzintersec
    #nz_dens_test[indx] = 0
    #block["nz_forecast_sample_density","raw"] = nz_dens_test * 3e-4
    #nz_dens_test /= np.trapz(nz_dens_test,z_dens_target)
    #block["nz_forecast_sample_density","bin_1"] = nz_dens_test
    ###
    # end of the test
    ###
    
    
    return 0













































