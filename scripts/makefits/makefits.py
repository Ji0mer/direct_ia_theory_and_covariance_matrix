import os
import sys
import time
import numpy as np
import fitsio as fi
from scipy import integrate
from scipy.interpolate import interp1d
from cosmosis.datablock import names, option_section
import matplotlib.pyplot as plt

def interp_func(x,y,xnew,axis=-1,kind='linear'):
    interp_func = interp1d(x,y,axis=axis,kind=kind,bounds_error=False,fill_value="extrapolate")
    y_new = interp_func(xnew)
    return y_new

def setup(options):
    fits_name = options.get_string(option_section,"save_fits")
    sample = options.get_string(option_section,"sample")
    survey = options.get_string(option_section,"survey")
    
    return fits_name,sample,survey


def execute(block, config):
    fits_name, sample, survey = config
    
    #############################################################################################
    # load data
    Cov = block["covmat","Cov"]
    rp0 = block["covmat","rp0"]
    rp2 = block["covmat","rp2"]
    rp04 = block["covmat","rp04"]
    
    wgp = block["galaxy_intrinsic_w","w_rp_1_1_%s_density_%s_shape"%(sample,sample)]
    wgp_rp = block["galaxy_intrinsic_w","r_p"]
    
    wpp = block["intrinsic_w","w_rp_1_1_%s_shape_%s_shape"%(sample,sample)]
    wpp_rp = block["intrinsic_w","r_p"]
    
    wgg = block["galaxy_w","w_rp_1_1_%s_density_%s_density"%(sample,sample)]
    wgg_rp = block["galaxy_w","r_p"]
    
    wgp_fits = interp_func(wgp_rp,wgp,rp2)
    wpp_fits = interp_func(wpp_rp,wpp,rp04)
    wgg_fits = interp_func(wgg_rp,wgg,rp0)
    
    z_shape = block["nz_%s_shape"%sample,"z"]
    nz_shape = block["nz_%s_shape"%sample,"bin_1"]
    
    z_dens = block["nz_%s_density"%sample,"z"]
    nz_dens = block["nz_%s_density"%sample,"bin_1"]
    
    # work out the edges of the redshift histogram bins
    dz_shape = z_shape[1] - z_shape[0]
    z_shape_low = z_shape - dz_shape/2
    z_shape_high = z_shape + dz_shape/2
    
    dz_dens = z_dens[1] - z_dens[0]
    z_dens_low = z_dens - dz_dens/2
    z_dens_high = z_dens + dz_dens/2
    
    #############################################################################################
    # make fits
    # remove the same output file if it exists
    os.system('rm %s'%fits_name)
    fits = fi.FITS(fits_name,'rw')
    
    # store the shape data in the FITS file
    out_dict = {}
    out_dict['Z_MID'] = z_shape
    out_dict['Z_LOW'] = z_shape_low
    out_dict['Z_HIGH'] = z_shape_high
    out_dict['BIN1'] = nz_shape
    
    fits.write(out_dict)
    fits[-1].write_key('EXTNAME','nz_%s_shape'%survey)
    
    # store the density in the FITS file
    out_dict = {}
    out_dict['Z_MID'] = z_dens
    out_dict['Z_LOW'] = z_dens_low
    out_dict['Z_HIGH'] = z_dens_high
    out_dict['BIN1'] = nz_dens
    
    fits.write(out_dict)
    fits[-1].write_key('EXTNAME','nz_%s_density'%survey)
    
    # index?
    nbins = int(len(rp0))
    sample_index_s = np.ones(nbins)
    sample_index_d = np.zeros(nbins)
    bin_index = np.ones(nbins)
    sep_bin_index = np.linspace(0,nbins-1,nbins).astype(int)
    
    # store the data arrays for w_g+
    out_dict = {}
    out_dict['SEP'] = rp2
    out_dict['SEPBIN'] = sep_bin_index
    out_dict['VALUE'] = wgp_fits
    out_dict['BIN1'] = bin_index
    out_dict['BIN2'] = bin_index
    out_dict['SAMPLE1'] = sample_index_s
    out_dict['SAMPLE2'] = sample_index_d
    
    fits.write(out_dict)
    
    # also include some metadata
    # in principle we could add more details by adding extra lines below
    fits[-1].write_key('EXTNAME','wgp')
    fits[-1].write_key('SAMPLE_0','%s_density'%survey)
    fits[-1].write_key('SAMPLE_1','%s_shape'%survey)
    fits[-1].write_key('SEP_UNITS','Mpc_h')
    
    # same thing for w_++
    out_dict = {}
    out_dict['SEP'] = rp04
    out_dict['SEPBIN'] = sep_bin_index
    out_dict['VALUE'] = wpp_fits
    out_dict['BIN1'] = bin_index
    out_dict['BIN2'] = bin_index
    out_dict['SAMPLE1'] = sample_index_s
    out_dict['SAMPLE2'] = sample_index_s
    
    fits.write(out_dict)
    fits[-1].write_key('EXTNAME','wpp')
    fits[-1].write_key('SAMPLE_1','%s_shape'%survey)
    fits[-1].write_key('SEP_UNITS','Mpc_h')
    
    # and for w_gg
    out_dict = {}
    out_dict['SEP'] = rp0
    out_dict['SEPBIN'] = sep_bin_index
    out_dict['VALUE'] = wgg_fits
    out_dict['BIN1'] = bin_index
    out_dict['BIN2'] = bin_index
    out_dict['SAMPLE1'] = sample_index_d
    out_dict['SAMPLE2'] = sample_index_d
    
    fits.write(out_dict)
    fits[-1].write_key('EXTNAME','wgg')
    fits[-1].write_key('SAMPLE_0','%s_density'%survey)
    fits[-1].write_key('SEP_UNITS','Mpc_h')
    
    # finally write the covariance matrix
    fits.write(Cov)
    fits[-1].write_key('EXTNAME','COVMAT')
    
    fits[-1].write_key('STRT_0',0)
    fits[-1].write_key('STRT_1',nbins) 
    fits[-1].write_key('STRT_2',nbins*2) 
    fits[-1].write_key('NAME_0', 'wgp')
    fits[-1].write_key('NAME_1', 'wpp') 
    fits[-1].write_key('NAME_2', 'wgg')
    
    fits.close()
    
    return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    