import os
import sys
import time
import numpy as np
from dht_simpson import Compute_covmat
from scipy import integrate
import scipy.integrate as sint
import scipy.interpolate as spi
from scipy.interpolate import interp1d
from cosmosis.datablock import names, option_section
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 #use Planck15 if you can

def interp_func(x,y,xnew,axis=0,kind='linear'):
    interp_func = interp1d(x,y,axis=axis,kind=kind,fill_value="extrapolate")
    y_new = interp_func(xnew)
    return y_new

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

def compute_c1(A1,Dz,z_out,z_piv=0,alpha1=0,Omega_m=0.3):
    C1_RHOCRIT = compute_c1_baseline()
    return -1.0*A1*C1_RHOCRIT*Omega_m/Dz*( (1.0+z_out)/(1.0+z_piv) )**alpha1

def get_pz_from_nz( z,nz,area,cosmo=Planck13.clone(H0=69) ):
    chi = cosmo.comoving_distance( z )
    dchidz = np.gradient( chi,z )
    N = np.trapz( area*chi**2*dchidz*nz,z )
    n2d = N / area
    pz = nz/n2d * chi**2*dchidz
    pz /= np.trapz( pz,z )
    return pz

def _get_covmat_param(block, defaults, name):
    if block.has_value("covmat", name):
        return block["covmat", name]
    return defaults[name]

def setup(options):

    sample = options.get_string(option_section,"sample",default="cmass")
    defaults = {
        "zeff": options.get_double(option_section, "zeff", default=0.52),
        "area_shape": options.get_double(option_section, "area_shape", default=5000.0),
        "area_dens": options.get_double(option_section, "area_dens", default=5000.0),
        "rmin": options.get_double(option_section, "rmin", default=0.1),
        "rmax": options.get_double(option_section, "rmax", default=350.0),
        "nr": options.get_int(option_section, "nr", default=21),
        "sigma_e": options.get_double(option_section, "sigma_e", default=0.25),
        "nbar_shape": options.get_double(option_section, "nbar_shape", default=2e-4),
        "nbar_dens": options.get_double(option_section, "nbar_dens", default=2e-4),
    }
    nk = 10000
    return sample, defaults, nk


def execute(block, config):

    sample, defaults, nk = config
    zeff = _get_covmat_param(block, defaults, "zeff")
    area_shape = _get_covmat_param(block, defaults, "area_shape")
    area_dens = _get_covmat_param(block, defaults, "area_dens")
    rmin = _get_covmat_param(block, defaults, "rmin")
    rmax = _get_covmat_param(block, defaults, "rmax")
    nr = int(_get_covmat_param(block, defaults, "nr"))
    sigma_e = _get_covmat_param(block, defaults, "sigma_e")
    nbar_shape = _get_covmat_param(block, defaults, "nbar_shape")
    nbar_dens = _get_covmat_param(block, defaults, "nbar_dens")
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), nr)
    Np = sigma_e**2 / nbar_shape
    Ng = 1 / nbar_dens
    
    h0 = block["cosmological_parameters","h0"]
    Pimax = block["LOS_bin","Pi_max"]/h0 #Mpc
    
    cosmo=Planck13.clone(H0=h0*100)
    omega_shape = area_shape*(np.pi/180)**2
    omega_dens = area_dens*(np.pi/180)**2
    
    A1 = block["intrinsic_alignment_parameters","A1"]
    b1 = block["bias_%s_density"%sample, "b1E_bin1"]
    b2 = block["bias_%s_density"%sample, "b2E_bin1"]
    
    plin = block['matter_power_lin','p_k']/h0**3
    z = block['matter_power_lin','z']
    kh = block['matter_power_lin','k_h']*h0
    
    pnl = block['matter_power_nl','p_k']/h0**3
    znl = block['matter_power_nl','z']
    khnl = block['matter_power_nl','k_h']*h0
    
    kuse = np.logspace( np.log10(khnl[0]),np.log10(khnl[-1]),nk )
    
    # compute Dz
    # use ind to handle mild scale-dependence in growth
    ind = np.where(kh > 0.03)[0][0]
    Dz = np.sqrt(plin[:, ind] / plin[0, ind])
    Dz_interp = interp1d(z,Dz)
    Dzeff = Dz_interp(zeff)
    C1 = compute_c1(A1,Dzeff,zeff)
    
    ptemp1 = interp_func(znl,pnl,zeff,axis=0)
    ptemp = interp_func(khnl,ptemp1,kuse)
    pgi_nl = b1*C1*ptemp
    pii_nl = C1**2*ptemp
    pgg_nl = b1**2*ptemp
    
    # load n(z)
    zuse = np.linspace(1e-5,4,401)
    zs = block['nz_'+sample+"_shape", 'z']
    nzs = block['nz_'+sample+"_shape", 'raw']
    zd = block['nz_'+sample+"_density", 'z']
    nzd = block['nz_'+sample+"_density", 'raw']
    
    #nzs
    nzs_interp = interp1d( zs,nzs,bounds_error=False,fill_value=0 )
    nzs = nzs_interp( zuse )
    nzs = get_pz_from_nz( zuse,nzs,omega_shape,cosmo )
    # nzd
    nzd_interp = interp1d( zd,nzd,bounds_error=False,fill_value=0 )
    nzd = nzd_interp( zuse )
    nzd = get_pz_from_nz( zuse,nzd,omega_dens,cosmo )
    # survey index where nz > 0
    survey_index = np.where( nzd > 0 )[0]
    print(survey_index)
    
    # compute w_dd, w_ds, w_ss
    z_chi = block["distances","z"]
    chi = block["distances","d_m"]
    chi_interp = interp1d(z_chi,chi)
    Chi = chi_interp(zuse)
    
    dchidz = dxdz = np.gradient(Chi,zuse[1]-zuse[0])
    W_ds = nzs * nzd /Chi/Chi/dchidz
    W_ds /= np.trapz( W_ds,zuse )
    W_dd = nzd * nzd /Chi/Chi/dchidz
    W_dd /= np.trapz( W_dd,zuse )
    W_ss = nzs * nzs /Chi/Chi/dchidz
    W_ss /= np.trapz( W_ss,zuse )
    
    pf_dsds = 1/omega_shape * np.trapz( W_ds[survey_index]**2*(chi[survey_index]**2*dchidz[survey_index])**-1,zuse[survey_index] ) * 2*Pimax
    pf_dsss = 1/omega_shape * np.trapz( W_ds[survey_index]*W_ss[survey_index]*(chi[survey_index]**2*dchidz[survey_index])**-1,zuse[survey_index] ) * 2*Pimax
    pf_dsdd = 1/omega_shape * np.trapz( W_ds[survey_index]*W_dd[survey_index]*(chi[survey_index]**2*dchidz[survey_index])**-1,zuse[survey_index] ) * 2*Pimax
    
    pf_ssds = 1/omega_shape * np.trapz( W_ss[survey_index]*W_ds[survey_index]*(chi[survey_index]**2*dchidz[survey_index])**-1,zuse[survey_index] ) * 2*Pimax
    pf_ssss = 1/omega_shape * np.trapz( W_ss[survey_index]**2*(chi[survey_index]**2*dchidz[survey_index])**-1,zuse[survey_index] ) * 2*Pimax
    pf_ssdd = 1/omega_shape * np.trapz( W_ss[survey_index]*W_dd[survey_index]*(chi[survey_index]**2*dchidz[survey_index])**-1,zuse[survey_index] ) * 2*Pimax
    
    pf_ddds = 1/omega_shape * np.trapz( W_dd[survey_index]*W_ds[survey_index]*(chi[survey_index]**2*dchidz[survey_index])**-1,zuse[survey_index] ) * 2*Pimax
    pf_ddss = 1/omega_shape * np.trapz( W_dd[survey_index]*W_ss[survey_index]*(chi[survey_index]**2*dchidz[survey_index])**-1,zuse[survey_index] ) * 2*Pimax
    pf_dddd = 1/omega_dens * np.trapz( W_dd[survey_index]**2*(chi[survey_index]**2*dchidz[survey_index])**-1,zuse[survey_index] ) * 2*Pimax

    
    cc = Compute_covmat(rbins, 1e-3, kuse, nv=[0,2,[0,4]], load_data=True, quad_limits=15000)
    #cc.save_jn_data()
    cov_gpgp = cc.covariance_wgpwgp(pgg_nl,pii_nl,pgi_nl,Ng,Np)
    cov_gpgp *= pf_dsds
    cov_gppp = cc.covariance_wgpwpp(pgg_nl,pii_nl,pgi_nl,Ng,Np)
    cov_gppp *= pf_dsss
    cov_gpgg = cc.covariance_wgpwgg(pgg_nl,pii_nl,pgi_nl,Ng,Np)
    cov_gpgg *= pf_dsdd
    
    cov_gggp = cc.covariance_wggwgp(pgg_nl,pii_nl,pgi_nl,Ng,Np)
    cov_gggp *= pf_ddds
    cov_ggpp = cc.covariance_wggwpp(pgg_nl,pii_nl,pgi_nl,Ng,Np)
    cov_ggpp *= pf_ddss
    cov_gggg = cc.covariance_wggwgg(pgg_nl,Ng)
    cov_gggg *= pf_dddd
    
    cov_ppgp = cc.covariance_wppwgp(pgg_nl,pii_nl,pgi_nl,Ng,Np)
    cov_ppgp *= pf_ssds
    cov_pppp = cc.covariance_wppwpp(pii_nl,Np)
    cov_pppp *= pf_ssss
    cov_ppgg = cc.covariance_wppwgg(pgg_nl,pii_nl,pgi_nl,Ng,Np)
    cov_ppgg *= pf_ssdd

    clen = len(cc.rp[0])
    Cov = np.zeros( (3*clen,3*clen) )
    
    Cov[0*clen:1*clen,0*clen:1*clen] += cov_gpgp
    Cov[0*clen:1*clen,1*clen:2*clen] += cov_gppp
    Cov[0*clen:1*clen,2*clen:3*clen] += cov_gpgg
    
    Cov[1*clen:2*clen,0*clen:1*clen] += cov_ppgp
    Cov[1*clen:2*clen,1*clen:2*clen] += cov_pppp
    Cov[1*clen:2*clen,2*clen:3*clen] += cov_ppgg
    
    Cov[2*clen:3*clen,0*clen:1*clen] += cov_gggp
    Cov[2*clen:3*clen,1*clen:2*clen] += cov_ggpp
    Cov[2*clen:3*clen,2*clen:3*clen] += cov_gggg

    
    #Cov /= 1.256
    #D = np.sqrt( np.diag(Cov) )
    #a,b= np.meshgrid(D,D)
    #Cov_plot = Cov/a/b
    
    #fig = plt.figure( figsize=(16,6) )
    #plt.subplot(1,2,1)
    #plt.title("Covariance Matrix",fontsize=18)
    #plt.imshow(Cov,aspect=1)
    #plt.colorbar()
    #plt.xlabel(r"$W_{g+}\,-\,W_{++}\,-\,W_{gg}$",fontsize=20)
    #plt.ylabel(r"$W_{gg}\,-\,W_{++}\,-\,W_{g+}$",fontsize=20)
    
    #plt.subplot(1,2,2)
    #plt.title("Correlation Matrix",fontsize=18)
    #plt.imshow(Cov_plot,aspect=1)
    #plt.colorbar()
    #plt.xlabel(r"$W_{g+}\,-\,W_{++}\,-\,W_{gg}$",fontsize=20)
    #plt.ylabel(r"$W_{gg}\,-\,W_{++}\,-\,W_{g+}$",fontsize=20)
    #plt.show()
    #plt.savefig("./cov_cor.png")
    
    block["covmat","Cov"] = Cov*h0**2
    block["covmat","rp0"] = cc.rp[0]*h0
    block["covmat","rp2"] = cc.rp[2]*h0
    block["covmat","rp04"] = cc.rp["[0, 4]"]*h0

    return 0













































