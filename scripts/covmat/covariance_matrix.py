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

def setup(options):

    zeff = options.get_double(option_section, "zeff",default=0.52)
    area_shape = options.get_double(option_section, "area_shape", default=5000.0)
    area_dens = options.get_double(option_section, "area_dens", default=5000.0)
    sample = options.get_string(option_section,"sample",default="cmass")
    rmin = options.get_double(option_section,"rmin",default=0.1)
    rmax = options.get_double(option_section,"rmax",default=350.0)
    nr = options.get_int(option_section,"nr",default=21)
    rbins = np.logspace( np.log10(rmin),np.log10(rmax),nr )
    nk = 10000
    sigma_e = options.get_double(option_section,"sigma_e",default=0.25)
    nbar_shape = options.get_double(option_section,"nbar_shape",default=2e-4)
    Np = sigma_e**2/nbar_shape
    nbar_dens = options.get_double(option_section,"nbar_dens",default=2e-4)
    Ng = 1/nbar_dens
    return zeff,area_shape,area_dens,sample,rbins,nk,Ng,Np


def execute(block, config):

    zeff, area_shape, area_dens, sample, rbins, nk, Ng, Np = config
    
    Pimax = block["LOS_bin","Pi_max"] #Mpc/h
    h0 = block["cosmological_parameters","h0"]
    
    cosmo=Planck13.clone(H0=h0*100)
    area_comoving=area_shape*(np.pi/180)**2*cosmo.comoving_distance(z=zeff)**2
    area_shape = area_comoving.value*h0**2
    area_comoving=area_dens*(np.pi/180)**2*cosmo.comoving_distance(z=zeff)**2
    area_dens = area_comoving.value*h0**2
    
    A1 = block["intrinsic_alignment_parameters","A1"]
    b1 = block["bias_%s_density"%sample, "b1E_bin1"]
    b2 = block["bias_%s_density"%sample, "b2E_bin1"]
    
    plin = block['matter_power_lin','p_k']
    z = block['matter_power_lin','z']
    kh = block['matter_power_lin','k_h']
    kuse = np.logspace( np.log10(kh[0]),np.log10(kh[-1]),nk )
    
    # compute Dz
    # use ind to handle mild scale-dependence in growth
    ind = np.where(kh > 0.03)[0][0]
    Dz = np.sqrt(plin[:, ind] / plin[0, ind])
    Dz_interp = interp1d(z,Dz)
    Dzeff = Dz_interp(zeff)
    C1 = compute_c1(A1,Dzeff,zeff)
    
    ptemp1 = interp_func(z,plin,zeff,axis=0)
    ptemp = interp_func(kh,ptemp1,kuse)
    pgi_lin = b1*C1*ptemp
    pii_lin = C1**2*ptemp
    pgg_lin = b1**2*ptemp
    
    # load n(z)
    zuse = np.linspace(1e-5,4,401)
    zs = block['nz_'+sample+"_shape", 'z']
    nzs = block['nz_'+sample+"_shape", 'raw_all']
    zd = block['nz_'+sample+"_density", 'z']
    nzd = block['nz_'+sample+"_density", 'raw_all']
    
    zmin = zd[np.where(nzd > 0)[0][0]]
    zmax = zd[np.where(nzd > 0)[0][-1]]
    Lw = cosmo.comoving_distance(z=zmax) - cosmo.comoving_distance(z=zmin)
    Lw = Lw.value*h0
    print("\n\n")
    print(Lw)
    print("\n\n")
    
    #nzs
    nzs_interp = interp1d( zs,nzs,bounds_error=False,fill_value=0 )
    nzs = nzs_interp( zuse )
    # nzd
    nzd_interp = interp1d( zd,nzd,bounds_error=False,fill_value=0 )
    nzd = nzd_interp( zuse )
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
    
    wdswds = np.trapz( W_ds**2,zuse )
    wdswss = np.trapz( W_ds*W_ss,zuse )
    wdswdd = np.trapz( W_ds*W_dd,zuse )
    
    wsswds = np.trapz( W_ss*W_ds,zuse )
    wsswss = np.trapz( W_ss**2,zuse )
    wsswdd = np.trapz( W_ss*W_dd,zuse )
    
    wddwds = np.trapz( W_dd*W_ds,zuse )
    wddwss = np.trapz( W_dd*W_ss,zuse )
    wddwdd = np.trapz( W_dd**2,zuse )
    
    
    cc = Compute_covmat(rbins,1e-3,kuse,nv=[0,2,[0,4]],load_data = True)
    #cc.save_jn_data()
    cov_gpgp = cc.covariance_wgpwgp(pgg_lin,pii_lin,pgi_lin,Ng,Np)
    cov_gpgp /= area_shape
    cov_gpgp *= Pimax/Lw * wdswds
    cov_gppp = cc.covariance_wgpwpp(pgg_lin,pii_lin,pgi_lin,Ng,Np)
    cov_gppp /= area_shape
    cov_gppp *= Pimax/Lw * wdswss
    cov_gpgg = cc.covariance_wgpwgg(pgg_lin,pii_lin,pgi_lin,Ng,Np)
    cov_gpgg /= area_shape
    cov_gpgg *= Pimax/Lw * wdswdd
    
    cov_gggp = cc.covariance_wggwgp(pgg_lin,pii_lin,pgi_lin,Ng,Np)
    cov_gggp /= area_shape
    cov_gggp *= Pimax/Lw * wddwds
    cov_ggpp = cc.covariance_wggwpp(pgg_lin,pii_lin,pgi_lin,Ng,Np)
    cov_ggpp /= area_shape
    cov_ggpp *= Pimax/Lw * wddwss
    cov_gggg = cc.covariance_wggwgg(pgg_lin,Ng)
    cov_gggg /= area_dens
    cov_gggg *= Pimax/Lw * wddwdd
    
    cov_ppgp = cc.covariance_wppwgp(pgg_lin,pii_lin,pgi_lin,Ng,Np)
    cov_ppgp /= area_shape
    cov_ppgp *= Pimax/Lw * wsswds
    cov_pppp = cc.covariance_wppwpp(pii_lin,Np)
    cov_pppp /= area_shape
    cov_pppp *= Pimax/Lw * wsswss
    cov_ppgg = cc.covariance_wppwgg(pgg_lin,pii_lin,pgi_lin,Ng,Np)
    cov_ppgg /= area_shape
    cov_ppgg *= Pimax/Lw * wsswdd

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
    D = np.sqrt( np.diag(Cov) )
    a,b= np.meshgrid(D,D)
    Cov_plot = Cov/a/b
    
    fig = plt.figure( figsize=(16,6) )
    plt.subplot(1,2,1)
    plt.title("Covariance Matrix",fontsize=18)
    plt.imshow(Cov,aspect=1)
    plt.colorbar()
    plt.xlabel(r"$W_{g+}\,-\,W_{++}\,-\,W_{gg}$",fontsize=20)
    plt.ylabel(r"$W_{gg}\,-\,W_{++}\,-\,W_{g+}$",fontsize=20)
    
    plt.subplot(1,2,2)
    plt.title("Correlation Matrix",fontsize=18)
    plt.imshow(Cov_plot,aspect=1)
    plt.colorbar()
    plt.xlabel(r"$W_{g+}\,-\,W_{++}\,-\,W_{gg}$",fontsize=20)
    plt.ylabel(r"$W_{gg}\,-\,W_{++}\,-\,W_{g+}$",fontsize=20)
    #plt.show()
    #plt.savefig("./cov_cor.png")
    
    block["covmat","Cov"] = Cov
    block["covmat","rp0"] = cc.rp[0]
    block["covmat","rp2"] = cc.rp[2]
    block["covmat","rp04"] = cc.rp["[0, 4]"]

    return 0













































