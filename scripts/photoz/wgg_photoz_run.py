from __future__ import print_function
from cosmosis.datablock import names, option_section
import os
import sys
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import quad
from hankl import FFTLog

# constants
clight = 299792.4580 # kms^-1

# 检查文件夹是否存在文件
def folder_has_files(folder_path):
    # 列出文件夹中的内容
    return any(os.path.isfile(os.path.join(folder_path, f)) for f in os.listdir(folder_path))

def pk_terms_sum(bv1,bv2,pk1,pk2,pk3,pk4,pk5,pk6,pk7,pk8):
    return ( bv1[0]*bv2[0]*pk1+0.5*(bv1[0]*bv2[1]+bv2[0]*bv1[1])*pk2+
                0.25*bv1[1]*bv2[1]*pk3+0.5*(bv1[0]*bv2[2]+bv2[0]*bv1[2])*pk4+
                0.25*(bv2[1]*bv1[2]+bv1[1]*bv2[2])*pk5+0.25*bv1[2]*bv2[2]*pk6+
                0.5*(bv1[0]*bv2[3]+bv2[0]*bv1[3])*pk7+(bv1[0]*bv2[4]+bv2[0]*bv1[4])*pk8 )

def interp_power(input_k,input_z,input_power,knew,znew):
    mini_power = np.min(input_power)
    modified_power = input_power - mini_power + 10
    inter = interp2d( np.log10(input_k), input_z, np.log10(modified_power) )
    Pnew = inter( np.log10(knew),znew )
    Pnew = 10**(Pnew) - 10 + mini_power
    return Pnew


def setup(options):
    
    sample_a = options.get_string(option_section, "sample_a", default="forecast_sample_density")
    sample_b = options.get_string(option_section, "sample_b", default="forecast_sample_density")
    timing = options.get_bool(option_section, "timing", default=True)
    constant_sigmaz = options.get_bool(option_section, "constant_sigmaz", default=True)
    pks_folder = options.get_string(option_section,"pks_folder")
    w_folder = options.get_string(option_section,"wgg_folder")
    os.makedirs(w_folder, exist_ok=True)
    
    return sample_a,sample_b,timing,constant_sigmaz,pks_folder,w_folder
    
def execute(block,config):
    sample_a,sample_b,timing,constant_sigmaz,pks_path,w_path = config
    
    # set LOS bins
    Npi = block['photoz_errors','N_pi']
    Nz = 200
    Pi = np.linspace(-block['photoz_errors','Pi_max'],block['photoz_errors','Pi_max'],Npi)
    z_low = np.linspace(0.01,4.00,Nz)
    
    z_distance = block["distances","z"]
    chi_distance = block["distances","d_m"]*block['cosmological_parameters', 'h0']
    a_distance = 1./(1+z_distance)
    chi_of_z_spline = interp1d(z_distance, chi_distance,bounds_error=False,fill_value="extrapolate")
    
    zf = np.linspace( 0.0,4.0,400 )
    chi = chi_of_z_spline(zf)
    
    # for simply case, only constant photoz
    if not constant_sigmaz:
        sigmaz_a=0.01
        sigmaz_b=0.01
    else:
        sigmaz_a=block['photoz_errors','sigmaz']
        sigmaz_b=block['photoz_errors','sigmaz']
    
    if folder_has_files(w_path):
        print("Has ve w files, sum togather...")
        rp = np.load( w_path+"w_1.npz" )["arr_0"]
        w_rp_1 = np.load( w_path+"w_1.npz" )["arr_1"]
        w_rp_2 = np.load( w_path+"w_2.npz" )["arr_1"]
        w_rp_3 = np.load( w_path+"w_3.npz" )["arr_1"]
        w_rp_4 = np.load( w_path+"w_4.npz" )["arr_1"]
        w_rp_5 = np.load( w_path+"w_5.npz" )["arr_1"]
        w_rp_6 = np.load( w_path+"w_6.npz" )["arr_1"]
        w_rp_7 = np.load( w_path+"w_7.npz" )["arr_1"]
        w_rp_8 = np.load( w_path+"w_8.npz" )["arr_1"]
        
        bv1 = block["galaxy_power","bias_values_a[bin1]"]
        bv2 = block["galaxy_power","bias_values_b[bin2]"]
        
        W = pk_terms_sum(bv1,bv2,w_rp_1,w_rp_2,w_rp_3,w_rp_4,w_rp_5,w_rp_6,w_rp_7,w_rp_8)
        
    else:
        print("No w files, start compute...")
        k_power = block['galaxy_power', 'k_h']
        z_power = block['galaxy_power','z']
        chi_power = chi_of_z_spline(z_power)
        
        Pterm1 = np.load( pks_path+"Pk1_Pd1d1.npz")['arr_0']
        Pterm2 = np.load( pks_path+"Pk2_Pd1d2.npz")['arr_0']
        Pterm3 = np.load( pks_path+"Pk3_Pd2d2.npz")['arr_0']
        Pterm4 = np.load( pks_path+"Pk4_Pd1s2.npz")['arr_0']
        Pterm5 = np.load( pks_path+"Pk5_Pd2s2.npz")['arr_0']
        Pterm6 = np.load( pks_path+"Pk6_Ps2s2.npz")['arr_0']
        Pterm7 = np.load( pks_path+"Pk7_sig3nl.npz")['arr_0']
        Pterm8 = np.load( pks_path+"Pk8_k2P.npz")['arr_0']
        
        rp,w_rp_1 = power_w_photoz(block,Nz,Npi,k_power,z_power,chi_power,Pterm1,chi,z_low,z_distance,chi_distance,Pi,sigmaz_a,sigmaz_b,block['cosmological_parameters','omega_m'],block['cosmological_parameters', 'omega_lambda'],zf,block['photoz_errors','Pi_mask_max'],sample_a,sample_b)
        rp,w_rp_2 = power_w_photoz(block,Nz,Npi,k_power,z_power,chi_power,Pterm2,chi,z_low,z_distance,chi_distance,Pi,sigmaz_a,sigmaz_b,block['cosmological_parameters','omega_m'],block['cosmological_parameters', 'omega_lambda'],zf,block['photoz_errors','Pi_mask_max'],sample_a,sample_b)
        rp,w_rp_3 = power_w_photoz(block,Nz,Npi,k_power,z_power,chi_power,Pterm3,chi,z_low,z_distance,chi_distance,Pi,sigmaz_a,sigmaz_b,block['cosmological_parameters','omega_m'],block['cosmological_parameters', 'omega_lambda'],zf,block['photoz_errors','Pi_mask_max'],sample_a,sample_b)
        rp,w_rp_4 = power_w_photoz(block,Nz,Npi,k_power,z_power,chi_power,Pterm4,chi,z_low,z_distance,chi_distance,Pi,sigmaz_a,sigmaz_b,block['cosmological_parameters','omega_m'],block['cosmological_parameters', 'omega_lambda'],zf,block['photoz_errors','Pi_mask_max'],sample_a,sample_b)
        rp,w_rp_5 = power_w_photoz(block,Nz,Npi,k_power,z_power,chi_power,Pterm5,chi,z_low,z_distance,chi_distance,Pi,sigmaz_a,sigmaz_b,block['cosmological_parameters','omega_m'],block['cosmological_parameters', 'omega_lambda'],zf,block['photoz_errors','Pi_mask_max'],sample_a,sample_b)
        rp,w_rp_6 = power_w_photoz(block,Nz,Npi,k_power,z_power,chi_power,Pterm6,chi,z_low,z_distance,chi_distance,Pi,sigmaz_a,sigmaz_b,block['cosmological_parameters','omega_m'],block['cosmological_parameters', 'omega_lambda'],zf,block['photoz_errors','Pi_mask_max'],sample_a,sample_b)
        rp,w_rp_7 = power_w_photoz(block,Nz,Npi,k_power,z_power,chi_power,Pterm7,chi,z_low,z_distance,chi_distance,Pi,sigmaz_a,sigmaz_b,block['cosmological_parameters','omega_m'],block['cosmological_parameters', 'omega_lambda'],zf,block['photoz_errors','Pi_mask_max'],sample_a,sample_b)
        rp,w_rp_8 = power_w_photoz(block,Nz,Npi,k_power,z_power,chi_power,Pterm8,chi,z_low,z_distance,chi_distance,Pi,sigmaz_a,sigmaz_b,block['cosmological_parameters','omega_m'],block['cosmological_parameters', 'omega_lambda'],zf,block['photoz_errors','Pi_mask_max'],sample_a,sample_b)
        
        np.savez( w_path+"w_1.npz",rp,w_rp_1 )
        np.savez( w_path+"w_2.npz",rp,w_rp_2 )
        np.savez( w_path+"w_3.npz",rp,w_rp_3 )
        np.savez( w_path+"w_4.npz",rp,w_rp_4 )
        np.savez( w_path+"w_5.npz",rp,w_rp_5 )
        np.savez( w_path+"w_6.npz",rp,w_rp_6 )
        np.savez( w_path+"w_7.npz",rp,w_rp_7 )
        np.savez( w_path+"w_8.npz",rp,w_rp_8 )
        
        bv1 = block["galaxy_power","bias_values_a[bin1]"]
        bv2 = block["galaxy_power","bias_values_b[bin2]"]
        
        W = pk_terms_sum(bv1,bv2,w_rp_1,w_rp_2,w_rp_3,w_rp_4,w_rp_5,w_rp_6,w_rp_7,w_rp_8)
        
    #finally save wg+ into data block
    block['galaxy_w','w_rp_1_1_%s_%s'%(sample_a,sample_b)] = W
    block['galaxy_w', 'r_p'] = rp
        
    return 0

def power_w_photoz(block,Nz,Npi,k_power,z_power,chi_power,P,chi,z_low,z_distance,chi_distance,Pi,sigmaz_a,sigmaz_b,omega_m,omega_lambda,zf,Pi_mask_max,sample_a,sample_b):
    
    P_interpolator = interp2d(k_power,chi_power,P,bounds_error=False, fill_value=0)
    
    Nell = 300
    ell = np.logspace(-6,np.log10(20000),Nell) 
    Cell_all = np.zeros((Nz, Npi, Nell))
    
    P_2d=[]
    for i, l in enumerate(ell):
        P1d = [P_interpolator((l+0.5)/x, x) for x in chi]
        P_2d.append(P1d)
    P_2d = np.array(P_2d)
    
    for i,z_l in enumerate(z_low):
        for j,pi in enumerate(Pi):
            # coordinate transform
            Hz = 100 * np.sqrt(omega_m*(1+z_l)**3 + omega_lambda) # no h because Pi is in units h^-1 Mpc
            z1 = z_l
            z2 = z_l + (1./clight * Hz * pi)
            
            Pz1 = gaussian(zf, sigmaz_a, z1)
            Pz1 = Pz1/np.trapz(Pz1,chi)

            Pz2 = gaussian(zf, sigmaz_b, z2)
            Pz2 = Pz2/np.trapz(Pz2,chi)
            
            C_gg = do_limber_integral(ell, P_2d, Pz1, Pz2, chi)
            Cell_all[i,j,:]=C_gg
            
    # Next do the Hankel transform
    xi_all = np.zeros_like(Cell_all)-9999.
    rp = np.logspace(np.log10(0.1), np.log10(300), xi_all.shape[2])
    
    chi_of_z_spline = interp1d(z_distance, chi_distance,bounds_error=False,fill_value="extrapolate")   
    for i, z in enumerate(z_low):
        x0 =  chi_of_z_spline(z)
        # do the coordinate transform to convert theta to rp at given redshift
        theta_radians = np.arctan(rp/x0)
        theta_degrees = theta_radians * 180./np.pi
        for j,pi in enumerate(Pi):
            Cell = Cell_all[i,j,:]

            theta_new,xi_new = FFTLog(ell, Cell*ell, 0, 0, lowring=True)
            xi_new = xi_new/theta_new/2/np.pi
            xi_interpolated = interp1d(theta_new,xi_new,fill_value="extrapolate")(theta_radians)
            xi_all[i,j,:]=xi_interpolated


    # integrate over los separation, between +/-Pi_max
    Pi_max=Pi_mask_max
    mask = ((Pi<Pi_max) & (Pi>-Pi_max))
    xi_rp = np.trapz(xi_all[:,mask,:], x=Pi[mask], axis=1)
    
    # and then over redshift
    za, W = get_redshift_kernel(block, 0, 0, zf, chi, sample_a, sample_b)
    W/=np.trapz(W,zf)
    Wofz = interp1d(zf,W)
    K = np.array([Wofz(z_low)]*len(rp)).T
    w_rp = np.trapz(xi_rp*K, x=z_low, axis=0) #/np.trapz(K, Zm, axis=0)
    
    return rp, w_rp

def gaussian(x,s,m):
    return 1/(s*np.sqrt(2*np.pi))*np.exp( -(x-m)**2/(2*s**2) )

def do_limber_integral(ell, P, p1, p2, X):

    I1 = interp1d(X,p1)
    I2 = interp1d(X,p2)
    cl = [] 
    Az = I1(X)*I2(X)/X**2
    Az[np.isinf(Az)]=0
    Az[np.isnan(Az)]=0

    Az_reshaped = np.array([Az]*P.shape[0])
    cl = np.trapz(Az_reshaped*P[:,:,0],X,axis=1)

    return np.array(cl)
    
def get_redshift_kernel(block, i, j, z0, x, sample_a, sample_b):


    dz = z0[1]-z0[0]
    dxdz = np.gradient(x,dz)
    #interp_dchi = spi.interp1d(z,Dchi)

    na = block['nz_%s'%sample_a, 'nbin']
    nb = block['nz_%s'%sample_b, 'nbin']
    zmin = 0.01

    nz_b = block['nz_%s'%sample_b, 'bin_%d'%(j+1)]
    zb = block['nz_%s'%sample_b, 'z']
    nz_a = block['nz_%s'%sample_a, 'bin_%d'%(i+1)]
    za = block['nz_%s'%sample_a, 'z']

    interp_nz = interp1d(zb, nz_b, fill_value='extrapolate')
    nz_b = interp_nz(z0)
    interp_nz = interp1d(za, nz_a, fill_value='extrapolate')
    nz_a = interp_nz(z0)


    X = nz_a * nz_b/x/x/dxdz
    X[0]=0
    interp_X = interp1d(z0, X, fill_value='extrapolate')

    # Inner integral over redshift
    V,Verr = quad(interp_X, zmin, z0.max())
    W = nz_a*nz_b/x/x/dxdz/np.trapz(X,x=z0)
    #V
    W[0]=0

    return z0,W













    
    
    