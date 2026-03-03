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
    
    sample_a = options.get_string(option_section, "sample_a", default="forecast_sample_shape")
    sample_b = options.get_string(option_section, "sample_b", default="forecast_sample_shape")
    constant_sigmaz = options.get_bool(option_section, "constant_sigmaz", default=True)
    timing = options.get_bool(option_section, "timing", default=True)
    w_folder = options.get_string(option_section,"wpp_folder")
    os.makedirs(w_folder, exist_ok=True)

    return sample_a,sample_b,timing,constant_sigmaz,w_folder

def execute(block, config):
    sample_a, sample_b, timing, constant_sigmaz, w_folder = config
    
    # set LOS bins
    Npi = block['photoz_errors','N_pi']
    Nz = 200
    Pi = np.linspace(-block['photoz_errors','Pi_max'],block['photoz_errors','Pi_max'],Npi)
    z_low = np.linspace(0.01,4.00,Nz)
    
    z_distance = block["distances","z"]
    chi_distance = block["distances","d_m"]*block['cosmological_parameters', 'h0']
    a_distance = 1./(1+z_distance)
    chi_of_z_spline = interp1d(z_distance, chi_distance, bounds_error=False, fill_value="extrapolate")
    
    zf = np.linspace( 0.0,4.0,400 )
    chi = chi_of_z_spline(zf)
    
    # for simply case, only constant photoz
    if not constant_sigmaz:
        sigmaz_a=0.01
        sigmaz_b=0.01
    else:
        sigmaz_a=block['photoz_errors','sigmaz']
        sigmaz_b=block['photoz_errors','sigmaz']
        
    ia_section = "intrinsic_alignment_parameters"
    A1 = block.get_double(ia_section, "A1", 1.0)
    
    if folder_has_files(w_folder):
        
        num_arrays = block['II_power','ia_model']
        
        if num_arrays[0] == 0:
            print("IA model is nla, using pre_computed xi_rp...")
            chi = np.load(w_folder+"chi.npz")["arr_0"]
            rp = np.load(w_folder+"rp.npz")["arr_0"]
            xi_rp = np.load(w_folder+"xi_p_nl.npz")["arr_0"]
            xi_rp = A1 * A1 * xi_rp
        
        elif num_arrays[0] == 1:
            pass
    
    else:
        
        print("This is first time computation, start compute w...")
        
        num_arrays = block['II_power','ia_model']
        k_power = block['II_power','k_h']
        z_power = block['II_power','z']
        if num_arrays[0] == 0:
            p_nl = block['II_power','p_k']
            C1 = block['II_power','c1']
            C1_d_A1 = C1 / A1
            P_II = C1_d_A1 * C1_d_A1 * p_nl
            
            chi,rp,xi_rp = get_xi_rp(block,zf,Nz,Npi,z_distance, chi_distance,sigmaz_a,sigmaz_b,P_II,k_power,z_power,z_low,Pi)
            
            np.savez(w_folder+"chi.npz",chi)
            np.savez(w_folder+"rp.npz",rp)
            np.savez(w_folder+"xi_p_nl.npz",xi_rp)
            
            xi_rp = A1 * A1 * xi_rp
            
        elif num_arrays[0] == 1:
            C1 = block['II_power','c1']
            Cdel = block['II_power','cdel']
            C2 = block['II_power','c2']
            
            p_nl = block["II_power","p_k"]
            p_ta_ee = block["II_power","p_ta_ee"]
            p_ta_de = block["II_power","2p_ta_de"]
            p_tt_ee = block["II_power","p_tt_ee"]
            p_mix_a = block["II_power","p_mix_a"]
            p_mix_b = block["II_power","p_mix_b"]
            p_mix_d_ee = block["II_power","p_mix_d_ee"]
            
            
            nla_II_EE = C1 * C1 * p_nl
            ta_II_EE = Cdel ** 2 * p_ta_ee + C1 * Cdel * p_ta_de
            tt_II_EE = C2 * C2 * p_tt_ee
            mix_II_EE = 2.0 * C2 * ( C1 * p_mix_a + C1 * p_mix_b + Cdel * p_mix_d_ee )
        
            P_II = nla_II_EE + ta_II_EE + tt_II_EE + mix_II_EE
            
            chi,rp,xi_rp = get_xi_rp(block,zf,Nz,Npi,z_distance, chi_distance,sigmaz_a,sigmaz_b,P_II,k_power,z_power,z_low,Pi)
            
    # and then over redshift
    za, W = get_redshift_kernel(block, 0, 0, zf, chi, sample_a, sample_b)
    W/=np.trapz(W,zf)
    Wofz = interp1d(zf,W)
    K = np.array([Wofz(z_low)]*len(rp)).T
    w_rp = np.trapz(xi_rp*K, x=z_low, axis=0) #/np.trapz(K, Zm, axis=0)

    #finally save wg+ into data block
    block['intrinsic_w','w_rp_1_1_%s_%s'%(sample_a,sample_b)] = w_rp
    #print(block['intrinsic_w','w_rp_1_1_%s_%s'%(sample_a,sample_b)])
    block['intrinsic_w', 'r_p'] = rp
        
    return 0

def get_xi_rp(block,zf,Nz,Npi,z_distance, chi_distance, sigmaz_a, sigmaz_b, power, k_power, z_power,z_low,Pi):
    
    chi_of_z_spline = interp1d(z_distance, chi_distance, bounds_error=False, fill_value="extrapolate")
    
    chi = chi_of_z_spline(zf)
    chi_power = chi_of_z_spline(z_power)
    
    P_interpolator = interp2d(k_power,chi_power,power,bounds_error=False, fill_value=0)
    
    Nell = 300
    ell = np.logspace(-6,np.log10(19000),Nell) 
    Cell_all = np.zeros((Nz, Npi, Nell))
    
    P_2d=[]
    for i, l in enumerate(ell):
        P1d = [P_interpolator((l+0.5)/x, x) for x in chi]
        P_2d.append(P1d)
    P_2d = np.array(P_2d)
    
    # we loop over a grid of los separation Pi and mean z z0
    for i, z_l in enumerate(z_low):
        for j,pi in enumerate(Pi):
            
            # coordinate transform
            Hz = 100 * np.sqrt(block['cosmological_parameters','omega_m']*(1+z_l)**3 + block['cosmological_parameters', 'omega_lambda']) # no h because Pi is in units h^-1 Mpc
            z1 = z_l
            z2 = z_l + (1./clight * Hz * pi)
            if z2<0: 
                continue

            Pz1 = gaussian(zf, sigmaz_a, z1)
            Pz1 = Pz1/np.trapz(Pz1,chi)

            Pz2 = gaussian(zf, sigmaz_b, z2)
            Pz2 = Pz2/np.trapz(Pz2,chi)

            C_gI = do_limber_integral(ell, P_2d, Pz1, Pz2, chi)
            Cell_all[i,j,:]=C_gI
            
    # Next do the Hankel transform
    xi_all = np.zeros_like(Cell_all)-9999.
    rp = np.logspace(np.log10(0.1), np.log10(300), xi_all.shape[2])
    
    for i, z in enumerate(z_low):
        x0 =  chi_of_z_spline(z)
        # do the coordinate transform to convert theta to rp at given redshift
        theta_radians = np.arctan(rp/x0)
        theta_degrees = theta_radians * 180./np.pi
        for j,pi in enumerate(Pi):
            Cell = Cell_all[i,j,:]

            theta_new_0,xi_new_0 = FFTLog(ell, Cell*ell, 0, 0, lowring=True)
            theta_new_4,xi_new_4 = FFTLog(ell, Cell*ell, 0, 4, lowring=True)
            xi_new_0 = xi_new_0/theta_new_0/2/np.pi
            xi_new_4 = xi_new_4/theta_new_0/2/np.pi
            xi_interpolated_0 = interp1d(theta_new_0,xi_new_0,fill_value="extrapolate")(theta_radians)
            xi_interpolated_4 = interp1d(theta_new_4,xi_new_4,fill_value="extrapolate")(theta_radians)
            xi_all[i,j,:]=xi_interpolated_0+xi_interpolated_4


    # integrate over los separation, between +/-Pi_max
    Pi_max=block['photoz_errors','Pi_mask_max']
    mask = ((Pi<Pi_max) & (Pi>-Pi_max))
    xi_rp = np.trapz(xi_all[:,mask,:], x=Pi[mask], axis=1)
    
    return chi,rp,xi_rp

def gaussian(x,s,m):
    return 1/(s*np.sqrt(2*np.pi))*np.exp( -(x-m)**2/(2*s**2) )


def do_limber_integral(ell, P, p1, p2, X):

    I1 = interp1d(X,p1)
    I2 = interp1d(X,p2)
    cl = [] 
    Az = 1./X/X*I1(X)*I2(X)
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
    
    
    
    
